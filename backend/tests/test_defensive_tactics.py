"""
Defensive pressing, plus the attacking off-ball parameters.

Pressing is the dial on the pursuit trade-off already at the heart of the
out-of-possession model: lambda = 0 is pure pursuit (go at the ball), lambda = 1
is full interception (align with its trajectory). Higher pressing weighs
approaching the ball over aligning with it.

Like the offensive sliders these are relative multipliers around 1.0, never
probabilities, and never constrained to sum with anything.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.decisions.pursuit import compute_lambda, prediction_confidence
from app.engine.decisions.tunables import (
    TACTIC_BIAS_MAX,
    TACTIC_BIAS_MIN,
    defensive_bias,
)
from app.engine.factory import build_test_state
from app.engine.game_loop import _decision_order, tick
from app.main import app
from tests.helpers import striker


# -- Resolution --------------------------------------------------------------

def test_team_setting_applies_without_an_override():
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.pressing = 1.5
    assert defensive_bias(team.players[3], team) == pytest.approx(1.5)


def test_player_override_wins():
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.pressing = 1.5
    player = team.players[3]
    player.bias_pressing = 0.7
    assert defensive_bias(player, team) == pytest.approx(0.7)


def test_default_is_neutral():
    state = build_test_state(seed=1)
    assert defensive_bias(state.home_team.players[3], state.home_team) == pytest.approx(1.0)


def test_bias_is_clamped():
    state = build_test_state(seed=1)
    team = state.home_team
    player = team.players[3]
    player.bias_pressing = 99.0
    assert defensive_bias(player, team) == TACTIC_BIAS_MAX
    player.bias_pressing = -5.0
    assert defensive_bias(player, team) == TACTIC_BIAS_MIN


def test_pressing_is_independent_of_the_offensive_biases():
    """Not a simplex -- raising one must not move the others."""
    from app.engine.decisions.tunables import tactical_bias

    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.pressing = 2.0
    for action in ("shot", "pass", "dribble"):
        assert tactical_bias(team.players[3], team, action) == pytest.approx(1.0)


# -- The lambda trade-off ----------------------------------------------------

def test_neutral_pressing_reproduces_the_original_lambda():
    """The regression guard that keeps the existing pursuit tests meaningful."""
    confidence = prediction_confidence(0.30, 1.0, 15.0)
    assert compute_lambda(confidence, 0.0, 0.0, 1.0) == pytest.approx(
        compute_lambda(confidence, 0.0, 0.0)
    )


def test_higher_pressing_lowers_lambda():
    """Lower lambda means aim closer to the ball itself rather than leading its
    trajectory -- which is exactly 'approach rather than align'."""
    confidence = prediction_confidence(0.30, 1.0, 15.0)
    previous = 2.0
    for pressing in (0.5, 1.0, 1.5, 2.0):
        value = compute_lambda(confidence, 0.0, 0.0, pressing)
        assert value < previous
        previous = value


def test_lambda_stays_bounded_for_every_pressing():
    for confidence in np.linspace(0.0, 1.0, 6):
        for cover in np.linspace(0.0, 1.0, 6):
            for outlets in np.linspace(0.0, 1.0, 6):
                for pressing in (0.5, 1.0, 2.0):
                    value = compute_lambda(
                        float(confidence), float(cover), float(outlets), pressing
                    )
                    assert 0.0 <= value <= 1.0 and np.isfinite(value)


def test_pressing_scales_rather_than_replaces():
    """The cover term must still modulate the result -- a pressing player still
    holds shape when teammates already cover the ball."""
    confidence = prediction_confidence(0.30, 1.0, 15.0)
    alone = compute_lambda(confidence, 0.0, 0.0, 2.0)
    covered = compute_lambda(confidence, 0.9, 0.0, 2.0)
    assert covered > alone


# -- The behaviour the slider exists to produce ------------------------------

def _mean_target_distance(state, depth: float) -> float:
    state.home_team.tactics.offensiveness = depth
    committed: list[tuple[int, np.ndarray]] = []
    distances = []
    from app.engine.decisions.off_ball import off_ball_decision

    for player in _decision_order(state):
        if player.team_id != 1 or player.position == "GK" or player.has_ball:
            continue
        action = off_ball_decision(player, state, committed)
        committed.append((player.player_id, action.target_pos))
        distances.append(float(np.hypot(*(action.target_pos - state.ball.pos))))
    return float(np.mean(distances))


def test_a_high_block_positions_closer_to_the_ball():
    """Depth's since decision D-16; it was Pressing's."""
    state = build_test_state(seed=0)
    for _ in range(3000):
        state = tick(state)
        if state.ball.owner_id is None and float(np.hypot(*state.ball.velocity)) > 0.3:
            break

    previous = None
    for depth in (0.5, 0.8, 1.0, 1.4, 1.8, 2.0):
        mean = _mean_target_distance(state, depth)
        if previous is not None:
            assert mean <= previous + 1e-9, f"not monotonic at {depth}"
        previous = mean

    assert _mean_target_distance(state, 2.0) < _mean_target_distance(state, 0.5)


# -- The API the sidebars talk to --------------------------------------------

@pytest.fixture
def client():
    c = TestClient(app)
    c.post("/simulate/start")
    return c


def test_endpoint_sets_team_pressing(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "pressing": 1.6})
    assert r.status_code == 200
    assert r.json()["home_team"]["tactics"]["pressing"] == pytest.approx(1.6)


def test_endpoint_sets_player_pressing(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 10, "bias_pressing": 0.5}],
    })
    assert r.status_code == 200
    player = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 10)
    assert player["bias_pressing"] == pytest.approx(0.5)


def test_endpoint_sets_formation(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "formation": "4-3-3"})
    assert r.status_code == 200
    body = r.json()
    assert body["home_team"]["tactics"]["formation"] == "4-3-3"
    assert [p["position"] for p in body["home_team"]["players"]].count("CM") == 3


def test_endpoint_rejects_unknown_formation(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "formation": "9-9-9"})
    assert r.status_code == 400


def test_formation_change_does_not_restart_the_match(client):
    for _ in range(20):
        client.post("/simulate/tick")
    before = client.post("/simulate/tick").json()["tick"]
    after = client.post(
        "/simulate/tactics", json={"team_id": 1, "formation": "3-5-2"}
    ).json()["tick"]
    assert after == before


def test_formation_change_moves_assigned_positions(client):
    before = client.post("/simulate/tick").json()
    home_before = {p["player_id"]: p["assigned_pos"] for p in before["home_team"]["players"]}
    after = client.post(
        "/simulate/tactics", json={"team_id": 1, "formation": "5-3-2"}
    ).json()
    moved = [
        p for p in after["home_team"]["players"]
        if p["assigned_pos"] != home_before[p["player_id"]]
    ]
    assert moved


@pytest.mark.slow
def test_pressing_side_presses_higher_over_a_match():
    """Same seed, opposite settings: the pressing side should end up closer to
    the ball while defending."""
    def mean_defensive_distance(pressing: float) -> float:
        state = build_test_state(seed=2)
        state.home_team.tactics.pressing = pressing
        total = 0.0
        samples = 0
        for _ in range(2000):
            state = tick(state)
            owner = state.get_ball_owner()
            if owner is None or owner.team_id != 2:
                continue
            for p in state.home_team.players:
                if p.position == "GK":
                    continue
                total += float(np.hypot(*(p.pos - state.ball.pos)))
                samples += 1
        return total / max(samples, 1)

    assert mean_defensive_distance(1.8) < mean_defensive_distance(0.5)


# -- Attacking off-ball: roaming --------------------------------------------

def test_roaming_resolution_and_clamping():
    from app.engine.decisions.tunables import roaming_bias

    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.roaming = 1.5
    player = team.players[3]
    assert roaming_bias(player, team) == pytest.approx(1.5)

    player.bias_roaming = 0.7
    assert roaming_bias(player, team) == pytest.approx(0.7)

    player.bias_roaming = 99.0
    assert roaming_bias(player, team) == TACTIC_BIAS_MAX
    player.bias_roaming = -1.0
    assert roaming_bias(player, team) == TACTIC_BIAS_MIN


def test_roaming_default_is_neutral():
    from app.engine.decisions.tunables import roaming_bias

    state = build_test_state(seed=1)
    assert roaming_bias(state.home_team.players[3], state.home_team) == pytest.approx(1.0)


def test_roaming_is_independent_of_pressing():
    from app.engine.decisions.tunables import roaming_bias

    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.roaming = 2.0
    assert defensive_bias(team.players[3], team) == pytest.approx(1.0)


def test_attacking_roaming_goes_further_from_station():
    """Keyed on distance from the player's OWN STATION, not from the ball: a
    full-back holding width on the far touchline is far from the carrier but
    exactly where he should be."""
    from app.engine.decisions.off_ball import off_ball_decision

    state = build_test_state(seed=0)
    for _ in range(200):
        state = tick(state)
    # Hand possession to home rather than waiting for it: which side has the
    # ball at any tick is seeded, and this test is about the in-possession path.
    _give_ball_to_home(state)

    def mean_roam(roaming: float) -> float:
        """Measured on the decision, not after simulating: the ball changes
        hands within a few ticks and the phase under test would be gone."""
        state.home_team.tactics.roaming = roaming
        committed: list[tuple[int, np.ndarray]] = []
        roam = []
        for p in _decision_order(state):
            if p.team_id != 1 or p.position == "GK" or p.has_ball:
                continue
            action = off_ball_decision(p, state, committed)
            committed.append((p.player_id, action.target_pos))
            roam.append(float(np.hypot(*(action.target_pos - p.assigned_pos))))
        return float(np.mean(roam))

    assert mean_roam(1.8) > mean_roam(0.6)


def test_endpoint_sets_roaming(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "roaming": 1.7})
    assert r.status_code == 200
    assert r.json()["home_team"]["tactics"]["roaming"] == pytest.approx(1.7)


def test_endpoint_sets_player_roaming(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 10, "bias_roaming": 0.6}],
    })
    assert r.status_code == 200
    player = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 10)
    assert player["bias_roaming"] == pytest.approx(0.6)


# -- Attacking off-ball: support keyed on the carrier's outlet ---------------

def test_teammates_come_short_for_a_smothered_carrier():
    """The behaviour ball position could not express: a carrier surrounded on
    the halfway line reads as "mid-pitch, go make runs" under the old proxy,
    when what he needs is someone to offer an outlet."""
    from app.engine.decisions.off_ball import off_ball_decision

    state = build_test_state(seed=0)
    for _ in range(200):
        state = tick(state)
    carrier = _give_ball_to_home(state)
    opponents = state.opponents_of(1)

    def mean_distance_to_ball() -> float:
        committed: list[tuple[int, np.ndarray]] = []
        distances = []
        for p in _decision_order(state):
            if p.team_id != 1 or p.position == "GK" or p.has_ball:
                continue
            action = off_ball_decision(p, state, committed)
            committed.append((p.player_id, action.target_pos))
            distances.append(float(np.hypot(*(action.target_pos - state.ball.pos))))
        return float(np.mean(distances))

    for o in opponents:
        o.pos = np.array([2.0, 2.0])
    free = mean_distance_to_ball()

    mates = [m for m in state.teammates_of(carrier.player_id) if m.position != "GK"]
    for o, m in zip(opponents, mates):
        o.pos = m.pos.copy()
    smothered = mean_distance_to_ball()

    assert smothered < free


def test_endpoint_sets_offensiveness_and_restations(client):
    before = client.post("/simulate/tick").json()
    deep = {p["player_id"]: p["assigned_pos"][0] for p in before["home_team"]["players"]}

    r = client.post("/simulate/tactics", json={"team_id": 1, "offensiveness": 1.5})
    assert r.status_code == 200
    body = r.json()
    assert body["home_team"]["tactics"]["offensiveness"] == pytest.approx(1.5)

    outfield = [p for p in body["home_team"]["players"] if p["position"] != "GK"]
    assert all(p["assigned_pos"][0] > deep[p["player_id"]] for p in outfield)

    keeper = next(p for p in body["home_team"]["players"] if p["position"] == "GK")
    assert keeper["assigned_pos"][0] == pytest.approx(deep[keeper["player_id"]])


def test_endpoint_sets_player_depth_and_restations(client):
    before = client.post("/simulate/tick").json()
    baseline = {p["player_id"]: p["assigned_pos"][0] for p in before["home_team"]["players"]}

    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 2, "bias_offensiveness": 1.8}],
    })
    assert r.status_code == 200
    players = r.json()["home_team"]["players"]
    target = next(p for p in players if p["player_id"] == 2)
    assert target["bias_offensiveness"] == pytest.approx(1.8)
    assert target["assigned_pos"][0] > baseline[2]
    for p in players:
        if p["player_id"] != 2:
            assert p["assigned_pos"][0] == pytest.approx(baseline[p["player_id"]])


def test_clearing_a_depth_override_restations_too(client):
    before = client.post("/simulate/tick").json()
    baseline = {p["player_id"]: p["assigned_pos"][0] for p in before["home_team"]["players"]}

    client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 2, "bias_offensiveness": 1.8}],
    })
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 2, "bias_offensiveness": None}],
    })
    restored = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 2)
    assert restored["assigned_pos"][0] == pytest.approx(baseline[2])


def _give_ball_to_home(state):
    """Put the ball at a home midfielder's feet.

    Which side is in possession at any tick is seeded, so a test about the
    in-possession path must arrange it rather than wait for it.
    """
    carrier = next(p for p in state.home_team.players if p.position == "CM")
    for p in state.all_players():
        p.has_ball = False
    carrier.has_ball = True
    state.ball.owner_id = carrier.player_id
    state.ball.pos = carrier.pos.copy()
    state.ball.velocity = np.zeros(2)
    state.ball.state = "stationary"
    return carrier
