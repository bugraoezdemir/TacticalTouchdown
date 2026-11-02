"""
Tactical bias: a user-facing preference, not a probability.

The engine takes an argmax over P x V x C rather than sampling, so a slider
cannot promise a share of actions. It tilts which situations an action wins.
The three biases are deliberately independent -- shoot/pass/dribble compete for
situations, not for a fixed budget, so "more pressing at everything" must be
expressible and a simplex would make each slider silently move the other two.
"""
import collections

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.decisions.on_ball import on_ball_decision
from app.engine.decisions.tunables import (
    TACTIC_BIAS_MAX,
    TACTIC_BIAS_MIN,
    tactical_bias,
)
from app.engine.factory import build_test_state
from tests.helpers import midfielder, striker
from app.engine.game_loop import tick
from app.main import app


# ── Resolution: player override beats team default ───────────────────────────

def test_team_setting_applies_when_player_has_no_override():
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.shoot = 1.5
    player = striker(state)
    assert player.bias_shoot is None
    assert tactical_bias(player, team, "shot") == pytest.approx(1.5)


def test_player_override_wins():
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.shoot = 1.5
    player = striker(state)
    player.bias_shoot = 0.8
    assert tactical_bias(player, team, "shot") == pytest.approx(0.8)


def test_defaults_are_neutral():
    state = build_test_state(seed=1)
    team = state.home_team
    for action in ("shot", "pass", "dribble"):
        assert tactical_bias(midfielder(state), team, action) == pytest.approx(1.0)


def test_bias_is_clamped():
    state = build_test_state(seed=1)
    team = state.home_team
    player = midfielder(state)
    player.bias_shoot = 99.0
    assert tactical_bias(player, team, "shot") == TACTIC_BIAS_MAX
    player.bias_shoot = -5.0
    assert tactical_bias(player, team, "shot") == TACTIC_BIAS_MIN


def test_biases_are_independent():
    """Not a simplex: raising one must not lower the others."""
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.shoot = 2.0
    assert tactical_bias(midfielder(state), team, "pass") == pytest.approx(1.0)
    assert tactical_bias(midfielder(state), team, "dribble") == pytest.approx(1.0)


# ── The behaviour a slider is supposed to produce ────────────────────────────

def _action_mix(seed: int, **tactics) -> dict[str, float]:
    state = build_test_state(seed=seed)
    for key, value in tactics.items():
        setattr(state.home_team.tactics, key, value)
    counts: collections.Counter = collections.Counter()
    for _ in range(2500):
        owner = state.get_ball_owner()
        if owner is not None and owner.has_ball and owner.team_id == 1:
            counts[on_ball_decision(owner, state).action_type] += 1
        state = tick(state)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


@pytest.mark.slow
def test_raising_a_bias_raises_that_action():
    neutral = _action_mix(3)
    more_dribble = _action_mix(3, dribble=2.0)
    more_pass = _action_mix(3, pass_=2.0)
    assert more_dribble.get("dribble", 0) > neutral.get("dribble", 0)
    assert more_pass.get("pass", 0) > neutral.get("pass", 0)


# ── The API the sidebars talk to ─────────────────────────────────────────────

@pytest.fixture
def client():
    c = TestClient(app)
    c.post("/simulate/start")
    return c


def test_tactics_endpoint_sets_team_values(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "shoot": 1.6})
    assert r.status_code == 200
    assert r.json()["home_team"]["tactics"]["shoot"] == pytest.approx(1.6)


def test_tactics_endpoint_sets_player_overrides(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 9, "bias_shoot": 2.0, "bias_pass": None,
                     "bias_dribble": None}],
    })
    assert r.status_code == 200
    player = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 9)
    assert player["bias_shoot"] == pytest.approx(2.0)
    assert player["bias_pass"] is None


def test_tactics_endpoint_rejects_a_player_from_another_team(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 15, "bias_shoot": 2.0, "bias_pass": None,
                     "bias_dribble": None}],
    })
    assert r.status_code == 400


def test_tactics_endpoint_rejects_unknown_player(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 999, "bias_shoot": 2.0, "bias_pass": None,
                     "bias_dribble": None}],
    })
    assert r.status_code == 404


def test_tactics_does_not_restart_the_match(client):
    for _ in range(20):
        client.post("/simulate/tick")
    before = client.post("/simulate/tick").json()["tick"]
    after = client.post("/simulate/tactics",
                        json={"team_id": 2, "dribble": 1.4}).json()["tick"]
    assert after == before


def test_tactics_requires_a_started_match():
    fresh = TestClient(app)
    fresh.post("/simulate/reset")
    r = fresh.post("/simulate/tactics", json={"team_id": 1, "shoot": 1.2})
    assert r.status_code in (200, 400)


# ── Direct pass: which flavour of pass, not how often to pass ────────────────

def test_direct_resolution_and_clamping():
    from app.engine.decisions.tunables import (
        TACTIC_BIAS_MAX,
        TACTIC_BIAS_MIN,
        direct_bias,
    )

    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.direct = 1.5
    player = team.players[5]
    assert direct_bias(player, team) == pytest.approx(1.5)

    player.bias_direct = 0.7
    assert direct_bias(player, team) == pytest.approx(0.7)

    player.bias_direct = 99.0
    assert direct_bias(player, team) == TACTIC_BIAS_MAX
    player.bias_direct = -1.0
    assert direct_bias(player, team) == TACTIC_BIAS_MIN


def test_direct_default_is_neutral():
    from app.engine.decisions.tunables import direct_bias

    state = build_test_state(seed=1)
    assert direct_bias(state.home_team.players[5], state.home_team) == pytest.approx(1.0)


def test_direct_trades_between_flavours():
    """It must not simply inflate passing: a ball in behind scales up as
    runway rises and a ball to feet scales down by the reciprocal, so the
    Pass slider keeps sole control of how often the ball is released."""
    from app.engine.decisions.candidates import Candidate
    from app.engine.decisions.on_ball import score_candidate

    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    # Which side kicks off is seeded, so set the CARRIER's team rather than
    # assuming it is home.
    carrier_team = state.get_team(carrier.team_id)
    mate = next(
        m for m in state.teammates_of(carrier.player_id) if m.position != "GK"
    )

    def context_of(direct: float, run: float) -> float:
        carrier_team.tactics.direct = direct
        target = mate.pos + np.array([0.0, run]) * (1.0 if mate.pos[1] < 34.0 else -1.0)
        cand = Candidate(
            kind="pass", target=target, speed=0.7, friction=0.02,
            recipient_id=mate.player_id,
        )
        return score_candidate(cand, carrier, state).context

    assert context_of(1.6, 0.0) > context_of(0.6, 0.0)      # to his feet
    assert context_of(1.6, 20.0) < context_of(0.6, 20.0)    # he must chase it
    assert context_of(1.6, 10.0) == pytest.approx(context_of(0.6, 10.0))

    # Neutral leaves every pass untouched.
    assert context_of(1.0, 0.0) == pytest.approx(context_of(1.0, 20.0))


@pytest.mark.slow
def test_direct_shifts_the_chosen_pass_flavour():
    import collections

    from app.engine.decisions.candidates import build_candidates
    from app.engine.decisions.on_ball import score_candidate
    from app.engine.game_loop import tick

    def behind_share(direct: float) -> float:
        state = build_test_state(seed=0)
        state.home_team.tactics.direct = direct
        counts: collections.Counter = collections.Counter()
        for _ in range(3000):
            owner = state.get_ball_owner()
            if owner is not None and owner.has_ball and owner.team_id == 1:
                best = max(
                    (score_candidate(c, owner, state) for c in build_candidates(owner, state)),
                    key=lambda c: c.utility,
                )
                if best.kind == "pass":
                    counts["behind" if best.is_runway else "feet"] += 1
            state = tick(state)
        total = sum(counts.values()) or 1
        return counts["behind"] / total

    assert behind_share(0.6) > behind_share(1.6)


def test_endpoint_sets_direct(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "direct": 1.6})
    assert r.status_code == 200
    assert r.json()["home_team"]["tactics"]["direct"] == pytest.approx(1.6)


def test_endpoint_sets_player_runway(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 6, "bias_direct": 0.6}],
    })
    assert r.status_code == 200
    player = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 6)
    assert player["bias_direct"] == pytest.approx(0.6)


# ── One-two: what the passer does after releasing ───────────────────────────

def test_one_two_resolution_and_clamping():
    from app.engine.decisions.tunables import (
        TACTIC_BIAS_MAX,
        TACTIC_BIAS_MIN,
        one_two_bias,
    )

    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.one_two = 1.5
    player = team.players[5]
    assert one_two_bias(player, team) == pytest.approx(1.5)

    player.bias_one_two = 0.7
    assert one_two_bias(player, team) == pytest.approx(0.7)

    player.bias_one_two = 99.0
    assert one_two_bias(player, team) == TACTIC_BIAS_MAX
    player.bias_one_two = -1.0
    assert one_two_bias(player, team) == TACTIC_BIAS_MIN


def test_one_two_default_is_neutral():
    from app.engine.decisions.tunables import one_two_bias

    state = build_test_state(seed=1)
    assert one_two_bias(state.home_team.players[5], state.home_team) == pytest.approx(1.0)


def test_passer_only_stands_off_a_pass_a_teammate_will_win():
    """A ball he shanked is still his to chase. Treating every pass as someone
    else's problem would abandon the ball whenever it went astray."""
    from app.engine.decisions.off_ball import _just_passed_safely

    state = build_test_state(seed=0)
    passer = state.home_team.players[5]
    state.ball.last_passer_id = passer.player_id
    state.ball.last_pass_tick = state.tick
    state.ball.owner_id = None
    state.ball.pos = passer.pos + np.array([6.0, 0.0])
    state.ball.velocity = np.array([0.5, 0.0])

    # A teammate ahead of the ball, opponents nowhere: it is going to arrive.
    mate = state.home_team.players[8]
    mate.pos = passer.pos + np.array([16.0, 0.0])
    for o in state.away_team.players:
        o.pos = np.array([5.0, 60.0])
    assert _just_passed_safely(passer, state)

    # An opponent about to cut it out: he must not stand off.
    state.away_team.players[4].pos = passer.pos + np.array([7.0, 0.0])
    assert not _just_passed_safely(passer, state)


def test_a_passer_who_already_lost_it_is_not_standing_off():
    from app.engine.decisions.off_ball import _just_passed_safely

    state = build_test_state(seed=0)
    passer = state.home_team.players[5]
    state.ball.last_passer_id = passer.player_id
    state.ball.last_pass_tick = state.tick - 500  # long ago
    assert not _just_passed_safely(passer, state)


def test_endpoint_sets_one_two(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "one_two": 1.6})
    assert r.status_code == 200
    assert r.json()["home_team"]["tactics"]["one_two"] == pytest.approx(1.6)


def test_endpoint_sets_player_one_two(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 6, "bias_one_two": 0.6}],
    })
    assert r.status_code == 200
    player = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 6)
    assert player["bias_one_two"] == pytest.approx(0.6)


# ── Forward runs: supporting a break ────────────────────────────────────────

def test_attack_call_resolution_and_clamping():
    from app.engine.decisions.tunables import (
        TACTIC_BIAS_MAX,
        TACTIC_BIAS_MIN,
        attack_call_bias,
    )

    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.attack_call = 1.5
    player = team.players[5]
    assert attack_call_bias(player, team) == pytest.approx(1.5)

    player.bias_attack_call = 0.7
    assert attack_call_bias(player, team) == pytest.approx(0.7)

    player.bias_attack_call = 99.0
    assert attack_call_bias(player, team) == TACTIC_BIAS_MAX
    player.bias_attack_call = -1.0
    assert attack_call_bias(player, team) == TACTIC_BIAS_MIN


def test_attack_call_default_is_neutral():
    from app.engine.decisions.tunables import attack_call_bias

    state = build_test_state(seed=1)
    assert attack_call_bias(state.home_team.players[5], state.home_team) == pytest.approx(1.0)


def test_progression_is_measured_toward_the_attacking_goal():
    from app.engine.decisions.off_ball import ball_progression

    state = build_test_state(seed=0)
    state.ball.velocity = np.array([0.8, 0.0])
    # Home attacks rightward, away leftward: the same ball advances one and
    # retreats for the other.
    assert ball_progression(state.ball, state.home_team) > 0
    assert ball_progression(state.ball, state.away_team) < 0


def test_only_genuine_progression_counts_as_a_break():
    """Ordinary play advances the ball ~0.16 m/tick; the threshold sits well
    above that so this fires on deep passes and carries, not on shuffling."""
    from app.engine.decisions.off_ball import _breaking_forward

    state = build_test_state(seed=0)
    state.ball.velocity = np.array([0.15, 0.0])
    assert _breaking_forward(state.ball, state.home_team) == 0.0

    state.ball.velocity = np.array([0.9, 0.0])
    assert _breaking_forward(state.ball, state.home_team) > 0.9

    # Backwards is never a break.
    state.ball.velocity = np.array([-0.9, 0.0])
    assert _breaking_forward(state.ball, state.home_team) == 0.0


def test_break_strength_is_bounded():
    from app.engine.decisions.off_ball import _breaking_forward

    state = build_test_state(seed=0)
    for vx in np.arange(-1.5, 1.6, 0.1):
        state.ball.velocity = np.array([float(vx), 0.0])
        assert 0.0 <= _breaking_forward(state.ball, state.home_team) <= 1.0


def test_endpoint_sets_attack_call(client):
    r = client.post("/simulate/tactics", json={"team_id": 1, "attack_call": 1.6})
    assert r.status_code == 200
    assert r.json()["home_team"]["tactics"]["attack_call"] == pytest.approx(1.6)


def test_endpoint_sets_player_attack_call(client):
    r = client.post("/simulate/tactics", json={
        "team_id": 1,
        "players": [{"player_id": 6, "bias_attack_call": 0.6}],
    })
    assert r.status_code == 200
    player = next(p for p in r.json()["home_team"]["players"] if p["player_id"] == 6)
    assert player["bias_attack_call"] == pytest.approx(0.6)
