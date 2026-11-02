"""
Statistical harness: does the engine produce recognisable football?

Marked slow -- a full match is 162,000 ticks. The ranges are deliberately wide,
so a failure means a real regression rather than drift. Every number here was a
concrete failure before the utility model landed: 0 goals, 0 dribbles, and a
34.6 m mean shot distance.
"""
import collections

import numpy as np
import pytest

from app.engine.factory import build_test_state
from app.engine.game_loop import tick

TICKS = 18_000  # 10 minutes at 30 Hz


def play(seed: int = 0, ticks: int = TICKS) -> dict:
    state = build_test_state()
    state.rng = np.random.default_rng(seed)

    touches = collections.Counter()
    possession = collections.Counter()
    shot_distances: list[float] = []
    goals = 0
    changes = 0

    prev_state = state.ball.state
    prev_owner = state.ball.owner_id
    prev_pos = state.ball.pos.copy()

    for _ in range(ticks):
        owner = state.get_ball_owner()
        owner_pos = owner.pos.copy() if owner else None
        owner_team = owner.team_id if owner else None
        if owner is not None:
            possession[owner.team_id] += 1

        state = tick(state)

        if state.ball.state != prev_state and state.ball.state in (
            "in_pass", "in_shot", "in_dribble"
        ):
            touches[state.ball.state] += 1
            if state.ball.state == "in_shot" and owner_pos is not None:
                goal_x = 105.0 if state.get_team(owner_team).direction == 1 else 0.0
                shot_distances.append(abs(float(owner_pos[0]) - goal_x))

        if state.ball.owner_id != prev_owner:
            changes += 1
        if np.allclose(state.ball.pos, [52.5, 34.0]) and not np.allclose(
            prev_pos, [52.5, 34.0]
        ):
            goals += 1

        prev_state = state.ball.state
        prev_owner = state.ball.owner_id
        prev_pos = state.ball.pos.copy()

    return {
        "touches": touches, "possession": possession, "goals": goals,
        "changes": changes, "shot_distances": shot_distances, "state": state,
    }


@pytest.fixture(scope="module")
def match():
    return play(seed=20260903)


@pytest.mark.slow
def test_ball_actually_moves(match):
    """The engine must not deadlock with one player holding forever."""
    assert match["changes"] > 100


@pytest.mark.slow
def test_players_pass(match):
    assert match["touches"]["in_pass"] > 50


@pytest.mark.slow
def test_players_dribble(match):
    """Was exactly 0: the touch distance sat inside CAPTURE_RADIUS, so a
    dribble was an instant loss of possession and never scored."""
    assert match["touches"]["in_dribble"] > 20


@pytest.mark.slow
def test_shots_come_from_sensible_range(match):
    """Was 34.6 m on average -- players shot from their own half because
    scoring asked only whether the ball would reach the goal."""
    dists = match["shot_distances"]
    assert len(dists) >= 3, f"only {len(dists)} shots"
    assert float(np.mean(dists)) < 24.0
    inside = sum(1 for d in dists if d < 18.0) / len(dists)
    assert inside > 0.4


@pytest.mark.slow
def test_possession_is_contested(match):
    poss = match["possession"]
    total = sum(poss.values())
    assert total > 0
    share = poss[1] / total
    assert 0.25 < share < 0.75, f"possession {share:.0%}/{1 - share:.0%}"


@pytest.mark.slow
def test_players_stay_on_the_pitch(match):
    for p in match["state"].all_players():
        assert -0.01 <= p.pos[0] <= 105.01
        assert -0.01 <= p.pos[1] <= 68.01


@pytest.mark.slow
def test_ball_stays_on_the_pitch(match):
    ball = match["state"].ball
    assert -0.01 <= ball.pos[0] <= 105.01
    assert -0.01 <= ball.pos[1] <= 68.01


@pytest.mark.slow
def test_deterministic_given_a_seed():
    """Execution noise must come from the seeded generator on GameState, never
    the global RNG, or nothing here is reproducible."""
    a = play(seed=42, ticks=600)
    b = play(seed=42, ticks=600)
    assert np.allclose(a["state"].ball.pos, b["state"].ball.pos)
    assert a["touches"] == b["touches"]
