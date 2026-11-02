"""Match events and score. The UI reports these; it must not have to guess."""
import numpy as np
import pytest

from app.engine.constants import GOAL_PAUSE_TICKS
from app.engine.factory import build_test_state
from app.engine.game_loop import tick
from app.engine.physics import resolve_possession


def test_goal_increments_the_right_score():
    """A ball crossing the line at x=105 is scored by whoever attacks rightward."""
    state = build_test_state()
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.home_team.team_id
    state.ball.pos = np.array([105.5, 34.0])
    resolve_possession(state)

    assert state.home_score == 1
    assert state.away_score == 0
    goals = [e for e in state.events if e.kind == "goal"]
    assert len(goals) == 1
    assert "GOAL" in goals[0].detail


def test_goal_at_other_end_credits_the_other_team():
    state = build_test_state()
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.away_team.team_id
    state.ball.pos = np.array([-0.5, 34.0])
    resolve_possession(state)

    assert state.away_score == 1
    assert state.home_score == 0


def test_throw_in_emits_an_event():
    state = build_test_state()
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.home_team.team_id
    state.ball.pos = np.array([50.0, -1.0])
    resolve_possession(state)

    kinds = [e.kind for e in state.events]
    assert "throw_in" in kinds


def test_goal_line_restart_emits_corner_or_goal_kick():
    state = build_test_state()
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.away_team.team_id
    state.ball.pos = np.array([-1.0, 10.0])  # wide of the left post
    resolve_possession(state)

    kinds = [e.kind for e in state.events]
    assert "corner" in kinds or "goal_kick" in kinds


def test_events_are_capped():
    """The log is a rolling tail, not an unbounded leak over a 90-minute match."""
    state = build_test_state()
    for i in range(200):
        state.add_event("throw_in", state.home_team.team_id, None, f"event {i}")
    assert len(state.events) <= 40
    assert state.events[-1].detail == "event 199"


def test_kickoff_follows_a_goal():
    state = build_test_state()
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.home_team.team_id
    state.ball.pos = np.array([105.5, 34.0])
    resolve_possession(state)
    for _ in range(GOAL_PAUSE_TICKS):   # it rests in the net first
        state = tick(state)

    kinds = [e.kind for e in state.events]
    assert kinds.index("goal") < kinds.index("kickoff")
    assert np.allclose(state.ball.pos, [52.5, 34.0])


@pytest.mark.slow
def test_a_played_match_records_events():
    state = build_test_state()
    state.rng = np.random.default_rng(20260903)
    for _ in range(9000):
        state = tick(state)
    assert len(state.events) > 0
    assert all(e.detail for e in state.events)
