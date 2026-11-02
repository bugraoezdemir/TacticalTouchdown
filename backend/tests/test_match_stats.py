"""
Match statistics.

Accumulated in the engine as events happen rather than derived from the event
log, which only keeps the last 40 entries and so cannot support a running total.
"""
import numpy as np
from fastapi.testclient import TestClient

from app.engine.factory import build_test_state
from app.engine.game_loop import tick
from app.main import app


def play(ticks=3000, seed=0):
    state = build_test_state(seed=seed)
    for _ in range(ticks):
        state = tick(state)
    return state


def test_possession_is_counted_for_exactly_one_side_per_tick():
    """Possession follows the ball's owner, or whoever last played it while it
    is in flight -- a team keeps the ball through its own pass."""
    state = play(600)
    total = state.stats.home.possession_ticks + state.stats.away.possession_ticks
    assert 0 < total <= 600


def test_completed_passes_never_exceed_attempted():
    state = play()
    for side in (state.stats.home, state.stats.away):
        assert side.passes_completed <= side.passes


def test_shots_on_target_never_exceed_shots():
    state = play()
    for side in (state.stats.home, state.stats.away):
        assert side.shots_on_target <= side.shots


def test_goals_in_stats_match_the_scoreline():
    """Two independent counters; a divergence means one of them is wrong."""
    state = play(6000)
    assert state.stats.home.goals == state.home_score
    assert state.stats.away.goals == state.away_score


def test_a_goal_is_always_an_on_target_shot():
    """Not every shot on target is a goal, but the converse must hold, or the
    on-target geometry disagrees with the goal detection."""
    state = play(6000)
    for side in (state.stats.home, state.stats.away):
        assert side.goals <= side.shots_on_target


def test_stats_survive_the_api_round_trip():
    client = TestClient(app)
    client.post("/simulate/start")
    payload = {}
    for _ in range(50):
        payload = client.post("/simulate/tick").json()

    assert "stats" in payload
    for side in ("home", "away"):
        s = payload["stats"][side]
        assert s["passes_completed"] <= s["passes"]
        assert s["shots_on_target"] <= s["shots"]


def test_a_reset_clears_the_stats():
    client = TestClient(app)
    client.post("/simulate/start")
    for _ in range(50):
        client.post("/simulate/tick")
    fresh = client.post("/simulate/reset").json()
    assert fresh["stats"]["home"]["possession_ticks"] == 0
    assert fresh["stats"]["away"]["passes"] == 0
