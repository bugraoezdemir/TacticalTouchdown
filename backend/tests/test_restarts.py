"""
Restarts: reforming and pausing.

Two things a dead ball needs that the engine did not do. Play resumed on the
very next tick, which is unreadable; and nobody returned to their station, so a
kick-off restarted with both teams still strung out from the move that had just
ended.
"""
import numpy as np
import pytest

from app.engine.constants import (
    CORNER_PAUSE_TICKS,
    GOAL_KICK_PAUSE_TICKS,
    DEAD_BALL_TICKS,
    GOAL_PAUSE_TICKS,
    KICKOFF_PAUSE_TICKS,
    THROW_IN_PAUSE_TICKS,
)
from app.engine.factory import build_test_state
from app.engine.game_loop import tick
from app.engine.physics import resolve_possession


def scatter(state):
    """Strew everyone across midfield, far from their stations."""
    for i, p in enumerate(state.all_players()):
        p.pos = np.array([50.0 + i * 0.5, 20.0])
        p.velocity = np.zeros(2)


def away_from_station(players) -> int:
    return sum(1 for p in players if float(np.hypot(*(p.pos - p.assigned_pos))) > 0.01)


def score_goal(state):
    """The ball goes in; it then rests in the net for GOAL_PAUSE_TICKS."""
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.home_team.team_id
    state.ball.pos = np.array([105.5, 34.0])
    resolve_possession(state)


def force_goal(state):
    """A goal, played on to the kick-off it leads to."""
    score_goal(state)
    for _ in range(GOAL_PAUSE_TICKS):
        state = tick(state)


def _out(state, team_id, pos):
    """The ball goes out; then the dead-ball beat, and the restart laid out."""
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = team_id
    state.ball.pos = np.array(pos)
    resolve_possession(state)
    for _ in range(DEAD_BALL_TICKS):
        state = tick(state)


def force_goal_kick(state):
    """Away attacks leftward, so away putting it out at x<0 is a goal kick."""
    _out(state, state.away_team.team_id, [-1.0, 20.0])


def force_throw_in(state):
    _out(state, state.home_team.team_id, [50.0, -1.0])


def force_corner(state):
    _out(state, state.home_team.team_id, [-1.0, 20.0])


def test_the_ball_is_seen_out_before_the_restart_is_laid_out():
    """Moved to the spot on the tick it crossed the line, the ball was never
    seen to go out and the throw-in message came first."""
    state = build_test_state(seed=0)
    scatter(state)
    positions = {p.player_id: p.pos.copy() for p in state.all_players()}
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.home_team.team_id
    state.ball.pos = np.array([50.0, -0.5])
    state.ball.velocity = np.array([0.0, -0.5])
    resolve_possession(state)
    assert [e.kind for e in state.events][-1] == "throw_in"   # announced at once
    for _ in range(DEAD_BALL_TICKS - 1):
        state = tick(state)
        assert state.ball.pos[1] < 0.0                        # still out of play
        for p in state.all_players():
            assert np.allclose(p.pos, positions[p.player_id])
    state = tick(state)
    taker = state.get_ball_owner()
    assert np.allclose(state.ball.pos, [50.0, 0.0])
    assert np.allclose(taker.pos, [50.0, 0.0])
    assert state.restart_pause == THROW_IN_PAUSE_TICKS


# -- Reforming ---------------------------------------------------------------

def test_kickoff_restations_both_teams():
    state = build_test_state(seed=0)
    scatter(state)
    force_goal(state)

    # Only the kick-off taker is off his station, standing on the centre spot.
    assert away_from_station(state.all_players()) == 1
    taker = state.get_ball_owner()
    assert taker is not None
    assert np.allclose(taker.pos, [52.5, 34.0])


def test_goal_kick_restations_both_teams():
    """Restationing only the defending side left the attackers wherever the
    move had stranded them: cleared out of the box, but bunched on its edge in
    a wall the keeper had to play through -- 9 of his 10 options scored P=0.000,
    and the giveaways that followed in his own third produced goals. A goal
    kick is a reset for both teams, like a kick-off."""
    state = build_test_state(seed=0)
    scatter(state)
    force_goal_kick(state)

    # Only the taker is off his station, standing on the ball.
    assert away_from_station(state.home_team.players) <= 1
    assert away_from_station(state.away_team.players) == 0


def test_throw_in_does_not_reform():
    """A throw-in restarts play from where it stopped, so sending everyone home
    would look wrong."""
    state = build_test_state(seed=0)
    scatter(state)
    force_throw_in(state)
    assert away_from_station(state.all_players()) >= len(state.all_players()) - 1


def test_restationing_clears_sticky_targets():
    """The sticky bonus would otherwise pull players straight back out of shape."""
    state = build_test_state(seed=0)
    scatter(state)
    for p in state.all_players():
        p.last_off_ball_target = np.array([50.0, 20.0])
    force_goal(state)
    assert all(p.last_off_ball_target is None for p in state.all_players())


def test_restationing_stops_players():
    state = build_test_state(seed=0)
    scatter(state)
    for p in state.all_players():
        p.velocity = np.array([0.3, 0.3])
    force_goal(state)
    for p in state.all_players():
        assert np.allclose(p.velocity, 0.0)


# -- Pausing -----------------------------------------------------------------

@pytest.mark.parametrize("force,expected", [
    (force_goal, KICKOFF_PAUSE_TICKS - GOAL_PAUSE_TICKS),
    (force_goal_kick, GOAL_KICK_PAUSE_TICKS),
    (force_throw_in, THROW_IN_PAUSE_TICKS),
    (force_corner, CORNER_PAUSE_TICKS),
])
def test_each_restart_pauses(force, expected):
    state = build_test_state(seed=0)
    scatter(state)
    force(state)
    assert state.restart_pause == expected


def test_pause_counts_down_and_ends():
    state = build_test_state(seed=0)
    force_throw_in(state)
    assert state.restart_pause == THROW_IN_PAUSE_TICKS

    for _ in range(THROW_IN_PAUSE_TICKS):
        state = tick(state)
    assert state.restart_pause == 0


def test_the_clock_still_runs_during_a_pause():
    """Otherwise the pause could never end and the match would hang."""
    state = build_test_state(seed=0)
    force_goal(state)
    before = state.tick
    for _ in range(10):
        state = tick(state)
    assert state.tick == before + 10


def test_the_ball_is_frozen_during_a_pause():
    state = build_test_state(seed=0)
    force_goal(state)
    ball_pos = state.ball.pos.copy()
    for _ in range(20):
        state = tick(state)
    assert np.allclose(state.ball.pos, ball_pos)


def test_players_do_not_move_during_a_pause():
    state = build_test_state(seed=0)
    force_goal(state)
    positions = {p.player_id: p.pos.copy() for p in state.all_players()}
    for _ in range(20):
        state = tick(state)
    for p in state.all_players():
        assert np.allclose(p.pos, positions[p.player_id])


def test_the_ball_rests_in_the_net_before_the_kick_off():
    """Set up for the kick-off on the tick it crossed the line, the ball was
    never seen in the goal and the goal banner seemed to come first."""
    state = build_test_state(seed=0)
    scatter(state)
    score_goal(state)
    positions = {p.player_id: p.pos.copy() for p in state.all_players()}
    for _ in range(GOAL_PAUSE_TICKS - 1):
        state = tick(state)
        assert state.ball.pos[0] > 105.0 and state.ball.owner_id is None
    for p in state.all_players():
        assert np.allclose(p.pos, positions[p.player_id])
    state = tick(state)
    assert np.allclose(state.ball.pos, [52.5, 34.0])
    assert state.restart_pause == KICKOFF_PAUSE_TICKS - GOAL_PAUSE_TICKS


def test_play_resumes_after_the_pause():
    state = build_test_state(seed=0)
    force_goal(state)
    for _ in range(KICKOFF_PAUSE_TICKS + 60):
        state = tick(state)
    assert state.restart_pause == 0
    # Somebody has moved or the ball has been played.
    moved = any(
        float(np.hypot(*p.velocity)) > 1e-9 for p in state.all_players()
    )
    assert moved or state.ball.state != "stationary"


@pytest.mark.slow
def test_pauses_do_not_dominate_the_match():
    """A restart should be a beat, not a stoppage-riddled match."""
    state = build_test_state(seed=0)
    paused = 0
    total = 6000
    for _ in range(total):
        if state.restart_pause > 0:
            paused += 1
        state = tick(state)
    assert paused / total < 0.15
