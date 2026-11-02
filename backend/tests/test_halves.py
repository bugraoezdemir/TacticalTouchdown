"""
Two halves, with the teams changing ends at the interval.

Direction is the single source of truth for which way a team attacks:
apply_formation rebuilds every assigned_pos from it and the decision layer reads
it for the attacking goal, so flipping it and reforming is the whole change.
"""
import numpy as np

from app.engine.constants import HALF_LENGTH_TICKS
from app.engine.factory import build_test_state
from app.engine.game_loop import tick


def run_to_half_time(seed=0):
    state = build_test_state(seed=seed)
    before = {
        "home_dir": state.home_team.direction,
        "away_dir": state.away_team.direction,
        "home_stations": [p.assigned_pos.copy() for p in state.home_team.players],
        "first_kickoff": state.first_kickoff_team_id,
    }
    while state.half == 1 and state.tick <= HALF_LENGTH_TICKS + 200:
        state = tick(state)
    return state, before


def test_the_half_turns_over_at_the_configured_length():
    state, _ = run_to_half_time()
    assert state.half == 2
    assert state.tick >= HALF_LENGTH_TICKS


def test_both_teams_change_ends():
    state, before = run_to_half_time()
    assert state.home_team.direction == -before["home_dir"]
    assert state.away_team.direction == -before["away_dir"]


def test_stations_are_mirrored_lengthwise():
    """The formation is rebuilt from the new direction rather than patched, so
    every station reflects across the halfway line."""
    from app.engine.constants import PITCH_LENGTH

    state, before = run_to_half_time()
    for now, was in zip(state.home_team.players, before["home_stations"]):
        assert float(now.assigned_pos[0]) == \
            np.float64(PITCH_LENGTH - float(was[0])).item()


def test_the_other_side_kicks_off_the_second_half():
    """Whoever started the match does not also restart it: taking every
    kick-off was previously worth roughly a 6-to-1 goal advantage."""
    state, before = run_to_half_time()
    owner = state.get_ball_owner()
    assert owner is not None
    assert owner.team_id != before["first_kickoff"]


def test_the_score_survives_the_interval():
    state, _ = run_to_half_time()
    # Scores are cumulative across the match, not reset per half.
    assert state.home_score >= 0 and state.away_score >= 0
    assert any(e.kind == "half_time" for e in state.events)


def test_the_second_half_plays_on():
    state, _ = run_to_half_time()
    tick_at_break = state.tick
    for _ in range(300):
        state = tick(state)
    assert state.half == 2
    assert state.tick > tick_at_break


def test_half_time_happens_once():
    state, _ = run_to_half_time()
    for _ in range(400):
        state = tick(state)
    assert sum(1 for e in state.events if e.kind == "half_time") <= 1


# -- Full time ----------------------------------------------------------------

def test_the_match_finishes_after_two_halves():
    state = build_test_state(seed=0)
    for _ in range(2 * HALF_LENGTH_TICKS + 400):
        state = tick(state)
        if state.finished:
            break
    assert state.finished
    assert state.half == 2
    assert any(e.kind == "full_time" for e in state.events)


def test_nothing_moves_after_full_time():
    """The final positions and stats must be exactly what the last live tick
    left, so a finished match can be inspected without it drifting."""
    state = build_test_state(seed=0)
    for _ in range(2 * HALF_LENGTH_TICKS + 400):
        state = tick(state)
        if state.finished:
            break
    assert state.finished

    before = (
        state.tick,
        state.ball.pos.copy(),
        state.home_score,
        state.away_score,
        state.stats.home.passes,
        state.stats.away.passes,
    )
    for _ in range(200):
        state = tick(state)

    assert state.tick == before[0]
    assert np.allclose(state.ball.pos, before[1])
    assert (state.home_score, state.away_score) == before[2:4]
    assert (state.stats.home.passes, state.stats.away.passes) == before[4:]
    assert not any(
        float(np.hypot(*p.velocity)) > 1e-9 for p in state.all_players()
    )


def test_full_time_is_recorded_once():
    state = build_test_state(seed=0)
    for _ in range(2 * HALF_LENGTH_TICKS + 600):
        state = tick(state)
    assert sum(1 for e in state.events if e.kind == "full_time") == 1


# -- The displayed clock ------------------------------------------------------

def test_the_clock_reads_zero_to_forty_five_in_each_half():
    """Display only: 15000 ticks are shown as 45 minutes so a half reads like
    football. The simulation itself is unscaled -- every speed and friction
    constant stays in m/tick at 30 Hz."""
    from app.api.routes.simulation import _clock

    state = build_test_state(seed=0)
    for tick_value, half, expected in [
        (0, 1, (0, 0)),
        (HALF_LENGTH_TICKS // 2, 1, (22, 30)),
        (HALF_LENGTH_TICKS, 1, (45, 0)),
        (HALF_LENGTH_TICKS + 1, 2, (0, 0)),
        (2 * HALF_LENGTH_TICKS, 2, (45, 0)),
    ]:
        state.tick, state.half = tick_value, half
        assert _clock(state) == expected, f"tick {tick_value} half {half}"


def test_the_clock_never_overruns_the_half():
    """A finished match reads 45'00, not beyond."""
    from app.api.routes.simulation import _clock

    state = build_test_state(seed=0)
    state.half, state.tick = 2, 3 * HALF_LENGTH_TICKS
    assert _clock(state) == (45, 0)
