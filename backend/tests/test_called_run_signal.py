"""The called run: signal lifetime, and meeting the ball rather than the spot.

This mechanism was dead for the engine's entire history -- 421 tests passed
whether it worked, was one-tick, or never fired at all -- so every assertion
here is written to FAIL against one of the four defects that were found:

  1. clear_all_signals ran at the end of compute, wiping signals emitted in the
     same tick's apply phase: 0 of 85,665 off-ball decisions ever saw one.
  2. An age test of `emitted_tick < state.tick` discarded them one tick later,
     which is the same bug: 177 emitted, 177 set, 177 cleared unread.
  3. Signals lasted a single tick, so the runner took one ~0.25 m step and the
     shape anchor walked him back.
  4. The branch aimed at the signal's FIXED coordinate, so on arrival his target
     was his own feet: he stopped dead and let the ball roll past. 1810 of 5472
     live-signal ticks were spent parked with the ball moving 3.6 m away.
"""
import numpy as np
import pytest

from app.engine.constants import RUNWAY_MEET_RADIUS, RUNWAY_SIGNAL_MAX_TICKS
from app.engine.decisions.off_ball import off_ball_decision
from app.engine.factory import build_test_state
from app.engine.game_loop import tick
from app.engine.signals import clear_all_signals, emit_signal


def _runner(state, team):
    """An outfield player who is not the carrier."""
    return next(
        p for p in team.players
        if p.position != "GK" and not p.has_ball
    )


def test_signal_survives_the_next_decision_pass():
    """Defects 1 and 2: emitted in apply, it must live to be read in compute.

    emit_signal stamps emitted_tick = state.tick during APPLY of tick N, and
    state.tick is incremented at the END of that tick. So the signal must
    survive the compute where state.tick == N + 1.
    """
    state = build_test_state(seed=3)
    # A fresh state is at kickoff, where someone already HAS the ball -- and an
    # owned ball legitimately ends a called run. Loosen it first, or this tests
    # the ownership rule rather than the lifetime one.
    state.ball.owner_id = None
    runner = _runner(state, state.home_team)
    emit_signal(state, sender_id=1, recipient_id=runner.player_id,
                target_pos=np.array([60.0, 30.0]))

    state.tick += 1              # as game_loop does at the end of the tick
    clear_all_signals(state)     # start of the next compute

    assert runner.runway_signal is not None, (
        "signal cleared before any decision could read it"
    )


def test_signal_persists_while_the_ball_is_loose():
    """Defect 3: the run outlives the tick it was called on.

    The user's rule: keep running as long as nobody has touched the ball.
    """
    state = build_test_state(seed=3)
    state.ball.owner_id = None
    runner = _runner(state, state.home_team)
    emit_signal(state, sender_id=1, recipient_id=runner.player_id,
                target_pos=np.array([60.0, 30.0]))

    for _ in range(30):
        state.tick += 1
        clear_all_signals(state)

    assert runner.runway_signal is not None, (
        "signal died while the ball was still loose"
    )


def test_signal_dies_once_anyone_owns_the_ball():
    state = build_test_state(seed=3)
    state.ball.owner_id = None
    runner = _runner(state, state.home_team)
    emit_signal(state, sender_id=1, recipient_id=runner.player_id,
                target_pos=np.array([60.0, 30.0]))

    state.tick += 1
    clear_all_signals(state)
    assert runner.runway_signal is not None

    state.ball.owner_id = state.away_team.players[0].player_id
    state.tick += 1
    clear_all_signals(state)
    assert runner.runway_signal is None, (
        "the run should end the moment the ball is collected"
    )


def test_signal_expires_at_the_safety_cap():
    state = build_test_state(seed=3)
    state.ball.owner_id = None
    runner = _runner(state, state.home_team)
    emit_signal(state, sender_id=1, recipient_id=runner.player_id,
                target_pos=np.array([60.0, 30.0]))

    state.tick += int(RUNWAY_SIGNAL_MAX_TICKS) + 2
    clear_all_signals(state)

    assert runner.runway_signal is None, "a never-resolving ball stranded the runner"


def test_runner_heads_for_the_spot_while_still_far_away():
    """While running, the SPOT is the target -- that is the space he was called
    into. Chasing the ball's current position instead would abandon it."""
    state = build_test_state(seed=3)
    team = state.home_team
    runner = _runner(state, team)
    spot = np.array([float(runner.pos[0]) + 25.0 * team.direction, 34.0])
    emit_signal(state, sender_id=1, recipient_id=runner.player_id,
                target_pos=spot)

    action = off_ball_decision(runner, state, [])

    assert np.linalg.norm(action.target_pos - spot) < 1.0, (
        "a distant runner should be heading for the called spot"
    )


def test_runner_tracks_the_ball_once_he_reaches_the_spot():
    """Defect 4, the one that actually cost goals.

    Standing on the spot, his target must be the BALL's path, not the spot he
    is already on -- otherwise he stops dead and it rolls past him.
    """
    state = build_test_state(seed=3)
    team = state.home_team
    runner = _runner(state, team)

    spot = runner.pos.copy()                      # he has arrived
    # Ball passing nearby, still moving, not yet at his feet.
    state.ball.owner_id = None
    state.ball.pos = spot + np.array([6.0, 0.0])
    state.ball.velocity = np.array([-0.4, 0.0])
    state.ball.state = "in_pass"

    emit_signal(state, sender_id=1, recipient_id=runner.player_id,
                target_pos=spot)

    action = off_ball_decision(runner, state, [])

    assert np.linalg.norm(runner.pos - spot) <= RUNWAY_MEET_RADIUS
    assert np.linalg.norm(action.target_pos - spot) > 0.5, (
        "runner parked on the called spot instead of moving to meet the ball"
    )


def test_called_runners_close_on_the_spot_in_play():
    """Behavioural: over a real match, a runner still EN ROUTE must move toward
    the called spot far more often than away from it.

    Scoped to the running phase deliberately. Once he is inside
    RUNWAY_MEET_RADIUS he switches to tracking the ball, so moving away from the
    spot there is the fix working, not a regression -- measured 418 closing vs
    88 en route (4.8:1), against 1094/425 after arrival. Counting both phases
    together would assert against pre-fix behaviour.
    """
    state = build_test_state(seed=3)
    state.home_team.tactics.pressing = 1.6

    from app.engine.decisions import off_ball as ob
    import app.engine.game_loop as gl

    closing = drifting = 0
    previous: dict[int, float] = {}
    original = ob.off_ball_decision

    def wrapped(player, st, *args, **kwargs):
        nonlocal closing, drifting
        sig = player.runway_signal
        if sig is not None:
            d = float(np.linalg.norm(player.pos - sig.target_pos))
            was = previous.get(player.player_id)
            if was is not None and was > RUNWAY_MEET_RADIUS:
                if d < was - 1e-9:
                    closing += 1
                elif d > was + 1e-9:
                    drifting += 1
            previous[player.player_id] = d
        else:
            previous.pop(player.player_id, None)
        return original(player, st, *args, **kwargs)

    ob.off_ball_decision = wrapped
    gl.off_ball_decision = wrapped
    try:
        for _ in range(4000):
            if state.finished:
                break
            tick(state)
    finally:
        ob.off_ball_decision = original
        gl.off_ball_decision = original

    assert closing + drifting > 50, "no called runs happened; the path is dead again"
    assert closing > 3 * drifting, (
        f"called runners are not closing on the spot: {closing} closing "
        f"vs {drifting} drifting"
    )
