from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import GameState

# Attack call as joining an attack (decision D-13): a capped number of
# joiners, each on a run to a chosen spot with an end (playmaking.
# choose_attack_runners, expire_attack_calls). FB_JOIN=0 restores the old
# calls: a +20 m resting spot until the ball is lost.
JOIN = os.environ.get("FB_JOIN", "1") != "0"


@dataclass(slots=True)
class RunwaySignal:
    sender_id: int
    target_pos: np.ndarray  # where the recipient should run to
    emitted_tick: int


@dataclass(slots=True)
class AttackCall:
    """A call to break forward and support the attack.

    Unlike RunwaySignal, which is consumed the tick after it is emitted because
    the ball is already travelling to a specific spot, this one PERSISTS: the
    run lasts as long as the move does. It carries no target -- the runner
    picks his own point, the call only tells him to look upfield.
    """
    sender_id: int
    team_id: int
    emitted_tick: int
    # Under JOIN: the spot the carrier sent him to, and since when he has been
    # on it. (`ahead`: a lead kept on the ball instead, when not None.)
    spot: "np.ndarray | None" = None
    ahead: float | None = None
    arrived_tick: int | None = None
    # His closest approach to the spot so far, and when he last improved it
    # by JOIN_CLOSE_MIN_M: a run that stops closing cannot make it.
    best_gap: float = float("inf")
    best_tick: int = 0


def clear_all_signals(state: GameState) -> None:
    """Expire runway signals that have already had their decision pass.

    Attack calls have their own possession-scoped lifetime and are expired by
    expire_attack_calls, not wiped here.

    AGE-BASED, deliberately. Signals are emitted in APPLY (game_loop) and read
    by the NEXT tick's COMPUTE, so a blanket wipe destroys them before they are
    ever read -- whichever end of the tick it runs at:

      * wiping at the end of COMPUTE (the original bug) cleared signals emitted
        in the same tick's APPLY before any decision saw them, so
        off_ball's runway_signal branch was unreachable. Measured: called
        runners ended 2.5-3.0 m FURTHER from the spot than when the ball was
        played, and reached it 8-17% of the time -- the carrier was passing to
        a place nobody was going.
      * wiping at the start of COMPUTE would do exactly the same thing to the
        signal emitted one tick earlier.

    The run lasts until the ball's fate is DECIDED -- until someone, anyone,
    has it. A called runner has 12-28 m to cover, so a signal that survives one
    decision pass is no signal at all: measured, 159 of 165 signals lasted
    exactly 1 tick, the runner took a single ~0.25 m step toward the spot and
    the ordinary model then walked him back to his anchor for the other 59
    ticks. Net movement was 2.5-3.0 m AWAY from the spot he was called to.

    Keyed on ball.owner_id, NOT last_toucher_team_id. The latter is only
    assigned when a player *kicks* the ball and is never updated when someone
    *collects* one, so it lags by up to 91 ticks -- see expire_attack_calls,
    which was bitten by exactly that. _give_possession is the single choke
    point for every collection, tackle, throw-in, corner and kickoff, and it
    always sets owner_id.

    So: loose or in flight -> keep running. Anyone owns it -> the run is over,
    whether a teammate collected it or an opponent cut it out. Plus a tick cap
    as a safety valve, mirroring ATTACK_CALL_MAX_TICKS, so a ball that never
    resolves cleanly cannot strand a runner forever.

    Mind the off-by-one, which bit once already: emit_signal stamps
    emitted_tick = state.tick during APPLY of tick N, and state.tick is
    incremented to N+1 at the END of that same tick. The signal must therefore
    survive the COMPUTE where state.tick == N+1 (age 1). A
    `emitted_tick < state.tick` test reads N < N+1 and discards it before any
    decision sees it -- 177 signals emitted, 177 set, 177 cleared unread.
    """
    from .constants import RUNWAY_SIGNAL_MAX_TICKS

    ball_is_owned = state.ball.owner_id is not None
    for player in state.all_players():
        sig = player.runway_signal
        if sig is None:
            continue
        age = state.tick - sig.emitted_tick
        if age < 1:
            continue          # emitted this tick; it has not had its pass yet
        if ball_is_owned or age > RUNWAY_SIGNAL_MAX_TICKS:
            player.runway_signal = None


def expire_attack_calls(state: GameState) -> None:
    """End the attack when the move ends.

    The run lasts until an opponent wins the ball -- that is what a break IS --
    with a long tick cap purely as a safety valve so a player cannot be
    stranded upfield if possession never resolves cleanly. Measured possession
    spells: median 43 ticks, p95 163, longest 282, so the cap is above even the
    longest spell in practice and possession loss is what actually ends runs.
    """
    from .constants import ATTACK_CALL_MAX_TICKS, JOIN_COOLDOWN_TICKS

    # Who has the ball NOW -- not who kicked it last. last_toucher_team_id is
    # only assigned when a player *kicks* the ball and is never updated when
    # someone *collects* one, so after an opponent intercepts it still names
    # the team that played the pass. Keying the expiry on it left calls alive
    # while the opponent owned the ball on 7.2% of call-ticks (99% of those
    # traceable to exactly this lag, which reached 91 ticks), stranding players
    # a mean of 14 m and up to 55 m upfield of their own station.
    owner = state.get_ball_owner()
    holder = (
        owner.team_id if owner is not None else state.ball.last_toucher_team_id
    )
    for player in state.all_players():
        call = player.attack_call
        if call is None:
            continue
        if state.tick - call.emitted_tick > ATTACK_CALL_MAX_TICKS:
            player.attack_call = None
        elif holder is not None and holder != call.team_id:
            # The opponent has the ball: stop attacking, get back to shape.
            player.attack_call = None
        elif JOIN and call.spot is not None and _join_over(player, call, state, owner):
            player.attack_call = None
            player.join_cooldown_until = state.tick + JOIN_COOLDOWN_TICKS


def join_spot_now(call, state: GameState) -> np.ndarray:
    """Where a joiner is running to this tick: his channel (the y he was called
    to), call.ahead metres ahead of the ball, held onside. The spot moves on
    with the ball: fixed on the pitch, the ball overtook it and 86% of runs
    ended with the joiner 13 m behind the play."""
    from .constants import PITCH_LENGTH, TOUCHLINE_MARGIN
    from .offside import onside_limit

    team = state.get_team(call.team_id)
    if call.ahead is None:
        x = float(call.spot[0])
    else:
        x = float(state.ball.pos[0]) + team.direction * call.ahead
    line = onside_limit(state, team)
    if (x - line) * team.direction > 0.0:
        x = line
    x = min(max(x, TOUCHLINE_MARGIN), PITCH_LENGTH - TOUCHLINE_MARGIN)
    return np.array([x, float(call.spot[1])])


def _join_over(player, call, state: GameState, owner) -> bool:
    """A run to join the attack is over once it has done its job, the attack
    has stalled, or he cannot make it: he has the ball; he has been on his
    spot for JOIN_WAIT_TICKS without it coming -- which, with the spot moving
    on with the ball, only happens when the ball is not going forward; or his
    gap to the spot has not shrunk by JOIN_CLOSE_MIN_M for JOIN_CLOSE_TICKS."""
    from .constants import JOIN_ARRIVE_M, JOIN_CLOSE_MIN_M, JOIN_CLOSE_TICKS, JOIN_WAIT_TICKS

    if owner is not None and owner.player_id == player.player_id:
        return True
    spot = join_spot_now(call, state)
    gap = math.hypot(float(player.pos[0] - spot[0]), float(player.pos[1] - spot[1]))
    on_it = gap <= JOIN_ARRIVE_M
    if not on_it:
        call.arrived_tick = None
        if gap < call.best_gap - JOIN_CLOSE_MIN_M:
            call.best_gap, call.best_tick = gap, state.tick
        return state.tick - call.best_tick > JOIN_CLOSE_TICKS
    if call.arrived_tick is None:
        call.arrived_tick = state.tick
    return state.tick - call.arrived_tick > JOIN_WAIT_TICKS


@dataclass(slots=True)
class JoinPlan:
    """The carrier's runs this tick (playmaking._choose_joiners): ((player id,
    spot), ...) to be sent, every other run of his team dropped; or None when
    he did not judge them this tick."""
    runs: "tuple | None"


def set_attack_runs(state: GameState, sender_id: int, runs) -> None:
    """Apply the carrier's judgement: new runs are called, kept ones get the
    spot he chose (the clock restarting if it moved), and runs he no longer
    wants are dropped."""
    from .constants import JOIN_ARRIVE_M
    team_id = state.get_player(sender_id).team_id
    wanted = {pid: spot for pid, spot in runs}
    for player in state.get_team(team_id).players:
        call = player.attack_call
        if player.player_id in wanted:
            spot = np.asarray(wanted[player.player_id], dtype=float)
            if call is None or call.spot is None:
                emit_attack_call(state, sender_id, player.player_id, spot=spot)
            elif math.hypot(float(spot[0] - call.spot[0]), float(spot[1] - call.spot[1])) > JOIN_ARRIVE_M:
                call.spot = spot.copy(); call.arrived_tick = None
                call.best_gap, call.best_tick = float("inf"), state.tick
        elif call is not None and call.spot is not None:
            player.attack_call = None


def emit_attack_call(state: GameState, sender_id: int, recipient_id: int,
                     spot: "np.ndarray | None" = None, ahead: float | None = None) -> None:
    """Call one teammate forward. Refreshes an existing call rather than
    stacking, so a sustained attack keeps the run alive. Under JOIN the call
    carries the spot he runs to, and only players with no call are called."""
    try:
        recipient = state.get_player(recipient_id)
    except KeyError:
        return
    recipient.attack_call = AttackCall(
        sender_id=sender_id,
        team_id=recipient.team_id,
        emitted_tick=state.tick,
        spot=None if spot is None else np.asarray(spot, dtype=float).copy(),
        ahead=ahead,
        best_tick=state.tick,
    )


def emit_signal(
    state: GameState,
    sender_id: int,
    recipient_id: int,
    target_pos: np.ndarray,
) -> None:
    try:
        recipient = state.get_player(recipient_id)
        recipient.runway_signal = RunwaySignal(
            sender_id=sender_id,
            target_pos=target_pos.copy(),
            emitted_tick=state.tick,
        )
    except KeyError:
        pass
