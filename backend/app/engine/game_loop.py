from __future__ import annotations

import numpy as np

from .decisions.off_ball import OffBallAction, off_ball_decision
from .decisions.on_ball import OnBallAction, on_ball_decision
from .decisions.playmaking import choose_attack_runners
from .decisions.shape_call import update_shape_calls
from .formations import apply_formation
from .geometry import normalize
from .models import GameState, Player
from .constants import GOAL_PAUSE_TICKS, HALF_LENGTH_TICKS, HALF_TIME_PAUSE_TICKS, KICKOFF_PAUSE_TICKS
from .physics import (
    _kickoff_restart,
    apply_pending_places,
    step_dead_ball,
    apply_ball_update,
    clamp_players_to_pitch,
    control_radius_of,
    compute_ball_update,
    resolve_possession,
)
from .constants import DRIBBLE_NUDGE, DRIBBLE_TOUCH_INTERVAL, DRIBBLE_TOUCH_MODEL, SHIELD_BALL_OFFSET
from .constants import CONTROL_HEIGHT, JOG_SHARE, KICK_LOCKOUT_TICKS, PRESS_ARRIVE_TAUS, TARGET_MOTION_MAX
from .decisions.candidates import dribble_nudge, dribble_top_speed
from .decisions import pace as _pace
from .decisions import tick_cache
from . import kinematics
from .kinematics import desired_velocity, pursuit_velocity, step_velocity
from .offside import is_offside_position
from .signals import (
    clear_all_signals,
    JoinPlan,
    emit_attack_call,
    set_attack_runs,
    emit_signal,
    expire_attack_calls,
)

_BALL_STATE_MAP = {
    "shot": "in_shot",
    "pass": "in_pass",
    "dribble": "in_dribble",
}



def tick(state: GameState) -> GameState:
    """
    Advance the simulation by one tick.

    Phase 1 — COMPUTE: all decisions read from the unmodified current state.
    Phase 2 — APPLY:   all results are written simultaneously.

    While `restart_pause` is positive the match is held: the ball sits, nobody
    decides, and the clock still advances so the pause ends on its own. This is
    what gives a goal or a goal kick a readable beat instead of play snapping
    straight back on.
    """
    # The match is over: freeze everything. Checked before the restart pause so
    # a pause running at full time cannot tick the clock past the whistle.
    if state.finished:
        return state

    # Full time: the second half has run its length. Nothing moves afterwards,
    # so the final positions and stats are exactly what the last live tick left.
    if state.half == 2 and state.tick >= 2 * HALF_LENGTH_TICKS:
        _full_time(state)
        return state

    if state.restart_pause > 0:
        state.restart_pause -= 1
        state.tick += 1
        if state.dead_ball_motion is not None:
            step_dead_ball(state)
        if state.pending_places is not None and state.restart_pause <= state.pending_places_at:
            # The ball has been seen to go out: now the taker goes to it.
            apply_pending_places(state)
        if state.restart_pause == 0 and state.kickoff_pending:
            # The goal has been seen in the net: now both sides reform for
            # the kick-off, for the rest of the 3 s.
            state.kickoff_pending = False
            _kickoff_restart(state)
            state.restart_pause = KICKOFF_PAUSE_TICKS - GOAL_PAUSE_TICKS
        # Restarts place players by teleport. With inertia a velocity is real
        # momentum, so a player set down on a restart spot must not carry the
        # run he was making before the whistle.
        if kinematics.INERTIA:
            for player in state.all_players():
                player.velocity = np.zeros(2)
                player.prev_move_target = None
        return state

    # Half-time: teams change ends so no side keeps whatever edge the first
    # half gave it. Handled before the compute phase and it returns, so nobody
    # decides on a tick where their own stations are being rewritten.
    if state.half == 1 and state.tick >= HALF_LENGTH_TICKS:
        _half_time(state)
        state.tick += 1
        return state

    # ── Phase 1: Compute ──────────────────────────────────────────────────────

    # End any attack whose move is over before anyone decides on it, so the
    # runners' own decisions this tick already reflect it.
    expire_attack_calls(state)

    # Same discipline for runway signals: expire the stale ones BEFORE anyone
    # decides, never after. Clearing them at the end of COMPUTE wiped signals
    # emitted in the same tick's APPLY before a single decision could read
    # them, so the runway branch in off_ball was dead code -- called runners
    # drifted 2.5-3.0 m AWAY from the spot and arrived 8-17% of the time.
    # The clear is age-based (see signals.clear_all_signals), so a signal
    # emitted last tick survives exactly this one decision pass.
    clear_all_signals(state)

    # A keeper holding the ball is counting down to his distribution. Unlike a
    # restart pause the rest of the match carries on -- only he is waiting, and
    # the box stays clear because opponents were moved out when he gathered.
    if state.gk_hold_ticks > 0:
        state.gk_hold_ticks -= 1

    # Possession is measured on who the ball currently belongs to -- the owner
    # if someone has it, otherwise whoever last played it, so a team keeps
    # possession through its own pass rather than losing it mid-flight.
    _owner = state.get_ball_owner()
    _holder = (
        _owner.team_id if _owner is not None else state.ball.last_toucher_team_id
    )
    if _holder is not None:
        state.stats.side(state, _holder).possession_ticks += 1

    ball_delta = compute_ball_update(state.ball)

    # All decisions read the current snapshot, including runway signals emitted
    # in the PREVIOUS tick's apply phase and not yet expired above.
    # Players decide in a deterministic order so that each can condition on the
    # targets teammates have already committed to. Without that ordering, four
    # symmetric defenders each compute the same small marginal contribution,
    # all conclude someone else has it covered, and nobody goes. The first
    # player in the order faces an empty committed set, so somebody always does.
    #
    # Commitments live in this local dict and are never written to `state`, so
    # the two-phase guarantee holds: every decision still reads the frozen
    # snapshot.
    player_actions: dict[int, OffBallAction | OnBallAction] = {}
    # Decided here in COMPUTE, from the frozen snapshot, and emitted in APPLY:
    # resolve_possession runs before the emit block, so by then the carrier has
    # already released the ball and has_ball is false for 91% of ticks.
    attack_runners: tuple[int, ...] = ()
    attack_caller_id: int | None = None
    committed: dict[int, list[tuple[int, np.ndarray]]] = {
        state.home_team.team_id: [],
        state.away_team.team_id: [],
    }
    # Shared answers (zone centres, the marking assignment, the ball's path)
    # are worked out once for this frozen snapshot -- see tick_cache.
    tick_cache.open_epoch()
    try:
        for player in _decision_order(state):
            if player.has_ball and attack_caller_id is None:
                attack_caller_id = player.player_id
                attack_runners = choose_attack_runners(player, state)
            if player.has_ball:
                action = on_ball_decision(player, state)
            else:
                action = off_ball_decision(player, state, committed[player.team_id])
                committed[player.team_id].append((player.player_id, action.target_pos))
            player_actions[player.player_id] = action
    finally:
        tick_cache.close_epoch()

    # ── Phase 2: Apply ────────────────────────────────────────────────────────

    # 2a. Ball physics (only meaningful when ball is loose / in-flight)
    state.ball_prev_pos = state.ball.pos.copy()
    apply_ball_update(state.ball, ball_delta)

    # 2b. Player movements
    for player in state.all_players():
        action = player_actions[player.player_id]
        _apply_player_movement(player, action)
        player.marking_id = action.marking_id if isinstance(action, OffBallAction) else None

    # 2c. Ball owner actions
    for player in state.all_players():
        action = player_actions[player.player_id]
        if not isinstance(action, OnBallAction):
            continue
        if action.action_type == "hold":
            # Ball sticks to the player this tick
            if player.has_ball:
                # On the far side of his body from the presser, not at his toes.
                ahead = np.zeros(2)
                if action.move is not None:
                    n = float(np.linalg.norm(action.move))
                    if n > 1e-9:
                        ahead = action.move / n * SHIELD_BALL_OFFSET
                state.ball.pos = player.pos + ahead
                player.hold_ticks += 1
            continue
        if action.action_type == "control":
            if player.has_ball:
                _apply_control(player, state)
            continue
        player.hold_ticks = 0
        if player.has_ball:
            # Kicked this tick: no swept capture (see GameState.ball_prev_pos).
            state.ball_prev_pos = None
            state.ball.velocity = _apply_execution_error(
                action.velocity, action.action_type, player, state, lofted=action.vz > 0.0
            )
            # Lofted, or headed on from the height he met it at; else along
            # the ground.
            state.ball.vz = (_lift_error(action.vz, player, state)
                             if action.action_type == "shot" and action.vz > 0.0 else action.vz)
            state.ball.curl = _curl_error(action.curl, player, state) if action.curl != 0.0 else 0.0
            state.ball.height = (player.first_touch_height
                                 if player.first_touch_height > CONTROL_HEIGHT else 0.0)
            # Whatever he did with it, the first touch has been taken.
            player.first_touch_speed = 0.0
            player.first_touch_velocity = None
            player.first_touch_height = 0.0
            state.ball.state = _BALL_STATE_MAP[action.action_type]
            state.ball.is_cross = action.action_type == "pass" and action.is_cross
            if action.action_type != "pass":
                state.ball.intended_receiver_id = None
            state.ball.owner_id = None
            state.ball.last_toucher_team_id = player.team_id
            # He cannot take back the ball he has just struck: with captures at
            # a player's full reach (1 m) the kicker would otherwise catch his
            # own pass as it left his foot.
            state.ball.no_capture_id = player.player_id
            state.ball.no_capture_until = state.tick + KICK_LOCKOUT_TICKS
            state.ball.struck_from = state.ball.pos.copy()
            state.ball.struck_tick = state.tick
            _side = state.stats.side(state, player.team_id)
            # Per-player counters alongside the team ones. Attributed here
            # because this is the only place the KICKER is unambiguous -- by
            # the time the ball is resolved he may no longer own it.
            _mine = state.stats.player(state, player)
            if action.action_type == "shot":
                _side.shots += 1
                _mine.shots += 1
                if _shot_is_on_target(state, player):
                    _side.shots_on_target += 1
                    _mine.shots_on_target += 1
                # The only moment the finisher is knowable: once struck, the
                # ball has no owner and no passer until it is next collected.
                state.ball.last_shooter_id = player.player_id
            elif action.action_type == "pass":
                _side.passes += 1
                _mine.passes += 1
                state.ball.last_shooter_id = None
                # Freeze who was offside at the moment it left his foot. The
                # line moves while the ball travels, so judging on arrival
                # would flag players who were onside when it was played and
                # clear ones who were not.
                _team = state.get_team(player.team_id)
                state.ball.intended_receiver_id = action.receiver_id
                if action.runway_signal_target is not None:
                    state.ball.pass_target = action.runway_signal_target.copy()
                elif action.pass_point is not None:
                    state.ball.pass_target = action.pass_point.copy()
                elif action.receiver_id is not None:
                    state.ball.pass_target = state.get_player(action.receiver_id).pos.copy()
                else:
                    state.ball.pass_target = None
                if action.is_cross:
                    _side.crosses += 1
                state.ball.offside_suspects = frozenset(
                    p.player_id
                    for p in _team.players
                    if p.player_id != player.player_id
                    and is_offside_position(p, state, _team,
                                            ball_x=float(player.pos[0]))
                )
                # fall through
            elif action.action_type == "dribble":
                _mine.dribbles += 1
                state.ball.last_shooter_id = None
                # Remember who pushed it: the loss can only be attributed once
                # somebody else wins the contest, several ticks later.
                state.ball.last_dribbler_id = player.player_id
            if action.action_type == "pass":
                state.ball.last_passer_id = player.player_id
                state.ball.last_pass_tick = state.tick
            else:
                state.ball.last_passer_id = None
            player.has_ball = False
            # Advance ball far enough that it's outside the kicker's capture
            # radius — otherwise resolve_possession re-assigns it immediately.
            kick_dir = normalize(action.velocity)
            if float(np.linalg.norm(kick_dir)) > 1e-9:
                from .constants import CAPTURE_RADIUS, DRIBBLE_LOOKAHEAD_DIST
                if action.action_type == "dribble":
                    if DRIBBLE_TOUCH_MODEL:
                        # The touch was SOLVED in the decision; its length is
                        # carried by the ball's speed. Nudge it just clear of
                        # his feet and let it ROLL the rest -- a 6 m knock is a
                        # ball rolling into space, not one teleported there.
                        touch = action.touch or DRIBBLE_NUDGE
                        state.ball.pos = state.ball.pos + kick_dir * dribble_nudge(touch)
                        state.ball.dribble_meet = player.pos + kick_dir * touch
                        state.ball.dribble_meet_tick = state.tick + int(round(
                            max(DRIBBLE_TOUCH_INTERVAL, touch / max(dribble_top_speed(player), 1e-9))))
                        if _pace.PACE:
                            # His touch is his until the meeting point: a walking
                            # touch stays inside his 1 m reach, and taking it back
                            # the moment the kick lockout ended had him re-decide
                            # every 6 ticks, turning on the spot. Opponents can
                            # still win it in the meantime.
                            state.ball.no_capture_until = max(state.ball.no_capture_until,
                                                              state.ball.dribble_meet_tick - 1)
                    else:
                        # Push the ball ahead and chase it. The touch must land
                        # beyond CAPTURE_RADIUS or the "dribble" is a no-op, but
                        # a better dribbler keeps it closer to their feet.
                        state.ball.pos = state.ball.pos + kick_dir * _dribble_touch_distance(player)
                else:
                    # Shot / pass: advance at least beyond capture radius
                    advance = max(
                        float(np.linalg.norm(action.velocity)),
                        CAPTURE_RADIUS + 0.3,
                    )
                    state.ball.pos = state.ball.pos + kick_dir * advance

    # 2d. Possession resolution and goal/out-of-bounds detection
    resolve_possession(state)

    # 2e. Emit runway signals (recipients will pick them up next tick)
    for player in state.all_players():
        action = player_actions[player.player_id]
        if (
            isinstance(action, OnBallAction)
            and action.runway_signal_target is not None
        ):
            # A called run can name more than one runner: the carrier picks the
            # spot and sends the best-placed teammates to it.
            recipients = action.signal_runners or (
                (action.signal_recipient_id,) if action.signal_recipient_id else ()
            )
            for recipient_id in recipients:
                emit_signal(
                    state,
                    sender_id=player.player_id,
                    recipient_id=recipient_id,
                    target_pos=action.runway_signal_target,
                )

    # The carrier's call to break forward, decided in COMPUTE above. Emitted
    # here so it is read by next tick's decisions and never by one taken in the
    # same tick.
    if attack_caller_id is not None:
        if isinstance(attack_runners, JoinPlan):          # signals.JOIN
            if attack_runners.runs is not None:
                set_attack_runs(state, attack_caller_id, attack_runners.runs)
        else:
            for runner_id in attack_runners:
                emit_attack_call(state, attack_caller_id, runner_id)

    update_shape_calls(state)

    clamp_players_to_pitch(state)
    state.tick += 1
    return state


def _half_time(state: GameState) -> None:
    """Turn the teams around and restart from the centre.

    Direction is the single source of truth for which way a team attacks:
    apply_formation rebuilds every assigned_pos from it, and the decision layer
    reads it for the attacking goal. So flipping it and reforming is the whole
    change -- nothing else needs to know about ends.

    The team that did NOT kick off the match kicks off the second half.
    """
    state.half = 2
    state.shape_calls.clear()
    state.shape_calls_waiting.clear()
    for team in (state.home_team, state.away_team):
        team.direction *= -1
        apply_formation(team)

    # _kickoff_restart hands the ball to whoever did NOT last touch it, so
    # naming the first-half kickoff team here gives the second half to the
    # other side -- neither team takes both.
    state.ball.last_toucher_team_id = (
        state.first_kickoff_team_id
        if state.first_kickoff_team_id is not None
        else state.home_team.team_id
    )
    _kickoff_restart(state)
    # _kickoff_restart already set a kick-off pause; the interval gets its own
    # length so it reads as a break rather than another goal restart.
    state.restart_pause = HALF_TIME_PAUSE_TICKS
    state.add_event(
        "half_time", None, None,
        f"Half time: {state.home_team.name} "
        f"{state.home_score}-{state.away_score} {state.away_team.name}",
    )


def _full_time(state: GameState) -> None:
    """Blow the whistle. The state stops advancing from here."""
    state.finished = True
    state.restart_pause = 0
    state.ball.velocity = np.zeros(2)
    state.ball.state = "stationary"
    state.ball.ground()
    for player in state.all_players():
        player.velocity = np.zeros(2)
    state.add_event(
        "full_time", None, None,
        f"Full time: {state.home_team.name} "
        f"{state.home_score}-{state.away_score} {state.away_team.name}",
    )


def _shot_is_on_target(state: GameState, shooter: Player) -> bool:
    """Would this shot, as actually struck, cross the goal line inside the posts?

    Measured on the ball's post-error velocity rather than the aim point, so a
    shot dragged wide by a poor finisher is correctly off target. Whether a
    keeper then saves it is a separate question -- an on-target shot is one he
    has to deal with.
    """
    from .ball_flight import crossing, height_after
    from .constants import CROSSBAR_HEIGHT, GOAL_BOTTOM_Y, GOAL_TOP_Y, LEFT_GOAL_X, RIGHT_GOAL_X

    team = state.get_team(shooter.team_id)
    goal_x = RIGHT_GOAL_X if team.direction == 1 else LEFT_GOAL_X
    if state.ball.curl != 0.0:
        where = crossing(state.ball.pos, state.ball.velocity, state.ball.height, state.ball.vz,
                         state.ball.curl, goal_x)
        return where is not None and GOAL_TOP_Y <= where[0] <= GOAL_BOTTOM_Y and where[1] <= CROSSBAR_HEIGHT
    vx = float(state.ball.velocity[0])
    if abs(vx) < 1e-9:
        return False
    ticks = (goal_x - float(state.ball.pos[0])) / vx
    if ticks <= 0.0:
        return False  # struck away from the goal
    y = float(state.ball.pos[1]) + float(state.ball.velocity[1]) * ticks
    if not GOAL_TOP_Y <= y <= GOAL_BOTTOM_Y:
        return False
    return height_after(state.ball.height, state.ball.vz, ticks) <= CROSSBAR_HEIGHT


def _lift_error(vz: float, player: Player, state: GameState) -> float:
    """A lofted shot's lift, as struck: off by a skill-scaled share, which is
    what puts one over the bar. Noise on execution, as for the aim."""
    from .constants import SHOT_LIFT_ERROR_MAX

    quality = min(1.0, max(0.0, player.shooting / 100.0))
    return vz * max(0.0, 1.0 + float(state.rng.normal(0.0, SHOT_LIFT_ERROR_MAX * (1.0 - quality))))


def _curl_error(curl: float, player: Player, state: GameState) -> float:
    """A bent shot's spin, as struck: off by a skill-scaled share."""
    from .constants import SHOT_CURL_ERROR_MAX

    quality = min(1.0, max(0.0, player.shooting / 100.0))
    return curl * (1.0 + float(state.rng.normal(0.0, SHOT_CURL_ERROR_MAX * (1.0 - quality))))


def _apply_player_movement(player: Player, action: OffBallAction | OnBallAction) -> None:
    if kinematics.INERTIA:
        _apply_inertial_movement(player, action)
        return
    if isinstance(action, OnBallAction):
        # Ball owner stands still until they kick -- EXCEPT when shielding, where
        # he drifts away from the presser with the ball (see _shield_step).
        if action.action_type == "hold" and action.move is not None:
            player.velocity = action.move.copy()
            player.pos = player.pos + player.velocity
        else:
            player.velocity = np.zeros(2)
        return

    direction = action.target_pos - player.pos
    dist = float(np.linalg.norm(direction))
    if dist < 1e-9:
        player.velocity = np.zeros(2)
        return

    if action.pace is not None:
        speed = action.pace
    else:
        speed = player.speed if action.sprint else player.speed * JOG_SHARE
    step = min(dist, speed)
    player.velocity = (direction / dist) * step
    player.pos = player.pos + player.velocity


def _apply_inertial_movement(player: Player, action: OffBallAction | OnBallAction) -> None:
    """Movement under inertia: the action says where to go, physics says how.

    The carrier stands for the tick he strikes the ball, as before -- but he
    keeps his MOMENTUM through it. Under the touch model a dribbler owns the
    ball for one tick per touch, so zeroing his velocity there would make him
    restart from a standstill at every touch.
    """
    if isinstance(action, OnBallAction):
        player.prev_move_target = None
        if action.action_type == "hold" and action.move is not None:
            # Shielding: drift away from the presser, within the same limits.
            player.velocity = step_velocity(player.velocity, action.move, player.speed)
            player.pos = player.pos + player.velocity
        elif action.action_type == "hold":
            player.velocity = step_velocity(player.velocity, np.zeros(2), player.speed)
        return

    target = action.target_pos
    if action.dive:
        # A keeper's dive at a shot: an explosive launch, not a run, so no
        # acceleration ramp -- the same instant step the old movement used.
        offset = target - player.pos
        dist = float(np.linalg.norm(offset))
        step = min(dist, player.speed if action.pace is None else action.pace)
        player.velocity = offset / dist * step if dist > 1e-9 else np.zeros(2)
        player.pos = player.pos + player.velocity
        player.prev_move_target = None
        return
    if action.pace is not None:
        # A dribbler following his own touch runs THROUGH the meeting point
        # with the ball, so he does not brake to stop on it -- aimed the way
        # reach_ticks assumes, so he meets it when the touch was solved for.
        desired = pursuit_velocity(player.pos, player.velocity, target, action.pace,
                                   control_radius_of(player))
    elif action.through:
        # Attacking a loose ball: flat out through the meeting point, aimed
        # to get there when the decisions predicted he would.
        desired = pursuit_velocity(player.pos, player.velocity, target, player.speed,
                                   control_radius_of(player))
    elif action.press_velocity is not None:
        # Closing down the carrier, or tracking a marked man: hard, braking
        # late, on HIS motion.
        cap = player.speed if action.sprint else player.speed * JOG_SHARE
        desired = desired_velocity(
            player.pos, target, cap,
            target_velocity=action.press_velocity, arrive_taus=PRESS_ARRIVE_TAUS,
        )
    else:
        cap = player.speed if action.sprint else player.speed * JOG_SHARE
        desired = desired_velocity(
            player.pos, target, cap, target_velocity=_target_motion(player, target)
        )
    player.prev_move_target = target.copy()
    player.velocity = step_velocity(player.velocity, desired, player.speed)
    player.pos = player.pos + player.velocity


def _target_motion(player: Player, target: np.ndarray) -> np.ndarray | None:
    """How far his target moved since last tick, if it MOVED rather than jumped.

    A marker's spot slides with his man a fraction of a metre per tick; a
    re-chosen target leaps. Only the first is a velocity to match.
    """
    prev = player.prev_move_target
    if prev is None:
        return None
    moved = target - prev
    if float(np.hypot(moved[0], moved[1])) > TARGET_MOTION_MAX * player.speed:
        return None
    return moved


def _dribble_touch_distance(player: Player) -> float:
    """How far the ball escapes on a dribble touch.

    Must exceed CAPTURE_RADIUS, else the ball never leaves the player and the
    dribble does nothing -- that plus zero-value scoring is why the old engine
    produced no dribbles at all. Better dribblers keep it tighter.
    """
    from .constants import DRIBBLE_TOUCH_MAX, DRIBBLE_TOUCH_MIN

    quality = min(1.0, max(0.0, player.dribbling / 100.0))
    return DRIBBLE_TOUCH_MAX - (DRIBBLE_TOUCH_MAX - DRIBBLE_TOUCH_MIN) * quality


def _apply_execution_error(
    velocity: np.ndarray,
    action_type: str,
    player: Player,
    state: GameState,
    lofted: bool = False,
) -> np.ndarray:
    """Rotate the delivery by a skill-scaled random angle.

    Noise belongs on execution, not on the decision: a weak passer still chooses
    the right ball, they just fail to hit it. Noise on the *choice* would instead
    make players erratic, which is not the same thing.

    A lofted ball is harder to place, and its LENGTH can be off too: its
    horizontal pace is scaled, which moves where it lands.
    """
    from .constants import (
        LOFT_ANGLE_ERROR_GAIN,
        LOFT_RANGE_ERROR_MAX,
        PASS_ERROR_MAX_RAD,
        SHOT_ERROR_MAX_RAD,
    )

    if action_type == "pass":
        skill, max_err = player.passing, PASS_ERROR_MAX_RAD
    elif action_type == "shot":
        skill, max_err = player.shooting, SHOT_ERROR_MAX_RAD
    else:
        return velocity.copy()

    quality = min(1.0, max(0.0, skill / 100.0))
    sigma = max_err * (1.0 - quality)
    if player.first_touch_speed > 0.0:
        # Played first time off a fast ball: the quicker it came, the harder
        # it is to place -- and a header is never placed like a kick.
        from .constants import FIRST_TIME_ERROR_GAIN
        from .decisions.control import hardness

        hard = hardness(player.first_touch_speed, player)
        if player.first_touch_height > CONTROL_HEIGHT:
            hard = max(hard, 0.5)
        sigma *= 1.0 + FIRST_TIME_ERROR_GAIN * hard
    if lofted:
        sigma *= LOFT_ANGLE_ERROR_GAIN
    if sigma <= 0.0:
        return velocity.copy()

    out = _rotate(velocity, float(state.rng.normal(0.0, sigma)))
    if lofted:
        out = out * max(0.5, 1.0 + float(state.rng.normal(0.0, LOFT_RANGE_ERROR_MAX * (1.0 - quality))))
    return out


def _rotate(velocity: np.ndarray, angle: float) -> np.ndarray:
    c, s_ = np.cos(angle), np.sin(angle)
    return np.array([
        c * velocity[0] - s_ * velocity[1],
        s_ * velocity[0] + c * velocity[1],
    ])


def _apply_control(player: Player, state: GameState) -> None:
    """He tries to trap a ball that came in too fast.

    Clean (control_probability) -> it is at his feet and play goes on as
    normal. Otherwise it squirms loose off him: a fraction of its incoming pace,
    knocked off its line by up to MISCONTROL_SPREAD, and he cannot retouch it
    for MISCONTROL_LOCKOUT ticks.
    """
    from .constants import (
        CAPTURE_RADIUS,
        MISCONTROL_LOCKOUT,
        MISCONTROL_RESTITUTION,
        MISCONTROL_SPREAD,
    )
    from .decisions.control import trap_probability

    ball = state.ball
    p_trap = trap_probability(player)
    incoming = player.first_touch_velocity
    player.first_touch_speed = 0.0
    player.first_touch_velocity = None
    player.first_touch_height = 0.0
    if incoming is None or state.rng.random() < p_trap:
        ball.pos = player.pos.copy()      # trapped: his ball, at his feet
        return
    spread = float(state.rng.uniform(-MISCONTROL_SPREAD, MISCONTROL_SPREAD))
    loose = _rotate(np.asarray(incoming, float), spread) * MISCONTROL_RESTITUTION
    n = float(np.linalg.norm(loose))
    state.ball_prev_pos = None             # he touched it this tick
    player.has_ball = False
    ball.owner_id = None
    ball.velocity = loose
    ball.state = "in_pass" if n > 1e-9 else "stationary"
    ball.last_toucher_team_id = player.team_id
    ball.last_passer_id = None
    ball.intended_receiver_id = None
    if n > 1e-9:
        ball.pos = player.pos + loose / n * (CAPTURE_RADIUS + 0.3)
    ball.no_capture_id = player.player_id
    ball.no_capture_until = state.tick + MISCONTROL_LOCKOUT
    state.stats.player(state, player).touches += 1


def _decision_order(state: GameState) -> list[Player]:
    """Players sorted by how well placed they are to reach the ball.

    Ties break on player_id so the order is a pure function of the snapshot --
    reproducible regardless of roster list order.
    """
    ball_pos = state.ball.pos

    # Ties break on a per-match shuffled ranking rather than player_id: home
    # ids are 1-10 and away 12-21, so an id tie-break silently let a home
    # player commit first in every 50/50 race.
    rank = _tiebreak_rank(state)

    if kinematics.INERTIA:
        # Best placed = soonest there from how he is actually moving.
        everyone = state.all_players()
        movers = [p for p in everyone if p.speed > 1e-9]
        eta = {}
        if movers:
            t = kinematics.reach_ticks(
                np.array([p.pos for p in movers]), np.array([p.velocity for p in movers]),
                np.array([p.speed for p in movers]), ball_pos[None, :],
            )
            eta = {p.player_id: float(x) for p, x in zip(movers, t)}
        return sorted(
            everyone, key=lambda p: (eta.get(p.player_id, float("inf")), rank[p.player_id])
        )

    def key(p: Player) -> tuple[float, int]:
        if p.speed <= 1e-9:
            return (float("inf"), rank[p.player_id])
        return (float(np.linalg.norm(p.pos - ball_pos)) / p.speed, rank[p.player_id])

    return sorted(state.all_players(), key=key)


def _tiebreak_rank(state: GameState) -> dict[int, int]:
    """A stable, match-specific shuffle of player ids.

    Derived from the seeded RNG once per match and cached, so ordering stays
    deterministic for a given seed while carrying no home/away bias.
    """
    if state._tiebreak_rank is None:
        ids = [p.player_id for p in state.all_players()]
        order = state.rng.permutation(len(ids))
        state._tiebreak_rank = {pid: int(order[i]) for i, pid in enumerate(ids)}
    return state._tiebreak_rank
