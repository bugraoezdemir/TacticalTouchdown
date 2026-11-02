"""
Off-ball decisions.

Defensive behaviour is a PHASE, not a role: it is how any player behaves when
their team lacks the ball, strikers included. There is deliberately no per-role
branching here -- role enters only through the context multiplier and through
the shape leash, which is what stops a striker sprinting 40 m back.

Two objectives share one skeleton:
  - out of possession: aim at the speed/safety trade-off point (see pursuit.py),
    gated by marginal contribution so players do not all chase the same ball
  - in possession:     sample reachable points, score by position_value x
                       receivability -- "where can I get the ball and hurt them
                       from"
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from ..constants import SUPPORT_BEHIND_MARGIN, SUPPORT_SPRINT_GAP, TIMED_RUN_EARLY_TICKS, TIMED_RUN_MIN_SHARE
from ..constants import (
    COME_TO_BALL_FROM_TICKS,
    COME_TO_BALL_HORIZON,
    CROSSBAR_HEIGHT,
    GK_DIVE_SPEED,
    GK_HIGH_SHOT,
    GK_LEAVE_OVER_BAR,
    GK_REACTION_TICKS,
    BALL_FRICTION_COEFF,
    FUNNEL_MAX,
    FUNNEL_POWER,
    FUNNEL_REACH_FLOOR,
    FUNNEL_SCALE,
    GOAL_BOTTOM_Y,
    GOAL_TOP_Y,
    LEFT_GOAL_X,
    CLAIM_MARGIN_TICKS,
    DANGER_FAR,
    DANGER_NEAR,
    DANGER_PRESS_GAIN,
    GOAL_CENTER_Y,
    GOAL_SIDE_MIN_AHEAD,
    GOAL_SIDE_STANDOFF,
    CONTEST_MARGIN_TICKS,
    CONTEST_SAME_POINT_RADIUS,
    OFFBALL_RING_POINTS,
    ONE_TWO_PULL_BASE,
    ATTACK_CALL_FULL_RATE,
    ATTACK_CALL_MIN_RATE,
    ATTACK_RUN_RANGE,
    BOX_ATTACK_FROM,
    BOX_ATTACK_FULL,
    BOX_CLEARANCE,
    BOX_SLOTS,
    PENALTY_AREA_BOTTOM,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_TOP,
    OFFSIDE_FADE_DISTANCE,
    OFFSIDE_MIN_FACTOR,
    ONE_TWO_CROWD_PENALTY,
    ONE_TWO_STANDOFF,
    ONE_TWO_SUPPORT_RANGE,
    ONE_TWO_SAFE_ROUTE,
    ONE_TWO_SAFETY_TICKS,
    ONE_TWO_WINDOW_TICKS,
    OFFBALL_RING_RADII,
    PASS_SPEED_HIGH,
    RECEIVABILITY_FLOOR,
    PITCH_LENGTH,
    PITCH_WIDTH,
    RIGHT_GOAL_X,
    MEET_TARGET_HYSTERESIS,
    RUNWAY_MEET_RADIUS,
    ROLE_LEASH,
    SHAPE_BLOCK_SLIDE_X,
    SHAPE_BLOCK_SLIDE_Y,
    HOLD_BALL_PULL,
    HOLD_BALL_PULL_MAX,
    STICKY_BONUS,
    SUPPORT_MAX_BONUS,
    SUPPORT_RANGE,
    SPACING_SOFTNESS,
    TEAMMATE_SPACING_FLOOR,
    TEAMMATE_SPACING_RADIUS,
    TEAMMATE_SPREAD_STRENGTH,
    TACTICAL_LEASH,
    COMPACT_BASE_PULL,
    COMPACT_PRESSURE_FAR,
    COMPACT_PRESSURE_PULL,
    COMPACT_REACH,
    COMPACT_SUPPORT_DISTANCE,
    COMPACT_FULL,
    COMPACT_LEASH_GAIN,
    COMPACT_SLIDE_GAIN,
    COMPACT_SLIDE_NEUTRAL,
    RECOVER_BASE,
    RECOVER_DEPTH,
    RECOVER_SPRINT_GAP,
    RECOVER_TRIANGLE_MARGIN,
    JOG_SHARE,
    GK_ADVANCE,
    LINE_SHARE,
    LINE_FADE_FAR,
    STATION_MIN_FROM_GOAL,
    ZONE_RADIUS,
    ZONE_FULL,
    ZONE_POS_WEIGHT,
    ZONE_THREAT_FAR,
    ZONE_THREAT_NEAR,
    ZONE_MARK_FLOOR,
    MARK_STANDOFF_FAR,
    MARK_STANDOFF_NEAR,
    MARK_BALL_SIDE,
    MARK_SPRINT_GAP,
    MIN_PASS_DIST,
    FRONT_MAN_RUN,
    FRONT_MAN_TICKS,
    PRESS_EXTRA_PER_UNIT,
    PRESS_EXTRA_WINDOW,
    PRESS_FIRST_TICKS,
    PRESS_LANE_MIN_M,
    MARK_KEEP_CLOSENESS,
    DRIBBLE_TOUCH_MODEL,
    OUTFIELD_CONTROL_RADIUS,
    DRIBBLE_FRICTION_COEFF,
    HEADER_HEIGHT,
    RISING_REACH,
    PRESS_LOOKAHEAD_TICKS,
    PRESS_MAX,
    PRESS_REACH_TICKS,
    PRESS_TAU,
    TACTIC_BIAS_MAX,
    TICK_RATE,
)
from .. import kinematics
from ..ball_flight import airborne, height_after, trajectory, within_reach
from ..geometry import distance, normalize
from ..kinematics import reach_ticks
from ..offside import offside_line, onside_limit
from ..models import Ball, GameState, Player, Team
from ..ball_physics import nominal_pass_speed, travel_ticks
from .candidates import dribble_top_speed
from . import pace as _pace
from . import tick_cache
from .interception import caught_out, control_radius, route_survival, route_survival_many
from .marginal import marginal_contribution
from .shape_call import SHAPE_CALL, ShapeCall, called_station
from .pursuit import (
    COME_TO_BALL,
    _comfortably_ours,
    aim_point,
    best_meeting_point,
    best_pass_safety,
    closing_on_path,
    space_quality,
    support_need,
    ball_intercept_point,
    carrier_straightness,
    compute_lambda,
    opponent_outlets,
    prediction_confidence,
)
from .tunables import (
    compactness_bias,
    context_multiplier,
    defensive_bias,
    man_marking_bias,
    offensiveness_bias,
    one_two_bias,
    press_aggression_bias,
    roaming_bias,
    tempo_bias,
)
from .value import position_value, threat
from .zones import Zone, classify_zone


@dataclass(slots=True)
class OffBallAction:
    target_pos: np.ndarray
    sprint: bool
    # An explicit pace in m/tick, overriding sprint/jog. Used by a dribbler
    # following his own touch: his speed is set by how far he knocked it.
    pace: float | None = None
    # Under inertia: attack the target flat out and run THROUGH it rather than
    # braking to stop on it. For collecting a moving ball -- a receiver who
    # brakes to stand on his meeting point arrives slowly, so the earliest
    # point he can reach stays far down the ball's path and he ends up waiting
    # for a ball a sprinting defender takes off him. Ignored without inertia.
    through: bool = False
    # Under inertia: a keeper DIVING at a shot. A dive is not a run -- he
    # launches sideways at once rather than accelerating -- and that is what
    # his save model (GK_REACTION_TICKS + GK_DIVE_SPEED) already assumes.
    # Running to it under inertia: saves 8 -> 1, goals/on target 0.28 -> 0.81,
    # keeper 4.5 m from the ball at 0.29 x top speed. Ignored without inertia.
    dive: bool = False
    # Under inertia: closing down the CARRIER, whose velocity this is. He runs
    # at him and brakes only as late as he physically can (PRESS_ARRIVE_TAUS),
    # matching the carrier's own motion. The ordinary arrive eased off over
    # the last metres and reached him at walking pace: near our goal the
    # carrier was more than 3 m free on 45-49% of ticks (24% without inertia).
    press_velocity: np.ndarray | None = None


# -- Entry point --------------------------------------------------------------

def off_ball_decision(
    player: Player,
    state: GameState,
    committed: list[tuple[int, np.ndarray]] | None = None,
) -> OffBallAction:
    committed = committed or []

    team = state.get_team(player.team_id)
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X

    # Hard override: the keeper solves a constrained 1-D problem on its line,
    # not a value argmax over the pitch.
    if player.position == "GK":
        return _gk_decision(player, state, own_goal_x, team)

    # Hard override: while a keeper holds the ball in his hands, the referee
    # keeps everyone else out of his area. Teleporting them out on the catch
    # was not enough on its own -- the off-ball model pulled them straight back
    # toward the ball, and 3 of 4 were inside again within 5 ticks (208 of 270
    # hold-ticks had an opponent in the box). They have to be kept out for as
    # long as he holds it, not just moved once.
    if state.gk_hold_ticks > 0:
        # No role check needed: gk_hold_ticks is only ever set when a keeper
        # gathers the ball, so possession alone identifies him. Role stays out
        # of off-ball control flow (see test_no_role_branching_in_off_ball).
        holder = state.get_ball_owner()
        if holder is not None and holder.team_id != player.team_id:
            keeper_goal_x = (
                LEFT_GOAL_X
                if float(holder.assigned_pos[0]) < PITCH_LENGTH / 2.0
                else RIGHT_GOAL_X
            )
            # Repel anyone inside, and -- just as important -- stop anyone
            # outside from walking back in. Players 1 m beyond the line were
            # targeting the keeper himself, so clearing the box only bought a
            # few ticks before they returned.
            retreat = _retreat_from_box(player, keeper_goal_x)
            if retreat is not None:
                return OffBallAction(retreat, sprint=False)
            return OffBallAction(
                _hold_outside_box(player, state, team, keeper_goal_x), sprint=False
            )

    # Following his own touch. The touch length IS the dribbling pace: he runs
    # to where the ball will come to rest, arriving as it does -- a walk after a
    # short touch, a sprint after a long one. Chasing every touch flat out made
    # close control a sprint-stop-sprint at 7 m/s whatever the touch.
    if (
        DRIBBLE_TOUCH_MODEL
        and state.ball.state == "in_dribble"
        and state.ball.last_dribbler_id == player.player_id
        and state.ball.owner_id is None
    ):
        meet = state.ball.dribble_meet
        left = float(state.ball.dribble_meet_tick - state.tick)
        rolling = float(np.linalg.norm(state.ball.velocity)) > 1e-6
        # Follow the plan while it still holds: run to the meeting point at the
        # pace that arrives with the ball. But the point is a plan, not a
        # destination -- once he is there, the moment has passed, or the ball
        # has stopped short, he goes to the BALL. Holding to the point left him
        # standing on it while the ball rolled on past (or stopped before it):
        # measured, 11% of the time he was following his own touch he stood
        # still with the ball out of reach.
        if meet is not None and left > 0.0 and rolling and distance(player.pos, meet) > 0.5:
            pace = min(dribble_top_speed(player), distance(player.pos, meet) / left)
            return OffBallAction(_clamp_to_pitch(meet), sprint=False, pace=pace)
        if _pace.PACE and meet is not None and left > 0.0:
            # A paced touch that has stopped before the time is up: walk on to
            # it at the touch's pace rather than sprint -- the walk stays a walk.
            pace = min(dribble_top_speed(player), distance(player.pos, state.ball.pos) / left)
            return OffBallAction(_clamp_to_pitch(state.ball.pos), sprint=False, pace=pace)
        return OffBallAction(_clamp_to_pitch(state.ball.pos), sprint=False,
                             pace=dribble_top_speed(player))

    # Hard override: a shot heading at our own goal is decided geometry; one
    # tick of hesitation is a conceded goal.
    if state.ball.state == "in_shot" and _shot_threatens(state.ball, own_goal_x):
        action = _intercept_action(player, state.ball)
        if action is not None:
            return action

    # A pass in flight FOR HIM with an opponent getting too close to it: first,
    # before any plan for the point it was played to.
    coming = _comes_to_ball(player, state, team)
    if coming is not None:
        return coming

    # A pass in flight FOR HIM, into space or to where his run takes him:
    # under grip he times his run to the point it was played to (_timed_run),
    # whatever kind of pass it is. Into space he used to run to the spot and
    # then chase the ball wherever it was, arriving late; near a line the ball
    # had meanwhile run out -- most passes lost out of play were these.
    if (
        kinematics.GRIP
        and state.ball.owner_id is None
        and state.ball.state == "in_pass"
        and state.ball.intended_receiver_id == player.player_id
    ):
        timed = _timed_run(player, state)
        if timed is not None:
            return timed

    # A teammate has played a ball into space for us. Treat it as a strong
    # candidate rather than an unconditional order: if the lane has since been
    # cut off, the normal model should still be able to win.
    if player.runway_signal is not None:
        spot = _clamp_to_pitch(player.runway_signal.target_pos)
        # Running to the spot is only HALF the job; the other half is meeting
        # the ball once it is on its way. Aiming at the fixed point for the
        # whole run meant that the moment he stood on it his target was where
        # he already was, so he stopped dead and let the ball roll past:
        # measured, 2454 of 5472 live-signal ticks were spent parked on the
        # spot, 1810 of them with the ball still moving and a median 3.6 m
        # away. The named runner received the pass 17% of the time while
        # opponents took it 68%.
        #
        # So once he is there, switch to the same trajectory-aware collection
        # the loose-ball path below already uses. Do it only on arrival: while
        # he is still running, the spot IS the right target -- that is the run
        # the carrier called for, and chasing the ball's current position
        # instead would abandon the space he was sent into.
        # LATCHED, not a bare threshold. The two branches aim in different
        # directions -- the spot is ahead of the runner, the meeting point is
        # wherever the ball is -- so a player hovering on the boundary flipped
        # between them and shivered on the spot: measured, the branch changed on
        # 10.2% of signalled ticks and the chosen target moved a mean of 26 m
        # per tick while he covered 7 m in 31 ticks. Once he switches to
        # tracking the ball he stays on it until the signal ends.
        if player.meet_target is None and (
            distance(player.pos, spot) <= RUNWAY_MEET_RADIUS
            and float(np.linalg.norm(state.ball.velocity)) > 1e-9
        ):
            player.meet_target = player.pos.copy()   # latch: tracking from now on
        if player.meet_target is not None:
            attacking_goal_x = RIGHT_GOAL_X if team.direction == 1 else LEFT_GOAL_X
            meet, _ = best_meeting_point(
                state.ball.pos, state.ball.velocity, _ball_friction(state.ball),
                player, attacking_goal_x, state.opponents_of(player.team_id),
                height=state.ball.height, vz=state.ball.vz,
            )
            meet = _clamp_to_pitch(meet)
            if float(np.linalg.norm(state.ball.velocity)) <= 1e-9:
                # Ball has stopped: go and get it rather than standing off.
                return OffBallAction(_clamp_to_pitch(state.ball.pos), sprint=True)
            # best_meeting_point re-solves from scratch every tick, and against
            # a properly decelerating ball its answer swings hard -- measured,
            # the target moved a MEAN of 28 m per tick while the runner covered
            # 5.7 m in 24 ticks. He was being sent somewhere new before he could
            # take a step, so he shivered on the spot instead of running.
            #
            # Commit to the first solution and only revise it when the ball has
            # genuinely gone somewhere else. The ball's own path is smooth; it
            # is the re-solving that is not.
            if player.meet_target is not None and distance(
                meet, player.meet_target
            ) < MEET_TARGET_HYSTERESIS:
                meet = player.meet_target
            player.meet_target = meet.copy()
            return OffBallAction(meet, sprint=True, through=True)
        player.meet_target = None
        return OffBallAction(spot, sprint=True)

    # The pass in flight is FOR HIM: he goes and gets it, whatever else is true.
    # Without this rule the receiver of our own pass fell through to the
    # out-of-possession logic and its zonal gate, which lets a man leave his
    # zone for a loose ball only if he beats every opponent to it -- right for
    # a defender, wrong for the man the ball was played to. Measured: on lost
    # passes he was held in his zone on 50% of in-flight ticks (5% on kept
    # ones) and his speed toward the ball was +0.01 x top; the ball rolled to
    # a defender who started 9-15 m from the lane. Support routing pulled him
    # away on another 14-23%.
    ball = state.ball
    if (
        ball.state == "in_pass"
        and ball.owner_id is None
        and ball.intended_receiver_id == player.player_id
        and ball.last_toucher_team_id == team.team_id
    ):
        return _collect_own_pass(player, state, team)

    # A player who has just released a safe pass should move on to support the
    # move, not stand over the ball he played. 18% of passes were being
    # collected by the passer himself -- a wasted touch that stalls the attack.
    if _just_passed_safely(player, state):
        return _in_possession(player, state, team, committed)

    owner = state.get_ball_owner()
    if owner is not None and owner.team_id == player.team_id:
        return _in_possession(player, state, team, committed)

    # A ball of ours being driven upfield is still ours to support. Narrow on
    # purpose: routing EVERY in-flight ball here made teammates position for a
    # future pass instead of contesting the current one, and completion fell
    # from 76% to 59%. Only a genuine break qualifies.
    if (
        owner is None
        and _breaking_forward(state.ball, team) > 0.0
        and _our_ball_in_flight(player, state)
    ):
        return _in_possession(player, state, team, committed)

    # Called forward and the move is still alive. Possession here means the
    # MOVE is ours, not that a teammate is standing on the ball: the ball is in
    # flight for most of a spell, so keying on `owner` alone sent 85% of called
    # players down the out-of-possession path, where the call is never read and
    # the run never happened. expire_attack_calls has already cleared the call
    # if an opponent won the ball, so reaching here means it is still on.
    #
    # Gated, but only on the question that actually matters to a runner: am I
    # the one who should be collecting this? Without any gate a called player
    # ran upfield past a ball he was the closest man to (8.6% of loose-ball
    # ticks) and completion fell 9 points. But gating on _our_ball_in_flight --
    # which demands a teammate be *clearly* favourite over every opponent --
    # was far too strict: it is false through most of a contested flight, so it
    # blocked the run exactly when breaking forward matters, and gave back
    # nearly all the attacking gain (box presence 4.7% -> 0.9%).
    #
    # Whoever is best placed to collect goes and gets it; everyone else called
    # is free to run.
    #
    # But "a teammate is nearer" is not the same as "the ball is ours".
    # _is_our_best_collector iterates teammates only, and this gate was reading
    # its answer as a guarantee: a runner 2.5 m from the ball sprinted upfield
    # because a teammate was nominally nearer, while an opponent got there
    # first (2.2 ticks against the teammate's 6.0). He was the one man who
    # could have contested it -- 26.9% of gate-freeings had an opponent
    # arriving first. _can_win_the_ball asks that personally, so the player who
    # can actually win it stays and everyone else keeps going.
    if (
        player.attack_call is not None
        and owner is None
        and not _is_our_best_collector(player, state)
        and not _can_win_the_ball(player, state)
    ):
        return _in_possession(player, state, team, committed)

    # The same rule for EVERYONE while the ball is ours and travelling. The
    # ball is held only 8% of a possession spell -- every dribble touch
    # releases it -- so keying possession mode on `owner` sent our own players
    # down the DEFENDING path for 54% of our possession: chasing the ball or
    # holding defensive shape instead of supporting, which is why the team
    # never contracted around the ball and a pressed carrier was left with one
    # teammate within 10 m against 2.6 opponents.
    #
    # Kept safe by the two gates learned above: whoever is best placed to
    # collect, or can personally win the ball, still goes for it -- routing
    # every in-flight ball to support without that cut completion 76% -> 59%.
    #
    # And never the INTENDED RECEIVER of the pass in flight. The teammate
    # nearest the ball just after it is struck is often not the one it is
    # for, so without this the receiver was routed away to a support position
    # while the ball travelled to him -- passer and receiver lost their
    # synchrony and completion fell 84% -> 77%.
    if (
        owner is None
        and state.ball.last_toucher_team_id == team.team_id
        and player.player_id != state.ball.intended_receiver_id
        and not _is_our_best_collector(player, state)
        and not _can_win_the_ball(player, state)
    ):
        return _in_possession(player, state, team, committed)

    return _out_of_possession(player, state, team, owner, committed)


# -- Out of possession: the speed/safety trade-off ----------------------------

def _out_of_possession(
    player: Player,
    state: GameState,
    team: Team,
    owner: Player | None,
    committed: list[tuple[int, np.ndarray]],
) -> OffBallAction:
    ball = state.ball
    t_arrive = _ticks_to_reach(player, ball.pos)
    pressing = defensive_bias(player, team)
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    # How dangerous is it where the ball is? A gradient, not a switch: far out
    # a player keeps his zone, near goal he must close down. It scales the
    # SAME pressing that drives everything below -- claim margin (how many
    # contest), leash width and the pull toward the ball -- so one number
    # expresses "urgency rises as the ball approaches our goal" instead of four
    # separately tuned knobs.
    danger = _danger(ball.pos, own_goal_x)
    pressing *= 1.0 + DANGER_PRESS_GAIN * danger

    # A player DRIBBLING the ball is the carrier too. Every dribble touch
    # releases ownership, so keyed on `owner` alone a player running at goal was
    # 'not the owner' for most of his run -- measured, a carrier held the ball
    # inside 25 m of goal on only 62 ticks in 5000 -- and the danger-zone
    # closing-down below switched on only in the instant before the shot,
    # while defenders treated his approach as a loose ball to race for.
    if owner is None:
        owner = _dribbler(state, team)

    if owner is None:
        # Loose ball. Its own velocity matters: chasing where it currently sits
        # misses by however far it rolls while you run. Aim at the earliest
        # point on its trajectory we can actually reach.
        friction = _ball_friction(ball)
        # Normally take the ball at the earliest reachable point. Letting it run
        # on is a luxury allowed only where the chosen spot is comfortably ours
        # AND clearly worth more -- see best_meeting_point.
        attacking_goal_x = RIGHT_GOAL_X if team.direction == 1 else LEFT_GOAL_X
        target, t_arrive = best_meeting_point(
            ball.pos, ball.velocity, friction, player, attacking_goal_x,
            state.opponents_of(player.team_id), height=ball.height, vz=ball.vz,
        )
        if t_arrive <= 0.0:
            t_arrive = _ticks_to_reach(player, target)
    else:
        heading = normalize(owner.velocity)
        carrier_speed = float(np.linalg.norm(owner.velocity))
        confidence = prediction_confidence(
            carrier_speed, carrier_straightness(owner), t_arrive
        )
        _, cover = marginal_contribution(player, ball.pos, t_arrive, committed, state)
        outlets = opponent_outlets(state, owner)
        lam = compute_lambda(confidence, cover, outlets, 1.0 + DANGER_PRESS_GAIN * danger)
        target = aim_point(ball.pos, heading, carrier_speed, t_arrive, lam)
        # Press aggression: on the way to the ball, onto the lane of the pass
        # he is most likely to play (_press_lane_spot, PRESS_LANE_MIN_M).
        share = press_lane_share(press_aggression_bias(player, team))
        # The man in front of a carrier running at goal stands in his way: the
        # lane would take him off the run and open it.
        if share > 0.0 and not (_running_at_goal(owner, own_goal_x)
                                and _goal_side_of(player, owner, own_goal_x)):
            spot = _press_lane_spot(player, owner, state, own_goal_x)
            if spot is not None:
                target = target + (spot - target) * share
        # ...and near goal, close down the SHOT, not the ball. aim_point is
        # built to cut off the carrier's run, which beside the box parks the
        # defender alongside or behind him: measured at the moment of the
        # shot, the nearest defender was 1.0-1.2 m away yet 75% of shots from
        # 12-25 m had nobody in their path, and 100% inside 12 m. Blend toward
        # a point just goal-side of the carrier, on his line to our goal, as
        # danger rises -- that is what denies a comfortable shot.
        if danger > 0.0:
            to_goal = normalize(np.array([own_goal_x, GOAL_CENTER_Y]) - owner.pos)
            goal_side = owner.pos + to_goal * GOAL_SIDE_STANDOFF
            target = target + (goal_side - target) * danger

    target = _clamp_to_pitch(target)


    # Engagement is a question of who arrives FIRST, not of who could arrive at
    # all. Absolute reach probability is ~0.99 for everyone within a few metres,
    # so prod(1 - r_j) collapsed to zero with two teammates anywhere nearby and
    # every player concluded they were redundant -- including one standing 3 m
    # from a free ball.
    if not ZONAL_MARKING:
        if _should_engage(
            player, target, t_arrive, committed, state, pressing,
            carrier=owner, own_goal_x=own_goal_x, danger=danger,
        ):
            return OffBallAction(
                _spread_from_teammates(target, player, state, committed), sprint=True
            )
        return _hold_shape(player, state, team, pressing, committed)

    engaged = _engaged_this_tick(state, team, committed)
    if COVER_PRESS and owner is not None:
        # A carrier: the team decides together who steps out (_press_set);
        # everyone else marks, with the pressers' men shared out among them.
        pressers = _press_set(state, team, owner, own_goal_x)
        if player.player_id in pressers:
            return OffBallAction(
                _spread_from_teammates(target, player, state, committed), sprint=True,
                press_velocity=owner.velocity.copy(),
            )
        mark = _mark_action(player, state, team, owner, own_goal_x, pressers)
        if mark is not None:
            return mark
        return _hold_shape(player, state, team, defensive_bias(player, team), committed)
    if _should_engage(
        player, target, t_arrive, committed, state, pressing,
        carrier=owner, own_goal_x=own_goal_x, danger=danger,
    ) and _may_leave_zone(
        player, state, team, owner, own_goal_x, danger, target, t_arrive, engaged,
    ):
        first = not engaged
        engaged.append(player.player_id)
        # A loose ball is attacked through its meeting point; a carrier is
        # closed down hard, braking only as late as he can, on his motion.
        # The FIRST man on a carrier goes straight to him: spreading him off
        # teammates' spots kept him ~2 m away and the carrier free to shoot
        # (more than 3 m free on 49% of ticks 12-18 m out). Later pressers
        # still spread, so they arrive from other angles.
        aim = target if (first and owner is not None) else \
            _spread_from_teammates(target, player, state, committed)
        return OffBallAction(
            aim, sprint=True,
            through=owner is None,
            press_velocity=owner.velocity.copy() if owner is not None else None,
        )

    # Not going at the ball: defend the zone. The station drift uses the
    # player's OWN pressing, not the danger-raised figure -- urgency near goal
    # is spent on the presser and on tighter marking, not on the whole team
    # collapsing toward the ball and leaving its men.
    base_pressing = defensive_bias(player, team)
    mark = _mark_action(player, state, team, owner, own_goal_x, set(engaged))
    if mark is not None:
        return mark
    # Holding his ground, he keeps off the others holding theirs, not off the
    # men going at the ball: stepping away from them opens the way behind them.
    if FRONT_MAN:
        committed = [(pid, spot) for pid, spot in committed if pid not in engaged]
    return _hold_shape(player, state, team, base_pressing, committed)


# Box roles per attacking team, worked out once per tick: the same for every
# teammate, and the role holders must be chosen jointly (one per role).
_BOX_ROLES: dict = {}


def _box_roles(state: GameState, team: Team) -> tuple[float, dict[int, np.ndarray]]:
    """(attack level, {player id: box spot}) for a team with the ball.

    The level is a smoothstep of how far up the pitch the ball is (BOX_ATTACK_*).
    Each role in BOX_SLOTS -- penalty spot, near post, far post, cut-back,
    mirrored to the ball's side -- goes to the nearest eligible attacker: not
    the keeper, not a defender, not the carrier. Nearest-first per role, in
    role order, so the spot and the near post are filled before the cut-back.
    """
    ball = state.ball
    key = (id(state), state.tick, float(ball.pos[0]), float(ball.pos[1]))
    cached = _BOX_ROLES.get(team.team_id)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    up = float(ball.pos[0]) if team.direction == 1 else PITCH_LENGTH - float(ball.pos[0])
    t = min(1.0, max(0.0, (up - BOX_ATTACK_FROM) / (BOX_ATTACK_FULL - BOX_ATTACK_FROM)))
    level = t * t * (3.0 - 2.0 * t)
    roles: dict[int, np.ndarray] = {}
    if level > 0.0:
        side = 1.0 if float(ball.pos[1]) >= GOAL_CENTER_Y else -1.0
        carrier = state.get_ball_owner()
        eligible = [
            p for p in team.players
            if p.position not in ("GK", "LD", "CD", "RD")
            and (carrier is None or p.player_id != carrier.player_id)
        ]
        for _, depth, across in BOX_SLOTS:
            if not eligible:
                break
            x = depth if team.direction == 1 else PITCH_LENGTH - depth
            spot = np.array([x, GOAL_CENTER_Y + side * across])
            holder = min(eligible, key=lambda p: distance(p.pos, spot))
            roles[holder.player_id] = spot
            eligible.remove(holder)
    _BOX_ROLES[team.team_id] = (key, level, roles)
    return level, roles


def _in_their_box(point: np.ndarray, team: Team) -> bool:
    """Is `point` inside the penalty area this team is attacking?"""
    up = float(point[0]) if team.direction == 1 else PITCH_LENGTH - float(point[0])
    return up >= PITCH_LENGTH - PENALTY_AREA_DEPTH and (
        PENALTY_AREA_TOP <= float(point[1]) <= PENALTY_AREA_BOTTOM
    )


# The danger level each defending team faces, worked out once per tick per
# team: the same for every defender, and it needs a lane per attacker.
_DANGER_LEVEL: dict = {}


def _danger_level(state: GameState, team: Team, own_goal_x: float) -> float:
    """How much attacking threat this team faces right now. >= 0.

    threat(ball) plus, for every attacker, threat where he stands x how
    reachable he is from the ball (the lane, floored: a man the ball cannot
    reach this instant can still move). value.threat is continuous and steep
    toward our goal, so this rises smoothly as the ball and the attackers come
    closer -- the input to the goal funnel in _zone_centre.
    """
    ball = state.ball
    key = (id(state), state.tick, float(ball.pos[0]), float(ball.pos[1]))
    cached = _DANGER_LEVEL.get(team.team_id)
    if cached is not None and cached[0] == key:
        return cached[1]
    level = threat(ball.pos, own_goal_x)
    attackers = [o for o in state.opponents_of(team.team_id) if o.position != "GK"]
    if attackers:
        pts = np.array([o.pos for o in attackers])
        gaps = np.hypot(pts[:, 0] - ball.pos[0], pts[:, 1] - ball.pos[1])
        paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps])
        lanes = route_survival_many(ball.pos, pts, paces, BALL_FRICTION_COEFF, team.players)
        for o, lane in zip(attackers, lanes):
            reach = FUNNEL_REACH_FLOOR + (1.0 - FUNNEL_REACH_FLOOR) * float(lane)
            level += threat(o.pos, own_goal_x) * reach
    _DANGER_LEVEL[team.team_id] = (key, level)
    return level


def _collect_own_pass(player: Player, state: GameState, team: Team) -> OffBallAction:
    """The intended receiver meets his pass at the best point on its path.

    The same trajectory-aware meeting point the loose-ball chase uses, so the
    choice of WHERE is unchanged -- only the gates that could send him
    elsewhere are skipped. A ball that has stopped is simply collected.
    """
    ball = state.ball
    if float(np.linalg.norm(ball.velocity)) <= 1e-9:
        return OffBallAction(_clamp_to_pitch(ball.pos), sprint=True, through=True)
    if kinematics.GRIP:
        timed = _timed_run(player, state)
        if timed is not None:
            return timed
    attacking_goal_x = RIGHT_GOAL_X if team.direction == 1 else LEFT_GOAL_X
    meet, _ = best_meeting_point(
        ball.pos, ball.velocity, _ball_friction(ball), player, attacking_goal_x,
        state.opponents_of(player.team_id), height=ball.height, vz=ball.vz,
    )
    return OffBallAction(_clamp_to_pitch(meet), sprint=True, through=True)


def _comes_to_ball(player: Player, state: GameState, team: Team) -> OffBallAction | None:
    """The pass is for him and an opponent is getting too close to it, so he
    stops waiting for it and goes to meet it: flat out to the earliest point
    of its path he can reach. Common sense, and it is what the passer planned
    on: _reception_probability races him to that same earliest point.

    Everywhere else he times his run or stands on the spot, and the only
    question those ask is whether the POINT is contested: a slow ball cut out
    on its way there was nobody's business. A ball past its point is watched
    all the way to where it stops, or he lets it run on until a man arrives.
    Once he goes he stays gone for this pass, whatever the opponent does next.
    None if the pass is not his, or it is not in danger.
    """
    ball = state.ball
    if not (
        COME_TO_BALL
        and ball.state == "in_pass"
        and ball.owner_id is None
        and ball.intended_receiver_id == player.player_id
        and ball.last_toucher_team_id == team.team_id
    ):
        return None
    friction = _ball_friction(ball)
    if player.coming_to_strike != ball.struck_tick:
        arrival = _pass_arrival(ball, friction)
        since = state.tick - ball.struck_tick
        if not closing_on_path(
            ball.pos, ball.velocity, friction, state.opponents_of(team.team_id),
            max(1, COME_TO_BALL_FROM_TICKS - since),
            arrival[1] if arrival is not None else COME_TO_BALL_HORIZON,
            height=ball.height, vz=ball.vz,
        ):
            return None
        player.coming_to_strike = ball.struck_tick
    point, _ = ball_intercept_point(
        ball.pos, ball.velocity, friction, player, height=ball.height, vz=ball.vz,
        reach=control_radius(player),
    )
    return OffBallAction(_clamp_to_pitch(point), sprint=True, through=True)


def _pass_arrival(ball: Ball, friction: float) -> tuple[np.ndarray, float] | None:
    """The pass point, moved onto where the ball is ACTUALLY going (the strike
    is never perfect), and in how many ticks the ball gets there. None if it
    has no point, has stopped or has gone past it."""
    from ..ball_physics import carry_distance

    if ball.pass_target is None:
        return None
    speed = float(np.linalg.norm(ball.velocity))
    if speed < 1e-9:
        return None
    direction = ball.velocity / speed
    along = float(np.dot(ball.pass_target - ball.pos, direction))
    if along <= 0.0:
        return None                          # it has gone past the point
    if airborne(ball.height, ball.vz):
        t_ball = along / speed               # no rolling resistance in the air
    else:
        along = min(along, carry_distance(speed, friction) * 0.98)
        t_ball = travel_ticks(speed, friction, along)
    if not math.isfinite(t_ball):
        return None
    return _clamp_to_pitch(ball.pos + direction * along), t_ball


def _timed_run(player: Player, state: GameState) -> OffBallAction | None:
    """Go to the pass point at the pace that arrives just before the ball.

    The passer chose WHERE; how fast to get there is the receiver's call.
    Flat out, he arrives committed -- under grip a sprinter cannot adjust to
    a ball that is a metre off -- and early for nothing. So when no opponent
    can contest the point he runs at the slowest pace that still has him there
    TIMED_RUN_EARLY_TICKS before the ball. None (sprint to meet it, as
    before) when the point is contested, he cannot make it with time to
    spare, or the ball is in the air.
    """
    arrival = _pass_arrival(state.ball, _ball_friction(state.ball))
    if arrival is None:
        return None
    point, t_ball = arrival
    reach = control_radius(player)
    if distance(player.pos, point) <= reach:
        # There: settle and wait for it.
        return OffBallAction(point, sprint=False)
    budget = t_ball - TIMED_RUN_EARLY_TICKS
    top = player.speed
    flat_out = float(reach_ticks(player.pos, player.velocity, top, point, reach))
    if flat_out > t_ball:
        return None                          # he cannot get there: meet it earlier
    if flat_out > budget or not _comfortably_ours(
        point, t_ball, player, state.opponents_of(player.team_id)
    ):
        # Contested, or only just in time: flat out to the point.
        return OffBallAction(point, sprint=True, through=True)
    lo, hi = TIMED_RUN_MIN_SHARE * top, top
    if float(reach_ticks(player.pos, player.velocity, lo, point, reach)) <= budget:
        pace = lo
    else:
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            if float(reach_ticks(player.pos, player.velocity, mid, point, reach)) <= budget:
                hi = mid
            else:
                lo = mid
        pace = hi
    return OffBallAction(point, sprint=False, pace=pace, through=True)


def _dribbler(state: GameState, team: Team) -> Player | None:
    """The opponent currently dribbling the ball, if any.

    A dribbled ball has no owner between touches, but it is plainly his: he is
    the man to defend against, not a loose ball to race for.
    """
    ball = state.ball
    if ball.state != "in_dribble" or ball.last_dribbler_id is None:
        return None
    try:
        p = state.get_player(ball.last_dribbler_id)
    except KeyError:
        return None
    return p if p.team_id != team.team_id else None


def _danger(pos: np.ndarray, own_goal_x: float) -> float:
    """0 far from our goal, rising to 1 at the edge of the six-yard box.

    Distance to goal rather than threat(): the user's rule is about a ZONE --
    keep shape out there, close down in here -- and distance says that plainly.
    Linear between DANGER_FAR and DANGER_NEAR so there is no line to hover on.
    """
    d = float(np.hypot(float(pos[0]) - own_goal_x, float(pos[1]) - GOAL_CENTER_Y))
    return float(min(1.0, max(0.0, (DANGER_FAR - d) / (DANGER_FAR - DANGER_NEAR))))


def _should_engage(
    player: Player,
    target: np.ndarray,
    t_arrive: float,
    committed: list[tuple[int, np.ndarray]],
    state: GameState,
    pressing: float = 1.0,
    carrier: Player | None = None,
    own_goal_x: float | None = None,
    danger: float = 0.0,
) -> bool:
    """Go if nobody already committed beats us to the contest point.

    CLAIM_MARGIN_TICKS is the hysteresis that stops a whole team converging:
    a teammate who arrives clearly sooner takes it, and everyone else keeps
    their shape. Pressing WIDENS that margin: the deferral test is
    `mate_time + margin < my_time`, so a bigger margin makes it harder for a
    teammate to crowd you out and you contest balls a patient player would
    leave. That is what makes the setting feel like pressing rather than just a
    different run angle.
    """
    if t_arrive == float("inf"):
        return False
    claim_margin = CLAIM_MARGIN_TICKS * max(pressing, 1e-6)
    for pid, mate_target in committed:
        try:
            mate = state.get_player(pid)
        except KeyError:
            continue
        if mate.speed <= 1e-9:
            continue
        # Only teammates actually contesting this same point crowd us out.
        if distance(mate_target, target) > CONTEST_SAME_POINT_RADIUS:
            continue
        # Near our goal, a teammate BEHIND the carrier is not cover. He may
        # reach the ball first, but he is chasing: he cannot get between the
        # carrier and the goal, so he cannot stop the shot. Standing down for
        # him left the player who COULD block it holding his zone while the
        # carrier ran on into clear space and shot.
        if (
            carrier is not None
            and own_goal_x is not None
            and (danger > 0.0 or _running_at_goal(carrier, own_goal_x))
            and not _goal_side_of(mate, carrier, own_goal_x)
        ):
            continue
        if _eta(mate, target) + claim_margin < t_arrive:
            return False
    return True


# A/B switch for the man in front of a running carrier (on); 0 restores the old
# rule, where nobody stepped up to him further out than DANGER_FAR.
FRONT_MAN = os.environ.get("FB_FRONT_MAN", "1") != "0"


def _running_at_goal(carrier: Player, own_goal_x: float) -> bool:
    """Is the carrier moving toward our goal at FRONT_MAN_RUN or more?"""
    to_goal = normalize(np.array([own_goal_x, GOAL_CENTER_Y]) - carrier.pos)
    return FRONT_MAN and float(carrier.velocity @ to_goal) * TICK_RATE >= FRONT_MAN_RUN


def _goal_side_of(defender: Player, carrier: Player, own_goal_x: float) -> bool:
    """Is `defender` between `carrier` and our goal?

    Measured along the carrier's line to goal, so a defender level with him
    out wide is NOT goal-side -- he has no body in the shot.
    """
    to_goal = np.array([own_goal_x, GOAL_CENTER_Y]) - carrier.pos
    length = float(np.linalg.norm(to_goal))
    if length < 1e-9:
        return True
    ahead = float(np.dot(defender.pos - carrier.pos, to_goal)) / length
    return ahead > GOAL_SIDE_MIN_AHEAD


def _ball_friction(ball: Ball) -> float:
    from ..constants import (
        BALL_FRICTION_COEFF as _PASS_F,
        DRIBBLE_FRICTION_COEFF as _DRIB_F,
        SHOT_FRICTION_COEFF as _SHOT_F,
    )

    return {"in_shot": _SHOT_F, "in_dribble": _DRIB_F}.get(ball.state, _PASS_F)


# -- In possession: find space worth receiving in -----------------------------

# Tempo off the ball (TEMPO_RUN_M, and the sprint gap). A switch only to
# measure it.
TEMPO_RUNS = True


def _in_possession(
    player: Player,
    state: GameState,
    team: Team,
    committed: list[tuple[int, np.ndarray]] | None = None,
) -> OffBallAction:
    attacking_goal_x = RIGHT_GOAL_X if team.direction == 1 else LEFT_GOAL_X
    opponents = state.opponents_of(player.team_id)
    ball = state.ball

    # One number for the whole team this tick -- the carrier's outlet quality is
    # the same regardless of which teammate is asking, so computing it per
    # player (let alone per candidate point) would be pure waste.
    carrier = state.get_ball_owner()
    support = support_need(best_pass_safety(state, carrier)) if carrier else 1.0
    roaming = roaming_bias(player, team)
    # A player who has just laid the ball off should support the move rather
    # than trudging back to station. one_two decides how strongly: it pulls his
    # anchor toward the ball, so the candidate points near the play win instead
    # of the one 19 m behind it.
    just_passed = _just_passed_safely(player, state)

    anchor = _shape_anchor(player, ball, team, state)
    # Called forward by the carrier: move the ANCHOR upfield rather than adding
    # a bonus to points around the station. A bonus could never work -- every
    # candidate is scored through the Gaussian leash centred on the anchor, so
    # the leash decayed faster than the bonus grew and the net score peaked at
    # 6 m of gain; a genuine 20 m supporting run scored WORSE than standing
    # still. Moving the anchor moves the whole scoring surface with it, so the
    # run becomes the thing the argmax actually wants.
    if player.attack_call is not None and player.attack_call.spot is not None:
        # Joining the attack (signals.JOIN): his run goes to his spot, which
        # keeps its lead on the ball as the attack moves on.
        from ..signals import join_spot_now
        anchor = _clamp_to_pitch(join_spot_now(player.attack_call, state))
    elif player.attack_call is not None:
        anchor = _clamp_to_pitch(
            anchor + np.array([team.direction * ATTACK_RUN_RANGE, 0.0])
        )
    # Tempo: midfielders and forwards run ahead of the ball (high) or come
    # short (low) -- the anchor again, for the same reason as the attack call.
    tempo = tempo_bias(player, team) if TEMPO_RUNS else 1.0
    if tempo != 1.0 and player.position[1] in "MF":
        from ..constants import TEMPO_RUN_M
        anchor = _clamp_to_pitch(
            anchor + np.array([team.direction * TEMPO_RUN_M * math.log2(tempo), 0.0])
        )
    # Come in to SUPPORT the carrier. Moving the anchor, not adding a bonus:
    # support used to be a score multiplier, and a multiplier cannot beat the
    # Gaussian leash around a player's station -- the same lesson the attack
    # call learned. Measured: a carrier under pressure had 2.6 opponents
    # within 10 m and 1.0 teammate, a quarter of the time none at all, and the
    # team's spacing barely changed whether he was pressed or not (26.5 m vs
    # 28.0 m). The press swarmed the ball and support stayed on the outer ring.
    #
    # So pull the station toward a ring at COMPACT_SUPPORT_DISTANCE around the
    # ball. Players already near come in fully; the far side barely moves, so
    # width survives. Always on, and stronger when the carrier is pressed.
    if carrier is None or carrier.player_id != player.player_id:
        anchor = _support_anchor(anchor, ball.pos, _ball_pressure(state, team))
    # Attack the box: a player holding a box role draws his anchor toward it
    # as the ball gets up the pitch (see BOX_* in constants).
    box_level, box_roles = _box_roles(state, team)
    box_spot = box_roles.get(player.player_id)
    if box_spot is not None:
        anchor = _clamp_to_pitch(anchor + (box_spot - anchor) * box_level)
    carrier_pressure = 0.0
    if carrier is None or carrier.player_id != player.player_id:
        carrier_pressure = _ball_pressure(state, team)
    one_two = one_two_bias(player, team) if just_passed else 0.0
    compactness = compactness_bias(player, team)
    best_target = anchor
    best_score = -1.0

    # A player looking for a pass checks himself against the last defender.
    # This is a CONSTRAINT, not a preference: the old 0.05 discount competed
    # against position_value and the shape anchor, so a high offensiveness
    # pushed the anchor upfield until being offside was worth it -- discipline
    # failed exactly when attacking intent was highest. A striker does not
    # weigh offside against goal proximity; he holds the line and times his run.
    #
    # Only while our own side has the ball: out of possession there is no
    # offside to be caught by, and a defender chasing back must be free to go
    # wherever the ball is.
    onside_x = None
    if _we_have_the_ball(state, team):
        onside_x = onside_limit(state, team)

    points = _candidate_points(player, anchor, roaming)
    if onside_x is not None:
        points = [_hold_onside(pt, onside_x, team.direction) for pt in points]
    # Every candidate's lane in ONE batched call -- identical numbers to calling
    # route_survival per point (verified to 4e-16), but this loop runs for most
    # of the team on most ticks, and the scalar version was 26 of every 66
    # seconds of simulation.
    pts = np.array(points)
    gaps_ball = np.hypot(pts[:, 0] - ball.pos[0], pts[:, 1] - ball.pos[1])
    if kinematics.INERTIA:
        arrive = reach_ticks(player.pos, player.velocity, max(player.speed, 1e-9), pts)
    else:
        arrive = np.hypot(pts[:, 0] - player.pos[0], pts[:, 1] - player.pos[1]) / max(player.speed, 1e-9)
    pace = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps_ball])
    lanes = route_survival_many(ball.pos, pts, pace, BALL_FRICTION_COEFF, opponents, arrive)

    if kinematics._rust is not None:
        claimed = dict(committed or [])
        mates = [claimed.get(m.player_id, m.pos) for m in state.teammates_of(player.player_id)]
        best = kinematics._rust.best_spot(
            pts, lanes, np.array([o.pos for o in opponents]).reshape(-1, 2),
            np.array(mates).reshape(-1, 2), _move_context(player),
            (float(ball.pos[0]), float(ball.pos[1])), (float(anchor[0]), float(anchor[1])),
            attacking_goal_x, float(team.direction),
            _leash_sigma(player, 1.0, roaming, compactness), _spacing_radius(player, state),
            carrier_pressure, box_level, support, float(one_two), onside_limit(state, team),
            None if player.last_off_ball_target is None else (
                float(player.last_off_ball_target[0]), float(player.last_off_ball_target[1])),
        )
        best_target = anchor if best < 0 else points[best]
        return _chosen_spot(player, state, team, best_target)

    for idx, point in enumerate(points):
        value = position_value(point, attacking_goal_x)
        # Receivability is judged against the *long* ball, and floored. Scoring
        # it with PASS_SPEED_LOW alone (32 m carry) meant any point beyond
        # current passing range scored ~0, so forwards never ran ahead of the
        # play and only 1% of possession had anyone in the box. A striker makes
        # the run first and the ball follows.
        # Judged at the moment he could actually BE there. The pass cannot be
        # played until he arrives, so the defenders get his whole run to close
        # the lane -- scoring it against their positions right now ran the race
        # from the wrong starting gun and every chosen point was open only at
        # the instant it was picked. Measured: 0.766 at choice, 0.383 by
        # arrival, while the carrier saw teammates on a mean lane of 0.135.
        receivable = max(float(lanes[idx]), RECEIVABILITY_FLOOR)
        # Space is what makes a pass possible. Running at the ball brings your
        # marker with you and clogs the very lane you want played into, so a
        # crowded point is worth much less than a free one.
        space = space_quality(point, opponents)
        # ...but not when coming short to a PRESSED carrier. Space measures
        # distance from the nearest opponent, so beside a swarm it is low and
        # it pushed supporters back to the outer ring -- measured, the best
        # point within 10 m of a pressed carrier had a 43% BETTER lane from the
        # ball than the point chosen, but 31% less "space". What makes a pass
        # possible is a clear line from the ball, which receivable already
        # scores; empty grass is not the point. So near a pressed carrier the
        # space term fades toward 1 and the lane decides.
        if carrier_pressure > 0.0:
            near_ball = 1.0 - min(1.0, distance(point, ball.pos) / COMPACT_REACH)
            space = space ** (1.0 - carrier_pressure * near_ball)
        # ...nor inside their box once we are attacking it: a packed box is
        # where the cross and the cut-back go, and scoring it as "no space"
        # kept our attackers outside it.
        if box_level > 0.0 and _in_their_box(point, team):
            space = space ** (1.0 - box_level)

        # Support vs threat, keyed on whether the carrier actually has an outlet.
        # A carrier with no safe pass needs someone to come short; one with good
        # options is better served by runs into dangerous space.
        proximity = 1.0 - min(1.0, distance(point, ball.pos) / SUPPORT_RANGE)
        support_term = 1.0 + SUPPORT_MAX_BONUS * support * proximity

        zone = classify_zone(point, attacking_goal_x)
        ctx = context_multiplier(zone, player.position, player.tactical, "move")
        score = value * receivable * ctx * _shape_factor(
            point, anchor, player, roaming=roaming, compactness=compactness,
        )
        # Two players occupying the same space act as one, wasting a body. Same-
        # role teammates share a leash and evaluate the same candidate points,
        # so without this they converge -- two centre-forwards were measured
        # 0.42 m apart.
        #
        # Kept gentler than the out-of-possession displacement: a strong penalty
        # here fights the roaming leash, because a longer leash reaches further
        # and therefore nearer to teammates, which inverted the slider.
        score *= _teammate_spacing(point, player, state, committed) ** SPACING_SOFTNESS
        if one_two:
            # Reward staying available for the return ball. Applied to the score
            # rather than the anchor: a ring point wins the argmax regardless of
            # where the anchor sits, so moving the anchor changed nothing.
            gap = distance(point, ball.pos)
            if gap < ONE_TWO_STANDOFF:
                # Do not hover over the ball just played. Supporting means being
                # an OPTION for the return, not being the receiver -- crowding it
                # pushed self-retakes from 18% to 23%.
                score *= ONE_TWO_CROWD_PENALTY
            else:
                near = 1.0 - min(1.0, gap / ONE_TWO_SUPPORT_RANGE)
                score *= 1.0 + (one_two - 1.0 + ONE_TWO_PULL_BASE) * near
        score *= space * support_term
        score *= _onside_factor(point, state, team)

        # Hysteresis: without it the argmax flips between near-equal points
        # every few ticks and the player visibly vibrates.
        if player.last_off_ball_target is not None and distance(
            point, player.last_off_ball_target
        ) < 2.0:
            score *= 1.0 + STICKY_BONUS

        if score > best_score:
            best_score, best_target = score, point

    return _chosen_spot(player, state, team, best_target)


def _chosen_spot(
    player: Player, state: GameState, team: Team, target: np.ndarray
) -> OffBallAction:
    player.last_off_ball_target = target.copy()
    if kinematics.GRIP:
        return OffBallAction(target, sprint=_support_sprint(player, state, team, target))
    return OffBallAction(target, sprint=distance(player.pos, target) > 4.0)


_MOVE_CONTEXT: dict = {}


def _move_context(player: Player) -> list[float]:
    """context_multiplier for a run into each zone, in fb_core's order."""
    key = (player.position, player.tactical)
    table = _MOVE_CONTEXT.get(key)
    if table is None:
        table = _MOVE_CONTEXT[key] = [
            context_multiplier(zone, player.position, player.tactical, "move")
            for zone in (Zone.ATTACKING_CENTRAL, Zone.ATTACKING_WING,
                         Zone.MIDFIELD_CENTRAL, Zone.MIDFIELD_WING,
                         Zone.DEFENSIVE_CENTRAL, Zone.DEFENSIVE_WING)
        ]
    return table


def _support_sprint(player: Player, state: GameState, team: Team, target: np.ndarray) -> bool:
    """Under grip, support play is a jog: a sprinter cannot turn with the
    ball when it comes, and sprinting to every spot 4 m away had outfield
    players flat out 45% of the time (real ones: about a tenth). Sprint only
    when it is far (SUPPORT_SPRINT_GAP), or a run in behind the defence, where
    arriving in time is the whole point."""
    # Tempo: a high-tempo side sprints sooner, a low-tempo one later.
    tempo = tempo_bias(player, team) if TEMPO_RUNS else 1.0
    if distance(player.pos, target) > SUPPORT_SPRINT_GAP / tempo:
        return True
    line = offside_line(state, team)   # where OUR players become offside
    beyond = (float(target[0]) - line) * team.direction
    return beyond > -SUPPORT_BEHIND_MARGIN and distance(player.pos, target) > 4.0


def _candidate_points(
    player: Player, anchor: np.ndarray, roaming: float = 1.0
) -> list[np.ndarray]:
    """Ring samples around the player, plus the shape anchor as a fallback that
    is always present so the argmax is never degenerate.

    The rings scale with roaming. Fixed radii meant a longer leash could only
    reweight the same points, never reach further -- so the slider changed the
    scores but not where anyone actually ended up.
    """
    points = [anchor.copy()]
    for radius in OFFBALL_RING_RADII:
        scaled = radius * roaming
        for i in range(OFFBALL_RING_POINTS):
            angle = 2.0 * np.pi * i / OFFBALL_RING_POINTS
            offset = np.array([np.cos(angle), np.sin(angle)]) * scaled
            points.append(_clamp_to_pitch(player.pos + offset))
    return points


# -- Shape --------------------------------------------------------------------

def press_lane_share(aggression: float) -> float:
    """How far a presser moves from his aim onto the lane to the carrier's best
    outlet: (1 - log2(aggression)) / 2, 1 at 0.5, 0.5 at 1.0, 0 at 2.0."""
    return 0.5 * (1.0 - math.log2(aggression))


_BEST_OUTLET: dict = {}


def _best_outlet(carrier: Player, state: GameState, attacking_goal_x: float) -> np.ndarray | None:
    """Where the carrier's likeliest pass goes: the teammate 6 m or more away
    with the highest lane x position value. Once per tick and carrier."""
    key = (id(state), state.tick, carrier.player_id)
    if _BEST_OUTLET.get("key") == key:
        return _BEST_OUTLET["spot"]
    mates = [m for m in state.teammates_of(carrier.player_id)
             if m.position != "GK" and distance(m.pos, carrier.pos) >= MIN_PASS_DIST]
    spot = None
    if mates:
        pts = np.array([m.pos for m in mates])
        gaps = np.hypot(pts[:, 0] - carrier.pos[0], pts[:, 1] - carrier.pos[1])
        paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps])
        lanes = route_survival_many(carrier.pos, pts, paces, BALL_FRICTION_COEFF,
                                    state.opponents_of(carrier.team_id))
        worth = lanes * np.array([position_value(p, attacking_goal_x) for p in pts])
        spot = pts[int(np.argmax(worth))].copy()
    _BEST_OUTLET.update(key=key, spot=spot)
    return spot


def _press_lane_spot(
    player: Player, carrier: Player, state: GameState, attacking_goal_x: float
) -> np.ndarray | None:
    """The nearest point to him on the lane from the carrier to his best outlet,
    at least PRESS_LANE_MIN_M out from the carrier. None if he has no outlet."""
    outlet = _best_outlet(carrier, state, attacking_goal_x)
    if outlet is None:
        return None
    lane = outlet - carrier.pos
    length = float(np.linalg.norm(lane))
    unit = lane / max(length, 1e-9)
    along = float(np.dot(player.pos - carrier.pos, unit))
    return carrier.pos + unit * min(length, max(PRESS_LANE_MIN_M, along))


def _hold_outside_box(
    player: Player, state: GameState, team: Team, goal_x: float
) -> np.ndarray:
    """Where to wait while the keeper has it: normal shape, but never a point
    inside his area."""
    from ..physics import in_penalty_area

    target = _hold_shape(player, state, team).target_pos
    if not in_penalty_area(target, goal_x):
        return target
    pushed = _retreat_from_box_point(target, goal_x)
    return pushed


def _retreat_from_box_point(pos: np.ndarray, goal_x: float) -> np.ndarray:
    """The nearest point outside the area to `pos`."""
    edge_x = (
        goal_x + PENALTY_AREA_DEPTH + BOX_CLEARANCE
        if goal_x < PITCH_LENGTH / 2.0
        else goal_x - PENALTY_AREA_DEPTH - BOX_CLEARANCE
    )
    out_x = abs(float(pos[0]) - edge_x)
    top_gap = abs(float(pos[1]) - (PENALTY_AREA_TOP - BOX_CLEARANCE))
    bottom_gap = abs(float(pos[1]) - (PENALTY_AREA_BOTTOM + BOX_CLEARANCE))
    if out_x <= min(top_gap, bottom_gap):
        return _clamp_to_pitch(np.array([edge_x, float(pos[1])]))
    if top_gap <= bottom_gap:
        return _clamp_to_pitch(np.array([float(pos[0]), PENALTY_AREA_TOP - BOX_CLEARANCE]))
    return _clamp_to_pitch(np.array([float(pos[0]), PENALTY_AREA_BOTTOM + BOX_CLEARANCE]))


def _retreat_from_box(player: Player, goal_x: float) -> np.ndarray | None:
    """Where to stand when the keeper has the ball, or None if already clear.

    Mirrors physics.clear_opponents_from_box so the walked-to point and the
    teleported-to point are the same place; only the means differ.
    """
    from ..physics import in_penalty_area

    if not in_penalty_area(player.pos, goal_x):
        return None
    return _retreat_from_box_point(player.pos, goal_x)


def _ball_pressure(state: GameState, team: Team) -> float:
    """How pressed our ball is: 0 free, 1 with an opponent on it.

    Measured at the BALL, not the carrier, because the ball is held only a
    small fraction of a possession spell -- it is travelling between touches
    most of the time, and support must not switch off in between.
    """
    nearest = min(
        (distance(o.pos, state.ball.pos) for o in state.opponents_of(team.team_id)),
        default=float("inf"),
    )
    return min(1.0, max(0.0, (COMPACT_PRESSURE_FAR - nearest) / COMPACT_PRESSURE_FAR))


def _support_anchor(
    anchor: np.ndarray, ball_pos: np.ndarray, pressure: float
) -> np.ndarray:
    """Pull an in-possession station toward the support ring around the ball."""
    offset = anchor - ball_pos
    d = float(np.linalg.norm(offset))
    if d <= COMPACT_SUPPORT_DISTANCE:
        return anchor
    # How much this player takes part: fully if his station is near the play,
    # not at all if it is on the far side.
    reach = min(1.0, max(0.0, (COMPACT_REACH - d) / (COMPACT_REACH - COMPACT_FULL)))
    pull = reach * (COMPACT_BASE_PULL + COMPACT_PRESSURE_PULL * pressure)
    ring = ball_pos + offset / d * COMPACT_SUPPORT_DISTANCE
    return _clamp_to_pitch(anchor + (ring - anchor) * min(1.0, pull))


def _shape_anchor(
    player: Player, ball: Ball, team: Team, state: GameState | None = None
) -> np.ndarray:
    """The assigned position, slid toward the ball -- and never offside.

    Replaces _clamp_x_to_zone, _blend_y and the press-radius magic numbers with
    one idea: the whole block shifts with the ball.

    The station is held onside ALWAYS, not only in possession. Depth pushes
    stations upfield, and at depth 2.0 both centre-forwards were stationed
    14.9 m behind the opposition defence: every time his team won the ball a
    striker was already offside and had seconds of retreating to do, so he was
    flagged over and over however carefully his runs were chosen. A forward
    monitors the last defender continuously, and his resting position moves
    with that line.
    """
    # A compact team attacks and defends TOGETHER: the whole block follows the
    # ball more closely, so defenders push up behind an attack and forwards
    # drop back behind the ball when defending, and the lines stay connected.
    # Clamped below 1 -- the block tracks the ball, it does not overtake it.
    # The clamp is on the slide, not on the follow: clamping the follow at
    # 0.95, which neutral already reached, made every compactness above ~0.94
    # follow the ball exactly like neutral.
    follow = COMPACT_SLIDE_NEUTRAL * compactness_bias(player, team) ** COMPACT_SLIDE_GAIN
    slide_x = (float(ball.pos[0]) - PITCH_LENGTH / 2.0) * min(0.95, SHAPE_BLOCK_SLIDE_X * follow)
    slide_y = (float(ball.pos[1]) - PITCH_WIDTH / 2.0) * min(0.95, SHAPE_BLOCK_SLIDE_Y * follow)
    anchor = _clamp_to_pitch(player.assigned_pos + np.array([slide_x, slide_y]))
    if state is not None:
        anchor = _hold_onside(anchor, onside_limit(state, team), team.direction)
    return anchor


def _shape_factor(
    point: np.ndarray,
    anchor: np.ndarray,
    player: Player,
    pressing: float = 1.0,
    roaming: float = 1.0,
    compactness: float = 1.0,
) -> float:
    """Gaussian leash. The only place role appears in off-ball movement: a
    centre-back is on a short leash, a striker roams.

    Pressing lengthens the leash while defending: a pressing player accepts
    leaving his station to hunt the ball, a patient one holds shape. Tie-breaks
    alone were not enough -- arrival-time gaps between a midfielder and a
    defender run to tens of ticks, so a few ticks of claim margin never decided
    anything.

    Roaming does the same for the attacking phase: an attack-minded player
    tolerates being far from his station to join the move, a defensive one
    returns to shape sooner.
    """
    sigma = _leash_sigma(player, pressing, roaming, compactness)
    return float(np.exp(-((distance(point, anchor) / sigma) ** 2)))


def _leash_sigma(player: Player, pressing: float, roaming: float, compactness: float) -> float:
    sigma = ROLE_LEASH.get(player.position, 15.0) * TACTICAL_LEASH.get(player.tactical, 1.0)
    sigma *= pressing * roaming
    # Compactness: a compact team strays less from its (already contracted)
    # stations, so the shape holds while play moves instead of drifting apart.
    sigma *= compactness ** -COMPACT_LEASH_GAIN
    return sigma


def _hold_shape(
    player: Player,
    state: GameState,
    team: Team,
    pressing: float = 1.0,
    committed: list[tuple[int, np.ndarray]] | None = None,
) -> OffBallAction:
    """Return toward station -- but a pressing player drifts toward the ball
    rather than sitting exactly on his anchor."""
    anchor, recovering = _zone_centre(player, state, team, pressing)
    anchor = _spread_from_teammates(anchor, player, state, committed)
    sprint = distance(player.pos, anchor) > (RECOVER_SPRINT_GAP if recovering else 10.0)
    return OffBallAction(anchor, sprint=sprint)


# A/B switch for the shooting-triangle recovery (on); 0 restores the old line
# across the pitch.
TRIANGLE_RECOVERY = os.environ.get("FB_TRIANGLE_RECOVERY", "1") != "0"


def _zone_centre(
    player: Player,
    state: GameState,
    team: Team,
    pressing: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """_zone_centre_compute, once per player per tick (see tick_cache):
    zonal marking asks for every defender's centre on behalf of each one, ~80
    computations a tick for ~20 distinct answers."""
    cache = tick_cache.get()
    if cache is None:
        return _zone_centre_compute(player, state, team, pressing)
    key = ("zone_centre", player.player_id, pressing)
    hit = cache.get(key)
    if hit is None:
        hit = cache[key] = _zone_centre_compute(player, state, team, pressing)
    return hit[0].copy(), hit[1]


def _zone_centre_compute(
    player: Player,
    state: GameState,
    team: Team,
    pressing: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """The middle of the ground this player defends, and whether he is
    recovering to it.

    His station slid with the ball, drifted toward it by pressing, and pulled
    back behind it on a recovery run. Separate from _hold_shape so zonal
    marking can work out EVERY teammate's zone, not just his own.
    """
    if SHAPE_CALL and not _we_have_the_ball(state, team):
        call = state.shape_calls.get(team.team_id)
        if call is not None:
            return _called_centre(player, state, team, call)
    anchor = _shape_anchor(player, state.ball, team, state)
    # Depth: symmetric about neutral: above 1.0 a player drifts toward the ball,
    # below it he sits deeper than his station and stays compact. (This was
    # Pressing's until decision D-16: how high the block sits is Depth's job.)
    pull = (offensiveness_bias(player, team) - 1.0) * HOLD_BALL_PULL
    pull = min(HOLD_BALL_PULL_MAX, max(-HOLD_BALL_PULL_MAX, pull))
    if abs(pull) > 1e-9:
        anchor = _clamp_to_pitch(anchor + (state.ball.pos - anchor) * pull)

    # Goal funnel: the whole block draws in toward the middle of our goal --
    # deeper and narrower together, so it can block shots -- by an amount that
    # grows smoothly with the danger the team faces (see FUNNEL_* constants).
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    level = _danger_level(state, team, own_goal_x)
    funnel = FUNNEL_MAX * (1.0 - math.exp(-((level / FUNNEL_SCALE) ** FUNNEL_POWER)))
    if funnel > 1e-9:
        goal = np.array([own_goal_x, GOAL_CENTER_Y])
        anchor = _clamp_to_pitch(goal + (anchor - goal) * (1.0 - funnel))

    # Recovery run: get back behind the ball. Measured at every shot, 42% came
    # with at most ONE defender goal-side of the ball and 82% with three or
    # fewer -- and most goals arrived 5-15 s after the ball was won, time
    # enough to recover. Nobody tracked back: a player whose station was upfield
    # held it while the attack went past him.
    #
    # So a player ahead of the ball has his station pulled back goal-side of
    # it, more strongly the nearer the ball is to our goal, and he sprints
    # there. The most advanced player stays up as the outlet -- a team that
    # sends everyone back can never counter.
    #
    # "Goal-side" is measured along the line from the ball to the middle of
    # our goal, not down the length of the pitch. A line across the pitch
    # RECOVER_DEPTH behind the ball (as this was) sat between the ball and the
    # BYLINE when the ball was wide near our goal: defenders recovered to the
    # side of the goal, behind the ball, and left the carrier the whole goal
    # to shoot at. And he is drawn into the shooting triangle -- ball to the
    # two posts, plus RECOVER_TRIANGLE_MARGIN -- where his body cuts the angle.
    recovering = False
    if not TRIANGLE_RECOVERY and not _is_outlet(player, state, team):
        # The old rule, kept behind the A/B switch: a line across the pitch.
        own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
        line_x = float(state.ball.pos[0]) - team.direction * RECOVER_DEPTH
        ahead = (float(anchor[0]) - line_x) * team.direction
        if ahead > 0.0:
            strength = RECOVER_BASE + (1.0 - RECOVER_BASE) * _danger(state.ball.pos, own_goal_x)
            new_x = float(anchor[0]) - team.direction * ahead * strength
            old_depth = abs(float(anchor[0]) - own_goal_x)
            new_depth = abs(new_x - own_goal_x)
            y = float(anchor[1])
            if old_depth > 1e-6 and new_depth < old_depth:
                y = GOAL_CENTER_Y + (y - GOAL_CENTER_Y) * (new_depth / old_depth)
            anchor = _clamp_to_pitch(np.array([new_x, y]))
            recovering = strength > 0.0
    elif not _is_outlet(player, state, team):
        own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
        ball = state.ball.pos
        to_goal = np.array([own_goal_x, GOAL_CENTER_Y]) - ball
        dist = float(np.hypot(to_goal[0], to_goal[1]))
        if dist > 1e-6:
            u = to_goal / dist
            danger = _danger(ball, own_goal_x)
            need = min(RECOVER_DEPTH, dist)
            ahead = need - float((anchor - ball) @ u)
            # Into the triangle: fully for a player recovering, and for one
            # already goal-side by how near our goal the ball is -- a man
            # standing beside the goal is goal-side but cuts no angle.
            strength = danger
            if ahead > 0.0:
                strength = RECOVER_BASE + (1.0 - RECOVER_BASE) * danger
                anchor = anchor + u * (ahead * strength)
                recovering = strength > 0.0
            depth = float((anchor - ball) @ u)
            if strength > 0.0 and depth > 0.0:
                # Beyond the goal along the ball's line -- the far side of
                # the goal -- he is measured against the goal mouth itself.
                lateral = (anchor - ball) - u * depth
                width = (GOAL_BOTTOM_Y - GOAL_TOP_Y) / 2.0 * min(depth, dist) / dist \
                    + RECOVER_TRIANGLE_MARGIN
                off = float(np.hypot(lateral[0], lateral[1]))
                if off > width:
                    anchor = anchor - lateral / off * ((off - width) * strength)
            anchor = _clamp_to_pitch(anchor)
    return _not_too_deep(anchor, state, team), recovering


def _called_centre(
    player: Player, state: GameState, team: Team, call: ShapeCall
) -> tuple[np.ndarray, bool]:
    """The middle of his ground under the back caller's call: his station, held
    onside, with Depth's drift toward the ball on top. He is recovering while
    he is upfield of it."""
    station = called_station(player, call, team, state.tick)
    anchor = _hold_onside(station, onside_limit(state, team), team.direction)
    pull = (offensiveness_bias(player, team) - 1.0) * HOLD_BALL_PULL
    pull = min(HOLD_BALL_PULL_MAX, max(-HOLD_BALL_PULL_MAX, pull))
    if abs(pull) > 1e-9:
        anchor = _clamp_to_pitch(anchor + (state.ball.pos - anchor) * pull)
    return anchor, (float(player.pos[0]) - float(anchor[0])) * team.direction > 0.0


def _not_too_deep(anchor: np.ndarray, state: GameState, team: Team) -> np.ndarray:
    """Hold a station no deeper than needed (see LINE_SHARE).

    No nearer our goal than LINE_SHARE of the ball's distance from it, so the
    gap behind the ball shrinks as the ball comes closer -- a back line on the
    edge of our box with the ball 30 m out, not on the penalty spot. Full with
    the ball within DANGER_FAR, fading out by LINE_FADE_FAR: dropping back is
    legitimate while the ball still has a long way to come.

    Measured before (back four's zone centres): 0.41-0.50 of the ball's
    distance -- 12 m with the ball 29 m out, 17 m with it 41 m out. The formation
    slides only 45% of the ball's movement and the goal funnel deepens that.
    """
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    goal = np.array([own_goal_x, GOAL_CENTER_Y])
    ball = state.ball.pos
    ball_r = float(np.hypot(*(ball - goal)))
    weight = min(1.0, max(0.0, (LINE_FADE_FAR - ball_r) / (LINE_FADE_FAR - DANGER_FAR)))
    if weight <= 0.0:
        return anchor
    floor = max(STATION_MIN_FROM_GOAL, LINE_SHARE * ball_r)
    out = anchor - goal
    r = float(np.hypot(out[0], out[1]))
    if r >= floor:
        return anchor
    direction = out / r if r > 1e-6 else (ball - goal) / max(ball_r, 1e-6)
    return _clamp_to_pitch(anchor + direction * ((floor - r) * weight))


# -- Zonal marking ------------------------------------------------------------

# Who has left his zone to go at the ball THIS tick, per team, in decision
# order. Compute-phase scratch exactly like the game loop's `committed` list --
# never written to the game state, so the two-phase guarantee holds. Reset on a
# new tick, and whenever a decider arrives with an empty committed list (the
# team's first, or a lone decision made outside the loop).
_ENGAGED: dict[int, tuple[int, list[int]]] = {}
# A/B switch for measurement; on by default.
ZONAL_MARKING = os.environ.get("FB_ZONAL", "1") != "0"


def _engaged_this_tick(
    state: GameState, team: Team, committed: list[tuple[int, np.ndarray]]
) -> list[int]:
    entry = _ENGAGED.get(team.team_id)
    if not committed or entry is None or entry[0] != state.tick:
        entry = (state.tick, [])
        _ENGAGED[team.team_id] = entry
    return entry[1]


def _zonal_assignment(
    state: GameState,
    team: Team,
    carrier: Player | None,
    exclude: set[int],
    centres: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, Player], dict[int, np.ndarray]]:
    """_zonal_assignment_compute, once per tick per team, carrier and
    excluded set (see tick_cache): it is the same for every defender who
    asks, and each one did. The result is shared -- callers only read it."""
    cache = tick_cache.get()
    if cache is None or centres is not None:
        return _zonal_assignment_compute(state, team, carrier, exclude, centres)
    key = ("zonal", team.team_id, None if carrier is None else carrier.player_id,
           frozenset(exclude))
    hit = cache.get(key)
    if hit is None:
        hit = cache[key] = _zonal_assignment_compute(state, team, carrier, exclude)
    return hit


def _zonal_assignment_compute(
    state: GameState,
    team: Team,
    carrier: Player | None,
    exclude: set[int],
    centres: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, Player], dict[int, np.ndarray]]:
    """Share the attackers out among the defenders still holding their zones.

    One-to-one, nearest zone centre first: the attacker standing in a
    defender's zone is his, and when a teammate leaves his zone to press, the
    man he left is picked up by whoever's zone is next nearest -- the cover
    moves with the press instead of leaving a gap behind it. The carrier is the
    pressers' job, not anyone's mark. `centres` may be passed in when the
    caller assigns repeatedly from the same snapshot (see _press_set).
    """
    own_goal = np.array([LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X, GOAL_CENTER_Y])
    defenders = [
        p for p in team.players
        if p.position != "GK" and p.player_id not in exclude
    ]
    threats = [
        o for o in state.opponents_of(team.team_id)
        if o.position != "GK"
        and (carrier is None or o.player_id != carrier.player_id)
        and distance(o.pos, own_goal) < ZONE_THREAT_FAR
    ]
    if centres is None:
        centres = {
            p.player_id: _zone_centre(p, state, team, defensive_bias(p, team))[0]
            for p in defenders
        }
    else:
        centres = {
            p.player_id: centres[p.player_id] if p.player_id in centres
            else _zone_centre(p, state, team, defensive_bias(p, team))[0]
            for p in defenders
        }
    if not defenders or not threats:
        return {}, centres
    dangers = [_danger(t.pos, own_goal[0]) for t in threats]
    pairs = sorted(
        (_zone_gap(d, centres[d.player_id], t), i, j)
        for i, d in enumerate(defenders)
        for j, t in enumerate(threats)
    )
    marks: dict[int, Player] = {}
    taken: set[int] = set()
    for gap, i, j in pairs:
        if gap >= _zone_reach(defenders[i], team, dangers[j]):
            continue
        did = defenders[i].player_id
        if did in marks or j in taken:
            continue
        marks[did] = threats[j]
        taken.add(j)
    return marks, centres


def _zone_reach(defender: Player, team: Team, danger: float) -> float:
    """How far from his ground a man can be and still be his: Man marking
    scales ZONE_RADIUS, so a man marker follows him out and a zonal one hands
    him on sooner. Leaning on the blend alone left the number of men marked
    where it was: most defenders have no man within ZONE_RADIUS to lean on.
    The urgency near our goal is built in: `danger`, the man's, takes a low
    setting back to ZONE_RADIUS."""
    setting = man_marking_bias(defender, team)
    return ZONE_RADIUS * (setting + max(0.0, 1.0 - setting) * danger)


def _zone_gap(defender: Player, centre: np.ndarray, threat: Player) -> float:
    """How far `threat` is from this defender's ground.

    Half from the zone centre, half from where the defender actually stands.
    Zone centres alone handed a striker in the box to a midfielder whose
    centre had slid nearer him than the centre-back's, while the midfielder
    himself was 9 m upfield and the centre-back stood alongside the striker:
    he received the ball unmarked. A zone is ground you can actually defend.
    """
    return ZONE_POS_WEIGHT * distance(defender.pos, threat.pos) + (
        1.0 - ZONE_POS_WEIGHT
    ) * distance(centre, threat.pos)


def _zonal_mark(
    player: Player,
    state: GameState,
    team: Team,
    carrier: Player | None,
    own_goal_x: float,
    exclude: set[int],
) -> tuple[np.ndarray | None, float]:
    """Where to stand to defend my zone, and how strongly I am marking.

    Goal-side of the attacker in my zone and a little toward the ball, so I am
    both between him and the goal and in the lane a pass to him has to use.
    Tighter the nearer he is to goal. Blended with my zone centre by how
    squarely he stands in my zone and how dangerous he is there, so a man on
    the edge of my ground 50 m out draws me a little, and a man in my zone in
    front of goal gets marked tight.
    """
    spot, strength, _ = _zonal_mark_detail(player, state, team, carrier, own_goal_x, exclude)
    return spot, strength


def _mark_action(
    player: Player,
    state: GameState,
    team: Team,
    carrier: Player | None,
    own_goal_x: float,
    exclude: set[int],
) -> OffBallAction | None:
    """The move that marks my man, or None if I have none.

    Under inertia he tracks his MAN's own motion, braking late, and sprints
    whenever the man is quicker than his jog: jogging until 3 m adrift, a
    marker 2.7 m off a runner could never close the gap, and a receiver near
    our goal was free when the ball arrived.
    """
    spot, strength, man = _zonal_mark_detail(player, state, team, carrier, own_goal_x, exclude)
    if spot is None or strength <= 0.0:
        return None
    man_speed = float(np.hypot(man.velocity[0], man.velocity[1]))
    sprint = distance(player.pos, spot) > MARK_SPRINT_GAP or man_speed > JOG_SHARE * player.speed
    return OffBallAction(
        spot, sprint=sprint,
        press_velocity=man.velocity * strength if kinematics.INERTIA else None,
    )


def _zonal_mark_detail(
    player: Player,
    state: GameState,
    team: Team,
    carrier: Player | None,
    own_goal_x: float,
    exclude: set[int],
) -> tuple[np.ndarray | None, float, Player | None]:
    """_zonal_mark, plus the man being marked."""
    marks, centres = _zonal_assignment(state, team, carrier, exclude)
    man = marks.get(player.player_id)
    if man is None:
        return None, 0.0, None
    centre = centres[player.player_id]
    goal = np.array([own_goal_x, GOAL_CENTER_Y])
    from_goal = distance(man.pos, goal)
    closeness = min(1.0, max(0.0, (ZONE_THREAT_FAR - from_goal) / (ZONE_THREAT_FAR - ZONE_THREAT_NEAR)))
    # Man marking: the same shape over his own reach, then leaned on towards
    # the man or the zone. An exponent, so 0 and 1 stay where they are.
    danger = _danger(man.pos, own_goal_x)
    reach = _zone_reach(player, team, danger)
    full = ZONE_FULL * reach / ZONE_RADIUS
    in_zone = min(1.0, max(0.0, (reach - _zone_gap(player, centre, man)) / (reach - full)))
    strength = in_zone * (ZONE_MARK_FLOOR + (1.0 - ZONE_MARK_FLOOR) * closeness)
    strength **= 1.0 / man_marking_bias(player, team)
    # The nearer our goal, the tighter: a man near it is marked fully however
    # squarely he stands in the zone. Blended with the zone centre instead,
    # attackers receiving within 25 m of our goal were more than 3 m free 55%
    # of the time, with the nearest defender goal-side of them only 26%.
    strength = max(strength, danger)

    side = normalize(goal - man.pos) + MARK_BALL_SIDE * normalize(state.ball.pos - man.pos)
    side = normalize(side)
    # Tight from ZONE_THREAT_NEAR in, like the marking strength: standing off by
    # the shot-danger range kept a man 20 m out marked from 2.2 m.
    standoff = MARK_STANDOFF_FAR + (MARK_STANDOFF_NEAR - MARK_STANDOFF_FAR) * closeness
    spot = man.pos + side * standoff
    return _clamp_to_pitch(centre + (spot - centre) * strength), strength, man


def _first_collect_tick(state: GameState, players: list[Player], horizon: int = 120) -> float:
    """_first_collect_tick_compute, once per tick per group of players."""
    cache = tick_cache.get()
    if cache is None:
        return _first_collect_tick_compute(state, players, horizon)
    key = ("first_collect", tuple(p.player_id for p in players), horizon)
    hit = cache.get(key)
    if hit is None:
        hit = cache[key] = _first_collect_tick_compute(state, players, horizon)
    return hit


def _first_collect_tick_compute(state: GameState, players: list[Player], horizon: int = 120) -> float:
    """The earliest tick at which any of `players` can reach the loose ball.

    Walks the ball's real, decelerating path tick by tick (vectorised), and a
    player collects once he can cover the gap within that many ticks.
    """
    ball = state.ball
    movers = [p for p in players if p.speed > 1e-9]
    if not movers:
        return float("inf")
    speed = float(np.linalg.norm(ball.velocity))
    k = np.arange(horizon + 1, dtype=float)
    low = np.ones(horizon + 1, dtype=bool)   # within reach (a ball in the air may not be)
    if airborne(ball.height, ball.vz):
        path, hs = trajectory(ball.pos, ball.velocity, ball.height, ball.vz,
                              _ball_friction(ball), horizon)
        path = np.vstack([ball.pos[None, :], path])
        now = ball.height <= (RISING_REACH if ball.vz > 0.0 else HEADER_HEIGHT)
        low = np.concatenate([[now], within_reach(ball.height, hs, HEADER_HEIGHT)])
    elif speed > 1e-9:
        f = _ball_friction(ball)
        stop = speed / f
        kk = np.minimum(k, np.floor(stop))
        travelled = speed * kk - f * kk * (kk + 1.0) / 2.0
        path = ball.pos[None, :] + normalize(ball.velocity)[None, :] * travelled[:, None]
    else:
        path = np.repeat(ball.pos[None, :], horizon + 1, axis=0)
    pos = np.array([p.pos for p in movers])
    spd = np.array([p.speed for p in movers])
    if kinematics.INERTIA:
        # Every other tick of the path: a (121 x players) table was this
        # check's whole cost, and a tick's precision changes no decision.
        vel = np.array([p.velocity for p in movers])
        path, k, low = path[::2], k[::2], low[::2]
        need = reach_ticks(pos[None, :, :], vel[None, :, :], spd[None, :],
                           path[:, None, :], OUTFIELD_CONTROL_RADIUS)
        reach = need <= k[:, None]
        hit = np.nonzero(reach.any(axis=1) & low)[0]
        return float(k[hit[0]]) if hit.size else float("inf")
    else:
        gap = np.linalg.norm(path[:, None, :] - pos[None, :, :], axis=2)
        reach = gap <= OUTFIELD_CONTROL_RADIUS + spd[None, :] * k[:, None]
    hit = np.nonzero(reach.any(axis=1) & low)[0]
    return float(hit[0]) if hit.size else float("inf")


# Cover-aware pressing: the A/B switch, and the team's press set per tick.
# OFF by default (2026-10-26): at parity with the old first-man-goes rule on
# expected goals conceded, not better (memory: fb-cover-aware-pressing). Turn
# on with FB_COVER_PRESS=1; the version with it on is saved in
# WS-FB_snapshot_2026-10-26_cover_press.
COVER_PRESS = os.environ.get("FB_COVER_PRESS", "0") == "1"
_PRESS_SET: dict = {}


def _press_set(
    state: GameState, team: Team, carrier: Player, own_goal_x: float
) -> set[int]:
    """Who steps out to the carrier this tick (see PRESS_* in constants).

    Joint, from the frozen snapshot, so it does not depend on who happens to
    decide first: the nearest man no longer goes just because he is nearest
    -- a centre-back with their striker beside him stays on him, and a
    defender whose ground is empty goes instead.
    """
    key = (id(state), state.tick, carrier.player_id, float(carrier.pos[0]), float(carrier.pos[1]),
           float(sum(p.pos[0] + 3.0 * p.pos[1] for p in state.all_players())))
    cached = _PRESS_SET.get(team.team_id)
    if cached is not None and cached[0] == key:
        return cached[1]

    defenders = [p for p in team.players if p.position != "GK" and p.speed > 1e-9]
    goal = np.array([own_goal_x, GOAL_CENTER_Y])
    to_goal = normalize(goal - carrier.pos)
    # The point to win is goal-side of where the carrier WILL be, along his
    # run: timed against where he stands now, a chaser right behind him scored
    # as the best presser -- he is close to that spot -- yet never gets in
    # front of a man running at goal, and the defender who was goal-side was
    # stood down as not needed. Shots against got better (shot value 0.49 ->
    # 0.55) and goals rose.
    ts = np.arange(0.0, 2.0 * PRESS_REACH_TICKS * TACTIC_BIAS_MAX + 1.0, 2.0)
    path = carrier.pos[None, :] + np.outer(ts, carrier.velocity) + to_goal[None, :] * GOAL_SIDE_STANDOFF
    path[:, 0] = np.clip(path[:, 0], 0.0, PITCH_LENGTH)
    path[:, 1] = np.clip(path[:, 1], 0.0, PITCH_WIDTH)
    ahead = _clamp_to_pitch(carrier.pos + carrier.velocity * PRESS_LOOKAHEAD_TICKS)
    carrier_threat = max(threat(carrier.pos, own_goal_x), threat(ahead, own_goal_x))

    # Attackers left behind, and how dangerous each is from where the ball is.
    others = [
        o for o in state.opponents_of(team.team_id)
        if o.position != "GK" and o.player_id != carrier.player_id
        and distance(o.pos, goal) < ZONE_THREAT_FAR
    ]
    danger = np.zeros(len(others))
    t_pass = np.zeros(len(others))
    if others:
        pts = np.array([o.pos for o in others])
        gaps = np.hypot(pts[:, 0] - carrier.pos[0], pts[:, 1] - carrier.pos[1])
        paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps])
        lanes = route_survival_many(carrier.pos, pts, paces, BALL_FRICTION_COEFF, team.players)
        for j, (o, lane) in enumerate(zip(others, lanes)):
            reach = FUNNEL_REACH_FLOOR + (1.0 - FUNNEL_REACH_FLOOR) * float(lane)
            danger[j] = threat(o.pos, own_goal_x) * reach
            t_pass[j] = travel_ticks(float(paces[j]), BALL_FRICTION_COEFF, float(gaps[j]))

    # cover[i, j]: can defender i get to attacker j before a pass from the
    # carrier does? The engine's own race. A zone-width measure counted a man
    # 10 m away as half-covering a striker, so leaving him looked cheap and
    # extra pressers stepped out leaving attackers free.
    cover = np.zeros((len(defenders), len(others)))
    gain = np.zeros(len(defenders))
    win = np.zeros(len(defenders))
    centres: dict[int, np.ndarray] = {}
    for i, d in enumerate(defenders):
        pressing = defensive_bias(d, team)
        centres[d.player_id] = _zone_centre(d, state, team, pressing)[0]
        for j, o in enumerate(others):
            t_d = _eta(d, o.pos)
            cover[i, j] = 1.0 / (1.0 + math.exp((t_d - t_pass[j]) / PRESS_TAU))
        # First moment he can stand goal-side of the carrier's run.
        if kinematics.INERTIA:
            need = reach_ticks(d.pos, d.velocity, d.speed, path)
        else:
            need = np.hypot(path[:, 0] - d.pos[0], path[:, 1] - d.pos[1]) / d.speed
        ok = np.nonzero(need <= ts)[0]
        t = float(ts[ok[0]]) if ok.size else float("inf")
        win[i] = 0.0 if t == float("inf") else (
            1.0 / (1.0 + math.exp((t - PRESS_REACH_TICKS * pressing) / PRESS_TAU)))
        gain[i] = pressing * carrier_threat * win[i]

    # How well each attacker is marked by the marking that will ACTUALLY
    # happen: the real zonal assignment among the defenders who stay, each
    # attacker held as well as his marker can reach him. A cost from "could
    # anyone else reach him" assumed cover the assignment then did not
    # provide -- centre-backs left strikers to men never given them, and goals
    # from passes into the box rose (22 -> 30 over 16 x 6000).
    index = {d.player_id: i for i, d in enumerate(defenders)}
    slot = {o.player_id: j for j, o in enumerate(others)}

    def marked(away: set[int]) -> np.ndarray:
        marks, _ = _zonal_assignment(state, team, carrier, away, centres)
        q = np.zeros(len(others))
        for did, man in marks.items():
            if did in index and man.player_id in slot:
                q[slot[man.player_id]] = cover[index[did], slot[man.player_id]]
        return q

    chosen: list[int] = []
    remaining = list(range(len(defenders)))
    unwon = 1.0
    while remaining and len(chosen) < PRESS_MAX:
        away = {defenders[k].player_id for k in chosen}
        held = marked(away)
        best, best_net = None, -float("inf")
        for i in remaining:
            # What leaving costs: the danger of every attacker marked worse
            # once he goes and the assignment is redone without him.
            worse = held - marked(away | {defenders[i].player_id})
            cost = float(np.sum(danger * np.maximum(worse, 0.0)))
            net = gain[i] * unwon - cost
            if net > best_net:
                best, best_net = i, net
        if best is None or (chosen and best_net <= 0.0):
            break
        chosen.append(best)
        remaining.remove(best)
        unwon *= 1.0 - win[best]
    result = {defenders[i].player_id for i in chosen}
    _PRESS_SET[team.team_id] = (key, result)
    return result


def _may_leave_zone(
    player: Player,
    state: GameState,
    team: Team,
    carrier: Player | None,
    own_goal_x: float,
    danger: float,
    target: np.ndarray,
    t_arrive: float,
    engaged: list[int],
) -> bool:
    """May this player leave his zone to go at the ball?

    Only so many may, or the zones behind them empty and the men in them are
    free to receive and score: measured near our goal, 2.6 defenders went at
    a carrier and 5.6 at a loose ball, on average.

      - The first man always goes; somebody has to.
      - A loose ball: only someone who reaches it before every opponent. A
        pass he cannot cut out is better defended by marking its receiver.
      - A carrier near goal, or running at it with a man near enough in front
        of him (FRONT_MAN_TICKS): one goal-side man, if the presser is chasing
        from behind.
      - Beyond that, only the pressing setting buys extra pressers -- none at
        neutral, graded above it (PRESS_EXTRA_WINDOW) -- and never one with a
        dangerous man in his zone.
      - Below neutral even the first man at a carrier needs to be near enough
        (PRESS_FIRST_TICKS).
    """
    others = [pid for pid in engaged if pid != player.player_id]
    if not others:
        if carrier is None:
            return True
        # The first man: below neutral he must be able to get there in time.
        s = defensive_bias(player, team) * (1.0 + DANGER_PRESS_GAIN * danger)
        return s >= 1.0 or t_arrive <= PRESS_FIRST_TICKS * s / (1.0 - s)
    if carrier is None:
        # Against when they can FIRST collect it, not against when they could
        # reach my point: a defender's meeting point far down the ball's path
        # is always "his" -- the receiver takes the ball long before it gets
        # there. Judged that way, 8 of 10 still chased a single pass.
        return t_arrive <= _first_collect_tick(state, state.opponents_of(team.team_id))
    mates = [state.get_player(pid) for pid in others]
    # Further out, against a carrier running at goal, only a man near enough.
    in_front = danger > 0.0 or (
        _running_at_goal(carrier, own_goal_x)
        and t_arrive <= FRONT_MAN_TICKS * defensive_bias(player, team)
    )
    if in_front and _goal_side_of(player, carrier, own_goal_x) and not any(
        _goal_side_of(m, carrier, own_goal_x) for m in mates
    ):
        return True
    pressers = len(others)
    # A goal-side blocker sent in behind a chasing presser is not pressing.
    if in_front and pressers >= 2 and not _goal_side_of(mates[0], carrier, own_goal_x):
        pressers -= 1
    # Whole places, then the part of one left over for a man who is near enough.
    extras = PRESS_EXTRA_PER_UNIT * max(0.0, defensive_bias(player, team) - 1.0)
    whole = int(extras + 1e-9)
    if pressers > whole and (pressers > whole + 1
                             or t_arrive > PRESS_EXTRA_WINDOW * (extras - whole)):
        return False
    _, strength = _zonal_mark(player, state, team, carrier, own_goal_x, set(others))
    return strength < MARK_KEEP_CLOSENESS


def _is_outlet(player: Player, state: GameState, team: Team) -> bool:
    """Is he the one player left up the pitch for the counter?

    The most advanced outfield player -- measured by current position, so the
    outlet is whoever is actually highest, not a fixed role.
    """
    outfield = [p for p in team.players if p.position != "GK"]
    if not outfield:
        return False
    top = max(outfield, key=lambda p: float(p.pos[0]) * team.direction)
    return top.player_id == player.player_id


def _we_have_the_ball(state: GameState, team: Team) -> bool:
    """Is this move ours? Owned by us, or in flight from one of ours."""
    owner = state.get_ball_owner()
    if owner is not None:
        return owner.team_id == team.team_id
    return state.ball.last_toucher_team_id == team.team_id


def _hold_onside(point: np.ndarray, onside_x: float, direction: int) -> np.ndarray:
    """Pull a candidate point back to the offside line, keeping its width.

    Clamping x only -- the lateral position is what makes the run useful and it
    has nothing to do with offside. A point a stride beyond the line becomes a
    point ON the line, which is where a striker actually wants to be.
    """
    px = float(point[0])
    if (px - onside_x) * direction <= 0.0:
        return point
    return np.array([onside_x, float(point[1])])


def _onside_factor(point: np.ndarray, state: GameState, team: Team) -> float:
    """How much a candidate point is discounted for straying offside.

    Was a flat 0.05 past the second-deepest opponent -- a stand-in for a rule
    that did not exist, and wrong in three ways: it ignored the ball (you
    cannot be offside level with or behind it), ignored the halfway line, and
    was a hard cliff, so a forward either hugged the line exactly or gave up on
    the position entirely.

    Now it steers off the real offside line (see offside.py) and fades rather
    than snapping: a stride beyond is a mistimed run worth discouraging, ten
    metres beyond is not a football position at all. The gradient is what lets
    a striker hold the shoulder of the last defender instead of oscillating
    across a threshold.
    """
    from ...engine.offside import onside_limit

    limit = onside_limit(state, team)
    over = (float(point[0]) - limit) * team.direction
    if over <= 0.0:
        return 1.0
    fade = max(0.0, 1.0 - over / OFFSIDE_FADE_DISTANCE)
    return OFFSIDE_MIN_FACTOR + (1.0 - OFFSIDE_MIN_FACTOR) * fade


# -- Goalkeeper (kept as control flow) ----------------------------------------

def _gk_decision(
    player: Player, state: GameState, own_goal_x: float, team: Team
) -> OffBallAction:
    ball = state.ball

    if ball.state == "in_shot":
        crossing_y = _predict_goal_crossing_y(ball, own_goal_x)
        if crossing_y is not None:
            target_y = float(np.clip(crossing_y, GOAL_TOP_Y - 1.0, GOAL_BOTTOM_Y + 1.0))
            target = np.array([own_goal_x + 0.5 * team.direction, target_y])
            if airborne(ball.height, ball.vz) and caught_out(player.pos, own_goal_x):
                # Going over his head: a dive will not take him back to his
                # line. He stops, turns and runs like anyone else.
                return OffBallAction(target, sprint=True, through=True)
            if _crossing_height(ball, own_goal_x) > GK_HIGH_SHOT:
                # Up under the bar he cannot run across and take it: he reads
                # it, then leaps, and the far top corner may be too far.
                if state.tick - ball.struck_tick < GK_REACTION_TICKS:
                    return OffBallAction(player.pos.copy(), sprint=False, dive=True, pace=0.0)
                return OffBallAction(target, sprint=True, dive=True, pace=GK_DIVE_SPEED)
            return OffBallAction(target, sprint=True, dive=True)

    if ball.owner_id is None and distance(ball.pos, player.pos) < 10.0:
        return OffBallAction(ball.pos.copy(), sprint=True)

    # On the line from the ball to the middle of his goal, GK_ADVANCE out: it
    # cuts the angle to both posts equally. He stood on the goal line at the
    # ball's y (clamped to the posts) -- for a carrier 12 m out and 10 m wide,
    # at the near post, ~3 m off the shooting line; under inertia 81% of goals
    # came with him more than 2 m off it. With the ball in range he moves flat
    # out and tracks how his spot moves with the ball, braking late; jogging
    # until 4 m adrift, he was always behind it.
    target = _gk_spot(ball.pos, own_goal_x, team)
    owner = state.get_ball_owner()
    moving = owner.velocity if owner is not None else ball.velocity
    tracking = _gk_spot(ball.pos + moving, own_goal_x, team) - target
    urgent = _danger(ball.pos, own_goal_x) > 0.0
    return OffBallAction(
        target, sprint=urgent or distance(player.pos, target) > 4.0,
        press_velocity=tracking if kinematics.INERTIA and urgent else None,
    )


def _gk_spot(ball_pos: np.ndarray, own_goal_x: float, team: Team) -> np.ndarray:
    """Where the keeper stands: GK_ADVANCE out from the middle of his goal,
    toward the ball, never behind his line."""
    goal = np.array([own_goal_x, GOAL_CENTER_Y])
    u = normalize(ball_pos - goal)
    if float(np.hypot(u[0], u[1])) < 1e-9:
        u = np.array([float(team.direction), 0.0])
    spot = goal + u * GK_ADVANCE
    line = own_goal_x + 0.5 * team.direction
    if (float(spot[0]) - line) * team.direction < 0.0:
        spot[0] = line
    return spot


def _predict_goal_crossing_y(ball: Ball, goal_x: float) -> float | None:
    vx = ball.velocity[0]
    if abs(vx) < 1e-9:
        return None
    t = (goal_x - ball.pos[0]) / vx
    if t < 0:
        return None
    if _crossing_height(ball, goal_x) > CROSSBAR_HEIGHT + GK_LEAVE_OVER_BAR:
        return None                          # going over: he leaves it
    return float(ball.pos[1] + ball.velocity[1] * t)


def _crossing_height(ball: Ball, goal_x: float) -> float:
    """How high the ball is when it reaches the line; below 0, it has landed."""
    return height_after(ball.height, ball.vz, float((goal_x - ball.pos[0]) / ball.velocity[0]))


def _shot_threatens(ball: Ball, own_goal_x: float) -> bool:
    """Is this shot travelling toward our goal?"""
    return (own_goal_x - float(ball.pos[0])) * float(ball.velocity[0]) > 0


def _intercept_action(player: Player, ball: Ball) -> OffBallAction | None:
    ball_dir = normalize(ball.velocity)
    if float(np.linalg.norm(ball_dir)) < 1e-9:
        return None
    proj = float(np.dot(player.pos - ball.pos, ball_dir))
    if proj <= 0.0:
        return None
    return OffBallAction(_clamp_to_pitch(ball.pos + ball_dir * proj), sprint=True)


# -- Utilities ----------------------------------------------------------------

def _ticks_to_reach(player: Player, point: np.ndarray) -> float:
    if player.speed <= 1e-9:
        return float("inf")
    return _eta(player, point)


def _clamp_to_pitch(point: np.ndarray) -> np.ndarray:
    """Off-pitch points would otherwise score well for having no defenders near
    them, which drags players into the touchline."""
    # Plain floats: np.clip per coordinate was ~2 s of every 25 (263k calls).
    return np.array([
        min(PITCH_LENGTH, max(0.0, float(point[0]))),
        min(PITCH_WIDTH, max(0.0, float(point[1]))),
    ])


def _spacing_radius(player: Player, state: GameState) -> float:
    """Inside this distance a teammate counts as crowding him.

    Shrinks with compactness: a compact team accepts standing closer together,
    which is what lets supporting triangles form instead of being pushed apart.
    """
    return TEAMMATE_SPACING_RADIUS / compactness_bias(player, state.get_team(player.team_id))


def _teammate_spacing(
    point: np.ndarray,
    player: Player,
    state: GameState,
    committed: list[tuple[int, np.ndarray]] | None,
) -> float:
    """Discourage occupying ground a teammate already holds or is heading to.

    Measured against teammates' CURRENT positions and, for those who have
    already decided this tick, the target they committed to -- otherwise two
    players converge on the same spot simultaneously, each unaware of the other.

    A smooth penalty rather than a hard exclusion: crowding is sometimes right
    (a wall, a near-post run), it just should not be free.
    """
    claimed = {pid: target for pid, target in (committed or [])}
    radius = _spacing_radius(player, state)
    worst = 1.0
    for mate in state.teammates_of(player.player_id):
        gap = distance(point, claimed.get(mate.player_id, mate.pos))
        if gap >= radius:
            continue
        worst = min(worst, gap / radius)
    return float(max(TEAMMATE_SPACING_FLOOR, worst))


def _spread_from_teammates(
    target: np.ndarray,
    player: Player,
    state: GameState,
    committed: list[tuple[int, np.ndarray]] | None,
) -> np.ndarray:
    """Nudge a target off ground a teammate already holds or has claimed.

    The in-possession sampler can simply score crowded points lower, but out of
    possession there is a single computed target, so it has to be displaced
    instead. Several players converging on one ball should arrive from
    different angles rather than stacking on the same point.
    """
    # Only against teammates who have already committed. Pushing off every
    # nearby teammate summed to "stay exactly where you are" when the whole team
    # converged on one ball, deadlocking the match completely.
    radius = _spacing_radius(player, state)
    for pid, spot in (committed or []):
        offset = target - spot
        gap = float(np.hypot(offset[0], offset[1]))
        if gap >= radius:
            continue
        if gap < 1e-6:
            # Exactly coincident: break the tie deterministically by id so the
            # two players do not both step the same way.
            sign = 1.0 if player.player_id > pid else -1.0
            offset = np.array([0.0, sign])
            gap = 1.0
        push = (radius - gap) * TEAMMATE_SPREAD_STRENGTH
        target = target + (offset / gap) * push
    return _clamp_to_pitch(target)


def _ball_is_contested(player: Player, state: GameState) -> bool:
    """Is an opponent close enough to this loose ball to make it a race?

    Judged on ARRIVAL TIME, not raw distance: players differ in speed, and a
    distance margin called almost everything contested -- 99% of situations,
    which left the uncontested logic dead. The margin is in ticks, so it means
    "could get there within a fifth of a second of me".
    """
    if player.speed <= 1e-9:
        return True
    mine = _eta_to_ball(player, state)
    for opponent in state.opponents_of(player.team_id):
        if opponent.speed <= 1e-9:
            continue
        theirs = _eta_to_ball(opponent, state)
        if theirs <= mine + CONTEST_MARGIN_TICKS:
            return True
    return False


# Under inertia, every player's time to the ball, worked out once per tick.
# The loose-ball predicates below run for most players every tick and each
# compares up to 21 others, so one vectorised pass replaces ~460 scalar ones.
# Exact, not approximate: the ball does not move during the compute phase.
_BALL_ETA: dict = {"key": None, "eta": {}}


def _eta_to_ball(player: Player, state: GameState) -> float:
    """Ticks for `player` to reach the ball where it lies now."""
    if not kinematics.INERTIA:
        return distance(player.pos, state.ball.pos) / player.speed
    key = (id(state), state.tick, float(state.ball.pos[0]), float(state.ball.pos[1]))
    if _BALL_ETA["key"] != key:
        everyone = state.all_players()
        P = np.array([p.pos for p in everyone]); V = np.array([p.velocity for p in everyone])
        S = np.array([p.speed for p in everyone])
        t = reach_ticks(P, V, S, state.ball.pos[None, :])
        _BALL_ETA["key"] = key
        _BALL_ETA["eta"] = {p.player_id: float(x) for p, x in zip(everyone, t)}
    return _BALL_ETA["eta"][player.player_id]


def _eta(player: Player, point: np.ndarray) -> float:
    """Ticks for `player` to reach `point`: from his actual motion under
    inertia, else the old straight line at top speed."""
    if kinematics.INERTIA:
        return float(reach_ticks(player.pos, player.velocity, player.speed, point))
    return distance(player.pos, point) / player.speed



def _just_passed_safely(player: Player, state: GameState) -> bool:
    """Did this player play a pass that is going to reach a teammate?

    Only then should he leave it alone. A loose ball he shanked is still his to
    chase, and so is one an opponent is about to intercept -- treating every
    pass as "someone else's problem" would abandon the ball whenever it went
    astray.
    """
    ball = state.ball
    if ball.last_passer_id != player.player_id:
        return False
    if state.tick - ball.last_pass_tick > ONE_TWO_WINDOW_TICKS:
        return False

    if ball.owner_id is not None:
        return False

    # STRICT ON PURPOSE. DO NOT LOOSEN -- this has been tried and it regressed.
    #
    # _teammate_is_favourite demands a teammate beat EVERY opponent to the ball
    # by 8 ticks. That holds on only 12% of in-flight ticks (median margin
    # -1.3), so the one-two is active for ~11% of its own window, and that
    # looks like a defect worth tuning away. It is not: the low rate is the
    # PRICE of correctness.
    #
    # Replacing it with "no opponent threatens the route AND a teammate is
    # nominally nearer" raised activation 11% -> 34% and produced stand-offs
    # where an opponent was actually arriving first:
    #
    #   pass slider | strict (this) | loosened
    #   1.0         | 0 bad / 419   | 367 bad / 1111  (33.0%)
    #   2.0         | 0 bad / 607   | 301 bad / 1489  (20.2%)
    #
    # Zero against one in four. It breaks the PASS SLIDER specifically, because
    # this predicate fires on every pass, so its error rate scales with pass
    # volume -- every extra pass the slider buys becomes another chance to walk
    # away from a live ball.
    #
    # Judge this guard by its error rate, never by how often it fires. Both
    # scenario tests pass under the broken version too; only an aggregate count
    # over thousands of live ticks catches it.
    return _teammate_is_favourite(player, state)


def ball_progression(ball: Ball, team: Team) -> float:
    """How fast the ball is advancing toward the goal this team attacks.

    In m/tick, from the ball's own velocity -- no history, so the check stays
    inside the frozen snapshot. Negative when the ball is going backwards.
    """
    return float(ball.velocity[0]) * team.direction


def _breaking_forward(ball: Ball, team: Team) -> float:
    """How strongly a break is on, in [0, 1].

    Zero for ordinary shuffling; ramps up over genuine progression. The median
    advance is 0.16 m/tick and the 95th percentile 0.86, so the threshold sits
    well above ordinary play and fires on deep passes and carries.
    """
    rate = ball_progression(ball, team)
    if rate <= ATTACK_CALL_MIN_RATE:
        return 0.0
    span = max(1e-6, ATTACK_CALL_FULL_RATE - ATTACK_CALL_MIN_RATE)
    return float(min(1.0, (rate - ATTACK_CALL_MIN_RATE) / span))


def _our_ball_in_flight(player: Player, state: GameState) -> bool:
    """Is a loose ball ours to support rather than to chase?

    Deliberately cheap: this runs for every player every tick, and the honest
    version -- walking the ball trajectory for all 22 players -- cost a 17x
    slowdown. Straight-line arrival time is a good enough proxy for "is a
    teammate clearly favourite", and the expensive trajectory walk still
    happens later for the player who actually goes.
    """
    ball = state.ball
    if ball.last_toucher_team_id != player.team_id:
        return False
    if float(np.hypot(ball.velocity[0], ball.velocity[1])) < 1e-6:
        return False

    return _teammate_is_favourite(player, state)





def _is_our_best_collector(
    player: Player, state: GameState, at: np.ndarray | None = None
) -> bool:
    """Of our players, is this the one who should go for the loose ball?

    Cheap on purpose -- straight-line arrival, like the rest of this module's
    loose-ball reasoning. Only the nearest man is held back from an attacking
    run; the others have no reason to converge on a ball already covered.

    `at` overrides the point judged. Attack-call routing asks about the ball
    where it lies; a passer deciding whether to chase his own pass has to ask
    about where it is GOING, or a ball rolling away from him still counts as
    his because he happens to be nearest to it right now.
    """
    target = state.ball.pos if at is None else at
    if player.speed <= 1e-9:
        return False

    def eta(p: Player) -> float:
        return _eta_to_ball(p, state) if at is None else _eta(p, target)

    mine = eta(player)
    for mate in state.teammates_of(player.player_id):
        if mate.speed <= 1e-9:
            continue
        if eta(mate) < mine:
            return False
    return True


def _can_win_the_ball(player: Player, state: GameState) -> bool:
    """Can THIS player reach the loose ball before any opponent?

    _is_our_best_collector answers an intra-team question -- of our players,
    whose ball is this? -- and iterates teammates only. That is right for
    allocating a chase, but says nothing about whether the ball stays ours,
    and the attack-call gate was reading it as though it did.

    The obvious repair -- "is the ball threatened at all?" -- is the one that
    already failed. `_our_ball_in_flight` asked exactly that and was far too
    strict: most threatened balls involve a player 25 m away, for whom the
    answer is yes and irrelevant, and holding him back cost the entire
    attacking gain (box presence 4.7% -> 0.9%).

    So the question is personal, not global: not "is someone going to take it"
    but "am I the one who can stop them". That scopes the override to short
    distances without hardcoding a radius. Verified on 10 paired held-out
    seeds: box presence 0.095 -> 0.150 (8/10 seeds, CI clears zero),
    completion and goals unchanged.

    Straight-line arrival, matching _is_our_best_collector, _should_engage and
    _teammate_is_favourite -- a pursuit question, not a question about a ball
    rolling along a lane.
    """
    if player.speed <= 1e-9:
        return False
    mine = _eta_to_ball(player, state)
    for opponent in state.opponents_of(player.team_id):
        if opponent.speed <= 1e-9:
            continue
        if _eta_to_ball(opponent, state) < mine:
            return False
    return True


def _teammate_is_favourite(player: Player, state: GameState) -> bool:
    """Will one of our players reach this loose ball clearly before any of
    theirs? Straight-line arrival time, deliberately cheap."""
    ball = state.ball
    ours = float("inf")
    for mate in state.teammates_of(player.player_id):
        if mate.speed <= 1e-9:
            continue
        # The keeper counts too: he is a legitimate receiver of a back-pass and
        # excluding him would need a role check this module deliberately avoids.
        ours = min(ours, _eta_to_ball(mate, state))
    theirs = float("inf")
    for opponent in state.opponents_of(player.team_id):
        if opponent.speed <= 1e-9:
            continue
        theirs = min(theirs, _eta_to_ball(opponent, state))
    return ours + ONE_TWO_SAFETY_TICKS < theirs
