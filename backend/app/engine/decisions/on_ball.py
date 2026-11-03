"""
On-ball decisions: enumerate every option, score each as P x V x C, take the best.

This replaces a priority cascade in which a shot scoring 0.35 beat a pass
scoring 0.99 because shots were simply tested first. Because P, V and C are all
bounded and mean the same thing for every action, the argmax is meaningful:
utility is an expected goal-probability contribution, comparable across kinds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..constants import (
    PASS_RANGE_EXPONENT,
    PASS_RANGE_PIVOT_M,
    ANGLE_GAIN_MAX,
    ANGLE_GAIN_NEED_BELOW,
    ANGLE_GAIN_WEIGHT,
    BALL_OVERRUN_DIST,
    MIN_PASS_DIST,
    PASS_UNVIABLE_PENALTY,
    GK_SAFE_PASS_FLOOR,
    GK_SAFETY_EXPONENT,
    PASS_VIABILITY_FLOOR,
    SHOT_DECISIVENESS,
    DRIBBLE_CONTEST_FLOOR,
    DRIBBLE_CONTEST_RADIUS,
    DRIBBLE_CONTEST_WEIGHT,
    DRIBBLE_LOOKAHEAD_FACTOR,
    DRIBBLE_ROOM_FLOOR,
    DRIBBLE_ROOM_RADIUS,
    DRIBBLE_CONTINUATION_DISCOUNT,
    OUT_OF_PLAY_FLOOR,
    PITCH_WIDTH,
    TOUCHLINE_MARGIN,
    RECEPTION_GRACE_TICKS,
    RECEPTION_TAU,
    RECEPTION_FROM_FRACTION,
    THROUGH_BEYOND_M,
    THROUGH_SHOT_FROM_M,
    RECEPTION_HORIZON_TICKS,
    PITCH_WIDTH,
    HOLD_CONTEST_RADIUS,
    HOLD_LOSS_PER_PRESSER,
    HOLD_PATIENCE_TICKS,
    HOLD_VALUE_DISCOUNT,
    PASS_CONTINUATION_DISCOUNT,
    BALL_FRICTION_COEFF,
    LOOKAHEAD_CLOSING,
    LOOKAHEAD_EXPAND,
    LOOKAHEAD_WEIGHT,
    OUTFIELD_CONTROL_RADIUS,
    SHOT_FRICTION_COEFF,
    SHOT_SPEED,
    PITCH_LENGTH,
    POSSESSION_LOSS_BASE,
    POSSESSION_LOSS_THREAT,
    SHIELD_AWARENESS,
    SHIELD_EDGE_MARGIN,
    SHIELD_SPEED_FRACTION,
)
from ..constants import CONTROL_HEIGHT, CROSS_FINISH_FACTOR, GK_REACH_HEIGHT, HEADER_HEIGHT
from ..constants import CROSS_FREE_FLOOR, CROSS_FREE_TAU
from ..constants import CROSSBAR_HEIGHT, GOAL_BOTTOM_Y, GOAL_TOP_Y, LOFT_ANGLE_ERROR_GAIN, LOFT_SHOT_VALUE
from ..constants import SHOT_CURL_ERROR_MAX, SHOT_ERROR_MAX_RAD, SHOT_LIFT_ERROR_MAX
from ..constants import DIRECT_RUN_M
from ..constants import WING_INNER_M, WING_OUTER_M, WING_TILT_GAIN
from ..constants import IDLE_FREE_TICKS, IDLE_GAIN_MAX, IDLE_MAX_TICKS, IDLE_MIN_GAP, IDLE_PACE_SHARE
from .. import kinematics
from ..ball_flight import bend_start, height_after, trajectory, turned, within_reach
from ..ball_physics import nominal_pass_speed, travel_ticks
from ..geometry import distance, normalize
from ..kinematics import reach_ticks
from ..models import GameState, Player
from . import pace as _pace
from .candidates import Candidate, build_candidates
from .interception import (
    control_radius,
    lofted_shot_survival,
    lofted_survival,
    route_survival,
    route_survival_many,
)
from .tunables import (
    context_multiplier,
    crossing_bias,
    lofted_bias,
    risk_bias,
    tempo_bias,
    wing_attacks_bias,
    pass_range_bias,
    possession_bias,
    direct_bias,
    tactical_bias,
)
from .value import (
    clear_path_bonus,
    goalside_value,
    penetration_value,
    position_value,
    shot_value,
    threat,
)
from .zones import classify_zone


@dataclass(slots=True)
class OnBallAction:
    # "control": trying to trap a ball that arrived too fast (see control.py).
    action_type: Literal["shot", "pass", "dribble", "hold", "control"]
    velocity: np.ndarray                     # ball velocity to apply
    runway_signal_target: np.ndarray | None  # only set for runway passes
    signal_recipient_id: int | None          # teammate to signal
    signal_runners: tuple[int, ...] = ()     # everyone told to run onto it
    # The carrier's own step this tick, for a shielding hold. Holding is not
    # standing still: he turns away from the presser and keeps the ball alive.
    move: np.ndarray | None = None
    # Who a pass is FOR, to-feet passes included. Runway passes carry this as
    # signal_recipient_id; a to-feet pass used to carry nobody, so while it was
    # in flight nothing knew who was meant to take it.
    receiver_id: int | None = None
    touch: float | None = None   # a dribble's solved touch length
    vz: float = 0.0              # upward launch speed of a lofted ball
    curl: float = 0.0            # how much a bent shot turns, rad/tick
    # Where a pass is played TO (his feet, or where his run takes him); the
    # receiver times his run to it.
    pass_point: np.ndarray | None = None
    is_cross: bool = False


def _pass_range_factor(dist: float, bias: float) -> float:
    """Preference multiplier for a pass of this length.

    Symmetric about PASS_RANGE_PIVOT_M and about bias 1.0: a ball at the pivot
    always scores exactly 1.0, and at bias 1.0 every length scores 1.0, so the
    middle of the slider really does express no preference. Longer balls scale
    with `bias` and shorter ones with its reciprocal -- log(ratio) changes sign
    at the pivot, which is what makes the pair reciprocal without a branch.

    Raising the slider therefore does not make the team pass MORE; it moves
    which lengths win the argmax. The exponent is fractional so this stays a
    tilt on P x V rather than an override of it.
    """
    # Neutral is the default and by far the commonest setting, and at exactly
    # 1.0 this returns 1.0 for every distance (1**x == 1). Short-circuiting it
    # skips a math.log per pass candidate per tick -- pure waste on a hot path,
    # since the result is provably the identity. Pinned by
    # test_neutral_setting_has_no_preference_at_any_distance.
    if bias == 1.0 or dist <= 1e-9 or bias <= 0.0:
        return 1.0
    return float(bias ** (PASS_RANGE_EXPONENT * math.log(dist / PASS_RANGE_PIVOT_M)))


def direct_factor(run: float, direct: float) -> float:
    """Preference multiplier for a pass the receiver must run `run` metres to:
    Direct ** (1 - 2 x min(1, run / DIRECT_RUN_M)). Above 1.0 a ball to his feet
    gains and one he must chase loses by the reciprocal; 1.0 at 10 m, and for
    every pass at Direct 1.0."""
    if direct == 1.0 or direct <= 0.0:
        return 1.0
    return float(direct ** (1.0 - 2.0 * min(1.0, run / DIRECT_RUN_M)))


def _receiver_run(cand: Candidate, player: Player, state: GameState) -> float:
    """How far the man it is for must go to the ball's target, from where he is
    now; the nearest teammate's distance when it is for nobody in particular."""
    if cand.recipient_id is not None:
        return distance(state.get_player(cand.recipient_id).pos, cand.target)
    return min((distance(m.pos, cand.target) for m in state.teammates_of(player.player_id)),
               default=0.0)


def score_candidate(cand: Candidate, player: Player, state: GameState) -> Candidate:
    """Fill in p / v / context / utility for one option."""
    team = state.get_team(player.team_id)
    attacking_goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    opponents = state.opponents_of(player.team_id)

    if cand.kind == "hold":
        # Holding keeps what you already have, discounted for making no
        # progress. P is not 1.0: standing still with the ball invites pressure,
        # and the closer the nearest opponent the likelier you are dispossessed.
        # Without this, holding is a risk-free option that beats every pass with
        # P < 1, and carriers freeze whenever the pitch is congested.
        #
        # P was `nearest / 8 m` -- a guess, never measured, and badly wrong: it
        # scored shielding at 0.19 with a man 1.5 m away, so a pressed carrier
        # NEVER held (0% of pressured actions) and was pushed into forced
        # passes. Measured by forcing carriers to hold for 0.5 s at random under
        # pressure: the ball was still ours 1.5 s later 80% of the time, at
        # <1.5 m and at 1.5-3 m alike -- better than the passes (70%) and
        # dribbles (61%) they chose instead.
        #
        # So model what actually decides it: dispossession is a strength
        # contest, and holding is lost to the combined strength of opponents
        # close enough to join it. One presser of equal strength -> 0.80.
        contest = sum(
            o.strength / max(player.strength, 1e-9)
            for o in opponents
            if distance(player.pos, o.pos) <= HOLD_CONTEST_RADIUS
        )
        cand.p = 1.0 / (1.0 + HOLD_LOSS_PER_PRESSER * contest)
        # ...for a SHORT wait. Shielding erodes the longer it goes on: forced
        # 0.5 s holds kept the ball 80% of the time, sustained holding only
        # 69%, and with no decay a carrier whose options were all mediocre
        # could stand on the ball indefinitely.
        cand.p *= math.exp(-player.hold_ticks / HOLD_PATIENCE_TICKS)
        cand.v = position_value(player.pos, attacking_goal_x) * HOLD_VALUE_DISCOUNT
    elif cand.is_cross:
        # A cross is contested where it arrives, receiver against defenders,
        # and that is the reception race's job. The lane counted the same
        # defenders again with nobody contesting them and scored every cross
        # P = 0, so it covers only the flight before the arrival.
        end = player.pos + (cand.target - player.pos) * RECEPTION_FROM_FRACTION
        if cand.vz > 0.0:
            cand.p = lofted_survival(player.pos, cand.target, cand.speed, cand.vz, opponents,
                                     upto=RECEPTION_FROM_FRACTION)
        else:
            cand.p = route_survival(player.pos, end, cand.speed, cand.friction, opponents)
    elif cand.kind == "shot" and cand.vz > 0.0:
        cand.p = lofted_shot_survival(player.pos, cand.target, cand.speed, cand.vz, opponents, cand.curl)
    elif cand.vz > 0.0:
        # Lofted: only cut out where it is low enough (see lofted_survival).
        # A through ball's too, drop and all: stopped where the race starts
        # (_lane_end) its drop was never counted, it scored as if nobody could
        # meet it, and 82-88% of through balls from midfield went in the air.
        cand.p = lofted_survival(player.pos, cand.target, cand.speed, cand.vz, opponents)
    elif cand.lane is not None:
        cand.p = cand.lane              # worked out with the others (_batch_lanes)
    else:
        cand.p = route_survival(
            player.pos, _lane_end(cand, player, state), cand.speed, cand.friction, opponents,
            is_shot=cand.kind == "shot",
            # A dribble touch stays at his feet, where a presser can tackle;
            # only a struck ball gets past the man beside him.
            struck=cand.kind != "dribble",
        )
    if cand.kind != "hold":
        if cand.kind == "pass":
            # A pass succeeds only if a TEAMMATE ends up with it. Surviving the
            # flight is necessary but not sufficient: the ball also has to be
            # won at the far end. Scoring only the flight is why balls were
            # played into space nobody claimed.
            cand.p *= (cand.reception if cand.reception is not None
                       else _reception_probability(cand, player, state, opponents))
            # ...and he has to be able to control it at the pace it arrives.
            # This is the passer weighting the ball: faster always survives
            # the lane better, so without it every pass drifted to its
            # hardest option and receivers took 24 m/s balls like gentle ones.
            cand.p *= _receivable_pace(cand, player, state)
        if cand.kind == "dribble":
            # route_survival over a 3 m touch barely notices anyone, so count
            # the opponents who could realistically contest the carry.
            cand.p *= _dribble_contest(cand, player, opponents)
            if cand.race is not None:
                # A paced dribble: the touch is loose until he reaches it, and
                # a longer, faster one is exposed for longer (pace.py).
                cand.p *= cand.race
        if cand.kind in ("pass", "dribble"):
            # A ball that runs out of play is lost as surely as an intercepted
            # one. Nothing modelled the touchline, so passes were aimed at
            # targets on or near the line and simply rolled out -- the largest
            # single source of throw-ins.
            cand.p *= _in_play_probability(cand, player)
        if cand.kind == "shot":
            # A shot's value is terminal: it either scores or it does not.
            # Everything else is discounted as a continuation, and dribbles can
            # additionally earn a clear-path bonus of up to 1.9x. Without a
            # matching term a shot loses to a carry even at 15 m with P=0.67 --
            # measured on eight declined chances in a single match.
            cand.v = shot_value(player.pos, attacking_goal_x) * SHOT_DECISIVENESS
            if cand.vz > 0.0:
                cand.v = _on_frame(cand, player) * LOFT_SHOT_VALUE * SHOT_DECISIVENESS
            else:
                from .candidates import SHOT_AIM
                if SHOT_AIM:
                    # ...and along the ground he can miss too (_on_target).
                    cand.p *= _on_target(cand, player)
        elif cand.is_cross:
            # A cross is worth the first-time finish it sets up: the shot from
            # the spot, through the defenders there. The pass multipliers
            # (defenders beaten, goal-side) stacked to V 1.4 on a ball into
            # the box -- more than a goal -- because a cross beats everyone
            # by definition.
            goal = np.array([attacking_goal_x, PITCH_WIDTH / 2.0])
            from .candidates import CROSS_REASONING
            if kinematics.GRIP and CROSS_REASONING:
                # Under grip: worth what a ball to that spot is worth, as for
                # any pass, times how free the man on the end of it is. Valued
                # as a shot from the spot through every defender in front
                # (the keeper included), a cross scored ~0.02-0.06 in a normal
                # box, so it only ever won when everything else was blocked --
                # crosses came as a last resort, into crowds, and never to the
                # free man at the far post.
                cand.v = (position_value(cand.target, attacking_goal_x)
                          * PASS_CONTINUATION_DISCOUNT
                          * _finisher_freedom(cand, state, opponents))
            else:
                cand.v = (shot_value(cand.target, attacking_goal_x) * SHOT_DECISIVENESS
                          * CROSS_FINISH_FACTOR
                          * route_survival(cand.target, goal, SHOT_SPEED, SHOT_FRICTION_COEFF,
                                           opponents, is_shot=True))
        else:
            # Value of the ball ending up where it lands. Because position_value
            # already contains shot_value, a ball into the box is worth more than
            # a square one without any separate "leads to a shot" term.
            #
            # Passes AND dribbles are both continuations, so both are discounted:
            # applying it to passes alone made dribbling win almost every
            # decision, since a 3 m touch is nearly certain to succeed while a
            # 20 m pass must survive the defence.
            cand.v = position_value(cand.target, attacking_goal_x)
            cand.v *= PASS_CONTINUATION_DISCOUNT
            # A move is worth more than its destination when it takes defenders
            # out of the game. Without this a ball played in behind scored the
            # same as a square ball to the same square metre, so nobody ever
            # tried to break the line.
            cand.v *= penetration_value(
                player.pos, cand.target, opponents, attacking_goal_x
            )
            # Does this move leave MORE or FEWER opponents between the ball and
            # the goal? A ball into a good-looking pocket is worth much less if
            # it puts three defenders back in front of it, and a carry that
            # takes defenders out of the picture is worth more than the ground
            # it gains.
            cand.v *= goalside_value(
                player.pos, cand.target, opponents, attacking_goal_x
            )
            if cand.kind == "dribble":
                # Carrying to open a passing angle is a real reason to dribble,
                # and it was invisible: a dribble was scored only on where the
                # ball lands, never on the lanes it unlocks. A 3 m step
                # sideways can lift the best available pass from 0.5 to 0.97.
                cand.v *= _angle_gain(cand, player, state, opponents)
                # A clear run at goal is worth carrying. With nobody in front,
                # passing usually hands the ball to a teammate with more
                # defenders ahead of them -- so the carry keeps its value and
                # the player runs until he is close enough to shoot.
                cand.v *= clear_path_bonus(player.pos, opponents, attacking_goal_x)
                # Where you dribble INTO matters more than the 3 m of ground
                # you gain. route_survival over such a short hop is high in
                # almost every direction, so without this the direction was
                # chosen purely on the destination's static value and players
                # carried the ball straight into crowds.
                cand.v *= _dribble_room(cand, player, opponents)
                # A dribble also leaves the ball loose at your feet with the
                # defence closing, so it is worth less than the same position
                # reached by a pass that is already complete.
                cand.v *= DRIBBLE_CONTINUATION_DISCOUNT
                if cand.pace == "walk":
                    # Walking it keeps the ball while the runs develop: raised
                    # by what the best pass a second from now is worth over
                    # the ball where he stands (pace.wait_gain). A factor, not
                    # a floor: a floor gave every walk direction the same V,
                    # and he turned back and forth on the spot.
                    cand.v *= _pace.wait_gain(player, state, opponents)
            elif THROUGH_CHANCE and _through_ball(cand, player, state):
                # A through ball is worth the chance it makes, not the men it
                # beats: penetration scored every ball beyond the line alike,
                # wide or central, 25 m out or 12, and through balls were
                # played where they led to nothing.
                cand.v = PASS_CONTINUATION_DISCOUNT * max(
                    position_value(cand.target, attacking_goal_x),
                    _through_chance(cand, state, opponents, attacking_goal_x),
                )

    if cand.kind == "pass" and cand.p < PASS_VIABILITY_FLOOR:
        # NOTE: an absolute floor. Weighted passes are slower and so score
        # lower across the board, which is why this is a gentle ramp rather
        # than a cliff -- the aim is to stop genuine giveaways, not to make the
        # carrier hold the ball whenever nothing is perfect.
        # Below this the ball is a giveaway, and it was still being played
        # because every alternative scored lower -- the argmax had no notion
        # that some options simply should not be on the table.
        #
        # Scaled by how far below the floor it sits, so a marginal ball is
        # nudged while a hopeless one is crushed. Suppress rather than veto: a
        # desperate clearance out of defence must stay reachable.
        shortfall = 1.0 - cand.p / PASS_VIABILITY_FLOOR
        cand.p *= 1.0 - (1.0 - PASS_UNVIABLE_PENALTY) * shortfall

    zone = classify_zone(player.pos, attacking_goal_x)
    cand.context = context_multiplier(zone, player.position, player.tactical, cand.kind)
    cand.context *= tactical_bias(player, team, cand.kind)
    if cand.kind == "pass" and not cand.is_clearance:
        # Air or ground: the same reciprocal trade, for crosses as for passes.
        lofted = lofted_bias(player, team)
        tilt = lofted if cand.vz > 0.0 else 1.0 / lofted
        cand.context *= tilt
        cand.flavour = tilt
    if cand.pace is not None:
        # Tempo: the faster paces against the walk, as Pass range trades length.
        cand.context *= _pace.tempo_factor(cand.pace_share, tempo_bias(player, team))
    if cand.kind == "pass" and not cand.is_clearance:
        # Tempo: how fast the ball goes forward -- a forward ball against a
        # square or back one, the same reciprocal trade (pace.progress_factor).
        cand.context *= _pace.progress_factor(
            (float(cand.target[0]) - float(player.pos[0])) * team.direction,
            tempo_bias(player, team),
        )
    if cand.is_cross:
        # Whether to cross at all. Runway and range are left out: a cross
        # goes to a box role by definition, whatever its length.
        cand.context *= crossing_bias(player, team)
    elif cand.kind == "pass":
        # Direct pass: by how far the receiver must run to it (direct_factor).
        tilt = direct_factor(_receiver_run(cand, player, state), direct_bias(player, team))
        cand.context *= tilt
        cand.flavour *= tilt
        # Wing attacks: a pass to the flank against the rest (_wing_tilt).
        tilt = _wing_tilt(cand.target, wing_attacks_bias(player, team))
        cand.context *= tilt
        cand.flavour *= tilt
        # Pass RANGE, the same trade applied to distance rather than flavour:
        # a ball longer than the pivot scales up as the slider rises and a
        # shorter one scales down by the reciprocal, so the team passes just as
        # often -- it simply prefers a different length. At exactly 1.0 both
        # factors are 1.0 and the slider expresses no preference at all.
        tilt = _pass_range_factor(
            float(np.linalg.norm(cand.target - player.pos)),
            pass_range_bias(player, team),
        )
        cand.context *= tilt
        cand.flavour *= tilt
    _set_utility(cand, player, team)
    return cand


def _wing_tilt(target: np.ndarray, wing: float) -> float:
    """The Wing attacks factor for a pass to `target`: wing ** (WING_TILT_GAIN
    x how far out on the flank it is, 0 in the middle to 1 from WING_OUTER_M).
    1.0 at neutral and for any ball to the middle."""
    if wing == 1.0:
        return 1.0
    out = abs(float(target[1]) - PITCH_WIDTH / 2.0)
    flank = min(1.0, max(0.0, (out - WING_INNER_M) / (WING_OUTER_M - WING_INNER_M)))
    return float(wing ** (WING_TILT_GAIN * flank)) if flank > 0.0 else 1.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _strike_sigma(player: Player) -> float:
    """The angle error game_loop puts on his strike, as a standard deviation:
    SHOT_ERROR_MAX_RAD by his skill, more off a fast ball
    (FIRST_TIME_ERROR_GAIN), a header counted as at least half hard."""
    from ..constants import FIRST_TIME_ERROR_GAIN
    from .control import hardness

    sigma = SHOT_ERROR_MAX_RAD * (1.0 - min(1.0, max(0.0, player.shooting / 100.0)))
    if player.first_touch_speed > 0.0:
        hard = hardness(player.first_touch_speed, player)
        if player.first_touch_height > CONTROL_HEIGHT:
            hard = max(hard, 0.5)
        sigma *= 1.0 + FIRST_TIME_ERROR_GAIN * hard
    return sigma


def _inside_posts(origin: np.ndarray, target: np.ndarray, sigma: float) -> float:
    """P(a ball struck from `origin` at `target` with this angle error passes
    between the posts): the error against the angle each post stands off the
    aim. An angle, so the same error misses by more metres from further out."""
    if sigma <= 0.0:
        return 1.0
    aim = target - origin
    bearing = math.atan2(aim[1], aim[0])
    off = []
    for post in (GOAL_TOP_Y, GOAL_BOTTOM_Y):
        angle = math.atan2(post - origin[1], target[0] - origin[0]) - bearing
        off.append(math.atan2(math.sin(angle), math.cos(angle)))
    return _normal_cdf(max(off) / sigma) - _normal_cdf(min(off) / sigma)


def _on_frame(cand: Candidate, player: Player) -> float:
    """His chance of hitting the frame with this lofted shot: inside the posts
    and under the bar, from the errors game_loop will put on the strike. The
    aim's angle error against the angle each post stands off the aim, and the
    lift error, which grows with the ticks the ball is up, against how far the
    bar is above it. A bent shot's spin error moves where it arrives by half
    its turn x that error, which adds to the aim's. Off a fast ball the aim is
    worse, as game_loop has it (_strike_sigma)."""
    from .candidates import AIM_IN_THE_AIR

    miss = 1.0 - min(1.0, max(0.0, player.shooting / 100.0))
    if miss <= 0.0:
        return 1.0
    aim = cand.target - player.pos
    ticks = float(np.linalg.norm(aim)) / cand.speed
    angle = _strike_sigma(player) if AIM_IN_THE_AIR else SHOT_ERROR_MAX_RAD * miss
    sigma = math.hypot(angle * LOFT_ANGLE_ERROR_GAIN,
                       0.5 * cand.curl * ticks * SHOT_CURL_ERROR_MAX * miss)
    wide = _inside_posts(player.pos, cand.target, sigma)
    rise = height_after(0.0, cand.vz, ticks)
    return wide * _normal_cdf((CROSSBAR_HEIGHT - rise) / (cand.vz * SHOT_LIFT_ERROR_MAX * miss * ticks))


def _on_target(cand: Candidate, player: Player) -> float:
    """His chance of putting this ground shot inside the posts, from the angle
    error on his strike (_strike_sigma, _inside_posts): the same skill misses by
    more metres from further out, and an aim nearer the post misses more often.
    Left out, every shot was aimed 0.5 m inside a post from any range:
    one-on-ones from 12-18 m went wide 23% of the time."""
    return _inside_posts(player.pos, cand.target, _strike_sigma(player))


def _receivable_pace(cand: Candidate, passer: Player, state: GameState) -> float:
    """The receiver's chance of controlling this pass at its arrival pace.

    Arrival pace from the ball's own deceleration over the distance, felt
    relative to how the receiver is moving -- a runner taking a ball along his
    run feels it slower than a man standing still.
    """
    from .control import control_probability, first_time_quality, relative_speed

    if cand.recipient_id is None:
        return 1.0
    try:
        receiver = state.get_player(cand.recipient_id)
    except KeyError:
        return 1.0
    offset = cand.target - passer.pos
    dist = float(np.hypot(offset[0], offset[1]))
    if dist < 1e-9:
        return 1.0
    # A cross is not meant to be controlled: it is finished first time.
    take = first_time_quality if cand.is_cross else control_probability
    if cand.vz > 0.0:
        # Dropping onto him: its pace in the air plus its fall, which is the
        # launch's own upward speed coming back down.
        from ..constants import AERIAL_CONTROL_FACTOR

        across = relative_speed(offset / dist * cand.speed, receiver.velocity)
        pace = float(np.hypot(across, cand.vz))
        return take(pace, receiver) * (1.0 if cand.is_cross else AERIAL_CONTROL_FACTOR)
    t = travel_ticks(cand.speed, cand.friction, dist)
    if t == float("inf"):
        return 1.0   # it dies before it gets there: no pace problem
    arrival = max(0.0, cand.speed - cand.friction * t)
    return take(relative_speed(offset / dist * arrival, receiver.velocity), receiver)


def _set_utility(cand: Candidate, player: Player, team, risky: bool = True) -> None:
    """P x V x C, less the turnover cost of a possession-minded side.

    For passes and dribbles P is weighed by the carrier's appetite for risk,
    P ** (1 / risk) (risk_bias). The turnover cost keeps the plain P: it is the
    real chance of losing the ball. `risky=False` for a trap, which is scored
    as a pass but is not a gamble on anything.
    """
    p = cand.p
    if risky and cand.kind in ("pass", "dribble") and p > 0.0:
        risk = risk_bias(player, team)
        if risk != 1.0:
            p = p ** (1.0 / risk)
    cand.utility = p * cand.v * cand.context
    cost = _turnover_cost(cand, player, team) if cand.kind in ("pass", "dribble", "shot") else 0.0
    cand.utility -= cost
    # Without the pass-kind tilts: what the pass is really worth (PASS_STAGES).
    cand.plain = (cand.utility if cand.flavour == 1.0
                  else p * cand.v * (cand.context / cand.flavour) - cost)


# Lookahead switch. OFF: measured on the large sample (16 x 7500, 2026-10-15)
# no variant beat plain one-step scoring against a 1.6 press -- pressed
# side's box entries 53 (off) vs 25 (blend), 32 (max), 37 (max, discounted
# 0.6). Valuing the 2-step play was not the bottleneck: its FIRST pass must
# still get through the press. Kept for re-testing once that changes (e.g.
# with lofted balls).
LOOKAHEAD = False
# How the follow-up combines with a pass's own value: "max" (it can only add)
# or "blend" (weighted average, LOOKAHEAD_WEIGHT). Under test.
LOOKAHEAD_MODE = "max"
LOOKAHEAD_DISCOUNT = 0.6


def _add_lookahead(scored: list[Candidate], player: Player, state: GameState) -> None:
    """Re-value the likeliest passes by what the receiver can do next.

    See LOOKAHEAD_* in constants. One expansion per receiver -- his most
    likely pass -- ranked by P, not utility: a switch to a free man is a
    low-value, high-P ball, and ranking by utility would prune exactly the
    pass this exists to find before its follow-up was ever seen.
    """
    team = state.get_team(player.team_id)
    best_per_receiver: dict[int, Candidate] = {}
    for c in scored:
        if c.kind != "pass" or c.recipient_id is None or c.p <= 0.0:
            continue
        cur = best_per_receiver.get(c.recipient_id)
        if cur is None or c.p > cur.p:
            best_per_receiver[c.recipient_id] = c
    chosen = sorted(best_per_receiver.values(), key=lambda c: c.p, reverse=True)[:LOOKAHEAD_EXPAND]
    follow: dict[int, float] = {}
    for c in chosen:
        follow[c.recipient_id] = _follow_up_value(c, player, state, team)
    for c in scored:
        if c.kind == "pass" and c.recipient_id in follow:
            f = follow[c.recipient_id]
            if LOOKAHEAD_MODE == "max":
                # The follow-up can only ADD: a pass whose next move is better
                # than its landing spot is promoted, a direct ball keeps its
                # own value. Averaging instead devalued every forward pass
                # into a crowd and taught safe recycling, not 2-3-step play
                # (large sample: pressed side's box entries vs a press 53 -> 25).
                # Discounted per step: a move one pass later is worth less than
                # the same move now. Undiscounted, a safe pass to a free man
                # always borrowed the best next move from anywhere and players
                # circulated forever (back passes 18% -> 34%, shots 44 -> 26).
                c.v = max(c.v, LOOKAHEAD_DISCOUNT * f)
            else:
                c.v = (1.0 - LOOKAHEAD_WEIGHT) * c.v + LOOKAHEAD_WEIGHT * f
            _set_utility(c, player, team)


def _follow_up_value(cand: Candidate, passer: Player, state: GameState, team) -> float:
    """The receiver's best next move, judged where and when he gets the ball."""
    from types import SimpleNamespace

    receiver = state.get_player(cand.recipient_id)
    at = cand.target
    attacking_goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    dist = float(np.hypot(*(at - passer.pos)))
    t = travel_ticks(cand.speed, cand.friction, dist)
    if t == float("inf"):
        t = dist / max(cand.speed, 1e-9)
    # Defenders close in toward the receiving point while the ball travels,
    # at LOOKAHEAD_CLOSING of their full run, stopping a control radius short.
    opponents = []
    for o in state.opponents_of(passer.team_id):
        off = at - o.pos
        n = float(np.hypot(*off))
        step = min(max(0.0, n - OUTFIELD_CONTROL_RADIUS), o.speed * t * LOOKAHEAD_CLOSING)
        pos = o.pos + (off / n * step if n > 1e-9 else 0.0)
        opponents.append(SimpleNamespace(
            pos=pos, velocity=np.zeros(2), speed=o.speed, position=o.position,
            goalkeeping=o.goalkeeping, player_id=o.player_id,
        ))
    nearest = min((float(np.hypot(*(o.pos - at))) for o in opponents), default=99.0)

    # Carrying on: the spot is his, discounted like any carry, and worth what
    # space he has there -- this is what makes the free man worth finding.
    best = (position_value(at, attacking_goal_x) * DRIBBLE_CONTINUATION_DISCOUNT
            * min(1.0, nearest / DRIBBLE_ROOM_RADIUS))
    # A shot from there.
    goal = np.array([attacking_goal_x, PITCH_WIDTH / 2.0])
    shot_p = route_survival(at, goal, SHOT_SPEED, SHOT_FRICTION_COEFF, opponents, is_shot=True)
    best = max(best, shot_p * shot_value(at, attacking_goal_x) * SHOT_DECISIVENESS)
    # A pass on, to any teammate (the passer included: the return ball).
    mates = [m for m in team.players if m.player_id != receiver.player_id and m.position != "GK"]
    if mates:
        pts = np.array([m.pos for m in mates])
        gaps = np.hypot(pts[:, 0] - at[0], pts[:, 1] - at[1])
        keep = gaps >= MIN_PASS_DIST
        if keep.any():
            pts = pts[keep]
            paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps[keep]])
            lanes = route_survival_many(at, pts, paces, BALL_FRICTION_COEFF, opponents)
            for pt, lane in zip(pts, lanes):
                v = (position_value(pt, attacking_goal_x) * PASS_CONTINUATION_DISCOUNT
                     * penetration_value(at, pt, opponents, attacking_goal_x))
                best = max(best, float(lane) * v)
    return float(best)


# Possession as a gradient over its whole range (POSSESSION_SCALE in constants).
# OFF: tried 2026-10-30 and rejected after watching -- behaviour regressed, even
# though the numbers moved smoothly. FB_POSSESSION_GRADUAL=1 turns it on to compare.
POSSESSION_GRADUAL = __import__("os").environ.get("FB_POSSESSION_GRADUAL", "0") != "0"


def _turnover_cost(cand: Candidate, player: Player, team) -> float:
    """What losing the ball on this move costs.

    Only above possession 1.0, (Possession - 1), and nothing at or below it, so
    the neutral setting is exactly the plain scoring. (Under POSSESSION_GRADUAL,
    off, every side pays it, scaled by Possession x POSSESSION_SCALE.) A pass or dribble is charged at its TARGET: lost passes are
    won at the receiving end (74-96% of them, measured), and a dribble is lost
    where it lands. A shot is charged too, at the shooter: a miss hands the
    ball over. Exempting shots made them the one free action -- at possession
    1.5-2.0 a side took 88-130 shots instead of 6 and lost the ball each time.
    """
    if POSSESSION_GRADUAL:
        from ..constants import POSSESSION_SCALE
        extra = possession_bias(player, team) * POSSESSION_SCALE
    else:
        extra = possession_bias(player, team) - 1.0
    if extra <= 0.0:
        return 0.0
    own_goal_x = 0.0 if team.direction == 1 else PITCH_LENGTH
    lost_at = player.pos if cand.kind == "shot" else cand.target
    danger = threat(lost_at, own_goal_x)
    return (1.0 - cand.p) * extra * (
        POSSESSION_LOSS_BASE + POSSESSION_LOSS_THREAT * danger
    )


def on_ball_decision(player: Player, state: GameState) -> OnBallAction:
    # A keeper who has just gathered the ball holds it while the box clears.
    # He is not under pressure -- opponents were moved out when he caught it --
    # so there is nothing to decide until he is ready to distribute.
    if state.gk_hold_ticks > 0 and player.position == "GK":
        return OnBallAction("hold", np.zeros(2), None, None)

    candidates = build_candidates(player, state)
    if player.first_touch_height > CONTROL_HEIGHT and player.position != "GK":
        # Headed on like any first-time ball, but along the ground only.
        candidates = [c for c in candidates if c.kind in ("pass", "shot") and c.vz == 0.0]
    _batch_lanes(candidates, player, state)
    if (
        PRUNE_RECEPTION and not LOOKAHEAD and player.position != "GK"
        and player.first_touch_speed <= 0.0
    ):
        if PASS_STAGES:
            return _action_from(_best_two_stage(candidates, player, state), player, state)
        return _action_from(_best_pruned(candidates, player, state), player, state)
    scored = [score_candidate(c, player, state) for c in candidates]
    if LOOKAHEAD:
        _add_lookahead(scored, player, state)

    # A ball that reached him too fast to simply trap: first touch first.
    if player.first_touch_speed > 0.0 and player.position != "GK":
        return _first_touch_decision(player, state, scored)

    # Distribution from his own hands is the one moment a keeper is never under
    # pressure, so there is no reason to gamble: a giveaway here is a shot on
    # his own goal. Weight safety far above value, rather than letting the
    # value model tempt him into a hopeful ball upfield.
    if player.position == "GK":
        passes = [c for c in scored if c.kind == "pass" and c.p >= GK_SAFE_PASS_FLOOR]
        if passes:
            best = max(passes, key=lambda c: c.p * (c.v ** GK_SAFETY_EXPONENT))
            direction = normalize(best.target - player.pos)
            if float(np.linalg.norm(direction)) >= 1e-9:
                if best.is_runway:
                    return OnBallAction(
                        "pass", direction * best.speed, best.target.copy(),
                        best.recipient_id, best.signal_runners, vz=best.vz,
                    )
                return OnBallAction("pass", direction * best.speed, None, None,
                                    receiver_id=best.recipient_id, vz=best.vz)

    best = _two_stage(scored)[0] if PASS_STAGES else max(scored, key=lambda c: c.utility)
    return _action_from(best, player, state)


# Skip the reception race for options that cannot win (_best_pruned). The
# chosen action is provably the same; a switch only to measure that.
PRUNE_RECEPTION = True

# Pass in two steps (2026-10-30). Whether to pass at all: the pass he would
# play -- and the cross he would play -- competes with every shot, dribble and
# hold at its utility WITHOUT the pass-kind tilts (Lofted, Direct, Pass range).
# Which pass (which cross): the tilted utility, within passes (within crosses). With the tilts inside the one argmax, the favoured kind's x2
# lifted the best pass over the dribble: with a flat tilt at 2.0 and 0.5 a side passed
# on 81-89% of decisions against 61-62%, about twice as often per minute.
# Identical choices whenever every tilt is 1.0. FB_PASS_STAGES=0: one argmax.
PASS_STAGES = __import__("os").environ.get("FB_PASS_STAGES", "1") != "0"


def _stage_group(c: Candidate) -> str | None:
    """Which tilted group an option is chosen within: passes, crosses, or none."""
    if c.kind != "pass":
        return None
    return "cross" if c.is_cross else "pass"


def _two_stage(scored: list[Candidate]) -> tuple[Candidate, float]:
    """(the chosen option, the utility it competed with). Within passes, and
    within crosses, the best by the tilted utility; then the best of those
    (at their plain utility) and every other option (at its utility); ties go
    to the option listed first."""
    group_best: dict[str, tuple[int, float]] = {}
    for i, c in enumerate(scored):
        g = _stage_group(c)
        if g is not None and (g not in group_best or c.utility > group_best[g][1]):
            group_best[g] = (i, c.utility)
    return _final_stage(scored, {i for i, _ in group_best.values()})


def _final_stage(options: list[Candidate], finalists: set[int]) -> tuple[Candidate, float]:
    best_i, best_u = -1, -float("inf")
    for i, c in enumerate(options):
        g = _stage_group(c)
        if g is not None and i not in finalists:
            continue
        u = c.plain if g is not None else c.utility
        if u > best_u:
            best_i, best_u = i, u
    return options[best_i], best_u


def _best_two_stage(candidates: list[Candidate], player: Player, state: GameState) -> Candidate:
    """_two_stage, without running the reception race for passes or crosses
    that cannot be their group's tilted best (the race only lowers a pass's
    tilted and plain utility alike, so _best_pruned's bound argument holds
    within each group)."""
    bounds = []
    for i, c in enumerate(candidates):
        racing = c.kind == "pass" and c.recipient_id is not None and c.reception is None
        if racing:
            c.reception = 1.0
        score_candidate(c, player, state)
        if racing:
            c.reception = None
        bounds.append((c.utility, i, racing))
    finalists = set()
    for g in ("pass", "cross"):
        group = sorted((b for b in bounds if _stage_group(candidates[b[1]]) == g), key=lambda b: (-b[0], b[1]))
        best_u, best_i = -float("inf"), -1
        for bound, i, racing in group:
            if bound < best_u:
                break
            c = candidates[i]
            u = score_candidate(c, player, state).utility if racing else bound
            if u > best_u or (u == best_u and i < best_i):
                best_u, best_i = u, i
        if best_i >= 0:
            # Its plain utility is exact: a racing winner was scored in full
            # above, and a pass with no race had its exact score as its bound.
            finalists.add(best_i)
    return _final_stage(candidates, finalists)[0]


def _batch_lanes(candidates: list[Candidate], player: Player, state: GameState) -> None:
    """Every ordinary ground pass's lane in one call (route_survival_many,
    pinned equal to route_survival): one at a time, ~100 passes x 11
    opponents, it was most of a decision's cost under inertia."""
    group = [c for c in candidates
             if c.kind == "pass" and c.vz == 0.0 and not c.is_cross and not c.is_clearance]
    if not group:
        return
    friction = group[0].friction
    if any(c.friction != friction for c in group):
        return
    last = _last_line(player, state)
    lanes = route_survival_many(
        player.pos, np.array([_lane_end(c, player, state, last) for c in group]),
        np.array([c.speed for c in group]), friction, state.opponents_of(player.team_id),
    )
    for c, lane in zip(group, lanes):
        c.lane = float(lane)


# A/B switch (on); 0 runs a through ball's lane all the way to its target.
RUNWAY_LANE = __import__("os").environ.get("FB_RUNWAY_LANE", "1") != "0"
# A/B switch (on); 0 values a through ball like any pass (penetration_value).
THROUGH_CHANCE = __import__("os").environ.get("FB_THROUGH_CHANCE", "1") != "0"


def _last_line(player: Player, state: GameState) -> float:
    """How far up the pitch, in metres from his own goal line, the deepest
    outfield opponent stands."""
    direction = state.get_team(player.team_id).direction
    return max(
        (float(o.pos[0]) if direction == 1 else PITCH_LENGTH - float(o.pos[0]))
        for o in state.opponents_of(player.team_id) if o.position != "GK"
    )


def _through_ball(cand: Candidate, player: Player, state: GameState, last: float | None = None) -> bool:
    """A runway ball landing THROUGH_BEYOND_M or more beyond every outfield
    opponent: the runner on the end of it is through on the keeper."""
    if not (RUNWAY_LANE and cand.is_runway and not cand.is_cross):
        return False
    direction = state.get_team(player.team_id).direction
    up = float(cand.target[0]) if direction == 1 else PITCH_LENGTH - float(cand.target[0])
    return up > (_last_line(player, state) if last is None else last) + THROUGH_BEYOND_M


def _through_chance(cand: Candidate, state: GameState, opponents: list[Player],
                    attacking_goal_x: float) -> float:
    """The shot a through ball sets up, on a shot's own terms (shot_value x
    SHOT_DECISIVENESS, past the keeper, inside the posts): from where the
    runner gets it, or further on as he carries it toward goal
    (THROUGH_SHOT_FROM_M, PACE_SPRINT_FROM), each spot taken by how free he
    still is there -- how much sooner he gets to it than the nearest outfield
    defender (CROSS_FREE_TAU, CROSS_FREE_FLOOR). A ball that puts him in
    alone at 14 m is a goal chance; one to the corner flag is not."""
    from ..constants import PACE_SPRINT_FROM, SHOT_AIM_INSIDE_M

    if cand.recipient_id is None:
        return 0.0
    runner = state.get_player(cand.recipient_id)
    goal = np.array([attacking_goal_x, PITCH_WIDTH / 2.0])
    to_goal = goal - cand.target
    dist = float(np.hypot(*to_goal))
    run = max(0.0, dist - THROUGH_SHOT_FROM_M)
    shares = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0) if run > 0.0 else (0.0,)
    spots = np.array([cand.target + to_goal / max(dist, 1e-9) * run * k for k in shares])
    t_runner = float(reach_ticks(runner.pos, runner.velocity, runner.speed,
                                 cand.target, control_radius(runner)))
    t_runner = t_runner + np.array(shares) * run / max(PACE_SPRINT_FROM * runner.speed, 1e-9)
    field = [o for o in opponents if o.position != "GK"]
    keeper = [o for o in opponents if o.position == "GK"]
    if field:
        P = np.array([o.pos for o in field])
        V = np.array([o.velocity for o in field])
        tops = np.array([o.speed for o in field])
        ctrl = np.array([control_radius(o) for o in field])
        t_def = np.min(reach_ticks(P[None], V[None], tops[None], spots[:, None, :], ctrl[None]), axis=1)
    else:
        t_def = np.full(len(spots), math.inf)
    sigma = _strike_sigma(runner)
    best = 0.0
    for spot, mine, theirs in zip(spots, t_runner, t_def):
        spare = float(theirs) - float(mine)
        if spare < 0.0 and best > 0.0:
            break       # caught on the way: he shot before here
        free = CROSS_FREE_FLOOR + (1.0 - CROSS_FREE_FLOOR) * (1.0 - math.exp(-max(0.0, spare) / CROSS_FREE_TAU))
        shot = 0.0
        for y in (GOAL_TOP_Y + SHOT_AIM_INSIDE_M, GOAL_BOTTOM_Y - SHOT_AIM_INSIDE_M):
            aim = np.array([attacking_goal_x, y])
            shot = max(shot, route_survival(spot, aim, SHOT_SPEED, SHOT_FRICTION_COEFF, keeper, is_shot=True)
                       * _inside_posts(spot, aim, sigma))
        best = max(best, shot_value(spot, attacking_goal_x) * SHOT_DECISIVENESS * shot * free)
    return best


def _lane_end(cand: Candidate, player: Player, state: GameState, last: float | None = None) -> np.ndarray:
    """Where a pass's lane ends: its target, but for a through ball
    (_through_ball) the start of the reception race (RECEPTION_FROM_FRACTION),
    as for a cross. Run to the target, the defenders racing the runner were
    counted twice -- in the lane and in the race: forced through balls scored
    P 0.07 on average and reached the runner 59% of the time; with the lane
    stopped there, 0.30, and in the right order (0.17 / 0.40 / 0.69 -> 41 /
    79 / 80%; scratch/forced_wing_through.py). Every runway ball stopped there
    took crosses from 51 to 6 in 16 matches: a ball from the flank into the
    box scored as a pass beat the same ball as a cross."""
    if _through_ball(cand, player, state, last):
        return player.pos + (cand.target - player.pos) * RECEPTION_FROM_FRACTION
    return cand.target


def _first_touch_decision(
    player: Player, state: GameState, scored: list[Candidate]
) -> OnBallAction:
    """A fast ball has arrived: play it first time, or try to control it.

    No dribble and no hold -- he does not have it under control yet. A
    first-time pass or shot keeps its P, scaled by how well a ball this quick
    can be played on (first_time_quality). Controlling it is worth having the
    ball here, like a completed pass to this spot, with the chance of a clean
    trap; a failed one squirms loose (game_loop applies it).
    """
    from .control import first_time_quality, trap_probability

    team = state.get_team(player.team_id)
    header = player.first_touch_height > CONTROL_HEIGHT
    quality = first_time_quality(player.first_touch_speed, player)
    options = [c for c in scored if c.kind in ("pass", "shot")]
    for c in options:
        c.p *= quality
        _set_utility(c, player, team)
    if header:
        # Nothing to trap: it is headed on, wherever it goes -- in his own
        # half possibly out of play (see CLEAR_* in constants).
        options += _clearing_headers(player, state, team, options)
        best = max(options, key=lambda c: c.utility, default=None)
        if best is None:
            return OnBallAction("control", np.zeros(2), None, None)
        return _action_from(best, player, state)
    if PASS_STAGES and options:
        best, best_u = _two_stage(options)
    else:
        best = max(options, key=lambda c: c.utility, default=None)
        best_u = best.utility if best is not None else -float("inf")
    attacking_goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    trap = Candidate(
        kind="pass", target=player.pos.copy(), speed=0.0, friction=BALL_FRICTION_COEFF,
    )
    trap.p = trap_probability(player)
    trap.v = position_value(player.pos, attacking_goal_x) * PASS_CONTINUATION_DISCOUNT
    trap.context = 1.0
    _set_utility(trap, player, team, risky=False)
    if best is None or trap.utility >= best_u:
        return OnBallAction("control", np.zeros(2), None, None)
    return _action_from(best, player, state)


def _clearing_headers(
    player: Player, state: GameState, team, options: list[Candidate]
) -> list[Candidate]:
    """A defender's header in his own half: head it on, or put it out.

    Charges every header option for where the ball ends up (CLEAR_DANGER): a
    header that is lost costs the danger of their ball where he stands, and
    a clearance costs a share of the danger of their restart. Adds the two
    clearances -- into touch a little upfield, and behind for a corner when
    he is close to his own goal line -- scored on that cost alone. Nothing
    changes for a header in the opponents' half.
    """
    from ..constants import (
        CLEAR_CORNER_RANGE,
        CLEAR_DANGER,
        CLEAR_RESTART_SHARE,
        PASS_SPEED_MAX,
    )

    own_goal_x = 0.0 if team.direction == 1 else PITCH_LENGTH
    own_up = abs(float(player.pos[0]) - own_goal_x)
    if own_up > PITCH_LENGTH / 2.0:
        return []
    here = CLEAR_DANGER * threat(player.pos, own_goal_x)
    for c in options:
        c.utility -= (1.0 - c.p) * here

    x, y = float(player.pos[0]), float(player.pos[1])
    side = -1.0 if y < PITCH_WIDTH / 2.0 else 1.0
    line_y = 0.0 if side < 0 else PITCH_WIDTH
    upfield = float(team.direction)
    outs = [(np.array([x + 10.0 * upfield, line_y + 2.0 * side]),
             np.array([x + 10.0 * upfield, line_y]))]
    if own_up <= CLEAR_CORNER_RANGE:
        outs.append((np.array([own_goal_x - 2.0 * upfield, PITCH_WIDTH / 2.0 + 18.0 * side]),
                     np.array([own_goal_x, line_y])))
    opponents = state.opponents_of(player.team_id)
    out = []
    for target, restart in outs:
        c = Candidate(kind="pass", target=target, speed=PASS_SPEED_MAX,
                      friction=BALL_FRICTION_COEFF, is_clearance=True)
        c.p = route_survival(player.pos, target, c.speed, c.friction, opponents)
        c.utility = -(c.p * CLEAR_RESTART_SHARE * CLEAR_DANGER * threat(restart, own_goal_x)
                      + (1.0 - c.p) * here)
        out.append(c)
    return out


def _action_from(best: Candidate, player: Player, state: GameState) -> OnBallAction:
    """The action that plays the chosen candidate."""
    # Holding is judged on the same scale as everything else, so there is no
    # absolute floor to clear. An absolute floor deadlocks a defender pinned in
    # his own corner: every option there is legitimately worth very little, and
    # a fixed threshold would reject them all and freeze him on the ball
    # forever. The only question that matters is whether acting beats waiting.
    if best.kind == "hold":
        return OnBallAction(
            "hold", np.zeros(2), None, None, move=_shield_step(player, state)
        )

    direction = normalize(best.target - player.pos)
    if float(np.linalg.norm(direction)) < 1e-9:
        return OnBallAction("hold", np.zeros(2), None, None)
    if (kinematics.GRIP and not _pace.PACE and best.kind == "dribble"
            and _may_idle(best, player, state)):
        # Nobody to pass to, nobody near, nowhere much to go: walk it, ball at
        # his feet, while the others move -- the same direction, at a walk.
        return OnBallAction("hold", np.zeros(2), None, None,
                            move=direction * player.speed * IDLE_PACE_SHARE)

    velocity = direction * best.speed
    if best.curl != 0.0:
        velocity = turned(velocity, bend_start(best.curl, distance(best.target, player.pos) / best.speed))
    if best.kind == "pass" and best.is_runway:
        return OnBallAction(
            "pass", velocity, best.target.copy(), best.recipient_id,
            best.signal_runners, receiver_id=best.recipient_id, vz=best.vz,
            is_cross=best.is_cross,
        )
    return OnBallAction(best.kind, velocity, None, None,
                        receiver_id=best.recipient_id if best.kind == "pass" else None,
                        pass_point=best.target.copy() if best.kind == "pass" else None,
                        touch=distance(best.target, player.pos) if best.kind == "dribble" else None,
                        vz=best.vz, curl=best.curl, is_cross=best.is_cross)


def _finisher_freedom(cand: Candidate, state: GameState, opponents: list[Player]) -> float:
    """How free the man on the end of a cross is: how much sooner he gets to
    the spot than the nearest defender (CROSS_FREE_TAU, CROSS_FREE_FLOOR).
    Whether a defender cuts the ball out on the way is the reception race's
    business; this is the time he has once it is his -- a header a defender
    meets at the same moment is rarely a goal, one alone at the far post is."""
    if cand.recipient_id is None or not opponents:
        return 1.0
    receiver = state.get_player(cand.recipient_id)
    t_recv = float(reach_ticks(receiver.pos, receiver.velocity, receiver.speed,
                               cand.target, control_radius(receiver)))
    P = np.array([o.pos for o in opponents])
    V = np.array([o.velocity for o in opponents])
    tops = np.array([o.speed for o in opponents])
    ctrl = np.array([control_radius(o) for o in opponents])
    t_def = float(np.min(reach_ticks(P, V, tops, cand.target[None, :], ctrl)))
    spare = max(0.0, t_def - t_recv)
    return CROSS_FREE_FLOOR + (1.0 - CROSS_FREE_FLOOR) * (1.0 - math.exp(-spare / CROSS_FREE_TAU))


def _may_idle(best: Candidate, player: Player, state: GameState) -> bool:
    """He may slow down and wait: his best option is a dribble that gains
    little, no opponent can reach him soon, and he has not idled too long.
    A dribble that gains ground -- a break into space -- is never slowed."""
    from .value import position_value

    if player.position == "GK" or player.hold_ticks >= IDLE_MAX_TICKS:
        return False
    team = state.get_team(player.team_id)
    goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    here = position_value(player.pos, goal_x)
    if position_value(best.target, goal_x) > here + IDLE_GAIN_MAX * max(here, 1e-6):
        return False
    reach = control_radius(player)
    for opp in state.opponents_of(player.team_id):
        if distance(opp.pos, player.pos) < IDLE_MIN_GAP:
            return False
        if float(reach_ticks(opp.pos, opp.velocity, opp.speed, player.pos, reach)) < IDLE_FREE_TICKS:
            return False
    return True


_RECEPTION_TICKS = np.arange(1.0, RECEPTION_HORIZON_TICKS + 1.0, 2.0)


def _shield_step(player: Player, state: GameState) -> np.ndarray:
    """The carrier's step while holding: shield, do not freeze.

    A shielding player turns his back to the presser and drifts away from him,
    keeping the ball alive while he waits for a pass to open. Standing dead
    still looked like freezing and gave the presser a stationary target.

    Away from every opponent close enough to contest, weighted by closeness,
    and steered in off the touchline so shielding never walks him out of play.
    Recomputed each tick, so he turns as the presser circles him.
    """
    push = np.zeros(2)
    for o in state.opponents_of(player.team_id):
        d = player.pos - o.pos
        n = float(np.linalg.norm(d))
        if 1e-6 < n <= SHIELD_AWARENESS:
            push += d / (n * n)
    # Keep off the touchlines and goal lines.
    x, y = float(player.pos[0]), float(player.pos[1])
    edge = SHIELD_EDGE_MARGIN
    if y < edge:
        push[1] += (edge - y) / edge
    elif y > PITCH_WIDTH - edge:
        push[1] -= (y - (PITCH_WIDTH - edge)) / edge
    if x < edge:
        push[0] += (edge - x) / edge
    elif x > PITCH_LENGTH - edge:
        push[0] -= (x - (PITCH_LENGTH - edge)) / edge
    n = float(np.linalg.norm(push))
    if n < 1e-9:
        return np.zeros(2)
    return push / n * player.speed * SHIELD_SPEED_FRACTION


def _reception_probability(
    cand: Candidate,
    passer: Player,
    state: GameState,
    opponents: list[Player],
) -> float:
    """P(the intended receiver, not an opponent, is first to the MOVING ball).

    The old version raced receiver and opponents to the target POINT and never
    looked at the ball: it computed the ball's arrival time and then ignored
    it. So pace had no effect on whether the passer expected his man to collect
    it -- a ball struck far too hard for a late receiver scored exactly like a
    well-weighted one, and 21% of passes lost under pressure were lost PAST the
    target, run through to whoever stood behind.

    Now the ball is followed along its real path, past the target too, until it
    stops or leaves the pitch, and each player's earliest reachable moment on
    that path is found. Too fast, and the receiver's first touch slides further
    down the line, where an opponent may already be. Too slow, and an opponent
    steps in front. The pace the argmax prefers is the one that wins this race.

    The early part of the path is left to route_survival (the lane); this is the
    contest for the ball as it arrives and runs on.
    """
    if cand.recipient_id is None:
        return 1.0
    try:
        receiver = state.get_player(cand.recipient_id)
    except KeyError:
        return 0.0

    origin = passer.pos
    offset = cand.target - origin
    dist = float(np.linalg.norm(offset))
    if dist < 1e-9 or cand.speed <= 1e-9:
        return 0.0
    direction = offset / dist

    # Ball position at each sampled tick under constant deceleration; it stops
    # (and stays) once its speed is spent.
    t = _RECEPTION_TICKS
    reach_out = reach_gk = None
    if cand.vz > 0.0:
        # Lofted: the real flight, landing and bounces. Where it is still
        # above a player's reach he cannot meet it there.
        path, hs = trajectory(origin, direction * cand.speed, 0.0, cand.vz,
                              cand.friction, int(t[-1]))
        idx = t.astype(int) - 1
        pts = path[idx]
        reach_out = within_reach(0.0, hs, HEADER_HEIGHT)[idx]
        reach_gk = (hs <= GK_REACH_HEIGHT)[idx]
        along = (pts - origin) @ direction
    else:
        stop = cand.speed / max(cand.friction, 1e-12)
        tt = np.minimum(t, stop)
        along = cand.speed * tt - cand.friction * tt * (tt + 1.0) / 2.0
        pts = origin + np.outer(along, direction)

    # Only the arrival and run-on: the flight before it is route_survival's.
    on_pitch = (
        (pts[:, 0] >= 0.0) & (pts[:, 0] <= PITCH_LENGTH)
        & (pts[:, 1] >= 0.0) & (pts[:, 1] <= PITCH_WIDTH)
    )
    mask = (along >= RECEPTION_FROM_FRACTION * dist) & on_pitch
    if not mask.any():
        return 0.0
    t, pts = t[mask], pts[mask]
    if reach_out is not None:
        reach_out, reach_gk = reach_out[mask], reach_gk[mask]

    def earliest(player: Player) -> float:
        # Reach is in METRES, not a grace in ticks. The old 6-tick grace let a
        # receiver arrive after the ball and still "collect" it, which was true
        # of the feeble passes it was tuned for; at realistic pace a ball covers
        # ~3 m in 6 ticks and is gone -- that fiction is why passes kept being
        # lost PAST the receiver. A player can stretch to a ball passing within
        # his control radius, and no further. Same rule for both sides.
        if kinematics.INERTIA:
            # From his actual motion: a receiver standing still needs time to
            # get going, one already running the wrong way must stop first.
            need = reach_ticks(player.pos, player.velocity, player.speed, pts,
                               control_radius(player))
        else:
            gap = np.hypot(pts[:, 0] - player.pos[0], pts[:, 1] - player.pos[1])
            need = np.maximum(0.0, gap - control_radius(player)) / max(player.speed, 1e-9)
        ok = need <= t
        if reach_out is not None:
            ok &= reach_gk if player.position == "GK" else reach_out
        return float(t[ok.argmax()]) if ok.any() else float("inf")

    t_receiver = earliest(receiver)
    if t_receiver == float("inf"):
        return 0.0   # he never gets to it before it stops or goes out
    if kinematics.INERTIA and opponents:
        # The same race for every opponent in one call: one at a time it was
        # the cost of a pass decision under inertia (~12k calls / 300 ticks).
        P = np.array([o.pos for o in opponents])
        V = np.array([o.velocity for o in opponents])
        tops = np.array([o.speed for o in opponents])
        ctrl = np.array([control_radius(o) for o in opponents])
        need = reach_ticks(P[None, :, :], V[None, :, :], tops[None, :], pts[:, None, :], ctrl[None, :])
        ok = need <= t[:, None]                                   # (points, opponents)
        if reach_out is not None:
            is_gk = np.array([o.position == "GK" for o in opponents])
            ok &= np.where(is_gk[None, :], reach_gk[:, None], reach_out[:, None])
        hit = ok.any(axis=0)
        first = np.where(hit, t[ok.argmax(axis=0)], np.inf)
        t_opponent = float(first.min())
    else:
        t_opponent = min((earliest(o) for o in opponents), default=float("inf"))
    if t_opponent == float("inf"):
        return 1.0
    return 1.0 / (1.0 + math.exp(-(t_opponent - t_receiver) / RECEPTION_TAU))


def _in_play_probability(cand: Candidate, passer: Player) -> float:
    """P(the ball is still on the pitch when it gets there).

    The ball does not stop at its target: it carries on until friction kills
    it. So the honest test is whether the *overrun* stays inside the lines,
    not whether the aim point does.
    """
    from ..ball_physics import carry_distance

    direction = cand.target - passer.pos
    length = float(np.hypot(direction[0], direction[1]))
    if length < 1e-9:
        return 1.0
    unit = direction / length

    # Where the ball ends up if nobody touches it. A lofted ball lands at its
    # target and bounces on from there.
    carry = float("inf") if cand.vz > 0.0 else carry_distance(cand.speed, cand.friction)
    overrun = min(carry, length + BALL_OVERRUN_DIST)
    end = passer.pos + unit * overrun

    margin = min(
        float(end[0]), PITCH_LENGTH - float(end[0]),
        float(end[1]), PITCH_WIDTH - float(end[1]),
    )
    if margin >= TOUCHLINE_MARGIN:
        return 1.0
    if margin <= 0.0:
        return OUT_OF_PLAY_FLOOR
    # Ramp: the closer the resting point to the line, the likelier it runs out.
    return OUT_OF_PLAY_FLOOR + (1.0 - OUT_OF_PLAY_FLOOR) * (margin / TOUCHLINE_MARGIN)


def _dribble_room(cand: Candidate, player: Player, opponents: list[Player]) -> float:
    """How much room a dribble carries the ball into.

    Looks past the touch itself: a 3 m knock into a gap that closes immediately
    is worthless, so this measures space at the destination AND a little beyond
    it, along the direction of travel.
    """
    direction = cand.target - player.pos
    length = float(np.hypot(direction[0], direction[1]))
    if length < 1e-9:
        return 1.0
    unit = direction / length

    room = 1.0
    for reach in (1.0, DRIBBLE_LOOKAHEAD_FACTOR):
        probe = player.pos + unit * (length * reach)
        if _pace.PACE and MOTION_GAPS:
            gaps = _motion_gaps(opponents, probe)
            nearest = float(gaps.min()) if gaps.size else DRIBBLE_ROOM_RADIUS
        else:
            nearest = min(
                (distance(probe, o.pos) for o in opponents),
                default=DRIBBLE_ROOM_RADIUS,
            )
        room = min(room, min(1.0, nearest / DRIBBLE_ROOM_RADIUS))
    return max(DRIBBLE_ROOM_FLOOR, room)


# Under pace: room and contest read each opponent's motion (_motion_gaps).
# A switch only to measure it against the plain distance.
MOTION_GAPS = True


def _motion_gaps(opponents: list[Player], point: np.ndarray) -> np.ndarray:
    """Each opponent's distance to `point` as his MOTION makes it:
    distance x (ticks he needs from how he is running / ticks he would need
    from a standstill), both from reach_ticks with grip. Standing, it is the
    plain distance; running at the spot, less; sprinting past or away, far
    more, because he must brake and turn first. Without it a defender who had
    just overrun the carrier counted as fully in the way, so nobody ever cut
    back against a man at full speed."""
    if not opponents:
        return np.zeros(0)
    P = np.array([o.pos for o in opponents]); V = np.array([o.velocity for o in opponents])
    S = np.array([o.speed for o in opponents])
    point = point[None, :]
    moving = np.asarray(reach_ticks(P, V, S, point), dtype=float)
    still = np.asarray(reach_ticks(P, np.zeros_like(V), S, point), dtype=float)
    dist = np.hypot(P[:, 0] - point[0, 0], P[:, 1] - point[0, 1])
    return dist * moving / np.maximum(still, 1e-9)


def _dribble_contest(
    cand: Candidate,
    player: Player,
    opponents: list[Player],
) -> float:
    """P(the carry survives contact), from the number of realistic challengers.

    A dribble is a 3 m touch, so route_survival hardly registers anyone. What
    actually decides a carry is how many opponents are close enough to step in
    -- one is a duel, three is a turnover.
    """
    challengers = 0.0
    outfield = [o for o in opponents if o.position != "GK"]
    gaps = _motion_gaps(outfield, cand.target) if _pace.PACE and MOTION_GAPS else None
    for i, opp in enumerate(outfield):
        gap = float(gaps[i]) if gaps is not None else distance(opp.pos, cand.target)
        if gap < DRIBBLE_CONTEST_RADIUS:
            # Closer opponents count for more; the nearest is the real threat.
            challengers += 1.0 - gap / DRIBBLE_CONTEST_RADIUS
    return float(max(DRIBBLE_CONTEST_FLOOR, 1.0 / (1.0 + DRIBBLE_CONTEST_WEIGHT * challengers)))


_ANGLE_HERE: dict = {}


def _angle_gain(
    cand: Candidate,
    player: Player,
    state: GameState,
    opponents: list[Player],
) -> float:
    """How much better the passing options become from the dribble's target.

    Returns >= 1.0. A carry that opens a lane is worth more than the three
    metres of ground it gains, which is the whole reason to dribble sideways
    when the direct ball is blocked.
    """
    from ..constants import BALL_FRICTION_COEFF
    from ..ball_physics import nominal_pass_speed

    mates = [
        m for m in state.teammates_of(player.player_id)
        if m.position != "GK" and distance(m.pos, player.pos) >= MIN_PASS_DIST
    ]
    if not mates:
        return 1.0

    pts = np.array([m.pos for m in mates])

    def best_lane(origin: np.ndarray) -> float:
        # All the lanes at once (route_survival_many, pinned equal to the
        # one-at-a-time form): this ran ~16 times a decision, lane by lane.
        gaps = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
        paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps])
        return float(np.max(route_survival_many(origin, pts, paces, BALL_FRICTION_COEFF, opponents)))

    # Where he stands is the same for every dribble option: once per decision.
    key = (id(state), state.tick, player.player_id, float(player.pos[0]), float(player.pos[1]),
           float(sum(o.pos[0] + 3.0 * o.pos[1] for o in opponents)))
    if _ANGLE_HERE.get("key") != key:
        _ANGLE_HERE.update(key=key, here=best_lane(player.pos))
    here = _ANGLE_HERE["here"]
    # Only reward opening a lane the carrier does not already have. Rewarding
    # any improvement made 70% of dribbles eligible at a mean of 1.41x, which
    # is not a discriminator -- it just inflated every dribble against passing
    # and shooting until shots stopped happening altogether.
    if here >= ANGLE_GAIN_NEED_BELOW:
        return 1.0
    there = best_lane(cand.target)
    gain = max(0.0, there - here)
    return float(min(ANGLE_GAIN_MAX, 1.0 + ANGLE_GAIN_WEIGHT * gain))
