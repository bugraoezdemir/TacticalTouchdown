"""
Enumerate the actions a ball carrier could take.

Separating enumeration from scoring is what lets on_ball.py be an argmax rather
than a cascade of thresholds: every option is built the same way, scored on the
same scale, and compared directly.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..constants import (
    BALL_FRICTION_COEFF,
    DRIBBLE_SPEED,
    RUNWAY_LATE_TOLERANCE,
    GOAL_BOTTOM_Y,
    GOAL_CENTER_Y,
    GOAL_TOP_Y,
    MIN_PASS_DIST,
    PASS_CHASE_TICKS,
    PASS_OVERRUN_MAX,
    LANE_SAFE_ENOUGH,
    PASS_DRIVEN_OVERRUN,
    PASS_OVERRUN_MIN,
    PASS_SPEED_HIGH,
    PASS_SPEED_LOW,
    PASS_PACE_FACTORS,
    PASS_SPEED_MAX,
    PASS_SPEED_MIN,
    PASS_TICKS_DRIVEN,
    PASS_TICKS_PER_M,
    PASS_TICKS_BASE,
    RUNWAY_DIST,
    RUNWAY_MAX_LEAD,
    PASS_RECEIVER_FREE_REACH,
    RUNWAY_ARRIVE_MARGIN,
    RUNWAY_CROSS_MARGIN,
    RUNWAY_MIN_LEAD,
    SHOT_SPEED,
    DRIBBLE_LOOKAHEAD_DIST,
    HOLD_MAX_TICKS,
    DRIBBLE_TOUCH_MODEL,
    DRIBBLE_DIRECTIONS,
    DRIBBLE_MIN_TOUCH,
    DRIBBLE_MAX_TOUCH_POOR,
    DRIBBLE_MAX_TOUCH_GREAT,
    DRIBBLE_TOUCH_SAMPLES,
    DRIBBLE_RACE_TAU,
    DRIBBLE_NUDGE,
    REACTION_TICKS,
    REACTION_RAMP_DISTANCE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TOUCHLINE_MARGIN,
    DRIBBLE_TOUCH_INTERVAL,
    OUTFIELD_CONTROL_RADIUS,
    DRIBBLE_PACE_POOR,
    DRIBBLE_PACE_GREAT,
    BALL_GRAVITY,
    CROSS_FLOAT_DEG,
    CROSS_FROM,
    CROSS_WHIP_DEG,
    CROSS_WIDE,
    LOFT_ANGLES_DEG,
    LOFT_MIN_DIST,
    LOFT_RUNWAY_MAX_DEG,
    LOFT_RUNWAY_MIN_DEG,
    LOFT_SPEED_MAX,
)
from .. import kinematics
from ..ball_flight import loft_launch
from ..kinematics import reach_distance, reach_ticks
from ..offside import is_offside_position
from ..ball_physics import nominal_pass_speed, speed_for_arrival, travel_ticks, weighted_speed
from ..constants import LEAD_MAX_TICKS, LEAD_MIN_SPEED_SHARE, PITCH_LENGTH, PITCH_WIDTH
from ..geometry import normalize
from ..models import GameState, Player
from .control import comfortable_strike
from .playmaking import find_signal_target

from . import pace as _pace

ActionKind = Literal["shot", "pass", "dribble", "hold"]

# A/B switch (on); 0 restores the race to his own touch that ignores how he is
# moving when he plays it.
OWN_MOMENTUM = os.environ.get("FB_OWN_MOMENTUM", "1") != "0"

# 8 candidate dribble directions at 45 degree increments.
_DRIBBLE_ANGLES = [i * math.pi / 4 for i in range(8)]
_DRIBBLE_ANGLES_16 = [i * 2.0 * math.pi / DRIBBLE_DIRECTIONS for i in range(DRIBBLE_DIRECTIONS)]


@dataclass(slots=True)
class Candidate:
    kind: ActionKind
    target: np.ndarray        # where the ball is being sent
    speed: float              # ball speed to impart
    friction: float           # which friction regime applies in flight
    recipient_id: int | None = None   # teammate this is aimed at, if any
    is_runway: bool = False           # true for a lead ball into space
    is_called_run: bool = False       # spot chosen first, runners summoned to it
    # Teammates to send onto this ball. Set for a called run, where the carrier
    # picks the spot first and tells people to go there, rather than only
    # exploiting runs that already happen to be underway.
    signal_runners: tuple[int, ...] = ()
    # Launch speed upward (m/tick) for a lofted ball; `speed` is then its
    # horizontal pace in the air. 0 for a ball along the ground.
    vz: float = 0.0
    is_cross: bool = False            # into a box role, to be finished first time
    is_clearance: bool = False        # put out of play on purpose
    # The lane's survival, when it was worked out for many passes at once
    # (on_ball_decision); None and the scorer works it out itself.
    lane: float | None = None
    # The reception race's result, when already known; None and the scorer
    # runs it. on_ball._best_pruned sets 1.0 to score an upper bound.
    reception: float | None = None
    # A paced dribble (decisions.pace): its band ("walk", "run", "sprint"),
    # its pace as a share of his top pace with the ball, and P(he wins the
    # race to his own ball) from the touch solver. None / 0 otherwise.
    pace: str | None = None
    pace_share: float = 0.0
    race: float | None = None
    # The pass-kind tilts in its C (Lofted, Direct, Pass range), and its
    # utility without them (on_ball.PASS_STAGES).
    flavour: float = 1.0
    plain: float = 0.0
    # Filled in by the scorer.
    p: float = 0.0
    v: float = 0.0
    context: float = 1.0
    utility: float = 0.0


def build_candidates(player: Player, state: GameState) -> list[Candidate]:
    from ..constants import BALL_FRICTION_COEFF, SHOT_FRICTION_COEFF, DRIBBLE_FRICTION_COEFF

    team = state.get_team(player.team_id)
    teammates = state.teammates_of(player.player_id)
    opponents = state.opponents_of(player.team_id)
    attacking_goal_x = 105.0 if team.direction == 1 else 0.0

    out: list[Candidate] = []

    # ── Shots: both corners plus the centre of the goal ──────────────────────
    for y in (GOAL_TOP_Y + 0.5, GOAL_CENTER_Y, GOAL_BOTTOM_Y - 0.5):
        out.append(Candidate(
            kind="shot", target=np.array([attacking_goal_x, y]),
            speed=SHOT_SPEED, friction=SHOT_FRICTION_COEFF,
        ))

    # ── Passes: direct and runway, each at both speeds ───────────────────────
    # Offering both speeds matters now that PASS_SPEED_LOW genuinely cannot
    # carry 40 m -- the argmax picks the weight instead of a hard-coded rule.
    team = state.get_team(player.team_id)
    for mate in teammates:
        # A teammate standing offside is not an option. Offside is judged where
        # he is when the ball is PLAYED, so this is decided now, before any
        # speed or lead is chosen: playing to him is a free kick to them. The
        # carrier never checked, and 21% of passes at high depth went to a
        # receiver who was offside as the ball left the foot.
        #
        # A runway ball INTO the space beyond the line is still legal -- he only
        # has to be onside when it is struck -- so that option survives for any
        # mate who is onside now. That is exactly how a striker beats a line.
        if is_offside_position(mate, state, team, ball_x=float(player.pos[0])):
            continue
        dist = float(np.linalg.norm(mate.pos - player.pos))
        if dist < MIN_PASS_DIST:
            # Too close to be a pass rather than a touch. Filtering these out
            # rather than scoring them is what stopped a dribbling player from
            # ever reconsidering: mid-carry his nearest options vanish.
            continue
        # Weight the ball to arrive rather than striking every pass at the same
        # speed: a fixed-speed 10 m ball carried 32 m, ran straight through the
        # receiver and was collected by whoever stood behind them.
        #
        # The overrun is what the RECEIVER can chase down, not a constant. A
        # fixed 20 m overrun kept passes fast but WAS the miss distance -- 74 of
        # 94 lost balls overshot by more than 8 m for exactly this reason.
        # Offered at SEVERAL paces, not one. _weight_for_lane's rule is a good
        # starting point but it cannot see who stands beyond the receiver; the
        # reception race in the scorer can, so let it pick the pace -- softer
        # when an opponent waits behind him, firmer when one closes the lane.
        # Plus the judged pace: as firm as he can still take cleanly.
        base = _weight_for_lane(player.pos, mate.pos, mate, opponents)
        strikes = [base * f for f in PASS_PACE_FACTORS]
        strikes.append(comfortable_strike(dist, mate, BALL_FRICTION_COEFF))
        paces = {round(min(PASS_SPEED_MAX, max(PASS_SPEED_MIN, s)), 4) for s in strikes}
        for pace in sorted(paces):
            out.append(Candidate(
                kind="pass", target=_lead_point(player.pos, mate, pace), speed=pace,
                friction=BALL_FRICTION_COEFF, recipient_id=mate.player_id,
            ))
        # Lofted to him, over whoever is in between: a low and a high ball.
        if dist >= LOFT_MIN_DIST:
            for angle in LOFT_ANGLES_DEG:
                spot, launch = _lofted_lead(player.pos, mate, angle)
                if launch is not None:
                    out.append(Candidate(
                        kind="pass", target=spot, speed=launch[0],
                        friction=BALL_FRICTION_COEFF, recipient_id=mate.player_id,
                        vz=launch[1],
                    ))
        # Runway ball into the space ahead of a moving teammate.
        #
        # The lead distance is NOT fixed: it is whatever the receiver can
        # actually reach before the ball does. A constant 8 m lead ignored the
        # receiver entirely, so most runway balls arrived at a point their
        # target was still 5-16 m away from -- they were the single biggest
        # source of passes handed straight to the opposition.
        mate_speed = float(np.linalg.norm(mate.velocity))
        if mate_speed > 1e-9:
            for lead_dist in _runway_leads(player, mate, mate_speed):
                lead = mate.pos + normalize(mate.velocity) * lead_dist
                lead_dist_from_passer = float(np.linalg.norm(lead - player.pos))
                out.append(Candidate(
                    kind="pass", target=lead,
                    speed=_runway_weight(player.pos, lead, mate),
                    friction=BALL_FRICTION_COEFF, recipient_id=mate.player_id,
                    is_runway=True,
                ))
                # ...or over the top: dropped onto the spot as he arrives.
                lofted = _lofted_runway(player.pos, lead, mate)
                if lofted is not None:
                    out.append(Candidate(
                        kind="pass", target=lead, speed=lofted[0],
                        friction=BALL_FRICTION_COEFF, recipient_id=mate.player_id,
                        is_runway=True, vz=lofted[1],
                    ))

    out.extend(_crosses(player, state, team))

    # ── Dribbles ─────────────────────────────────────────────────────────────
    if DRIBBLE_TOUCH_MODEL and _pace.PACE and player.position != "GK":
        # 16 directions x a walk, a run and a sprint, each band with its own
        # solved touch; the scorer picks the pace (decisions.pace).
        for angle in _DRIBBLE_ANGLES_16:
            d = np.array([math.cos(angle), math.sin(angle)])
            for band, target, share, p_win in _solve_sprint(player, d, opponents, attacking_goal_x):
                touch = float(np.linalg.norm(target - player.pos))
                roll = max(0.0, touch - dribble_nudge(touch))
                ticks = _touch_ticks(player, touch)
                out.append(Candidate(
                    kind="dribble", target=target,
                    speed=speed_for_arrival(DRIBBLE_FRICTION_COEFF, roll, ticks) if roll > 1e-6 else 0.0,
                    friction=DRIBBLE_FRICTION_COEFF,
                    pace=band, pace_share=share, race=p_win,
                ))
        for band, target, share, p_win in _paced_slow(player, opponents):
            touch = float(np.linalg.norm(target - player.pos))
            roll = max(0.0, touch - dribble_nudge(touch))
            out.append(Candidate(
                kind="dribble", target=target,
                speed=speed_for_arrival(DRIBBLE_FRICTION_COEFF, roll, _touch_ticks(player, touch))
                if roll > 1e-6 else 0.0,
                friction=DRIBBLE_FRICTION_COEFF,
                pace=band, pace_share=share, race=p_win,
            ))
    elif DRIBBLE_TOUCH_MODEL:
        # 16 directions, each with its touch length SOLVED from the context.
        for angle in _DRIBBLE_ANGLES_16:
            d = np.array([math.cos(angle), math.sin(angle)])
            touch = _solve_touch(player, d, opponents, attacking_goal_x)
            if touch is None:
                continue
            roll = max(0.0, touch - DRIBBLE_NUDGE)
            ticks = _touch_ticks(player, touch)
            out.append(Candidate(
                kind="dribble", target=player.pos + d * touch,
                # launched to reach the meeting point AS HE DOES, still rolling
                speed=speed_for_arrival(DRIBBLE_FRICTION_COEFF, roll, ticks) if roll > 1e-6 else 0.0,
                friction=DRIBBLE_FRICTION_COEFF,
            ))
    else:
        for angle in _DRIBBLE_ANGLES:
            d = np.array([math.cos(angle), math.sin(angle)])
            out.append(Candidate(
                kind="dribble", target=player.pos + d * DRIBBLE_LOOKAHEAD_DIST,
                speed=DRIBBLE_SPEED, friction=DRIBBLE_FRICTION_COEFF,
            ))

    # ── Called run: pick the best spot, then send people to it ───────────────
    called = find_signal_target(player, state)
    if called is not None:
        spot, runners = called
        # The summoned runner is the one who has to reach it, so the overrun is
        # theirs to chase.
        runner = state.get_player(runners[0]) if runners else player
        out.append(Candidate(
            kind="pass", target=spot,
            speed=_weight_for_lane(player.pos, spot, runner, opponents),
            friction=BALL_FRICTION_COEFF,
            recipient_id=runners[0] if runners else None,
            is_runway=True, is_called_run=True, signal_runners=tuple(runners),
        ))

    # ── Hold ─────────────────────────────────────────────────────────────────
    # Only while the wait is still short. Past the ceiling holding is not
    # offered at all, whatever it scores, so a carrier can never freeze on the
    # ball -- he must act.
    # Under the touch model an outfield player never simply holds: his
    # shortest touch away from pressure is the shield. The keeper still holds
    # the ball in his hands, which is a different thing.
    if (not DRIBBLE_TOUCH_MODEL or player.position == "GK") and player.hold_ticks < HOLD_MAX_TICKS:
        out.append(Candidate(
            kind="hold", target=player.pos.copy(), speed=0.0, friction=BALL_FRICTION_COEFF,
        ))
    return out


def _moving(mate: Player) -> bool:
    """Running, not standing or shuffling: worth leading (under grip only)."""
    return kinematics.GRIP and float(np.linalg.norm(mate.velocity)) > LEAD_MIN_SPEED_SHARE * mate.speed


def _lead_point(passer_pos: np.ndarray, mate: Player, speed: float) -> np.ndarray:
    """Where to play a ground ball to a teammate: his feet if he is standing,
    else where his run takes him by the time a ball struck at `speed` gets
    there. To his feet as he runs, it arrived where he HAD been: under
    inertia he could not stop and come back for it, and 40% of the short
    passes lost went behind him. He then times his run to the point."""
    if not _moving(mate):
        return mate.pos.copy()
    point = mate.pos.copy()
    for _ in range(3):
        t = travel_ticks(speed, BALL_FRICTION_COEFF, float(np.linalg.norm(point - passer_pos)))
        if not math.isfinite(t):
            return mate.pos.copy()
        point = _on_pitch(mate.pos + mate.velocity * min(t, LEAD_MAX_TICKS))
    return point


def _lofted_lead(passer_pos: np.ndarray, mate: Player, angle: float):
    """(spot, launch) for a lofted ball dropped where his run takes him."""
    spot = mate.pos.copy()
    launch = loft_launch(float(np.linalg.norm(spot - passer_pos)), angle)
    if launch is None or not _moving(mate):
        return spot, launch
    for _ in range(3):
        spot = _on_pitch(mate.pos + mate.velocity * min(launch[2], LEAD_MAX_TICKS))
        launch = loft_launch(float(np.linalg.norm(spot - passer_pos)), angle)
        if launch is None:
            return spot, None
    return spot, launch


def _on_pitch(point: np.ndarray) -> np.ndarray:
    return np.array([min(max(float(point[0]), 1.0), PITCH_LENGTH - 1.0),
                     min(max(float(point[1]), 1.0), PITCH_WIDTH - 1.0)])


def _ticks_to(player: Player, point: np.ndarray, reach: float = 0.0) -> float:
    """Ticks for `player` to get within `reach` of `point`.

    Under inertia, from his actual motion (kinematics.reach_ticks); otherwise
    the old straight line at top speed.
    """
    if kinematics.INERTIA:
        return float(reach_ticks(player.pos, player.velocity, player.speed, point, reach))
    run = max(0.0, float(np.linalg.norm(point - player.pos)) - reach)
    return run / max(player.speed, 1e-9)


def _touch_ticks(player: Player, touch: float) -> float:
    """How long a touch of this length takes him to reach: the touch cadence,
    or longer if even his top speed cannot cover it in one cadence."""
    return max(DRIBBLE_TOUCH_INTERVAL, touch / max(dribble_top_speed(player), 1e-9))


def dribble_nudge(touch: float) -> float:
    """How far the ball is put ahead of him at once when he touches it; it
    rolls the rest of the touch.

    DRIBBLE_NUDGE (2 m) for every touch, unless paced: then half the touch,
    at least clear of his capture radius, never past the touch itself and
    never more than 2 m. A 2 m nudge on a 1 m walking touch left the ball dead
    2 m ahead -- further than the touch -- and he sprinted onto it: walking
    was stop-and-go. (He cannot take a paced touch back before the meeting, so
    a short one may sit at the meeting point.)
    """
    from ..constants import CAPTURE_RADIUS
    if not _pace.PACE:
        return DRIBBLE_NUDGE
    return min(DRIBBLE_NUDGE, touch, max(CAPTURE_RADIUS + 0.1, 0.5 * touch))


def dribble_top_speed(player: Player) -> float:
    """His fastest pace WITH the ball: a skill-dependent share of his sprint."""
    skill = min(1.0, max(0.0, player.dribbling / 100.0))
    return player.speed * (DRIBBLE_PACE_POOR + (DRIBBLE_PACE_GREAT - DRIBBLE_PACE_POOR) * skill)


def _solve_touch(
    player: Player, direction: np.ndarray, opponents: "list[Player]", attacking_goal_x: float
) -> float | None:
    """The touch length in `direction` that maximises P(win the race to my own
    ball) x V(the spot it leaves me in). None if every length leaves the pitch.

    The race: he runs the whole touch; each opponent needs his distance to the
    spot at his own speed, plus a reaction for anyone not already there. The
    value: position_value, which already contains shot value, so a touch into a
    free shooting position is worth far more than the same metres at the back.
    """
    from .value import position_value

    skill = min(1.0, max(0.0, player.dribbling / 100.0))
    reach = DRIBBLE_MAX_TOUCH_POOR + (DRIBBLE_MAX_TOUCH_GREAT - DRIBBLE_MAX_TOUCH_POOR) * skill
    lengths = np.linspace(DRIBBLE_MIN_TOUCH, reach, DRIBBLE_TOUCH_SAMPLES)
    spots = player.pos[None, :] + lengths[:, None] * direction[None, :]
    inside = (
        (spots[:, 0] >= TOUCHLINE_MARGIN) & (spots[:, 0] <= PITCH_LENGTH - TOUCHLINE_MARGIN)
        & (spots[:, 1] >= TOUCHLINE_MARGIN) & (spots[:, 1] <= PITCH_WIDTH - TOUCHLINE_MARGIN)
    )
    if not inside.any():
        return None
    # Only the time the ball is OUT OF HIS REACH is exposed. Within about a
    # metre he still has it -- an opponent must take it off him, not beat him to
    # it -- so a close touch is nearly safe and a long one is exposed for most
    # of the cadence. Racing the full cadence made every touch up to ~3.6 m look
    # equally risky, and the longer one then always won on value.
    p_win = _touch_race(player, lengths, spots, opponents)
    value = np.array([position_value(sp, attacking_goal_x) for sp in spots])
    utility = np.where(inside, p_win * value, -1.0)
    return float(lengths[int(np.argmax(utility))])


def _solve_sprint(
    player: Player, direction: np.ndarray, opponents: "list[Player]", attacking_goal_x: float
) -> "list[tuple[str, np.ndarray, float, float]]":
    """The sprint in `direction`: _solve_touch over the lengths whose pace is
    at least PACE_SPRINT_FROM of his top pace with the ball, up to the longest
    touch his skill recovers. [(band, target, pace share 1.0, race)], or []
    if none of it is on the pitch."""
    from .value import position_value
    from ..constants import PACE_SPRINT_FROM, PACE_TOUCH_SAMPLES

    skill = min(1.0, max(0.0, player.dribbling / 100.0))
    reach = DRIBBLE_MAX_TOUCH_POOR + (DRIBBLE_MAX_TOUCH_GREAT - DRIBBLE_MAX_TOUCH_POOR) * skill
    lo = max(DRIBBLE_MIN_TOUCH, PACE_SPRINT_FROM * DRIBBLE_TOUCH_INTERVAL * dribble_top_speed(player))
    if reach <= lo + 1e-6:
        return []
    lengths = np.linspace(lo, reach, PACE_TOUCH_SAMPLES)
    spots = player.pos[None, :] + lengths[:, None] * direction[None, :]
    inside = _on_pitch_mask(spots)
    if not inside.any():
        return []
    p_win = _touch_race(player, lengths, spots, opponents)
    value = np.array([position_value(sp, attacking_goal_x) for sp in spots])
    k = int(np.argmax(np.where(inside, p_win * value, -1.0)))
    return [("sprint", spots[k], 1.0, float(p_win[k]))]


def _paced_slow(player: Player, opponents: "list[Player]") -> "list[tuple[str, np.ndarray, float, float]]":
    """The walk and the run in each of the 16 directions, planned from his
    motion: he eases toward share x top pace with the ball along the direction
    for one touch cadence (kinematics.step_velocity_rows, grip included), and
    the touch goes where that leaves him. Moving fast, his walk is a braking
    touch that ends where his momentum carries him; from a standstill it is a
    short step. [(band, target, target pace share, race)] for touches on the
    pitch and at least PACE_MIN_TOUCH long."""
    from ..constants import PACE_MIN_TOUCH, PACE_SLOW
    from ..kinematics import step_velocity_rows

    top = dribble_top_speed(player)
    dirs = np.array([[math.cos(a), math.sin(a)] for a in _DRIBBLE_ANGLES_16])
    bands = [(b, s) for b, s in PACE_SLOW for _ in dirs]
    desired = np.concatenate([dirs * (s * top) for _, s in PACE_SLOW])
    v = np.tile(np.asarray(player.velocity, dtype=float), (len(desired), 1))
    moved = np.zeros_like(desired)
    for _ in range(int(DRIBBLE_TOUCH_INTERVAL)):
        v = step_velocity_rows(v, desired, player.speed)
        moved += v
    spots = player.pos[None, :] + moved
    lengths = np.hypot(moved[:, 0], moved[:, 1])
    keep = _on_pitch_mask(spots) & (lengths >= PACE_MIN_TOUCH)
    if not keep.any():
        return []
    p_win = _touch_race(player, lengths[keep], spots[keep], opponents)
    idx = np.flatnonzero(keep)
    return [(bands[i][0], spots[i], bands[i][1], float(p)) for i, p in zip(idx, p_win)]


def _on_pitch_mask(spots: np.ndarray) -> np.ndarray:
    return ((spots[:, 0] >= TOUCHLINE_MARGIN) & (spots[:, 0] <= PITCH_LENGTH - TOUCHLINE_MARGIN)
            & (spots[:, 1] >= TOUCHLINE_MARGIN) & (spots[:, 1] <= PITCH_WIDTH - TOUCHLINE_MARGIN))


def _touch_race(player: Player, lengths: np.ndarray, spots: np.ndarray,
                opponents: "list[Player]") -> np.ndarray:
    """P(he gets to his own ball first), per touch length.

    Each side from how it is moving now. His touch cadence alone left his run
    out of it: at speed a touch back the way he came was thought 93% safe,
    kept 48%, and he reached it 29 ticks late, because he had to stop first.
    """
    reach = OUTFIELD_CONTROL_RADIUS
    t_me = np.array([_touch_ticks(player, float(L)) * max(0.0, L - reach) / L for L in lengths])
    if OWN_MOMENTUM and kinematics.INERTIA:
        t_me = np.maximum(t_me, reach_ticks(
            player.pos[None, None, :], player.velocity[None, None, :],
            np.array([[dribble_top_speed(player)]]), spots[:, None, :], reach,
        )[:, 0])
    outfield = [o for o in opponents if o.speed > 1e-9]
    if outfield:
        P = np.array([o.pos for o in outfield]); S = np.array([o.speed for o in outfield])
        if kinematics.INERTIA:
            V = np.array([o.velocity for o in outfield])
            t_opp = reach_ticks(P[None, :, :], V[None, :, :], S[None, :], spots[:, None, :])
        else:
            gap = np.hypot(spots[:, None, 0] - P[None, :, 0], spots[:, None, 1] - P[None, :, 1])
            t_opp = gap / S[None, :] + REACTION_TICKS * np.minimum(1.0, gap / REACTION_RAMP_DISTANCE)
        risk = 1.0 / (1.0 + np.exp(-(t_me[:, None] - t_opp) / DRIBBLE_RACE_TAU))
        return 1.0 - risk.max(axis=1)
    return np.ones(len(lengths))


def _runway_leads(passer: Player, mate: Player, mate_speed: float) -> list[float]:
    """Lead distances that put the receiver on the ball as it arrives.

    The lead is SOLVED for, not chosen from fixed options. Offering a constant
    4 m and 8 m meant the lead bore no relation to how far the ball had to
    travel: measured, the receiver reached the spot a median 6 ticks before the
    ball and 26% of the time more than 10 ticks early, so he stood waiting
    while a defender closed -- the timing problem that survived the physics
    change.

    Both sides of the race move together: a longer lead is further for the ball
    AND further for the runner, so solve for the lead where their arrival times
    match, then offer that plus a shorter, safer option.
    """
    from ..ball_physics import travel_ticks

    run_dir = normalize(mate.velocity)
    if float(np.linalg.norm(run_dir)) < 1e-9:
        return []

    def mismatch(lead: float) -> float:
        """Receiver arrival minus ball arrival, in ticks."""
        target = mate.pos + run_dir * lead
        ball_dist = float(np.linalg.norm(target - passer.pos))
        t_ball = travel_ticks(
            nominal_pass_speed(BALL_FRICTION_COEFF, ball_dist),
            BALL_FRICTION_COEFF, ball_dist,
        )
        if t_ball == float("inf"):
            return float("inf")
        return _ticks_to(mate, target) - t_ball

    # The runner is behind at lead 0 (the ball still has to get there) and
    # ahead at a long lead, so the crossing is bracketed. Bisect for it.
    lo, hi = 0.0, RUNWAY_MAX_LEAD
    if mismatch(hi) < 0.0:
        matched = hi          # even the longest lead leaves him chasing
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if mismatch(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        matched = hi

    # Offer a spread around the matched lead rather than the matched lead
    # alone. Matching arrival is a geometry result, not a football decision:
    # the matched lead is often long, which means the ball spends longer in
    # contested space. Handing the argmax several leads lets route_survival
    # decide how much of that timing to trade for a lane that survives --
    # on the matched lead alone, timing was right (median -0.6 ticks vs -6.2)
    # but reception fell from 53% to 41% and the match produced fewer shots.
    leads: list[float] = []
    for frac in (0.4, 0.7, 1.0):
        lead = matched * frac
        if lead < RUNWAY_MIN_LEAD:
            continue
        m = mismatch(lead)
        # Reject only what he cannot reach in time. Arriving early is no longer
        # free either: a shorter lead than `matched` means waiting, and the
        # scorer sees the extra exposure as a worse lane.
        if m != float("inf") and m <= RUNWAY_LATE_TOLERANCE:
            leads.append(lead)
    return leads


def _reachable_overruns(receiver: Player, dist: float) -> tuple[float, ...]:
    """How far past the receiver a ball may run and still plausibly be theirs.

    They are already moving toward it, so a few metres of run is free; beyond
    that the ball is nobody's.
    """
    if kinematics.INERTIA:
        # The ball runs on PAST him, so he must turn and chase from a standstill
        # in that direction: what he covers in the chase window from rest.
        reach = reach_distance(receiver.speed, PASS_CHASE_TICKS)
    else:
        reach = receiver.speed * PASS_CHASE_TICKS
    return (PASS_OVERRUN_MIN, float(min(PASS_OVERRUN_MAX, max(PASS_OVERRUN_MIN, reach))))


def _runway_weight(
    origin: np.ndarray,
    lead: np.ndarray,
    mate: Player,
) -> float:
    """Speed for a ball played into space ahead of a runner.

    A runway target is EMPTY GROUND, so the only thing that matters is that the
    ball and the runner get there together. _weight_for_lane cannot do this: it
    prices a pass from the danger on its lane and knows nothing about the
    receiver, so it applied the driven discount (70% of the flight) to runway
    balls too and the ball beat the runner to a spot the lead solver had
    carefully matched. Every runway pass arrived too early to be collected.

    So weight it to the RUNNER's arrival, with a small margin so the ball is
    still travelling into his path rather than sitting waiting for him.
    """
    ball_dist = float(np.linalg.norm(lead - origin))
    run_dist = float(np.linalg.norm(lead - mate.pos))
    t_mate = _ticks_to(mate, lead)

    # How much the ball CROSSES his run, as sin(angle): 0 when it travels along
    # his run or straight at him, 1 when it cuts square across it. Along his run
    # a ball arriving a touch early is ideal -- he runs onto it. Across it, an
    # early ball passes through the meeting point and carries on sideways, into
    # whoever is on the far side or over the line. Measured: runners collected
    # 76% of balls played along their run and 67% played towards them, but only
    # 44-48% of crossing balls, with 41-43% lost to an opponent. So a crossing
    # ball must arrive AS he does, or just after.
    cross = 0.0
    if run_dist > 1e-6 and ball_dist > 1e-6:
        cos_a = float(np.dot(lead - mate.pos, lead - origin)) / (run_dist * ball_dist)
        cross = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    margin = RUNWAY_ARRIVE_MARGIN + (RUNWAY_CROSS_MARGIN - RUNWAY_ARRIVE_MARGIN) * cross
    ticks = max(1.0, t_mate * margin)
    return min(
        PASS_SPEED_MAX,
        max(PASS_SPEED_MIN, speed_for_arrival(BALL_FRICTION_COEFF, ball_dist, ticks)),
    )


def _crosses(player: Player, state: GameState, team) -> list[Candidate]:
    """Balls from the flank into the box roles (see CROSS_* constants).

    Each role's spot goes to the teammate filling it (off_ball._box_roles),
    so the carrier and the runners agree on where the ball is going. Offered
    along the ground and in the air, both timed to the runner and both struck
    flat-out, and the scorer's reception race picks.
    """
    from .off_ball import _box_roles

    up = float(player.pos[0]) if team.direction == 1 else PITCH_LENGTH - float(player.pos[0])
    if up < CROSS_FROM or abs(float(player.pos[1]) - PITCH_WIDTH / 2.0) < CROSS_WIDE:
        return []
    _, roles = _box_roles(state, team)
    out: list[Candidate] = []
    for pid, spot in roles.items():
        mate = state.get_player(pid)
        if is_offside_position(mate, state, team, ball_x=float(player.pos[0])):
            continue
        dist = float(np.linalg.norm(spot - player.pos))
        if dist < MIN_PASS_DIST:
            continue
        common = dict(kind="pass", target=spot.copy(), friction=BALL_FRICTION_COEFF,
                      recipient_id=pid, is_runway=True, is_cross=True)
        for pace in {_runway_weight(player.pos, spot, mate), PASS_SPEED_MAX}:
            out.append(Candidate(speed=pace, **common))
        lofts = [_lofted_runway(player.pos, spot, mate)]
        for angle in (CROSS_WHIP_DEG, CROSS_FLOAT_DEG):
            launch = loft_launch(dist, angle)
            lofts.append(launch[:2] if launch is not None else None)
        for loft in lofts:
            if loft is not None:
                out.append(Candidate(speed=loft[0], vz=loft[1], **common))
    if kinematics.GRIP and CROSS_REASONING:
        out.extend(_crosses_to_players(player, state, team, set(roles)))
    return out


# A/B switch for crossing to players and valuing the finisher's freedom
# (under grip); FB_CROSS_REASONING=0 restores the box-spot crosses.
CROSS_REASONING = __import__("os").environ.get("FB_CROSS_REASONING", "1") != "0"


def _crosses_to_players(player: Player, state: GameState, team, taken: set) -> list[Candidate]:
    """Crosses to the attackers themselves -- where their runs take them --
    not only to the box spots. A teammate standing free at the back post who
    was not the holder of a spot was never offered as a target, so the ball
    went to an empty spot whose runner was marked, and the free man watched."""
    from .off_ball import _in_their_box

    out: list[Candidate] = []
    for mate in team.players:
        if mate.player_id == player.player_id or mate.position in ("GK", "LD", "CD", "RD"):
            continue
        if is_offside_position(mate, state, team, ball_x=float(player.pos[0])):
            continue
        ground_spot = _lead_point(player.pos, mate, PASS_SPEED_MAX)
        if not _in_their_box(ground_spot, team):
            continue
        common = dict(kind="pass", friction=BALL_FRICTION_COEFF,
                      recipient_id=mate.player_id, is_cross=True)
        if float(np.linalg.norm(ground_spot - player.pos)) >= MIN_PASS_DIST:
            out.append(Candidate(target=ground_spot, speed=PASS_SPEED_MAX, **common))
        for angle in (CROSS_WHIP_DEG, CROSS_FLOAT_DEG):
            spot, launch = _lofted_lead(player.pos, mate, angle)
            if launch is not None and _in_their_box(spot, team):
                out.append(Candidate(target=spot, speed=launch[0], vz=launch[1], **common))
    return out


def _lofted_runway(
    origin: np.ndarray,
    lead: np.ndarray,
    mate: Player,
) -> tuple[float, float] | None:
    """(horizontal pace, vz0) for a ball dropped onto `lead` as the runner arrives.

    The flight time IS his run time, so the trajectory follows from it: vz0 =
    g*(T+1)/2 lands it after T ticks, at dist/T along the ground. None when
    that is too flat to clear anyone, too steep to be a pass, or too hard to
    strike -- the ground ball covers those.
    """
    dist = float(np.linalg.norm(lead - origin))
    if dist < LOFT_MIN_DIST:
        return None
    T = _ticks_to(mate, lead)
    if not np.isfinite(T) or T < 1.0:
        return None
    vh = dist / T
    vz0 = BALL_GRAVITY * (T + 1.0) / 2.0
    angle = math.degrees(math.atan2(vz0, vh))
    if angle < LOFT_RUNWAY_MIN_DEG or angle > LOFT_RUNWAY_MAX_DEG:
        return None
    if math.hypot(vh, vz0) > LOFT_SPEED_MAX:
        return None
    return vh, vz0


def _weight_for_lane(
    origin: np.ndarray,
    target: np.ndarray,
    receiver: Player,
    opponents: "list[Player]",
) -> float:
    """Speed to strike a ball, chosen from how threatened its lane is.

    The passer picks an ARRIVAL TIME. He is not trying to make the ball stop at
    the receiver -- the ball stops when someone controls it, since capture
    zeroes its velocity -- he is trying to get it there before anyone else can
    reach the lane. The old rule weighted every pass to die at the receiver's
    feet, which forced feeble strikes (10 m at 9 m/s, 1.83 s in the air) and
    was the single biggest reason a press won the ball so easily.

    A clear lane gets an ordinary pass. A threatened one is driven: the same
    ball delivered in a fraction of the time, which is exactly what a player
    does when a defender is closing.
    """
    from .interception import route_survival

    dist = float(np.linalg.norm(target - origin))
    if dist < 1e-9:
        return PASS_SPEED_MAX

    normal_ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * dist
    gentle = min(
        PASS_SPEED_MAX,
        max(PASS_SPEED_MIN,
            speed_for_arrival(BALL_FRICTION_COEFF, dist, normal_ticks)),
    )

    # How safe is the ordinary ball? If nobody can reach it, keep the pace:
    # an accurately weighted pass is easier to control than a driven one.
    safety = route_survival(origin, target, gentle, BALL_FRICTION_COEFF, opponents)
    if safety >= LANE_SAFE_ENOUGH:
        return gentle

    # Otherwise buy time, in proportion to the danger: compress the flight.
    urgency = 1.0 - safety / LANE_SAFE_ENOUGH
    ticks = normal_ticks * (1.0 - (1.0 - PASS_TICKS_DRIVEN) * urgency)

    # ...but never faster than the RECEIVER can get there. How hard a ball may
    # be struck is governed by how far he has to travel, not by how far the
    # ball does: a ball to his feet can be hit as hard as you like because he
    # only has to control it, while the same pace to a target he must run 8 m
    # to reach simply beats him there. Driving every threatened pass at 20 m/s
    # is why receivers never caught them.
    run = float(np.linalg.norm(target - receiver.pos))
    if run > PASS_RECEIVER_FREE_REACH:
        t_mate = _ticks_to(receiver, target, PASS_RECEIVER_FREE_REACH)
        ticks = max(ticks, t_mate * RUNWAY_ARRIVE_MARGIN)

    return min(
        PASS_SPEED_MAX,
        max(PASS_SPEED_MIN,
            speed_for_arrival(BALL_FRICTION_COEFF, dist, ticks)),
    )

