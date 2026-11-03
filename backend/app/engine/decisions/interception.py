"""
The single interception primitive.

Every question of the form "will the ball get from A to B before someone cuts
it out?" goes through route_survival -- passes, shots, dribbles, off-ball
receivability, and defensive contest points alike. One implementation means one
place to fix the physics and one set of behaviours to tune.

Model, per opponent:
  - clamp them onto the route segment -> the point they must reach
  - t_ball = ticks for the ball to reach that point (decelerating)
  - t_opp  = reaction time + ticks for them to cover the remaining ground
  - risk   = sigmoid((t_ball - t_opp) / TAU)   -- a smooth probability

Survival = prod(1 - risk_i), i.e. nobody intercepts. Bounded in [0, 1].
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import numpy as np

from ..ball_physics import travel_ticks
from ..constants import (
    BLOCKED_LANE_RISK,
    BLOCKING_BODY_RADIUS,
    BLOCKING_MIN_ALONG,
    GK_CONTROL_RADIUS,
    GK_DIVE_SPEED,
    GK_REACTION_TICKS,
    SHOT_BLOCK_REACTION_TICKS,
    REACTION_RAMP_DISTANCE,
    OUTFIELD_CONTROL_RADIUS,
    PRESSER_BLOCK_ANGLE_DEG,
    PRESSER_BLOCK_RANGE,
    PRESSER_SIDE_RISK,
    REACTION_TICKS,
    RISK_TAU,
)
from .. import kinematics
from ..geometry import point_segment_distance
from ..kinematics import reach_ticks
from ..models import Player

_EPS = 1e-9
_PRESSER_BLOCK_COS = math.cos(math.radians(PRESSER_BLOCK_ANGLE_DEG))

# How multiple opponents' risks COMBINE into one route survival.
#   "product" -- the original: every opponent is an independent coin flip.
#   "max"     -- Option A under test: only the best-placed defender counts.
# Overridable from the environment so the test suite and probes can run under
# either mode without editing the file. A temporary experiment switch, not a
# permanent knob: once the question is settled one branch gets deleted.
COMBINE_MODE = os.environ.get("WSFB_COMBINE_MODE", "max")


def control_radius(player: Player) -> float:
    """How far a player can reach to touch the ball. Keepers use their hands."""
    if player.position == "GK":
        # goalkeeping quality scales the keeper's reach around a nominal 50.
        # Kept modest deliberately: keepers are beaten at the corners precisely
        # because their reach is limited, and an over-long reach made every
        # shot score as saved.
        return GK_CONTROL_RADIUS * (0.75 + 0.005 * player.goalkeeping)
    return OUTFIELD_CONTROL_RADIUS


def keeper_reach(keeper: Player, height: float) -> float:
    """How far from his feet he can get to a ball at `height`: his whole
    control radius up to his chest, narrowing above it. The reach at the top
    corner is what makes it the hardest part of the goal. Physics and the
    shooter's plan both ask this."""
    from ..constants import GK_HANDS_HEIGHT, GK_REACH_HEIGHT

    full = control_radius(keeper)
    if height <= GK_HANDS_HEIGHT:
        return full
    over = min(1.0, (height - GK_HANDS_HEIGHT) / (GK_REACH_HEIGHT - GK_HANDS_HEIGHT))
    return full * math.sqrt(1.0 - over * over)


def opponent_risk(
    opponent: Player,
    origin: np.ndarray,
    target: np.ndarray,
    speed: float,
    friction: float,
    reaction_ticks: float = REACTION_TICKS,
    is_shot: bool = False,
    delay_ticks: float = 0.0,
    struck: bool = True,
    reach: float | None = None,
) -> float:
    """Probability this one opponent cuts the route out.

    `reach`: his reach for this ball, if not his control radius.

    `delay_ticks` is how long before the pass can actually be PLAYED. It is
    zero for a ball being kicked now, but an off-ball player scoring a point he
    must first run to cannot be passed to until he arrives, and the defender
    has that whole time to close the lane. Leaving it out ran the race from the
    wrong starting gun: measured, a point scoring 0.766 at the moment it was
    chosen was worth 0.383 by the time the runner got there, and the carrier
    saw teammates with a mean lane of 0.135 while those teammates believed they
    were moving to 0.674. The defender was being credited with roughly half the
    time he really had.
    """
    # Momentum: during the reaction window they carry on at their current
    # velocity, so a defender already moving toward the lane closes it sooner
    # and one moving away cannot simply stop and turn.
    ex = float(opponent.pos[0]) + float(opponent.velocity[0]) * reaction_ticks
    ey = float(opponent.pos[1]) + float(opponent.velocity[1]) * reaction_ticks

    ox, oy = float(origin[0]), float(origin[1])
    sx, sy = float(target[0]) - ox, float(target[1]) - oy
    seg_len_sq = sx * sx + sy * sy
    if seg_len_sq < _EPS:
        return 0.0

    def _clamp(dx: float, dy: float) -> tuple[float, float]:
        """Where they meet the segment, as (fraction along, perpendicular gap)."""
        frac = min(1.0, max(0.0, (dx * sx + dy * sy) / seg_len_sq))
        return frac, math.hypot(dx - sx * frac, dy - sy * frac)

    t_moved, gap_moved = _clamp(ex - ox, ey - oy)
    t_now, gap_now = _clamp(
        float(opponent.pos[0]) - ox, float(opponent.pos[1]) - oy
    )

    # Momentum may bring someone TO the lane; it must never carry them out of
    # one they are already standing in. Projecting 6 ticks forward moved 76% of
    # the defenders whose body genuinely blocked a chosen lane far enough away
    # that BLOCKED_LANE_RISK stopped applying -- a marker 0.08 m off the line,
    # jogging across it, scored 0.26 risk instead of 0.90 and the ball was
    # played straight into him. He is not obliviously vacating the lane: he is
    # a defender who will react to the pass. So take whichever position is the
    # bigger threat, and score him where he actually blocks.
    t, raw_gap = (t_now, gap_now) if gap_now < gap_moved else (t_moved, gap_moved)

    gap = max(0.0, raw_gap - (control_radius(opponent) if reach is None else reach))

    # Where along the route they meet it -- the ball only has to reach that far.
    dist_along = t * math.sqrt(seg_len_sq)

    t_ball = travel_ticks(speed, friction, dist_along)
    if t_ball == float("inf"):
        return 0.0  # ball dies before reaching them
    # The defender's clock starts now; the ball's does not start until the pass
    # can be played. See delay_ticks above.
    t_ball += delay_ticks

    if opponent.speed < _EPS:
        return 0.0

    if kinematics.INERTIA and not (opponent.position == "GK" and is_shot):
        # Under inertia the cost of starting, stopping and turning is physics,
        # so there is no reaction delay to add: the time is how long HE, from
        # where he stands and how he is moving, needs to get a foot to that
        # point of the lane. (A keeper diving at a shot is not running; his
        # dive model below is kept.)
        meet = np.array([ox + sx * t, oy + sy * t])
        t_opp = float(reach_ticks(
            opponent.pos, opponent.velocity, opponent.speed, meet, control_radius(opponent)
        ))
        return _finish_risk(
            t_ball, t_opp, raw_gap, dist_along, opponent, ox, oy, sx, sy, seg_len_sq, struck
        )

    # Reaction time is the cost of *starting to move*, so it scales with how far
    # they actually have to go. A defender already standing in the lane covers
    # zero distance and blocks with his body -- charging him a full 6-tick
    # reaction made the ball "beat" him and passes straight into a marker
    # scored as safe.
    # A keeper diving at a struck ball is not the same as a defender jogging
    # into a lane: he must read the shot, choose a side and leave his feet.
    # Modelling that as ordinary running made him near-unbeatable and shots
    # collapsed to zero.
    reach_speed = opponent.speed
    if opponent.position == "GK" and is_shot:
        # Only against a struck ball. Reading and diving at a shot is slow;
        # stepping across to a rolling pass is not, and applying the dive
        # penalty there made keepers worse than outfield players at ordinary
        # interceptions.
        reaction_ticks = GK_REACTION_TICKS
        # He dives rather than runs. Letting him cover ground at running speed
        # made him better the FURTHER out the shot was struck, because the
        # longer flight gave him time his legs could not otherwise buy.
        reach_speed = GK_DIVE_SPEED
    elif is_shot:
        # An outfield player facing a ball struck at 33 m/s has almost no time
        # to adjust. Charging him the same reaction as against a rolling pass
        # made a defender 1.5 m off the line a near-certain block, which is why
        # players declined shots from good positions.
        reaction_ticks = SHOT_BLOCK_REACTION_TICKS

    travel_ticks_needed = gap / reach_speed
    if gap <= _EPS:
        t_opp = 0.0
    else:
        ramp = min(1.0, gap / REACTION_RAMP_DISTANCE)
        t_opp = reaction_ticks * ramp + travel_ticks_needed
    return _finish_risk(
        t_ball, t_opp, raw_gap, dist_along, opponent, ox, oy, sx, sy, seg_len_sq, struck
    )


def _finish_risk(
    t_ball: float,
    t_opp: float,
    raw_gap: float,
    dist_along: float,
    opponent: Player,
    ox: float,
    oy: float,
    sx: float,
    sy: float,
    seg_len_sq: float,
    struck: bool = True,
) -> float:
    """The race turned into a risk, plus the two body-in-the-way rules.

    Shared by the instant-movement and the inertia branch of opponent_risk,
    which differ only in how t_opp is worked out.
    """
    # Smooth in the time margin. A hard 1 - t_opp/t_ball had a cliff: arriving
    # one tick early scored ~0.03 while arriving on time scored exactly 0.
    risk = 1.0 / (1.0 + math.exp(-(t_ball - t_opp) / RISK_TAU))

    # A body already standing in the lane is not in a race at all -- the ball
    # has to go through them. The sigmoid is centred on zero margin, so on its
    # own it gave a marker two metres away only 0.68 risk, which is how passes
    # played straight at an opponent scored as safe.
    #
    # This is measured on the RAW gap, not the control-radius-adjusted one: a
    # keeper's 2 m reach does not mean his body fills the lane, and applying the
    # floor to him made every shot score as 90% saved.
    #
    # ...and only when he is far enough ALONG the lane to be in the way. A
    # presser at the carrier's feet sits ~5% of the way down EVERY lane at
    # once, so without this he pinned all of them at 0.90 and the carrier saw
    # no pass anywhere: median best pass 0.215, viable passes on 2% of pressed
    # decisions. He still earns ordinary sigmoid risk -- passing under pressure
    # IS harder -- he simply stops blocking every line simultaneously.
    #
    # The hard cut is DELIBERATE, not a rough edge to be smoothed. See
    # BLOCKING_MIN_ALONG: a convex ramp in its place was a severe regression in
    # play despite improving every offline metric.
    if (
        raw_gap <= BLOCKING_BODY_RADIUS
        and dist_along >= BLOCKING_MIN_ALONG
        and t_ball > 0.0
    ):
        risk = max(risk, BLOCKED_LANE_RISK)

    # ...EXCEPT a presser at the passer's feet who stands IN FRONT of the lane.
    # The cliff above exists because a presser at your feet is ~5% of the way
    # down every lane at once, and treating him as a blocker of all of them left
    # the carrier with no pass anywhere. But he does block the lanes his body is
    # actually in: measured under tight pressure, passes played within 30 deg of
    # the presser kept the ball 38% of the time, and 52% were taken by that very
    # presser within 3 m -- against 67-77% kept in every other direction. So a
    # presser blocks what is in front of him, not everything around him.
    #
    # His CURRENT position, not the momentum projection: at this range he is a
    # body in the way, not a runner racing to the lane.
    px = float(opponent.pos[0]) - ox
    py = float(opponent.pos[1]) - oy
    p_dist = math.hypot(px, py)
    if _EPS < p_dist <= PRESSER_BLOCK_RANGE:
        cos_to_presser = (px * sx + py * sy) / (p_dist * math.sqrt(seg_len_sq))
        if cos_to_presser >= _PRESSER_BLOCK_COS:
            risk = max(risk, BLOCKED_LANE_RISK)
        elif struck:
            # Outside the cone he is not in this lane: see PRESSER_SIDE_RISK.
            # Only for a STRUCK ball -- a dribble touch stays at the carrier's
            # feet, where this man really can tackle.
            risk = min(risk, PRESSER_SIDE_RISK)
    return risk


def route_survival(
    origin: np.ndarray,
    target: np.ndarray,
    speed: float,
    friction: float,
    blockers: Sequence[Player],
    exclude_ids: Iterable[int] = (),
    is_shot: bool = False,
    delay_ticks: float = 0.0,
    struck: bool = True,
) -> float:
    """P(ball travels origin -> target without being intercepted). In [0, 1].

    `delay_ticks` defers the moment the pass is played -- see opponent_risk.

    EXPERIMENT (Option A): how independent risks COMBINE is under test.

    The product treats every opponent as a separate coin flip, which is wrong
    when the ball takes one path: only the best-placed defender actually gets
    to contest it, and a second defender standing somewhere else does not help
    the first one reach the ball. Measured: four defenders 3 m OFF a lane, none
    of them blocking it, took a 20 m pass from 1.000 to 0.000 -- a swarm needs
    no interceptor, only bodies.

    Option A keeps only the largest risk. It is deliberately the extreme end of
    the axis: if compounding is the cause it will show a large effect, and
    Option B (a discounted product) can then dial it back. If A changes
    nothing, the hypothesis is wrong.
    """
    excluded = set(exclude_ids)
    if COMBINE_MODE == "max":
        worst = 0.0
        for opp in blockers:
            if opp.player_id in excluded:
                continue
            r = opponent_risk(opp, origin, target, speed, friction,
                          is_shot=is_shot, delay_ticks=delay_ticks,
                              struck=struck)
            if r > worst:
                worst = r
                if worst >= 1.0:
                    return 0.0
        return 1.0 - worst

    clear = 1.0
    for opp in blockers:
        if opp.player_id in excluded:
            continue
        r = opponent_risk(opp, origin, target, speed, friction,
                          is_shot=is_shot, delay_ticks=delay_ticks,
                              struck=struck)
        if r > 0.0:
            clear *= 1.0 - r
            if clear < 1e-6:
                return 0.0  # already hopeless; the rest cannot rescue it
    return clear


def lofted_survival(
    origin: np.ndarray,
    target: np.ndarray,
    vh: float,
    vz0: float,
    blockers: Sequence[Player],
    exclude_ids: Iterable[int] = (),
    upto: float = 1.0,
) -> float:
    """P(a lofted ball origin -> target is not cut out in flight). In [0, 1].

    `upto` (fraction of the distance) stops the lane early, for a ball whose
    arrival is contested by the reception race instead -- a cross.

    Over the top of everyone in between: it can only be reached where it is
    low enough -- rising off the passer's foot, and dropping onto the landing
    spot. Each part is raced as a ball at the lofted pace (no friction in the
    air), the drop with its clock started when the ball comes down into reach.
    The rising part is the ordinary struck-ball lane, presser rules included:
    a man at the passer's feet can still charge it down.
    """
    from ..ball_flight import below_window, flight_ticks
    from ..constants import GK_REACH_HEIGHT, HEADER_HEIGHT, RISING_REACH
    from ..physics import in_penalty_area

    offset = np.asarray(target, float) - np.asarray(origin, float)
    dist = float(np.hypot(offset[0], offset[1]))
    if dist < _EPS or vh < _EPS:
        return 1.0
    unit = offset / dist
    T = flight_ticks(vz0)
    excluded = set(exclude_ids)
    windows: dict[float, tuple[float, float]] = {}
    worst = 0.0
    for opp in blockers:
        if opp.player_id in excluded:
            continue
        ceiling = HEADER_HEIGHT
        if opp.position == "GK" and in_penalty_area(target, 0.0 if opp.assigned_pos[0] < 52.5 else 105.0):
            ceiling = GK_REACH_HEIGHT
        if ceiling not in windows:
            # Rising, he reaches no higher than RISING_REACH: no time to jump.
            rise = ceiling if opp.position == "GK" else min(ceiling, RISING_REACH)
            up, _ = below_window(vz0, rise)
            over, down = below_window(vz0, ceiling)
            if over >= down:
                down = T / 2.0   # never over his reach: reachable from the apex on
            windows[ceiling] = (up, down)
        t_up, t_down = windows[ceiling]
        risk = 0.0
        d_up = min(dist * upto, t_up * vh)
        if d_up > 0.05:
            risk = opponent_risk(opp, origin, origin + unit * d_up, vh, 0.0, struck=True)
        d_down = min(dist, t_down * vh)
        d_end = dist * upto
        if d_down < d_end - 0.05 and t_down < T:
            risk = max(risk, opponent_risk(
                opp, origin + unit * d_down, origin + unit * d_end, vh, 0.0,
                delay_ticks=t_down, struck=False,
            ))
        if risk > worst:
            worst = risk
            if worst >= 1.0:
                return 0.0
    return 1.0 - worst


def caught_out(keeper_pos: np.ndarray, goal_x: float) -> bool:
    """Is he too far off his line to get back to a ball over his head by a
    dive? Further than he ever stands to cut the angle (GK_ADVANCE)."""
    from ..constants import GK_CAUGHT_OUT

    return abs(float(keeper_pos[0]) - goal_x) > GK_CAUGHT_OUT


def bend_miss(keeper: Player, origin: np.ndarray, target: np.ndarray, vh: float, curl: float) -> float:
    """How far from a bent shot's real arrival his leap leaves him: he goes
    where it was heading GK_LEAP_TICKS before the line, and it bends on by
    curl x pace x ticks^2 / 2. Bending away from him that leaves him short.
    Bending toward him he meets it on his way across, unless he has time to
    get to where he read it, past it."""
    from ..constants import GK_DIVE_SPEED, GK_LEAP_TICKS, GK_REACTION_TICKS

    offset = np.asarray(target, float) - np.asarray(origin, float)
    bends_to = math.copysign(1.0, curl * float(offset[0]))
    miss = abs(curl) * vh * GK_LEAP_TICKS * GK_LEAP_TICKS / 2.0
    across = float(target[1]) - float(keeper.pos[1])
    if across * bends_to > 0.0:
        return miss
    ticks = float(np.hypot(offset[0], offset[1])) / vh
    return miss if abs(across) + miss <= GK_DIVE_SPEED * (ticks - GK_REACTION_TICKS) else 0.0


def lofted_shot_survival(
    origin: np.ndarray,
    target: np.ndarray,
    vh: float,
    vz0: float,
    blockers: Sequence[Player],
    curl: float = 0.0,
) -> float:
    """P(a lofted shot gets to the goal line past everybody). In [0, 1].

    lofted_survival for a ball that does not come to earth at `target`: it
    crosses the line in the air. It can be reached rising off his foot, and
    again once it has dropped within a man's reach, the keeper's hands
    included: he is raced by his dive, like any shot, with his reach at the
    height it crosses -- unless he is caught out, when he has to run back.

    A bent shot (`curl`) goes round the straight line: each part of the lane
    is raced in two chords of its arc, so a man on the line may not be in it.
    """
    from ..ball_flight import arc_point, below_window, height_at
    from ..constants import GK_REACH_HEIGHT, HEADER_HEIGHT, RISING_REACH

    offset = np.asarray(target, float) - np.asarray(origin, float)
    dist = float(np.hypot(offset[0], offset[1]))
    if dist < _EPS or vh < _EPS:
        return 1.0
    at_line = dist / vh
    turn = curl * at_line
    lane = [(0.0, 0.5), (0.5, 1.0)] if curl != 0.0 else [(0.0, 1.0)]

    def along(t0: float, t1: float, **race) -> float:
        """The worst of `opp` over the ball's flight from tick t0 to t1."""
        out = 0.0
        for a, b in lane:
            start = t0 + (t1 - t0) * a
            out = max(out, opponent_risk(
                opp, arc_point(origin, target, min(1.0, start / at_line), turn),
                arc_point(origin, target, min(1.0, (t0 + (t1 - t0) * b) / at_line), turn),
                vh, 0.0, delay_ticks=start, **race))
        return out

    windows: dict[bool, tuple[float, float]] = {}
    worst = 0.0
    for opp in blockers:
        keeper = opp.position == "GK"
        if keeper not in windows:
            ceiling = GK_REACH_HEIGHT if keeper else HEADER_HEIGHT
            rise = ceiling if keeper else min(ceiling, RISING_REACH)
            windows[keeper] = (below_window(vz0, rise)[0], below_window(vz0, ceiling)[1])
        t_up, t_down = windows[keeper]
        dives = not (keeper and caught_out(opp.pos, float(target[0])))
        reach = keeper_reach(opp, height_at(vz0, at_line)) if keeper else None
        if keeper and curl != 0.0:
            reach = max(0.0, reach - bend_miss(opp, origin, target, vh, curl))
        risk = along(0.0, min(at_line, t_up), is_shot=dives, struck=True, reach=reach)
        if t_down < at_line:
            risk = max(risk, along(t_down, at_line, is_shot=dives, struck=False, reach=reach))
        if risk > worst:
            worst = risk
            if worst >= 1.0:
                return 0.0
    return 1.0 - worst


# ── Batched form, for the one hot path that needs it ──────────────────────────
def _travel_ticks_many(speed: np.ndarray, friction: float, dist: np.ndarray) -> np.ndarray:
    """Vectorised ball_physics.travel_ticks: same closed form, same inf rules."""
    from ..constants import BALL_STOP_THRESHOLD as thr

    speed = np.asarray(speed, dtype=float)
    dist = np.asarray(dist, dtype=float)
    out = np.full(np.broadcast(speed, dist).shape, np.inf)
    if friction <= _EPS:
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(speed > _EPS, dist / speed, np.inf)
        return np.where(dist <= 0.0, 0.0, res)
    n = np.floor((speed - thr) / friction) + 1.0
    n = np.minimum(n, np.floor(speed / friction))
    carry = np.where((speed > thr) & (n > 0), n * speed - friction * n * (n + 1.0) / 2.0, 0.0)
    a = friction / 2.0
    b = friction / 2.0 - speed
    disc = b * b - 4.0 * a * dist
    ok = (speed >= _EPS) & (dist <= carry) & (disc >= 0.0)
    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    out = np.where(ok, t, out)
    return np.where(dist <= 0.0, 0.0, out)


def route_survival_many(
    origin: np.ndarray,
    targets: np.ndarray,
    speeds: np.ndarray,
    friction: float,
    blockers: Sequence[Player],
    delays: np.ndarray | None = None,
) -> np.ndarray:
    """route_survival for many targets at once (passes only, no exclusions).

    Identical rules to opponent_risk + route_survival -- momentum projection,
    the BLOCKING_MIN_ALONG cliff, the presser cone, and the combine mode -- but
    computed as a (targets x opponents) array. The off-ball support search asks
    this for 17 points x 11 opponents per decision, and doing it one
    opponent_risk at a time in pure Python was 26 of every 66 seconds of
    simulation. Verified equal to the scalar version on real match states.
    """
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    m = targets.shape[0]
    speeds = np.broadcast_to(np.asarray(speeds, dtype=float), (m,))
    delays = np.zeros(m) if delays is None else np.broadcast_to(np.asarray(delays, dtype=float), (m,))
    if COMBINE_MODE != "max":
        return np.array([
            route_survival(origin, targets[i], float(speeds[i]), friction, blockers,
                           delay_ticks=float(delays[i]))
            for i in range(m)
        ])
    opps = [o for o in blockers if o.speed >= _EPS]
    if not opps:
        return np.ones(m)
    P = np.array([o.pos for o in opps], dtype=float)                  # (N,2)
    V = np.array([o.velocity for o in opps], dtype=float)
    spd = np.array([o.speed for o in opps], dtype=float)              # (N,)
    ctrl = np.array([control_radius(o) for o in opps], dtype=float)
    if kinematics.INERTIA and kinematics._rust is not None:
        if kinematics.GRIP and not kinematics._grip_flat:
            kinematics.grip_table()
        return kinematics._rust.lanes(
            float(origin[0]), float(origin[1]), targets, np.array(speeds), np.array(delays),
            friction, P, V, spd, ctrl, kinematics.GRIP,
        )
    E = P + V * REACTION_TICKS

    o = np.asarray(origin, dtype=float)
    S = targets - o                                                    # (M,2)
    seg2 = np.sum(S * S, axis=1)                                       # (M,)
    valid = seg2 >= _EPS
    seg2s = np.where(valid, seg2, 1.0)

    def clamp(D):                                                      # D (N,2)
        frac = np.clip((S[:, None, 0] * D[None, :, 0] + S[:, None, 1] * D[None, :, 1])
                       / seg2s[:, None], 0.0, 1.0)                     # (M,N)
        gx = D[None, :, 0] - S[:, None, 0] * frac
        gy = D[None, :, 1] - S[:, None, 1] * frac
        return frac, np.hypot(gx, gy)

    t_m, g_m = clamp(E - o)
    t_n, g_n = clamp(P - o)
    use_now = g_n < g_m
    t = np.where(use_now, t_n, t_m)
    raw_gap = np.where(use_now, g_n, g_m)
    gap = np.maximum(0.0, raw_gap - ctrl[None, :])
    dist_along = t * np.sqrt(seg2s)[:, None]
    t_ball = _travel_ticks_many(speeds[:, None], friction, dist_along)
    alive = np.isfinite(t_ball)
    t_ball = np.where(alive, t_ball + delays[:, None], 0.0)

    if kinematics.INERTIA:
        # Same rule as opponent_risk: physics from his position and velocity
        # to the lane point, no reaction delay. Passes only, so no keeper dive.
        meet = o[None, None, :] + S[:, None, :] * t[:, :, None]          # (M,N,2)
        t_opp = reach_ticks(P[None, :, :], V[None, :, :], spd[None, :], meet, ctrl[None, :])
    else:
        ramp = np.minimum(1.0, gap / REACTION_RAMP_DISTANCE)
        t_opp = np.where(gap <= _EPS, 0.0, REACTION_TICKS * ramp + gap / spd[None, :])
    with np.errstate(over="ignore"):
        risk = 1.0 / (1.0 + np.exp(-(t_ball - t_opp) / RISK_TAU))
    block = (raw_gap <= BLOCKING_BODY_RADIUS) & (dist_along >= BLOCKING_MIN_ALONG) & (t_ball > 0.0)
    risk = np.where(block, np.maximum(risk, BLOCKED_LANE_RISK), risk)

    Dp = P - o
    pd = np.hypot(Dp[:, 0], Dp[:, 1])                                  # (N,)
    cosp = ((S[:, None, 0] * Dp[None, :, 0] + S[:, None, 1] * Dp[None, :, 1])
            / (np.maximum(pd, _EPS)[None, :] * np.sqrt(seg2s)[:, None]))
    near = (pd[None, :] > _EPS) & (pd[None, :] <= PRESSER_BLOCK_RANGE)
    cone = near & (cosp >= _PRESSER_BLOCK_COS)
    risk = np.where(cone, np.maximum(risk, BLOCKED_LANE_RISK), risk)
    # Passes only, so always a struck ball: see PRESSER_SIDE_RISK.
    risk = np.where(near & ~cone, np.minimum(risk, PRESSER_SIDE_RISK), risk)
    risk = np.where(alive, risk, 0.0)

    worst = risk.max(axis=1)
    surv = np.where(worst >= 1.0, 0.0, 1.0 - worst)
    return np.where(valid, surv, 1.0)
