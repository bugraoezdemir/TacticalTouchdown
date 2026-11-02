"""
The speed/safety trade-off for off-ball movement.

The primitive objective of an off-ball action is to pick a point and turn
toward it. Two obvious choices are both wrong:

  - run at the ball's current position -> you arrive where it *was*
  - run at the full interception point -> safe but slow, and it bets on a
    trajectory that may never happen

So the real optimisation is the trade-off between them, and that is one bounded
scalar:

    aim(lam) = ball + heading * (lam * carrier_speed * t_my_arrival)
    lam = 0  pure pursuit    lam = 1  full interception

lam is not a tuned constant. It means "how much do I trust this trajectory
prediction", and it is derived from three things:

  1. prediction confidence -- fast, straight, and soon is predictable
  2. teammate cover        -- if others are already pressing, prioritise
                              positioning over speed (raises lam)
  3. opponent outlets      -- if the carrier has open passes, the carry is a
                              bad bet (lowers lam)
"""
from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np

from ..constants import (
    COME_TO_BALL_MARGIN,
    PRESSING_LAMBDA_WEIGHT,
    MIN_SUPPORT_PASS_DIST,
    PITCH_LENGTH,
    SPACE_RADIUS,
    SUPPORT_FALLOFF,
    LAMBDA_COVER_FLOOR,
    WAIT_HORIZON_TICKS,
    WAIT_SAFETY_TICKS,
    WAIT_STRIDE_TICKS,
    WAIT_VALUE_MARGIN,
    LAMBDA_CONFIDENCE_HORIZON,
    LAMBDA_OUTLET_WEIGHT,
    PASS_SPEED_LOW,
    BALL_FRICTION_COEFF,
    PLAYER_BASE_SPEED,
    GK_REACH_HEIGHT,
    HEADER_HEIGHT,
)
from .. import kinematics
from . import tick_cache
from ..ball_flight import airborne, trajectory, within_reach
from ..ball_physics import nominal_pass_speed
from ..geometry import distance, normalize
from ..kinematics import reach_ticks
from ..models import GameState, Player

_EPS = 1e-9

# A/B switch (on); 0 restores the receiver who keeps to his timed run whoever
# is closing on the ball.
COME_TO_BALL = os.environ.get("FB_COME_TO_BALL", "1") != "0"


def prediction_confidence(
    carrier_speed: float,
    straightness: float,
    t_arrive: float,
) -> float:
    """Fast x straight x soon. In [0, 1]."""
    spd = min(1.0, carrier_speed / max(PLAYER_BASE_SPEED, _EPS))
    decay = math.exp(-t_arrive / LAMBDA_CONFIDENCE_HORIZON)
    return float(max(0.0, min(1.0, spd * straightness * decay)))


def compute_lambda(
    confidence: float,
    cover: float,
    outlets: float,
    pressing: float = 1.0,
) -> float:
    """Blend the terms into the speed/safety trade-off. In [0, 1].

    `pressing` SCALES the result rather than replacing it, so the confidence,
    cover and outlet terms all still modulate the outcome -- a pressing player
    still holds shape when teammates already cover the ball. Above 1.0 it pulls
    lambda toward 0 (go straight at the ball); below 1.0 it pushes toward
    aligning with the ball's trajectory and waiting.
    """
    c = max(0.0, min(1.0, cover))
    o = max(0.0, min(1.0, outlets))
    raw = confidence * (LAMBDA_COVER_FLOOR + (1.0 - LAMBDA_COVER_FLOOR) * c)
    raw *= 1.0 - LAMBDA_OUTLET_WEIGHT * o
    raw /= 1.0 + PRESSING_LAMBDA_WEIGHT * (pressing - 1.0)
    return float(max(0.0, min(1.0, raw)))


def aim_point(
    ball_pos: np.ndarray,
    carrier_heading: np.ndarray,
    carrier_speed: float,
    t_arrive: float,
    lam: float,
) -> np.ndarray:
    """Where to run. lam=0 is the ball itself; lam=1 leads it fully."""
    lead = lam * carrier_speed * t_arrive
    return ball_pos + carrier_heading * lead


def opponent_outlets(state: GameState, carrier: Player) -> float:
    """Fraction of the carrier's teammates who are open for a pass. In [0, 1].

    Open outlets mean the ball probably will not stay on its current carry, so
    committing to an interception point is a poor bet.

    The same for every defender in a tick, so worked out once per tick, and
    the lanes batched (route_survival_many, pinned equal to the scalar form).
    """
    from .interception import route_survival_many

    key = (id(state), state.tick, carrier.player_id, float(carrier.pos[0]), float(carrier.pos[1]),
           float(sum(p.pos[0] + 3.0 * p.pos[1] for p in state.all_players())))
    if _OUTLETS.get("key") == key:
        return _OUTLETS["value"]
    mates = state.teammates_of(carrier.player_id)
    value = 0.0
    if mates:
        far = [m for m in mates if float(np.linalg.norm(m.pos - carrier.pos)) >= 5.0]
        if far:
            pts = np.array([m.pos for m in far])
            gaps = np.hypot(pts[:, 0] - carrier.pos[0], pts[:, 1] - carrier.pos[1])
            paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps])
            lanes = route_survival_many(carrier.pos, pts, paces, BALL_FRICTION_COEFF,
                                        state.opponents_of(carrier.team_id))
            open_count = float(np.sum(lanes > 0.5))
            value = min(1.0, open_count / max(1.0, len(mates) * 0.5))
    _OUTLETS.update(key=key, value=value)
    return value


_OUTLETS: dict = {}


def carrier_straightness(carrier: Player) -> float:
    """How committed the carrier is to their current heading. In [0, 1].

    Uses speed as the proxy for commitment: a player at full tilt cannot turn
    sharply, so their heading is informative. A near-stationary one tells us
    nothing, which correctly drives lambda toward pure pursuit.
    """
    speed = float(np.linalg.norm(carrier.velocity))
    return min(1.0, speed / max(PLAYER_BASE_SPEED, _EPS))


def _shared_path(key: tuple, build):
    """The ball's path for this tick, built once (see tick_cache). Callers only
    read it."""
    cache = tick_cache.get()
    if cache is None:
        return build()
    hit = cache.get(key)
    if hit is None:
        hit = cache[key] = build()
    return hit


def ball_intercept_point(
    ball_pos: np.ndarray,
    ball_velocity: np.ndarray,
    friction: float,
    chaser: Player,
    max_ticks: int = 120,
    height: float = 0.0,
    vz: float = 0.0,
    reach: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Earliest point on the ball's own trajectory the chaser can reach.

    A loose ball is not a stationary target. Chasing where it currently sits
    misses by however far it rolls while you run -- up to 15 m for a ball
    travelling at pass speed. This walks the ball's decaying path and returns
    the first point the chaser gets to in time, falling back to the ball's
    resting place if they cannot catch it at all.

    A ball in the air is met only where it has come down within his reach.

    `reach`: how close he must get. Getting exactly ONTO the path is later,
    and for a ball going past him at twice his pace much later: the answer
    slides on down the path with the ball and he runs alongside it, 2 m away,
    without ever putting a foot out.
    """
    from ..ball_physics import travel_distance

    speed = math.hypot(float(ball_velocity[0]), float(ball_velocity[1]))
    if airborne(height, vz) and chaser.speed > 1e-9:
        key = ("flight", float(ball_pos[0]), float(ball_pos[1]), float(ball_velocity[0]),
               float(ball_velocity[1]), height, vz, friction, max_ticks)
        path, hs = _shared_path(
            key, lambda: trajectory(ball_pos, ball_velocity, height, vz, friction, max_ticks),
        )
        ts = np.arange(1, max_ticks + 1, dtype=float)
        up = GK_REACH_HEIGHT if chaser.position == "GK" else HEADER_HEIGHT
        if kinematics.INERTIA:
            need = reach_ticks(chaser.pos, chaser.velocity, chaser.speed, path, reach)
        else:
            need = np.maximum(0.0, np.hypot(path[:, 0] - chaser.pos[0], path[:, 1] - chaser.pos[1])
                              - reach) / chaser.speed
        low = (hs <= up) if chaser.position == "GK" else within_reach(height, hs, up)
        ok = np.nonzero((need <= ts) & low)[0]
        if ok.size:
            i = int(ok[0])
            return path[i].copy(), float(ts[i])
        return path[-1].copy(), float(max_ticks)
    if speed < 1e-9 or chaser.speed <= 1e-9:
        return ball_pos.copy(), 0.0

    direction = normalize(ball_velocity)
    px, py = float(chaser.pos[0]), float(chaser.pos[1])

    if kinematics.INERTIA:
        # The whole path at once, raced against his real reach time. The path
        # is the ball's, the same for every chaser: built once a tick.
        ts = np.arange(1, max_ticks + 1, dtype=float)

        def build():
            along = np.array([travel_distance(speed, friction, float(t)) for t in ts])
            return ball_pos[None, :] + np.outer(along, direction)
        key = ("roll", float(ball_pos[0]), float(ball_pos[1]), float(ball_velocity[0]),
               float(ball_velocity[1]), friction, max_ticks)
        path = _shared_path(key, build)
        need = reach_ticks(chaser.pos, chaser.velocity, chaser.speed, path, reach)
        ok = np.nonzero(need <= ts)[0]
        if ok.size:
            i = int(ok[0])
            return path[i].copy(), float(ts[i])
    else:
        for t in range(1, max_ticks + 1):
            travelled = travel_distance(speed, friction, t)
            x = float(ball_pos[0]) + float(direction[0]) * travelled
            y = float(ball_pos[1]) + float(direction[1]) * travelled
            if math.hypot(x - px, y - py) - reach <= chaser.speed * t:
                return np.array([x, y]), float(t)

    # Cannot catch it in flight: aim for where it comes to rest.
    from ..ball_physics import carry_distance

    rest = carry_distance(speed, friction)
    end = ball_pos + direction * rest
    return end, float(max_ticks)


def closing_on_path(
    ball_pos: np.ndarray,
    ball_velocity: np.ndarray,
    friction: float,
    opponents: "Sequence[Player]",
    from_tick: int,
    until_tick: float,
    height: float = 0.0,
    vz: float = 0.0,
) -> bool:
    """Is an opponent getting too close to this ball on its way?

    From `from_tick` to `until_tick` ticks ahead: can anyone reach the ball's
    path, within his control radius and from how he is moving, in the ball's
    own time there plus COME_TO_BALL_MARGIN of it? In the air, only where it
    has come down within his reach.
    """
    from ..ball_physics import travel_distance
    from .interception import control_radius

    last = int(math.ceil(until_tick))
    outfield = [o for o in opponents if o.speed > 1e-9]
    if last < from_tick or not outfield:
        return False
    ts = np.arange(1, last + 1, dtype=float)
    low = None
    if airborne(height, vz):
        key = ("flight", float(ball_pos[0]), float(ball_pos[1]), float(ball_velocity[0]),
               float(ball_velocity[1]), height, vz, friction, last)
        path, hs = _shared_path(
            key, lambda: trajectory(ball_pos, ball_velocity, height, vz, friction, last),
        )
        low = np.array([hs <= GK_REACH_HEIGHT if o.position == "GK"
                        else within_reach(height, hs, HEADER_HEIGHT) for o in outfield]).T
    else:
        speed = math.hypot(float(ball_velocity[0]), float(ball_velocity[1]))
        if speed < _EPS:
            return False
        along = np.array([travel_distance(speed, friction, float(t)) for t in ts])
        path = ball_pos[None, :] + np.outer(along, normalize(ball_velocity))
    P = np.array([o.pos for o in outfield])
    V = np.array([o.velocity for o in outfield])
    S = np.array([o.speed for o in outfield])
    ctrl = np.array([control_radius(o) for o in outfield])
    need = reach_ticks(P[None, :, :], V[None, :, :], S[None, :], path[:, None, :], ctrl[None, :])
    close = need <= ts[:, None] * (1.0 + COME_TO_BALL_MARGIN)
    if low is not None:
        close &= low
    return bool(close[from_tick - 1:].any())


def support_need(best_pass_safety: float) -> float:
    """How much an off-ball teammate should prioritise SUPPORT over THREAT.

    Keyed on the carrier's best available pass, not on where the ball is. A
    carrier with no safe option needs someone to come short and offer one; a
    carrier with good options is better served by runs into dangerous space.

    Ball position was the original input and is only a +0.63 proxy for this,
    with roughly 50% spread inside each bucket -- so about a third of the time
    it gave the wrong answer. The case it always got wrong is the one that
    matters most: a carrier surrounded on the halfway line reads as
    "mid-pitch, go make runs" when what he needs is an outlet.
    """
    safety = min(1.0, max(0.0, best_pass_safety))
    return float((1.0 - safety) ** SUPPORT_FALLOFF)


def best_pass_safety(state: GameState, carrier: Player) -> float:
    """Survival of the carrier's most reliable pass to a teammate. In [0, 1].

    Computed once per tick per team and shared -- it is the same number for
    every off-ball teammate, so evaluating it per player (let alone per
    candidate point) would be pure waste.
    """
    from .interception import route_survival_many

    mates = [
        mate.pos for mate in state.teammates_of(carrier.player_id)
        if mate.position != "GK" and distance(mate.pos, carrier.pos) >= MIN_SUPPORT_PASS_DIST
    ]
    if not mates:
        return 0.0
    speeds = [nominal_pass_speed(BALL_FRICTION_COEFF, distance(pos, carrier.pos)) for pos in mates]
    return float(route_survival_many(
        carrier.pos, np.array(mates), np.array(speeds), BALL_FRICTION_COEFF,
        state.opponents_of(carrier.team_id),
    ).max())


def space_quality(
    point: np.ndarray,
    opponents: Sequence[Player],
    radius: float = SPACE_RADIUS,
) -> float:
    """How free `point` is of opponents. 1.0 = nobody near, 0.0 = crowded.

    Running into space is what makes a pass possible; running at the ball
    brings your marker with you and clogs the very lane you want played into.
    """
    if not opponents:
        return 1.0
    nearest = min(
        math.hypot(float(point[0]) - float(o.pos[0]), float(point[1]) - float(o.pos[1]))
        for o in opponents
    )
    return float(min(1.0, nearest / radius))


def _on_pitch(point: "np.ndarray") -> bool:
    """A meeting point off the pitch is not a meeting point.

    Under the real ball model a pass carries hundreds of metres, so projecting
    the trajectory forward now runs the search far beyond the touchline -- it
    was proposing points kilometres away and overflowing the value model.
    """
    from ..constants import PITCH_LENGTH, PITCH_WIDTH

    return (
        0.0 <= float(point[0]) <= PITCH_LENGTH
        and 0.0 <= float(point[1]) <= PITCH_WIDTH
    )


def best_meeting_point(
    ball_pos: np.ndarray,
    ball_velocity: np.ndarray,
    friction: float,
    receiver: Player,
    attacking_goal_x: float,
    opponents: "Sequence[Player]",
    max_ticks: int = 120,
    height: float = 0.0,
    vz: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Where along the ball's path this player should meet it.

    In the air: the earliest point it comes within his reach -- there is no
    waiting for a dropping ball to run on, since it is contested where it lands.

    Normally the earliest reachable point: if anyone can contest the ball, first
    touch is all that matters. Letting it run on is a luxury, allowed only when
    BOTH conditions hold at the point actually chosen:

      * no opponent can get there anywhere near as soon as we can, and
      * waiting is worth appreciably more than taking it immediately.

    The safety test must be applied to the CHOSEN point, not to where the ball
    is now. Testing the current position let a receiver wait until tick 120 for
    a spot an opponent reached at tick 20 -- it lost the ball every time.
    """
    from ..ball_physics import travel_distance
    from .value import position_value

    earliest = ball_intercept_point(
        ball_pos, ball_velocity, friction, receiver, max_ticks, height, vz
    )
    speed = math.hypot(float(ball_velocity[0]), float(ball_velocity[1]))
    if speed < 1e-9 or receiver.speed <= 1e-9 or airborne(height, vz):
        return earliest

    early_point, early_tick = earliest
    early_value = position_value(early_point, attacking_goal_x)

    direction = normalize(ball_velocity)
    px, py = float(receiver.pos[0]), float(receiver.pos[1])

    best_point, best_tick, best_value = early_point, early_tick, early_value
    # Sample the trajectory coarsely. Walking every tick and testing safety at
    # each one dominated the whole engine; the ball moves under a metre per
    # tick, so a stride costs nothing in fidelity.
    horizon = min(max_ticks, int(early_tick) + WAIT_HORIZON_TICKS)
    for t in range(int(early_tick) + WAIT_STRIDE_TICKS, horizon + 1, WAIT_STRIDE_TICKS):
        travelled = travel_distance(speed, friction, t)
        x = float(ball_pos[0]) + float(direction[0]) * travelled
        y = float(ball_pos[1]) + float(direction[1]) * travelled
        mine = _ticks(receiver, x, y)
        if mine > t:
            continue  # cannot get there in time

        point = np.array([x, y])
        # A ball now carries hundreds of metres, so the projection runs clean
        # off the pitch: waiting for it there is not a decision anyone can make.
        if not _on_pitch(point):
            break
        if not _comfortably_ours(point, mine, receiver, opponents):
            continue

        value = position_value(point, attacking_goal_x)
        if value > best_value * (1.0 + WAIT_VALUE_MARGIN) and value > best_value:
            best_point, best_tick, best_value = point, float(t), value

    return best_point, best_tick


def _comfortably_ours(
    point: np.ndarray,
    my_ticks: float,
    receiver: Player,
    opponents: "Sequence[Player]",
) -> bool:
    """Can we hold this spot with room to spare?

    Not "is the ball uncontested right now" -- that was the mistake. The
    question is whether an opponent beats us to the spot we intend to wait at.
    """
    x, y = float(point[0]), float(point[1])
    # The same opponent and the same point on the ball's path come up for each
    # teammate weighing whether to let the ball run: worked out once a tick.
    cache = tick_cache.get()
    for opponent in opponents:
        if opponent.speed <= 1e-9:
            continue
        if cache is None:
            theirs = _ticks(opponent, x, y)
        else:
            key = ("opp_ticks", opponent.player_id, x, y)
            theirs = cache.get(key)
            if theirs is None:
                theirs = cache[key] = _ticks(opponent, x, y)
        if theirs <= my_ticks + WAIT_SAFETY_TICKS:
            return False
    return True


def _ticks(player: Player, x: float, y: float) -> float:
    """Ticks for `player` to reach (x, y): from his actual motion under
    inertia, else the old straight line at top speed."""
    if kinematics.INERTIA:
        # The one-point form reach_ticks itself uses, without wrapping the
        # point in an array only to unwrap it again.
        return kinematics._reach_ticks_scalar(
            float(player.pos[0]), float(player.pos[1]),
            float(player.velocity[0]), float(player.velocity[1]),
            float(player.speed), x, y, 0.0,
        )
    return math.hypot(x - float(player.pos[0]), y - float(player.pos[1])) / player.speed
