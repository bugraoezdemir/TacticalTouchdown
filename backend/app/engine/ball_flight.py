"""
The ball in the air: height, gravity and bounce.

The ground model (ball_physics) is unchanged and still governs a rolling ball.
A ball with height or upward speed is airborne instead: no rolling friction,
horizontal pace kept, and each tick, in the same decay-then-move order as the
ground model,

    vz <- vz - BALL_GRAVITY
    h  <- h + vz
    pos <- pos + v

so after n ticks a ball launched at vz0 stands at  h = n*vz0 - g*n*(n+1)/2,
and one launched at vz0 = g*(T+1)/2 lands exactly T ticks later. On landing it
bounces -- BOUNCE_RESTITUTION of its fall back up, BOUNCE_KEEP of its pace kept
-- until a bounce is under BOUNCE_STOP_VZ, after which it rolls.

Plain floats in `step`, used by the simulation itself; `trajectory` samples the
same steps for the decision layer, so prediction and physics cannot drift.

A bent ball also turns `curl` radians a tick while it is in the air, its pace
unchanged, so it flies an arc; the first bounce ends that (physics). Nobody on
the pitch reads the spin: `trajectory` and the keeper's crossing point go by
the ball's flight as it is now, so they are wrong early and come right late.
That is what makes a bent ball hard to reach. Only the man who struck it knows
the arc (`crossing`, `arc_point`).
"""
from __future__ import annotations

import math

import numpy as np

from .constants import (
    BALL_GRAVITY,
    BALL_STOP_THRESHOLD,
    BOUNCE_KEEP,
    BOUNCE_RESTITUTION,
    BOUNCE_STOP_VZ,
    LOFT_SPEED_MAX,
    RISING_REACH,
)

_EPS = 1e-9


def airborne(height: float, vz: float) -> bool:
    return height > _EPS or vz > _EPS


def turned(vel: np.ndarray, angle: float) -> np.ndarray:
    """`vel` turned `angle` radians, left of travel positive."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vel[0] - s * vel[1], s * vel[0] + c * vel[1]])


def step(
    pos: np.ndarray,
    vel: np.ndarray,
    height: float,
    vz: float,
    friction: float,
    curl: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One tick of ball motion, in the air or along the ground."""
    if airborne(height, vz):
        vz -= BALL_GRAVITY
        height += vz
        if curl != 0.0:
            vel = turned(vel, curl)
        pos = pos + vel
        if height <= 0.0:
            height = 0.0
            up = -vz * BOUNCE_RESTITUTION
            vz = up if up >= BOUNCE_STOP_VZ else 0.0
            vel = vel * BOUNCE_KEEP
        return pos, vel, height, vz
    speed = math.hypot(float(vel[0]), float(vel[1]))
    if speed > _EPS:
        new_speed = max(0.0, speed - friction)
        vel = vel * (new_speed / speed)
    if math.hypot(float(vel[0]), float(vel[1])) < BALL_STOP_THRESHOLD:
        vel = np.zeros(2)
    return pos + vel, vel, 0.0, 0.0


def trajectory(
    pos: np.ndarray,
    vel: np.ndarray,
    height: float,
    vz: float,
    friction: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the ball is, and how high, at ticks 1..n: (n,2) and (n,)."""
    pts = np.empty((n, 2))
    hs = np.empty(n)
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    for i in range(n):
        if height > _EPS or vz > _EPS:
            vz -= BALL_GRAVITY
            height += vz
            x += vx
            y += vy
            if height <= 0.0:
                height = 0.0
                up = -vz * BOUNCE_RESTITUTION
                vz = up if up >= BOUNCE_STOP_VZ else 0.0
                vx *= BOUNCE_KEEP
                vy *= BOUNCE_KEEP
        else:
            s = math.hypot(vx, vy)
            if s > _EPS:
                k = max(0.0, s - friction) / s
                vx *= k
                vy *= k
            if math.hypot(vx, vy) < BALL_STOP_THRESHOLD:
                vx = vy = 0.0
            x += vx
            y += vy
        pts[i, 0], pts[i, 1] = x, y
        hs[i] = height
    return pts, hs


def within_reach(height0: float, hs: np.ndarray, ceiling: float) -> np.ndarray:
    """Which samples of a trajectory a player can touch, for heights `hs`
    following a start at `height0`: up to `ceiling` while the ball drops,
    but no higher than RISING_REACH while it still climbs (see constants)."""
    prev = np.concatenate([[height0], hs[:-1]])
    rising = hs > prev
    return hs <= np.where(rising, min(ceiling, RISING_REACH), ceiling)


def height_at(vz0: float, t: float) -> float:
    """Height t ticks after a launch from the ground at vz0, before it lands."""
    return vz0 * t - BALL_GRAVITY * t * (t + 1.0) / 2.0


def height_after(height: float, vz: float, t: float) -> float:
    """Height t ticks on, for a ball at `height` rising at `vz` now; below 0
    means it has landed first."""
    return height + vz * t - BALL_GRAVITY * t * (t + 1.0) / 2.0


def drive_launch(dist: float, height: float, pace: float) -> tuple[float, float, float]:
    """(horizontal speed, vz0, ticks) for a ball struck at `pace` (3-D) that is
    at `height` when it has gone `dist` metres. Twice round, because the lift
    takes a little from the horizontal speed that fixes how long it needs."""
    vh = pace
    for _ in range(2):
        T = dist / vh
        vz0 = (height + BALL_GRAVITY * T * (T + 1.0) / 2.0) / T
        vh = math.sqrt(max(pace * pace - vz0 * vz0, 1e-9))
    return vh, vz0, dist / vh


def bend_start(curl: float, ticks: float) -> float:
    """How far off the straight line to its target a ball must set off, in
    radians, to curl onto it in `ticks`: turning a little every tick before it
    moves, its path's chord lies (ticks + 1) / 2 turns from where it started."""
    return -curl * (ticks + 1.0) / 2.0


def arc_point(origin: np.ndarray, target: np.ndarray, share: float, turn: float) -> np.ndarray:
    """Where a ball from `origin` to `target` is `share` of the way along, for
    an arc that turns `turn` radians in all, left of travel positive: on the
    chord, off to the side by up to an eighth of its length x turn."""
    chord = target - origin
    bulge = 0.5 * turn * share * (1.0 - share)
    return origin + chord * share + np.array([chord[1], -chord[0]]) * bulge


def crossing(
    pos: np.ndarray, vel: np.ndarray, height: float, vz: float, curl: float, goal_x: float,
) -> tuple[float, float] | None:
    """(y, height) where a ball in flight reaches `goal_x`, on its arc, as long
    as it stays in the air; None if it comes down or turns back first."""
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    c, s = math.cos(curl), math.sin(curl)
    side = goal_x - x
    while (goal_x - x) * side > 0.0:
        if vx * side <= 0.0:
            return None
        vz -= BALL_GRAVITY
        height += vz
        if height <= 0.0:
            return None
        vx, vy = c * vx - s * vy, s * vx + c * vy
        x += vx
        y += vy
    return y, height


def chip_launch(dist: float, height: float, angle_deg: float, pace: float) -> tuple[float, float, float] | None:
    """(horizontal speed, vz0, ticks) for a ball going up at `angle_deg` that
    is at `height`, coming down, when it has gone `dist` metres. None if that
    needs more than `pace` (3-D). From tan(angle) = vz0 / vh, with vh = dist/T
    and height_at(vz0, T) = height."""
    span = dist * math.tan(math.radians(angle_deg)) - height
    if span <= _EPS:
        return None
    T = (-1.0 + math.sqrt(1.0 + 8.0 * span / BALL_GRAVITY)) / 2.0
    vh = dist / T
    vz0 = (height + BALL_GRAVITY * T * (T + 1.0) / 2.0) / T
    if math.hypot(vh, vz0) > pace:
        return None
    return vh, vz0, T


def flight_ticks(vz0: float) -> float:
    """Ticks from a launch at vz0 to landing."""
    return max(0.0, 2.0 * vz0 / BALL_GRAVITY - 1.0)


def below_window(vz0: float, ceiling: float) -> tuple[float, float]:
    """Ticks (rising, falling) at which the ball crosses `ceiling` metres.

    It is reachable before the first and after the second. (T, T) when it
    never rises above the ceiling, i.e. reachable all the way.
    """
    g = BALL_GRAVITY
    a, b, c = g / 2.0, g / 2.0 - vz0, ceiling
    disc = b * b - 4.0 * a * c
    T = flight_ticks(vz0)
    if disc <= 0.0:
        return T, T
    r = math.sqrt(disc)
    return max(0.0, (-b - r) / (2.0 * a)), min(T, (-b + r) / (2.0 * a))


def loft_launch(dist: float, angle_deg: float) -> tuple[float, float, float] | None:
    """(horizontal speed, vz0, flight ticks) to land `dist` metres away.

    None when it would need more than LOFT_SPEED_MAX. From
    tan(angle) = vz0 / vh with vz0 = g*(T+1)/2 and vh = dist/T.
    """
    if dist <= _EPS:
        return None
    tan = math.tan(math.radians(angle_deg))
    T = (-1.0 + math.sqrt(1.0 + 8.0 * dist * tan / BALL_GRAVITY)) / 2.0
    if T <= _EPS:
        return None
    vh = dist / T
    vz0 = BALL_GRAVITY * (T + 1.0) / 2.0
    if math.hypot(vh, vz0) > LOFT_SPEED_MAX:
        return None
    return vh, vz0, T
