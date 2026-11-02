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


def step(
    pos: np.ndarray,
    vel: np.ndarray,
    height: float,
    vz: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One tick of ball motion, in the air or along the ground."""
    if airborne(height, vz):
        vz -= BALL_GRAVITY
        height += vz
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
