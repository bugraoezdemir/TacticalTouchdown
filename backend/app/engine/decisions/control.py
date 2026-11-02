"""
Receiving the ball: how pace affects a first touch.

Two numbers, both in [0, 1], both 1 for anything slower than a player's easy
speed and falling linearly to the hard speed (see CONTROL_* in constants):

  control_probability -- a clean trap: the ball is his, at his feet
  first_time_quality  -- how well he can pass or shoot it without trapping

Relative speed is what the receiver feels: the ball's velocity minus his own,
so a runner taking a ball along his run feels it slower than a man standing
still. Pure functions; physics sets the pending first touch at capture, the
on-ball decision reads it, the game loop applies the outcome.
"""
from __future__ import annotations

import numpy as np

from ..constants import (
    AERIAL_CONTROL_FACTOR,
    CONTROL_COMFORT,
    CONTROL_HEIGHT,
    CONTROL_EASY_SPEED,
    CONTROL_HARD_SPEED,
    CONTROL_MIN_P,
    CONTROL_SKILL_SPAN,
    FIRST_TIME_MIN_Q,
)


def easy_speed(player) -> float:
    """The pace this player still takes cleanly; better dribblers take more."""
    skill = min(1.0, max(0.0, float(player.dribbling) / 100.0))
    return CONTROL_EASY_SPEED * (1.0 - CONTROL_SKILL_SPAN / 2.0 + CONTROL_SKILL_SPAN * skill)


def hardness(speed: float, player) -> float:
    """0 at or below his easy speed, 1 at the hard speed and beyond."""
    easy = easy_speed(player)
    if speed <= easy:
        return 0.0
    return min(1.0, (speed - easy) / max(CONTROL_HARD_SPEED - easy, 1e-9))


def control_probability(speed: float, player) -> float:
    """Chance a ball arriving at `speed` (relative, m/tick) is trapped cleanly."""
    return 1.0 - (1.0 - CONTROL_MIN_P) * hardness(speed, player)


def first_time_quality(speed: float, player) -> float:
    """How well a ball arriving at `speed` can be played on without a trap."""
    return 1.0 - (1.0 - FIRST_TIME_MIN_Q) * hardness(speed, player)


def trap_probability(player) -> float:
    """His chance of controlling the first touch he has pending.

    A header cannot be trapped at all; a ball still in the air below the
    chest can, but less surely than one along the ground.
    """
    if player.first_touch_height > CONTROL_HEIGHT:
        return 0.0
    p = control_probability(player.first_touch_speed, player)
    if player.first_touch_height > 0.0:
        p *= AERIAL_CONTROL_FACTOR
    return p


def comfortable_strike(dist: float, receiver, friction: float) -> float:
    """The strike that reaches `dist` at just under the receiver's easy speed.

    The judged pace: as firm as it can be while still a clean first touch.
    Friction takes `friction` off every tick, so a ball arriving at a after t
    ticks left at a + friction*t, having covered a*t + friction*t*(t-1)/2;
    solve that for t.
    """
    a = CONTROL_COMFORT * easy_speed(receiver)
    if friction <= 0.0:
        return a
    b = a - friction / 2.0
    t = (-b + np.sqrt(b * b + 2.0 * friction * dist)) / friction
    return a + friction * t


def relative_speed(ball_velocity: np.ndarray, player_velocity: np.ndarray) -> float:
    d = np.asarray(ball_velocity, float) - np.asarray(player_velocity, float)
    return float(np.hypot(d[0], d[1]))
