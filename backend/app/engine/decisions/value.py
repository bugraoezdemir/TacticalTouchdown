"""
The value model: how much is it worth to have the ball at a given point?

This is the V in utility = P x V x C, and it is what the engine has been
missing. Without it, scoring only asked "will the ball arrive?", which is why
players shot from 35 m: the shot reached the goal unblocked, so it scored well.

Deliberately analytic rather than a lookup grid -- a few flops, evaluates
anywhere with no interpolation seams, and nothing to invalidate when re-tuned.
A grid would only earn its keep if V were learned from real event data.

Defenders are deliberately absent here. Their effect belongs to P; keeping them
out of V is what stops P x V double-counting the same obstruction.
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import (
    CLEAR_PATH_BLOCKER_LIMIT,
    CLEAR_PATH_BONUS,
    CLEAR_PATH_LANE_WIDTH,
    GOALSIDE_MIN,
    GOALSIDE_WEIGHT,
    GOAL_BOTTOM_Y,
    PENETRATION_MAX,
    PENETRATION_WEIGHT,
    GOAL_CENTER_Y,
    GOAL_TOP_Y,
    PITCH_LENGTH,
    SHOT_VALUE_ANGLE_POWER,
    SHOT_VALUE_D0,
    SHOT_VALUE_K,
    VALUE_PROGRESSION_POWER,
    VALUE_W_SHOT,
)

# Reference angle: the goal mouth as seen from the penalty spot. Normalising by
# this (rather than by pi) puts the angle term near 1.0 for a normal chance, so
# the logistic in distance carries the shape and the angle term only penalises
# genuinely tight positions.
_REFERENCE_ANGLE = 0.6424


def _subtended_goal_angle(pos: np.ndarray, goal_x: float) -> float:
    """Angle the goal mouth subtends from `pos`, in radians.

    This is the standard xG geometry feature: it collapses "how far out" and
    "how tight the angle" into one number, so a wing position is penalised
    without needing a separate centrality term.
    """
    # Plain floats: this runs ~200k times per 600 ticks, and the numpy calls
    # cost more than the arithmetic. The same operations in the same order, so
    # the results are bit-identical (sqrt(x*x + y*y) is exactly what
    # np.linalg.norm computes for a 2-vector, and a 2-term dot likewise).
    px, py = float(pos[0]), float(pos[1])
    tx, ty = goal_x - px, GOAL_TOP_Y - py
    bx, by = goal_x - px, GOAL_BOTTOM_Y - py
    n_top = math.sqrt(tx * tx + ty * ty)
    n_bottom = math.sqrt(bx * bx + by * by)
    if n_top < 1e-9 or n_bottom < 1e-9:
        return math.pi
    cos_a = (tx * bx + ty * by) / (n_top * n_bottom)
    return math.acos(max(-1.0, min(1.0, cos_a)))


def shot_value(pos: np.ndarray, attacking_goal_x: float) -> float:
    """P(a shot from here becomes a goal), ignoring defenders. In [0, 1]."""
    dx = attacking_goal_x - float(pos[0])
    dy = GOAL_CENTER_Y - float(pos[1])
    dist = math.sqrt(dx * dx + dy * dy)
    angle = _subtended_goal_angle(pos, attacking_goal_x)

    # Logistic in distance, scaled by how much goal you can actually see.
    # Coefficients fitted to real-football anchors; see tests/test_value.py.
    # Clamped: under real ball physics a pass carries hundreds of metres, so a
    # trajectory projection can propose a point far off the pitch and the raw
    # exponent overflows. Saturating is correct anyway -- a shot from 2 km is
    # exactly as worthless as one from 200 m.
    z = (dist - SHOT_VALUE_D0) / SHOT_VALUE_K
    base = 0.0 if z > 700.0 else 1.0 / (1.0 + math.exp(z))
    visibility = min(1.0, angle / _REFERENCE_ANGLE) ** SHOT_VALUE_ANGLE_POWER
    return float(base * visibility)


def progression_value(pos: np.ndarray, attacking_goal_x: float) -> float:
    """Value of simply being further upfield. Smooth, monotone, in [0, 1].

    Accelerates near the goal so the gradient off-ball sampling climbs points
    toward dangerous areas rather than being flat across midfield.
    """
    own_goal_x = PITCH_LENGTH - attacking_goal_x
    progress = abs(pos[0] - own_goal_x) / PITCH_LENGTH
    return float(min(1.0, max(0.0, progress)) ** VALUE_PROGRESSION_POWER)


def position_value(pos: np.ndarray, attacking_goal_x: float) -> float:
    """Value of my team having the ball at `pos`. Strictly positive, in [0, 1].

    Used for pass targets, dribble destinations and off-ball run targets alike.
    Because it already contains shot_value, a pass into the box is automatically
    worth more than a square ball -- no separate "leads to a shot" term, which
    is how these models start double-counting.
    """
    return float(
        VALUE_W_SHOT * shot_value(pos, attacking_goal_x)
        + (1.0 - VALUE_W_SHOT) * progression_value(pos, attacking_goal_x)
    )


def threat(pos: np.ndarray, own_goal_x: float) -> float:
    """Danger to *my* goal of the opponent having the ball at `pos`.

    Literally position_value mirrored -- the opponent's attacking value measured
    against my goal. One value surface serves attack and defence, so tuning the
    shot model keeps both consistent by construction.
    """
    return position_value(pos, own_goal_x)


def defenders_beaten(
    origin: np.ndarray,
    destination: np.ndarray,
    opponents: "list",
    attacking_goal_x: float,
) -> int:
    """How many opponents end up behind the ball after this move.

    A ball that takes three defenders out of the game is worth far more than
    the destination's static value suggests. This is what makes a through-ball
    different from a square pass to the same spot.
    """
    own_goal_x = PITCH_LENGTH - attacking_goal_x
    start = abs(float(origin[0]) - own_goal_x)
    end = abs(float(destination[0]) - own_goal_x)
    if end <= start:
        return 0
    beaten = 0
    for opp in opponents:
        depth = abs(float(opp.pos[0]) - own_goal_x)
        if start <= depth < end:
            beaten += 1
    return beaten


def penetration_value(
    origin: np.ndarray,
    destination: np.ndarray,
    opponents: "list",
    attacking_goal_x: float,
) -> float:
    """Multiplier for how much a move breaks the opposition shape. >= 1.0.

    position_value alone scores where the ball LANDS. It says nothing about
    whether the move beat anybody -- so a ball played into space behind the
    defence scored the same as a sideways ball to the same square metre.
    """
    beaten = defenders_beaten(origin, destination, opponents, attacking_goal_x)
    return float(min(PENETRATION_MAX, 1.0 + PENETRATION_WEIGHT * beaten))


def opponents_goalside(
    point: np.ndarray,
    opponents: "list",
    attacking_goal_x: float,
    include_gk: bool = False,
) -> int:
    """How many opponents stand between `point` and the goal being attacked.

    This is the count that decides whether moving the ball somewhere actually
    improves the attack. A pass that lands with three more defenders in front of
    it has made things worse, however good the destination looks in isolation.
    """
    goal_x = attacking_goal_x
    count = 0
    for opp in opponents:
        if not include_gk and opp.position == "GK":
            continue
        # Strictly between the point and the goal along the attacking axis.
        if abs(float(opp.pos[0]) - goal_x) < abs(float(point[0]) - goal_x):
            count += 1
    return count


def goalside_value(
    origin: np.ndarray,
    destination: np.ndarray,
    opponents: "list",
    attacking_goal_x: float,
) -> float:
    """Penalty for a move that puts MORE defenders between ball and goal.

    Deliberately one-sided: at most 1.0, never a bonus. Rewarding forward balls
    is already penetration_value's job, and doing it twice double-counts the
    same idea. This only discourages the move that makes the attack worse --
    a square or backward ball into a pocket with three defenders newly in front
    of it, which looks fine to position_value on its own.

    It is a discount, not a veto: a genuinely valuable ball can still be worth
    playing backwards.
    """
    before = opponents_goalside(origin, opponents, attacking_goal_x)
    after = opponents_goalside(destination, opponents, attacking_goal_x)
    added = after - before  # positive = we put defenders back in the way
    if added <= 0:
        return 1.0
    return float(max(GOALSIDE_MIN, 1.0 - GOALSIDE_WEIGHT * added))


def clear_path_bonus(
    origin: np.ndarray,
    opponents: "list",
    attacking_goal_x: float,
) -> float:
    """Extra value for carrying when nothing stands between you and the goal.

    A player with a clear run should keep going until he is close enough to
    shoot; passing there usually hands the ball to someone with more defenders
    in front of them.
    """
    blockers = _effective_blockers(origin, opponents, attacking_goal_x)
    if blockers >= CLEAR_PATH_BLOCKER_LIMIT:
        return 1.0
    # Smooth decay so the bonus still bites in the common 1-2 defender case,
    # not only on a completely empty pitch: carriers face 3+ opponents 88% of
    # the time, so an all-or-nothing bonus almost never fired.
    fade = 1.0 - blockers / CLEAR_PATH_BLOCKER_LIMIT
    return float(1.0 + (CLEAR_PATH_BONUS - 1.0) * fade)


def _effective_blockers(
    origin: np.ndarray,
    opponents: "list",
    attacking_goal_x: float,
) -> float:
    """Opponents goal-side who are actually in a position to stop the carry.

    A raw goal-side count treats a full-back stranded on the far touchline as
    an obstacle. What matters is whether they are near the running lane toward
    goal, so each one is weighted by how far off that lane they stand.
    """
    goal = np.array([attacking_goal_x, GOAL_CENTER_Y])
    lane = goal - origin
    lane_len = float(np.hypot(lane[0], lane[1]))
    if lane_len < 1e-9:
        return 0.0
    unit = lane / lane_len

    total = 0.0
    for opp in opponents:
        if opp.position == "GK":
            continue
        rel = opp.pos - origin
        along = float(rel[0] * unit[0] + rel[1] * unit[1])
        if along <= 0.0 or along > lane_len:
            continue  # behind us, or beyond the goal
        lateral = abs(float(rel[0] * -unit[1] + rel[1] * unit[0]))
        if lateral >= CLEAR_PATH_LANE_WIDTH:
            continue  # too wide to intervene
        total += 1.0 - lateral / CLEAR_PATH_LANE_WIDTH
    return total
