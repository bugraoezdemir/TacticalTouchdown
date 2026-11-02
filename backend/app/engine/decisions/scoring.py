"""
Route scoring via interception-time triangles.

For each opponent:
  - Project them onto the ball's movement line → interception point
  - t_ball  = ticks for ball to reach that point (accounts for friction)
  - t_opp   = ticks for opponent to close the perpendicular distance
  - If t_opp < t_ball the opponent can intercept; risk ∝ (1 - t_opp/t_ball)

Total risk = 1 - product(1 - risk_i) over all dangerous opponents
           = probability that at least one opponent intercepts.
Score      = 1 - total_risk
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import (
    BACK_DRIBBLE_PENALTY,
    BALL_FRICTION_COEFF,
    CLOSE_SHOT_DISTANCE,
    DRIBBLE_FRICTION_COEFF,
    DRIBBLE_LOOKAHEAD_DIST,
    GK_SPEED,
    SHOT_FRICTION_COEFF,
)
from ..ball_physics import travel_ticks
from ..geometry import distance, normalize, perp_distance_unsigned
from ..models import Player


# ── Physics helper ────────────────────────────────────────────────────────────

def _ball_travel_ticks(speed: float, friction: float, dist: float) -> float:
    """
    Ticks for a decelerating ball to travel `dist` metres, or inf if it stops short.

    Delegates to ball_physics so there is one implementation of the kinematics.
    That version also accounts for BALL_STOP_THRESHOLD, which this one ignored --
    at realistic friction that omission overestimated the ball's reach by ~33%.
    """
    return travel_ticks(speed, friction, dist)


def _opponent_risk(
    perp_dist: float,
    opp_speed: float,
    t_ball_to_point: float,
) -> float:
    """
    Risk contribution from one opponent who must cover `perp_dist` metres
    while the ball needs `t_ball_to_point` ticks to reach the interception point.
    Returns 0 if the opponent can't make it in time.
    """
    if t_ball_to_point <= 0 or t_ball_to_point == float("inf"):
        return 0.0
    t_opp = perp_dist / opp_speed if opp_speed > 1e-9 else float("inf")
    if t_opp >= t_ball_to_point:
        return 0.0
    return 1.0 - t_opp / t_ball_to_point   # in (0, 1]


# ── Public scoring functions ──────────────────────────────────────────────────

def score_shot_route(
    shooter: Player,
    goal_corner: np.ndarray,
    opponents: list[Player],
    gk: Player | None,
    shot_speed: float,
) -> float:
    shot_dir = normalize(goal_corner - shooter.pos)
    if float(np.linalg.norm(shot_dir)) < 1e-9:
        return 0.0

    target_dist = distance(shooter.pos, goal_corner)
    t_total = _ball_travel_ticks(shot_speed, SHOT_FRICTION_COEFF, target_dist)
    if t_total == float("inf"):
        return 0.0   # shot won't reach goal

    clear_prob = 1.0   # probability no one intercepts

    # Field players
    for opp in opponents:
        if gk is not None and opp.player_id == gk.player_id:
            continue
        proj = float(np.dot(opp.pos - shooter.pos, shot_dir))
        if proj <= 0.0 or proj >= target_dist:
            continue   # behind shooter or beyond goal
        perp = perp_distance_unsigned(opp.pos, shooter.pos, shot_dir)
        t_ball_here = _ball_travel_ticks(shot_speed, SHOT_FRICTION_COEFF, proj)
        r = _opponent_risk(perp, opp.speed, t_ball_here)
        clear_prob *= (1.0 - r)

    # Goalkeeper — must cover lateral distance to where ball crosses goal line
    if gk is not None:
        if abs(shot_dir[0]) > 1e-9:
            crossing_y = shooter.pos[1] + shot_dir[1] / shot_dir[0] * (goal_corner[0] - shooter.pos[0])
        else:
            crossing_y = shooter.pos[1]
        gk_perp = abs(crossing_y - gk.pos[1])
        r = _opponent_risk(gk_perp, GK_SPEED, t_total)
        clear_prob *= (1.0 - r)

    score = float(clear_prob)

    # Bonus for close-range shots
    if target_dist < CLOSE_SHOT_DISTANCE:
        score = min(1.0, score + 0.2 * (1.0 - target_dist / CLOSE_SHOT_DISTANCE))

    return score


def score_pass_route(
    passer: Player,
    target_pos: np.ndarray,
    opponents: list[Player],
    pass_speed: float,
) -> float:
    pass_dir = normalize(target_pos - passer.pos)
    if float(np.linalg.norm(pass_dir)) < 1e-9:
        return 0.0

    target_dist = distance(passer.pos, target_pos)

    clear_prob = 1.0

    for opp in opponents:
        proj = float(np.dot(opp.pos - passer.pos, pass_dir))
        if proj <= 0.0 or proj >= target_dist:
            continue
        perp = perp_distance_unsigned(opp.pos, passer.pos, pass_dir)
        t_ball_here = _ball_travel_ticks(pass_speed, BALL_FRICTION_COEFF, proj)
        r = _opponent_risk(perp, opp.speed, t_ball_here)
        clear_prob *= (1.0 - r)

    return float(clear_prob)


def score_dribble_route(
    player: Player,
    dribble_dir: np.ndarray,
    opponents: list[Player],
    dribble_speed: float,
    attacking_dir: np.ndarray,
) -> float:
    target_dist = DRIBBLE_LOOKAHEAD_DIST

    clear_prob = 1.0

    for opp in opponents:
        proj = float(np.dot(opp.pos - player.pos, dribble_dir))
        if proj <= 0.0 or proj >= target_dist:
            continue
        perp = perp_distance_unsigned(opp.pos, player.pos, dribble_dir)
        t_ball_here = _ball_travel_ticks(dribble_speed, DRIBBLE_FRICTION_COEFF, proj)
        r = _opponent_risk(perp, opp.speed, t_ball_here)
        clear_prob *= (1.0 - r)

    score = float(clear_prob)

    # Penalise backward dribbles
    dot = float(np.dot(dribble_dir, attacking_dir))
    angle = math.acos(max(-1.0, min(1.0, dot)))
    if angle > math.radians(120):
        score *= BACK_DRIBBLE_PENALTY

    return score
