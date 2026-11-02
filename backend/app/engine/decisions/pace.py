"""
Pace with the ball, and Tempo (workplan 2.9, decision D-12 (a)).

Every dribble direction is offered at a walk, a run and a sprint
(candidates._paced_slow, _solve_sprint); the scorer picks the pace by the same argmax as
everything else. Two terms here make the choice a real one:

- tempo_factor: the Tempo slider's tilt, a context multiplier per pace.
- wait_value: what walking buys -- the best pass a second from now, from
  where teammates' runs will have taken them. Without it the slow pace only
  ever wins on safety, and Tempo would do all the work.

Idling is the walk winning; there is no separate idle rule. FB_PACE=0 turns
it all off and restores the solved touch plus the idle rule (needs grip).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import kinematics
from ..constants import (
    BALL_FRICTION_COEFF,
    MIN_PASS_DIST,
    PACE_WAIT_TICKS,
    PASS_CONTINUATION_DISCOUNT,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TEMPO_EXPONENT,
    TEMPO_PIVOT_SHARE,
)

PACE = kinematics.GRIP and os.environ.get("FB_PACE", "1") != "0"


def tempo_factor(share: float, tempo: float) -> float:
    """Context multiplier for a dribble at this pace share.

    The same reciprocal trade as Pass range: a pace above TEMPO_PIVOT_SHARE
    scales with `tempo`, a slower one with its reciprocal, and at tempo 1.0
    every pace scores 1.0.
    """
    if tempo == 1.0 or share <= 1e-9 or tempo <= 0.0:
        return 1.0
    return float(tempo ** (TEMPO_EXPONENT * math.log(share / TEMPO_PIVOT_SHARE)))


_WAIT: dict = {}


def wait_value(player, state, opponents) -> float:
    """The best pass a second from now: max over teammates of
    lane(here -> where his run takes him in PACE_WAIT_TICKS)
    x position_value(that spot) x PASS_CONTINUATION_DISCOUNT x penetration.

    Opponents are moved along their own motion for the same second, so a
    defender tracking the runner closes the lane he would open. Once per
    decision (cached on the tick and the carrier).
    """
    from types import SimpleNamespace
    from ..ball_physics import nominal_pass_speed
    from .interception import route_survival_many
    from .value import penetration_value, position_value

    key = (id(state), state.tick, player.player_id, float(player.pos[0]), float(player.pos[1]))
    if _WAIT.get("key") == key:
        return _WAIT["value"]
    team = state.get_team(player.team_id)
    goal_x = PITCH_LENGTH if team.direction == 1 else 0.0

    def later(p):
        pos = p.pos + p.velocity * PACE_WAIT_TICKS
        return np.array([min(max(float(pos[0]), 1.0), PITCH_LENGTH - 1.0),
                         min(max(float(pos[1]), 1.0), PITCH_WIDTH - 1.0)])

    mates = [m for m in state.teammates_of(player.player_id) if m.position != "GK"]
    best = 0.0
    if mates:
        pts = np.array([later(m) for m in mates])
        gaps = np.hypot(pts[:, 0] - player.pos[0], pts[:, 1] - player.pos[1])
        keep = gaps >= MIN_PASS_DIST
        if keep.any():
            pts = pts[keep]
            moved = [SimpleNamespace(pos=later(o), velocity=o.velocity, speed=o.speed,
                                     position=o.position, goalkeeping=o.goalkeeping,
                                     player_id=o.player_id) for o in opponents]
            paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps[keep]])
            lanes = route_survival_many(player.pos, pts, paces, BALL_FRICTION_COEFF, moved)
            for pt, lane in zip(pts, lanes):
                v = (position_value(pt, goal_x) * PASS_CONTINUATION_DISCOUNT
                     * penetration_value(player.pos, pt, moved, goal_x))
                best = max(best, float(lane) * v)
    _WAIT.update(key=key, value=best)
    return best


def wait_gain(player, state, opponents) -> float:
    """The walk's V multiplier: max(1, PACE_WAIT_DISCOUNT ** Tempo x wait_value / V_stay),
    V_stay = position_value(where he stands) x PASS_CONTINUATION_DISCOUNT x
    DRIBBLE_CONTINUATION_DISCOUNT -- the ball kept where it is. The same
    factor for every walk direction, so it decides walk against run and pass,
    and the direction is still chosen on its own V."""
    from ..constants import DRIBBLE_CONTINUATION_DISCOUNT, PACE_WAIT_DISCOUNT
    from .tunables import tempo_bias
    from .value import position_value

    team = state.get_team(player.team_id)
    goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    stay = (position_value(player.pos, goal_x) * PASS_CONTINUATION_DISCOUNT
            * DRIBBLE_CONTINUATION_DISCOUNT)
    discount = PACE_WAIT_DISCOUNT ** tempo_bias(player, team)
    return max(1.0, discount * wait_value(player, state, opponents) / max(stay, 1e-9))


def progress_factor(forward: float, tempo: float) -> float:
    """Tempo in pass choices: Tempo ** (TEMPO_PASS_EXPONENT x clamp(forward /
    TEMPO_PROGRESS_M, -1, 1)), `forward` the metres the pass moves the ball
    toward goal. A forward ball gains with Tempo, a back pass by the
    reciprocal, a square ball not at all; 1.0 for every pass at Tempo 1."""
    from ..constants import TEMPO_PASS_EXPONENT, TEMPO_PROGRESS_M
    if tempo == 1.0 or tempo <= 0.0:
        return 1.0
    x = min(1.0, max(-1.0, forward / TEMPO_PROGRESS_M))
    return float(tempo ** (TEMPO_PASS_EXPONENT * x))
