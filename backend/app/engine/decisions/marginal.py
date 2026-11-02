"""
Marginal contribution: how much do *I* add to what my team already achieves?

Without this, every player computes their own chance of winning the ball in
isolation, so five teammates all "correctly" converge on the same loose ball
and a striker sprints 40 m back to help with an interception four defenders
already have covered.

    marginal = P_team_with_me - P_team_without_me
             = prod_{j committed}(1 - r_j) * r_me

The factored form on the right is exact (verified to 1.4e-16) and costs one
multiply, where the literal subtraction costs a second full team evaluation.

Conditioning on *predecessors only* is what prevents deadlock. If everyone
conditions on everyone, four symmetric defenders each compute the same small
marginal and all defer -- nobody goes. Processed in a fixed order, the first
player faces an empty committed set, so somebody always goes by construction.
"""
from __future__ import annotations

import math

import numpy as np

from .. import kinematics
from ..constants import REACTION_TICKS, RISK_TAU
from ..kinematics import reach_ticks
from ..models import GameState, Player


def _reach_probability(dist: float, speed: float, t_available: float) -> float:
    """P(a player covering `dist` at `speed` arrives within `t_available`)."""
    if speed <= 1e-9 or t_available == float("inf"):
        return 0.0
    t_needed = REACTION_TICKS + dist / speed
    return _arrives_in_time(t_needed, t_available)


def _arrives_in_time(t_needed: float, t_available: float) -> float:
    return 1.0 / (1.0 + math.exp(-(t_available - t_needed) / RISK_TAU))


def _inertial_risk(player: Player, point: np.ndarray, t_available: float) -> float:
    """Under inertia: his time from how he is actually moving, no reaction
    delay -- the momentum the old version approximated is in his velocity."""
    if player.speed <= 1e-9 or t_available == float("inf"):
        return 0.0
    t_needed = float(reach_ticks(player.pos, player.velocity, player.speed, point))
    return _arrives_in_time(t_needed, t_available)


def player_risk_at(player: Player, point: np.ndarray, t_available: float) -> float:
    """P(this player reaches `point` within `t_available` ticks). In [0, 1]."""
    if kinematics.INERTIA:
        return _inertial_risk(player, point, t_available)
    return _reach_probability(
        float(np.linalg.norm(point - player.pos)), player.speed, t_available
    )


def marginal_contribution(
    me: Player,
    point: np.ndarray,
    t_available: float,
    committed: list[tuple[int, np.ndarray]],
    state: GameState,
) -> tuple[float, float]:
    """Return (marginal, cover) for me contesting `point`.

    `committed` is [(player_id, chosen_target)] for teammates who already
    decided this tick. Their risk is measured from a position advanced along the
    path they committed to, not from where they currently stand -- a teammate
    already running at the ball covers it better than their current position
    suggests.

    `cover` = P(the team wins it without me), which is exactly the "teammate
    cover" term the pursuit model needs, so it is computed once and used twice.
    """
    survive = 1.0
    for pid, target in committed:
        if pid == me.player_id:
            continue
        try:
            mate = state.get_player(pid)
        except KeyError:
            continue
        if mate.speed <= 1e-9:
            continue

        if kinematics.INERTIA:
            survive *= 1.0 - _inertial_risk(mate, point, t_available)
            if survive < 1e-6:
                break
            continue

        # Advance them along their committed path by however far they can run.
        to_target = target - mate.pos
        d = float(np.linalg.norm(to_target))
        if d > 1e-9 and t_available != float("inf"):
            step = min(d, mate.speed * t_available)
            advanced = mate.pos + to_target * (step / d)
        else:
            advanced = mate.pos

        remaining = float(np.linalg.norm(point - advanced))
        survive *= 1.0 - _reach_probability(remaining, mate.speed, t_available)
        if survive < 1e-6:
            break

    r_me = player_risk_at(me, point, t_available)
    return survive * r_me, 1.0 - survive
