"""
Team shapes.

A formation is 11 slots, each a (position, x, y) for a team attacking rightward.
Slots drive `Player.assigned_pos`, which anchors the off-ball shape leash in
`off_ball._shape_anchor` -- so changing formation changes where the whole team
lives without touching any decision code.

Position strings deliberately REPEAT within a formation: a 4-4-2 has two "CD"
and two "CF". Every lookup keyed on position is a `.get(position, default)`
(ROLE_LEASH, ROLE_ACTION_MULT) or a first-match scan (_find_gk, _find_position),
so duplicates share a value -- which is what you want, since two centre-backs
should have the same leash. Player identity is carried by `player_id`, not by
the position string. A tuple of triples rather than a dict, because a dict
cannot hold two "CD" keys.

Depths follow the original hand-tuned shape (GK 5, defence 22, midfield 45,
attack 70) so that SHAPE_BLOCK_SLIDE_* and ROLE_LEASH stay calibrated.
"""
from __future__ import annotations

import numpy as np

from .constants import (
    GK_DEPTH_LIMIT,
    PITCH_LENGTH,
    PITCH_WIDTH,
    STATION_BREADTH_GAIN,
    STATION_BREADTH_MARGIN,
    STATION_DEPTH_GAIN,
    STATION_DEPTH_MARGIN,
    STATION_COMPACT_GAIN,
)
from .models import Team

# name -> 11 slots of (position, x, y), for a team attacking rightward.
FORMATIONS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "4-4-2": (
        ("GK",  5.0, 34.0),
        ("LD", 22.0, 12.0), ("CD", 22.0, 27.0), ("CD", 22.0, 41.0), ("RD", 22.0, 56.0),
        ("LM", 45.0, 12.0), ("CM", 45.0, 27.0), ("CM", 45.0, 41.0), ("RM", 45.0, 56.0),
        ("CF", 70.0, 27.0), ("CF", 70.0, 41.0),
    ),
    "4-3-3": (
        ("GK",  5.0, 34.0),
        ("LD", 22.0, 12.0), ("CD", 22.0, 27.0), ("CD", 22.0, 41.0), ("RD", 22.0, 56.0),
        ("CM", 45.0, 20.0), ("CM", 45.0, 34.0), ("CM", 45.0, 48.0),
        ("LF", 70.0, 14.0), ("CF", 70.0, 34.0), ("RF", 70.0, 54.0),
    ),
    "4-2-3-1": (
        ("GK",  5.0, 34.0),
        ("LD", 22.0, 12.0), ("CD", 22.0, 27.0), ("CD", 22.0, 41.0), ("RD", 22.0, 56.0),
        ("CM", 38.0, 26.0), ("CM", 38.0, 42.0),
        ("LM", 56.0, 14.0), ("CM", 56.0, 34.0), ("RM", 56.0, 54.0),
        ("CF", 72.0, 34.0),
    ),
    "3-5-2": (
        ("GK",  5.0, 34.0),
        ("CD", 20.0, 20.0), ("CD", 20.0, 34.0), ("CD", 20.0, 48.0),
        ("LM", 42.0,  9.0), ("CM", 45.0, 24.0), ("CM", 45.0, 34.0),
        ("CM", 45.0, 44.0), ("RM", 42.0, 59.0),
        ("CF", 70.0, 27.0), ("CF", 70.0, 41.0),
    ),
    "5-3-2": (
        ("GK",  5.0, 34.0),
        ("LD", 20.0, 10.0), ("CD", 20.0, 24.0), ("CD", 20.0, 34.0),
        ("CD", 20.0, 44.0), ("RD", 20.0, 58.0),
        ("CM", 45.0, 22.0), ("CM", 45.0, 34.0), ("CM", 45.0, 46.0),
        ("CF", 70.0, 27.0), ("CF", 70.0, 41.0),
    ),
}

DEFAULT_FORMATION = "4-4-2"

SQUAD_SIZE = 11


def slot_position(
    x: float,
    y: float,
    direction: int,
    offensiveness: float = 1.0,
    breadth: float = 1.0,
    compactness: float = 1.0,
    centre: tuple[float, float] | None = None,
) -> np.ndarray:
    """A slot's absolute pitch position for a team attacking `direction`.

    Rotates the shape 180 degrees for a team attacking leftward -- BOTH axes.
    Mirroring x alone leaves a left-back at the same absolute y, which is that
    team's RIGHT touchline.

    `offensiveness` pushes the whole shape up or down the pitch: above 1.0 every
    station moves toward the opponent goal, below it the team sits deeper. The
    keeper is not moved -- a shifted keeper is an empty net, not a tactic -- and
    the shift is applied before mirroring so both sides behave identically.

    `breadth` does the same on the lateral axis: above 1.0 the shape spreads
    toward the touchlines, below it the team packs centrally. Both shifts are
    applied before mirroring, so a wide team is wide attacking either way.
    """
    x, y = _contract(x, y, compactness, centre)
    x = _shift_depth(x, offensiveness)
    y = _spread_width(y, breadth)
    if direction == -1:
        return np.array([PITCH_LENGTH - x, PITCH_WIDTH - y], dtype=float)
    return np.array([x, y], dtype=float)


def _contract(
    x: float, y: float, compactness: float, centre: tuple[float, float] | None
) -> tuple[float, float]:
    """Draw a station toward the centre of the outfield block, on both axes.

    Applied to the formation's own shape BEFORE depth and breadth, so a compact
    team is compact whatever its line height or width. The keeper is left on
    his line. Damped by a power so the extremes stay football shapes: at 2.0
    the block is ~0.66 of its size, at 0.5 ~1.5x.
    """
    if centre is None or x <= GK_DEPTH_LIMIT or abs(compactness - 1.0) < 1e-9:
        return x, y
    k = compactness ** -STATION_COMPACT_GAIN
    return centre[0] + (x - centre[0]) * k, centre[1] + (y - centre[1]) * k


def _shift_depth(x: float, offensiveness: float) -> float:
    """Move a station toward (or away from) the opponent goal.

    Scales the distance from the team's own goal line, so deep players move
    least and advanced ones most -- a high line compresses toward the halfway
    line rather than translating the whole block rigidly.
    """
    if x <= GK_DEPTH_LIMIT:
        return x  # the keeper stays on his line
    # Damped rather than a raw multiply: at 1.4x a raw scale puts strikers on
    # the goal line at x=98, which is goal-hanging, not a high line.
    shifted = x + (x - GK_DEPTH_LIMIT) * (offensiveness - 1.0) * STATION_DEPTH_GAIN
    return float(min(PITCH_LENGTH - STATION_DEPTH_MARGIN, max(GK_DEPTH_LIMIT, shifted)))


def _spread_width(y: float, breadth: float) -> float:
    """Move a station toward (or away from) the touchlines.

    Scales the distance from the centre line, so central players barely move
    and wide ones move most -- narrowing pulls the flanks in rather than
    translating everyone sideways, which is what actually shortens the gaps
    between teammates and makes short passing viable.

    Damped and clamped for the same reason as the depth shift: at 2.0 a raw
    scale puts wingers past the touchline, which is not a tactic.
    """
    half = PITCH_WIDTH / 2.0
    spread = y + (y - half) * (breadth - 1.0) * STATION_BREADTH_GAIN
    return float(
        min(PITCH_WIDTH - STATION_BREADTH_MARGIN, max(STATION_BREADTH_MARGIN, spread))
    )


def apply_formation(team: Team, name: str | None = None) -> None:
    """Re-station a team into the named formation.

    Rewrites `position` and `assigned_pos` for all 11 players. Deliberately does
    NOT touch `player.pos` or `has_ball`: players run to their new stations over
    the following ticks rather than teleporting, and a carrier keeps the ball.

    Must only be called BETWEEN ticks (e.g. from the tactics endpoint), never
    from inside `game_loop.tick`, which would put a write in the decision phase
    and break the frozen-snapshot guarantee.
    """
    from .decisions.tunables import breadth_bias, compactness_bias, offensiveness_bias

    name = name or team.tactics.formation
    slots = FORMATIONS[name]
    outfield = [(x, y) for _, x, y in slots if x > GK_DEPTH_LIMIT]
    centre = (
        sum(x for x, _ in outfield) / len(outfield),
        sum(y for _, y in outfield) / len(outfield),
    )
    for player, (position, x, y) in zip(team.players, slots):
        player.position = position  # type: ignore[assignment]
        # Resolved per player: a marauding full-back or a deep-lying forward is
        # a real instruction, and the team value is only the default.
        player.assigned_pos = slot_position(
            x, y, team.direction,
            offensiveness_bias(player, team), breadth_bias(player, team),
            compactness_bias(player, team), centre,
        )
        # The sticky bonus in off_ball favours points near last tick's target.
        # After a shape change that target belongs to the old formation and
        # would fight the new anchor for several ticks.
        player.last_off_ball_target = None
    team.tactics.formation = name
