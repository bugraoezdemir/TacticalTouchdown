"""
Pitch zones -- the six regions from the design doc.

These exist so context multipliers can be looked up as data. The decision tree
in the design doc lives here and in tunables.py as a table, not as control flow:
the tree is good at saying "be cautious near your own goal" and bad at being
1000 hand-written branches.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from ..constants import GOAL_CENTER_Y, PITCH_LENGTH, ZONE_CENTRAL_HALF_WIDTH


class Zone(str, Enum):
    ATTACKING_CENTRAL = "attacking_central"
    ATTACKING_WING = "attacking_wing"
    MIDFIELD_CENTRAL = "midfield_central"
    MIDFIELD_WING = "midfield_wing"
    DEFENSIVE_CENTRAL = "defensive_central"
    DEFENSIVE_WING = "defensive_wing"


_THIRD = PITCH_LENGTH / 3.0


def classify_zone(pos: np.ndarray, attacking_goal_x: float) -> Zone:
    """Which of the six zones `pos` falls in, from the perspective of a team
    attacking toward `attacking_goal_x`.

    Direction-aware: the same absolute coordinate is the attacking third for one
    team and the defensive third for the other.
    """
    own_goal_x = PITCH_LENGTH - attacking_goal_x
    # Distance upfield from our own goal, regardless of which way we attack.
    progress = abs(float(pos[0]) - own_goal_x)

    central = abs(float(pos[1]) - GOAL_CENTER_Y) <= ZONE_CENTRAL_HALF_WIDTH

    if progress >= 2.0 * _THIRD:
        return Zone.ATTACKING_CENTRAL if central else Zone.ATTACKING_WING
    if progress >= _THIRD:
        return Zone.MIDFIELD_CENTRAL if central else Zone.MIDFIELD_WING
    return Zone.DEFENSIVE_CENTRAL if central else Zone.DEFENSIVE_WING
