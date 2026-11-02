"""
The value model decides what is *worth* doing, which is the thing the engine
previously had no opinion about. These assert shape, not fitted precision.
"""
import numpy as np
import pytest

from app.engine.constants import GOAL_CENTER_Y, LEFT_GOAL_X, PITCH_LENGTH, RIGHT_GOAL_X
from app.engine.decisions.value import (
    position_value,
    progression_value,
    shot_value,
    threat,
)

RIGHT = RIGHT_GOAL_X


def P(x, y=GOAL_CENTER_Y):
    return np.array([float(x), float(y)])


# ── xG anchors, from real football ───────────────────────────────────────────

@pytest.mark.parametrize("name,pos,expected", [
    ("six yard box",  P(99.0), 0.90),
    ("penalty spot",  P(94.0), 0.76),
    ("18 yard line",  P(88.5), 0.20),
    ("25 metres",     P(80.0), 0.06),
    ("tight wing",    P(88.0, 12.0), 0.03),
])
def test_shot_value_anchors(name, pos, expected):
    assert shot_value(pos, RIGHT) == pytest.approx(expected, abs=0.08), name


def test_shot_value_bounded():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        p = P(rng.uniform(0, PITCH_LENGTH), rng.uniform(0, 68))
        v = shot_value(p, RIGHT)
        assert 0.0 <= v <= 1.0 and np.isfinite(v)


def test_shot_value_decreases_with_distance():
    prev = 2.0
    for x in (100.0, 95.0, 90.0, 85.0, 75.0, 60.0):
        v = shot_value(P(x), RIGHT)
        assert v < prev
        prev = v


def test_central_beats_wide_at_equal_distance():
    """Same radius from goal centre, different angle."""
    central = P(90.0, 34.0)
    wide = P(99.6148, 34.0 - 14.0)
    assert np.linalg.norm(central - P(RIGHT)) == pytest.approx(
        np.linalg.norm(wide - P(RIGHT)), abs=1.0)
    assert shot_value(central, RIGHT) > shot_value(wide, RIGHT)


def test_35m_shot_is_worthless():
    """The direct cause of the old engine's 34.6 m mean shot distance."""
    assert shot_value(P(70.0), RIGHT) < 0.02


# ── Direction awareness ──────────────────────────────────────────────────────

def test_mirror_symmetry():
    """Value for a team attacking right at x equals that for a team attacking
    left at 105-x. One surface, two directions."""
    for x in (20.0, 52.5, 88.0, 99.0):
        for y in (10.0, 34.0, 58.0):
            assert shot_value(P(x, y), RIGHT) == pytest.approx(
                shot_value(P(PITCH_LENGTH - x, y), LEFT_GOAL_X), abs=1e-9)
            assert position_value(P(x, y), RIGHT) == pytest.approx(
                position_value(P(PITCH_LENGTH - x, y), LEFT_GOAL_X), abs=1e-9)


def test_threat_is_position_value_mirrored():
    """The defensive surface must BE the attacking one, not a parallel model."""
    for x in (5.0, 16.5, 35.0, 52.5, 90.0):
        assert threat(P(x), LEFT_GOAL_X) == pytest.approx(
            position_value(P(x), LEFT_GOAL_X), abs=1e-12)
    # Danger rises as the ball nears my goal.
    assert threat(P(5.0), LEFT_GOAL_X) > threat(P(35.0), LEFT_GOAL_X)


# ── position_value: the surface off-ball sampling climbs ─────────────────────

def test_position_value_strictly_positive_everywhere():
    """Off-ball sampling needs a gradient everywhere; a flat zero region would
    leave players with no reason to prefer any point."""
    for x in np.arange(0.0, 105.1, 5.0):
        for y in np.arange(0.0, 68.1, 8.5):
            assert position_value(P(x, y), RIGHT) > 0.0


def test_position_value_monotone_toward_goal():
    prev = -1.0
    for x in np.arange(5.0, 100.1, 5.0):
        v = position_value(P(x), RIGHT)
        assert v > prev
        prev = v


def test_position_value_bounded():
    rng = np.random.default_rng(11)
    for _ in range(2000):
        v = position_value(P(rng.uniform(0, 105), rng.uniform(0, 68)), RIGHT)
        assert 0.0 <= v <= 1.0


def test_progression_accelerates():
    """A convex progression term keeps midfield from being as valuable as the
    final third; a linear one made every forward pass look equally good."""
    near = progression_value(P(90.0), RIGHT) - progression_value(P(80.0), RIGHT)
    far = progression_value(P(30.0), RIGHT) - progression_value(P(20.0), RIGHT)
    assert near > far
