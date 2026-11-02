"""Pass range and Breadth: the two flavour/shape sliders added for the user.

Both follow the house convention (fb-tactics-sliders-are-bias-not-probability):
a relative multiplier clamped to [0.5, 2.0], team value with a per-player
override, and NEUTRAL AT EXACTLY 1.0 -- which for these two is the load-bearing
property, since the user asked for "middle values balanced, no preference".
"""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.constants import (
    PASS_RANGE_PIVOT_M,
    PITCH_WIDTH,
    TACTIC_BIAS_MAX,
    TACTIC_BIAS_MIN,
)
from app.engine.decisions.on_ball import _pass_range_factor
from app.engine.decisions.tunables import breadth_bias, pass_range_bias
from app.engine.factory import build_test_state
from app.engine.formations import _spread_width, apply_formation


# -- Pass range ---------------------------------------------------------------

def test_neutral_setting_has_no_preference_at_any_distance():
    """The whole point of the middle of the slider."""
    for dist in (2.0, 10.0, PASS_RANGE_PIVOT_M, 30.0, 60.0):
        assert _pass_range_factor(dist, 1.0) == pytest.approx(1.0)


def test_the_pivot_is_neutral_at_every_setting():
    for bias in (TACTIC_BIAS_MIN, 0.8, 1.0, 1.5, TACTIC_BIAS_MAX):
        assert _pass_range_factor(PASS_RANGE_PIVOT_M, bias) == pytest.approx(1.0)


def test_high_setting_prefers_long_and_low_prefers_short():
    short, long_ = 8.0, 40.0
    assert _pass_range_factor(long_, 2.0) > 1.0 > _pass_range_factor(short, 2.0)
    assert _pass_range_factor(short, 0.5) > 1.0 > _pass_range_factor(long_, 0.5)


def test_the_pair_is_reciprocal_so_volume_is_unchanged():
    """A ball as far above the pivot as another is below scores the inverse.

    This is what stops Pass range from acting as a second Pass slider: it moves
    which lengths win, not how attractive passing is overall.
    """
    for ratio in (1.5, 2.0, 3.0):
        near = PASS_RANGE_PIVOT_M / ratio
        far = PASS_RANGE_PIVOT_M * ratio
        for bias in (0.5, 0.75, 1.5, 2.0):
            assert _pass_range_factor(near, bias) * _pass_range_factor(
                far, bias
            ) == pytest.approx(1.0)


def test_monotonic_in_distance_and_in_the_setting():
    lengths = [5.0, 12.0, 20.0, 35.0, 55.0]
    rising = [_pass_range_factor(d, 2.0) for d in lengths]
    assert rising == sorted(rising)
    falling = [_pass_range_factor(d, 0.5) for d in lengths]
    assert falling == sorted(falling, reverse=True)
    at40 = [_pass_range_factor(40.0, b) for b in (0.5, 1.0, 1.5, 2.0)]
    assert at40 == sorted(at40)


def test_degenerate_inputs_are_neutral_not_explosive():
    assert _pass_range_factor(0.0, 2.0) == 1.0
    assert _pass_range_factor(10.0, 0.0) == 1.0


# -- Breadth ------------------------------------------------------------------

def test_the_centre_line_never_moves():
    half = PITCH_WIDTH / 2.0
    for bias in (0.5, 1.0, 2.0):
        assert _spread_width(half, bias) == pytest.approx(half)


def test_wide_spreads_and_narrow_packs():
    wing, half = 8.0, PITCH_WIDTH / 2.0
    assert _spread_width(wing, 2.0) < wing          # closer to the touchline
    assert _spread_width(wing, 0.5) > wing          # pulled infield
    # ...and symmetrically on the far side.
    far = PITCH_WIDTH - wing
    assert _spread_width(far, 2.0) > far
    assert _spread_width(far, 0.5) < far


def test_nobody_is_stationed_off_the_pitch():
    for y in (0.0, 2.0, 8.0, 34.0, 60.0, PITCH_WIDTH):
        for bias in (TACTIC_BIAS_MIN, 1.0, TACTIC_BIAS_MAX):
            out = _spread_width(y, bias)
            assert 0.0 < out < PITCH_WIDTH


def test_breadth_changes_stations_and_narrow_shortens_teammate_gaps():
    def spread_and_gap(bias: float) -> tuple[float, float]:
        state = build_test_state(seed=1)
        team = state.home_team
        team.tactics.breadth = bias
        apply_formation(team)
        outfield = [p for p in team.players if p.position != "GK"]
        ys = [float(p.assigned_pos[1]) for p in outfield]
        spread = float(np.std(ys))
        gaps = [
            float(np.linalg.norm(a.assigned_pos - b.assigned_pos))
            for i, a in enumerate(outfield)
            for b in outfield[i + 1:]
        ]
        return spread, float(np.mean(gaps))

    narrow_spread, narrow_gap = spread_and_gap(0.5)
    neutral_spread, neutral_gap = spread_and_gap(1.0)
    wide_spread, wide_gap = spread_and_gap(2.0)

    assert narrow_spread < neutral_spread < wide_spread
    # The user's stated reason for wanting it: packing them in shortens the
    # distance between teammates, which is what makes short passing viable.
    assert narrow_gap < neutral_gap < wide_gap


def test_breadth_leaves_depth_alone():
    """Breadth is lateral. If it moved x it would silently be a Depth slider."""
    base = build_test_state(seed=1)
    apply_formation(base.home_team)
    before = [float(p.assigned_pos[0]) for p in base.home_team.players]

    wide = build_test_state(seed=1)
    wide.home_team.tactics.breadth = 2.0
    apply_formation(wide.home_team)
    after = [float(p.assigned_pos[0]) for p in wide.home_team.players]

    assert before == pytest.approx(after)


# -- Both, as sliders ---------------------------------------------------------

@pytest.mark.parametrize(
    "bias_fn, team_attr, player_attr",
    [
        (pass_range_bias, "pass_range", "bias_pass_range"),
        (breadth_bias, "breadth", "bias_breadth"),
    ],
)
def test_slider_conventions(bias_fn, team_attr, player_attr):
    state = build_test_state(seed=1)
    team = state.home_team
    player = team.players[1]

    assert bias_fn(player, team) == 1.0                      # neutral default

    setattr(team.tactics, team_attr, 1.4)
    assert bias_fn(player, team) == pytest.approx(1.4)       # team value

    setattr(player, player_attr, 0.7)
    assert bias_fn(player, team) == pytest.approx(0.7)       # override wins

    setattr(player, player_attr, 99.0)
    assert bias_fn(player, team) == TACTIC_BIAS_MAX          # clamped
    setattr(player, player_attr, -99.0)
    assert bias_fn(player, team) == TACTIC_BIAS_MIN
