"""Per-player tactical rules.

Thresholds here are MEASURED, not chosen: they are the p75 of 120 outfield
player-matches (interception 0.22, dribble loss 0.28). These tests pin that the
rules fire where the data says they should, and -- just as important -- that
they stay quiet on ordinary performances and thin samples.
"""
from __future__ import annotations

import pytest

from app.engine.models import PlayerStats
from app.engine.player_rules import (
    BIAS_MAX,
    BIAS_MIN,
    DECAY,
    DRIBBLE_LOSS_THRESHOLD,
    INTERCEPTION_THRESHOLD,
    MIN_DRIBBLES,
    MIN_PASSES,
    MIN_SHOTS,
    RULES,
    STEP,
    evaluate,
)


# -- Calibration --------------------------------------------------------------

def test_thresholds_match_the_measured_p75():
    """If these drift, the rules fire on a different share of the league than
    intended. Measured over 120 outfield player-matches."""
    assert INTERCEPTION_THRESHOLD == pytest.approx(0.22)
    assert DRIBBLE_LOSS_THRESHOLD == pytest.approx(0.28)


def test_minimum_samples_suit_a_recent_slice_not_a_season():
    """Rules judge the slice since the last evaluation, not the season total.

    A busy player makes ~50 passes and ~53 dribbles a match, so 12 of either is
    a few windows' worth: enough to mean something, small enough that a rule
    can still act within a match. Season-sized minimums would never be met by a
    slice, and the rules would go silent.
    """
    assert MIN_PASSES == 12
    assert MIN_DRIBBLES == 12
    # Shots are the exception and stay season-long: only 35 of 120
    # player-matches contain even one, median 2 among those. No recent slice
    # could ever carry 15 shots, so that rule reads the cumulative total.
    assert MIN_SHOTS == 15


# -- Quiet on ordinary performance --------------------------------------------

def test_a_median_player_triggers_nothing():
    """Median interception 0.19 and dribble loss 0.20 are both below threshold."""
    median = PlayerStats(
        passes=50, passes_completed=40, passes_intercepted=9,
        dribbles=53, dribbles_lost=11,
    )
    assert median.interception_rate < INTERCEPTION_THRESHOLD
    assert median.dribble_loss_rate < DRIBBLE_LOSS_THRESHOLD
    biases: dict[str, float] = {}
    assert evaluate(median, 5, biases) == []
    assert biases == {}


@pytest.mark.parametrize(
    "stats",
    [
        PlayerStats(passes=10, passes_intercepted=8),      # 0.80 but n=10
        PlayerStats(dribbles=10, dribbles_lost=9),         # 0.90 but n=10
        PlayerStats(shots=3, goals=0),                     # 0.00 but n=3
    ],
)
def test_a_thin_sample_never_fires(stats):
    """A terrible rate on a handful of events is luck, not a problem."""
    biases: dict[str, float] = {}
    assert evaluate(stats, 5, biases) == []


# -- Firing where the data says ------------------------------------------------

def test_a_player_whose_passes_are_cut_out_shortens_his_range():
    stats = PlayerStats(passes=60, passes_completed=35, passes_intercepted=18)
    assert stats.interception_rate > INTERCEPTION_THRESHOLD
    biases: dict[str, float] = {}
    firings = evaluate(stats, 7, biases)

    assert {f.rule for f in firings} == {"passes_cut_out"}
    assert biases["bias_pass_range"] < 1.0      # shorter balls
    assert biases["bias_dribble"] > 1.0         # carry when lanes are shut


def test_a_player_losing_dribbles_passes_more():
    stats = PlayerStats(dribbles=60, dribbles_lost=25)
    assert stats.dribble_loss_rate > DRIBBLE_LOSS_THRESHOLD
    biases: dict[str, float] = {}
    firings = evaluate(stats, 9, biases)

    assert {f.rule for f in firings} == {"dribbles_lost"}
    assert biases["bias_dribble"] < 1.0
    assert biases["bias_pass"] > 1.0


def test_a_wasteful_finisher_shoots_less_but_only_with_enough_shots():
    wasteful = PlayerStats(shots=MIN_SHOTS, goals=0)
    biases: dict[str, float] = {}
    firings = evaluate(wasteful, 10, biases)
    assert {f.rule for f in firings} == {"shots_not_scoring"}
    assert biases["bias_shoot"] < 1.0

    one_short = PlayerStats(shots=MIN_SHOTS - 1, goals=0)
    biases2: dict[str, float] = {}
    assert evaluate(one_short, 10, biases2) == []


def test_two_problems_fire_two_rules():
    stats = PlayerStats(
        passes=60, passes_completed=35, passes_intercepted=18,
        dribbles=60, dribbles_lost=25,
    )
    firings = evaluate(stats, 4, {})
    assert {f.rule for f in firings} == {"passes_cut_out", "dribbles_lost"}


# -- Steps stay small and reversible -------------------------------------------

def test_one_firing_moves_a_bias_by_one_step():
    stats = PlayerStats(dribbles=60, dribbles_lost=25)
    biases: dict[str, float] = {}
    evaluate(stats, 3, biases)
    # A full-weight adjustment is exactly STEP; decay then pulls it back a
    # little, so the net move is smaller than a step but in the right direction.
    assert 1.0 - STEP - 1e-9 <= biases["bias_dribble"] < 1.0


def test_repeated_firing_compounds_but_stays_clamped():
    stats = PlayerStats(dribbles=60, dribbles_lost=40)
    biases: dict[str, float] = {}
    for _ in range(200):
        evaluate(stats, 3, biases)
    assert BIAS_MIN <= biases["bias_dribble"] <= BIAS_MAX
    assert biases["bias_dribble"] == pytest.approx(BIAS_MIN)


def test_decay_returns_a_recovered_player_to_base():
    """An adjustment is a claim that has to keep being re-earned."""
    biases = {"bias_dribble": 1.30}
    clean = PlayerStats()
    for _ in range(100):
        evaluate(clean, 3, biases)
        if "bias_dribble" not in biases:
            break
    assert "bias_dribble" not in biases, "override never decayed away"


def test_decay_drops_the_override_entirely_when_it_arrives_home():
    """A recovered player should not carry a pointless entry forever."""
    biases = {"bias_pass": 1.0 + DECAY / 2.0}
    evaluate(PlayerStats(), 3, biases)
    assert biases == {}


# -- Structure -----------------------------------------------------------------

def test_every_rule_adjusts_a_real_bias_field():
    """A typo here is a rule that silently does nothing."""
    from app.engine.models import Player
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Player)}
    for rule in RULES:
        for field_name in rule.adjust:
            assert field_name in fields, f"{rule.name} adjusts unknown {field_name}"


def test_every_rule_explains_itself():
    """The `why` is shown to the user; an empty one is a silent change."""
    for rule in RULES:
        assert rule.why.strip()
        assert rule.name.strip()
