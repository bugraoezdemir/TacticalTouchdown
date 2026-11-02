"""The context table must stay a tilt, never the thing doing the deciding."""
import pytest

from app.engine.decisions.tunables import (
    MULTIPLIER_MAX,
    MULTIPLIER_MIN,
    ROLE_ACTION_MULT,
    TACTIC_ACTION_MULT,
    ZONE_ACTION_MULT,
    context_multiplier,
)
from app.engine.decisions.zones import Zone

ACTIONS = ("shot", "pass", "dribble", "hold", "move")


@pytest.mark.parametrize("table,label", [
    (ZONE_ACTION_MULT, "zone"), (ROLE_ACTION_MULT, "role"), (TACTIC_ACTION_MULT, "tactic"),
])
def test_multipliers_stay_in_band(table, label):
    """If a multiplier needs to be 3.0 for sane behaviour, the value model is
    wrong -- fix V rather than inflating the tilt."""
    for key, row in table.items():
        for action, value in row.items():
            assert MULTIPLIER_MIN <= value <= MULTIPLIER_MAX, f"{label}[{key}][{action}]={value}"


def test_every_zone_and_role_covered():
    for zone in Zone:
        assert set(ZONE_ACTION_MULT[zone]) == set(ACTIONS)
    for pos in ("GK", "LD", "CD", "RD", "LM", "CM", "RM", "LF", "CF", "RF"):
        assert set(ROLE_ACTION_MULT[pos]) == set(ACTIONS)
    for tac in ("defensive", "normal", "offensive"):
        assert set(TACTIC_ACTION_MULT[tac]) == set(ACTIONS)


def test_combined_multiplier_is_bounded():
    """Three tilts multiply, so the compound range is what actually matters."""
    for zone in Zone:
        for pos in ROLE_ACTION_MULT:
            for tac in TACTIC_ACTION_MULT:
                for action in ACTIONS:
                    m = context_multiplier(zone, pos, tac, action)
                    assert 0.3 <= m <= 2.5, f"{zone}/{pos}/{tac}/{action} = {m}"


def test_striker_shoots_more_than_centre_back():
    assert (context_multiplier(Zone.ATTACKING_CENTRAL, "CF", "normal", "shot")
            > context_multiplier(Zone.ATTACKING_CENTRAL, "CD", "normal", "shot"))


def test_own_box_discourages_dribbling():
    assert (context_multiplier(Zone.DEFENSIVE_CENTRAL, "CD", "normal", "dribble")
            < context_multiplier(Zone.MIDFIELD_CENTRAL, "CD", "normal", "dribble"))


def test_unknown_keys_fall_back_to_neutral():
    assert context_multiplier(Zone.MIDFIELD_CENTRAL, "XX", "normal", "shot") > 0.0
    assert context_multiplier(Zone.MIDFIELD_CENTRAL, "CM", "normal", "nonsense") > 0.0
