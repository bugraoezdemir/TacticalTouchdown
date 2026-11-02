"""
Press aggression and Press numbers (decision D-16).

Press aggression: how the man going at the carrier approaches him: from straight
at him at 2.0 to fully on the lane of the carrier's likeliest pass at 0.5.
Press numbers (the Pressing setting): who goes, graded, not in steps.
"""
import numpy as np
import pytest

from app.engine.decisions import off_ball
from app.engine.decisions.off_ball import _may_leave_zone, _press_lane_spot, press_lane_share
from app.engine.decisions.tunables import press_aggression_bias
from app.engine.factory import build_test_state


def test_the_share_on_the_lane_is_graded_over_the_range():
    shares = [press_lane_share(a) for a in (0.5, 0.75, 1.0, 1.5, 2.0)]
    assert shares[0] == pytest.approx(1.0) and shares[2] == pytest.approx(0.5) and shares[4] == pytest.approx(0.0)
    assert shares == sorted(shares, reverse=True)


def test_it_has_its_own_setting_and_override():
    state = build_test_state(seed=1)
    team = state.home_team
    player = next(p for p in team.players if p.position == "CM")
    assert press_aggression_bias(player, team) == 1.0
    team.tactics.press_aggression = 1.6
    assert press_aggression_bias(player, team) == pytest.approx(1.6)
    player.bias_press_aggression = 0.7
    assert press_aggression_bias(player, team) == pytest.approx(0.7)
    player.bias_press_aggression = 99.0
    assert press_aggression_bias(player, team) == 2.0


def _carrier_scene():
    state = build_test_state(seed=1)
    carrier = next(p for p in state.away_team.players if p.position == "CM")
    defender = next(p for p in state.home_team.players if p.position == "CM")
    for p in state.all_players():
        p.velocity = np.zeros(2)
    carrier.pos = np.array([50.0, 34.0])
    for p in state.away_team.players:
        if p is not carrier and p.position != "GK":
            p.pos = np.array([95.0, 5.0 + 6.0 * (p.player_id % 10)])     # nobody for him to mark
    return state, carrier, defender


def test_the_spot_is_on_the_lane_and_clear_of_the_carrier(monkeypatch):
    state, carrier, defender = _carrier_scene()
    outlet = np.array([30.0, 34.0])
    monkeypatch.setattr(off_ball, "_best_outlet", lambda *a: outlet)
    defender.pos = np.array([40.0, 44.0])
    assert _press_lane_spot(defender, carrier, state, 0.0) == pytest.approx(np.array([40.0, 34.0]))
    defender.pos = np.array([60.0, 34.0])                   # behind the carrier
    assert _press_lane_spot(defender, carrier, state, 0.0) == pytest.approx(np.array([48.0, 34.0]))
    defender.pos = np.array([10.0, 34.0])                   # beyond the outlet
    assert _press_lane_spot(defender, carrier, state, 0.0) == pytest.approx(outlet)


def _may(pressing, t_arrive, engaged, danger=0.0):
    state, carrier, defender = _carrier_scene()
    team = state.home_team
    team.tactics.pressing = pressing
    others = [p.player_id for p in team.players if p.position != "GK" and p is not defender][:engaged]
    return _may_leave_zone(defender, state, team, carrier, 0.0, danger, np.array([50.0, 34.0]), t_arrive, others)


def test_the_first_man_always_goes_from_neutral_up():
    assert _may(1.0, 500.0, 0) and _may(2.0, 500.0, 0)


def test_below_neutral_the_first_man_must_be_near_enough():
    assert _may(0.5, 25.0, 0) and not _may(0.5, 35.0, 0)        # 1 s at 0.5
    assert _may(0.75, 85.0, 0) and not _may(0.75, 95.0, 0)      # 3 s at 0.75
    assert _may(0.5, 500.0, 0, danger=1.0)                      # near our goal he goes


def test_extra_pressers_are_graded_not_stepped():
    """No second man at 1.0, one only if near at 1.25, one at any range from 1.5."""
    assert not _may(1.0, 5.0, 1)
    assert _may(1.25, 25.0, 1) and not _may(1.25, 35.0, 1)
    assert not _may(1.25, 25.0, 2)
