"""
Crosses into the box roles, and a defender's clearing header.
"""
import numpy as np

from app.engine.factory import build_test_state
from app.engine.decisions.candidates import build_candidates
from app.engine.decisions.on_ball import _clearing_headers


def _wide_carrier(state, y):
    team = state.home_team
    carrier = next(p for p in team.players if p.position == "RM")
    x = 88.0 if team.direction == 1 else 105.0 - 88.0
    for p in state.all_players():
        p.has_ball = False
    carrier.pos = np.array([x, y])
    carrier.has_ball = True
    state.ball.pos = carrier.pos.copy()
    state.ball.owner_id = carrier.player_id
    return carrier


def test_a_wide_carrier_in_the_final_third_is_offered_crosses():
    state = build_test_state()
    carrier = _wide_carrier(state, 60.0)
    crosses = [c for c in build_candidates(carrier, state) if c.is_cross]
    assert crosses
    assert any(c.vz > 0 for c in crosses) and any(c.vz == 0 for c in crosses)
    assert all(c.is_runway and c.recipient_id is not None for c in crosses)


def test_no_crosses_from_the_middle_or_the_half_space():
    """A cross comes from the flank: outside the width of the penalty area."""
    state = build_test_state()
    for y in (34.0, 48.0):
        carrier = _wide_carrier(state, y)
        assert not any(c.is_cross for c in build_candidates(carrier, state))


def _scored(state, carrier):
    from app.engine.decisions.on_ball import score_candidate
    return [score_candidate(c, carrier, state) for c in build_candidates(carrier, state)]


def test_crossing_slider_scales_crosses_only():
    state = build_test_state()
    carrier = _wide_carrier(state, 60.0)
    base = _scored(state, carrier)
    state.home_team.tactics.crossing = 2.0
    keen = _scored(state, carrier)
    for a, b in zip(base, keen):
        if a.is_cross:
            assert np.isclose(b.context, 2.0 * a.context)
        else:
            assert np.isclose(b.context, a.context)


def test_lofted_slider_trades_air_against_ground():
    state = build_test_state()
    carrier = _wide_carrier(state, 60.0)
    base = _scored(state, carrier)
    state.home_team.tactics.lofted = 1.6
    airy = _scored(state, carrier)
    for a, b in zip(base, airy):
        if a.kind != "pass":
            assert np.isclose(b.context, a.context)
        elif a.vz > 0:
            assert np.isclose(b.context, 1.6 * a.context)
        else:
            assert np.isclose(b.context, a.context / 1.6)


def test_a_player_override_beats_the_team_setting():
    state = build_test_state()
    carrier = _wide_carrier(state, 60.0)
    base = _scored(state, carrier)
    state.home_team.tactics.crossing = 0.5
    carrier.bias_crossing = 1.0
    same = _scored(state, carrier)
    assert all(np.isclose(a.context, b.context) for a, b in zip(base, same))


def test_clearing_headers_only_in_his_own_half():
    state = build_test_state()
    team = state.home_team
    defender = next(p for p in team.players if p.position == "CD")
    own_goal_x = 0.0 if team.direction == 1 else 105.0
    defender.pos = np.array([abs(own_goal_x - 10.0), 20.0])
    clears = _clearing_headers(defender, state, team, [])
    assert len(clears) == 2                     # into touch, and behind for a corner
    assert all(c.is_clearance and c.recipient_id is None for c in clears)
    defender.pos = np.array([abs(own_goal_x - 80.0), 20.0])
    assert _clearing_headers(defender, state, team, []) == []
