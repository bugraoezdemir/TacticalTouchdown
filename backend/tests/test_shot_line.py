"""In shooting range the man in front going at a carrier stands in his shot."""
import numpy as np
import pytest

import app.engine.decisions.off_ball as ob
from app.engine.constants import GOAL_SIDE_STANDOFF, SHOT_LINE_FAR, SHOT_LINE_NEAR
from app.engine.factory import build_test_state

GOAL = np.array([105.0, 34.0])
LEAD = np.array([86.0, 37.0])


@pytest.fixture
def scene(monkeypatch):
    """Away defends x = 105; a carrier 20 m out running across, one defender
    near him, everyone else far off. aim_point is pinned to LEAD."""
    state = build_test_state(seed=1)
    carrier = next(p for p in state.home_team.players if p.position != "GK")
    defender = next(p for p in state.away_team.players if p.position != "GK")
    for p in state.home_team.players:
        p.has_ball = False
        if p is not carrier:
            p.pos = np.array([2.0, 2.0])
    carrier.pos = np.array([85.0, 34.0])
    carrier.velocity = np.array([0.0, 0.15])
    carrier.has_ball = True
    state.ball.owner_id = carrier.player_id
    state.ball.pos = carrier.pos.copy()
    for p in state.away_team.players:
        if p is not defender and p.position != "GK":
            p.pos = np.array([2.0, 66.0])
    monkeypatch.setattr(ob, "aim_point", lambda *a: LEAD.copy())
    return state, carrier, defender


def _target(state, carrier, defender):
    return ob._out_of_possession(defender, state, state.away_team, carrier, []).target_pos


def test_the_weight():
    assert ob._shot_line_weight(GOAL - np.array([SHOT_LINE_NEAR - 1.0, 0.0]), 105.0) == 1.0
    assert ob._shot_line_weight(GOAL - np.array([SHOT_LINE_FAR + 1.0, 0.0]), 105.0) == 0.0
    mid = 0.5 * (SHOT_LINE_NEAR + SHOT_LINE_FAR)
    assert ob._shot_line_weight(GOAL - np.array([mid, 0.0]), 105.0) == pytest.approx(0.5)


def test_the_man_in_front_stands_in_the_shot_where_the_carrier_is_going(scene):
    state, carrier, defender = scene
    defender.pos = np.array([89.0, 38.0])
    expected = LEAD + (GOAL - LEAD) / np.linalg.norm(GOAL - LEAD) * GOAL_SIDE_STANDOFF
    assert np.allclose(_target(state, carrier, defender), expected)


def test_a_man_chasing_from_behind_keeps_the_blend(scene, monkeypatch):
    state, carrier, defender = scene
    defender.pos = np.array([80.0, 36.0])
    on = _target(state, carrier, defender)
    monkeypatch.setattr(ob, "SHOT_LINE", False)
    assert np.allclose(on, _target(state, carrier, defender))


def test_the_switch(scene, monkeypatch):
    state, carrier, defender = scene
    defender.pos = np.array([89.0, 38.0])
    on = _target(state, carrier, defender)
    monkeypatch.setattr(ob, "SHOT_LINE", False)
    assert not np.allclose(on, _target(state, carrier, defender))
    assert ob._shot_line_weight(carrier.pos, 105.0) == 0.0


def test_in_the_shot(scene):
    state, carrier, defender = scene
    mate = next(p for p in state.away_team.players if p.position != "GK" and p is not defender)
    defender.pos = np.array([89.0, 38.0])
    assert ob._in_the_shot(defender, state, carrier, 105.0, [])
    mate.pos = np.array([80.0, 34.0])        # going at him from behind
    assert ob._in_the_shot(defender, state, carrier, 105.0, [mate.player_id])
    mate.pos = np.array([87.0, 34.0])        # going at him, goal-side already
    assert not ob._in_the_shot(defender, state, carrier, 105.0, [mate.player_id])
    defender.pos = np.array([82.0, 34.0])    # behind him
    assert not ob._in_the_shot(defender, state, carrier, 105.0, [])
    carrier.pos = np.array([65.0, 34.0])     # out of range
    defender.pos = np.array([70.0, 34.0])
    assert not ob._in_the_shot(defender, state, carrier, 105.0, [])
