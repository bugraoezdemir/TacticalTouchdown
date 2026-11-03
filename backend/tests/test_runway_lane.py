"""A through ball's lane stops where the reception race starts, as a cross's does."""
import numpy as np
import pytest

import app.engine.decisions.on_ball as onb
from app.engine.constants import BALL_FRICTION_COEFF, RECEPTION_FROM_FRACTION
from app.engine.decisions.candidates import Candidate
from app.engine.factory import build_test_state


@pytest.fixture
def scene():
    """Home attacks x = 105; the away outfielders' deepest man at x = 80."""
    state = build_test_state(seed=1)
    passer = next(p for p in state.home_team.players if p.position != "GK")
    passer.pos = np.array([60.0, 34.0])
    for i, p in enumerate(q for q in state.away_team.players if q.position != "GK"):
        p.pos = np.array([70.0 + i, 2.0])
        p.velocity = np.zeros(2)
    deepest = next(p for p in state.away_team.players if p.position != "GK")
    deepest.pos = np.array([80.0, 60.0])
    return state, passer


def _pass(target, runway=True):
    return Candidate(kind="pass", target=np.array(target), speed=1.0,
                     friction=BALL_FRICTION_COEFF, recipient_id=None, is_runway=runway)


def test_a_through_ball_lane_ends_where_the_race_starts(scene):
    state, passer = scene
    end = onb._lane_end(_pass([85.0, 34.0]), passer, state)
    assert np.allclose(end, [60.0 + 25.0 * RECEPTION_FROM_FRACTION, 34.0])


def test_a_runway_ball_short_of_the_last_man_runs_to_its_target(scene):
    state, passer = scene
    assert np.allclose(onb._lane_end(_pass([78.0, 34.0]), passer, state), [78.0, 34.0])


def test_a_pass_to_feet_runs_to_its_target(scene):
    state, passer = scene
    assert np.allclose(onb._lane_end(_pass([85.0, 34.0], runway=False), passer, state), [85.0, 34.0])


def test_the_switch(scene, monkeypatch):
    state, passer = scene
    monkeypatch.setattr(onb, "RUNWAY_LANE", False)
    assert np.allclose(onb._lane_end(_pass([85.0, 34.0]), passer, state), [85.0, 34.0])


def test_a_defender_racing_the_runner_is_left_to_the_race(scene):
    """A defender beside the target, nowhere near the first part of the flight,
    cuts a through ball's lane only through the reception race."""
    state, passer = scene
    defender = next(p for p in state.away_team.players if p.position != "GK")
    defender.pos = np.array([86.0, 36.0])
    through, feet = _pass([88.0, 34.0]), _pass([88.0, 34.0], runway=False)
    onb._batch_lanes([through, feet], passer, state)
    assert through.lane > 0.9
    assert feet.lane < through.lane
