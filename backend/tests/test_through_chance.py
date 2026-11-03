"""A through ball is worth the chance it sets up; Wing attacks tilts passes to the flank."""
import numpy as np
import pytest

import app.engine.decisions.on_ball as onb
from app.engine.constants import BALL_FRICTION_COEFF
from app.engine.decisions.candidates import Candidate
from app.engine.factory import build_test_state


@pytest.fixture
def scene():
    """Home attacks x = 105. The away outfielders are all back at x 70-79 near
    the touchline, the keeper on his line; a home runner at x 82 in the middle."""
    state = build_test_state(seed=1)
    outfield = [p for p in state.home_team.players if p.position != "GK"]
    passer, runner = outfield[0], outfield[1]
    passer.pos = np.array([60.0, 34.0])
    runner.pos, runner.velocity = np.array([82.0, 34.0]), np.zeros(2)
    for i, p in enumerate(q for q in state.away_team.players if q.position != "GK"):
        p.pos, p.velocity = np.array([70.0 + i, 2.0]), np.zeros(2)
    keeper = next(p for p in state.away_team.players if p.position == "GK")
    keeper.pos, keeper.velocity = np.array([104.0, 34.0]), np.zeros(2)
    return state, passer, runner


def _ball(target, runner):
    return Candidate(kind="pass", target=np.array(target), speed=1.0, friction=BALL_FRICTION_COEFF,
                     recipient_id=runner.player_id, is_runway=True)


def _chance(state, passer, cand):
    return onb._through_chance(cand, state, state.opponents_of(passer.team_id), 105.0)


def test_through_in_the_middle_beats_through_to_the_corner(scene):
    state, passer, runner = scene
    middle = _chance(state, passer, _ball([88.0, 34.0], runner))
    corner = _chance(state, passer, _ball([98.0, 64.0], runner))
    assert middle > 2.0 * corner


def test_a_defender_on_his_heels_makes_it_worth_less(scene):
    state, passer, runner = scene
    alone = _chance(state, passer, _ball([88.0, 34.0], runner))
    chaser = next(p for p in state.away_team.players if p.position != "GK")
    chaser.pos = np.array([87.0, 36.0])
    assert _chance(state, passer, _ball([88.0, 34.0], runner)) < 0.8 * alone


def test_a_through_ball_is_valued_by_its_chance(scene, monkeypatch):
    state, passer, runner = scene
    cand = onb.score_candidate(_ball([88.0, 34.0], runner), passer, state)
    assert cand.v == pytest.approx(onb.PASS_CONTINUATION_DISCOUNT * max(
        onb.position_value(cand.target, 105.0), _chance(state, passer, cand)))
    monkeypatch.setattr(onb, "THROUGH_CHANCE", False)
    assert onb.score_candidate(_ball([88.0, 34.0], runner), passer, state).v != pytest.approx(cand.v)


def test_wing_tilt_is_neutral_at_one_and_in_the_middle():
    assert onb._wing_tilt(np.array([60.0, 4.0]), 1.0) == 1.0
    assert onb._wing_tilt(np.array([60.0, 34.0]), 2.0) == 1.0
    assert onb._wing_tilt(np.array([60.0, 34.0 + onb.WING_INNER_M]), 2.0) == 1.0


def test_wing_tilt_is_bounded_and_both_ways():
    wide = np.array([60.0, 34.0 + onb.WING_OUTER_M + 2.0])
    assert onb._wing_tilt(wide, 2.0) == pytest.approx(2.0 ** onb.WING_TILT_GAIN)
    assert onb._wing_tilt(wide, 0.5) == pytest.approx(0.5 ** onb.WING_TILT_GAIN)
    half = np.array([60.0, 34.0 - (onb.WING_INNER_M + onb.WING_OUTER_M) / 2.0])
    assert 1.0 < onb._wing_tilt(half, 2.0) < onb._wing_tilt(wide, 2.0)


def test_the_slider_reaches_the_pass(scene):
    state, passer, runner = scene
    runner.pos = np.array([66.0, 60.0])
    to_flank = Candidate(kind="pass", target=np.array([66.0, 60.0]), speed=1.0,
                         friction=BALL_FRICTION_COEFF, recipient_id=runner.player_id)
    base = onb.score_candidate(to_flank, passer, state).context
    state.home_team.tactics.wing_attacks = 2.0
    tilted = Candidate(kind="pass", target=np.array([66.0, 60.0]), speed=1.0,
                       friction=BALL_FRICTION_COEFF, recipient_id=runner.player_id)
    assert onb.score_candidate(tilted, passer, state).context == pytest.approx(base * 2.0 ** onb.WING_TILT_GAIN)
