"""Shots: the shooter plans on missing, picks his margin, and the keeper halves
the angle; from the flank a ball into the box is a cross."""
import numpy as np
import pytest

import app.engine.decisions.candidates as ca
import app.engine.decisions.off_ball as ob
import app.engine.decisions.on_ball as onb
from app.engine.constants import GK_ADVANCE, GOAL_BOTTOM_Y, GOAL_TOP_Y, SHOT_FRICTION_COEFF, SHOT_SPEED
from app.engine.decisions.candidates import Candidate
from app.engine.factory import build_test_state


def _shot(player, y):
    return Candidate(kind="shot", target=np.array([105.0, y]), speed=SHOT_SPEED,
                     friction=SHOT_FRICTION_COEFF)


@pytest.fixture
def shooter():
    state = build_test_state(seed=1)
    p = next(q for q in state.home_team.players if q.position != "GK")
    p.shooting = 60.0
    p.first_touch_speed = 0.0
    return state, p


def test_he_misses_more_from_further(shooter):
    _, p = shooter
    p.pos = np.array([95.0, 34.0])
    near = onb._on_target(_shot(p, GOAL_TOP_Y + 0.5), p)
    p.pos = np.array([80.0, 34.0])
    far = onb._on_target(_shot(p, GOAL_TOP_Y + 0.5), p)
    assert far < near < 1.0


def test_a_margin_inside_the_post_misses_less(shooter):
    _, p = shooter
    p.pos = np.array([85.0, 34.0])
    assert onb._on_target(_shot(p, GOAL_TOP_Y + 1.2), p) > onb._on_target(_shot(p, GOAL_TOP_Y + 0.5), p)


def test_the_margin_is_offered(shooter):
    state, p = shooter
    p.pos = np.array([85.0, 34.0])
    ys = {round(float(c.target[1]), 2) for c in ca.build_candidates(p, state)
          if c.kind == "shot" and c.vz == 0.0 and c.curl == 0.0}
    assert round(GOAL_TOP_Y + 1.2, 2) in ys and round(GOAL_BOTTOM_Y - 1.2, 2) in ys


def test_the_keeper_halves_the_angle():
    """From wide on the top side he stands toward the near (top) post, off the
    line to the middle of the goal, GK_ADVANCE out."""
    state = build_test_state(seed=1)
    team = state.away_team if state.away_team.direction == -1 else state.home_team
    goal_x = 105.0
    ball = np.array([92.0, 14.0])
    spot = ob._gk_spot(ball, goal_x, team)
    to_centre = np.array([goal_x, 34.0]) - ball
    along = (spot - ball) @ to_centre / np.linalg.norm(to_centre)
    on_centre_line = ball + to_centre / np.linalg.norm(to_centre) * along
    assert spot[1] < on_centre_line[1]           # toward the near post
    a = np.arctan2(GOAL_TOP_Y - ball[1], goal_x - ball[0])
    b = np.arctan2(GOAL_BOTTOM_Y - ball[1], goal_x - ball[0])
    s = np.arctan2(spot[1] - ball[1], spot[0] - ball[0])
    assert s == pytest.approx(0.5 * (a + b), abs=1e-6)
    assert np.hypot(*(spot - np.array([goal_x, ball[1] + (goal_x - ball[0]) * np.tan(s)]))) == pytest.approx(GK_ADVANCE)


def test_from_the_flank_a_ball_into_the_box_is_a_cross():
    state = build_test_state(seed=1)
    team = state.home_team
    p = next(q for q in team.players if q.position not in ("GK", "LD", "CD", "RD"))
    p.pos = np.array([90.0, 6.0]) if team.direction == 1 else np.array([15.0, 6.0])
    into_box = Candidate(kind="pass", target=np.array([97.0 if team.direction == 1 else 8.0, 34.0]),
                         speed=1.0, friction=0.005, is_runway=True)
    back = Candidate(kind="pass", target=np.array([75.0 if team.direction == 1 else 30.0, 20.0]),
                     speed=1.0, friction=0.005)
    ca._flank_balls_are_crosses(p, team, [into_box, back])
    assert into_box.is_cross and not back.is_cross
