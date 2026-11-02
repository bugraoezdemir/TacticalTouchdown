"""
The ball in the air (ball_flight.py) and the lofted pass built on it.

The launch solver, the trajectory the decision layer samples and the step the
simulation runs must agree, or players chase balls that land elsewhere.
"""
import numpy as np

from app.engine.ball_flight import below_window, loft_launch, step, trajectory
from app.engine.constants import BALL_FRICTION_COEFF, HEADER_HEIGHT, LOFT_SPEED_MAX
from app.engine.decisions.interception import lofted_survival, route_survival
from app.engine.ball_physics import nominal_pass_speed
from app.engine.factory import build_test_state


def _land(vh, vz0):
    pos, vel, h, vz = np.zeros(2), np.array([vh, 0.0]), 0.0, vz0
    for t in range(1, 400):
        pos, vel, h, vz = step(pos, vel, h, vz, BALL_FRICTION_COEFF)
        if h == 0.0:
            return float(pos[0]), t
    raise AssertionError("never landed")


def test_lofted_ball_lands_where_it_was_aimed():
    for dist in (15.0, 30.0, 50.0):
        for angle in (22.0, 38.0):
            vh, vz0, T = loft_launch(dist, angle)
            x, t = _land(vh, vz0)
            assert abs(x - dist) < 0.8, (dist, angle, x)
            assert abs(t - T) <= 1.0


def test_launch_respects_the_strike_limit():
    assert loft_launch(200.0, 38.0) is None
    vh, vz0, _ = loft_launch(40.0, 22.0)
    assert np.hypot(vh, vz0) <= LOFT_SPEED_MAX


def test_trajectory_matches_the_simulation_step():
    vh, vz0, _ = loft_launch(30.0, 30.0)
    pts, hs = trajectory(np.zeros(2), np.array([vh, 0.0]), 0.0, vz0, BALL_FRICTION_COEFF, 120)
    pos, vel, h, vz = np.zeros(2), np.array([vh, 0.0]), 0.0, vz0
    for i in range(120):
        pos, vel, h, vz = step(pos, vel, h, vz, BALL_FRICTION_COEFF)
        assert np.allclose(pos, pts[i]) and abs(h - hs[i]) < 1e-9


def test_it_bounces_and_then_rolls():
    vh, vz0, _ = loft_launch(30.0, 38.0)
    _, hs = trajectory(np.zeros(2), np.array([vh, 0.0]), 0.0, vz0, BALL_FRICTION_COEFF, 400)
    landings = np.nonzero((hs[1:] == 0.0) & (hs[:-1] > 0.0))[0]
    assert len(landings) >= 2              # landed and bounced at least once
    assert np.all(hs[-50:] == 0.0)         # and is rolling by the end


def test_reach_window_brackets_the_flight():
    vh, vz0, T = loft_launch(40.0, 38.0)
    up, down = below_window(vz0, HEADER_HEIGHT)
    assert 0.0 < up < down < T


def test_a_lofted_ball_goes_over_the_man_in_between():
    """A defender standing on the lane kills the ground pass, not the chip."""
    state = build_test_state()
    opp = state.away_team.players[5]
    opp.pos = np.array([45.0, 34.0]); opp.velocity = np.zeros(2)
    origin, target = np.array([35.0, 34.0]), np.array([65.0, 34.0])
    ground = route_survival(origin, target, nominal_pass_speed(BALL_FRICTION_COEFF, 30.0),
                            BALL_FRICTION_COEFF, [opp])
    vh, vz0, _ = loft_launch(30.0, 30.0)
    lofted = lofted_survival(origin, target, vh, vz0, [opp])
    assert ground < 0.2 < 0.8 < lofted


def test_a_lofted_ball_is_still_contested_where_it_lands():
    state = build_test_state()
    opp = state.away_team.players[5]
    opp.pos = np.array([64.0, 34.5]); opp.velocity = np.zeros(2)
    vh, vz0, _ = loft_launch(30.0, 30.0)
    assert lofted_survival(np.array([35.0, 34.0]), np.array([65.0, 34.0]), vh, vz0, [opp]) < 0.3
