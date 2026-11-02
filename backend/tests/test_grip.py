"""
Grip (kinematics.GRIP): the braking and sideways part of a player's change of
velocity is limited, so a cut or a stop at a sprint costs far more than at a
jog, and he eases off into a turn rather than sail past.

These pin what grip changes in the movement, and that reach_ticks -- the plain
law's exact answer plus the tabulated cost of grip -- still says when a player
steered by pursuit_velocity gets there.
"""
import math

import numpy as np
import pytest

from app.engine import kinematics
from app.engine.kinematics import pursuit_velocity, reach_ticks, step_velocity

TOP = 0.236  # m/tick, 7.08 m/s


@pytest.fixture
def grip(monkeypatch):
    monkeypatch.setattr(kinematics, "GRIP", True)


def _simulate(pos, vel, top, point, reach, cap=2000):
    """What game_loop does for a player sent flat out at a point."""
    t = 0
    while math.hypot(*(point - pos)) > reach and t < cap:
        vel = step_velocity(vel, pursuit_velocity(pos, vel, point, top, reach), top)
        pos = pos + vel
        t += 1
    return t


def _turn_ticks(speed_share):
    """Ticks to swing a run at `speed_share` of top through 90 degrees."""
    vel = np.array([TOP * speed_share, 0.0])
    t = 0
    while math.atan2(vel[1], vel[0]) < math.radians(85) and t < 500:
        vel = step_velocity(vel, np.array([0.0, TOP * speed_share]), TOP)
        t += 1
    return t


def test_a_start_from_rest_is_not_slowed(grip):
    """Pushing on along his run is the plain law: sprint starts are real."""
    v_grip = np.zeros(2)
    for _ in range(30):
        v_grip = step_velocity(v_grip, np.array([TOP, 0.0]), TOP)
    kinematics.GRIP = False
    v_plain = np.zeros(2)
    for _ in range(30):
        v_plain = step_velocity(v_plain, np.array([TOP, 0.0]), TOP)
    assert np.allclose(v_grip, v_plain)


def test_a_cut_at_a_sprint_costs_more_than_at_a_jog(grip):
    """The point of grip: a jogger cuts sharply, a sprinter takes far longer.
    Under the plain law the two take the same time."""
    jog, sprint = _turn_ticks(0.5), _turn_ticks(1.0)
    kinematics.GRIP = False
    assert _turn_ticks(0.5) == _turn_ticks(1.0)
    assert sprint > 1.4 * jog


def _stop_distance(share):
    vel, dist = np.array([TOP * share, 0.0]), 0.0
    while vel[0] > 0.02 * TOP:
        vel = step_velocity(vel, np.zeros(2), TOP)
        dist += vel[0]
    return dist


def test_stopping_distance_grows_with_speed_squared(grip):
    """Twice the speed, about four times the distance (braking at his grip);
    the plain law only doubles it."""
    ratio = _stop_distance(1.0) / _stop_distance(0.5)
    kinematics.GRIP = False
    plain = _stop_distance(1.0) / _stop_distance(0.5)
    assert plain == pytest.approx(2.0, rel=0.05)
    assert ratio > 3.0


def test_he_eases_off_into_a_tight_turn(grip):
    """Sprinting at a point close and off to the side: slower than top."""
    v = pursuit_velocity(np.zeros(2), np.array([TOP, 0.0]), np.array([3.0, 3.0]), TOP, 1.0)
    assert math.hypot(*v) < 0.9 * TOP
    # ...but a point he will pass within reach of sets no limit.
    v = pursuit_velocity(np.zeros(2), np.array([TOP, 0.0]), np.array([8.0, 0.5]), TOP, 1.0)
    assert math.hypot(*v) == pytest.approx(TOP)


def test_reach_ticks_matches_the_simulated_movement(grip):
    """The licence for using reach_ticks in the decisions under grip."""
    rng = np.random.default_rng(7)
    errors = []
    for _ in range(300):
        top = rng.uniform(0.19, 0.29)
        d = rng.uniform(1.5, 40.0)
        ang = rng.uniform(-math.pi, math.pi)
        vel = np.array([math.cos(ang), math.sin(ang)]) * top * rng.uniform(0.0, 1.0)
        point = np.array([d, 0.0])
        sim = _simulate(np.zeros(2), vel.copy(), top, point, 1.0)
        pred = float(reach_ticks(np.zeros(2), vel, top, point, 1.0))
        errors.append(sim - pred)
    errors = np.abs(np.array(errors))
    assert np.median(errors) <= 1.0
    assert np.percentile(errors, 95) <= 3.0


def test_the_scalar_path_equals_the_array_path(grip):
    rng = np.random.default_rng(3)
    for _ in range(50):
        pos = rng.uniform(0, 50, 2)
        vel = rng.uniform(-TOP, TOP, 2) * 0.7
        q = rng.uniform(0, 50, 2)
        a = float(reach_ticks(pos[None], vel[None], TOP, q[None], 1.0)[0])
        s = kinematics._reach_ticks_scalar(*pos, *vel, TOP, *q, 1.0)
        assert s == pytest.approx(a, abs=1e-6)


# -- Choosing to run slower ----------------------------------------------------

def _quiet_state():
    """Everyone far away except whoever a test places."""
    from app.engine.factory import build_test_state
    state = build_test_state(seed=0)
    for p in state.all_players():
        p.pos = np.array([5.0, 5.0]) if p.team_id == state.home_team.team_id else np.array([100.0, 63.0])
        p.velocity = np.zeros(2)
    return state


def test_an_uncontested_pass_is_met_at_a_timed_pace(grip):
    """Told where, the receiver decides how fast: he arrives just before the
    ball, not flat out."""
    from app.engine.decisions.off_ball import _timed_run
    state = _quiet_state()
    passer, receiver = state.home_team.players[3], state.home_team.players[5]
    passer.pos, receiver.pos = np.array([30.0, 34.0]), np.array([52.0, 38.0])
    ball = state.ball
    ball.owner_id, ball.state = None, "in_pass"
    ball.pos = passer.pos.copy()
    ball.velocity = np.array([0.6, 0.0])
    ball.intended_receiver_id = receiver.player_id
    ball.pass_target = np.array([55.0, 34.0])
    act = _timed_run(receiver, state)
    assert act is not None and act.pace is not None
    assert act.pace < 0.8 * receiver.speed


def test_a_contested_pass_point_is_sprinted_to(grip):
    """Contested, he does not dawdle: flat out to the same point."""
    from app.engine.decisions.off_ball import _timed_run
    state = _quiet_state()
    receiver = state.home_team.players[5]
    receiver.pos = np.array([52.0, 38.0])
    rival = state.away_team.players[5]
    rival.pos = np.array([57.0, 37.0])
    ball = state.ball
    ball.owner_id, ball.state = None, "in_pass"
    ball.pos = np.array([30.0, 34.0])
    ball.velocity = np.array([0.6, 0.0])
    ball.intended_receiver_id = receiver.player_id
    ball.pass_target = np.array([55.0, 34.0])
    act = _timed_run(receiver, state)
    assert act is not None and act.sprint and act.pace is None
    assert abs(act.target_pos[1] - 34.0) < 1e-6


def test_a_running_teammate_is_played_where_his_run_takes_him(grip):
    """To his feet as he ran, the ball arrived where he had been."""
    from app.engine.decisions.candidates import _lead_point
    state = _quiet_state()
    mate = state.home_team.players[5]
    mate.pos = np.array([50.0, 34.0])
    mate.velocity = np.array([0.0, 0.2])
    lead = _lead_point(np.array([35.0, 34.0]), mate, 0.6)
    assert lead[1] > 36.0 and abs(lead[0] - 50.0) < 1e-6
    mate.velocity = np.zeros(2)
    assert np.allclose(_lead_point(np.array([35.0, 34.0]), mate, 0.6), mate.pos)


def test_a_carrier_with_nobody_near_idles(grip):
    from app.engine.decisions.candidates import Candidate
    from app.engine.decisions.on_ball import _may_idle
    from app.engine.constants import BALL_FRICTION_COEFF
    state = _quiet_state()
    carrier = state.home_team.players[5]
    carrier.pos = np.array([40.0, 34.0])
    step = Candidate(kind="dribble", target=carrier.pos + np.array([2.0, 0.0]),
                     speed=0.2, friction=BALL_FRICTION_COEFF)
    assert _may_idle(step, carrier, state)
    state.away_team.players[5].pos = carrier.pos + np.array([3.0, 0.0])   # pressure
    assert not _may_idle(step, carrier, state)


# -- Crossing to the free man --------------------------------------------------

def _crossing_state():
    """Home attacks toward x = 105; a winger on the right flank."""
    state = _quiet_state()
    team = state.home_team
    winger = next(p for p in team.players if p.position not in ("GK", "LD", "CD", "RD"))
    winger.pos = np.array([92.0, 60.0])
    for p in state.all_players():
        p.has_ball = False
    winger.has_ball = True
    state.ball.owner_id = winger.player_id
    state.ball.pos = winger.pos.copy()
    return state, winger


def test_a_free_teammate_in_the_box_is_a_cross_target(grip):
    from app.engine.decisions.candidates import _crosses_to_players
    state, winger = _crossing_state()
    assert state.home_team.direction == 1   # the setup assumes home attacks x = 105
    mate = next(p for p in state.home_team.players
                if p.position not in ("GK", "LD", "CD", "RD") and p is not winger)
    mate.pos = np.array([99.0, 30.0])
    crosses = _crosses_to_players(winger, state, state.home_team, set())
    assert any(c.recipient_id == mate.player_id and c.is_cross for c in crosses)


def test_a_free_finisher_is_worth_more_than_a_marked_one(grip):
    from app.engine.decisions.candidates import Candidate
    from app.engine.decisions.on_ball import _finisher_freedom
    from app.engine.constants import BALL_FRICTION_COEFF
    state, winger = _crossing_state()
    mate = next(p for p in state.home_team.players
                if p.position not in ("GK", "LD", "CD", "RD") and p is not winger)
    mate.pos = np.array([99.0, 30.0])
    cross = Candidate(kind="pass", target=mate.pos.copy(), speed=0.5, friction=BALL_FRICTION_COEFF,
                      recipient_id=mate.player_id, is_cross=True, vz=0.3)
    marker = state.away_team.players[4]
    marker.pos = np.array([90.0, 40.0])                      # far: he is free
    free = _finisher_freedom(cross, state, state.away_team.players)
    marker.pos = mate.pos + np.array([0.8, 0.0])             # tight: he is marked
    marked = _finisher_freedom(cross, state, state.away_team.players)
    assert free > 0.8 and marked < 0.5
