"""What a player can see coming: a pass in danger, and his own run before a turn."""
import numpy as np
import pytest

import app.engine.decisions.candidates as ca
import app.engine.decisions.off_ball as ob
import app.engine.decisions.pursuit as pu
from app.engine import kinematics
from app.engine.ball_physics import travel_distance
from app.engine.constants import BALL_FRICTION_COEFF
from app.engine.factory import build_test_state

inertia = pytest.mark.skipif(not kinematics.INERTIA, reason="needs FB_INERTIA=1")


def _clear_pitch(state):
    """Nobody near the middle: each test places the men it needs."""
    for i, p in enumerate(state.all_players()):
        p.pos = np.array([2.0 + i, 2.0])
        p.velocity = np.zeros(2)
        p.has_ball = False


def _pass_scene(speed=0.3, point=(60.0, 34.0), since=20):
    """A slow ball rolling along y = 34 from x = 40 to a receiver standing on
    the pass point. Returns (state, team, receiver)."""
    state = build_test_state(seed=1)
    _clear_pitch(state)
    team = state.home_team
    receiver = next(p for p in team.players if p.position != "GK")
    receiver.pos = np.array(point, float)
    ball = state.ball
    ball.state = "in_pass"
    ball.owner_id = None
    ball.pos = np.array([40.0, 34.0])
    ball.velocity = np.array([speed, 0.0])
    ball.height = 0.0
    ball.vz = 0.0
    ball.intended_receiver_id = receiver.player_id
    ball.last_toucher_team_id = team.team_id
    ball.pass_target = np.array(point, float)
    ball.struck_tick = state.tick - since
    return state, team, receiver


def _opponent(state, at, velocity=(0.0, 0.0)):
    o = next(p for p in state.away_team.players if p.position != "GK")
    o.pos = np.array(at, float)
    o.velocity = np.array(velocity, float)
    return o


def _ball_at(ticks, speed=0.3):
    return 40.0 + travel_distance(speed, BALL_FRICTION_COEFF, ticks)


def test_an_opponent_stepping_onto_the_path_is_seen():
    state, team, receiver = _pass_scene()
    _opponent(state, (_ball_at(30), 37.0))
    assert ob._comes_to_ball(receiver, state, team) is not None


def test_an_opponent_nowhere_near_it_is_not():
    state, team, receiver = _pass_scene()
    _opponent(state, (_ball_at(30), 66.0))
    assert ob._comes_to_ball(receiver, state, team) is None


def test_the_receiver_standing_on_the_point_goes_to_the_ball():
    """The old answer: settle on the point and wait."""
    state, team, receiver = _pass_scene()
    _opponent(state, (_ball_at(30), 37.0))
    action = ob.off_ball_decision(receiver, state)
    assert action.sprint and action.through
    assert action.target_pos[0] < receiver.pos[0] - 3.0
    assert abs(action.target_pos[1] - 34.0) < 1e-6


def test_a_man_the_ball_is_leaving_behind_is_no_danger():
    state, team, receiver = _pass_scene(speed=0.5, since=8)
    _opponent(state, (38.0, 34.0), velocity=(0.2, 0.0))
    assert ob._comes_to_ball(receiver, state, team) is None


def test_the_man_at_the_passers_feet_does_not_count_at_the_strike():
    state, team, receiver = _pass_scene(speed=0.5, since=0)
    _opponent(state, (40.5, 34.5))
    assert ob._comes_to_ball(receiver, state, team) is None


def test_once_he_goes_he_stays_gone_for_that_pass():
    state, team, receiver = _pass_scene()
    o = _opponent(state, (_ball_at(30), 37.0))
    assert ob._comes_to_ball(receiver, state, team) is not None
    o.pos = np.array([100.0, 66.0])
    assert ob._comes_to_ball(receiver, state, team) is not None
    state.ball.struck_tick += 1            # a new pass
    assert ob._comes_to_ball(receiver, state, team) is None


def test_a_ball_past_its_point_is_watched_to_where_it_stops():
    state, team, receiver = _pass_scene(point=(39.0, 34.0))
    state.ball.pass_target = np.array([39.0, 34.0])
    _opponent(state, (_ball_at(40), 37.0))
    assert ob._pass_arrival(state.ball, ob._ball_friction(state.ball)) is None
    assert ob._comes_to_ball(receiver, state, team) is not None


def test_only_the_man_it_was_played_to():
    state, team, receiver = _pass_scene()
    other = [p for p in team.players if p.position != "GK"][1]
    _opponent(state, (_ball_at(30), 37.0))
    assert ob._comes_to_ball(other, state, team) is None


def test_a_ball_of_theirs_is_not_his():
    state, team, receiver = _pass_scene()
    _opponent(state, (_ball_at(30), 37.0))
    state.ball.last_toucher_team_id = state.away_team.team_id
    assert ob._comes_to_ball(receiver, state, team) is None


def test_the_switch(monkeypatch):
    state, team, receiver = _pass_scene()
    _opponent(state, (_ball_at(30), 37.0))
    monkeypatch.setattr(ob, "COME_TO_BALL", False)
    assert ob._comes_to_ball(receiver, state, team) is None


def test_he_only_has_to_get_a_foot_to_it():
    """A ball going past 3 m away: with his reach the meeting is earlier, up
    the path, not somewhere further along it he can never get ahead of."""
    state, team, receiver = _pass_scene(speed=0.45, point=(60.0, 34.0))
    receiver.pos = np.array([46.0, 37.0])
    args = (state.ball.pos, state.ball.velocity, BALL_FRICTION_COEFF, receiver)
    onto, t_onto = pu.ball_intercept_point(*args)
    foot, t_foot = pu.ball_intercept_point(*args, reach=1.0)
    assert t_foot < t_onto
    assert foot[0] < onto[0]


def test_in_the_air_only_where_it_has_come_down():
    state, team, receiver = _pass_scene()
    ball = state.ball
    ball.velocity = np.array([0.6, 0.0])
    ball.height = 4.0
    ball.vz = 0.25
    from app.engine.ball_flight import trajectory

    path, hs = trajectory(ball.pos, ball.velocity, ball.height, ball.vz, BALL_FRICTION_COEFF, 20)
    under = path[9]
    assert hs[9] > 3.0
    opponents = [_opponent(state, under)]
    assert not pu.closing_on_path(ball.pos, ball.velocity, BALL_FRICTION_COEFF, opponents,
                                  6, 12, height=ball.height, vz=ball.vz)


# -- The turn ----------------------------------------------------------------

def _carrier(velocity):
    state = build_test_state(seed=1)
    _clear_pitch(state)
    carrier = next(p for p in state.home_team.players if p.position != "GK")
    carrier.pos = np.array([50.0, 34.0])
    carrier.velocity = np.array(velocity, float) * carrier.speed
    return state, carrier


def _race(carrier, opponents, direction, length=4.0):
    spots = carrier.pos[None, :] + np.array(direction, float)[None, :] * length
    return float(ca._touch_race(carrier, np.array([length]), spots, opponents)[0])


@inertia
def test_a_touch_back_against_his_run_is_a_race_he_can_lose():
    state, carrier = _carrier((0.9, 0.0))
    o = _opponent(state, (42.0, 34.0))
    assert _race(carrier, [o], (-1.0, 0.0)) < 0.5 * _race(carrier, [o], (1.0, 0.0))


@inertia
def test_standing_still_every_direction_is_the_same_to_him():
    state, carrier = _carrier((0.0, 0.0))
    o = _opponent(state, (50.0, 26.0))
    assert _race(carrier, [o], (1.0, 0.0)) == pytest.approx(_race(carrier, [o], (-1.0, 0.0)), abs=1e-6)


@inertia
def test_the_challengers_run_counts_too():
    """The same man in the same place: running at the spot he wins it, running
    away he has to stop first."""
    state, carrier = _carrier((0.0, 0.0))
    o = _opponent(state, (50.0, 26.0), velocity=(0.0, 0.25))
    towards = _race(carrier, [o], (0.0, 1.0))
    o.velocity = np.array([0.0, -0.25])
    assert towards < _race(carrier, [o], (0.0, 1.0))


@inertia
def test_the_own_momentum_switch(monkeypatch):
    state, carrier = _carrier((0.9, 0.0))
    o = _opponent(state, (42.0, 34.0))
    on = _race(carrier, [o], (-1.0, 0.0))
    monkeypatch.setattr(ca, "OWN_MOMENTUM", False)
    assert _race(carrier, [o], (-1.0, 0.0)) > on


@inertia
def test_the_sprint_back_is_not_chosen_with_a_man_there_to_take_it():
    state, carrier = _carrier((0.9, 0.0))
    o = _opponent(state, (45.0, 34.0))
    d = np.array([-1.0, 0.0])
    sprint = ca._solve_sprint(carrier, d, [o], 105.0)
    straight = ca._solve_sprint(carrier, -d, [o], 105.0)
    assert sprint and straight
    assert sprint[0][3] < straight[0][3]
