"""Bent shots: the arc, who can read it, what the shooter plans on."""
import math

import numpy as np
import pytest

import app.engine.decisions.candidates as ca
import app.engine.decisions.off_ball as ob
import app.engine.game_loop as gl
import app.engine.physics as ph
from app.engine.ball_flight import arc_point, bend_start, crossing, drive_launch, step, trajectory, turned
from app.engine.constants import (
    BEND_CURL,
    BEND_MAX_DIST,
    BEND_MIN_DIST,
    BEND_PACE,
    GK_LEAP_TICKS,
    GOAL_BOTTOM_Y,
    SHOT_FRICTION_COEFF,
    SHOT_SPEED,
    TOP_CORNER_HEIGHT,
)
from app.engine.decisions.candidates import _bent_shots
from app.engine.decisions.interception import bend_miss, lofted_shot_survival
from app.engine.decisions.on_ball import _action_from, _on_frame
from app.engine.factory import build_test_state

TOP_CORNER = np.array([105.0, GOAL_BOTTOM_Y - 0.5])


def _clear_pitch(state):
    for i, p in enumerate(state.all_players()):
        p.pos = np.array([2.0 + i, 2.0])
        p.velocity = np.zeros(2)
        p.has_ball = False


def _shooter(depth=25.0, wide=0.0):
    state = build_test_state(seed=1)
    _clear_pitch(state)
    p = next(q for q in state.home_team.players if q.position != "GK")
    p.pos = np.array([105.0 - depth, 34.0 + wide])
    p.has_ball = True
    state.ball.pos = p.pos.copy()
    state.ball.owner_id = p.player_id
    keeper = next(q for q in state.away_team.players if q.position == "GK")
    keeper.pos = np.array([104.0, 34.0])
    return state, p, keeper


def _flight(state, cand):
    """The candidate as struck, with no error on it: (pos, vel, vz)."""
    p = state.get_player(state.ball.owner_id)
    action = _action_from(cand, p, state)
    return p.pos.copy(), action.velocity, action.vz, action.curl


# -- The arc -----------------------------------------------------------------

def test_a_ball_turns_in_the_air_and_keeps_its_pace():
    vel = np.array([1.0, 0.0])
    pos, h, vz = np.zeros(2), 0.0, 0.2
    for _ in range(10):
        pos, vel, h, vz = step(pos, vel, h, vz, SHOT_FRICTION_COEFF, curl=0.02)
    assert np.hypot(*vel) == pytest.approx(1.0)
    assert math.atan2(vel[1], vel[0]) == pytest.approx(0.2)


def test_a_rolling_ball_does_not_turn():
    vel = np.array([1.0, 0.0])
    _, after, _, _ = step(np.zeros(2), vel, 0.0, 0.0, SHOT_FRICTION_COEFF, curl=0.02)
    assert after[1] == 0.0


def test_the_arc_bulges_away_from_the_way_it_turns():
    # Turning left, it starts out to the right of the straight line.
    mid = arc_point(np.zeros(2), np.array([20.0, 0.0]), 0.5, 0.4)
    assert mid[0] == pytest.approx(10.0)
    assert mid[1] == pytest.approx(-20.0 * 0.4 / 8.0)


def test_arc_point_is_the_flight_itself():
    dist, pace = 25.0, SHOT_SPEED * BEND_PACE
    vh, vz, T = drive_launch(dist, TOP_CORNER_HEIGHT, pace)
    start = np.zeros(2)
    aim = np.array([dist, 0.0])
    vel = turned(np.array([vh, 0.0]), bend_start(BEND_CURL, T))
    pos, h = start, 0.0
    for k in range(1, int(T)):
        pos, vel, h, vz = step(pos, vel, h, vz, SHOT_FRICTION_COEFF, BEND_CURL)
        assert np.linalg.norm(pos - arc_point(start, aim, k / T, BEND_CURL * T)) < 0.15


@pytest.mark.parametrize("curl", [BEND_CURL, -BEND_CURL])
@pytest.mark.parametrize("depth,wide", [(14.0, 0.0), (20.0, 8.0), (22.0, -7.0)])
def test_struck_true_it_arrives_where_it_was_aimed(depth, wide, curl):
    state, _, _ = _shooter(depth, wide)
    cand = next(c for c in _bent_shots(state.get_player(state.ball.owner_id), TOP_CORNER) if c.curl == curl)
    pos, vel, vz, spin = _flight(state, cand)
    y, h = crossing(pos, vel, 0.0, vz, spin, 105.0)
    # Its last tick takes it past the line at an angle: up to a tick's slant.
    assert y == pytest.approx(TOP_CORNER[1], abs=0.4)
    assert h == pytest.approx(TOP_CORNER_HEIGHT, abs=0.3)


def test_it_leaves_his_foot_on_a_line_well_off_the_corner():
    state, _, _ = _shooter(25.0, 0.0)
    for cand in _bent_shots(state.get_player(state.ball.owner_id), TOP_CORNER):
        pos, vel, _, _ = _flight(state, cand)
        straight = pos[1] + vel[1] * (105.0 - pos[0]) / vel[0]
        assert abs(straight - TOP_CORNER[1]) > 3.0


# -- Nobody reads it ---------------------------------------------------------

def test_the_keeper_reads_the_flight_not_the_spin_so_he_comes_right_late():
    state, _, _ = _shooter(25.0, 0.0)
    cand = next(c for c in _bent_shots(state.get_player(state.ball.owner_id), TOP_CORNER) if c.curl > 0.0)
    pos, vel, vz, spin = _flight(state, cand)
    ball = state.ball
    ball.owner_id, ball.state = None, "in_shot"
    ball.pos, ball.velocity, ball.vz, ball.curl, ball.height = pos, vel, vz, spin, 0.0
    errors = []
    for _ in range(26):
        ph.apply_ball_update(ball, ph.compute_ball_update(ball))
        if ball.pos[0] < 104.0:
            errors.append(abs(ob._predict_goal_crossing_y(ball, 105.0) - TOP_CORNER[1]))
    assert errors[0] > 3.0
    assert errors[-1] < 0.5


def test_prediction_goes_by_the_flight_as_it_is_now():
    pts, _ = trajectory(np.zeros(2), np.array([1.0, 0.0]), 1.0, 0.1, SHOT_FRICTION_COEFF, 5)
    assert pts[-1, 1] == 0.0


def _flying_at(state, corner_y, curl, depth=25.0):
    """A ball in flight at the top corner, the keeper reading it."""
    ball = state.ball
    origin = ball.pos.copy()
    target = np.array([105.0, corner_y])
    dist = float(np.linalg.norm(target - origin))
    pace = SHOT_SPEED * BEND_PACE
    vh, vz, _ = drive_launch(dist - pace, TOP_CORNER_HEIGHT, pace)
    ball.owner_id, ball.state, ball.struck_tick = None, "in_shot", state.tick
    ball.velocity = turned((target - origin) / dist * vh, bend_start(curl, dist / vh))
    ball.vz, ball.curl, ball.height = vz, curl, 0.0


def _keeper_targets(state, keeper):
    """His target y on every tick up to the line."""
    ball, team = state.ball, state.away_team
    out = []
    while ball.pos[0] < 104.0:
        out.append((105.0 - float(ball.pos[0]), ob._gk_decision(keeper, state, 105.0, team).target_pos[1]))
        ph.apply_ball_update(ball, ph.compute_ball_update(ball))
        state.tick += 1
    return out


def test_on_his_feet_he_follows_the_ball_then_he_is_committed():
    state, p, keeper = _shooter(25.0, 0.0)
    state.away_team.direction = -1
    _flying_at(state, TOP_CORNER[1], BEND_CURL)
    seen = _keeper_targets(state, keeper)
    last = len(seen) - 1
    pace = np.hypot(*state.ball.velocity)
    left = [y for gap, y in seen if gap / pace <= GK_LEAP_TICKS - 1.0]
    early = [y for gap, y in seen if GK_LEAP_TICKS + 1.0 <= gap / pace < GK_LEAP_TICKS + 8.0]
    assert last > 0 and len(set(round(y, 3) for y in left)) == 1
    assert len(set(round(y, 3) for y in early)) > 1


def test_a_straight_shot_is_where_he_read_it():
    state, p, keeper = _shooter(25.0, 0.0)
    state.away_team.direction = -1
    _flying_at(state, TOP_CORNER[1], 0.0)
    assert len({round(y, 6) for gap, y in _keeper_targets(state, keeper)}) <= 2


def test_bending_away_from_him_he_is_short_of_it():
    state, p, keeper = _shooter(22.0, 0.0)
    keeper.pos = np.array([104.0, 34.0])
    cand = _bent_shots(p, TOP_CORNER)[0]
    away = abs(cand.curl)            # moving +x, left is up to the far corner
    assert bend_miss(keeper, p.pos, TOP_CORNER, cand.speed, away) == pytest.approx(
        BEND_CURL * cand.speed * GK_LEAP_TICKS ** 2 / 2.0)


def test_bending_toward_him_he_meets_it_on_his_way_unless_he_has_time():
    pace = SHOT_SPEED * BEND_PACE
    state, p, keeper = _shooter(14.0, 0.0)
    assert bend_miss(keeper, p.pos, TOP_CORNER, pace, -BEND_CURL) == 0.0
    state, p, keeper = _shooter(32.0, 0.0)
    assert bend_miss(keeper, p.pos, TOP_CORNER, pace, -BEND_CURL) > 0.0


def test_a_shooter_rates_the_bend_higher_for_it():
    state, p, keeper = _shooter(22.0, 0.0)
    cand = _bent_shots(p, TOP_CORNER)[0]
    args = (p.pos, TOP_CORNER, cand.speed, cand.vz, [keeper])
    plain = lofted_shot_survival(*args)
    assert lofted_shot_survival(*args, abs(cand.curl)) > plain


# -- What ends it ------------------------------------------------------------

def test_the_first_bounce_takes_the_spin_off_it():
    state, _, _ = _shooter()
    ball = state.ball
    ball.owner_id, ball.state = None, "in_shot"
    ball.height, ball.vz, ball.curl = 0.05, -0.3, BEND_CURL
    ball.velocity = np.array([1.0, 0.0])
    ph.apply_ball_update(ball, ph.compute_ball_update(ball))
    assert ball.curl == 0.0


def test_a_touch_takes_the_spin_off_it():
    state, _, _ = _shooter()
    state.ball.curl = BEND_CURL
    state.ball.ground()
    assert state.ball.curl == 0.0


# -- What the shooter plans on -----------------------------------------------

def test_offered_both_ways_in_range_only():
    state, p, _ = _shooter(25.0)
    assert sorted(c.curl for c in _bent_shots(p, TOP_CORNER)) == [-BEND_CURL, BEND_CURL]
    p.pos = np.array([105.0 - BEND_MIN_DIST + 1.0, TOP_CORNER[1]])
    assert _bent_shots(p, TOP_CORNER) == []
    p.pos = np.array([105.0 - BEND_MAX_DIST - 1.0, TOP_CORNER[1]])
    assert _bent_shots(p, TOP_CORNER) == []


def test_keepers_do_not_get_one():
    state = build_test_state(seed=1)
    keeper = next(q for q in state.home_team.players if q.position == "GK")
    assert not any(c.curl for c in ca.build_candidates(keeper, state))


def test_a_switch_turns_it_off(monkeypatch):
    state, p, _ = _shooter(25.0)
    assert any(c.curl for c in ca.build_candidates(p, state))
    monkeypatch.setattr(ca, "BENT_SHOTS", False)
    assert not any(c.curl for c in ca.build_candidates(p, state))


def test_a_man_on_the_straight_line_is_gone_round():
    state, p, keeper = _shooter(25.0, 0.0)
    keeper.pos = np.array([120.0, 34.0])
    blocker = next(q for q in state.away_team.players if q.position != "GK")
    origin = p.pos.copy()
    blocker.pos = origin + (TOP_CORNER - origin) * 0.5
    cand = _bent_shots(p, TOP_CORNER)[0]
    straight = lofted_shot_survival(origin, TOP_CORNER, cand.speed, cand.vz, [blocker])
    bent = lofted_shot_survival(origin, TOP_CORNER, cand.speed, cand.vz, [blocker], cand.curl)
    assert bent > straight


def test_a_man_on_the_arc_is_not():
    state, p, keeper = _shooter(25.0, 0.0)
    keeper.pos = np.array([120.0, 34.0])
    blocker = next(q for q in state.away_team.players if q.position != "GK")
    origin = p.pos.copy()
    cand = _bent_shots(p, TOP_CORNER)[0]
    T = np.linalg.norm(TOP_CORNER - origin) / cand.speed
    blocker.pos = arc_point(origin, TOP_CORNER, 0.15, cand.curl * T)
    assert lofted_shot_survival(origin, TOP_CORNER, cand.speed, cand.vz, [blocker], cand.curl) < 0.5


def test_no_spin_is_the_lofted_shot_as_it_was():
    state, p, keeper = _shooter(20.0, 3.0)
    blocker = next(q for q in state.away_team.players if q.position != "GK")
    blocker.pos = np.array([95.0, 33.0])
    vh, vz, _ = drive_launch(20.0, TOP_CORNER_HEIGHT, SHOT_SPEED)
    args = (p.pos, TOP_CORNER, vh, vz, [blocker, keeper])
    assert lofted_shot_survival(*args) == lofted_shot_survival(*args, 0.0)


def test_spin_error_costs_the_frame():
    state, p, _ = _shooter(25.0, 0.0)
    p.shooting = 50.0
    cand = _bent_shots(p, TOP_CORNER)[0]
    bent = _on_frame(cand, p)
    cand.curl = 0.0
    assert bent < _on_frame(cand, p)


def test_on_target_goes_by_the_arc():
    state, p, _ = _shooter(25.0, 0.0)
    # Curling in from the side beyond the post: on the line it left on, wide.
    cand = next(c for c in _bent_shots(p, TOP_CORNER) if c.curl < 0.0)
    pos, vel, vz, spin = _flight(state, cand)
    state.ball.pos, state.ball.velocity, state.ball.vz, state.ball.height = pos, vel, vz, 0.0
    state.ball.curl = spin
    assert gl._shot_is_on_target(state, p)
    state.ball.curl = 0.0
    assert not gl._shot_is_on_target(state, p)
