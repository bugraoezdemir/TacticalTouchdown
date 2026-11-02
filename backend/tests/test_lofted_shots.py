"""Shots in the air: the top corner, the chip, the bar, and what a keeper can reach."""
import numpy as np
import pytest

import app.engine.decisions.candidates as ca
import app.engine.decisions.off_ball as ob
import app.engine.game_loop as gl
import app.engine.physics as ph
from app.engine.ball_flight import chip_launch, drive_launch, height_after, step
from app.engine.constants import (
    CHIP_ANGLE_DEG,
    CHIP_HEIGHT,
    CHIP_MAX_DIST,
    CROSSBAR_HEIGHT,
    GK_CONTROL_RADIUS,
    GK_DIVE_SPEED,
    GK_GATHER_SPEED,
    GK_HANDS_HEIGHT,
    GK_LEAVE_OVER_BAR,
    GK_REACH_HEIGHT,
    GK_REACTION_TICKS,
    GOAL_BOTTOM_Y,
    LOFT_SHOT_MAX_DIST,
    LOFT_SHOT_MIN_DIST,
    SHOT_FRICTION_COEFF,
    SHOT_SPEED,
    TOP_CORNER_HEIGHT,
)
from app.engine.decisions.candidates import _lofted_shots
from app.engine.decisions.interception import control_radius, keeper_reach, lofted_shot_survival, route_survival
from app.engine.decisions.on_ball import _on_frame
from app.engine.factory import build_test_state

TOP_CORNER = np.array([105.0, GOAL_BOTTOM_Y - 0.5])


def _clear_pitch(state):
    for i, p in enumerate(state.all_players()):
        p.pos = np.array([2.0 + i, 2.0])
        p.velocity = np.zeros(2)
        p.has_ball = False


def _shooter(depth=20.0, wide=0.0):
    """Home attacks x = 105: its carrier `depth` m out, the away keeper on his line."""
    state = build_test_state(seed=1)
    _clear_pitch(state)
    assert state.home_team.direction == 1
    p = next(q for q in state.home_team.players if q.position != "GK")
    p.pos = np.array([105.0 - depth, 34.0 + wide])
    keeper = next(q for q in state.away_team.players if q.position == "GK")
    keeper.pos = np.array([104.0, 34.0])
    return state, p, keeper


def _height_on_arrival(vh, vz, dist):
    pos, vel, h = np.zeros(2), np.array([vh, 0.0]), 0.0
    while pos[0] < dist:
        pos, vel, h, vz = step(pos, vel, h, vz, SHOT_FRICTION_COEFF)
    return h, vz


# -- Where the shot goes -----------------------------------------------------

@pytest.mark.parametrize("dist", [12.0, 20.0, 30.0, 40.0])
def test_a_top_corner_shot_crosses_under_the_bar(dist):
    vh, vz, _ = drive_launch(dist, TOP_CORNER_HEIGHT, SHOT_SPEED)
    assert np.hypot(vh, vz) == pytest.approx(SHOT_SPEED, abs=1e-2)
    height = _height_on_arrival(vh, vz, dist)[0]
    assert height == pytest.approx(TOP_CORNER_HEIGHT, abs=0.25)
    assert height < CROSSBAR_HEIGHT


def test_it_is_no_faster_than_a_shot_along_the_ground_when_struck():
    vh, vz, _ = drive_launch(20.0, TOP_CORNER_HEIGHT, SHOT_SPEED)
    assert np.hypot(vh, vz) <= SHOT_SPEED + 1e-9


@pytest.mark.parametrize("dist", [10.0, 18.0, 28.0])
def test_a_chip_is_coming_down_under_the_bar(dist):
    vh, vz, _ = chip_launch(dist, CHIP_HEIGHT, CHIP_ANGLE_DEG, SHOT_SPEED)
    height, falling = _height_on_arrival(vh, vz, dist)
    assert height < CROSSBAR_HEIGHT
    assert falling < 0.0


def test_a_chip_that_needs_more_than_a_foot_can_give_is_not_on():
    assert chip_launch(200.0, CHIP_HEIGHT, CHIP_ANGLE_DEG, SHOT_SPEED) is None


@pytest.mark.parametrize("depth,count", [(5.0, 0), (20.0, 2), (35.0, 1), (60.0, 0)])
def test_range_of_the_two_shots(depth, count):
    state, p, _ = _shooter(depth)
    assert len(_lofted_shots(p, TOP_CORNER, corner=True)) == count


def test_the_top_of_the_goal_means_its_corners():
    state, p, _ = _shooter(35.0)
    assert _lofted_shots(p, np.array([105.0, 34.0]), corner=False) == []


def test_range_constants_hold_together():
    assert LOFT_SHOT_MIN_DIST < CHIP_MAX_DIST < LOFT_SHOT_MAX_DIST


def test_keepers_do_not_shoot_in_the_air():
    state, _, keeper = _shooter()
    home_keeper = next(q for q in state.home_team.players if q.position == "GK")
    home_keeper.pos = np.array([80.0, 34.0])
    shots = [c for c in ca.build_candidates(home_keeper, state) if c.kind == "shot"]
    assert shots and all(c.vz == 0.0 for c in shots)


def test_the_switch(monkeypatch):
    state, p, _ = _shooter()
    monkeypatch.setattr(ca, "LOFTED_SHOTS", False)
    assert all(c.vz == 0.0 for c in ca.build_candidates(p, state) if c.kind == "shot")


# -- The trade-off: the bar --------------------------------------------------

def test_the_frame_gets_harder_to_hit_from_further_out():
    state, near, _ = _shooter(14.0)
    close = _on_frame(_lofted_shots(near, TOP_CORNER, corner=True)[0], near)
    near.pos = np.array([105.0 - 38.0, 34.0])
    assert _on_frame(_lofted_shots(near, TOP_CORNER, corner=True)[0], near) < close


def test_a_better_shooter_hits_it_more_often():
    state, p, _ = _shooter(30.0)
    cand = _lofted_shots(p, TOP_CORNER, corner=True)[0]
    p.shooting = 50.0
    poor = _on_frame(cand, p)
    p.shooting = 95.0
    assert _on_frame(cand, p) > poor


def test_lift_error_is_nothing_at_skill_100_and_real_below_it():
    state, p, _ = _shooter()
    p.shooting = 100.0
    assert gl._lift_error(0.2, p, state) == 0.2
    p.shooting = 50.0
    assert len({gl._lift_error(0.2, p, state) for _ in range(5)}) == 5


def test_over_the_bar_is_not_on_target():
    state, p, _ = _shooter(20.0)
    ball = state.ball
    ball.pos = p.pos.copy()
    ball.height = 0.0
    ball.velocity = np.array([1.0, 0.0])
    ball.vz = 0.1
    assert gl._shot_is_on_target(state, p)
    ball.vz = 0.5
    assert not gl._shot_is_on_target(state, p)


def test_over_the_bar_is_no_goal():
    state, _, _ = _shooter()
    ball = state.ball
    ball.pos = np.array([105.5, 34.0])
    ball.velocity = np.array([1.0, 0.0])
    ball.height = CROSSBAR_HEIGHT + 0.5
    ball.state = "in_shot"
    ball.owner_id = None
    score = state.home_score
    ph.resolve_possession(state)
    assert state.home_score == score


# -- What the keeper can reach -----------------------------------------------

def test_his_reach_is_unchanged_low_and_narrows_above_his_chest():
    state, _, keeper = _shooter()
    full = control_radius(keeper)
    assert keeper_reach(keeper, 0.0) == full
    assert keeper_reach(keeper, GK_HANDS_HEIGHT) == full
    assert keeper_reach(keeper, TOP_CORNER_HEIGHT) < 0.9 * full
    assert keeper_reach(keeper, GK_REACH_HEIGHT) == 0.0


def _high_shot(state, y=34.0, height=2.1):
    ball = state.ball
    ball.state = "in_shot"
    ball.owner_id = None
    ball.pos = np.array([92.0, y])
    ball.velocity = np.array([1.0, 0.0])
    ball.height = height
    ball.vz = 0.05
    ball.struck_tick = state.tick
    ball.struck_from = ball.pos.copy()
    return ball


def test_he_reads_a_high_shot_before_he_leaps():
    state, _, keeper = _shooter(14.0)
    ball = _high_shot(state, y=36.5)
    action = ob.off_ball_decision(keeper, state)
    assert action.dive and action.pace == 0.0
    ball.struck_tick = state.tick - int(GK_REACTION_TICKS)
    action = ob.off_ball_decision(keeper, state)
    assert action.dive and action.pace == GK_DIVE_SPEED


def test_along_the_ground_he_goes_at_once_and_flat_out():
    state, _, keeper = _shooter(14.0)
    ball = _high_shot(state, y=36.5, height=0.0)
    ball.vz = 0.0
    action = ob.off_ball_decision(keeper, state)
    assert action.dive and action.pace is None


def _top_corner_outcome(depth):
    """A shot at the top corner, struck from straight in front, into an empty
    pitch but for the keeper in the middle of his goal: goal?"""
    state, p, keeper = _shooter(depth)
    keeper.pos = np.array([103.0, 34.0])
    cand = _lofted_shots(p, TOP_CORNER, corner=True)[0]
    ball = state.ball
    ball.state = "in_shot"
    ball.owner_id = None
    ball.pos = p.pos.copy()
    ball.height = 0.0
    ball.vz = cand.vz
    ball.velocity = (cand.target - p.pos) / np.linalg.norm(cand.target - p.pos) * cand.speed
    ball.struck_tick = state.tick
    ball.struck_from = ball.pos.copy()
    ball.last_toucher_team_id = state.home_team.team_id
    ball.last_shooter_id = p.player_id
    score = state.home_score
    for _ in range(60):
        gl.tick(state)
        if state.home_score != score:
            return True
        if ball.owner_id is not None or state.stats.away.saves or state.restart_pause > 0:
            return False
    return False


def test_from_close_the_top_corner_is_out_of_his_reach():
    assert _top_corner_outcome(14.0)


def test_from_far_he_gets_there():
    assert not _top_corner_outcome(32.0)


def test_the_plan_sees_the_same_keeper():
    state, p, keeper = _shooter(14.0)
    keeper.pos = np.array([103.0, 34.0])
    cand = _lofted_shots(p, TOP_CORNER, corner=True)[0]
    close = lofted_shot_survival(p.pos, TOP_CORNER, cand.speed, cand.vz, [keeper])
    state, p, keeper = _shooter(32.0)
    keeper.pos = np.array([103.0, 34.0])
    cand = _lofted_shots(p, TOP_CORNER, corner=True)[0]
    far = lofted_shot_survival(p.pos, TOP_CORNER, cand.speed, cand.vz, [keeper])
    assert close > 0.5 > far


def test_the_keeper_leaves_one_going_over():
    state, p, keeper = _shooter(10.0)
    ball = state.ball
    ball.state = "in_shot"
    ball.pos = np.array([100.0, 34.0])
    ball.velocity = np.array([1.0, 0.0])
    ball.height = 3.5
    ball.vz = 0.0
    assert height_after(ball.height, ball.vz, 5.0) > CROSSBAR_HEIGHT + GK_LEAVE_OVER_BAR
    assert ob._predict_goal_crossing_y(ball, 105.0) is None
    ball.height = 2.0
    assert ob._predict_goal_crossing_y(ball, 105.0) == pytest.approx(34.0)


def test_a_chip_beats_a_keeper_who_has_come_out_and_not_one_on_his_line():
    state, p, keeper = _shooter(18.0)
    target = np.array([105.0, 34.0])
    chip = _lofted_shots(p, target, corner=False)[0]
    on_his_line = lofted_shot_survival(p.pos, target, chip.speed, chip.vz, [keeper])
    keeper.pos = np.array([99.0, 34.0])
    keeper.velocity = np.array([-0.25, 0.0])
    out = lofted_shot_survival(p.pos, target, chip.speed, chip.vz, [keeper])
    assert out > on_his_line + 0.5


def test_a_man_under_a_chip_is_less_in_the_way_than_of_a_shot_along_the_ground():
    state, p, keeper = _shooter(18.0)
    target = np.array([105.0, 34.0])
    chip = _lofted_shots(p, target, corner=False)[0]
    d = next(q for q in state.away_team.players if q.position != "GK")
    d.pos = np.array([95.0, 34.0])
    along_the_ground = route_survival(p.pos, target, SHOT_SPEED, SHOT_FRICTION_COEFF, [d], is_shot=True)
    assert lofted_shot_survival(p.pos, target, chip.speed, chip.vz, [d]) > along_the_ground + 0.3


def test_a_keeper_caught_out_runs_back_for_a_ball_over_his_head():
    state, _, keeper = _shooter(14.0)
    keeper.pos = np.array([97.0, 34.0])
    ball = state.ball
    ball.state = "in_shot"
    ball.owner_id = None
    ball.pos = np.array([95.0, 34.0])
    ball.velocity = np.array([0.4, 0.0])
    ball.height = 4.0
    ball.vz = -0.05
    action = ob.off_ball_decision(keeper, state)
    assert not action.dive and action.through


def test_a_parry_is_as_it_was():
    """No rebounds: slower than he can gather, it is his again next tick."""
    state, _, keeper = _shooter()
    ball = state.ball
    ball.state = "in_shot"
    ball.owner_id = None
    ball.velocity = np.array([GK_GATHER_SPEED + 0.05, 0.0])
    ball.pos = keeper.pos - np.array([0.5, 0.0])
    ball.height = 0.5
    ball.vz = 0.0
    ball.last_toucher_team_id = state.home_team.team_id
    state.ball_prev_pos = None
    ph.resolve_possession(state)
    assert state.ball.owner_id is None
    state.ball_prev_pos = state.ball.pos.copy()
    state.tick += 1
    ph.resolve_possession(state)
    assert state.ball.owner_id == keeper.player_id
