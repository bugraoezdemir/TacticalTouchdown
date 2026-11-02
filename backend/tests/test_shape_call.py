"""The back caller's shape call: the formation moved and squeezed as one object."""
import numpy as np
import pytest

import app.engine.decisions.off_ball as ob
import app.engine.decisions.shape_call as sc
import app.engine.game_loop as gl
from app.engine.constants import (
    BLOCK_LINE_MIN,
    GOAL_CENTER_Y,
    SHAPE_CALL_BALL_M,
    SHAPE_CALL_EASE_TICKS,
    SHAPE_CALL_HEAR_TICKS,
    SHAPE_CALL_MIN_TICKS,
    SHAPE_CALL_TICKS,
)
from app.engine.factory import build_test_state
from app.engine.formations import FORMATIONS, apply_formation
from app.engine.models import TeamTactics

BOX_HALF_WIDTH = 20.16


def _scene(formation="4-4-2", depth=27.0, wide=25.0):
    """Home defending, with the ball `depth` m from its line, `wide` m off centre."""
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.formation = formation
    apply_formation(team)
    own = 0.0 if team.direction == 1 else 105.0
    state.ball.pos = np.array([own + team.direction * depth, GOAL_CENTER_Y + wide])
    return state, team


def _stations(state, team):
    call = sc.make_call(state, team)
    return {p.player_id: sc.called_station(p, call, team, state.tick) for p in team.players if p.position != "GK"}, call


def _depth(point, team):
    return sc._depth(point, team)


@pytest.mark.parametrize("formation", sorted(FORMATIONS))
@pytest.mark.parametrize("depth,wide", [(12, 28), (27, 25), (27, 3), (45, 25), (80, 0)])
def test_slots_level_in_the_formation_stay_level(formation, depth, wide):
    """No arc: the same depth in the formation is the same depth on the pitch,
    whichever way the wide men's distance from goal lies."""
    state, team = _scene(formation, depth, wide)
    stations, call = _stations(state, team)
    by_slot = {}
    for p in team.players:
        if p.position != "GK" and p.player_id != call.outlet_id:
            key = round(_depth(p.assigned_pos, team), 3)
            by_slot.setdefault(key, []).append(_depth(stations[p.player_id], team))
    for depths in by_slot.values():
        assert max(depths) - min(depths) < 1e-9


@pytest.mark.parametrize("formation", sorted(FORMATIONS))
@pytest.mark.parametrize("depth,wide", [(12, 28), (27, 25), (45, 25)])
def test_every_formation_keeps_its_order(formation, depth, wide):
    """It stays that formation: who is deeper and who is further across holds."""
    state, team = _scene(formation, depth, wide)
    stations, call = _stations(state, team)
    men = [p for p in team.players if p.position != "GK" and p.player_id != call.outlet_id]
    for a in men:
        for b in men:
            if _depth(a.assigned_pos, team) < _depth(b.assigned_pos, team) - 1e-6:
                assert _depth(stations[a.player_id], team) < _depth(stations[b.player_id], team)
            if a.assigned_pos[1] < b.assigned_pos[1] - 1e-6:
                assert stations[a.player_id][1] < stations[b.player_id][1]


@pytest.mark.parametrize("wide", [-28, 28])
def test_centre_backs_stay_in_front_of_goal_with_the_ball_wide_near_it(wide):
    state, team = _scene("4-4-2", 12.0, wide)
    stations, _ = _stations(state, team)
    backs = [stations[p.player_id] for p in team.players if p.position == "CD"]
    assert all(abs(s[1] - GOAL_CENTER_Y) < BOX_HALF_WIDTH for s in backs)
    assert abs(backs[0][1] - backs[1][1]) < 10.5


def test_the_near_full_back_stays_on_the_ball_side_and_above_the_line_minimum():
    state, team = _scene("4-4-2", 27.0, 25.0)
    stations, call = _stations(state, team)
    full_back = max((p for p in team.players if p.position in ("LD", "RD")),
                    key=lambda p: stations[p.player_id][1])
    assert stations[full_back.player_id][1] - GOAL_CENTER_Y > 18.0
    assert _depth(stations[full_back.player_id], team) >= BLOCK_LINE_MIN
    assert call.line > 14.0


def test_a_call_is_heard_late_and_holds_between_calls():
    state, team = _scene()
    sc.update_shape_calls(state)
    first = state.shape_calls[team.team_id]
    state.tick += SHAPE_CALL_TICKS
    sc.update_shape_calls(state)
    assert state.shape_calls[team.team_id] is first
    waiting = state.shape_calls_waiting[team.team_id]
    state.tick += SHAPE_CALL_HEAR_TICKS - 1
    sc.update_shape_calls(state)
    assert state.shape_calls[team.team_id] is first
    state.tick += 1
    sc.update_shape_calls(state)
    assert state.shape_calls[team.team_id] is waiting


def test_the_shape_glides_onto_a_new_call():
    state, team = _scene("4-4-2", 45.0, 0.0)
    player = next(p for p in team.players if p.position == "CD")
    before = sc.make_call(state, team)
    state.ball.pos = state.ball.pos + np.array([-team.direction * 18.0, 0.0])
    call = sc.make_call(state, team)
    call.previous = before
    call.heard_tick = 100
    start = sc.called_station(player, call, team, 100)
    assert np.allclose(start, sc.called_station(player, before, team, 100))
    step = sc.called_station(player, call, team, 101) - start
    end = sc.called_station(player, call, team, 100 + SHAPE_CALL_EASE_TICKS)
    assert np.hypot(*step) < 0.1 * np.hypot(*(end - start))
    assert np.allclose(end, sc.called_station(player, call, team, 500))


def test_a_long_ball_brings_the_call_forward():
    state, team = _scene()
    sc.update_shape_calls(state)
    state.tick += SHAPE_CALL_MIN_TICKS
    state.ball.pos = state.ball.pos + np.array([SHAPE_CALL_BALL_M + 1.0, 0.0])
    sc.update_shape_calls(state)
    assert team.team_id in state.shape_calls_waiting


def test_the_station_is_used_only_out_of_possession():
    state, team = _scene()
    sc.update_shape_calls(state)
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.away_team.team_id
    player = next(p for p in team.players if p.position == "CD")
    call = state.shape_calls[team.team_id]
    centre, _ = ob._zone_centre_compute(player, state, team)
    assert np.allclose(centre, ob._hold_onside(sc.called_station(player, call, team, state.tick),
                                               ob.onside_limit(state, team), team.direction))
    state.ball.last_toucher_team_id = team.team_id
    centre, _ = ob._zone_centre_compute(player, state, team)
    assert not np.allclose(centre, sc.called_station(player, call, team, state.tick))


def test_depth_still_moves_the_shape():
    state, team = _scene("4-4-2", 60.0, 0.0)
    low = sc.make_call(state, team).line
    team.tactics = TeamTactics(offensiveness=1.8)
    apply_formation(team)
    assert sc.make_call(state, team).line > low


def test_a_match_runs_with_calls_on_both_sides():
    state = build_test_state(seed=2)
    for _ in range(300):
        gl.tick(state)
    assert set(state.shape_calls) == {state.home_team.team_id, state.away_team.team_id}
