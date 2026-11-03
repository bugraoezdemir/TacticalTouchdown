"""A marker keeps his man until the man is clearly out of his reach."""
import numpy as np
import pytest

import app.engine.decisions.off_ball as ob
import app.engine.game_loop as gl
from app.engine.constants import MARK_KEEP
from app.engine.decisions.off_ball import _zonal_assignment_compute, _zonal_mark_detail, _zone_reach
from app.engine.factory import build_test_state


@pytest.fixture
def scene():
    """Away defends x = 105: one marker, one man, everyone else far off."""
    state = build_test_state(seed=1)
    carrier = next(p for p in state.home_team.players if p.position != "GK")
    man = next(p for p in state.home_team.players if p.position != "GK" and p is not carrier)
    marker = next(p for p in state.away_team.players if p.position != "GK")
    for p in state.home_team.players:
        p.has_ball = False
        if p is not man:
            p.pos = np.array([2.0, 2.0])
    carrier.pos = np.array([60.0, 34.0])
    carrier.has_ball = True
    state.ball.owner_id = carrier.player_id
    man.pos = np.array([92.0, 34.0])
    for p in state.away_team.players:
        if p is not marker and p.position != "GK":
            p.pos = np.array([2.0, 66.0])
    return state, carrier, man, marker


def _assigned(state, carrier, marker):
    team = state.away_team
    centres = {p.player_id: p.pos.copy() for p in team.players if p.position != "GK"}
    return _zonal_assignment_compute(state, team, carrier, set(), centres)[0].get(marker.player_id)


def _stand_off(man, marker, reach, share):
    marker.pos = man.pos + np.array([reach * share, 0.0])


def test_he_keeps_his_man_past_his_reach(scene):
    state, carrier, man, marker = scene
    _stand_off(man, marker, _zone_reach(marker, state.away_team, 0.0), 1.15)
    assert _assigned(state, carrier, marker) is None
    marker.marking_id = man.player_id
    assert _assigned(state, carrier, marker) is man


def test_he_lets_him_go_beyond_the_keep(scene):
    state, carrier, man, marker = scene
    _stand_off(man, marker, _zone_reach(marker, state.away_team, 0.0), MARK_KEEP * 1.05)
    marker.marking_id = man.player_id
    assert _assigned(state, carrier, marker) is None


def test_the_man_on_the_ball_is_nobodys_mark(scene):
    state, carrier, man, marker = scene
    marker.pos = carrier.pos + np.array([1.0, 0.0])
    marker.marking_id = carrier.player_id
    assert _assigned(state, carrier, marker) is not carrier


def test_two_men_cannot_keep_the_same_man(scene):
    state, carrier, man, marker = scene
    other = next(p for p in state.away_team.players if p.position != "GK" and p is not marker)
    marker.pos = man.pos + np.array([1.0, 0.0])
    other.pos = man.pos + np.array([0.0, 1.0])
    marker.marking_id = other.marking_id = man.player_id
    team = state.away_team
    centres = {p.player_id: p.pos.copy() for p in team.players if p.position != "GK"}
    marks = _zonal_assignment_compute(state, team, carrier, set(), centres)[0]
    assert sum(m is man for m in marks.values()) == 1


def test_his_marking_stays_firm_out_to_the_keep(scene, monkeypatch):
    state, carrier, man, marker = scene
    monkeypatch.setattr(ob, "_zone_centre", lambda p, *a: (p.pos.copy(), False))
    _stand_off(man, marker, _zone_reach(marker, state.away_team, 0.0), 1.15)
    marker.marking_id = man.player_id
    spot, strength, kept = _zonal_mark_detail(marker, state, state.away_team, carrier, 105.0, set())
    assert kept is man and strength > 0.0


def test_the_switch_turns_it_off(scene, monkeypatch):
    state, carrier, man, marker = scene
    monkeypatch.setattr(ob, "STICKY_MARKS", False)
    _stand_off(man, marker, _zone_reach(marker, state.away_team, 0.0), 1.15)
    marker.marking_id = man.player_id
    assert _assigned(state, carrier, marker) is None


def test_it_is_remembered_from_tick_to_tick():
    state = build_test_state(seed=3)
    seen = False
    for _ in range(400):
        gl.tick(state)
        seen = seen or any(p.marking_id is not None for p in state.all_players())
    assert seen
    assert all(p.marking_id is None for p in state.all_players() if p.has_ball)
