"""Man marking: how a defender shares himself between his zone and his man."""
import numpy as np
import pytest

import app.engine.decisions.off_ball as ob
from app.engine.constants import LEFT_GOAL_X, RIGHT_GOAL_X, ZONE_RADIUS
from app.engine.decisions.tunables import defensive_bias, man_marking_bias
from app.engine.factory import build_test_state
from app.engine.geometry import distance


def test_it_has_its_own_setting_and_override():
    state = build_test_state(seed=1)
    team = state.home_team
    player = next(p for p in team.players if p.position == "CM")
    assert man_marking_bias(player, team) == 1.0
    team.tactics.man_marking = 1.6
    assert man_marking_bias(player, team) == pytest.approx(1.6)
    player.bias_man_marking = 0.7
    assert man_marking_bias(player, team) == pytest.approx(0.7)
    player.bias_man_marking = 99.0
    assert man_marking_bias(player, team) == 2.0


def _scene(gap, setting=1.0):
    """A home midfielder alone on his zone centre, one away man `gap` metres
    across from it, the rest out of reach."""
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.man_marking = setting
    player = next(p for p in team.players if p.position == "CM")
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    far_x = LEFT_GOAL_X + RIGHT_GOAL_X - own_goal_x
    for o in state.away_team.players:
        o.pos = np.array([far_x, 34.0])
        o.velocity = np.zeros(2)
    centre = ob._zone_centre(player, state, team, defensive_bias(player, team))[0]
    player.pos = centre.copy()
    player.velocity = np.zeros(2)
    man = next(o for o in state.away_team.players if o.position != "GK")
    # Across the pitch, so he stays as far from goal whatever the gap.
    side = 1.0 if centre[1] < 34.0 else -1.0
    man.pos = centre + np.array([0.0, side * gap])
    exclude = {p.player_id for p in team.players if p.player_id != player.player_id}
    spot, strength, marked = ob._zonal_mark_detail(player, state, team, None, own_goal_x, exclude)
    return spot, strength, marked, man


def _gap_to_man(gap, setting):
    spot, _, marked, man = _scene(gap, setting)
    assert marked is man
    return distance(spot, man.pos)


def test_a_higher_setting_stands_nearer_his_man():
    assert _gap_to_man(5.0, 0.5) > _gap_to_man(5.0, 1.0)
    assert _gap_to_man(9.0, 1.0) > _gap_to_man(9.0, 2.0)


def test_a_higher_setting_follows_him_further():
    assert _scene(ZONE_RADIUS * 0.75, 0.5)[2] is None
    assert _scene(ZONE_RADIUS * 0.75, 1.0)[2] is not None
    assert _scene(ZONE_RADIUS * 1.25, 1.0)[2] is None
    assert _scene(ZONE_RADIUS * 1.25, 2.0)[2] is not None


def test_the_urgency_near_our_goal_takes_a_low_setting_back():
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.man_marking = 0.5
    player = team.players[3]
    assert ob._zone_reach(player, team, 0.0) == ZONE_RADIUS * 0.5
    assert ob._zone_reach(player, team, 1.0) == ZONE_RADIUS


def test_neutral_leaves_the_share_alone():
    # Neutral: reach ZONE_RADIUS whatever the danger, no lean.
    state = build_test_state(seed=1)
    for danger in (0.0, 0.5, 1.0):
        assert ob._zone_reach(state.home_team.players[3], state.home_team, danger) == ZONE_RADIUS
