"""The man in front of a carrier running at our goal steps up to him."""
import numpy as np

import app.engine.decisions.off_ball as ob
from app.engine.constants import FRONT_MAN_TICKS, LEFT_GOAL_X, RIGHT_GOAL_X
from app.engine.factory import build_test_state


def _scene(ahead=8.0, speed=0.15):
    """A home defender `ahead` metres goal-side of an away carrier 45 m out,
    running at our goal, with a home chaser just behind him."""
    state = build_test_state(seed=1)
    team = state.home_team
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    sign = 1.0 if own_goal_x == LEFT_GOAL_X else -1.0
    carrier = next(p for p in state.away_team.players if p.position == "CM")
    carrier.pos = np.array([own_goal_x + sign * 45.0, 34.0])
    carrier.velocity = np.array([-sign * speed, 0.0])
    outfield = [p for p in team.players if p.position != "GK"]
    front, chaser = outfield[0], outfield[1]
    front.pos = carrier.pos + np.array([-sign * ahead, 1.0])
    chaser.pos = carrier.pos + np.array([sign * 2.0, 0.0])
    return state, team, carrier, front, chaser, own_goal_x


def _may_go(state, team, carrier, front, chaser, own_goal_x):
    t_arrive = ob._ticks_to_reach(front, state.ball.pos)
    return t_arrive, ob._may_leave_zone(front, state, team, carrier, own_goal_x, 0.0,
                                        carrier.pos, t_arrive, [chaser.player_id])


def test_he_steps_up_beyond_the_danger_zone():
    state, team, carrier, front, chaser, own_goal_x = _scene()
    state.ball.pos = carrier.pos.copy()
    t_arrive, go = _may_go(state, team, carrier, front, chaser, own_goal_x)
    assert t_arrive <= FRONT_MAN_TICKS
    assert go


def test_not_if_the_carrier_is_not_running_at_goal():
    state, team, carrier, front, chaser, own_goal_x = _scene(speed=0.0)
    state.ball.pos = carrier.pos.copy()
    assert not _may_go(state, team, carrier, front, chaser, own_goal_x)[1]


def test_not_from_too_far_away():
    state, team, carrier, front, chaser, own_goal_x = _scene(ahead=25.0)
    state.ball.pos = carrier.pos.copy()
    t_arrive, go = _may_go(state, team, carrier, front, chaser, own_goal_x)
    assert t_arrive > FRONT_MAN_TICKS
    assert not go


def test_a_chaser_behind_him_does_not_stand_the_front_man_down():
    state, team, carrier, front, chaser, own_goal_x = _scene()
    state.ball.pos = carrier.pos.copy()
    t_arrive = ob._ticks_to_reach(front, carrier.pos)
    committed = [(chaser.player_id, carrier.pos.copy())]
    assert ob._should_engage(front, carrier.pos, t_arrive, committed, state, 1.0,
                             carrier=carrier, own_goal_x=own_goal_x, danger=0.0)
