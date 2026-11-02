"""
Defenders get goal-side of the ball along its line to OUR GOAL and into the
shooting triangle (ball to our posts) -- not to a line across the pitch
behind it, which with the ball wide near our goal left them beside the goal,
blocking nothing.

Stations (zone centres) are what is tested: they come from the formation, so
a player is placed by choosing who he is, not where he stands.
"""
import numpy as np

from app.engine.factory import build_test_state
from app.engine.decisions import off_ball as OB


def _carrier_at_the_edge_of_our_box(monkeypatch, triangle):
    monkeypatch.setattr(OB, "TRIANGLE_RECOVERY", triangle)
    state = build_test_state()
    home, away = state.home_team, state.away_team
    gx = 0.0 if home.direction == 1 else 105.0
    carrier = next(p for p in away.players if p.position == "RM")
    for p in state.all_players():
        p.has_ball = False
    carrier.pos = np.array([abs(gx - 16.0), 45.0])   # edge of our box, at an angle
    carrier.has_ball = True
    state.ball.pos = carrier.pos.copy()
    state.ball.owner_id = carrier.player_id
    return state, home, gx


def _off_the_shooting_line(centre, ball, goal_x):
    goal = np.array([goal_x, 34.0])
    u = (goal - ball) / np.hypot(*(goal - ball))
    rel = centre - ball
    return float(np.hypot(*(rel - u * (rel @ u)))), float(rel @ u)


def test_defenders_close_the_shooting_angle_better_than_the_old_line(monkeypatch):
    state, home, gx = _carrier_at_the_edge_of_our_box(monkeypatch, False)
    outfield = [p for p in home.players if p.position != "GK" and not OB._is_outlet(p, state, home)]
    old = [OB._zone_centre(p, state, home)[0] for p in outfield]
    monkeypatch.setattr(OB, "TRIANGLE_RECOVERY", True)
    new = [OB._zone_centre(p, state, home)[0] for p in outfield]
    off_old = [_off_the_shooting_line(c, state.ball.pos, gx)[0] for c in old]
    off_new = [_off_the_shooting_line(c, state.ball.pos, gx)[0] for c in new]
    # Nearer the shooting line on average, and the widest man much nearer: the
    # old line across the pitch left a full-back 11-12 m off it, beside the goal.
    assert np.mean(off_new) < np.mean(off_old)
    assert max(off_new) < max(off_old)


def test_far_from_goal_the_shape_is_left_alone(monkeypatch):
    """No danger, no pull: only recovering players move."""
    state, home, gx = _carrier_at_the_edge_of_our_box(monkeypatch, True)
    state.ball.pos = np.array([52.5, 34.0])
    for p in home.players:
        if p.position == "GK" or OB._is_outlet(p, state, home):
            continue
        monkeypatch.setattr(OB, "TRIANGLE_RECOVERY", False)
        old, rec_old = OB._zone_centre(p, state, home)
        monkeypatch.setattr(OB, "TRIANGLE_RECOVERY", True)
        new, rec_new = OB._zone_centre(p, state, home)
        if not rec_old and not rec_new:
            assert np.allclose(old, new)
