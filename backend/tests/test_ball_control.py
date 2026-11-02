"""
Receiving the ball: pace has a price at the receiving end (decisions/control.py).

A ball slower than a player's easy speed is a clean first touch; faster than
that he must play it first time or try to control it, and a failed trap
squirms loose. The passer's P includes the receiver's chance of controlling
it, so a pass he cannot take is not a good pass however safe its lane.
"""
import numpy as np

from app.engine.constants import CONTROL_HARD_SPEED, CONTROL_MIN_P, FIRST_TIME_MIN_Q
from app.engine.decisions.control import (
    control_probability,
    easy_speed,
    first_time_quality,
)
from app.engine.factory import build_test_state
from app.engine.physics import resolve_possession


def _player(state, position="CM"):
    return next(p for p in state.home_team.players if p.position == position)


def test_clean_below_the_easy_speed_and_floored_at_the_hard_speed():
    state = build_test_state()
    p = _player(state)
    assert control_probability(easy_speed(p) * 0.99, p) == 1.0
    assert control_probability(CONTROL_HARD_SPEED, p) == CONTROL_MIN_P
    assert control_probability(CONTROL_HARD_SPEED * 2, p) == CONTROL_MIN_P
    assert first_time_quality(CONTROL_HARD_SPEED, p) == FIRST_TIME_MIN_Q


def test_control_falls_with_pace_and_better_dribblers_take_more():
    state = build_test_state()
    p = _player(state)
    speeds = np.linspace(0.3, 1.2, 10)
    probs = [control_probability(s, p) for s in speeds]
    assert all(a >= b for a, b in zip(probs, probs[1:]))
    poor, great = _player(state), _player(state, "CF")
    poor.dribbling, great.dribbling = 20.0, 90.0
    assert easy_speed(great) > easy_speed(poor)
    assert control_probability(0.6, great) > control_probability(0.6, poor)


def _receive(speed: float):
    """A pass arriving at `speed` m/tick at a player standing still."""
    state = build_test_state()
    receiver = _player(state)
    for p in state.all_players():          # nobody else near the ball
        if p is not receiver:
            p.pos = np.array([5.0, 5.0]) if p.team_id == 1 else np.array([100.0, 60.0])
    receiver.pos = np.array([60.0, 34.0]); receiver.velocity = np.zeros(2)
    for p in state.all_players():
        p.has_ball = False
    ball = state.ball
    ball.owner_id = None
    ball.state = "in_pass"
    ball.pos = receiver.pos + np.array([-0.2, 0.0])
    ball.velocity = np.array([speed, 0.0])
    ball.last_toucher_team_id = receiver.team_id
    state.ball_prev_pos = None
    resolve_possession(state)
    return state, receiver


def test_a_gentle_pass_is_simply_his():
    state, receiver = _receive(0.3)
    assert state.ball.owner_id == receiver.player_id
    assert receiver.first_touch_speed == 0.0


def test_a_fast_pass_leaves_a_first_touch_to_take():
    state, receiver = _receive(0.9)
    assert state.ball.owner_id == receiver.player_id
    assert receiver.first_touch_speed > easy_speed(receiver)
    assert receiver.first_touch_velocity is not None
