"""
A ball just struck from a player's feet (physics.resolve_possession).

The receiver takes the ball to his feet, so his next kick leaves from there.
An opponent right beside him then had it inside his 1 m reach at once, took
it, and the two traded it every tick; a teammate beside him did the same with
passes meant for someone else. For STRIKE_REACT_TICKS an opponent can only
block it (in its path, ahead of the kicker's foot) and a teammate it was not
played to lets it run.
"""
import numpy as np

from app.engine.constants import CAPTURE_RADIUS
from app.engine.factory import build_test_state
from app.engine.physics import _give_possession, resolve_possession


def _strike(near: np.ndarray, same_team: bool = False, receiver_far: bool = True):
    """A kicker at (60, 34) passes along +x; one other player at `near`."""
    state = build_test_state()
    kicker = next(p for p in state.home_team.players if p.position == "CM")
    others = state.home_team.players if same_team else state.away_team.players
    other = next(p for p in others if p is not kicker and p.position != "GK")
    for p in state.all_players():
        p.pos = np.array([5.0, 5.0]) if p.team_id == 1 else np.array([100.0, 60.0])
        p.velocity = np.zeros(2)
    kicker.pos = np.array([60.0, 34.0])
    other.pos = kicker.pos + near
    _give_possession(state, state.ball, kicker)
    ball = state.ball
    ball.pos = kicker.pos.copy()
    # Struck as game_loop strikes it: lockout, origin, then advanced.
    ball.owner_id = None
    kicker.has_ball = False
    ball.state = "in_pass"
    ball.velocity = np.array([0.6, 0.0])
    ball.last_toucher_team_id = kicker.team_id
    ball.last_passer_id = kicker.player_id
    ball.last_pass_tick = state.tick
    ball.intended_receiver_id = None if receiver_far else other.player_id
    ball.no_capture_id = kicker.player_id
    ball.no_capture_until = state.tick + 4
    ball.struck_from = ball.pos.copy()
    ball.struck_tick = state.tick
    ball.pos = ball.pos + np.array([CAPTURE_RADIUS + 0.3, 0.0])
    state.ball_prev_pos = None
    resolve_possession(state)
    return state, other


def test_an_opponent_beside_the_kicker_cannot_take_it_at_once():
    state, other = _strike(np.array([0.0, 0.15]))
    assert state.ball.owner_id is None


def test_an_opponent_behind_the_kicker_cannot_take_it_at_once():
    state, other = _strike(np.array([-0.3, 0.2]))
    assert state.ball.owner_id is None


def test_an_opponent_in_its_path_blocks_it_and_it_comes_off_him_loose():
    state, other = _strike(np.array([0.6, 0.1]))
    ball = state.ball
    # Nobody has it: at his feet it sat inside the kicker's reach, and the two
    # blocked each other's kicks every tick.
    assert ball.owner_id is None
    assert ball.last_toucher_team_id == other.team_id
    assert 0.0 < float(np.linalg.norm(ball.velocity)) < 0.6 * 0.5
    assert ball.velocity[0] < 0.0          # back the way it came, off his body
    assert ball.no_capture_id == other.player_id


def test_without_deflection_the_blocker_takes_it(monkeypatch):
    import app.engine.physics as physics
    monkeypatch.setattr(physics, "BLOCK_DEFLECT", False)
    state, other = _strike(np.array([0.6, 0.1]))
    assert state.ball.owner_id == other.player_id


def test_a_teammate_beside_him_lets_a_pass_for_someone_else_run():
    state, other = _strike(np.array([0.0, 0.15]), same_team=True)
    assert state.ball.owner_id is None


def test_the_intended_receiver_still_takes_it():
    state, other = _strike(np.array([1.2, 0.3]), same_team=True, receiver_far=False)
    assert state.ball.owner_id == other.player_id
