"""
Compactness and the block's follow of the ball (off_ball._shape_anchor).

The follow used to be capped at 0.95, which neutral already reached, so no
compactness above ~0.94 made the block follow the ball any closer. It is now a
scale: neutral is unchanged, higher follows closer, lower less, and the slide
stays below 0.95 of the ball's offset.
"""
import numpy as np

from app.engine.constants import PITCH_LENGTH, SHAPE_BLOCK_SLIDE_X
from app.engine.decisions.off_ball import _shape_anchor
from app.engine.factory import build_test_state


def _slide(compactness: float, ball_x: float = 90.0) -> float:
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.compactness = compactness
    player = next(p for p in team.players if p.position == "CM")
    state.ball.pos = np.array([ball_x, 34.0])
    anchor = _shape_anchor(player, state.ball, team)        # no state: no onside hold
    return float(anchor[0] - player.assigned_pos[0])


def test_neutral_follow_is_unchanged():
    offset = 90.0 - PITCH_LENGTH / 2.0
    assert abs(_slide(1.0) - offset * SHAPE_BLOCK_SLIDE_X * 0.95) < 1e-9


def test_a_compact_team_follows_the_ball_closer():
    assert _slide(2.0) > _slide(1.5) > _slide(1.0) > _slide(0.5)


def test_the_block_never_overtakes_the_ball():
    offset = 90.0 - PITCH_LENGTH / 2.0
    assert _slide(2.0) <= 0.95 * offset + 1e-9
