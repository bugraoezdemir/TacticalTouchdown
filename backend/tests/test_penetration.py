"""
Value is more than the destination.

position_value scores where the ball LANDS. On its own, a ball played in behind
the defence scored the same as a square ball to the same square metre, so
nobody ever tried to break the line -- and dribble direction was chosen without
regard to whether the player was carrying into space or into a crowd.
"""
import numpy as np
import pytest

from app.engine.constants import (
    DRIBBLE_ROOM_FLOOR,
    PENETRATION_MAX,
    PITCH_LENGTH,
)
from app.engine.decisions.candidates import Candidate, build_candidates
from app.engine.decisions.on_ball import _dribble_room, score_candidate
from app.engine.decisions.value import defenders_beaten, penetration_value
from app.engine.factory import build_test_state
from tests.helpers import striker
from app.engine.models import Player

RIGHT = PITCH_LENGTH


def defender(x, y, pid=90):
    return Player(
        player_id=pid, team_id=2, name=f"D{pid}", position="CD", tactical="normal",
        pos=np.array([x, y], dtype=float), velocity=np.zeros(2), speed=0.3,
        strength=70.0, intelligence=70.0, passing=70.0, goalkeeping=40.0,
        assigned_pos=np.array([x, y], dtype=float), marking_target_id=None,
        has_ball=False, runway_signal=None,
    )


# ── Defenders beaten ─────────────────────────────────────────────────────────

def test_forward_ball_beats_the_defenders_it_passes():
    line = [defender(60.0, 20.0, 1), defender(62.0, 34.0, 2), defender(61.0, 48.0, 3)]
    assert defenders_beaten(np.array([50.0, 34.0]), np.array([70.0, 34.0]), line, RIGHT) == 3


def test_square_ball_beats_nobody():
    line = [defender(60.0, 20.0, 1), defender(62.0, 34.0, 2)]
    assert defenders_beaten(np.array([50.0, 20.0]), np.array([50.0, 48.0]), line, RIGHT) == 0


def test_backward_ball_beats_nobody():
    line = [defender(60.0, 34.0, 1)]
    assert defenders_beaten(np.array([70.0, 34.0]), np.array([40.0, 34.0]), line, RIGHT) == 0


def test_defenders_already_behind_are_not_counted_again():
    behind = [defender(30.0, 34.0, 1)]
    assert defenders_beaten(np.array([50.0, 34.0]), np.array([70.0, 34.0]), behind, RIGHT) == 0


def test_penetration_is_direction_aware():
    line = [defender(45.0, 34.0, 1)]
    # Attacking left: progress means DECREASING x.
    assert defenders_beaten(np.array([50.0, 34.0]), np.array([40.0, 34.0]), line, 0.0) == 1
    assert defenders_beaten(np.array([50.0, 34.0]), np.array([60.0, 34.0]), line, 0.0) == 0


# ── The multiplier ───────────────────────────────────────────────────────────

def test_penetration_rewards_line_breaking():
    line = [defender(60.0 + i, 20.0 + i * 12.0, i) for i in range(3)]
    through = penetration_value(np.array([50.0, 34.0]), np.array([75.0, 34.0]), line, RIGHT)
    square = penetration_value(np.array([50.0, 34.0]), np.array([50.0, 48.0]), line, RIGHT)
    assert through > square
    assert square == pytest.approx(1.0)


def test_penetration_is_capped():
    many = [defender(55.0 + i, 34.0, i) for i in range(20)]
    assert penetration_value(
        np.array([50.0, 34.0]), np.array([90.0, 34.0]), many, RIGHT
    ) <= PENETRATION_MAX


def test_penetration_never_below_one():
    line = [defender(60.0, 34.0, 1)]
    assert penetration_value(
        np.array([70.0, 34.0]), np.array([40.0, 34.0]), line, RIGHT
    ) >= 1.0


# ── Dribble direction ────────────────────────────────────────────────────────

def test_dribble_prefers_open_ground():
    """route_survival over a 3 m touch is high in almost every direction, so
    without a room term the direction was picked on destination value alone and
    players carried the ball into crowds."""
    state = build_test_state()
    carrier = striker(state)
    carrier.pos = np.array([80.0, 34.0])
    carrier.has_ball = True
    state.ball.pos = carrier.pos.copy()
    state.ball.owner_id = carrier.player_id

    # Wall directly ahead; open ground behind and to the sides.
    for i, o in enumerate(state.away_team.players[:4]):
        o.pos = np.array([84.0, 28.0 + i * 4.0])
    for o in state.away_team.players[4:]:
        o.pos = np.array([40.0, 34.0])

    opponents = state.opponents_of(carrier.team_id)
    scored = [score_candidate(c, carrier, state) for c in build_candidates(carrier, state)]
    dribbles = [c for c in scored if c.kind == "dribble"]
    best = max(dribbles, key=lambda c: c.utility)
    assert _dribble_room(best, carrier, opponents) > 0.9


def test_dribble_room_penalises_crowds():
    state = build_test_state()
    carrier = striker(state)
    carrier.pos = np.array([50.0, 34.0])
    crowd = [defender(52.0, 34.0, i) for i in range(3)]

    into_crowd = Candidate(kind="dribble", target=np.array([53.0, 34.0]),
                           speed=0.14, friction=0.06)
    into_space = Candidate(kind="dribble", target=np.array([47.0, 34.0]),
                           speed=0.14, friction=0.06)
    assert _dribble_room(into_crowd, carrier, crowd) < _dribble_room(into_space, carrier, crowd)


def test_dribble_room_is_bounded():
    state = build_test_state()
    carrier = striker(state)
    opponents = state.away_team.players
    for c in build_candidates(carrier, state):
        if c.kind == "dribble":
            room = _dribble_room(c, carrier, opponents)
            assert DRIBBLE_ROOM_FLOOR <= room <= 1.0


# ── Action-mix balance ───────────────────────────────────────────────────────

@pytest.mark.slow
def test_players_still_dribble():
    """Regression guard on a compounding failure.

    DRIBBLE_CONTINUATION_DISCOUNT was tuned before penetration_value existed.
    Once passes could earn a 2.2x line-breaking bonus that a 3 m touch almost
    never earns, the compounded gap reached 5-6x and dribbling stopped
    happening entirely -- the argmax picked a pass on 100% of touches.
    """
    from app.engine.game_loop import tick

    state = build_test_state()
    state.rng = np.random.default_rng(20260903)
    touches = {"in_pass": 0, "in_dribble": 0}
    previous = state.ball.state
    for _ in range(4500):
        state = tick(state)
        if state.ball.state != previous and state.ball.state in touches:
            touches[state.ball.state] += 1
        previous = state.ball.state

    total = touches["in_pass"] + touches["in_dribble"]
    assert total > 100, "the ball is barely being played at all"
    share = touches["in_dribble"] / total
    assert 0.10 < share < 0.60, f"dribble share {share:.0%} is not football"


def test_dribble_and_pass_multipliers_stay_comparable():
    """The two continuation discounts must not drift far apart: a dribble is a
    slower, safer way to move the same ball, not a categorically worse one."""
    from app.engine.constants import (
        DRIBBLE_CONTINUATION_DISCOUNT,
        PASS_CONTINUATION_DISCOUNT,
    )

    dribble_v = PASS_CONTINUATION_DISCOUNT * DRIBBLE_CONTINUATION_DISCOUNT
    # A pass may additionally earn penetration; a short dribble rarely does.
    # Guard the base gap so that bonus cannot compound into a wipe-out.
    assert dribble_v / PASS_CONTINUATION_DISCOUNT > 0.5
