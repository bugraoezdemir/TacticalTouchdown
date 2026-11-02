"""
Who is between the ball and the goal.

Two related ideas the model was missing:
  - a pass that puts MORE defenders in front of the ball has made the attack
    worse, however good the destination looks in isolation (a penalty, not a veto)
  - a carrier with a clear lane should keep running until he can shoot, because
    passing there usually hands the ball to someone with more defenders ahead
"""
import numpy as np
import pytest

from app.engine.constants import (
    CLEAR_PATH_BONUS,
    GOALSIDE_MIN,
    PITCH_LENGTH,
)
from app.engine.decisions.candidates import build_candidates
from app.engine.decisions.on_ball import score_candidate
from app.engine.decisions.value import (
    _effective_blockers,
    clear_path_bonus,
    goalside_value,
    opponents_goalside,
)
from app.engine.factory import build_test_state
from tests.helpers import midfielder, striker
from app.engine.models import Player

RIGHT = PITCH_LENGTH


def opp(x, y, pid=1, position="CD"):
    return Player(
        player_id=pid, team_id=2, name=f"O{pid}", position=position, tactical="normal",
        pos=np.array([x, y], dtype=float), velocity=np.zeros(2), speed=0.3,
        strength=70.0, intelligence=70.0, passing=70.0, goalkeeping=40.0,
        assigned_pos=np.array([x, y], dtype=float), marking_target_id=None,
        has_ball=False, runway_signal=None,
    )


# ── Counting ─────────────────────────────────────────────────────────────────

def test_counts_only_opponents_between_point_and_goal():
    line = [opp(80.0, 34.0, 1), opp(60.0, 34.0, 2)]
    assert opponents_goalside(np.array([70.0, 34.0]), line, RIGHT) == 1


def test_goalkeeper_is_excluded_by_default():
    keeper = [opp(104.0, 34.0, 1, position="GK")]
    assert opponents_goalside(np.array([90.0, 34.0]), keeper, RIGHT) == 0
    assert opponents_goalside(np.array([90.0, 34.0]), keeper, RIGHT, include_gk=True) == 1


def test_counting_is_direction_aware():
    line = [opp(40.0, 34.0, 1)]
    assert opponents_goalside(np.array([50.0, 34.0]), line, 0.0) == 1
    assert opponents_goalside(np.array([50.0, 34.0]), line, RIGHT) == 0


# ── The pass penalty ─────────────────────────────────────────────────────────

def test_pass_that_adds_defenders_is_penalised():
    line = [opp(70.0 + i * 2.0, 30.0 + i * 4.0, i) for i in range(3)]
    backward = goalside_value(
        np.array([80.0, 34.0]), np.array([60.0, 34.0]), line, RIGHT
    )
    assert backward < 1.0


def test_penalty_is_never_a_bonus():
    """Rewarding forward play is penetration_value's job -- doing it here too
    would double-count the same idea."""
    line = [opp(70.0, 34.0, 1), opp(72.0, 30.0, 2)]
    forward = goalside_value(
        np.array([60.0, 34.0]), np.array([85.0, 34.0]), line, RIGHT
    )
    assert forward == pytest.approx(1.0)


def test_penalty_is_a_discount_not_a_veto():
    """A genuinely valuable ball can still be worth playing backwards."""
    many = [opp(70.0 + i, 34.0, i) for i in range(9)]
    worst = goalside_value(
        np.array([90.0, 34.0]), np.array([40.0, 34.0]), many, RIGHT
    )
    assert worst >= GOALSIDE_MIN > 0.0


def test_square_pass_is_neutral():
    line = [opp(80.0, 34.0, 1)]
    assert goalside_value(
        np.array([70.0, 20.0]), np.array([70.0, 48.0]), line, RIGHT
    ) == pytest.approx(1.0)


# ── The clear-run bonus ──────────────────────────────────────────────────────

def test_clear_lane_earns_the_full_bonus():
    assert clear_path_bonus(np.array([75.0, 34.0]), [], RIGHT) == pytest.approx(
        CLEAR_PATH_BONUS
    )


def test_bonus_fades_as_defenders_enter_the_lane():
    origin = np.array([72.0, 34.0])
    scores = []
    blockers = []
    for spot in [(85.0, 34.0), (90.0, 32.0), (80.0, 36.0)]:
        blockers.append(opp(*spot, pid=len(blockers)))
        scores.append(clear_path_bonus(origin, list(blockers), RIGHT))
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < scores[0]


def test_wide_defenders_do_not_block_the_lane():
    """A raw goal-side count treated a full-back stranded on the touchline as
    an obstacle; what matters is proximity to the running lane."""
    origin = np.array([72.0, 34.0])
    in_lane = [opp(85.0, 34.0, 1)]
    out_wide = [opp(85.0, 2.0, 1)]
    assert _effective_blockers(origin, out_wide, RIGHT) < _effective_blockers(
        origin, in_lane, RIGHT
    )
    assert clear_path_bonus(origin, out_wide, RIGHT) > clear_path_bonus(
        origin, in_lane, RIGHT
    )


def test_defenders_behind_the_carrier_do_not_block():
    origin = np.array([72.0, 34.0])
    behind = [opp(50.0, 34.0, 1)]
    assert _effective_blockers(origin, behind, RIGHT) == pytest.approx(0.0)


# ── The behaviour all of this exists for ─────────────────────────────────────

def test_clear_run_prefers_carrying_to_passing_backwards():
    """Players were sending the ball back and throwing away the chance."""
    state = build_test_state()
    carrier = striker(state)
    carrier.pos = np.array([72.0, 34.0])
    carrier.has_ball = True
    state.ball.pos = carrier.pos.copy()
    state.ball.owner_id = carrier.player_id

    # Nobody ahead; everyone behind, including a tempting square outlet.
    for i, o in enumerate(state.away_team.players):
        o.pos = (np.array([104.0, 34.0]) if o.position == "GK"
                 else np.array([50.0, 10.0 + i * 6.0]))
    mate = midfielder(state)
    mate.pos = np.array([60.0, 34.0])
    for t in state.home_team.players:
        if t is not carrier and t is not mate:
            t.pos = np.array([45.0, 15.0 + t.player_id * 3.0])

    scored = [score_candidate(c, carrier, state) for c in build_candidates(carrier, state)]
    best = max(scored, key=lambda c: c.utility)
    assert best.kind == "dribble", f"chose {best.kind} with a clear run at goal"

    best_dribble = max(c.utility for c in scored if c.kind == "dribble")
    best_pass = max((c.utility for c in scored if c.kind == "pass"), default=0.0)
    assert best_dribble > best_pass * 1.5
