"""
The pitch has edges.

Nothing in the decision model knew that, so passes were aimed at targets sitting
exactly on the touchline (called runs clamped spots to y=0.0 / y=68.0) and the
ball simply rolled out. At 5.5 stoppages per minute this was the single largest
killer of attacking moves.
"""
import numpy as np
import pytest

from app.engine.constants import (
    BALL_FRICTION_COEFF,
    PASS_SPEED_HIGH,
    PASS_SPEED_LOW,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TOUCHLINE_MARGIN,
)
from app.engine.decisions.candidates import Candidate, build_candidates
from app.engine.decisions.on_ball import _in_play_probability, score_candidate
from app.engine.decisions.playmaking import _inside_playable_area, find_signal_target
from app.engine.factory import build_test_state
from app.engine.game_loop import tick


def pass_to(target, passer_pos, speed=PASS_SPEED_LOW):
    from app.engine.models import Player

    passer = Player(
        player_id=1, team_id=1, name="P", position="CM", tactical="normal",
        pos=np.array(passer_pos, dtype=float), velocity=np.zeros(2), speed=0.3,
        strength=70.0, intelligence=70.0, passing=70.0, goalkeeping=40.0,
        assigned_pos=np.array(passer_pos, dtype=float), marking_target_id=None,
        has_ball=True, runway_signal=None,
    )
    cand = Candidate(
        kind="pass", target=np.array(target, dtype=float),
        speed=speed, friction=BALL_FRICTION_COEFF,
    )
    return _in_play_probability(cand, passer)


# ── The in-play term ─────────────────────────────────────────────────────────

def test_pass_into_the_middle_is_unpenalised():
    assert pass_to([60.0, 34.0], [40.0, 34.0]) == 1.0


def test_pass_at_the_touchline_is_heavily_penalised():
    assert pass_to([60.0, 68.0], [40.0, 60.0]) < 0.2


def test_pass_past_the_touchline_is_worthless():
    assert pass_to([60.0, 75.0], [40.0, 60.0]) < 0.1


def test_penalty_ramps_smoothly_toward_the_line():
    """A cliff would make the model flip between two near-identical passes."""
    scores = [pass_to([60.0, y], [40.0, 50.0]) for y in (60.0, 64.0, 66.0, 67.5, 68.0)]
    assert scores == sorted(scores, reverse=True)


def test_overrun_is_what_counts_not_the_aim_point():
    """The ball does not stop at its target -- a pass aimed just inside the line
    still runs out."""
    safely_inside = pass_to([60.0, 34.0], [40.0, 34.0])
    aimed_just_inside = pass_to([60.0, 66.5], [55.0, 62.0])
    assert aimed_just_inside < safely_inside


def test_goal_line_end_is_also_respected():
    assert pass_to([105.0, 34.0], [95.0, 34.0]) < 0.5


# ── Called-run spots ─────────────────────────────────────────────────────────

def test_playable_area_excludes_the_margin():
    assert not _inside_playable_area(np.array([50.0, 0.0]))
    assert not _inside_playable_area(np.array([50.0, PITCH_WIDTH]))
    assert not _inside_playable_area(np.array([0.0, 34.0]))
    assert _inside_playable_area(np.array([50.0, 34.0]))
    assert _inside_playable_area(
        np.array([50.0, PITCH_WIDTH - TOUCHLINE_MARGIN])
    )


def test_called_runs_never_target_the_touchline():
    """Spots used to be clamped onto the line rather than rejected."""
    state = build_test_state()
    for _ in range(60):
        for player in state.all_players():
            if not player.has_ball:
                continue
            result = find_signal_target(player, state)
            if result is not None:
                spot, _ = result
                assert _inside_playable_area(spot), f"called run at {spot}"
        state = tick(state)


def test_no_chosen_action_targets_off_pitch():
    state = build_test_state()
    state.rng = np.random.default_rng(20260903)
    for _ in range(400):
        carrier = state.get_ball_owner()
        if carrier is not None and carrier.has_ball:
            best = max(
                (score_candidate(c, carrier, state)
                 for c in build_candidates(carrier, state)),
                key=lambda c: c.utility,
            )
            if best.kind in ("pass", "dribble"):
                assert -0.01 <= best.target[0] <= PITCH_LENGTH + 0.01
                assert -0.01 <= best.target[1] <= PITCH_WIDTH + 0.01
        state = tick(state)


@pytest.mark.slow
def test_out_of_play_rate_is_realistic():
    """Was 5.5 stoppages per minute; real football is roughly 1-2."""
    state = build_test_state()
    state.rng = np.random.default_rng(20260903)
    stoppages = 0
    minutes = 6000 / 30 / 60
    for _ in range(6000):
        before = len(state.events)
        state = tick(state)
        stoppages += sum(
            1 for e in state.events[before:]
            if e.kind in ("throw_in", "corner", "goal_kick")
        )
    assert stoppages / minutes < 3.0
