"""
Goal kicks and the keeper's hands.

The reported bug: at a goal kick the attacking side stayed exactly where the
move had left them -- standing on the keeper -- and simply took the ball off
him. Only the DEFENDING team was being restationed.
"""
import numpy as np
import pytest

from app.engine.constants import (
    DEAD_BALL_TICKS,
    GK_GATHER_SPEED,
    GK_HOLD_TICKS,
    GK_SAFE_PASS_FLOOR,
    PENALTY_AREA_DEPTH,
)
from app.engine.factory import build_test_state
from app.engine.game_loop import tick
from app.engine.physics import in_penalty_area, resolve_possession


def force_goal_kick(state):
    """Away attacks leftward, so away putting it out at x<0 is a goal kick."""
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = state.away_team.team_id
    state.ball.pos = np.array([-1.0, 20.0])
    resolve_possession(state)
    for _ in range(DEAD_BALL_TICKS):   # seen to go out, then laid out
        state = tick(state)


# -- Goal kicks ---------------------------------------------------------------

def test_a_goal_kick_clears_attackers_out_of_the_box():
    state = build_test_state(seed=0)
    # Camp four attackers on the keeper, as a real attack leaves them.
    for i, p in enumerate(state.away_team.players[:4]):
        p.pos = np.array([4.0 + i * 0.5, 34.0])

    force_goal_kick(state)

    inside = [p for p in state.away_team.players if in_penalty_area(p.pos, 0.0)]
    assert inside == [], f"{len(inside)} attackers still in the box"


def test_a_goal_kick_leaves_the_keeper_room():
    """0.5 m from the keeper was the measured worst case, which is how they
    took it off him and scored."""
    state = build_test_state(seed=0)
    for i, p in enumerate(state.away_team.players[:4]):
        p.pos = np.array([4.0 + i * 0.5, 34.0])

    force_goal_kick(state)

    gk = next(p for p in state.home_team.players if p.position == "GK")
    nearest = min(
        float(np.hypot(*(p.pos - gk.pos))) for p in state.away_team.players
    )
    assert nearest > PENALTY_AREA_DEPTH / 2.0


def test_a_goal_kick_does_not_move_the_kicking_team_out():
    """Only the opponents retreat; the defending side has just been restationed
    and several of them legitimately stand inside their own area."""
    state = build_test_state(seed=0)
    force_goal_kick(state)
    assert any(in_penalty_area(p.pos, 0.0) for p in state.home_team.players)


# -- The keeper's hands -------------------------------------------------------

def test_the_keeper_gathers_a_slow_ball_in_his_area():
    state = build_test_state(seed=0)
    gk = next(p for p in state.home_team.players if p.position == "GK")
    state.ball.owner_id = None
    for p in state.all_players():
        p.has_ball = False
    state.ball.pos = gk.pos.copy()
    state.ball.velocity = np.array([GK_GATHER_SPEED * 0.5, 0.0])

    resolve_possession(state)

    assert state.ball.owner_id == gk.player_id
    assert state.gk_hold_ticks == GK_HOLD_TICKS


def test_the_keeper_cannot_gather_a_ball_struck_at_pace():
    """He can only parry a shot; catching everything would make him unbeatable."""
    state = build_test_state(seed=0)
    gk = next(p for p in state.home_team.players if p.position == "GK")
    state.ball.owner_id = None
    for p in state.all_players():
        p.has_ball = False
    state.ball.pos = gk.pos.copy()
    state.ball.velocity = np.array([GK_GATHER_SPEED * 3.0, 0.0])

    resolve_possession(state)

    assert state.gk_hold_ticks == 0


def test_gathering_clears_the_box_and_the_hold_counts_down():
    state = build_test_state(seed=0)
    gk = next(p for p in state.home_team.players if p.position == "GK")
    for i, p in enumerate(state.away_team.players[:3]):
        p.pos = np.array([5.0 + i * 0.5, 34.0])
    state.ball.owner_id = None
    for p in state.all_players():
        p.has_ball = False
    state.ball.pos = gk.pos.copy()
    state.ball.velocity = np.array([GK_GATHER_SPEED * 0.5, 0.0])

    resolve_possession(state)
    assert not any(in_penalty_area(p.pos, 0.0) for p in state.away_team.players)

    before = state.gk_hold_ticks
    state = tick(state)
    assert state.gk_hold_ticks < before


def test_the_keeper_holds_rather_than_deciding_while_the_hold_runs():
    from app.engine.decisions.on_ball import on_ball_decision

    state = build_test_state(seed=0)
    gk = next(p for p in state.home_team.players if p.position == "GK")
    for p in state.all_players():
        p.has_ball = False
    gk.has_ball = True
    state.ball.owner_id = gk.player_id
    state.ball.pos = gk.pos.copy()
    state.gk_hold_ticks = GK_HOLD_TICKS

    assert on_ball_decision(gk, state).action_type == "hold"


def test_the_keeper_never_distributes_into_a_risky_pass():
    """A giveaway from his own hands is a shot on his own goal, so he is held
    to a floor no outfield player is."""
    from app.engine.decisions.candidates import build_candidates
    from app.engine.decisions.on_ball import on_ball_decision, score_candidate

    state = build_test_state(seed=0)
    gk = next(p for p in state.home_team.players if p.position == "GK")
    for p in state.all_players():
        p.has_ball = False
    gk.has_ball = True
    state.ball.owner_id = gk.player_id
    state.ball.pos = gk.pos.copy()
    state.gk_hold_ticks = 0

    action = on_ball_decision(gk, state)
    if action.action_type != "pass":
        return  # nothing safe enough was on; holding is the right answer

    chosen = [
        score_candidate(c, gk, state)
        for c in build_candidates(gk, state)
        if c.kind == "pass"
    ]
    assert max(c.p for c in chosen) >= GK_SAFE_PASS_FLOOR


def test_the_box_stays_clear_for_the_whole_hold():
    """Clearing the box on the catch was not enough on its own: the off-ball
    model pulled opponents straight back toward the ball, and 3 of 4 were
    inside again within 5 ticks. Players just outside the line were targeting
    the keeper himself, so they have to be kept out for as long as he holds it.
    """
    state = build_test_state(seed=0)
    gk = next(p for p in state.home_team.players if p.position == "GK")
    for i, p in enumerate(state.away_team.players[:4]):
        p.pos = np.array([5.0 + i * 0.5, 34.0])
    for p in state.all_players():
        p.has_ball = False
    state.ball.owner_id = None
    state.ball.pos = gk.pos.copy()
    state.ball.velocity = np.array([GK_GATHER_SPEED * 0.5, 0.0])

    resolve_possession(state)
    assert state.gk_hold_ticks > 0

    while state.gk_hold_ticks > 0:
        state = tick(state)
        intruders = [
            p for p in state.away_team.players if in_penalty_area(p.pos, 0.0)
        ]
        assert intruders == [], (
            f"{len(intruders)} opponents back in the box with "
            f"{state.gk_hold_ticks} hold ticks left"
        )
