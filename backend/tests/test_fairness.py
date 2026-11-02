"""
Neither side may be structurally favoured.

With identical player attributes, home and away must have the same chance of
winning. Four separate mechanisms quietly favoured home:

  1. home took EVERY kick-off
  2. possession ties resolved by dict insertion order (home inserted first)
  3. an unknown last-toucher awarded the restart to home
  4. decision-order ties broke on player_id, and home ids are 1-10 vs away 12-21

Together these were worth roughly a 6-to-1 goal advantage across seeds.
"""
import collections

import numpy as np
import pytest

from app.engine.factory import build_test_state
from app.engine.game_loop import _decision_order, tick
from app.engine.physics import _other_team_id, resolve_possession


# ── Kick-off ─────────────────────────────────────────────────────────────────

def test_kickoff_alternates_across_seeds():
    """Home took every kick-off, and first possession often runs straight into
    a chance."""
    takers = collections.Counter(
        build_test_state(seed=s).get_ball_owner().team_id for s in range(40)
    )
    assert takers[1] > 5 and takers[2] > 5, f"kick-offs are one-sided: {dict(takers)}"


def test_kickoff_team_can_be_forced():
    assert build_test_state(seed=1, kickoff_team=1).get_ball_owner().team_id == 1
    assert build_test_state(seed=1, kickoff_team=2).get_ball_owner().team_id == 2


def test_same_seed_reproduces_the_same_kickoff():
    assert (
        build_test_state(seed=7).get_ball_owner().player_id
        == build_test_state(seed=7).get_ball_owner().player_id
    )


# ── Tie-breaks ───────────────────────────────────────────────────────────────

def test_possession_ties_do_not_always_favour_home():
    """Two equally strong players on the ball: over many draws each side must
    win some. max() over a dict returned the first key, which was always home."""
    winners = collections.Counter()
    for seed in range(60):
        state = build_test_state(seed=seed)
        home = state.home_team.players[5]
        away = state.away_team.players[5]
        for p in state.all_players():
            p.has_ball = False
            p.pos = np.array([5.0, 5.0])
        state.ball.owner_id = None
        state.ball.pos = np.array([52.5, 34.0])
        state.ball.velocity = np.zeros(2)
        home.pos = state.ball.pos.copy()
        away.pos = state.ball.pos.copy()
        home.strength = away.strength = 70.0
        resolve_possession(state)
        owner = state.get_ball_owner()
        if owner is not None:
            winners[owner.team_id] += 1
    assert winners[1] > 5 and winners[2] > 5, f"ties are one-sided: {dict(winners)}"


def test_unknown_last_toucher_does_not_always_favour_home():
    sides = collections.Counter(
        _other_team_id(build_test_state(seed=s), None) for s in range(60)
    )
    assert sides[1] > 5 and sides[2] > 5, f"restarts are one-sided: {dict(sides)}"


def test_decision_order_ties_do_not_favour_lower_ids():
    """Home ids are 1-10, away 12-21, so an id tie-break let a home player
    commit first in every 50/50 race."""
    firsts = collections.Counter()
    for seed in range(40):
        state = build_test_state(seed=seed)
        # Mirror the two teams onto identical distances from the ball.
        state.ball.pos = np.array([52.5, 34.0])
        for i, p in enumerate(state.home_team.players):
            p.pos = np.array([52.5, 34.0 + 10.0])
        for i, p in enumerate(state.away_team.players):
            p.pos = np.array([52.5, 34.0 - 10.0])
        firsts[_decision_order(state)[0].team_id] += 1
    assert firsts[1] > 3 and firsts[2] > 3, f"ordering is one-sided: {dict(firsts)}"


def test_decision_order_is_still_deterministic_per_seed():
    a = [p.player_id for p in _decision_order(build_test_state(seed=3))]
    b = [p.player_id for p in _decision_order(build_test_state(seed=3))]
    assert a == b


# ── The end-to-end property ──────────────────────────────────────────────────

@pytest.mark.slow
def test_no_side_wins_systematically():
    """With identical squads, results must be even across a run of matches."""
    home_goals = away_goals = 0
    home_poss = away_poss = 0
    for seed in range(12):
        state = build_test_state(seed=seed)
        for _ in range(2400):
            owner = state.get_ball_owner()
            if owner is not None:
                if owner.team_id == 1:
                    home_poss += 1
                else:
                    away_poss += 1
            state = tick(state)
        home_goals += state.home_score
        away_goals += state.away_score

    total_poss = home_poss + away_poss
    assert total_poss > 0
    share = home_poss / total_poss
    assert 0.40 < share < 0.60, f"possession {share:.0%} favours one side"

    total_goals = home_goals + away_goals
    if total_goals >= 6:
        # Only meaningful once enough goals exist to be a signal.
        assert min(home_goals, away_goals) / total_goals > 0.2, (
            f"goals {home_goals}-{away_goals} are one-sided"
        )
