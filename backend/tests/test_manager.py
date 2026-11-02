"""Reactive in-match tactics: the pure logic.

Deliberately no match simulation here -- every one of these is a property of
the reward, the policy or the preset arithmetic, and they run in milliseconds.
The behavioural question (does reacting actually play better?) needs a season,
not a unit test, and the logged windows are what will answer it.
"""
from __future__ import annotations

import pytest

from app.engine.constants import DISPLAY_HALF_MINUTES, HALF_LENGTH_TICKS
from app.engine.manager import (
    WINDOW_TICKS,
    MatchContext,
    StatSnapshot,
    window_reward,
)
from app.engine.models import TeamTactics


# -- The reward ---------------------------------------------------------------

def test_a_quiet_window_scores_zero():
    """No shots, no goals, even possession, neutral territory -> nothing."""
    even = StatSnapshot(possession_ticks=800)
    assert window_reward(even, even, 0.0, WINDOW_TICKS) == pytest.approx(0.0)


def test_the_reward_is_antisymmetric():
    """Swapping the sides must flip the sign, or one team is silently favoured."""
    ours = StatSnapshot(goals=1, shots=3, shots_on_target=2, possession_ticks=1000)
    theirs = StatSnapshot(goals=0, shots=1, shots_on_target=0, possession_ticks=600)
    forward = window_reward(ours, theirs, 0.5, WINDOW_TICKS)
    reverse = window_reward(theirs, ours, -0.5, WINDOW_TICKS)
    assert forward == pytest.approx(-reverse)


def test_goals_outweigh_the_dense_terms():
    """The dense signals exist to make the reward steerable BETWEEN goals, not
    to replace the objective. A reward that pays better for shots than for
    goals finds tactics that shoot from 30 m and lose."""
    scored = StatSnapshot(goals=1)
    shot_happy = StatSnapshot(shots=6, shots_on_target=3)
    nothing = StatSnapshot()
    assert window_reward(scored, nothing, 0.0, WINDOW_TICKS) > window_reward(
        shot_happy, nothing, 0.0, WINDOW_TICKS
    )


def test_conceding_hurts_as_much_as_scoring_helps():
    scored = window_reward(StatSnapshot(goals=1), StatSnapshot(), 0.0, WINDOW_TICKS)
    conceded = window_reward(StatSnapshot(), StatSnapshot(goals=1), 0.0, WINDOW_TICKS)
    assert scored == pytest.approx(-conceded)


def test_territory_and_possession_are_shares_not_counts():
    """A short final window must not score differently just for being short."""
    ours = StatSnapshot(possession_ticks=500)
    theirs = StatSnapshot(possession_ticks=250)
    full = window_reward(ours, theirs, 0.3, 1000)
    half = window_reward(
        StatSnapshot(possession_ticks=250), StatSnapshot(possession_ticks=125),
        0.3, 500,
    )
    assert full == pytest.approx(half)


# -- Window arithmetic --------------------------------------------------------

def test_a_window_is_fifteen_displayed_minutes():
    per_minute = HALF_LENGTH_TICKS / DISPLAY_HALF_MINUTES
    assert WINDOW_TICKS == pytest.approx(per_minute * 15.0, abs=1.0)


def test_a_match_yields_roughly_six_windows():
    total = 2 * HALF_LENGTH_TICKS
    assert 5 <= total / WINDOW_TICKS <= 7


# -- Who is allowed to react --------------------------------------------------
#
# The user's rule: "I only want the opponent team to auto-adjust. I have control
# on my own team." True in BOTH modes. But when rivals play each other with
# nobody watching, both of them adapt.

def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    import app.api.routes.simulation as sim
    import app.api.routes.tournament as tr

    tr._tournament = None
    tr._live_fixture = None
    sim._state = None
    sim._manager = None
    return TestClient(app), sim, tr


def test_in_a_friendly_only_the_opponent_reacts():
    """The Match tab's user owns the HOME side.

    Attaching with no user id left both sides reacting, so the user's own
    sliders were rewritten mid-match -- the exact thing reactive tactics are
    not for.
    """
    client, sim, _ = _client()
    client.post("/simulate/start")

    assert sim._manager is not None
    assert sim._manager.home.enabled is False, "the user's own side reacted"
    assert sim._manager.away.enabled is True, "the opponent did not react"


def test_reset_keeps_the_user_exempt():
    client, sim, _ = _client()
    client.post("/simulate/start")
    client.post("/simulate/reset")

    assert sim._manager.home.enabled is False
    assert sim._manager.away.enabled is True


def test_in_a_league_fixture_the_users_side_never_reacts():
    """Whichever venue. Ids are rebuilt per fixture, so this must be resolved
    from the tournament's user_team_id, not from home/away."""
    client, sim, tr = _client()
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]
    client.post("/tournament/start-match")
    state = sim._state

    user_is_home = state.home_team.team_id == user_id
    ours = sim._manager.home if user_is_home else sim._manager.away
    theirs = sim._manager.away if user_is_home else sim._manager.home

    assert ours.enabled is False
    assert theirs.enabled is True


def test_when_rivals_play_each_other_both_adapt():
    """No human is watching, so there is nobody to exempt."""
    from app.engine.factory import build_test_state
    from app.engine.manager import MatchManager

    state = build_test_state(seed=1)
    manager = MatchManager(state, home_enabled=True, away_enabled=True)

    assert manager.home.enabled is True
    assert manager.away.enabled is True


def test_a_disabled_side_keeps_its_tactics_across_a_window():
    """The guard has to survive an actual window boundary, not just be set."""
    from app.engine.factory import build_test_state
    from app.engine.manager import MatchManager, WINDOW_TICKS

    state = build_test_state(seed=1)
    manager = MatchManager(state, home_enabled=False, away_enabled=True)
    state.home_team.tactics.attack_call = 1.4
    before = state.home_team.tactics.attack_call

    # Home trailing late: the bucket that would otherwise force Chasing.
    state.away_score = 1
    state.tick = 2 * HALF_LENGTH_TICKS - 1000
    for side in (manager.home, manager.away):
        side._window_start_tick = state.tick - WINDOW_TICKS - 1
    manager.observe(state)
    manager.maybe_adjust(state)

    assert state.home_team.tactics.attack_call == pytest.approx(before)


# -- Two id spaces: season ids (1-8) vs live match ids (home=1, away=2) -------
#
# `Tournament.user_team_id` means nothing to a GameState, whose sides are
# rebuilt per fixture. Passing it straight to attach_manager exempted whichever
# LIVE side happened to share that number -- so whenever the user played AWAY,
# their own team reacted and the opponent did not.
#
# The earlier tests missed this by only ever exercising matchday 1, where the
# user is home and the two id spaces coincidentally agree. These sweep the
# season so a venue-dependent bug cannot hide in an untested matchday.

def _exemption(sim, user_name: str) -> tuple[bool, bool]:
    """(does the user's side react, does the rival react) in the live match."""
    state = sim._state
    user_is_home = state.home_team.name == user_name
    ours = sim._manager.home if user_is_home else sim._manager.away
    theirs = sim._manager.away if user_is_home else sim._manager.home
    return ours.enabled, theirs.enabled


def test_the_user_is_exempt_at_every_venue_across_the_season():
    client, sim, tr = _client()
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]
    user_name = next(t["name"] for t in created["teams"] if t["is_user"])
    season = tr._tournament

    venues = set()
    for matchday in range(1, season.matchdays + 1):
        fixture = season.user_fixture(matchday)
        assert fixture is not None
        venues.add(fixture.home_id == user_id)

        client.post("/tournament/start-match")
        ours, theirs = _exemption(sim, user_name)
        assert ours is False, f"user reacted on matchday {matchday}"
        assert theirs is True, f"rival did not react on matchday {matchday}"

        # Finish the matchday so the season advances to the next fixture.
        for other in season.fixtures_for(matchday):
            season.record(other, 0, 0)
        tr._live_fixture = None

    assert venues == {True, False}, "the season must cover home AND away"


@pytest.mark.parametrize("endpoint", ["/simulate/start", "/simulate/reset"])
def test_pressing_play_or_reset_keeps_the_right_side_exempt_away(endpoint):
    """Both re-attach a manager, so both can pick the wrong side."""
    client, sim, tr = _client()
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]
    user_name = next(t["name"] for t in created["teams"] if t["is_user"])
    season = tr._tournament

    # Advance to a matchday where the user plays AWAY.
    for matchday in range(1, season.matchdays + 1):
        if season.user_fixture(matchday).away_id == user_id:
            break
        for other in season.fixtures_for(matchday):
            season.record(other, 0, 0)

    client.post("/tournament/start-match")
    assert sim._state.away_team.name == user_name, "expected the user away"

    client.post(endpoint)
    ours, theirs = _exemption(sim, user_name)
    assert ours is False
    assert theirs is True
