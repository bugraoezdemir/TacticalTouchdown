"""Tournament: structure, persistence and the rules that protect the season."""
from __future__ import annotations

from dataclasses import fields

import pytest
from fastapi.testclient import TestClient

from app.engine.tournament import (
    CITY_NAMES,
    TEAM_COUNT,
    build_fixtures,
    create_tournament,
    random_tactics,
)
from app.engine.constants import TACTIC_BIAS_MAX, TACTIC_BIAS_MIN
from app.engine.models import TeamTactics
from app.main import app
import numpy as np


# -- Structure ----------------------------------------------------------------

def test_eighteen_teams_with_distinct_german_city_names():
    t = create_tournament(seed=1)
    assert len(t.teams) == TEAM_COUNT == 18
    names = [x.name for x in t.teams]
    assert names == list(CITY_NAMES)
    assert len(set(names)) == 18


def test_double_round_robin_every_pair_twice_once_at_each_ground():
    t = create_tournament(seed=1)
    assert len(t.fixtures) == 18 * 17      # 306
    assert t.matchdays == 34
    ordered = {(f.home_id, f.away_id) for f in t.fixtures}
    assert len(ordered) == 306             # each ordered pair exactly once
    for home, away in ordered:
        assert (away, home) in ordered     # and its reverse exists


def test_every_team_plays_once_per_matchday():
    t = create_tournament(seed=3)
    for day in range(1, t.matchdays + 1):
        ids = [i for f in t.fixtures_for(day) for i in (f.home_id, f.away_id)]
        assert sorted(ids) == list(range(1, 19))


def test_home_and_away_games_are_balanced():
    """An unbalanced schedule would hand somebody a free advantage."""
    t = create_tournament(seed=7)
    for team in t.teams:
        home = sum(1 for f in t.fixtures if f.home_id == team.team_id)
        away = sum(1 for f in t.fixtures if f.away_id == team.team_id)
        assert home == away == 17


def test_user_plays_every_matchday_and_eight_ai_games_run_alongside():
    t = create_tournament(seed=2, user_team_index=4)
    for day in range(1, t.matchdays + 1):
        assert t.user_fixture(day) is not None
        assert len(t.ai_fixtures(day)) == 8


# -- Tactics ------------------------------------------------------------------

def test_ai_teams_get_distinct_tactics_and_the_user_starts_neutral():
    t = create_tournament(seed=11, user_team_index=0)
    user = t.team(t.user_team_id)
    assert user.tactics.pass_ == 1.0
    assert user.tactics.shoot == 1.0
    others = [x for x in t.teams if not x.is_user]
    profiles = {
        (round(x.tactics.pass_, 4), round(x.tactics.pressing, 4)) for x in others
    }
    assert len(profiles) == len(others)          # genuinely different sides


def test_every_slider_differs_between_ai_teams():
    """A slider left off the random lists plays at neutral for every AI team,
    and nothing else would show it."""
    t = create_tournament(seed=3)
    rivals = [team.tactics for team in t.teams if not team.is_user]
    sliders = [f.name for f in fields(TeamTactics) if f.type in ("float", float)]
    assert "man_marking" in sliders
    for slider in sliders:
        values = {getattr(tactics, slider) for tactics in rivals}
        assert len(values) > 1, slider
        assert all(TACTIC_BIAS_MIN <= v <= TACTIC_BIAS_MAX for v in values), slider


def test_the_users_side_stays_neutral():
    t = create_tournament(seed=3, user_team_index=4)
    assert t.team(t.user_team_id).tactics == TeamTactics()


def test_random_tactics_stay_inside_the_clamp():
    rng = np.random.default_rng(0)
    for _ in range(200):
        tactics = random_tactics(rng)
        for field in ("shoot", "pass_", "dribble", "pressing", "breadth"):
            value = getattr(tactics, field)
            assert TACTIC_BIAS_MIN <= value <= TACTIC_BIAS_MAX


def test_ai_teams_get_varied_possession_and_the_user_stays_neutral():
    t = create_tournament(seed=11, user_team_index=0)
    assert t.team(t.user_team_id).tactics.possession == 1.0
    values = [x.tactics.possession for x in t.teams if not x.is_user]
    assert all(1.0 <= v <= TACTIC_BIAS_MAX for v in values)
    assert len({round(v, 4) for v in values}) == len(values)


def test_possession_draws_do_not_shift_anything_else_the_seed_produces():
    """Possession has its own generator: the same seed must still give the
    same tactics, formations and fixtures as before it existed."""
    t = create_tournament(seed=5, user_team_index=2)
    rng = np.random.default_rng(5)
    for team in t.teams:
        if team.is_user:
            continue
        expected = random_tactics(rng)
        got = team.tactics
        for field in ("shoot", "pass_", "dribble", "pressing", "roaming",
                      "offensiveness", "breadth", "compactness", "formation"):
            assert getattr(got, field) == getattr(expected, field)
    expected_fixtures = build_fixtures([x.team_id for x in t.teams], rng)
    assert [(f.home_id, f.away_id) for f in t.fixtures] == \
        [(f.home_id, f.away_id) for f in expected_fixtures]


def test_a_match_is_built_from_stored_tactics_without_aliasing_them():
    t = create_tournament(seed=4, user_team_index=0)
    fixture = t.user_fixture(1)
    assert fixture is not None
    home = t.team(fixture.home_id)
    home.tactics.pass_ = 1.75

    state = t.build_match(fixture)
    assert state.home_team.name == home.name
    assert state.home_team.tactics.pass_ == pytest.approx(1.75)

    # Mutating the match must not reach back into the season's record.
    state.home_team.tactics.pass_ = 0.5
    assert home.tactics.pass_ == pytest.approx(1.75)


def test_match_seeds_are_stable_so_a_result_cannot_be_rerolled():
    a = create_tournament(seed=9)
    b = create_tournament(seed=9)
    for fa, fb in zip(a.fixtures, b.fixtures):
        assert a.match_seed(fa) == b.match_seed(fb)
    seeds = {a.match_seed(f) for f in a.fixtures}
    assert len(seeds) == len(a.fixtures)     # and every fixture differs


# -- Table --------------------------------------------------------------------

def test_table_orders_by_points_then_goal_difference():
    t = create_tournament(seed=6, user_team_index=0)
    day = t.fixtures_for(1)
    t.record(day[0], 3, 0)      # home win, +3
    t.record(day[1], 1, 1)      # draw
    t.record(day[2], 0, 1)      # away win
    rows = t.table()
    assert rows[0].points == 3
    assert rows[0].goal_difference == 3
    assert rows[0].team_id == day[0].home_id
    assert sum(r.played for r in rows) == 6                # three games, two sides


def test_current_matchday_advances_only_when_the_day_is_complete():
    t = create_tournament(seed=8, user_team_index=0)
    assert t.current_matchday == 1
    fixtures = t.fixtures_for(1)
    for f in fixtures[:-1]:
        t.record(f, 1, 0)
    assert t.current_matchday == 1          # still one game outstanding
    t.record(fixtures[-1], 1, 0)
    assert t.current_matchday == 2


# -- API ----------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_advance_is_refused_until_the_user_has_played(client: TestClient):
    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    res = client.post("/tournament/advance")
    assert res.status_code == 400
    assert "your match" in res.json()["detail"].lower()


def test_finish_match_is_refused_before_full_time(client: TestClient):
    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    client.post("/tournament/start-match")
    res = client.post("/tournament/finish-match")
    assert res.status_code == 400
    assert "not finished" in res.json()["detail"].lower()


def test_stored_tactics_persist_and_reach_the_match(client: TestClient):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]

    client.post(
        "/tournament/tactics",
        json={"team_id": user_id, "pass_": 1.6, "formation": "4-3-3"},
    )
    state = client.get("/tournament/state").json()
    stored = next(t for t in state["teams"] if t["team_id"] == user_id)
    assert stored["tactics"]["pass_"] == pytest.approx(1.6)
    assert stored["tactics"]["formation"] == "4-3-3"

    client.post("/tournament/start-match")
    live = client.post("/simulate/tick").json()
    name = stored["name"]
    side = live["home_team"] if live["home_team"]["name"] == name else live["away_team"]
    assert side["tactics"]["pass_"] == pytest.approx(1.6)
    assert side["tactics"]["formation"] == "4-3-3"


def test_team_names_replace_home_and_away(client: TestClient):
    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    client.post("/tournament/start-match")
    live = client.post("/simulate/tick").json()
    assert live["home_team"]["name"] in CITY_NAMES
    assert live["away_team"]["name"] in CITY_NAMES
    assert live["home_team"]["name"] != live["away_team"]["name"]


def test_unknown_formation_is_rejected(client: TestClient):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    res = client.post(
        "/tournament/tactics",
        json={"team_id": created["user_team_id"], "formation": "9-0-1"},
    )
    assert res.status_code == 400


def test_pressing_play_does_not_replace_the_tournament_match(client: TestClient):
    """The UI calls /simulate/start when it goes from idle to running.

    That used to rebuild a default friendly, so clicking "Play my match" and
    then play showed Home FC v Away FC instead of the league fixture.
    """
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 2}
    ).json()
    user_name = next(t["name"] for t in created["teams"] if t["is_user"])

    client.post("/tournament/start-match")
    started = client.post("/simulate/start").json()
    assert user_name in (started["home_team"]["name"], started["away_team"]["name"])
    assert "Home FC" not in (
        started["home_team"]["name"],
        started["away_team"]["name"],
    )


def test_reset_replays_the_fixture_rather_than_dropping_it(client: TestClient):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 2}
    ).json()
    user_name = next(t["name"] for t in created["teams"] if t["is_user"])

    client.post("/tournament/start-match")
    for _ in range(3):
        client.post("/simulate/tick")

    fresh = client.post("/simulate/reset").json()
    assert fresh["tick"] == 0
    assert user_name in (fresh["home_team"]["name"], fresh["away_team"]["name"])


def test_a_friendly_is_still_built_when_no_tournament_exists(client: TestClient):
    """The Match tab has to keep working on its own."""
    from app.api.routes import simulation, tournament

    tournament._tournament = None
    tournament._live_fixture = None
    simulation._state = None

    started = client.post("/simulate/start").json()
    assert started["home_team"]["name"] == "Home FC"
    assert started["away_team"]["name"] == "Away FC"


# -- Results, stats and non-blocking Advance ---------------------------------

def _play_user_fixture(client: TestClient) -> None:
    """Run the user's match to full time and record it."""
    client.post("/tournament/start-match")
    while not client.post("/simulate/tick").json()["finished"]:
        pass
    client.post("/tournament/finish-match")


def test_a_recorded_fixture_keeps_its_match_stats(client: TestClient):
    """A headless match's GameState dies with its worker, so the fixture is the
    only durable record. Without this the season keeps scorelines and nothing
    else."""
    from app.engine.models import MatchStats
    from app.engine.tournament import create_tournament

    tournament = create_tournament(seed=1)
    fixture = tournament.fixtures[0]
    stats = MatchStats()
    stats.home.shots = 11
    stats.away.passes = 402

    tournament.record(fixture, 2, 1, stats)
    assert fixture.played
    assert fixture.stats is not None
    assert fixture.stats.home.shots == 11
    assert fixture.stats.away.passes == 402


def test_recording_without_stats_is_still_allowed(client: TestClient):
    """Table-only callers should not have to fabricate a MatchStats."""
    from app.engine.tournament import create_tournament

    tournament = create_tournament(seed=1)
    fixture = tournament.fixtures[0]
    tournament.record(fixture, 1, 1)
    assert fixture.played
    assert fixture.stats is None


@pytest.mark.slow
def test_the_users_own_match_stores_its_stats_too(client: TestClient):
    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    _play_user_fixture(client)

    state = client.get("/tournament/state").json()
    played = [f for f in state["fixtures"] if f["played"]]
    assert len(played) == 1
    assert played[0]["stats"] is not None
    # A full match must have produced some passes at both ends.
    assert played[0]["stats"]["home"]["passes"] > 0
    assert played[0]["stats"]["away"]["passes"] > 0


@pytest.mark.slow
def test_advance_returns_immediately_rather_than_blocking(client: TestClient):
    """The whole point of the change: a matchday takes ~2.7 minutes, and the UI
    must not be frozen for it."""
    import time

    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    _play_user_fixture(client)

    started = time.time()
    res = client.post("/tournament/advance")
    elapsed = time.time() - started

    assert res.status_code == 200
    assert elapsed < 5.0, f"advance blocked for {elapsed:.1f}s"
    job = res.json()["sim_job"]
    assert job is not None
    assert job["running"] is True
    assert job["total"] == 8          # the eight AI fixtures
    assert job["done"] == 0


@pytest.mark.slow
def test_a_second_advance_while_one_runs_is_refused(client: TestClient):
    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    _play_user_fixture(client)
    client.post("/tournament/advance")

    res = client.post("/tournament/advance")
    assert res.status_code == 409


@pytest.mark.slow
def test_every_fixture_of_a_simulated_matchday_ends_with_stats(client: TestClient):
    """Follows the job to completion and checks what it left behind."""
    import time

    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    _play_user_fixture(client)
    client.post("/tournament/advance")

    deadline = time.time() + 900
    while time.time() < deadline:
        state = client.get("/tournament/state").json()
        if not state["sim_job"]["running"]:
            break
        time.sleep(2)
    else:
        pytest.fail("matchday simulation did not finish in time")

    assert state["sim_job"]["error"] is None
    assert state["sim_job"]["done"] == 8
    day_one = [f for f in state["fixtures"] if f["matchday"] == 1]
    assert all(f["played"] for f in day_one)
    assert all(f["stats"] is not None for f in day_one)
    # And the season moved on, which is what makes past results history.
    assert state["current_matchday"] == 2


@pytest.mark.slow
def test_results_of_past_matchdays_remain_available(client: TestClient):
    """The user's complaint was that other games' results were invisible. They
    were stored all along -- this pins that they stay readable once the season
    has moved past them."""
    import time

    client.post("/tournament/create", json={"seed": 5, "user_team_index": 0})
    _play_user_fixture(client)
    client.post("/tournament/advance")
    deadline = time.time() + 900
    while time.time() < deadline:
        state = client.get("/tournament/state").json()
        if not state["sim_job"]["running"]:
            break
        time.sleep(2)

    assert state["current_matchday"] == 2
    finished = [f for f in state["fixtures"] if f["matchday"] == 1]
    assert len(finished) == 9
    assert all(f["played"] for f in finished)
    # Every one carries a scoreline and stats, not just the user's.
    for fixture in finished:
        assert fixture["stats"]["home"]["passes"] > 0


# -- Per-player overrides persist too -----------------------------------------

def test_player_overrides_are_stored_on_the_team_preset(client: TestClient):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]

    res = client.post("/tournament/tactics", json={
        "team_id": user_id,
        "players": [{"slot": 7, "bias_shoot": 1.9, "bias_dribble": 0.6}],
    })
    assert res.status_code == 200
    team = next(t for t in res.json()["teams"] if t["team_id"] == user_id)
    stored = team["player_biases"]
    assert len(stored) == 1
    assert stored[0]["slot"] == 7
    assert stored[0]["bias_shoot"] == pytest.approx(1.9)
    assert stored[0]["bias_dribble"] == pytest.approx(0.6)


def test_player_overrides_reach_the_squad_of_a_built_match(client: TestClient):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]
    user_name = next(t["name"] for t in created["teams"] if t["is_user"])

    client.post("/tournament/tactics", json={
        "team_id": user_id,
        "players": [{"slot": 7, "bias_shoot": 1.9}],
    })
    client.post("/tournament/start-match")
    live = client.post("/simulate/tick").json()

    side = (
        live["home_team"]
        if live["home_team"]["name"] == user_name
        else live["away_team"]
    )
    assert side["players"][7]["bias_shoot"] == pytest.approx(1.9)
    # Nobody else was touched.
    assert side["players"][6]["bias_shoot"] is None


def test_overrides_survive_a_change_of_venue(client: TestClient):
    """Keyed on SQUAD SLOT, not player_id.

    The same footballer is id 8 at home and id 19 away, because ids are rebuilt
    per fixture. Keying storage on player_id would silently drop the override on
    every alternate matchday.
    """
    from app.api.routes import tournament as tr

    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]
    user_name = next(t["name"] for t in created["teams"] if t["is_user"])

    client.post("/tournament/tactics", json={
        "team_id": user_id,
        "players": [{"slot": 7, "bias_shoot": 1.9}],
    })

    def played_at_home() -> bool:
        client.post("/tournament/start-match")
        live = client.post("/simulate/tick").json()
        home = live["home_team"]["name"] == user_name
        side = live["home_team"] if home else live["away_team"]
        assert side["players"][7]["bias_shoot"] == pytest.approx(1.9)
        return home

    first = played_at_home()
    for fixture in tr._tournament.fixtures_for(1):
        tr._tournament.record(fixture, 0, 0)
    tr._live_fixture = None
    second = played_at_home()

    assert first != second, "fixture list should flip venue between legs"


def test_an_override_can_be_cleared(client: TestClient):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    user_id = created["user_team_id"]

    client.post("/tournament/tactics", json={
        "team_id": user_id, "players": [{"slot": 3, "bias_pass": 1.5}],
    })
    res = client.post("/tournament/tactics", json={
        "team_id": user_id, "players": [{"slot": 3}],
    })
    team = next(t for t in res.json()["teams"] if t["team_id"] == user_id)
    assert team["player_biases"] == []


@pytest.mark.parametrize("slot", [-1, 11, 99])
def test_a_slot_outside_the_squad_is_rejected(client: TestClient, slot: int):
    created = client.post(
        "/tournament/create", json={"seed": 5, "user_team_index": 0}
    ).json()
    res = client.post("/tournament/tactics", json={
        "team_id": created["user_team_id"],
        "players": [{"slot": slot, "bias_shoot": 1.5}],
    })
    assert res.status_code == 400


def test_capture_reads_overrides_back_off_a_squad():
    """finish-match writes the played squad's overrides back to the preset."""
    from app.engine.tournament import capture_player_biases, create_tournament

    tournament = create_tournament(seed=1, user_team_index=0)
    fixture = tournament.user_fixture(1)
    state = tournament.build_match(fixture)
    user = tournament.team(tournament.user_team_id)
    side = (
        state.home_team
        if fixture.home_id == user.team_id
        else state.away_team
    )

    side.players[2].bias_roaming = 1.6
    side.players[9].bias_one_two = 0.7
    captured = capture_player_biases(side)

    assert captured == {2: {"bias_roaming": 1.6}, 9: {"bias_one_two": 0.7}}
