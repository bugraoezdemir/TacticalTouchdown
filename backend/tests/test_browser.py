"""The browser build's route layer answers as the server does."""
import json
import pickle

import pytest
from fastapi.testclient import TestClient

from app import browser, browser_sim
from app.api.routes import tournament
from app.engine.headless import MatchResult
from app.engine.models import MatchStats
from app.main import app


def _shape(value):
    """A JSON value with its numbers and strings blanked: the same keys, nested."""
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(v) for v in value[:1]]
    return None


def test_every_route_it_serves_exists_on_the_server():
    # From the schema: newer FastAPI no longer lists included routers' routes in app.routes.
    served = {(m.upper(), path) for path, ops in app.openapi()["paths"].items() for m in ops}
    assert set(browser.ROUTES) <= served


def test_a_match_answers_in_the_same_shape():
    client = TestClient(app)
    theirs = client.post("/simulate/start").json()
    status, text = browser.handle("POST", "/simulate/start", {}, None)
    assert status == 200
    ours = json.loads(text)
    assert _shape(ours) == _shape(theirs)
    assert ours["home_team"]["tactics"] == theirs["home_team"]["tactics"]

    theirs = client.post("/simulate/tick?n=2").json()
    status, text = browser.handle("POST", "/simulate/tick", {"n": "2"}, None)
    assert status == 200
    assert _shape(json.loads(text)) == _shape(theirs)
    assert len(json.loads(text)["frames"]) == 2


def test_tactics_reach_the_match():
    browser.handle("POST", "/simulate/start", {}, None)
    body = json.dumps({"team_id": 1, "risk": 1.4})
    status, text = browser.handle("POST", "/simulate/tactics", {}, body)
    assert status == 200
    assert json.loads(text)["home_team"]["tactics"]["risk"] == 1.4


def test_errors_carry_a_detail():
    assert browser.handle("POST", "/nope", {}, None)[0] == 404
    status, text = browser.handle("POST", "/simulate/tactics", {}, '{"risk": "x"}')
    assert status == 422 and "detail" in json.loads(text)


@pytest.fixture(autouse=True)
def sim_workers(monkeypatch):
    monkeypatch.setattr(tournament, "start_matchday", browser._queue_matchday)
    yield
    # The job and the season are module state: leave none for the next test.
    tournament._sim_job = None
    browser.take_specs()


def _season():
    browser.take_specs()
    tournament._sim_job = None
    status, _ = browser.handle("POST", "/tournament/create", {},
                               json.dumps({"seed": 3, "user_team_index": 0}))
    assert status == 200
    return tournament._tournament


def _result(spec, home=2, away=1):
    return MatchResult(
        matchday=spec.matchday, home_id=spec.home_id, away_id=spec.away_id,
        home_goals=home, away_goals=away, ticks=30000, stats=MatchStats(),
    )


def _state():
    return json.loads(browser.handle("GET", "/tournament/state", {}, None)[1])


def test_the_ai_fixtures_start_behind_the_users_match():
    season = _season()
    assert browser.handle("POST", "/tournament/start-match", {}, None)[0] == 200
    specs = [pickle.loads(s) for s in browser.take_specs()]
    assert len(specs) == 8
    assert {(s.home_id, s.away_id) for s in specs} == {
        (f.home_id, f.away_id) for f in season.ai_fixtures(1)}
    assert browser.take_specs() == []
    assert tournament._sim_job.running


def test_results_and_scores_reach_the_season():
    season = _season()
    browser.handle("POST", "/tournament/start-match", {}, None)
    specs = [pickle.loads(s) for s in browser.take_specs()]

    browser.sim_live(specs[0].home_id, specs[0].away_id, 1, 0, 500)
    state = _state()
    assert state["sim_job"]["running"] and state["sim_job"]["done"] == 0

    for spec in specs:
        browser.sim_result(pickle.dumps(_result(spec)))
    state = _state()
    assert not state["sim_job"]["running"] and state["sim_job"]["done"] == 8
    assert all(f.played for f in season.ai_fixtures(1))


def _play_matchday(season):
    for fixture in season.fixtures_for(season.current_matchday):
        season.record(fixture, 1, 0, MatchStats())


def test_each_change_to_the_season_is_saved_once():
    _season()
    saved = browser.take_save()
    assert saved
    assert browser.take_save() is None
    browser.handle("POST", "/tournament/start-match", {}, None)
    assert browser.take_save()
    specs = browser.take_specs()
    browser.handle("GET", "/tournament/state", {}, None)
    assert browser.take_save() is None
    browser.sim_result(pickle.dumps(_result(pickle.loads(specs[0]))))
    assert browser.take_save()


def test_a_saved_season_comes_back_as_it_was():
    season = _season()
    _play_matchday(season)
    data = pickle.dumps(season)
    theirs = _state()
    browser.handle("POST", "/tournament/reset", {}, None)
    browser.restore(data)
    ours = _state()
    assert ours["table"] == theirs["table"]
    assert ours["current_matchday"] == theirs["current_matchday"] == 2


def test_clearing_a_season_clears_the_save():
    _season()
    browser.take_save()
    assert browser.handle("POST", "/tournament/reset", {}, None)[0] == 200
    assert browser.take_save() == b""
    assert browser.handle("GET", "/tournament/state", {}, None)[0] == 400


def test_ai_fixtures_left_behind_start_again_on_restore():
    season = _season()
    fixture = season.user_fixture(1)
    season.record(fixture, 1, 0, MatchStats())
    data = pickle.dumps(season)
    browser.handle("POST", "/tournament/reset", {}, None)
    browser.take_specs()
    browser.restore(data)
    assert len(browser.take_specs()) == 8


def test_ai_fixtures_wait_for_the_users_match_on_restore():
    season = _season()
    data = pickle.dumps(season)
    browser.handle("POST", "/tournament/reset", {}, None)
    browser.restore(data)
    assert browser.take_specs() == []


def test_a_failed_fixture_is_reported():
    _season()
    browser.handle("POST", "/tournament/start-match", {}, None)
    specs = browser.take_specs()
    for spec in specs[1:]:
        browser.sim_result(pickle.dumps(_result(pickle.loads(spec))))
    browser.sim_failed("worker crashed")
    state = _state()
    assert state["sim_job"]["error"] == "worker crashed" and not state["sim_job"]["running"]


def test_a_sim_worker_returns_what_the_season_records(monkeypatch):
    _season()
    browser.handle("POST", "/tournament/start-match", {}, None)
    data = browser.take_specs()[0]
    seen = []

    def quick(spec, live, reactive):
        live[(spec.home_id, spec.away_id)] = (1, 0, 900)
        return _result(spec)

    monkeypatch.setattr(browser_sim, "simulate", quick)
    result = pickle.loads(browser_sim.run(data, lambda *a: seen.append(a)))
    assert result.home_goals == 2 and seen[0][2:] == (1, 0, 900)


def test_a_season_saved_before_a_new_setting_still_plays():
    """A season saved by an older version lacks settings added since (0.0.8
    had no Wing attacks); restored, they take their defaults and it plays."""
    season = _season()
    old = pickle.loads(pickle.dumps(season))
    for team in old.teams:
        object.__delattr__(team.tactics, "wing_attacks")
    data = pickle.dumps(old)
    browser.handle("POST", "/tournament/reset", {}, None)
    browser.restore(data)
    assert all(team.tactics.wing_attacks == 1.0 for team in tournament._tournament.teams)
    assert browser.handle("POST", "/tournament/start-match", {}, None)[0] == 200
    assert browser.handle("POST", "/simulate/tick", {"n": "30"}, None)[0] == 200
