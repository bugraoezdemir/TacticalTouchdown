"""The API's routes as plain calls, for the game running in the browser.

No server there: the frontend's requests reach handle() in a Web Worker
(frontend/src/engineWorker.ts), which calls the same route functions FastAPI
does and answers with the same status and JSON.
"""
from __future__ import annotations

import json
import pickle

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from .api.routes import simulation, tournament
from .db.schemas import TacticsUpdate, TournamentCreate, TournamentTacticsUpdate


def _tick(query: dict[str, str], body: str | None):
    return simulation.simulation_tick(int(query.get("n", 1)))


def _start_match(query: dict[str, str], body: str | None):
    """The user's match, and the matchday's AI fixtures behind it: they run on
    the other workers during it, so full time leaves little to wait for."""
    tournament.start_match()
    if tournament._sim_job is None or not tournament._sim_job.running:
        tournament.begin_matchday(tournament._require())
    return tournament._to_dto(tournament._require())


# AI fixtures waiting to go to the sim workers; no threads or processes here.
_specs: list[bytes] = []


def _queue_matchday(season, pending, job) -> None:
    _specs.extend(pickle.dumps(tournament._spec_for(season, f)) for f in pending)


# The season, for the worker to keep in the browser (IndexedDB). A live match is
# not kept: closing the page during one replays that fixture from kick-off.
_changed = False


def take_save() -> bytes | None:
    """The season to store, once per change: empty bytes if there is none now,
    None if nothing changed."""
    global _changed
    if not _changed:
        return None
    _changed = False
    season = tournament._tournament
    return b"" if season is None else pickle.dumps(season)


def restore(data) -> None:
    """Pick a stored season up where it was. The matchday's AI fixtures run
    behind the user's match, so they start now only if that has been played."""
    season = pickle.loads(_bytes(data))
    tournament._tournament = season
    if not season.finished:
        fixture = season.user_fixture(season.current_matchday)
        if fixture is None or fixture.played:
            tournament.begin_matchday(season)


def _bytes(data) -> bytes:
    return data.to_bytes() if hasattr(data, "to_bytes") else data


def install() -> None:
    """Run matchdays through the sim workers. Called by the engine worker, so
    that importing this module changes nothing on a server."""
    tournament.start_matchday = _queue_matchday


def take_specs() -> list[bytes]:
    """The fixtures to run, once each. The worker asks after every request."""
    taken = list(_specs)
    _specs.clear()
    return taken


def sim_live(home: int, away: int, home_goals: int, away_goals: int, tick: int) -> None:
    job = tournament._sim_job
    if job is not None:
        job.live[(home, away)] = (home_goals, away_goals, tick)


def sim_result(data) -> None:
    """A finished AI fixture, from a sim worker."""
    global _changed
    job = tournament._sim_job
    tournament.record_result(tournament._require(), pickle.loads(_bytes(data)))
    _changed = True
    job.done += 1
    _finish_if_done(job)


def sim_failed(error: str) -> None:
    job = tournament._sim_job
    job.error = error
    job.done += 1
    _finish_if_done(job)


def _finish_if_done(job) -> None:
    if job.done >= job.total:
        job.running = False
        job.live = {}


ROUTES = {
    ("POST", "/simulate/start"): lambda q, b: simulation.start_simulation(),
    ("POST", "/simulate/tick"): _tick,
    ("POST", "/simulate/tactics"): lambda q, b: simulation.set_tactics(TacticsUpdate.model_validate_json(b)),
    ("POST", "/simulate/reset"): lambda q, b: simulation.reset_simulation(),
    ("POST", "/tournament/create"): lambda q, b: tournament.create(TournamentCreate.model_validate_json(b)),
    ("GET", "/tournament/state"): lambda q, b: tournament.state(),
    ("POST", "/tournament/tactics"): lambda q, b: tournament.set_tactics(
        TournamentTacticsUpdate.model_validate_json(b)),
    ("POST", "/tournament/start-match"): _start_match,
    ("POST", "/tournament/finish-match"): lambda q, b: tournament.finish_match(),
    ("POST", "/tournament/advance"): lambda q, b: tournament.advance(),
    ("POST", "/tournament/reset"): lambda q, b: tournament.reset(),
}


def handle(method: str, path: str, query: dict[str, str], body: str | None) -> tuple[int, str]:
    """(status, JSON text) for one request, as the server would answer it."""
    global _changed
    route = ROUTES.get((method, path))
    if route is None:
        return 404, _detail("Not Found")
    try:
        result = route(query, body)
    except HTTPException as e:
        return e.status_code, _detail(str(e.detail))
    except ValidationError as e:
        return 422, _detail(str(e))
    if method == "POST" and path.startswith("/tournament/"):
        _changed = True
    return 200, result.model_dump_json() if isinstance(result, BaseModel) else json.dumps(result)


def _detail(text: str) -> str:
    return json.dumps({"detail": text})
