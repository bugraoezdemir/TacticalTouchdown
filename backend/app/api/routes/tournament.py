"""
Tournament routes.

The season owns the tactics; a match borrows them. Two rules follow, and both
are enforced here rather than in the engine:

  * Starting the user's match builds a GameState FROM the stored tactics.
  * Finishing it writes whatever the user changed back to the store, so a
    season is a tuning arc rather than 14 unrelated games.

The user's live match is held by the simulation router, which already owns a
single `_state`. This router drives that same slot so the existing /simulate
endpoints -- tick, tactics, stats -- keep working unchanged for a tournament
match.
"""
from __future__ import annotations

import threading
from dataclasses import replace

from fastapi import APIRouter, HTTPException

from ...db.schemas import (
    FixtureDTO,
    PlayerStatsDTO,
    TournamentPlayerBiasDTO,
    MatchStatsDTO,
    SimJobDTO,
    TeamStatsDTO,
    TableRowDTO,
    TournamentCreate,
    TournamentDTO,
    TournamentTacticsUpdate,
    TournamentTeamDTO,
)
from ...engine.formations import FORMATIONS, SQUAD_SIZE, apply_formation
from ...engine.headless import MatchResult, MatchSpec, iter_simulated, live_score_map
from ...engine.tournament import (
    TEAM_COUNT,
    PLAYER_BIAS_FIELDS,
    Fixture,
    Tournament,
    capture_player_biases,
    create_tournament,
)
from . import simulation

router = APIRouter()

_tournament: Tournament | None = None
# The fixture whose match is currently loaded into the simulation router.
_live_fixture: Fixture | None = None


class _SimJob:
    """A matchday's AI fixtures, simulated off the request thread.

    Advance used to run `simulate_many` inline, so the HTTP request -- and with
    it the whole UI -- blocked for the ~2.7 minutes a matchday takes. Here the
    work happens on a background thread that owns a process pool, and the
    request returns immediately; the client polls /state to watch scores land.

    The lock guards `done`/`error` only. Results are written into the
    Tournament from the worker thread, which is safe because FastAPI's sync
    endpoints run in a threadpool and every mutation here is a whole-object
    assignment on a distinct fixture -- no two matches touch the same one.
    """

    def __init__(self, matchday: int, total: int) -> None:
        self.matchday = matchday
        self.total = total
        self.done = 0
        self.error: str | None = None
        self.running = True
        self.lock = threading.Lock()
        # (home_id, away_id) -> (home_goals, away_goals, tick), written by the
        # worker processes as goals go in and read by /state.
        self.live: dict = {}

    def dto(self) -> SimJobDTO:
        with self.lock:
            return SimJobDTO(
                running=self.running,
                matchday=self.matchday,
                done=self.done,
                total=self.total,
                error=self.error,
            )


_sim_job: _SimJob | None = None


def _require() -> Tournament:
    if _tournament is None:
        raise HTTPException(status_code=400, detail="No tournament; call /create")
    return _tournament


@router.post("/create", response_model=TournamentDTO)
def create(body: TournamentCreate) -> TournamentDTO:
    global _tournament, _live_fixture
    if not 0 <= body.user_team_index < TEAM_COUNT:
        raise HTTPException(
            status_code=400, detail=f"user_team_index must be 0-{TEAM_COUNT - 1}"
        )
    _tournament = create_tournament(
        seed=body.seed, user_team_index=body.user_team_index
    )
    _live_fixture = None
    return _to_dto(_tournament)


@router.post("/reset")
def reset() -> dict[str, bool]:
    """Forget the season. A matchday still running finishes unseen."""
    global _tournament, _live_fixture, _sim_job
    _tournament = _live_fixture = _sim_job = None
    return {"ok": True}


@router.get("/state", response_model=TournamentDTO)
def state() -> TournamentDTO:
    return _to_dto(_require())


@router.post("/tactics", response_model=TournamentDTO)
def set_tactics(update: TournamentTacticsUpdate) -> TournamentDTO:
    """Change a team's STORED tactics. Persists for the rest of the season."""
    tournament = _require()
    try:
        team = tournament.team(update.team_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"No team {update.team_id}"
        ) from None

    tactics = team.tactics
    for field in (
        "shoot", "pass_", "dribble", "pressing", "roaming", "offensiveness",
        "direct", "one_two", "attack_call", "pass_range", "breadth", "compactness",
        "possession", "crossing", "lofted", "risk", "tempo", "press_aggression",
        "man_marking", "wing_attacks",
    ):
        value = getattr(update, field)
        if value is not None:
            setattr(tactics, field, value)
    if update.formation is not None:
        if update.formation not in FORMATIONS:
            raise HTTPException(
                status_code=400, detail=f"Unknown formation {update.formation!r}"
            )
        tactics.formation = update.formation

    # Per-player overrides persist alongside the team sliders. A slot sent with
    # every value None clears it, which is why the entry is dropped rather than
    # stored empty.
    for entry in update.players:
        if not 0 <= entry.slot < SQUAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Slot {entry.slot} is outside the squad (0-{SQUAD_SIZE - 1})",
            )
        values = {
            name: getattr(entry, name)
            for name in PLAYER_BIAS_FIELDS
            if getattr(entry, name) is not None
        }
        if values:
            team.preset.player_biases[entry.slot] = values
        else:
            team.preset.player_biases.pop(entry.slot, None)

    # If this team is playing right now, the live match should reflect the
    # change immediately -- the user expects a slider to bite mid-match.
    _push_tactics_to_live_match(tournament, update.team_id)
    return _to_dto(tournament)


@router.post("/start-match", response_model=TournamentDTO)
def start_match() -> TournamentDTO:
    """Load the user's fixture for the current matchday into the live sim."""
    global _live_fixture
    tournament = _require()
    if tournament.finished:
        raise HTTPException(status_code=400, detail="Season is over")
    fixture = tournament.user_fixture(tournament.current_matchday)
    if fixture is None or fixture.played:
        raise HTTPException(
            status_code=400, detail="No unplayed user fixture on this matchday"
        )
    simulation._state = tournament.build_match(fixture)
    # The opponent's manager reacts; the user's side does not, because those
    # sliders belong to the user and the tactics panel owns them.
    #
    # TWO ID SPACES. Tournament teams are 1-18; a live GameState always has
    # home=1 and away=2. Passing the tournament id straight through exempted
    # whichever LIVE side happened to share that number -- so on a matchday
    # where the user played away, their own team reacted and the opponent did
    # not. Resolve through the fixture instead.
    opponent_id = (
        fixture.away_id
        if fixture.home_id == tournament.user_team_id
        else fixture.home_id
    )
    opponent = tournament.team(opponent_id)
    simulation.attach_manager(
        simulation._state,
        _live_user_team_id(tournament, fixture),
        opponent_biases={
            slot: dict(v) for slot, v in opponent.preset.player_biases.items()
        },
        opponent_is_home=fixture.home_id == opponent_id,
    )
    _live_fixture = fixture
    return _to_dto(tournament)


@router.post("/finish-match", response_model=TournamentDTO)
def finish_match() -> TournamentDTO:
    """Record the live match's score and persist the user's tactics.

    Deliberately requires the match to be FINISHED: recording a score at an
    arbitrary tick would let a losing position be abandoned.
    """
    global _live_fixture
    tournament = _require()
    if _live_fixture is None or simulation._state is None:
        raise HTTPException(status_code=400, detail="No live match")
    if not simulation._state.finished:
        raise HTTPException(status_code=400, detail="Match is not finished")

    live = simulation._state
    tournament.record(
        _live_fixture, live.home_score, live.away_score, live.stats
    )
    # Persist whatever the user tuned during the match: tactics belong to the
    # team, and the user asked for changes to carry forward.
    user = tournament.team(tournament.user_team_id)
    played_as = (
        live.home_team if _live_fixture.home_id == user.team_id else live.away_team
    )
    user.preset.tactics = replace(played_as.tactics)
    # Per-player overrides are part of the same tuning arc as the team sliders,
    # so they persist too -- otherwise a slider set mid-match survives and a
    # player override silently does not.
    user.preset.player_biases = capture_player_biases(played_as)
    _live_fixture = None
    return _to_dto(tournament)


@router.post("/advance", response_model=TournamentDTO)
def advance() -> TournamentDTO:
    """Kick off this matchday's AI fixtures and return IMMEDIATELY.

    Blocked while the user's own fixture is unplayed, so the season cannot run
    away from them. Progress is reported on the TournamentDTO's `sim_job`, and
    results appear in the fixture list as each match finishes.
    """
    tournament = _require()
    if tournament.finished:
        raise HTTPException(status_code=400, detail="Season is over")
    if _sim_job is not None and _sim_job.dto().running:
        raise HTTPException(status_code=409, detail="A matchday is already simulating")

    matchday = tournament.current_matchday
    user_fixture = tournament.user_fixture(matchday)
    if user_fixture is not None and not user_fixture.played:
        raise HTTPException(
            status_code=400,
            detail="Play your match for this matchday first",
        )

    begin_matchday(tournament)
    return _to_dto(tournament)


def begin_matchday(tournament: Tournament) -> None:
    """Start the current matchday's unplayed AI fixtures, if it has any."""
    global _sim_job
    matchday = tournament.current_matchday
    pending = [f for f in tournament.ai_fixtures(matchday) if not f.played]
    if pending:
        _sim_job = _SimJob(matchday=matchday, total=len(pending))
        start_matchday(tournament, pending, _sim_job)


def _start_on_a_thread(
    tournament: Tournament, pending: list[Fixture], job: _SimJob
) -> None:
    threading.Thread(
        target=_run_matchday, args=(tournament, pending, job), daemon=True
    ).start()


# How a matchday's fixtures get run. The browser build has no threads or
# processes and replaces this (app.browser).
start_matchday = _start_on_a_thread


def record_result(tournament: Tournament, result: MatchResult) -> None:
    """Write a finished AI fixture into the season."""
    fixture = next(
        (f for f in tournament.fixtures_for(result.matchday)
         if (f.home_id, f.away_id) == (result.home_id, result.away_id)),
        None,
    )
    if fixture is None:
        return
    tournament.record(fixture, result.home_goals, result.away_goals, result.stats)
    # Persist what the rules changed. Both sides are AI here, so both develop
    # across the season.
    tournament.team(result.home_id).preset.player_biases = {
        slot: dict(v) for slot, v in result.home_player_biases.items()
    }
    tournament.team(result.away_id).preset.player_biases = {
        slot: dict(v) for slot, v in result.away_player_biases.items()
    }


def _run_matchday(
    tournament: Tournament, pending: list[Fixture], job: _SimJob
) -> None:
    """Background worker: simulate the fixtures and record them as they land."""
    manager = None
    try:
        specs = [_spec_for(tournament, f) for f in pending]
        manager, live = live_score_map()
        with job.lock:
            job.live = live
        # Explicit rather than relying on the default: rival managers reacting
        # in-match is a behavioural choice, and it should be visible at the
        # call site rather than buried in a signature.
        for result in iter_simulated(specs, live=live, reactive=True):
            record_result(tournament, result)
            with job.lock:
                job.done += 1
    except Exception as exc:  # surfaced to the user rather than lost in a thread
        with job.lock:
            job.error = str(exc)
    finally:
        with job.lock:
            job.running = False
            job.live = {}
        if manager is not None:
            manager.shutdown()


def _live_user_team_id(tournament: Tournament, fixture: Fixture) -> int:
    """The user's team id IN THE LIVE MATCH (home=1 / away=2).

    `tournament.user_team_id` is a season-long id in 1-18 and means nothing to a
    GameState, whose sides are rebuilt per fixture. Comparing the two directly
    is only correct by coincidence when the numbers happen to match.
    """
    state = simulation._state
    assert state is not None
    if fixture.home_id == tournament.user_team_id:
        return state.home_team.team_id
    return state.away_team.team_id


def _spec_for(tournament: Tournament, fixture: Fixture) -> MatchSpec:
    home = tournament.team(fixture.home_id)
    away = tournament.team(fixture.away_id)
    return MatchSpec(
        match_id=f"md{fixture.matchday}-{home.name}-v-{away.name}",
        seed=tournament.match_seed(fixture),
        home_name=home.name,
        away_name=away.name,
        home_tactics=replace(home.tactics),
        away_tactics=replace(away.tactics),
        matchday=fixture.matchday,
        home_id=fixture.home_id,
        away_id=fixture.away_id,
        # Carry the season's per-player overrides in, so a player who has been
        # adjusting all season starts this match where he left off.
        home_player_biases={
            slot: dict(v) for slot, v in home.preset.player_biases.items()
        },
        away_player_biases={
            slot: dict(v) for slot, v in away.preset.player_biases.items()
        },
    )


def _push_tactics_to_live_match(tournament: Tournament, team_id: int) -> None:
    """Mirror a stored tactics change into the running match, if it applies."""
    if _live_fixture is None or simulation._state is None:
        return
    if team_id not in (_live_fixture.home_id, _live_fixture.away_id):
        return
    stored = tournament.team(team_id).tactics
    live = simulation._state
    target = live.home_team if _live_fixture.home_id == team_id else live.away_team
    restation = (
        target.tactics.formation != stored.formation
        or target.tactics.offensiveness != stored.offensiveness
        or target.tactics.breadth != stored.breadth
        or target.tactics.compactness != stored.compactness
    )
    target.tactics = replace(stored)
    for slot, player in enumerate(target.players):
        overrides = tournament.team(team_id).preset.player_biases.get(slot, {})
        for name in PLAYER_BIAS_FIELDS:
            setattr(player, name, overrides.get(name))
    if restation:
        # Safe here for the same reason the /simulate tactics endpoint is: a
        # request handler runs between ticks, never inside one.
        apply_formation(target)


# -- Serialisation ------------------------------------------------------------

def _to_dto(tournament: Tournament) -> TournamentDTO:
    names = {t.team_id: t.name for t in tournament.teams}
    # Running scores for fixtures being simulated right now. Copied under the
    # lock: the underlying dict is a manager proxy written by other processes.
    live_now: dict = {}
    if _sim_job is not None and _sim_job.running:
        with _sim_job.lock:
            try:
                live_now = dict(_sim_job.live)
            except Exception:
                live_now = {}  # manager gone; fall back to no live scores
    user_fixture = tournament.user_fixture(tournament.current_matchday)
    table = [
        TableRowDTO(
            position=i + 1,
            team_id=row.team_id,
            name=row.name,
            played=row.played,
            won=row.won,
            drawn=row.drawn,
            lost=row.lost,
            goals_for=row.goals_for,
            goals_against=row.goals_against,
            goal_difference=row.goal_difference,
            points=row.points,
            is_user=row.team_id == tournament.user_team_id,
        )
        for i, row in enumerate(tournament.table())
    ]
    return TournamentDTO(
        seed=tournament.seed,
        user_team_id=tournament.user_team_id,
        matchdays=tournament.matchdays,
        current_matchday=tournament.current_matchday,
        finished=tournament.finished,
        teams=[
            TournamentTeamDTO(
                team_id=t.team_id,
                name=t.name,
                is_user=t.is_user,
                preset_name=t.preset.name,
                tactics=simulation._tactics_dto(t.tactics),
                player_biases=[
                    TournamentPlayerBiasDTO(slot=slot, **biases)
                    for slot, biases in sorted(t.preset.player_biases.items())
                ],
            )
            for t in tournament.teams
        ],
        fixtures=[
            FixtureDTO(
                matchday=f.matchday,
                home_id=f.home_id,
                away_id=f.away_id,
                home_name=names[f.home_id],
                away_name=names[f.away_id],
                played=f.played,
                home_goals=(
                    f.home_goals
                    if f.played
                    else live_now.get((f.home_id, f.away_id), (0, 0, 0))[0]
                ),
                away_goals=(
                    f.away_goals
                    if f.played
                    else live_now.get((f.home_id, f.away_id), (0, 0, 0))[1]
                ),
                live=not f.played and (f.home_id, f.away_id) in live_now,
                involves_user=tournament.user_team_id in (f.home_id, f.away_id),
                stats=_stats_dto(
                    f.stats,
                    tournament.team(f.home_id).tactics.formation,
                    tournament.team(f.away_id).tactics.formation,
                ),
            )
            for f in tournament.fixtures
        ],
        table=table,
        user_match_pending=user_fixture is not None and not user_fixture.played,
        sim_job=_sim_job.dto() if _sim_job is not None else None,
    )


def _stats_dto(
    stats,
    home_formation: str | None = None,
    away_formation: str | None = None,
) -> MatchStatsDTO | None:
    """Serialise a finished fixture's stats, players included.

    This used to emit only the team totals, so the Matches tab showed "no
    player data recorded" for every fixture -- the counters were collected and
    stored correctly, and then dropped here. The simulation router has its own
    serialiser which I widened when PlayerStatsDTO was added; this one was
    missed. Two serialisers for one DTO is the hazard.

    A finished fixture has no live squad to read names from -- the season keeps
    tactics, not players -- so the position comes from the formation the team
    played and the label from the slot. That is honest about what is known:
    inventing a different name would be worse than a positional one.
    """
    if stats is None:
        return None
    # TeamStats is a slots dataclass, so `vars()` raises -- read the fields by
    # name. dataclasses.asdict would also work but allocates two dicts per call.
    return MatchStatsDTO(
        home=TeamStatsDTO(**{f: getattr(stats.home, f) for f in _STAT_FIELDS}),
        away=TeamStatsDTO(**{f: getattr(stats.away, f) for f in _STAT_FIELDS}),
        home_players=_player_rows(stats.home_players, home_formation),
        away_players=_player_rows(stats.away_players, away_formation),
    )


def _player_rows(bucket: dict, formation: str | None) -> list[PlayerStatsDTO]:
    """Per-player rows for one side, ordered by squad slot."""
    slots = FORMATIONS.get(formation or "", ())
    rows: list[PlayerStatsDTO] = []
    for slot, ps in sorted(bucket.items()):
        position = slots[slot][0] if 0 <= slot < len(slots) else "?"
        rows.append(
            PlayerStatsDTO(
                slot=slot,
                name=f"{position}{slot + 1}",
                position=position,
                **{f: getattr(ps, f) for f in _PLAYER_STAT_FIELDS},
            )
        )
    return rows


_PLAYER_STAT_FIELDS = (
    "passes", "passes_completed", "passes_intercepted", "dribbles",
    "dribbles_lost", "shots", "shots_on_target", "goals", "touches",
)


_STAT_FIELDS = (
    "possession_ticks", "passes", "passes_completed", "shots",
    "shots_on_target", "goals", "corners", "throw_ins", "goal_kicks", "saves",
)
