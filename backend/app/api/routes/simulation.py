from __future__ import annotations

from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, HTTPException

from ...db.schemas import (
    MatchStatsDTO,
    PlayerStatsDTO,
    TeamStatsDTO,
    BallDTO,
    GameStateDTO,
    MatchEventDTO,
    PlayerDTO,
    TacticsUpdate,
    TeamDTO,
    TeamTacticsDTO,
)
from ...engine.constants import DISPLAY_HALF_MINUTES, HALF_LENGTH_TICKS
from ...engine.factory import build_test_state
from ...engine.formations import FORMATIONS, apply_formation
from ...engine.game_loop import tick
from ...engine.manager import MatchManager
from ...engine.models import GameState, Player, Team

router = APIRouter()

# Server-owned simulation state — no client round-trip needed.
_state: GameState | None = None

# Reactive tactics for the live match. Rebuilt whenever `_state` is, because a
# manager caches each side's BASE tactics -- a stale one would carry the
# previous match's identity into the next fixture.
_manager: MatchManager | None = None


def attach_manager(
    state: GameState,
    user_team_id: int | None = None,
    opponent_biases: dict | None = None,
    opponent_is_home: bool | None = None,
) -> None:
    """Give the live match a reactive-tactics manager.

    `user_team_id` names the side the human controls; that side does NOT react,
    because its sliders belong to the user and a manager moving them would
    fight the tactics panel and silently overwrite what was set there.

    Passing None means NOBODY is exempt -- both sides react. That is right for
    an AI-vs-AI fixture nobody is watching, and wrong for anything the user is
    playing: in the Match tab the user owns the HOME side (it is what the
    tactics panel and the scoreboard treat as theirs), so that caller passes
    the home team id rather than None.
    """
    global _manager
    # Per-player rules run only for the side the user does NOT control, and
    # only when the caller supplied a store to persist them into.
    home_biases = away_biases = None
    if opponent_biases is not None and opponent_is_home is not None:
        if opponent_is_home:
            home_biases = opponent_biases
        else:
            away_biases = opponent_biases
    _manager = MatchManager(
        state,
        home_enabled=user_team_id != state.home_team.team_id,
        away_enabled=user_team_id != state.away_team.team_id,
        home_player_biases=home_biases,
        away_player_biases=away_biases,
    )


@router.post("/start", response_model=GameStateDTO)
def start_simulation() -> GameStateDTO:
    """Begin play. Builds a fresh friendly ONLY when nothing is loaded.

    The UI calls this whenever it goes from idle to running, which used to
    overwrite whatever was in the slot -- so a tournament fixture loaded by
    /tournament/start-match was replaced by a default Home FC v Away FC the
    moment the user pressed play.

    The guard asks whether the TOURNAMENT owns the current match, not merely
    whether some state exists. "Resume anything unfinished" also works for the
    user, but it makes /simulate/start non-idempotent for everyone else: the
    Match tab could never restart, and tests that call /start for a clean slate
    would silently inherit the previous test's match.
    """
    global _state
    if not _tournament_match_is_loaded():
        _state = build_test_state()
        # The user controls the home side in a friendly, so only the opponent
        # adapts. Passing None here left the user's own sliders being rewritten
        # mid-match, which is exactly what they are not for.
        attach_manager(_state, _state.home_team.team_id)
    return _to_dto(_state)


# Ticks one request may run. The browser asks for as many as are due since its
# last request (30 a second at 1x), so a slow frame catches up instead of the
# whole match slowing down; this only bounds a single catch-up.
MAX_TICKS_PER_REQUEST = 60


@router.post("/tick", response_model=GameStateDTO)
def simulation_tick(n: int = 1) -> GameStateDTO:
    """Advance `n` ticks (1..MAX_TICKS_PER_REQUEST) and return the last state.

    The event log keeps the last 40 events, far more than a batch produces,
    so the UI (which de-duplicates them) misses none.
    """
    global _state
    if _state is None:
        raise HTTPException(status_code=400, detail="Call /start first")
    players = _state.all_players()
    frames: list[list[float]] = []
    for _ in range(max(1, min(MAX_TICKS_PER_REQUEST, n))):
        if _state.finished:
            break
        _state = tick(_state)
        # Reactive tactics run here too, not only in headless matches --
        # otherwise a rival never adapts in the match the user is watching.
        if _manager is not None:
            _manager.observe(_state)
            _manager.maybe_adjust(_state)
        b = _state.ball
        frame = [float(b.pos[0]), float(b.pos[1]), float(b.height),
                 # Nobody is shown on the ball until the taker is on the spot.
                 float(b.owner_id if b.owner_id is not None and _state.pending_places is None
                       else -1),
                 float(_state.home_score), float(_state.away_score)]
        for p in players:
            frame += [float(p.pos[0]), float(p.pos[1])]
        frames.append(frame)
    dto = _to_dto(_state)
    dto.frames = frames
    dto.frame_order = [p.player_id for p in players]
    return dto


@router.post("/tactics", response_model=GameStateDTO)
def set_tactics(update: TacticsUpdate) -> GameStateDTO:
    """Adjust one team's action bias, and optionally individual players'.

    Takes effect on the next tick; the match is not restarted.
    """
    global _state
    if _state is None:
        raise HTTPException(status_code=400, detail="Call /start first")

    team = _state.get_team(update.team_id)
    if update.shoot is not None:
        team.tactics.shoot = update.shoot
    if update.pass_ is not None:
        team.tactics.pass_ = update.pass_
    if update.dribble is not None:
        team.tactics.dribble = update.dribble
    if update.pressing is not None:
        team.tactics.pressing = update.pressing
    if update.roaming is not None:
        team.tactics.roaming = update.roaming
    if update.offensiveness is not None:
        team.tactics.offensiveness = update.offensiveness
    if update.direct is not None:
        team.tactics.direct = update.direct
    if update.one_two is not None:
        team.tactics.one_two = update.one_two
    if update.attack_call is not None:
        team.tactics.attack_call = update.attack_call
    if update.pass_range is not None:
        team.tactics.pass_range = update.pass_range
    if update.breadth is not None:
        team.tactics.breadth = update.breadth
    if update.compactness is not None:
        team.tactics.compactness = update.compactness
    if update.possession is not None:
        team.tactics.possession = update.possession
    if update.crossing is not None:
        team.tactics.crossing = update.crossing
    if update.lofted is not None:
        team.tactics.lofted = update.lofted
    if update.risk is not None:
        team.tactics.risk = update.risk
    if update.tempo is not None:
        team.tactics.tempo = update.tempo
    if update.press_aggression is not None:
        team.tactics.press_aggression = update.press_aggression
    if update.man_marking is not None:
        team.tactics.man_marking = update.man_marking
    if update.formation is not None:
        # A name cannot be clamped the way a bias can, so reject it outright.
        if update.formation not in FORMATIONS:
            raise HTTPException(
                status_code=400, detail=f"Unknown formation {update.formation!r}"
            )
        team.tactics.formation = update.formation
    # Player overrides are applied BEFORE re-stationing, because
    # apply_formation resolves each player's depth from their own override.
    # Breadth joins depth here: both rewrite assigned_pos, so both must
    # re-station. Pass range does NOT -- it only tilts pass scoring.
    restation = (
        update.formation is not None
        or update.offensiveness is not None
        or update.breadth is not None
        or update.compactness is not None
    )

    for entry in update.players:
        try:
            player = _state.get_player(entry.player_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"No player {entry.player_id}"
            ) from None
        if player.team_id != update.team_id:
            raise HTTPException(
                status_code=400,
                detail=f"Player {entry.player_id} is not on team {update.team_id}",
            )
        player.bias_shoot = entry.bias_shoot
        player.bias_pass = entry.bias_pass
        player.bias_dribble = entry.bias_dribble
        player.bias_pressing = entry.bias_pressing
        player.bias_roaming = entry.bias_roaming
        # Compare before assigning: a depth override being CLEARED back to the
        # team value must re-station just as much as one being set.
        if player.bias_offensiveness != entry.bias_offensiveness:
            restation = True
        player.bias_offensiveness = entry.bias_offensiveness
        if player.bias_breadth != entry.bias_breadth:
            restation = True
        player.bias_breadth = entry.bias_breadth
        if player.bias_compactness != entry.bias_compactness:
            restation = True
        player.bias_compactness = entry.bias_compactness
        player.bias_direct = entry.bias_direct
        player.bias_one_two = entry.bias_one_two
        player.bias_attack_call = entry.bias_attack_call
        player.bias_pass_range = entry.bias_pass_range
        player.bias_possession = entry.bias_possession
        player.bias_crossing = entry.bias_crossing
        player.bias_lofted = entry.bias_lofted
        player.bias_risk = entry.bias_risk
        player.bias_tempo = entry.bias_tempo
        player.bias_press_aggression = entry.bias_press_aggression
        player.bias_man_marking = entry.bias_man_marking

    if restation:
        # Formation, team depth and per-player depth all feed assigned_pos.
        apply_formation(team)

    return _to_dto(_state)


@router.post("/reset", response_model=GameStateDTO)
def reset_simulation() -> GameStateDTO:
    """Start over. Replays the loaded tournament fixture, if there is one.

    Rebuilding a friendly here would silently drop the user's league match --
    and because the fixture's seed is fixed, replaying it is the same match
    again rather than a reroll for a better result.
    """
    global _state
    _state = _reload_tournament_match() or build_test_state()
    # A league fixture knows who the user is; a friendly is always home.
    attach_manager(
        _state, _tournament_user_team_id() or _state.home_team.team_id
    )
    return _to_dto(_state)


def _tournament_user_team_id() -> int | None:
    """The user's team id IN THE LIVE MATCH (home / away), or None.

    TWO ID SPACES. `Tournament.user_team_id` is a season-long id in 1-18; a
    GameState's sides are rebuilt per fixture and are always home=1, away=2.
    Returning the season id exempted whichever live side happened to share that
    number, so on a matchday where the user played away their own team reacted
    and the opponent did not. Resolve through the fixture's home/away ids.
    """
    from . import tournament

    season = tournament._tournament
    fixture = tournament._live_fixture
    if season is None or fixture is None or _state is None:
        return None
    if fixture.home_id == season.user_team_id:
        return _state.home_team.team_id
    return _state.away_team.team_id


def _tournament_match_is_loaded() -> bool:
    """Is the state slot currently holding a live league fixture?"""
    from . import tournament

    return (
        _state is not None
        and tournament._tournament is not None
        and tournament._live_fixture is not None
    )


def _reload_tournament_match() -> GameState | None:
    """Rebuild the live tournament fixture, or None if we are not in a season.

    Imported lazily: the tournament router imports this module, so a top-level
    import here would be circular.
    """
    from . import tournament

    if tournament._tournament is None or tournament._live_fixture is None:
        return None
    return tournament._tournament.build_match(tournament._live_fixture)


# ── Serialisation (domain → DTO) ─────────────────────────────────────────────

def _clock(state: GameState) -> tuple[int, int]:
    """Match clock within the current half, on the displayed 0-45 scale.

    Display only: the simulation is unscaled and nothing in the engine reads
    this. Capped at the half length so a full-time state reads 45'00 rather
    than overrunning.
    """
    into_half = state.tick - (state.half - 1) * HALF_LENGTH_TICKS
    into_half = max(0, min(into_half, HALF_LENGTH_TICKS))
    seconds = into_half / HALF_LENGTH_TICKS * DISPLAY_HALF_MINUTES * 60.0
    return int(seconds // 60), int(seconds % 60)


def _to_dto(state: GameState) -> GameStateDTO:
    return GameStateDTO(
        tick=state.tick,
        ball=BallDTO(
            pos=state.ball.pos.tolist(),
            velocity=state.ball.velocity.tolist(),
            state=state.ball.state,
            owner_id=state.ball.owner_id,
            last_toucher_team_id=state.ball.last_toucher_team_id,
            height=float(state.ball.height),
        ),
        home_team=_team_dto(state.home_team),
        away_team=_team_dto(state.away_team),
        match_id=state.match_id,
        half=state.half,
        home_score=state.home_score,
        away_score=state.away_score,
        restart_pause=state.restart_pause,
        finished=state.finished,
        clock_minute=_clock(state)[0],
        clock_second=_clock(state)[1],
        stats=_match_stats_dto(state),
        events=[
            MatchEventDTO(
                kind=e.kind, tick=e.tick, team_id=e.team_id,
                team_name=e.team_name, player_name=e.player_name, detail=e.detail,
            )
            for e in state.events[-8:]
        ],
    )


def _player_rows(state: GameState, home: bool) -> list[PlayerStatsDTO]:
    """Per-player rows for one side, ordered by squad slot.

    Names and positions are resolved from the live squad: PlayerStats is keyed
    by slot alone, because that is the only identity stable across fixtures.
    """
    team = state.home_team if home else state.away_team
    bucket = state.stats.home_players if home else state.stats.away_players
    rows: list[PlayerStatsDTO] = []
    for slot, stats in sorted(bucket.items()):
        if not 0 <= slot < len(team.players):
            continue
        player = team.players[slot]
        rows.append(
            PlayerStatsDTO(
                slot=slot,
                name=player.name,
                position=player.position,
                **asdict(stats),
            )
        )
    return rows


def _match_stats_dto(state: GameState) -> MatchStatsDTO:
    return MatchStatsDTO(
        home=TeamStatsDTO(**asdict(state.stats.home)),
        away=TeamStatsDTO(**asdict(state.stats.away)),
        home_players=_player_rows(state, home=True),
        away_players=_player_rows(state, home=False),
    )


def _tactics_dto(tactics) -> TeamTacticsDTO:
    """One definition, shared with the tournament router.

    Adding a slider means editing this once; two copies drifted apart is
    exactly how a new parameter ends up invisible in one tab.
    """
    return TeamTacticsDTO(
        shoot=tactics.shoot,
        pass_=tactics.pass_,
        dribble=tactics.dribble,
        pressing=tactics.pressing,
        roaming=tactics.roaming,
        offensiveness=tactics.offensiveness,
        direct=tactics.direct,
        one_two=tactics.one_two,
        attack_call=tactics.attack_call,
        pass_range=tactics.pass_range,
        breadth=tactics.breadth,
        compactness=tactics.compactness,
        possession=tactics.possession,
        crossing=tactics.crossing,
        lofted=tactics.lofted,
        risk=tactics.risk,
        tempo=tactics.tempo,
        press_aggression=tactics.press_aggression,
        man_marking=tactics.man_marking,
        formation=tactics.formation,
    )


def _team_dto(team: Team) -> TeamDTO:
    return TeamDTO(
        team_id=team.team_id,
        name=team.name,
        players=[_player_dto(p) for p in team.players],
        direction=team.direction,
        tactics=_tactics_dto(team.tactics),
    )


def _player_dto(p: Player) -> PlayerDTO:
    return PlayerDTO(
        player_id=p.player_id,
        team_id=p.team_id,
        name=p.name,
        position=p.position,
        tactical=p.tactical,
        pos=p.pos.tolist(),
        velocity=p.velocity.tolist(),
        speed=p.speed,
        strength=p.strength,
        intelligence=p.intelligence,
        passing=p.passing,
        goalkeeping=p.goalkeeping,
        shooting=p.shooting,
        dribbling=p.dribbling,
        bias_shoot=p.bias_shoot,
        bias_pass=p.bias_pass,
        bias_dribble=p.bias_dribble,
        bias_pressing=p.bias_pressing,
        bias_roaming=p.bias_roaming,
        bias_offensiveness=p.bias_offensiveness,
        bias_direct=p.bias_direct,
        bias_one_two=p.bias_one_two,
        bias_attack_call=p.bias_attack_call,
        bias_pass_range=p.bias_pass_range,
        bias_breadth=p.bias_breadth,
        bias_compactness=p.bias_compactness,
        bias_possession=p.bias_possession,
        bias_crossing=p.bias_crossing,
        bias_lofted=p.bias_lofted,
        bias_risk=p.bias_risk,
        bias_tempo=p.bias_tempo,
        bias_press_aggression=p.bias_press_aggression,
        bias_man_marking=p.bias_man_marking,
        assigned_pos=p.assigned_pos.tolist(),
        marking_target_id=p.marking_target_id,
        has_ball=p.has_ball,
        runway_signal_target=(
            p.runway_signal.target_pos.tolist() if p.runway_signal else None
        ),
        runway_signal_sender_id=(
            p.runway_signal.sender_id if p.runway_signal else None
        ),
    )
