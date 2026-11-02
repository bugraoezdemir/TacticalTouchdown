"""Pydantic DTOs used at the FastAPI boundary. numpy arrays are serialised as list[float]."""
from __future__ import annotations

from pydantic import BaseModel


class BallDTO(BaseModel):
    pos: list[float]        # [x, y]
    velocity: list[float]   # [vx, vy]
    state: str              # BallState literal
    owner_id: int | None
    last_toucher_team_id: int | None = None
    height: float = 0.0     # m above the pitch


class PlayerDTO(BaseModel):
    player_id: int
    team_id: int
    name: str
    position: str
    tactical: str
    pos: list[float]
    velocity: list[float]
    speed: float
    strength: float
    intelligence: float
    passing: float
    goalkeeping: float
    shooting: float
    dribbling: float
    bias_shoot: float | None = None
    bias_pass: float | None = None
    bias_dribble: float | None = None
    bias_pressing: float | None = None
    bias_roaming: float | None = None
    bias_offensiveness: float | None = None
    bias_direct: float | None = None
    bias_one_two: float | None = None
    bias_attack_call: float | None = None
    bias_pass_range: float | None = None
    bias_breadth: float | None = None
    bias_compactness: float | None = None
    bias_possession: float | None = None
    bias_crossing: float | None = None
    bias_lofted: float | None = None
    bias_risk: float | None = None
    bias_tempo: float | None = None
    bias_press_aggression: float | None = None
    bias_man_marking: float | None = None
    assigned_pos: list[float]
    marking_target_id: int | None
    has_ball: bool
    runway_signal_target: list[float] | None
    runway_signal_sender_id: int | None


class TeamTacticsDTO(BaseModel):
    shoot: float = 1.0
    pass_: float = 1.0
    dribble: float = 1.0
    pressing: float = 1.0
    roaming: float = 1.0
    offensiveness: float = 1.0
    direct: float = 1.0
    one_two: float = 1.0
    attack_call: float = 1.0
    pass_range: float = 1.0
    breadth: float = 1.0
    compactness: float = 1.0
    possession: float = 1.0
    crossing: float = 1.0
    lofted: float = 1.0
    risk: float = 1.0
    tempo: float = 1.0
    press_aggression: float = 1.0
    man_marking: float = 1.0
    formation: str = "4-4-2"


class TeamDTO(BaseModel):
    team_id: int
    name: str
    players: list[PlayerDTO]
    direction: int
    tactics: TeamTacticsDTO


class PlayerBiasUpdate(BaseModel):
    player_id: int
    bias_shoot: float | None = None
    bias_pass: float | None = None
    bias_dribble: float | None = None
    bias_pressing: float | None = None
    bias_roaming: float | None = None
    bias_offensiveness: float | None = None
    bias_direct: float | None = None
    bias_one_two: float | None = None
    bias_attack_call: float | None = None
    bias_pass_range: float | None = None
    bias_breadth: float | None = None
    bias_compactness: float | None = None
    bias_possession: float | None = None
    bias_crossing: float | None = None
    bias_lofted: float | None = None
    bias_risk: float | None = None
    bias_tempo: float | None = None
    bias_press_aggression: float | None = None
    bias_man_marking: float | None = None


class TacticsUpdate(BaseModel):
    """Sliders are a relative bias, not a probability: they do not need to sum
    to anything, and each is clamped to [0.5, 2.0] by the engine."""
    team_id: int
    shoot: float | None = None
    pass_: float | None = None
    dribble: float | None = None
    pressing: float | None = None
    roaming: float | None = None
    offensiveness: float | None = None
    direct: float | None = None
    one_two: float | None = None
    attack_call: float | None = None
    pass_range: float | None = None
    breadth: float | None = None
    compactness: float | None = None
    possession: float | None = None
    crossing: float | None = None
    lofted: float | None = None
    risk: float | None = None
    tempo: float | None = None
    press_aggression: float | None = None
    man_marking: float | None = None
    formation: str | None = None
    players: list[PlayerBiasUpdate] = []


class MatchEventDTO(BaseModel):
    kind: str
    tick: int
    team_id: int | None
    team_name: str
    player_name: str | None
    detail: str


class TeamStatsDTO(BaseModel):
    possession_ticks: int = 0
    passes: int = 0
    passes_completed: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    corners: int = 0
    throw_ins: int = 0
    goal_kicks: int = 0
    saves: int = 0
    crosses: int = 0
    crosses_completed: int = 0


class PlayerStatsDTO(BaseModel):
    """One player's match numbers. `slot` is the formation index, which is the
    only identity stable across fixtures."""
    slot: int
    name: str
    position: str
    passes: int = 0
    passes_completed: int = 0
    passes_intercepted: int = 0
    dribbles: int = 0
    dribbles_lost: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    touches: int = 0


class MatchStatsDTO(BaseModel):
    home: TeamStatsDTO
    away: TeamStatsDTO
    home_players: list[PlayerStatsDTO] = []
    away_players: list[PlayerStatsDTO] = []


class GameStateDTO(BaseModel):
    tick: int
    ball: BallDTO
    home_team: TeamDTO
    away_team: TeamDTO
    match_id: str
    half: int
    home_score: int
    away_score: int
    restart_pause: int = 0
    finished: bool = False
    # Minutes elapsed within the current half, on the displayed 0-45 scale.
    # Served by the backend so the tick->minute mapping lives in exactly one
    # place and the UI cannot drift from it.
    clock_minute: int = 0
    clock_second: int = 0
    events: list[MatchEventDTO]
    stats: MatchStatsDTO
    # Every tick of a batched /tick request, so the pitch can play the batch
    # back tick by tick: [ball x, ball y, ball height, owner id (-1 for none),
    # home score, away score, then x, y of each player in `frame_order`].
    # Drawing only the last state slid the ball in a straight line across the
    # tick where a player took it; the owner and score are per tick so the
    # carrier ring and the scoreboard change when the pitch shows it happen.
    frames: list[list[float]] = []
    frame_order: list[int] = []


# -- Tournament ---------------------------------------------------------------

class TournamentPlayerBiasDTO(BaseModel):
    """One squad slot's stored overrides. Slot, not player_id: ids are rebuilt
    per match, so only the formation slot is stable across fixtures."""
    slot: int
    bias_shoot: float | None = None
    bias_pass: float | None = None
    bias_dribble: float | None = None
    bias_pressing: float | None = None
    bias_roaming: float | None = None
    bias_offensiveness: float | None = None
    bias_direct: float | None = None
    bias_one_two: float | None = None
    bias_attack_call: float | None = None
    bias_pass_range: float | None = None
    bias_breadth: float | None = None
    bias_compactness: float | None = None
    bias_possession: float | None = None
    bias_crossing: float | None = None
    bias_lofted: float | None = None
    bias_risk: float | None = None
    bias_tempo: float | None = None
    bias_press_aggression: float | None = None
    bias_man_marking: float | None = None


class TournamentTeamDTO(BaseModel):
    team_id: int
    name: str
    is_user: bool
    preset_name: str
    tactics: TeamTacticsDTO
    # Stored per-player overrides, one entry per slot that has any.
    player_biases: list[TournamentPlayerBiasDTO] = []


class FixtureDTO(BaseModel):
    matchday: int
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    played: bool
    home_goals: int
    away_goals: int
    involves_user: bool
    # Present once the match has been played. None for a fixture recorded
    # without stats (only the table-only tests do that).
    stats: MatchStatsDTO | None = None
    # True while this fixture is being simulated right now. Its goals are the
    # RUNNING score, published from the worker as they go in.
    live: bool = False


class TableRowDTO(BaseModel):
    position: int
    team_id: int
    name: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    is_user: bool


class SimJobDTO(BaseModel):
    """Progress of a non-blocking Advance.

    The UI polls this so a matchday's ~2.7 minutes of simulation does not lock
    the page. `done`/`total` drive the progress readout; `results` grows as
    individual matches land, so scores appear one at a time rather than all at
    the end.
    """
    running: bool
    matchday: int
    done: int
    total: int
    error: str | None = None


class TournamentDTO(BaseModel):
    seed: int
    user_team_id: int
    matchdays: int
    current_matchday: int
    finished: bool
    teams: list[TournamentTeamDTO]
    fixtures: list[FixtureDTO]
    table: list[TableRowDTO]
    # True when the user's fixture for the current matchday is still unplayed,
    # so the UI knows whether Advance is blocked on the human.
    user_match_pending: bool
    # None when no Advance has ever been started this session.
    sim_job: SimJobDTO | None = None


class TournamentCreate(BaseModel):
    seed: int = 0
    user_team_index: int = 0


class TournamentTacticsUpdate(BaseModel):
    """Tactics are owned by the TEAM and persist across matches."""
    team_id: int
    shoot: float | None = None
    pass_: float | None = None
    dribble: float | None = None
    formation: str | None = None
    pressing: float | None = None
    roaming: float | None = None
    offensiveness: float | None = None
    direct: float | None = None
    one_two: float | None = None
    attack_call: float | None = None
    pass_range: float | None = None
    breadth: float | None = None
    compactness: float | None = None
    possession: float | None = None
    crossing: float | None = None
    lofted: float | None = None
    risk: float | None = None
    tempo: float | None = None
    press_aggression: float | None = None
    man_marking: float | None = None
    # Per-player overrides to store. A slot listed here REPLACES whatever was
    # stored for it, so clearing an override is sending that slot with every
    # value None.
    players: list[TournamentPlayerBiasDTO] = []
