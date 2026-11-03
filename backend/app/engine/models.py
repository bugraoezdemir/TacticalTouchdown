from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .signals import AttackCall, RunwaySignal

BallState = Literal["stationary", "in_pass", "in_shot", "in_dribble"]
Position = Literal["GK", "LD", "CD", "RD", "LM", "CM", "RM", "LF", "CF", "RF"]
TacticalInstruction = Literal["defensive", "offensive", "normal"]


@dataclass(slots=True)
class Ball:
    pos: np.ndarray       # shape (2,): [x, y]
    velocity: np.ndarray  # shape (2,): [vx, vy] in m/tick
    state: BallState
    owner_id: int | None            # player_id of current holder; None if loose
    last_toucher_team_id: int | None  # team that last played the ball (for restarts)
    # Who released the ball, and when. A passer who immediately collects his own
    # pass has achieved nothing, so the off-ball model needs to know it was him.
    last_passer_id: int | None = None
    last_pass_tick: int = -1
    # Who last knocked the ball forward on a dribble. A dribble releases
    # ownership by design, so the loss is only knowable when the next contest
    # resolves -- which may be an opponent, several ticks later.
    last_dribbler_id: int | None = None
    # Who struck the current shot. A shot in flight has no owner and no passer
    # -- measured at four goals: owner_id=None, last_passer_id=None, every time
    # -- so without this the finisher cannot be identified when it goes in.
    last_shooter_id: int | None = None
    # Teammates who were in an offside POSITION when this ball was played.
    # Being there is not an offence; receiving the ball there is. So the flag
    # is raised at the kick and only acted on if one of them collects it --
    # which is why it must be recorded here rather than recomputed later, by
    # which time the defensive line has moved.
    offside_suspects: "frozenset[int]" = frozenset()
    # Who the pass currently in flight is meant for. The off-ball model must
    # never send him somewhere else while it travels to him.
    intended_receiver_id: int | None = None
    # Where and when the dribbler meets his own touch -- set at the touch, read
    # by his next decisions so his pace is the one the touch was solved for.
    dribble_meet: "np.ndarray | None" = None
    dribble_meet_tick: int = 0
    # A player who just miscontrolled it cannot retouch it at once -- the bad
    # touch is what sent it away from him.
    no_capture_id: int | None = None
    no_capture_until: int = -1
    # Where and when it was last struck: an opponent close by has no time to
    # react for KICK_LOCKOUT_TICKS and can only block it (physics).
    struck_from: "np.ndarray | None" = None
    # Where the pass in flight was played TO: the receiver's feet, or the
    # spot of a ball into space. He decides how fast to get there.
    pass_target: "np.ndarray | None" = None
    struck_tick: int = -1
    # Height above the pitch (m) and vertical speed (m/tick, up positive).
    # Both 0 for a ball on the ground; see ball_flight.
    height: float = 0.0
    vz: float = 0.0
    # Radians a tick its flight turns in the air, left of travel positive; 0
    # for a ball struck straight. See ball_flight.
    curl: float = 0.0
    # The ball in flight is a cross (counted when a teammate collects it).
    is_cross: bool = False

    def ground(self) -> None:
        """Put it on the ground: at a player's feet or on a restart spot."""
        self.height = 0.0
        self.vz = 0.0
        self.curl = 0.0


@dataclass(slots=True)
class Player:
    player_id: int
    team_id: int
    name: str
    position: Position
    tactical: TacticalInstruction
    pos: np.ndarray          # shape (2,): current position
    velocity: np.ndarray     # shape (2,): displacement applied last tick
    speed: float             # max m/tick; derived from DB speed quality
    strength: float
    intelligence: float
    passing: float
    goalkeeping: float
    assigned_pos: np.ndarray  # formation home position, shape (2,)
    marking_target_id: int | None
    has_ball: bool
    runway_signal: RunwaySignal | None  # set when a teammate sends a runway pass signal
    # Set when the carrier calls this player forward to support an attack.
    # Persists across ticks; see signals.expire_attack_calls.
    attack_call: "AttackCall | None" = None
    # Under signals.JOIN: no new call to join an attack before this tick.
    join_cooldown_until: int = -1
    # Execution skills. These add error to *delivery*, not to the choice: a weak
    # passer still picks the right pass, they just miss it more often.
    shooting: float = 60.0
    dribbling: float = 60.0
    last_off_ball_target: np.ndarray | None = None
    # The meeting point he has committed to while running onto a pass. Held
    # across ticks so re-solving cannot jitter him -- see off_ball.
    meet_target: "np.ndarray | None" = None
    # The strike (Ball.struck_tick) of the pass he has seen is in danger and
    # gone to meet: once he goes he stays gone -- see off_ball._comes_to_ball.
    coming_to_strike: int = -2
    # A keeper's leap: the strike he left his feet for, and the y he went to.
    leapt_at: int = -2
    leap_y: float = 0.0
    # The man he marked last tick; see off_ball.STICKY_MARKS.
    marking_id: int | None = None
    # Consecutive ticks he has chosen to hold the ball. Updated in APPLY, read
    # by the next tick's decision: holding is a short wait for something to
    # open up, and its value decays the longer it goes on.
    hold_ticks: int = 0
    # The spot he was heading for last tick. Under inertia its change is how
    # fast his target is MOVING (a marker's spot moves with his man), which the
    # steering matches instead of braking to stop on a moving point.
    prev_move_target: "np.ndarray | None" = None
    # A first touch still to be taken: set by physics when he receives a ball
    # faster than he can simply trap (relative speed, m/tick) together with
    # its incoming velocity; read by his next on-ball decision, which must
    # play it first time or try to control it. 0 / None otherwise.
    first_touch_speed: float = 0.0
    first_touch_velocity: "np.ndarray | None" = None
    # How high it was when he met it. Above CONTROL_HEIGHT it is a header: it
    # can only be played first time, and with a header's pace.
    first_touch_height: float = 0.0
    # Per-player tactical bias. None means "inherit the team setting"; a float
    # is a direct multiplier on that action's utility. These are a PREFERENCE,
    # not a frequency: the engine takes an argmax over P x V x C rather than
    # sampling, so a bias makes an action win more situations, it does not
    # promise a share of them.
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
    bias_wing_attacks: float | None = None


@dataclass(slots=True)
class TeamTactics:
    """Team-wide action bias. 1.0 is neutral; players may override.

    Deliberately NOT constrained to sum to one. Shoot / pass / dribble compete
    for situations, not for a fixed budget -- a side that commits forward can
    reasonably do more of all three, and a simplex would make every slider
    silently move the other two.
    """
    shoot: float = 1.0
    pass_: float = 1.0
    dribble: float = 1.0
    # Team shape. Drives every player's assigned_pos via
    # formations.apply_formation; see that module for why it is a name and not
    # a resolved slot list.
    formation: str = "4-4-2"
    # Defensive pressing. Pulls the pursuit lambda toward 0 (go at the ball)
    # and away from 1 (align with its trajectory), and shrinks the margin by
    # which a teammate must beat you before you leave a contest to them.
    pressing: float = 1.0
    # How far from his own station a player will roam to join the attack.
    # Above 1.0 he pushes on; below it he returns to shape sooner.
    roaming: float = 1.0
    # How far up the pitch the whole shape is stationed. Above 1.0 every
    # outfield station moves toward the opponent goal; below it the team sits
    # deeper. Applied by formations.apply_formation.
    offensiveness: float = 1.0
    # Direct pass: which FLAVOUR of pass is preferred, not how often to pass.
    # Above 1.0 favours passes that do not make the receiver run; below it,
    # passes that do (on_ball.direct_factor). The Pass slider governs
    # pass-versus-anything-else.
    direct: float = 1.0
    # One-two: after a safe pass, how strongly the passer moves on to support
    # the move rather than standing over the ball he just played.
    one_two: float = 1.0
    # Attack call: how readily the player ON THE BALL summons teammates
    # forward to support the attack. It is the CARRIER's decision, not a
    # runner's own eagerness -- one player calls, several respond, and the
    # called runs last until possession is lost. Higher calls more runners.
    attack_call: float = 1.0
    # Pass range: which LENGTH of pass is preferred, not how often to pass.
    # Above 1.0 favours longer balls, below it favours short ones; 1.0 has no
    # preference either way. Like `runway` this trades between flavours via a
    # reciprocal, so it does not change pass VOLUME -- that is the Pass slider.
    pass_range: float = 1.0
    # Breadth: how far apart the shape stands LATERALLY. Above 1.0 spreads the
    # stations toward the touchlines; below it packs them centrally, which
    # shortens the gaps between teammates and so favours short passing.
    # Applied by formations.apply_formation, like offensiveness.
    breadth: float = 1.0
    # Compactness: how close teammates stay to one another, at ALL times and in
    # both phases. Above 1.0 the shape contracts toward its own centre, players
    # stray less from their stations and accept standing closer together;
    # below it the team spreads. Unlike breadth it acts on both axes, and
    # unlike the (disabled) support pull it contracts the team around its own
    # shape, not toward the ball.
    compactness: float = 1.0
    # Possession: how much keeping the ball is worth against moving it on.
    # Above 1.0 a pass or dribble that may lose the ball pays a turnover cost,
    # so when every forward option is a gamble the player recycles -- often
    # backwards -- rather than forcing it. 1.0 is exactly the old behaviour.
    possession: float = 1.0
    # Crossing: how readily a carrier on the flank crosses into the box,
    # against cutting inside, carrying on or recycling. A plain multiplier on
    # a cross's utility, like shoot / pass / dribble.
    crossing: float = 1.0
    # Lofted: how the ball travels, not how often it is passed. Above 1.0
    # balls in the air scale up and balls along the ground down by the
    # reciprocal -- crosses included, so high means high crosses and low means
    # driven ground crosses and cut-backs.
    lofted: float = 1.0
    # Risk: how much a chancy option is worth against a safe one. P is
    # replaced by P ** (1 / risk) for passes (crosses included) and dribbles:
    # above 1.0 a player gambles more, below it he plays safer. Shots are left
    # out (their P is only 'not blocked'; the chance of scoring is in V).
    risk: float = 1.0
    # Tempo: how fast a player runs with the ball. Every dribble is offered at
    # a walk, a run and a sprint (decisions.pace); above 1.0 the faster paces
    # gain and the walk loses by the reciprocal, below it the reverse. 1.0
    # expresses no preference: the pace is then chosen on P x V x C alone.
    tempo: float = 1.0
    # Press aggression: how the man going at the carrier approaches him: at 2.0
    # straight at him, at 0.5 fully onto the lane of his likeliest pass, between
    # them both at once (off_ball.press_lane_share).
    press_aggression: float = 1.0
    # Man marking: how a defender shares himself between his zone and his man.
    # Above 1.0 he stands nearer his man and follows him further from his zone,
    # below it he stays nearer the zone and hands him on sooner
    # (off_ball._zonal_mark_detail). 1.0 is the old share.
    man_marking: float = 1.0
    # Wing attacks: above 1.0 a pass to the flank gains on the rest, enough to
    # win when the two are close and never more than WING_TILT_MAX (on_ball
    # ._wing_tilt); below it the flank loses. Crossing stays its own slider.
    wing_attacks: float = 1.0

    def get(self, action: str) -> float:
        return {
            "shot": self.shoot,
            "pass": self.pass_,
            "dribble": self.dribble,
        }.get(action, 1.0)


@dataclass(slots=True)
class Team:
    team_id: int
    name: str
    players: list[Player]
    direction: int  # +1 = attacking rightward to x=105; -1 = attacking leftward to x=0
    tactics: TeamTactics = field(default_factory=TeamTactics)


@dataclass(slots=True)
class MatchEvent:
    """Something worth telling the viewer about."""
    kind: str           # goal | throw_in | corner | goal_kick | kickoff
    tick: int
    team_id: int | None      # team the restart favours, or that scored
    team_name: str
    player_name: str | None  # scorer / taker
    detail: str              # human-readable line for the UI


@dataclass(slots=True)
class TeamStats:
    """Counting stats for one side. Accumulated in the engine as the events
    happen, rather than derived afterwards from the event log, which only keeps
    the last 40 entries."""
    possession_ticks: int = 0
    passes: int = 0
    passes_completed: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    corners: int = 0
    throw_ins: int = 0
    offsides: int = 0
    goal_kicks: int = 0
    saves: int = 0
    crosses: int = 0
    crosses_completed: int = 0


@dataclass(slots=True)
class PlayerStats:
    """Counting stats for ONE player in one match.

    Keyed by squad SLOT wherever it is stored, never by player_id: ids are
    rebuilt every fixture (1-11 home, 12-22 away), so the same footballer has a
    different id depending on venue. Slot is the only identity that survives.

    These exist so a rule can ask "are MY passes being cut out?" -- a question
    TeamStats cannot answer, because it aggregates the whole side.
    """
    passes: int = 0
    passes_completed: int = 0
    passes_intercepted: int = 0
    dribbles: int = 0
    dribbles_lost: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    touches: int = 0
    offsides: int = 0

    @property
    def pass_completion(self) -> float:
        """Share of this player's passes that reached a teammate."""
        return self.passes_completed / self.passes if self.passes else 0.0

    @property
    def interception_rate(self) -> float:
        return self.passes_intercepted / self.passes if self.passes else 0.0

    @property
    def dribble_loss_rate(self) -> float:
        return self.dribbles_lost / self.dribbles if self.dribbles else 0.0

    @property
    def shot_conversion(self) -> float:
        return self.goals / self.shots if self.shots else 0.0


@dataclass(slots=True)
class MatchStats:
    home: TeamStats = field(default_factory=TeamStats)
    away: TeamStats = field(default_factory=TeamStats)
    # slot -> stats, one dict per side. Populated lazily: a player who never
    # touches the ball simply has no entry, which keeps a clean sheet clean.
    home_players: dict[int, PlayerStats] = field(default_factory=dict)
    away_players: dict[int, PlayerStats] = field(default_factory=dict)

    def side(self, state: "GameState", team_id: int) -> TeamStats:
        return self.home if team_id == state.home_team.team_id else self.away

    def player(self, state: "GameState", player: "Player") -> PlayerStats:
        """Per-player counters, created on first use.

        Resolves the player's SLOT from his position in his team's squad list,
        which is the index apply_formation assigned him.
        """
        home = player.team_id == state.home_team.team_id
        team = state.home_team if home else state.away_team
        bucket = self.home_players if home else self.away_players
        try:
            slot = team.players.index(player)
        except ValueError:
            slot = -1
        if slot not in bucket:
            bucket[slot] = PlayerStats()
        return bucket[slot]


@dataclass
class GameState:
    tick: int
    ball: Ball
    home_team: Team
    away_team: Team
    match_id: str
    half: int  # 1 or 2
    # Which side kicked off the match. The other side starts the second half,
    # so neither gets both. Team id, not the 1/2 selector.
    first_kickoff_team_id: int | None = None
    # Set at full time. Once true, tick() is a no-op: the match is over and
    # nothing further should move, decide, or be counted.
    finished: bool = False
    # Seeded so matches replay identically; never use the global numpy RNG.
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    # Ticks remaining before play resumes. While positive the world is frozen
    # except for players walking back to their stations, which gives the viewer
    # a beat to read what just happened.
    restart_pause: int = 0
    # A goal has been scored and the ball rests in the net: when the pause
    # ends, game_loop sets up the kick-off.
    kickoff_pending: bool = False
    # A restart decided at the whistle but not yet laid out: where everyone
    # and the ball go ({player_id: pos}, ball pos), applied by game_loop
    # once restart_pause has counted down to `pending_places_at`.
    pending_places: "tuple | None" = None
    pending_places_at: int = 0
    # The ball's motion while it is dead (velocity, height, vz), stepped
    # through the beat by physics.step_dead_ball; None once it is laid out.
    dead_ball_motion: "tuple | None" = None
    # Ticks left of a keeper holding the ball in his hands. While positive the
    # box stays cleared and he is not closed down; when it reaches zero he
    # distributes. Distinct from restart_pause: play continues normally
    # everywhere else on the pitch.
    gk_hold_ticks: int = 0
    home_score: int = 0
    away_score: int = 0
    stats: "MatchStats" = field(default_factory=lambda: MatchStats())
    # Rolling log; the API sends the tail so the UI can report what just
    # happened. The frontend previously had to guess at goals from ball
    # position, which missed some and invented others.
    events: list[MatchEvent] = field(default_factory=list)
    # The back caller's shape call per team id, in force and on its way
    # (decisions.shape_call). Written in APPLY only.
    shape_calls: dict = field(default_factory=dict)
    shape_calls_waiting: dict = field(default_factory=dict)

    def add_event(
        self,
        kind: str,
        team_id: int | None,
        player_name: str | None,
        detail: str,
    ) -> None:
        team_name = ""
        if team_id is not None:
            team_name = self.get_team(team_id).name
        self.events.append(
            MatchEvent(kind, self.tick, team_id, team_name, player_name, detail)
        )
        del self.events[:-40]

    # ── Lookup caches ─────────────────────────────────────────────────────────
    # Rosters do not change mid-match, so these are built once on first use.
    # Before caching, get_player was an O(n) scan over a freshly allocated list
    # and teammates_of allocated a set plus a list on every call -- both run
    # dozens of times per tick, per player.
    _all_players: list[Player] | None = field(default=None, repr=False, compare=False)
    _by_id: dict[int, Player] | None = field(default=None, repr=False, compare=False)
    _teammates: dict[int, list[Player]] | None = field(default=None, repr=False, compare=False)
    # Match-specific shuffle used to break decision-order ties without favouring
    # the side whose player ids happen to be lower.
    _tiebreak_rank: dict[int, int] | None = field(default=None, repr=False, compare=False)
    # Where the ball was before this tick's free motion, for the swept capture
    # test under inertia. None on a tick the ball was kicked: its path then
    # runs back through the kicker's feet, and he would retake his own pass.
    ball_prev_pos: "np.ndarray | None" = field(default=None, repr=False, compare=False)

    def _build_caches(self) -> None:
        self._all_players = self.home_team.players + self.away_team.players
        self._by_id = {p.player_id: p for p in self._all_players}
        self._teammates = {
            p.player_id: [q for q in team.players if q.player_id != p.player_id]
            for team in (self.home_team, self.away_team)
            for p in team.players
        }

    def invalidate_caches(self) -> None:
        """Call if a roster ever changes (substitutions, red cards)."""
        self._all_players = self._by_id = self._teammates = None

    def all_players(self) -> list[Player]:
        if self._all_players is None:
            self._build_caches()
        return self._all_players  # type: ignore[return-value]

    def get_player(self, player_id: int) -> Player:
        if self._by_id is None:
            self._build_caches()
        try:
            return self._by_id[player_id]  # type: ignore[index]
        except KeyError:
            raise KeyError(player_id) from None

    def opponents_of(self, team_id: int) -> list[Player]:
        if team_id == self.home_team.team_id:
            return self.away_team.players
        return self.home_team.players

    def teammates_of(self, player_id: int) -> list[Player]:
        if self._teammates is None:
            self._build_caches()
        try:
            return self._teammates[player_id]  # type: ignore[index]
        except KeyError:
            raise KeyError(player_id) from None

    def get_team(self, team_id: int) -> Team:
        if self.home_team.team_id == team_id:
            return self.home_team
        return self.away_team

    def get_ball_owner(self) -> Player | None:
        if self.ball.owner_id is None:
            return None
        try:
            return self.get_player(self.ball.owner_id)
        except KeyError:
            return None
