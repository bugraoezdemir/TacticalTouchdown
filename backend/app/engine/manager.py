"""
Reactive in-match tactics: a manager who responds to how the game is going.

Rules, not learning. The user's request was for rival teams to adjust within a
match; a hand-written policy delivers that today, and the window-by-window
reward is logged so a learned policy can be judged later against real data
rather than assumed to help.

Three pieces:

  * `window_reward` -- a dense score for a slice of a match. Goals alone are far
    too sparse to steer anything at 0-2 per match; shots, shots faced and
    territory give 10-100x the events.
  * `match_context` -- the situation a manager reacts to: are we ahead, how long
    is left, are we camped in their half or our own.
  * `MatchManager` -- swaps between named presets at window boundaries.

Deliberately NOT a learner. Half the sliders have never been shown to move an
outcome, and a bandit always converges on something -- it would launder that
uncertainty into confident-looking numbers. The window log is the evidence that
would justify learning; see the module's docstring in tournament.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .constants import DISPLAY_HALF_MINUTES, HALF_LENGTH_TICKS, PITCH_LENGTH
from .models import GameState, MatchStats, PlayerStats, TeamStats, TeamTactics
from .player_rules import RuleFiring, evaluate as evaluate_player_rules, since

# A "5 game minutes" window in ticks. 15000 ticks are displayed as 45 minutes,
# so one displayed minute is 15000/45 ticks.
TICKS_PER_DISPLAY_MINUTE = HALF_LENGTH_TICKS / DISPLAY_HALF_MINUTES
# How much play a manager judges at a time, in ticks of play rather than clock
# minutes: the rewards and player rules were measured on windows this long, and
# a shorter match gives each window no more events.
WINDOW_MINUTES = 15.0
WINDOW_TICKS = int(TICKS_PER_DISPLAY_MINUTE * WINDOW_MINUTES)  # ~1667

# Reward weights. Goals dominate because they are what actually wins matches --
# the denser terms are there to make the signal *steerable* between goals, not
# to replace the objective. A reward that pays too well for shots finds tactics
# that shoot from 30 m and lose.
#
# UNVALIDATED: these are a starting point, not a tuned result. Whether this
# correlates with winning is exactly what the logged windows are for.
REWARD_WEIGHTS: dict[str, float] = {
    "goal_for": 10.0,
    "goal_against": -10.0,
    "shot_on_target_for": 1.5,
    "shot_on_target_against": -1.5,
    "shot_for": 0.4,
    "shot_against": -0.4,
    "territory": 2.0,     # share of the window spent in their half, minus ours
    "possession": 1.0,    # share of the window we owned the ball
}


@dataclass(frozen=True, slots=True)
class StatSnapshot:
    """The counters that matter, frozen at a window boundary."""

    goals: int = 0
    shots: int = 0
    shots_on_target: int = 0
    possession_ticks: int = 0

    @classmethod
    def of(cls, side: TeamStats) -> "StatSnapshot":
        return cls(
            goals=side.goals,
            shots=side.shots,
            shots_on_target=side.shots_on_target,
            possession_ticks=side.possession_ticks,
        )

    def since(self, earlier: "StatSnapshot") -> "StatSnapshot":
        return StatSnapshot(
            goals=self.goals - earlier.goals,
            shots=self.shots - earlier.shots,
            shots_on_target=self.shots_on_target - earlier.shots_on_target,
            possession_ticks=self.possession_ticks - earlier.possession_ticks,
        )


def window_reward(
    ours: StatSnapshot,
    theirs: StatSnapshot,
    territory: float,
    window_ticks: int,
) -> float:
    """How well did this window go for us? Positive is good.

    `territory` is in [-1, 1]: +1 means the ball spent the whole window in the
    opponent half, -1 means our own. Possession is a share of the window, so
    both are scale-free and a short final window is not penalised for brevity.
    """
    span = max(1, window_ticks)
    possession = ours.possession_ticks / span
    opponent_possession = theirs.possession_ticks / span
    w = REWARD_WEIGHTS
    return (
        w["goal_for"] * ours.goals
        + w["goal_against"] * theirs.goals
        + w["shot_on_target_for"] * ours.shots_on_target
        + w["shot_on_target_against"] * theirs.shots_on_target
        + w["shot_for"] * ours.shots
        + w["shot_against"] * theirs.shots
        + w["territory"] * territory
        + w["possession"] * (possession - opponent_possession)
    )


@dataclass(frozen=True, slots=True)
class MatchContext:
    """The situation a manager reacts to."""

    margin: int           # our goals minus theirs
    minutes_left: float   # displayed minutes remaining in the match
    territory: float      # [-1, 1], + is their half

    @property
    def bucket(self) -> tuple[str, str]:
        """Coarse context, and the key a learned policy would index on.

        Three margin buckets and three time buckets. Kept coarse deliberately:
        a finer grid needs more data per cell than a season provides.
        """
        if self.margin > 0:
            margin = "leading"
        elif self.margin < 0:
            margin = "trailing"
        else:
            margin = "level"
        if self.minutes_left <= 15.0:
            phase = "late"
        elif self.minutes_left <= 45.0:
            phase = "second_half"
        else:
            phase = "early"
        return margin, phase


def match_context(state: GameState, team_id: int, territory: float) -> MatchContext:
    """Read the current situation from the live state, for one side."""
    home = team_id == state.home_team.team_id
    ours = state.home_score if home else state.away_score
    theirs = state.away_score if home else state.home_score
    total_ticks = 2 * HALF_LENGTH_TICKS
    remaining = max(0, total_ticks - state.tick)
    return MatchContext(
        margin=ours - theirs,
        minutes_left=remaining / TICKS_PER_DISPLAY_MINUTE,
        territory=territory,
    )


# Every per-player override the rules may write. Kept here so pushing stored
# values onto a live player cannot silently miss a field.
PLAYER_BIAS_NAMES = (
    "bias_shoot", "bias_pass", "bias_dribble", "bias_pressing", "bias_roaming",
    "bias_offensiveness", "bias_direct", "bias_one_two", "bias_attack_call",
    "bias_pass_range", "bias_breadth", "bias_compactness", "bias_possession",
    "bias_crossing", "bias_lofted", "bias_risk", "bias_tempo",
    "bias_press_aggression",
    "bias_man_marking",
)

# A brief hold when a player changes his own approach, so the change is
# readable rather than silent. Shorter than a throw-in: it is a note, not a
# stoppage.
RULE_CHANGE_PAUSE_TICKS = 20

# Friendly names for the sliders a rule can move, for the match log.
_SLIDER_LABELS = {
    "bias_shoot": "shooting",
    "bias_pass": "passing",
    "bias_dribble": "dribbling",
    "bias_pass_range": "pass range",
    "bias_pressing": "pressing",
    "bias_roaming": "roaming",
    "bias_offensiveness": "depth",
    "bias_direct": "direct passes",
    "bias_one_two": "one-twos",
    "bias_attack_call": "attacking runs",
    "bias_breadth": "width",
    "bias_compactness": "compactness",
    "bias_possession": "possession",
    "bias_crossing": "crossing",
    "bias_lofted": "lofted balls",
    "bias_risk": "risk",
    "bias_tempo": "tempo",
    "bias_press_aggression": "press aggression",
    "bias_man_marking": "man marking",
}


@dataclass(slots=True)
class WindowRecord:
    """One evaluated window. The evidence a learned policy would train on."""

    team_id: int
    tick: int
    preset: str
    margin: int
    minutes_left: float
    territory: float
    reward: float


@dataclass(slots=True)
class TeamManager:
    """Reactive tactics for one side of one match."""

    team_id: int
    base: TeamTactics
    enabled: bool = True
    # slot -> stored overrides, shared with the season record so rule-driven
    # changes persist. None when this side is not rule-managed (the user's).
    player_biases: dict[int, dict[str, float]] | None = None
    # slot -> his stats at the last evaluation. Rules judge the slice since
    # then, not the season total, so an adjustment can be re-earned or lapse.
    previous: dict[int, PlayerStats] = field(default_factory=dict)
    _last_ours: StatSnapshot = field(default_factory=StatSnapshot)
    _last_theirs: StatSnapshot = field(default_factory=StatSnapshot)
    _window_start_tick: int = 0
    _territory_ticks: int = 0
    _their_half_ticks: int = 0


class MatchManager:
    """Drives both sides' reactive tactics for a single match.

    Called once per tick, but only *acts* on window boundaries. The per-tick
    cost is one comparison plus a territory sample -- deliberately trivial,
    because `opponent_risk` already dominates the tick at ~1,076 calls and this
    must not become a second hot path.

    Mutates `team.tactics` in place BETWEEN ticks, which is the same contract
    the live tactics endpoint already relies on: decisions read `team.tactics`
    fresh every tick, so a write lands on the next one.
    """

    def __init__(
        self,
        state: GameState,
        home_enabled: bool = True,
        away_enabled: bool = True,
        home_player_biases: dict[int, dict[str, float]] | None = None,
        away_player_biases: dict[int, dict[str, float]] | None = None,
    ) -> None:
        self.home = TeamManager(
            team_id=state.home_team.team_id,
            base=replace(state.home_team.tactics),
            enabled=home_enabled,
            player_biases=home_player_biases,
        )
        self.away = TeamManager(
            team_id=state.away_team.team_id,
            base=replace(state.away_team.tactics),
            enabled=away_enabled,
            player_biases=away_player_biases,
        )
        self.log: list[WindowRecord] = []
        self.firings: list[RuleFiring] = []

    def observe(self, state: GameState) -> None:
        """Sample territory. Cheap enough to run every tick."""
        in_their_half_home = float(state.ball.pos[0]) > PITCH_LENGTH / 2.0
        for side, theirs in ((self.home, in_their_half_home),
                             (self.away, not in_their_half_home)):
            side._territory_ticks += 1
            if theirs:
                side._their_half_ticks += 1

    def maybe_adjust(self, state: GameState) -> None:
        """At a window boundary, score the window and pick the next preset."""
        for side in (self.home, self.away):
            if state.tick - side._window_start_tick < WINDOW_TICKS:
                continue
            self._close_window(state, side)

    def _close_window(self, state: GameState, side: TeamManager) -> None:
        home = side.team_id == state.home_team.team_id
        stats: MatchStats = state.stats
        ours_now = StatSnapshot.of(stats.home if home else stats.away)
        theirs_now = StatSnapshot.of(stats.away if home else stats.home)
        ours = ours_now.since(side._last_ours)
        theirs = theirs_now.since(side._last_theirs)

        span = max(1, side._territory_ticks)
        territory = (2.0 * side._their_half_ticks / span) - 1.0
        window_ticks = max(1, state.tick - side._window_start_tick)

        reward = window_reward(ours, theirs, territory, window_ticks)
        context = match_context(state, side.team_id, territory)

        self.log.append(
            WindowRecord(
                team_id=side.team_id,
                tick=state.tick,
                preset="",
                margin=context.margin,
                minutes_left=context.minutes_left,
                territory=territory,
                reward=reward,
            )
        )

        if side.enabled:
            self._apply_player_rules(state, side)

        side._last_ours = ours_now
        side._last_theirs = theirs_now
        side._window_start_tick = state.tick
        side._territory_ticks = 0
        side._their_half_ticks = 0

    def _apply_player_rules(self, state: GameState, side: TeamManager) -> None:
        """Let each player react to his OWN recent numbers.

        Judged on the slice since his last evaluation rather than the season
        total: a cumulative rate, once crossed, can never uncross, so the
        trigger stays on and the slider walks to its clamp. Measured before the
        fix: 207 firings in 18 windows, three players pinned at the floor in a
        single match.

        Writes into the season's stored overrides, so the adjustment persists
        past the final whistle. Skipped for a side without a store, which is
        how the user's team stays exempt.
        """
        if side.player_biases is None:
            return
        team = (
            state.home_team
            if side.team_id == state.home_team.team_id
            else state.away_team
        )
        for slot, player in enumerate(team.players):
            if player.position == "GK":
                continue          # a keeper's numbers mean something different
            total = state.stats.player(state, player)
            recent = since(total, side.previous.get(slot, PlayerStats()))
            # Shots are the exception: only 35 of 120 player-matches contain
            # even one, so a recent slice could never reach the 15-shot bar.
            recent = replace(recent, shots=total.shots, goals=total.goals)
            side.previous[slot] = replace(total)

            biases = side.player_biases.setdefault(slot, {})
            firings = evaluate_player_rules(recent, slot, biases)
            for name in PLAYER_BIAS_NAMES:
                setattr(player, name, biases.get(name))
            if not biases:
                side.player_biases.pop(slot, None)
            if firings:
                self.firings.extend(firings)
                self._announce(state, player, firings)

    def _announce(self, state: GameState, player, firings) -> None:
        """Log what the player changed, and hold play briefly so it reads.

        One line per adjustment would flood the log -- a player can move two
        sliders at once -- so the firings are summarised into a single note
        naming why he changed and which way each slider went.
        """
        parts = []
        for f in firings:
            label = _SLIDER_LABELS.get(f.field, f.field)
            direction = "more" if f.after > f.before else "less"
            parts.append(f"{direction} {label}")
        why = firings[0].why
        state.add_event(
            "player_adjust",
            player.team_id,
            player.name,
            f"{player.name}: {why} - {', '.join(parts)}",
        )
        # Only pause if play is otherwise running, so this never extends a
        # restart that is already holding the match.
        if state.restart_pause <= 0:
            state.restart_pause = RULE_CHANGE_PAUSE_TICKS

