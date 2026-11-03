"""
Tournament: eight teams, double round-robin, tactics that persist.

The engine simulates one match at a time and knows nothing about seasons, so a
tournament is a plain record of teams, fixtures and results plus the code to
turn a fixture into a GameState. Nothing here runs inside `tick`.

Two design points worth stating, because they shape the API:

  * TACTICS ARE OWNED BY THE TEAM, not by a match. Every match is built FROM a
    team's stored tactics and, for the human side, whatever the user changed
    during the match is written back afterwards. Squads are deliberately
    identical for now, so the league table is a pure test of tactics.

  * TACTICS ARE STORED AS A NAMED PRESET. A team currently holds exactly one,
    but the record is a `TacticsPreset` rather than a bare `TeamTactics` so that
    holding several named sets later is a list change, not a migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .constants import TACTIC_BIAS_MAX, TACTIC_BIAS_MIN
from .factory import build_test_state
from .formations import FORMATIONS
from .models import GameState, MatchStats, TeamTactics

# German cities, none of them a real club name.
CITY_NAMES: tuple[str, ...] = (
    "Aachen",
    "Augsburg",
    "Berlin",
    "Bochum",
    "Bremen",
    "Dortmund",
    "Dresden",
    "Essen",
    "Freiburg",
    "Hamburg",
    "Hannover",
    "Kiel",
    "Leipzig",
    "Mainz",
    "Munster",
    "Nurnberg",
    "Rostock",
    "Stuttgart",
)

TEAM_COUNT = len(CITY_NAMES)

# The sliders an AI team gets a random value for. `formation` is drawn
# separately because it is a name, not a number.
_RANDOM_SLIDERS: tuple[str, ...] = (
    "shoot",
    "pass_",
    "dribble",
    "pressing",
    "roaming",
    "offensiveness",
    "direct",
    "one_two",
    "attack_call",
    "pass_range",
    "breadth",
    "compactness",
)

# Random tactics are drawn from a normal centred on neutral rather than a
# uniform over the whole clamp range: a uniform draw makes every team an
# extremist on half its sliders, which reads as noise rather than as a
# tactical identity. This keeps most values near 1.0 with occasional strong
# preferences, and the clamp catches the tail.
_SLIDER_SIGMA = 0.28


# Every per-player override a Player carries. Kept as a tuple so the preset,
# the API and the write-back all agree on the list -- a field missing from one
# of them is an override that silently fails to persist.
PLAYER_BIAS_FIELDS: tuple[str, ...] = (
    "bias_shoot",
    "bias_pass",
    "bias_dribble",
    "bias_pressing",
    "bias_roaming",
    "bias_offensiveness",
    "bias_direct",
    "bias_one_two",
    "bias_attack_call",
    "bias_pass_range",
    "bias_breadth",
    "bias_compactness",
    "bias_possession",
    "bias_crossing",
    "bias_lofted",
    "bias_risk",
    "bias_tempo",
    "bias_press_aggression",
    "bias_man_marking",
    "bias_wing_attacks",
)


@dataclass(slots=True)
class TacticsPreset:
    """A named set of tactics: the team's sliders AND its per-player overrides.

    Teams hold one today. The name exists so that storing several per team --
    the user's stated next step -- is adding a list, not reshaping the record.

    `player_biases` maps SQUAD SLOT (0-10, the index into the formation) to that
    player's overrides. Deliberately not keyed on player_id: ids are rebuilt for
    every match as 1-11 for the home side and 12-22 for the away side, so the
    same footballer has a different id depending on where the fixture is played.
    Slot is the only identity that survives a rebuild.
    """

    name: str
    tactics: TeamTactics
    player_biases: dict[int, dict[str, float]] = field(default_factory=dict)

    def copy(self) -> "TacticsPreset":
        return TacticsPreset(
            name=self.name,
            tactics=replace(self.tactics),
            player_biases={
                slot: dict(biases) for slot, biases in self.player_biases.items()
            },
        )


@dataclass(slots=True)
class TournamentTeam:
    team_id: int
    name: str
    is_user: bool
    preset: TacticsPreset

    @property
    def tactics(self) -> TeamTactics:
        return self.preset.tactics


@dataclass(slots=True)
class Fixture:
    matchday: int          # 1-based
    home_id: int
    away_id: int
    played: bool = False
    home_goals: int = 0
    away_goals: int = 0
    # The full counting stats for the match, kept for the whole season. A
    # headless match's GameState is discarded the moment its worker exits, so
    # if the stats are not copied here they are gone: the fixture list is the
    # only durable record of a game the user never watched.
    stats: MatchStats | None = None

    @property
    def involves_user(self) -> bool:
        return False  # filled in by the Tournament, which knows who the user is


@dataclass(slots=True)
class TableRow:
    team_id: int
    name: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return self.won * 3 + self.drawn

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


def random_tactics(rng: np.random.Generator) -> TeamTactics:
    """A tactical identity: mostly sensible, with a few real preferences."""
    tactics = TeamTactics()
    for slider in _RANDOM_SLIDERS:
        value = float(rng.normal(1.0, _SLIDER_SIGMA))
        setattr(
            tactics, slider, min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value))
        )
    names = sorted(FORMATIONS)
    tactics.formation = names[int(rng.integers(len(names)))]
    return tactics


# Possession is drawn from its OWN generator (see create_tournament), not from
# the shared one: inserting a draw there would shift every later one, and the
# same seed would silently produce different tactics, formations and fixtures.
# Drawn at or above 1.0 because below it the slider has no effect yet -- a
# spread around 1.0 would show variety on half the teams that does nothing.
_POSSESSION_SPREAD = 0.35


def random_possession(rng: np.random.Generator) -> float:
    """A possession preference between 1.0 (direct, the default) and 2.0."""
    return float(min(TACTIC_BIAS_MAX, 1.0 + abs(rng.normal(0.0, _POSSESSION_SPREAD))))


# Crossing and lofted balls mean something on both sides of 1.0, so they are
# spread around it. Their own generator for the same reason as possession.
_BALL_STYLE_SIGMA = 0.30


def random_ball_style(rng: np.random.Generator) -> tuple[float, float]:
    """(crossing, lofted) preferences around 1.0, clamped like every slider."""
    return tuple(
        float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, rng.normal(1.0, _BALL_STYLE_SIGMA))))
        for _ in range(2)
    )


# Sliders added since, on their own generator for the same reason. A new slider
# goes here, or every AI team plays it at neutral.
_LATER_SLIDERS: tuple[str, ...] = ("risk", "tempo", "press_aggression", "man_marking", "wing_attacks")


def random_later_sliders(tactics: TeamTactics, rng: np.random.Generator) -> None:
    """Draw the later sliders, spread around 1.0 like the first ones."""
    for slider in _LATER_SLIDERS:
        value = float(rng.normal(1.0, _SLIDER_SIGMA))
        setattr(tactics, slider, min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def _round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Circle method: n-1 rounds in which every team plays exactly once.

    Indices are 0..n-1 into the team list. Requires n even, which TEAM_COUNT is.
    """
    if n % 2 != 0:
        raise ValueError("round robin needs an even number of teams")
    order = list(range(n))
    rounds: list[list[tuple[int, int]]] = []
    for _ in range(n - 1):
        pairs = [
            (order[i], order[n - 1 - i]) for i in range(n // 2)
        ]
        rounds.append(pairs)
        # Rotate all but the first.
        order = [order[0], order[-1], *order[1:-1]]
    return rounds


def build_fixtures(team_ids: list[int], rng: np.random.Generator) -> list[Fixture]:
    """Double round-robin: every pair meets twice, once at each ground.

    The second half of the season is the first with venues reversed, which is
    what makes it a fair double round-robin rather than two random halves.
    Home/away within the first leg is alternated per round so no team is handed
    a lopsided share of home games early.
    """
    n = len(team_ids)
    rounds = _round_robin(n)
    fixtures: list[Fixture] = []
    for r, pairs in enumerate(rounds):
        for i, (a, b) in enumerate(pairs):
            # Alternate venue by round and by position, else team 0 is always
            # at home in the first leg.
            home, away = (a, b) if (r + i) % 2 == 0 else (b, a)
            fixtures.append(
                Fixture(
                    matchday=r + 1,
                    home_id=team_ids[home],
                    away_id=team_ids[away],
                )
            )
    # Return leg: same pairings, venues swapped.
    first_leg_rounds = len(rounds)
    for fixture in list(fixtures):
        fixtures.append(
            Fixture(
                matchday=fixture.matchday + first_leg_rounds,
                home_id=fixture.away_id,
                away_id=fixture.home_id,
            )
        )
    return fixtures


@dataclass(slots=True)
class Tournament:
    seed: int
    teams: list[TournamentTeam]
    fixtures: list[Fixture]
    user_team_id: int

    @property
    def matchdays(self) -> int:
        return max((f.matchday for f in self.fixtures), default=0)

    def team(self, team_id: int) -> TournamentTeam:
        for candidate in self.teams:
            if candidate.team_id == team_id:
                return candidate
        raise KeyError(f"no team {team_id}")

    def fixtures_for(self, matchday: int) -> list[Fixture]:
        return [f for f in self.fixtures if f.matchday == matchday]

    def user_fixture(self, matchday: int) -> Fixture | None:
        for fixture in self.fixtures_for(matchday):
            if self.user_team_id in (fixture.home_id, fixture.away_id):
                return fixture
        return None

    def ai_fixtures(self, matchday: int) -> list[Fixture]:
        return [
            f
            for f in self.fixtures_for(matchday)
            if self.user_team_id not in (f.home_id, f.away_id)
        ]

    @property
    def current_matchday(self) -> int:
        """The first matchday with anything left to play, else one past the end."""
        for day in range(1, self.matchdays + 1):
            if any(not f.played for f in self.fixtures_for(day)):
                return day
        return self.matchdays + 1

    @property
    def finished(self) -> bool:
        return all(f.played for f in self.fixtures)

    def table(self) -> list[TableRow]:
        """Standings, sorted the way a league table is: points, GD, goals, name."""
        rows = {t.team_id: TableRow(t.team_id, t.name) for t in self.teams}
        for fixture in self.fixtures:
            if not fixture.played:
                continue
            home, away = rows[fixture.home_id], rows[fixture.away_id]
            home.played += 1
            away.played += 1
            home.goals_for += fixture.home_goals
            home.goals_against += fixture.away_goals
            away.goals_for += fixture.away_goals
            away.goals_against += fixture.home_goals
            if fixture.home_goals > fixture.away_goals:
                home.won += 1
                away.lost += 1
            elif fixture.home_goals < fixture.away_goals:
                away.won += 1
                home.lost += 1
            else:
                home.drawn += 1
                away.drawn += 1
        return sorted(
            rows.values(),
            key=lambda r: (-r.points, -r.goal_difference, -r.goals_for, r.name),
        )

    def match_seed(self, fixture: Fixture) -> int:
        """A stable seed per fixture.

        Derived from the tournament seed and the fixture's identity, so a
        matchday cannot be re-rolled for a better result and a replayed
        tournament reproduces exactly.
        """
        return (
            self.seed * 100_003
            + fixture.matchday * 1_009
            + fixture.home_id * 31
            + fixture.away_id
        ) % (2**31 - 1)

    def build_match(self, fixture: Fixture) -> GameState:
        """A kickoff GameState for this fixture, using both teams' stored tactics."""
        home = self.team(fixture.home_id)
        away = self.team(fixture.away_id)
        state = build_test_state(
            match_id=f"md{fixture.matchday}-{home.name}-v-{away.name}",
            seed=self.match_seed(fixture),
            home_name=home.name,
            away_name=away.name,
            home_tactics=home.tactics,
            away_tactics=away.tactics,
        )
        # Players are built fresh for every fixture, so stored overrides have to
        # be re-applied or they die with the previous match.
        _apply_player_biases(state.home_team, home.preset)
        _apply_player_biases(state.away_team, away.preset)
        return state

    def record(
        self,
        fixture: Fixture,
        home_goals: int,
        away_goals: int,
        stats: MatchStats | None = None,
    ) -> None:
        """Store a finished result, and its stats when the caller has them.

        `stats` is optional so the table-only tests can record a scoreline
        without fabricating a MatchStats, but both real callers -- the headless
        advance and the user's own finished match -- pass it.
        """
        fixture.home_goals = home_goals
        fixture.away_goals = away_goals
        fixture.played = True
        if stats is not None:
            fixture.stats = stats


def _apply_player_biases(team, preset: TacticsPreset) -> None:
    """Write a preset's stored overrides onto a freshly built squad."""
    for slot, player in enumerate(team.players):
        stored = preset.player_biases.get(slot)
        if not stored:
            continue
        for field_name in PLAYER_BIAS_FIELDS:
            if field_name in stored:
                setattr(player, field_name, stored[field_name])
    # A depth or breadth override moves a station, so the shape has to be
    # rebuilt -- otherwise the override is stored but invisible until something
    # else triggers a restation.
    if any(
        f in biases
        for biases in preset.player_biases.values()
        for f in ("bias_offensiveness", "bias_breadth", "bias_compactness")
    ):
        from .formations import apply_formation

        apply_formation(team)
        # This runs before kick-off, so the moved stations are where the
        # players START, exactly as build_test_state places them -- not a
        # spot they jog to after the whistle. The kick-off taker keeps the
        # centre spot.
        for player in team.players:
            if not player.has_ball:
                player.pos = player.assigned_pos.copy()


def capture_player_biases(team) -> dict[int, dict[str, float]]:
    """Read a squad's overrides back into storable form, keyed by slot.

    Only set values are kept: storing the Nones would be indistinguishable from
    an explicit override at the team default, and clearing one would then never
    take effect.
    """
    captured: dict[int, dict[str, float]] = {}
    for slot, player in enumerate(team.players):
        values = {
            name: getattr(player, name)
            for name in PLAYER_BIAS_FIELDS
            if getattr(player, name) is not None
        }
        if values:
            captured[slot] = values
    return captured


def create_tournament(seed: int = 0, user_team_index: int = 0) -> Tournament:
    """Eighteen teams with German city names; every side but the user's gets a
    random tactical identity, kept for the whole season."""
    rng = np.random.default_rng(seed)
    possession_rng = np.random.default_rng([abs(int(seed)), 1])
    style_rng = np.random.default_rng([abs(int(seed)), 2])
    later_rng = np.random.default_rng([abs(int(seed)), 3])
    names = list(CITY_NAMES)
    teams: list[TournamentTeam] = []
    for index, name in enumerate(names):
        is_user = index == user_team_index
        # The user's side starts neutral -- their tactics are theirs to choose,
        # and a randomised starting point would be a hidden handicap.
        tactics = TeamTactics() if is_user else random_tactics(rng)
        if not is_user:
            tactics.possession = random_possession(possession_rng)
            tactics.crossing, tactics.lofted = random_ball_style(style_rng)
            random_later_sliders(tactics, later_rng)
        teams.append(
            TournamentTeam(
                team_id=index + 1,
                name=name,
                is_user=is_user,
                preset=TacticsPreset(name="Default", tactics=tactics),
            )
        )
    fixtures = build_fixtures([t.team_id for t in teams], rng)
    return Tournament(
        seed=seed,
        teams=teams,
        fixtures=fixtures,
        user_team_id=teams[user_team_index].team_id,
    )
