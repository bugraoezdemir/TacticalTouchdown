"""
Per-player tactical rules: a footballer who notices what is going wrong.

The preset system reads only the scoreboard, so a player whose passes are all
being cut out plays exactly like one completing everything. These rules read
the outcomes of a player's OWN actions and adjust the parameter responsible --
a closed feedback loop rather than a scripted reaction to the score.

Applies to AI teams only; the user designs their own tactics. Changes persist
across the season, so a season is a development arc.

THRESHOLDS ARE MEASURED, NOT GUESSED. Calibrated from 120 outfield
player-matches:

    rate            p10   p25   med   p75   p90   max
    interception   0.11  0.15  0.19  0.22  0.26  0.42
    dribble loss   0.12  0.15  0.20  0.28  0.32  0.58

    events/match    p25   med   p75
    passes           33    50    76
    dribbles         17    53   102
    shots             0     0     1

Two consequences that shaped the design:

  * Pass and dribble rules can act WITHIN a match (medians of 50 and 53 events).
  * A shot rule cannot. Only 35 of 120 player-matches contain even one shot,
    median 2 among those, and conversion reads 0.00 at the median with 0.50 at
    p75 -- that is "missed both" versus "scored one of two", which is noise at
    n=2. So the shoot rule accumulates across the SEASON and needs 15 shots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .models import PlayerStats

# p75 of the measured distribution: fires on roughly the worst quarter of
# qualifying players. p90 was the alternative -- fewer, more clear-cut cases --
# but the user chose p75 for a more visibly reactive league.
INTERCEPTION_THRESHOLD = 0.22
DRIBBLE_LOSS_THRESHOLD = 0.28
SHOT_CONVERSION_THRESHOLD = 0.10

# Minimum events before a rule may fire. Below these, a "bad" rate is luck:
# 30 passes is under the p25 of 33, so a player must be genuinely involved.
# Minimums for a RECENT slice. A busy player makes ~50 passes and ~53 dribbles
# a match, so 12 of either is a few windows' worth -- enough to mean something,
# small enough that a rule can still act within a match.
MIN_PASSES = 12
MIN_DRIBBLES = 12
# Shots stay SEASON-long: only 35 of 120 player-matches contain even one shot,
# median 2. There is no recent slice that could ever carry 15 shots.
MIN_SHOTS = 15

# Small and reversible. At p75 a quarter of qualifying players adjust each
# window, so a large step would swing rather than steer -- and if a rule is
# wrong, decay pulls it back rather than compounding the error.
# Measured: with cumulative counters a triggered player fired EVERY window --
# 207 firings across 18 windows, the busiest man 64 times, which at the old
# 0.04 step is 2.56 of movement on a range only 0.5 wide. Five times the span.
#
# Two corrections. Rules now read a RECENT slice (see `since`), so a player who
# improves actually clears the trigger and decays home; and the step is sized
# so a full match of continuous triggering moves a slider by at most ~0.2,
# leaving a season to reach the clamp rather than a single game.
STEP = 0.012
DECAY = 0.006           # per evaluation, back toward the player's base value

BIAS_MIN, BIAS_MAX = 0.5, 2.0


def since(now: PlayerStats, earlier: PlayerStats) -> PlayerStats:
    """The stats accrued between two snapshots.

    This is what makes an adjustment re-earnable. Judging a player on his
    season-to-date rate means that once he crosses a threshold he can never
    uncross it -- each new window is a vanishing fraction of a growing total --
    so the trigger stays on, decay never applies, and the value walks to the
    clamp and stays pinned. Measured: three players hit the floor in one match.
    """
    return PlayerStats(
        passes=now.passes - earlier.passes,
        passes_completed=now.passes_completed - earlier.passes_completed,
        passes_intercepted=now.passes_intercepted - earlier.passes_intercepted,
        dribbles=now.dribbles - earlier.dribbles,
        dribbles_lost=now.dribbles_lost - earlier.dribbles_lost,
        shots=now.shots - earlier.shots,
        shots_on_target=now.shots_on_target - earlier.shots_on_target,
        goals=now.goals - earlier.goals,
        touches=now.touches - earlier.touches,
    )


@dataclass(frozen=True, slots=True)
class Rule:
    """One if-then: a trigger on a player's own numbers, and what to nudge.

    `adjust` maps bias field -> direction (+1 raises, -1 lowers). Magnitudes
    come from STEP so every rule moves at the same measured pace; a rule that
    wanted its own magnitude would be a tuning knob nobody could calibrate.
    """

    name: str
    why: str
    qualifies: Callable[[PlayerStats], bool]
    triggers: Callable[[PlayerStats], bool]
    adjust: dict[str, float]


RULES: tuple[Rule, ...] = (
    Rule(
        name="passes_cut_out",
        why="his passes keep being intercepted",
        qualifies=lambda s: s.passes >= MIN_PASSES,
        triggers=lambda s: s.interception_rate > INTERCEPTION_THRESHOLD,
        # Shorter balls survive better; carrying is the fallback when the
        # passing lanes are not there.
        adjust={"bias_pass_range": -1.0, "bias_dribble": +0.5},
    ),
    Rule(
        name="dribbles_lost",
        why="he keeps losing the ball carrying it",
        qualifies=lambda s: s.dribbles >= MIN_DRIBBLES,
        triggers=lambda s: s.dribble_loss_rate > DRIBBLE_LOSS_THRESHOLD,
        adjust={"bias_dribble": -1.0, "bias_pass": +0.5},
    ),
    Rule(
        name="shots_not_scoring",
        why="his shots are not going in",
        qualifies=lambda s: s.shots >= MIN_SHOTS,
        triggers=lambda s: s.shot_conversion < SHOT_CONVERSION_THRESHOLD,
        # Shoot less, pass more. Deliberately the gentlest consequence: a
        # striker who stops shooting entirely is worse than one who is wasteful,
        # and DECAY is what stops "shoot less -> fewer shots -> even less
        # evidence -> shoot less still" becoming a death spiral.
        adjust={"bias_shoot": -1.0, "bias_pass": +0.5},
    ),
)


@dataclass(slots=True)
class RuleFiring:
    """A record of one adjustment, for the UI and for later analysis."""

    slot: int
    rule: str
    why: str
    field: str
    before: float
    after: float


def _clamp(value: float) -> float:
    return min(BIAS_MAX, max(BIAS_MIN, value))


def _current(biases: dict[str, float], field_name: str, base: float) -> float:
    return biases.get(field_name, base)


def evaluate(
    stats: PlayerStats,
    slot: int,
    biases: dict[str, float],
    base: dict[str, float] | None = None,
) -> list[RuleFiring]:
    """Apply every rule to one player, mutating `biases` in place.

    `biases` is the stored per-slot override dict -- the same structure the
    tactics panel writes, so a rule-driven change and a hand-set one are the
    same kind of thing and persist the same way.

    `base` is what the values decay toward: the team default (1.0) unless the
    caller supplies the player's own starting point.
    """
    base = base or {}
    firings: list[RuleFiring] = []
    touched: set[str] = set()

    for rule in RULES:
        if not rule.qualifies(stats) or not rule.triggers(stats):
            continue
        for field_name, direction in rule.adjust.items():
            start = base.get(field_name, 1.0)
            before = _current(biases, field_name, start)
            after = _clamp(before + direction * STEP)
            touched.add(field_name)
            if abs(after - before) < 1e-9:
                continue          # already at the clamp; nothing to record
            biases[field_name] = after
            firings.append(
                RuleFiring(slot, rule.name, rule.why, field_name, before, after)
            )

    # Decay only what the rules did NOT touch this time. Decaying a field a
    # rule just fired on would halve every correction (net STEP - DECAY) and,
    # worse, keep lifting a clamped value off its floor -- an actively failing
    # player would never reach the adjustment his numbers call for.
    _decay(biases, base, skip=touched)
    return firings


def _decay(
    biases: dict[str, float],
    base: dict[str, float],
    skip: set[str] | None = None,
) -> None:
    """Drift overrides back toward base, except those a rule just re-earned.

    Without this, a rule that fired on a bad run would hold its adjustment
    forever even after the player's numbers recovered -- and the shoot rule in
    particular would ratchet a striker out of the game entirely. Decay makes
    every adjustment a claim that has to keep being re-earned.
    """
    skip = skip or set()
    for field_name in list(biases):
        if field_name in skip:
            continue
        start = base.get(field_name, 1.0)
        value = biases[field_name]
        if abs(value - start) <= DECAY:
            # Close enough: snap home and drop the override, so a player who
            # has recovered stops carrying a pointless entry.
            del biases[field_name]
            continue
        biases[field_name] = _clamp(value - DECAY if value > start else value + DECAY)
