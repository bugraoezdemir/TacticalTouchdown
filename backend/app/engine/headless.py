"""
Headless match simulation, run in parallel across processes.

A full match is 10,000 ticks. A matchday holds eight AI fixtures, so running
them in separate PROCESSES turns eight match-times into a few. Threads would not: the tick loop is pure Python and
holds the GIL throughout, so it does not overlap.

Only primitives cross the process boundary. A GameState carries a numpy
Generator, per-player caches and back-references; pickling one is both slow and
a good way to end up with two divergent copies. So a worker is handed the
fixture's inputs, builds its own state, and returns the score.
"""
from __future__ import annotations

import multiprocessing
from collections.abc import Iterator, MutableMapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace

from .factory import build_test_state
from .game_loop import tick
from .manager import MatchManager, WindowRecord
from .models import MatchStats, TeamTactics


@dataclass(frozen=True, slots=True)
class MatchSpec:
    """Everything a worker needs to replay one fixture, all picklable."""

    match_id: str
    seed: int
    home_name: str
    away_name: str
    home_tactics: TeamTactics
    away_tactics: TeamTactics
    # Identifies the fixture to the caller; the engine never reads these.
    matchday: int
    home_id: int
    away_id: int
    # The season's stored per-player overrides, by squad slot. Rules mutate a
    # copy, and the caller reads the result back off MatchResult to persist it.
    # Defaulted, so they must follow every required field.
    home_player_biases: dict = field(default_factory=dict)
    away_player_biases: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchResult:
    matchday: int
    home_id: int
    away_id: int
    home_goals: int
    away_goals: int
    ticks: int
    # The full counting stats for both sides. Carried back across the process
    # boundary because the worker's GameState dies with it -- without this the
    # only surviving record of a headless match is its scoreline.
    stats: MatchStats
    # Per-window reward log. The evidence for whether reactive tactics help,
    # and the training data a learned policy would need. Empty when the match
    # ran without managers.
    windows: tuple[WindowRecord, ...] = ()
    # Per-player overrides as they stood at full time, by squad slot. The
    # caller writes these back to the season record.
    home_player_biases: dict = field(default_factory=dict)
    away_player_biases: dict = field(default_factory=dict)


def simulate(
    spec: MatchSpec,
    max_ticks: int = 200_000,
    live: "MutableMapping | None" = None,
    reactive: bool = True,
) -> MatchResult:
    """Play one match to full time and return the score.

    `max_ticks` is a guard, not a schedule: `tick` becomes a no-op once the
    state is finished, so a bug that never sets `finished` would otherwise spin
    forever inside a worker process where it is invisible.

    `live` is an optional shared mapping (a multiprocessing Manager dict) into
    which the running score is published as goals go in. Without it a headless
    fixture shows nothing at all until full time, which for a ~3 minute match
    means three minutes of a blank scoreline. Written only when the score
    CHANGES -- a cross-process write every tick would cost more than the
    simulation.
    """
    state = build_test_state(
        match_id=spec.match_id,
        seed=spec.seed,
        home_name=spec.home_name,
        away_name=spec.away_name,
        home_tactics=replace(spec.home_tactics),
        away_tactics=replace(spec.away_tactics),
    )
    # Both sides get a manager: an AI opponent should react whether or not the
    # other side is the user. Scoring windows happens regardless, so the reward
    # log exists even with reactions disabled -- that is the control arm.
    manager = MatchManager(
        state,
        home_enabled=reactive,
        away_enabled=reactive,
        # Rule-managed only when reactive: a non-reactive control arm must not
        # quietly rewrite the season's overrides.
        home_player_biases=dict(spec.home_player_biases) if reactive else None,
        away_player_biases=dict(spec.away_player_biases) if reactive else None,
    )

    ticks = 0
    last = (-1, -1)
    key = (spec.home_id, spec.away_id)
    while not state.finished and ticks < max_ticks:
        tick(state)
        ticks += 1
        manager.observe(state)
        manager.maybe_adjust(state)
        if live is not None:
            score = (state.home_score, state.away_score)
            if score != last:
                last = score
                live[key] = (score[0], score[1], state.tick)
    return MatchResult(
        matchday=spec.matchday,
        home_id=spec.home_id,
        away_id=spec.away_id,
        home_goals=state.home_score,
        away_goals=state.away_score,
        ticks=ticks,
        stats=state.stats,
        windows=tuple(manager.log),
        home_player_biases=manager.home.player_biases or {},
        away_player_biases=manager.away.player_biases or {},
    )


def simulate_many(
    specs: list[MatchSpec],
    max_workers: int | None = None,
    reactive: bool = True,
) -> list[MatchResult]:
    """Run these fixtures in parallel and return every result."""
    return list(iter_simulated(specs, max_workers, reactive=reactive))


def iter_simulated(
    specs: list[MatchSpec],
    max_workers: int | None = None,
    live: "MutableMapping | None" = None,
    reactive: bool = True,
) -> Iterator[MatchResult]:
    """Yield each fixture's result AS IT FINISHES.

    `pool.map` preserves submission order, so a caller consuming it cannot see
    the second match until the first is done -- and `list(pool.map(...))` drains
    the whole iterator before returning anything at all. Either way a progress
    counter sits at zero for the entire matchday and then jumps to complete,
    which is exactly what it looked like.

    `as_completed` reports in finishing order instead, so progress moves as each
    match lands. The caller is responsible for matching a result back to its
    fixture (by team ids), since order is no longer positional.

    Falls back to running in-process for a single fixture: spawning a pool costs
    more than the match saves, and it keeps tests and debugging simple.
    """
    if not specs:
        return
    if len(specs) == 1:
        yield simulate(specs[0], live=live, reactive=reactive)
        return
    workers = max_workers or len(specs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(simulate, spec, 200_000, live, reactive) for spec in specs
        ]
        for future in as_completed(futures):
            yield future.result()


def live_score_map() -> tuple["multiprocessing.managers.SyncManager", MutableMapping]:
    """A manager and its dict, for publishing running scores out of workers.

    The caller owns the manager and must shut it down; it holds a child
    process. Returned as a pair rather than hidden inside iter_simulated so the
    reader (an HTTP handler on another thread) can hold the dict while the
    simulation runs.
    """
    manager = multiprocessing.Manager()
    return manager, manager.dict()
