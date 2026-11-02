"""
One tick's shared answers, worked out once.

Every decision in a tick reads the same frozen snapshot (game_loop's two-phase
rule), so anything that depends only on that snapshot -- a defender's zone
centre, the team's marking assignment, the ball's path -- is the same for
every player who asks. Asked per player it was recomputed up to 80 times a
tick. game_loop opens an epoch for the decision phase and closes it before
anything moves; outside an epoch (tests, probes, the apply phase) get() returns
None and callers compute directly, so a stale answer can never leak.
"""
from __future__ import annotations

import itertools

_COUNTER = itertools.count()
_EPOCH: list = [None]
_DATA: dict = {}


def open_epoch() -> int:
    """Start a decision phase: a fresh, empty cache."""
    _EPOCH[0] = next(_COUNTER)
    _DATA.clear()
    return _EPOCH[0]


def close_epoch() -> None:
    _EPOCH[0] = None
    _DATA.clear()


def get() -> dict | None:
    """The current decision phase's cache, or None outside one."""
    return _DATA if _EPOCH[0] is not None else None
