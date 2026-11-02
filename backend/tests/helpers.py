"""Shared test helpers.

Tests must not depend on roster list order. With 11 players and selectable
formations, `players[8]` is no longer reliably a forward -- and the failure mode
is silent: the test still passes, it just tests something else.
"""
from __future__ import annotations

from app.engine.models import GameState, Player


def player_at(
    state: GameState,
    position: str,
    team: str = "home",
    nth: int = 0,
) -> Player:
    """The `nth` player in `position` for a side.

    Formations may contain duplicate position strings (a 4-4-2 has two "CD" and
    two "CF"), so `nth` selects between them in roster order.
    """
    squad = state.home_team if team == "home" else state.away_team
    matches = [p for p in squad.players if p.position == position]
    if len(matches) <= nth:
        raise LookupError(
            f"{team} has {len(matches)} player(s) at {position!r}, wanted index {nth}"
        )
    return matches[nth]


def striker(state: GameState, team: str = "home", nth: int = 0) -> Player:
    """A centre-forward -- the usual stand-in for "an attacking player"."""
    return player_at(state, "CF", team=team, nth=nth)


def midfielder(state: GameState, team: str = "home", nth: int = 0) -> Player:
    """A central midfielder -- the usual stand-in for "a supporting player"."""
    return player_at(state, "CM", team=team, nth=nth)
