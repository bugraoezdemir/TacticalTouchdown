"""The offside line.

One primitive, used by three different questions:

  * off-ball positioning -- "am I straying beyond the last defender?"
  * pass selection       -- "would this ball put my man offside?"
  * the referee          -- "was he offside when it was played?"

The engine previously had only `_onside_factor`, a flat 0.05 discount on any
candidate point past the second-deepest opponent. That suppressed goal-hanging,
which is all it was for, but it is not the rule: it ignored the ball, ignored
the halfway line, and nothing was ever penalised, so a forward could simply
stand in an offside position and be passed to.

The real test, simplified to what this engine can represent (no active/passive
judgement, no daylight between body parts):

    A player is in an offside POSITION if, at the moment the ball is played by
    a teammate, he is in the opposition half AND ahead of both the ball and the
    second-last opponent.

Being in an offside position is not itself an offence -- it becomes one only
when the ball is played to him, which is why the check is made at the moment of
the pass and not continuously.
"""
from __future__ import annotations

import numpy as np

from .constants import PITCH_LENGTH
from .models import GameState, Player, Team


def offside_line(state: GameState, team: Team) -> float:
    """The x beyond which `team`'s players are in an offside position.

    The second-last opponent -- normally the last outfield defender, since the
    keeper is usually the deepest, but it is deliberately whoever is actually
    second-deepest so that a keeper caught upfield plays attackers onside.

    Never behind the halfway line: you cannot be offside in your own half.

    Once per team per decision phase (tick_cache): every off-ball candidate
    point asked for it, ~150 times a tick for two distinct answers.
    """
    from .decisions import tick_cache

    cache = tick_cache.get()
    if cache is not None:
        key = ("offside_line", team.team_id)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = _offside_line(state, team)
        return hit
    return _offside_line(state, team)


def _offside_line(state: GameState, team: Team) -> float:
    opponents = state.opponents_of(team.team_id)
    if len(opponents) < 2:
        return PITCH_LENGTH if team.direction == 1 else 0.0

    # Sort by depth from the defenders' own goal, deepest first.
    xs = sorted((float(o.pos[0]) for o in opponents), reverse=team.direction == 1)
    second_last = xs[1]

    half = PITCH_LENGTH / 2.0
    if team.direction == 1:
        return max(second_last, half)
    return min(second_last, half)


def beyond_line(x: float, line: float, direction: int) -> bool:
    """Is `x` past the offside line, for a team attacking in `direction`?"""
    return x > line if direction == 1 else x < line


def is_offside_position(
    player: Player, state: GameState, team: Team, ball_x: float | None = None
) -> bool:
    """Is this player currently in an offside position?

    Ahead of the second-last opponent AND ahead of the ball. The ball test is
    what makes a deep cross-field pass legal: a player level with or behind the
    ball can never be offside however far upfield he is.
    """
    if ball_x is None:
        ball_x = float(state.ball.pos[0])
    line = offside_line(state, team)
    px = float(player.pos[0])
    return beyond_line(px, line, team.direction) and beyond_line(
        px, ball_x, team.direction
    )


def onside_limit(state: GameState, team: Team, ball_x: float | None = None) -> float:
    """The furthest x a player may take up without being offside.

    What the off-ball model steers by: a forward holds this line rather than
    drifting past it. Includes the ball, so a player can always run beyond the
    defence while the ball is ahead of him.
    """
    if ball_x is None:
        ball_x = float(state.ball.pos[0])
    line = offside_line(state, team)
    return max(line, ball_x) if team.direction == 1 else min(line, ball_x)
