"""The back caller's shape call: where the team defends, as one object.

Players have no shared mind, so one of them says where the team stands, as the
carrier says who joins an attack. He is the most central of the deepest outfield
players, who has the whole attack in front of him.

The call is the team's formation moved and squeezed: how high, how far across,
how tight. Any formation stays itself and no line is assumed, because a station
is only ever the player's slot put through those numbers. Slots level in the
formation stay level, so the shape cannot bend back at the edges.

Made about once a second and heard a moment later (SHAPE_CALL_*), so stations
follow a plan rather than every touch, and a quick switch of play can beat it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..constants import (
    BLOCK_ACROSS_NEAR,
    BLOCK_FRONT_RANK,
    BLOCK_LENGTH_MIN,
    BLOCK_LINE_MIN,
    BLOCK_SQUEEZE_FAR,
    BLOCK_WIDTH_MIN,
    COMPACT_SLIDE_GAIN,
    COMPACT_SLIDE_NEUTRAL,
    DANGER_FAR,
    GOAL_CENTER_Y,
    LEFT_GOAL_X,
    LINE_FADE_FAR,
    LINE_SHARE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    RECOVER_DEPTH,
    RIGHT_GOAL_X,
    SHAPE_BLOCK_SLIDE_X,
    SHAPE_BLOCK_SLIDE_Y,
    SHAPE_CALL_BALL_M,
    SHAPE_CALL_EASE_TICKS,
    SHAPE_CALL_HEAR_TICKS,
    SHAPE_CALL_MIN_TICKS,
    SHAPE_CALL_OUTLET_KEEP_M,
    SHAPE_CALL_TICKS,
)

if TYPE_CHECKING:
    from ..models import GameState, Player, Team

# A/B switch (on); 0 restores the stations built from the slide, the goal
# funnel, the recovery run and the distance from goal.
SHAPE_CALL = os.environ.get("FB_SHAPE_CALL", "1") != "0"


@dataclass(slots=True)
class ShapeCall:
    caller_id: int
    made_tick: int
    heard_tick: int
    ball: np.ndarray       # where the ball was when he called
    line: float            # m from our goal line: where the deepest slot stands
    back: float            # the deepest slot's own depth in the formation
    length: float          # share of the formation's length kept
    across: float          # m the shape moves across, towards the ball
    width: float           # share of the formation's width kept
    centre_y: float        # the middle of the formation's width
    outlet_id: int | None  # the man left up for the counter, on his full length
    # The call it replaced: the shape glides from that one over
    # SHAPE_CALL_EASE_TICKS, or every station would step on each call.
    previous: "ShapeCall | None" = None


def _depth(point: np.ndarray, team: Team) -> float:
    """Metres from our own goal line."""
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    return (float(point[0]) - own_goal_x) * team.direction


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _outlet(outfield: list[Player], team: Team, previous: ShapeCall | None) -> int | None:
    """The highest man, kept while he is near enough: handing it between two
    level forwards moves both of them most of the team's length."""
    top = max(outfield, key=lambda p: _depth(p.pos, team))
    if previous is not None:
        for p in outfield:
            if p.player_id == previous.outlet_id and (
                _depth(p.pos, team) >= _depth(top.pos, team) - SHAPE_CALL_OUTLET_KEEP_M
            ):
                return p.player_id
    return top.player_id


def make_call(state: GameState, team: Team, previous: ShapeCall | None = None) -> ShapeCall:
    """What the back caller calls now, from where the ball is."""
    outfield = [p for p in team.players if p.position != "GK"]
    caller = min(
        outfield,
        key=lambda p: _depth(p.pos, team) + 0.5 * abs(float(p.pos[1]) - GOAL_CENTER_Y),
    )
    ball = state.ball.pos
    ball_depth = _depth(ball, team)
    slots = sorted(_depth(p.assigned_pos, team) for p in outfield)
    back, front = slots[0], slots[min(BLOCK_FRONT_RANK, len(slots) - 1)]

    # How high: the formation slid with the ball, and from DANGER_FAR in no
    # nearer our goal than LINE_SHARE of the ball's depth.
    follow = COMPACT_SLIDE_NEUTRAL * team.tactics.compactness ** COMPACT_SLIDE_GAIN
    slid = back + (ball_depth - PITCH_LENGTH / 2.0) * min(0.95, SHAPE_BLOCK_SLIDE_X * follow)
    weight = _clip01((LINE_FADE_FAR - ball_depth) / (LINE_FADE_FAR - DANGER_FAR))
    floor = LINE_SHARE * ball_depth
    line = max(BLOCK_LINE_MIN, slid + weight * max(0.0, floor - slid))

    # How long: the men behind the front rank fit in behind the ball.
    room = ball_depth - RECOVER_DEPTH - line
    length = min(1.0, max(BLOCK_LENGTH_MIN, room / max(front - back, 1e-6)))

    # How far across, and how tight: towards the ball, but less so near our
    # goal, where the middle of the shape stays in front of it.
    offset = float(ball[1]) - GOAL_CENTER_Y
    near = BLOCK_ACROSS_NEAR + (1.0 - BLOCK_ACROSS_NEAR) * _clip01(ball_depth / DANGER_FAR)
    across = offset * min(0.95, SHAPE_BLOCK_SLIDE_Y * follow) * near
    squeeze = _clip01((BLOCK_SQUEEZE_FAR - ball_depth) / (BLOCK_SQUEEZE_FAR - DANGER_FAR))
    width = 1.0 - (1.0 - BLOCK_WIDTH_MIN) * squeeze

    tick = state.tick
    return ShapeCall(
        caller_id=caller.player_id,
        made_tick=tick,
        heard_tick=tick + SHAPE_CALL_HEAR_TICKS,
        ball=ball.copy(),
        line=line,
        back=back,
        length=length,
        across=across,
        width=width,
        centre_y=sum(float(p.assigned_pos[1]) for p in outfield) / len(outfield),
        outlet_id=_outlet(outfield, team, previous),
    )


def called_station(player: Player, call: ShapeCall, team: Team, tick: int) -> np.ndarray:
    """His slot, put through the call, part way from the last one while the
    team moves to it."""
    station = _station(player, call, team)
    if call.previous is not None:
        done = (tick - call.heard_tick) / SHAPE_CALL_EASE_TICKS
        if done < 1.0:
            before = _station(player, call.previous, team)
            return before + (station - before) * max(0.0, done)
    return station


def _station(player: Player, call: ShapeCall, team: Team) -> np.ndarray:
    length = 1.0 if player.player_id == call.outlet_id else call.length
    depth = call.line + (_depth(player.assigned_pos, team) - call.back) * length
    own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
    y = call.centre_y + call.across + (float(player.assigned_pos[1]) - call.centre_y) * call.width
    return np.array([
        min(PITCH_LENGTH, max(0.0, own_goal_x + team.direction * depth)),
        min(PITCH_WIDTH, max(0.0, y)),
    ])


def update_shape_calls(state: GameState) -> None:
    """Hear the call that has arrived, and make a new one if it is time.

    In APPLY, like the carrier's calls, so decisions read it from the next
    tick on.
    """
    if not SHAPE_CALL:
        return
    for team in (state.home_team, state.away_team):
        tid = team.team_id
        waiting = state.shape_calls_waiting.get(tid)
        if waiting is not None and state.tick >= waiting.heard_tick:
            waiting.previous = state.shape_calls.get(tid)
            if waiting.previous is not None:
                waiting.previous.previous = None
            state.shape_calls[tid] = waiting
            del state.shape_calls_waiting[tid]
            waiting = None
        current = state.shape_calls.get(tid)
        if current is None:
            state.shape_calls[tid] = make_call(state, team)
            continue
        if waiting is not None:
            continue
        since = state.tick - current.made_tick
        moved = float(np.hypot(*(state.ball.pos - current.ball)))
        if since >= SHAPE_CALL_TICKS or (
            since >= SHAPE_CALL_MIN_TICKS and moved >= SHAPE_CALL_BALL_M
        ):
            state.shape_calls_waiting[tid] = make_call(state, team, current)
