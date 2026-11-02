"""
Builds GameState objects from structured data.

build_test_state() creates a minimal 11v11 game for engine verification.
build_game_state() will be used by the API once the DB layer is wired up.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import (
    PITCH_CENTER,
    PLAYER_BASE_SPEED,
    PLAYER_SPEED_SCALE,
)
from .formations import (
    DEFAULT_FORMATION,
    FORMATIONS,
    SQUAD_SIZE,
    apply_formation,
    slot_position,
)
from .models import Ball, GameState, Player, Team, TeamTactics

def _player_speed(speed_quality: float) -> float:
    return PLAYER_BASE_SPEED + (speed_quality - 50.0) * PLAYER_SPEED_SCALE


def _make_player(
    player_id: int,
    team_id: int,
    name: str,
    position: str,
    direction: int,
    slot_x: float,
    slot_y: float,
    speed_q: float = 70.0,
    strength_q: float = 70.0,
    intelligence_q: float = 70.0,
    passing_q: float = 70.0,
    goalkeeping_q: float = 40.0,
    shooting_q: float = 70.0,
    dribbling_q: float = 70.0,
) -> Player:
    pos = slot_position(slot_x, slot_y, direction)
    return Player(
        player_id=player_id,
        team_id=team_id,
        name=name,
        position=position,  # type: ignore[arg-type]
        tactical="normal",
        pos=pos.copy(),
        velocity=np.zeros(2),
        speed=_player_speed(speed_q),
        strength=strength_q,
        intelligence=intelligence_q,
        passing=passing_q,
        goalkeeping=goalkeeping_q,
        assigned_pos=pos.copy(),
        marking_target_id=None,
        has_ball=False,
        runway_signal=None,
        shooting=shooting_q,
        dribbling=dribbling_q,
    )


def build_test_state(
    match_id: str = "test-match",
    seed: int = 0,
    kickoff_team: int | None = None,
    formation: str = DEFAULT_FORMATION,
    home_name: str = "Home FC",
    away_name: str = "Away FC",
    home_tactics: "TeamTactics | None" = None,
    away_tactics: "TeamTactics | None" = None,
) -> GameState:
    """
    Creates a kickoff GameState with two generic teams.
    Home team attacks rightward (+1), away team attacks leftward (-1).

    `kickoff_team` selects who starts with the ball (1 = home, 2 = away).
    When omitted it is drawn from the seeded RNG, so neither side gets a
    systematic first-possession advantage across a run of matches.
    """
    rng = np.random.default_rng(seed)
    if kickoff_team is None:
        kickoff_team = 1 if rng.random() < 0.5 else 2
    # Each side is built in ITS OWN formation. Building both from the one
    # `formation` argument and only storing the team's real shape in its
    # tactics meant every first half was played in that default shape -- a
    # 3-5-2 kicked off as a 4-4-2, with 4-4-2 roles -- until the half-time
    # restation finally applied it. An explicit formation argument only wins
    # when no tactics were supplied, or it would silently override the stored
    # shape.
    home_formation = home_tactics.formation if home_tactics is not None else formation
    away_formation = away_tactics.formation if away_tactics is not None else formation

    def build_side(prefix: str, team_id: int, direction: int, shape: str, first_id: int):
        return [
            # Ids are 1..11 (home) and 12..22 (away): disjoint, and no gap.
            _make_player(
                player_id=first_id + i,
                team_id=team_id,
                name=f"{prefix} {pos}{i + 1}",
                position=pos,
                direction=direction,
                slot_x=sx,
                slot_y=sy,
            )
            for i, (pos, sx, sy) in enumerate(FORMATIONS[shape])
        ]

    home_players = build_side("Home", 1, 1, home_formation, 1)
    away_players = build_side("Away", 2, -1, away_formation, 1 + SQUAD_SIZE)

    home_team = Team(team_id=1, name=home_name, players=home_players, direction=1)
    away_team = Team(team_id=2, name=away_name, players=away_players, direction=-1)
    # Tournament teams carry their tactics between matches, so a caller may hand
    # in a stored set. Copied, never aliased: the match must not write back into
    # the tournament's record except where we explicitly choose to.
    if home_tactics is not None:
        home_team.tactics = replace(home_tactics)
    if away_tactics is not None:
        away_team.tactics = replace(away_tactics)
    home_team.tactics.formation = home_formation
    away_team.tactics.formation = away_formation
    # Station both sides from their tactics -- depth, breadth and compactness
    # too, which the raw formation slots ignore -- and start everyone ON that
    # station, as every later kick-off does (physics._restation_team).
    for team in (home_team, away_team):
        apply_formation(team)
        for p in team.players:
            p.pos = p.assigned_pos.copy()

    # Ball at centre; home CF has kick-off
    ball = Ball(
        pos=PITCH_CENTER.copy(),
        velocity=np.zeros(2),
        state="stationary",
        owner_id=None,
        last_toucher_team_id=None,
    )

    # Which side kicks off is decided by the caller, not fixed to home. Home
    # taking every kick-off was worth roughly a 6-to-1 goal advantage across
    # seeds, since the first possession often runs straight into a chance.
    kickoff_players = home_players if kickoff_team == 1 else away_players
    # Fall back to the most advanced outfielder rather than requiring every
    # formation to contain a literal "CF" -- a bare next() would raise
    # StopIteration and take the whole match down.
    direction = 1 if kickoff_team == 1 else -1
    kickoff_taker = next(
        (p for p in kickoff_players if p.position == "CF"),
        max(
            (p for p in kickoff_players if p.position != "GK"),
            key=lambda p: p.assigned_pos[0] * direction,
        ),
    )
    kickoff_taker.has_ball = True
    kickoff_taker.pos = PITCH_CENTER.copy()
    # NOTE: assigned_pos is deliberately NOT moved to the centre spot. It is the
    # player's formation home, and the off-ball shape leash is anchored to it --
    # overwriting it here pinned the striker to the halfway line for the entire
    # match, so no attacker ever entered the box.
    ball.owner_id = kickoff_taker.player_id

    return GameState(
        tick=0,
        rng=rng,
        ball=ball,
        home_team=home_team,
        away_team=away_team,
        match_id=match_id,
        half=1,
        first_kickoff_team_id=kickoff_taker.team_id,
    )
