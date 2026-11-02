from __future__ import annotations

import numpy as np

from .constants import (
    CORNER_PAUSE_TICKS,
    BOX_CLEARANCE,
    GK_GATHER_SPEED,
    GK_PARRY_RESTITUTION,
    GK_PARRY_SPREAD,
    GK_RERECEIVE_LOCKOUT,
    GK_HOLD_TICKS,
    GOAL_KICK_PAUSE_TICKS,
    GOAL_NET_DEPTH,
    GOAL_PAUSE_TICKS,
    DEAD_BALL_FRICTION,
    DEAD_BALL_LIMIT_M,
    DEAD_BALL_TICKS,
    PENALTY_AREA_BOTTOM,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_TOP,
    KICKOFF_PAUSE_TICKS,
    OFFSIDE_PAUSE_TICKS,
    THROW_IN_PAUSE_TICKS,
    BALL_FRICTION_COEFF,
    BALL_STOP_THRESHOLD,
    BLOCK_LOCKOUT_TICKS,
    BLOCK_MIN_AHEAD,
    BLOCK_RADIUS,
    BLOCK_RESTITUTION,
    BLOCK_SPREAD,
    CAPTURE_RADIUS,
    STRIKE_REACT_TICKS,
    CONTROL_HEIGHT,
    CROSSBAR_HEIGHT,
    GK_REACH_HEIGHT,
    HEADER_HEIGHT,
    RISING_REACH,
    DRIBBLE_FRICTION_COEFF,
    GOAL_BOTTOM_Y,
    GOAL_CENTER_Y,
    GOAL_TOP_Y,
    LEFT_GOAL_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
    RIGHT_GOAL_X,
    SHOT_FRICTION_COEFF,
)
from .ball_flight import airborne, step
from .models import Ball, GameState, Player

BallDelta = tuple[np.ndarray, np.ndarray, str, float, float]


_EPS = 1e-9


def compute_ball_update(ball: Ball) -> BallDelta:
    """Pure computation: returns the next ball state without mutating anything.

    (pos, velocity, state, height, vz).
    """
    if ball.state == "stationary":
        return ball.pos.copy(), ball.velocity.copy(), "stationary", 0.0, 0.0

    friction = {
        "in_shot": SHOT_FRICTION_COEFF,
        "in_dribble": DRIBBLE_FRICTION_COEFF,
    }.get(ball.state, BALL_FRICTION_COEFF)

    if airborne(ball.height, ball.vz):
        # In the air: gravity, no rolling resistance; bounces on landing.
        pos, vel, height, vz = step(ball.pos, ball.velocity, ball.height, ball.vz, friction)
        return pos, vel, ball.state, height, vz

    # Constant deceleration, not proportional decay. A rolling ball loses
    # speed to rolling resistance at a roughly CONSTANT rate; it does not shed
    # a fixed fraction of whatever it currently has. The old `v *= 1 - f` bled
    # speed fastest when the ball was fastest, so flight time exploded with
    # distance -- a 30 m pass took 3.4 s -- and no pair of constants could make
    # short passes firm AND long passes reach: matching one broke the other.
    # `friction` is now metres/tick^2.
    speed = float(np.linalg.norm(ball.velocity))
    if speed > _EPS:
        new_speed = max(0.0, speed - friction)
        new_vel = ball.velocity * (new_speed / speed)
    else:
        new_vel = ball.velocity * 0.0
    new_pos = ball.pos + new_vel
    new_state = ball.state

    if float(np.linalg.norm(new_vel)) < BALL_STOP_THRESHOLD:
        new_vel = np.zeros(2)
        new_state = "stationary"

    return new_pos, new_vel, new_state, 0.0, 0.0


def apply_ball_update(ball: Ball, delta: BallDelta) -> None:
    ball.pos = delta[0]
    ball.velocity = delta[1]
    ball.state = delta[2]
    ball.height = delta[3]
    ball.vz = delta[4]


# Swept ball capture (see resolve_possession). A switch only for A/B
# measurement; on is the correct physics.
SWEPT_CAPTURE = True
# A/B switch for deflecting blocks (on); FB_BLOCK_DEFLECT=0 restores the old
# rule, where the blocker took a struck ball to his feet.
BLOCK_DEFLECT = __import__("os").environ.get("FB_BLOCK_DEFLECT", "1") != "0"


def resolve_possession(state: GameState) -> None:
    """Detect goals, handle out-of-bounds restarts, and transfer possession."""
    ball = state.ball

    # ── Goal detection ───────────────────────────────────────────────────────
    # A goal at x=0 is scored by whoever attacks leftward, and vice versa.
    # Over the bar is not a goal: it is out over the line like any other ball.
    under_bar = ball.height <= CROSSBAR_HEIGHT
    if ball.pos[0] <= LEFT_GOAL_X and GOAL_TOP_Y <= ball.pos[1] <= GOAL_BOTTOM_Y and under_bar:
        _record_goal(state, conceding_goal_x=LEFT_GOAL_X)
        _goal_pause(state)
        return

    if ball.pos[0] >= RIGHT_GOAL_X and GOAL_TOP_Y <= ball.pos[1] <= GOAL_BOTTOM_Y and under_bar:
        _record_goal(state, conceding_goal_x=RIGHT_GOAL_X)
        _goal_pause(state)
        return

    # ── Out-of-bounds ────────────────────────────────────────────────────────
    clamped_x = float(np.clip(ball.pos[0], 0.0, PITCH_LENGTH))
    clamped_y = float(np.clip(ball.pos[1], 0.0, PITCH_WIDTH))
    if clamped_x != ball.pos[0] or clamped_y != ball.pos[1]:
        out_y = float(ball.pos[1])
        # It goes on moving through the beat (step_dead_ball).
        motion = (ball.velocity.copy(), float(ball.height), float(ball.vz))
        rest = ball.pos.copy()
        ball.velocity = np.zeros(2)
        ball.state = "stationary"
        ball.ground()

        # Sideline out (top or bottom of pitch)
        if ball.pos[1] < 0.0 or ball.pos[1] > PITCH_WIDTH:
            _after_beat(state, rest, _throw_in, np.array([clamped_x, clamped_y]))
        # Goal-line out (left or right end)
        elif ball.pos[0] < 0.0 or ball.pos[0] > PITCH_LENGTH:
            _after_beat(state, rest, _goal_line_restart, (ball.pos[0] < 0.0), out_y)
        state.dead_ball_motion = motion
        ball.height, ball.vz = motion[1], motion[2]
        return

    # ── Dribble ownership release ─────────────────────────────────────────────
    # Even the current owner loses the ball if it drifts beyond CAPTURE_RADIUS.
    if ball.owner_id is not None and ball.state == "in_dribble":
        owner = state.get_player(ball.owner_id)
        if owner is not None:
            if float(np.linalg.norm(ball.pos - owner.pos)) > CAPTURE_RADIUS:
                owner.has_ball = False
                ball.owner_id = None   # ball is now loose; last_toucher_team_id already set

    # ── Goalkeeper gathers the ball ───────────────────────────────────────────
    # A keeper catches what he can reach rather than merely touching it like an
    # outfield player. While he holds it the box is cleared and he is left
    # alone to distribute, which is what stops attackers standing on him.
    #
    # Tested against the ball's PATH this tick, not only where it ends up. A
    # shot travels 1.1 m per tick -- further than a capture circle is wide --
    # so testing the end point alone let it pass straight through the keeper
    # between two ticks. He could not stop a shot by physics at all: 47% of all
    # shots scored, and doubling his modelled dive speed changed nothing,
    # because that only fed the SHOOTER's belief, never the pitch.
    speed = float(np.linalg.norm(ball.velocity))
    if ball.owner_id is None:
        prev = ball.pos - ball.velocity      # compute_ball_update moved it by velocity
        for keeper in (
            _find_gk(state, state.home_team.team_id),
            _find_gk(state, state.away_team.team_id),
        ):
            if keeper is None:
                continue
            own_goal_x = LEFT_GOAL_X if keeper.assigned_pos[0] < PITCH_LENGTH / 2.0 else RIGHT_GOAL_X
            if not in_penalty_area(ball.pos, own_goal_x):
                continue
            if ball.height > GK_REACH_HEIGHT:
                continue   # over his hands
            if _segment_gap(keeper.pos, prev, ball.pos) > keeper_reach(keeper, ball.height):
                continue
            # ...and he cannot re-gather the ball he has just distributed. The
            # gather threshold is above a pass's launch speed (it has to be, or
            # he could never catch one), so without this he caught his own
            # distribution on the very next tick while it was still inside his
            # reach: pass, recapture, hold 45 ticks, pass again, forever. From
            # the outside it looked like a keeper who never released the ball.
            if (
                ball.last_passer_id == keeper.player_id
                and state.tick - ball.last_pass_tick <= GK_RERECEIVE_LOCKOUT
            ):
                continue
            if ball.state == "in_shot":
                # He has gathered a struck ball: that is a save, credited to
                # the keeper's own side.
                state.stats.side(state, keeper.team_id).saves += 1
            if speed > GK_GATHER_SPEED:
                # Too hot to hold: he gets a hand to it and PARRIES. Before,
                # anything faster than he could catch simply went through him.
                _parry(state, keeper, own_goal_x)
                return
            _give_possession(state, ball, keeper)
            ball.pos = keeper.pos.copy()
            # Both sides reform, exactly as at a goal kick: the keeper has the
            # ball in his hands and play restarts from him, so leaving the
            # attackers bunched on the edge of his area gave him nothing to aim
            # at but a wall. He keeps his own position -- he is holding it.
            keeper_pos = keeper.pos.copy()
            _restation_all(state)
            keeper.pos = keeper_pos
            ball.pos = keeper_pos.copy()
            state.gk_hold_ticks = GK_HOLD_TICKS
            state.add_event(
                "gk_hold", keeper.team_id, keeper.name,
                f"{keeper.name} gathers it",
            )
            return

    # ── Possession resolution by strength contest ─────────────────────────────
    # Collect all players within CAPTURE_RADIUS ("on-ball" players).
    # Rules (from spec):
    #   1. Sum strength of on-ball players per team.
    #   2. Team with higher total strength wins.
    #   3. Within the winning team the strongest individual holds the ball.
    if ball.owner_id is None:
        # The test is SWEPT, like the keeper's: a pass moves 0.5-0.8 m a tick
        # and a shot 1.1 m, so judging only where the ball ends up let it pass
        # a player 0.45 m off its line -- or a shot straight through a defender
        # -- without one sample inside the 0.51 m circle. Found under inertia
        # (receivers arriving slightly off the line, passes kept 90% -> 65%);
        # under instant movement it is why shots could not be blocked and
        # goals ran at 11-14 a match. Not on a tick the ball was kicked:
        # ball_prev_pos is None then, or the kicker would retake his own ball.
        swept = SWEPT_CAPTURE and state.ball_prev_pos is not None
        on_ball: list[tuple[Player, float]] = []   # (player, dist)
        # Over everyone's head: nobody can touch it this tick.
        reachable = ball.height <= (RISING_REACH if ball.vz > 0.0 else HEADER_HEIGHT)
        struck = (ball.struck_from is not None
                  and state.tick <= ball.struck_tick + STRIKE_REACT_TICKS)
        # A shot or pass meeting a body deflects off it; a dribble touch is
        # simply taken by the man it runs into.
        deflects = BLOCK_DEFLECT and ball.state in ("in_shot", "in_pass")
        blockers: list[Player] = []
        for player in (state.all_players() if reachable else ()):
            if ball.no_capture_id == player.player_id and state.tick <= ball.no_capture_until:
                continue   # he has just kicked or miscontrolled it
            if struck and player.team_id != ball.last_toucher_team_id:
                # Just struck past him: no time to react, only a block.
                if _blocks(player, ball, state.ball_prev_pos):
                    if deflects:
                        blockers.append(player)
                    else:
                        on_ball.append((player, 0.0))
                continue
            if struck and player.player_id != ball.intended_receiver_id:
                # A teammate it was not played to lets it run: taking it at
                # once, from a ball struck right beside him, had two of them
                # passing it between them every tick.
                continue
            if swept:
                d = _segment_gap(player.pos, state.ball_prev_pos, ball.pos)
            else:
                d = float(np.linalg.norm(ball.pos - player.pos))
            # His real reach -- the radius every decision plans with
            # (interception.control_radius). Capturing only within 0.51 m while
            # decisions assumed 1.0 m had players move to exactly where they
            # planned to take a ball and watch it roll past them.
            reach = keeper_reach(player, ball.height) if player.position == "GK" else control_radius_of(player)
            if d < reach:
                on_ball.append((player, d))

        if blockers:
            # It meets the first body along its path, and nobody owns it.
            start = ball.struck_from if state.ball_prev_pos is None else state.ball_prev_pos
            first = min(blockers, key=lambda p: float(np.dot(p.pos - start, ball.velocity)))
            _deflect(state, first, start)
            return

        if on_ball:
            # Group by team
            team_strength: dict[int, float] = {}
            team_players: dict[int, list[Player]] = {}
            for p, _ in on_ball:
                team_strength[p.team_id] = team_strength.get(p.team_id, 0.0) + p.strength
                team_players.setdefault(p.team_id, []).append(p)

            # Ties must not favour a side. max() returns the first key on a
            # tie and insertion order is home-then-away, so home silently won
            # every 50/50 contest. Break ties with the seeded RNG instead.
            best = max(team_strength.values())
            tied = [tid for tid, v in team_strength.items() if v >= best - 1e-9]
            if len(tied) == 1:
                winning_team_id = tied[0]
            else:
                winning_team_id = tied[int(state.rng.integers(len(tied)))]
            # Strongest player on winning team takes possession
            winner = max(team_players[winning_team_id], key=lambda p: p.strength)
            # A pass is completed when a TEAMMATE of the passer collects it --
            # not merely when the ball arrives, since an opponent may take it.
            if (
                ball.state == "in_pass"
                and ball.last_passer_id is not None
                and winner.player_id != ball.last_passer_id
            ):
                try:
                    passer = state.get_player(ball.last_passer_id)
                except KeyError:
                    passer = None
                if passer is not None:
                    if winner.team_id == passer.team_id:
                        # Reached a teammate: completed, for team and passer.
                        state.stats.side(state, winner.team_id).passes_completed += 1
                        state.stats.player(state, passer).passes_completed += 1
                        if ball.is_cross:
                            state.stats.side(state, winner.team_id).crosses_completed += 1
                    else:
                        # Cut out. Charged to the PASSER -- the rules ask "are my
                        # passes being intercepted", which is a question about
                        # his choice of ball, not about the interceptor.
                        state.stats.player(state, passer).passes_intercepted += 1
            if (
                ball.state == "in_dribble"
                and ball.last_dribbler_id is not None
                and winner.player_id != ball.last_dribbler_id
            ):
                try:
                    carrier = state.get_player(ball.last_dribbler_id)
                except KeyError:
                    carrier = None
                if carrier is not None and winner.team_id != carrier.team_id:
                    state.stats.player(state, carrier).dribbles_lost += 1
                ball.last_dribbler_id = None
            state.stats.player(state, winner).touches += 1
            # Offside: flagged when the ball was PLAYED, penalised only now
            # that he has actually received it. Standing in an offside position
            # is not an offence -- being involved in play from one is.
            if winner.player_id in ball.offside_suspects:
                _after_beat(state, ball.pos.copy(), _offside_free_kick, winner)
                return
            # How fast it reached him, relative to his own run -- before the
            # capture stops it. His own dribble touch is not a reception.
            incoming = ball.velocity.copy()
            height, vz = ball.height, ball.vz
            own_touch = ball.state == "in_dribble" and ball.last_dribbler_id == winner.player_id
            _give_possession(state, ball, winner)
            # He reaches out and takes it to his feet. Left where it was caught
            # -- up to his full reach, a metre away -- his next touch or pass
            # left from there, and the ball turned without ever reaching him.
            ball.pos = winner.pos.copy()
            if winner.position != "GK" and not own_touch:
                from .decisions.control import easy_speed, relative_speed

                # A dropping ball comes at him from above too.
                rel = float(np.hypot(relative_speed(incoming, winner.velocity), vz))
                header = height > CONTROL_HEIGHT
                if rel > easy_speed(winner) or header or airborne(height, vz):
                    # Too quick to just trap, or still in the air: his next
                    # decision is a first touch -- play it on, or try to
                    # control it (not possible for a header).
                    winner.first_touch_speed = max(rel, 1e-6)
                    winner.first_touch_velocity = incoming
                    winner.first_touch_height = max(height, 1e-3) if airborne(height, vz) else 0.0


def _blocks(player: Player, ball: Ball, prev: np.ndarray | None) -> bool:
    """A ball struck moments ago meets him: on his side of the kicker's foot
    and within BLOCK_RADIUS of its path this tick (from the foot on the tick
    it was struck)."""
    start = ball.struck_from if prev is None else prev
    path = ball.pos - ball.struck_from
    n = float(np.linalg.norm(path))
    if n < 1e-9:
        return False
    if float(np.dot(player.pos - ball.struck_from, path)) / n < BLOCK_MIN_AHEAD:
        return False
    return _segment_gap(player.pos, start, ball.pos) < BLOCK_RADIUS


def _deflect(state: GameState, blocker: Player, start: np.ndarray) -> None:
    """A struck ball comes off a body: reflected off the side it hit, turned
    by up to BLOCK_SPREAD, at BLOCK_RESTITUTION of its pace -- loose, like a
    parry, so the kicker and the blocker both have to go and win it.

    The side it hit is where its path from `start` first reaches him
    (BLOCK_RADIUS): its position at the end of the tick is already past him."""
    ball = state.ball
    speed = float(np.linalg.norm(ball.velocity))
    if speed < _EPS:
        return
    v = ball.velocity / speed
    d = start - blocker.pos
    half_b = float(np.dot(d, v))
    c = float(np.dot(d, d)) - BLOCK_RADIUS ** 2
    disc = half_b * half_b - c
    if c <= 0.0:                            # struck from inside his reach: straight back
        n = -v
    elif disc < 0.0:                        # only grazed him
        side = d - half_b * v
        ns = float(np.linalg.norm(side))
        n = -v if ns < _EPS else side / ns
    else:
        n = (d + v * (-half_b - np.sqrt(disc))) / BLOCK_RADIUS
    if float(np.dot(v, n)) < 0.0:           # coming into him: bounces off
        v = v - 2.0 * float(np.dot(v, n)) * n
    angle = float(state.rng.uniform(-BLOCK_SPREAD, BLOCK_SPREAD))
    c, s = np.cos(angle), np.sin(angle)
    v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    ball.velocity = v * speed * BLOCK_RESTITUTION
    ball.pos = blocker.pos + v * BLOCK_RADIUS
    ball.ground()
    ball.state = "in_pass"
    ball.is_cross = False
    ball.last_toucher_team_id = blocker.team_id
    ball.last_shooter_id = None
    ball.last_passer_id = None
    ball.intended_receiver_id = None
    ball.offside_suspects = frozenset()
    ball.no_capture_id = blocker.player_id
    ball.no_capture_until = state.tick + BLOCK_LOCKOUT_TICKS
    state.ball_prev_pos = None


def _segment_gap(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Closest distance from `point` to the segment a->b."""
    ab = b - a
    n2 = float(np.dot(ab, ab))
    if n2 < 1e-12:
        return float(np.linalg.norm(point - a))
    t = min(1.0, max(0.0, float(np.dot(point - a, ab)) / n2))
    return float(np.linalg.norm(point - (a + ab * t)))


def _parry(state: GameState, keeper: Player, own_goal_x: float) -> None:
    """Knock a shot away: back out, sideways, and much slower.

    The ball stays loose -- a parried shot is a rebound anyone can reach, which
    is what makes a save a save rather than a turnover.
    """
    ball = state.ball
    away = 1.0 if own_goal_x < PITCH_LENGTH / 2.0 else -1.0
    speed = float(np.linalg.norm(ball.velocity)) * GK_PARRY_RESTITUTION
    angle = float(state.rng.uniform(-GK_PARRY_SPREAD, GK_PARRY_SPREAD))
    ball.velocity = np.array([away * np.cos(angle), np.sin(angle)]) * speed
    ball.pos = keeper.pos + ball.velocity / max(speed, 1e-9) * 0.6
    ball.ground()
    ball.state = "in_pass"
    ball.last_toucher_team_id = keeper.team_id
    ball.last_shooter_id = None
    ball.last_passer_id = None
    ball.intended_receiver_id = None
    ball.offside_suspects = frozenset()
    state.add_event("save", keeper.team_id, keeper.name, f"{keeper.name} saves")


def _after_beat(state: GameState, ball_rest: np.ndarray, restart, *args) -> None:
    """Run `restart` now -- who takes it, the stats, the event: the whistle --
    but leave the ball at `ball_rest` and everyone where they stand for
    DEAD_BALL_TICKS before they are moved to it (game_loop applies the
    places). Moved on the tick the ball went out, it was never seen to go
    out, and the message seemed to come before it happened."""
    ball = state.ball
    held = {p.player_id: p.pos.copy() for p in state.all_players()}
    restart(state, *args)
    state.pending_places = (
        {p.player_id: p.pos.copy() for p in state.all_players()},
        ball.pos.copy(),
    )
    for p in state.all_players():
        p.pos = held[p.player_id]
    ball.pos = np.asarray(ball_rest, dtype=float).copy()
    state.pending_places_at = state.restart_pause
    state.restart_pause += DEAD_BALL_TICKS


def apply_pending_places(state: GameState) -> None:
    """Lay out the restart decided at the whistle (see _after_beat)."""
    places, ball_pos = state.pending_places
    for p in state.all_players():
        if p.player_id in places:
            p.pos = places[p.player_id].copy()
    state.ball.pos = ball_pos.copy()
    state.ball.ground()
    state.pending_places = None
    state.dead_ball_motion = None


def step_dead_ball(state: GameState) -> None:
    """One tick of a dead ball's own motion: on into the net, or on past the
    line until DEAD_BALL_LIMIT_M; falling if it is in the air. Nobody can
    touch it -- play has stopped."""
    ball = state.ball
    vel, height, vz = state.dead_ball_motion
    pos, vel, height, vz = step(ball.pos, vel, height, vz, DEAD_BALL_FRICTION)
    pos, vel = pos.copy(), vel.copy()
    if state.kickoff_pending:
        # In the net: its back and sides stop it, its roof holds it down.
        if pos[0] > PITCH_LENGTH / 2.0:
            depth = (RIGHT_GOAL_X, RIGHT_GOAL_X + GOAL_NET_DEPTH)
        else:
            depth = (LEFT_GOAL_X - GOAL_NET_DEPTH, LEFT_GOAL_X)
        for i, (lo, hi) in enumerate((depth, (GOAL_TOP_Y, GOAL_BOTTOM_Y))):
            if not lo <= pos[i] <= hi:
                pos[i] = min(max(pos[i], lo), hi)
                vel[i] = 0.0
        height = min(height, CROSSBAR_HEIGHT)
    else:
        for i, top in ((0, PITCH_LENGTH), (1, PITCH_WIDTH)):
            lo, hi = -DEAD_BALL_LIMIT_M, top + DEAD_BALL_LIMIT_M
            if not lo <= pos[i] <= hi:
                pos[i] = min(max(pos[i], lo), hi)
                vel = np.zeros(2)
    ball.pos = pos
    ball.height, ball.vz = height, vz
    state.dead_ball_motion = (vel, height, vz)


def _offside_free_kick(state: GameState, offender: Player) -> None:
    """Indirect free kick to the defending side, where the offence happened.

    Taken from where the offside player was when the ball reached him, which is
    the laws' "place where the offence occurred" simplified to the receiver's
    own position -- the engine has no notion of interfering with an opponent
    from elsewhere.
    """
    ball = state.ball
    defending_id = _other_team_id(state, offender.team_id)
    spot = offender.pos.copy()
    taker = _nearest_player(state, spot, defending_id)
    if taker is not None:
        taker.pos = spot.copy()
    _give_possession(state, ball, taker)
    ball.pos = spot.copy()
    ball.offside_suspects = frozenset()
    state.restart_pause = OFFSIDE_PAUSE_TICKS
    state.stats.side(state, offender.team_id).offsides += 1
    state.stats.player(state, offender).offsides += 1
    state.add_event(
        "offside", offender.team_id, offender.name,
        f"{offender.name} is offside",
    )


def _throw_in(state: GameState, restart_pos: np.ndarray) -> None:
    """Give the throw-in to the nearest player of the team that did NOT last touch."""
    ball = state.ball
    throw_team_id = _other_team_id(state, ball.last_toucher_team_id)
    player = _nearest_player(state, restart_pos, throw_team_id)
    if player is not None:
        player.pos = restart_pos.copy()
    _give_possession(state, ball, player)
    ball.pos = restart_pos.copy()
    # A pause but deliberately no reform: a throw-in restarts play from where it
    # stopped, so sending everyone home would look wrong.
    state.restart_pause = THROW_IN_PAUSE_TICKS
    if throw_team_id is not None:
        state.stats.side(state, throw_team_id).throw_ins += 1
    team_name = state.get_team(throw_team_id).name
    state.add_event(
        "throw_in", throw_team_id, player.name if player else None,
        f"Throw-in to {team_name}",
    )


def _goal_line_restart(state: GameState, left_line: bool, out_y: float) -> None:
    """
    Ball exited through a goal line (not in the goal):
      - Defending team last touched → corner kick for attacking team
      - Attacking team last touched → goal kick for defending team (GK)
    """
    ball = state.ball
    goal_x = LEFT_GOAL_X if left_line else RIGHT_GOAL_X
    defending_team = _team_defending_x(state, goal_x)
    attacking_team_id = _other_team_id(state, defending_team.team_id)

    last_team = ball.last_toucher_team_id
    if last_team is None or last_team == defending_team.team_id:
        # Defending team last touched → corner for attacking team
        corner_y = 0.0 if out_y < GOAL_CENTER_Y else PITCH_WIDTH
        restart_pos = np.array([goal_x, corner_y])
        player = _nearest_player(state, restart_pos, attacking_team_id)
        restart_kind, restart_team_id = "corner", attacking_team_id
    else:
        # Attacking team last touched → goal kick for defending team
        kick_x = 6.0 if left_line else PITCH_LENGTH - 6.0
        restart_pos = np.array([kick_x, GOAL_CENTER_Y])
        gk = _find_gk(state, defending_team.team_id)
        player = gk or _nearest_player(state, restart_pos, defending_team.team_id)
        restart_kind, restart_team_id = "goal_kick", defending_team.team_id

    if restart_kind == "goal_kick":
        # A goal kick is a genuine reset for BOTH sides. Restationing only the
        # defending team left the attackers wherever the move had stranded
        # them: cleared out of the box, but bunched on its edge in a wall the
        # keeper had to play through -- 9 of his 10 options scored P=0.000, and
        # the resulting giveaways in his own third were a real source of goals.
        # Reforming both teams is the same reset a kick-off already does.
        _restation_all(state)
    if player is not None:
        player.pos = restart_pos.copy()
    _give_possession(state, ball, player)
    ball.pos = restart_pos.copy()
    state.restart_pause = (
        GOAL_KICK_PAUSE_TICKS if restart_kind == "goal_kick" else CORNER_PAUSE_TICKS
    )
    side = state.stats.side(state, restart_team_id)
    if restart_kind == "corner":
        side.corners += 1
    else:
        side.goal_kicks += 1
    label = "Corner" if restart_kind == "corner" else "Goal kick"
    state.add_event(
        restart_kind, restart_team_id, player.name if player else None,
        f"{label} to {state.get_team(restart_team_id).name}",
    )


def _goal_pause(state: GameState) -> None:
    """The ball comes to rest in the net, where it went in; game_loop sets up
    the kick-off once GOAL_PAUSE_TICKS have passed."""
    ball = state.ball
    # It carries on into the net (step_dead_ball) rather than stopping on the
    # line -- and drops into it if it went in in the air.
    state.dead_ball_motion = (ball.velocity.copy(), float(ball.height), float(ball.vz))
    ball.velocity = np.zeros(2)
    ball.state = "stationary"
    state.kickoff_pending = True
    state.restart_pause = GOAL_PAUSE_TICKS


def _kickoff_restart(state: GameState) -> None:
    """After a goal: ball goes to centre circle, nearest opponent of scoring team kicks off."""
    ball = state.ball
    state.dead_ball_motion = None
    ball.pos = np.array([52.5, 34.0])
    ball.velocity = np.zeros(2)
    ball.state = "stationary"
    ball.ground()
    # The team that conceded kicks off
    conceding_team_id = _other_team_id(state, ball.last_toucher_team_id)
    cf = _find_position(state, conceding_team_id, "CF")
    player = cf or _nearest_player(state, ball.pos, conceding_team_id)
    # Both teams reform before a kick-off. Without this everyone restarts from
    # wherever the previous move left them, which looks nothing like football.
    _restation_all(state)
    _give_possession(state, ball, player)
    if player:
        ball.pos = np.array([52.5, 34.0])
        player.pos = np.array([52.5, 34.0])
    state.restart_pause = KICKOFF_PAUSE_TICKS
    state.add_event(
        "kickoff", conceding_team_id, player.name if player else None,
        f"Kick-off: {state.get_team(conceding_team_id).name}",
    )


def clamp_players_to_pitch(state: GameState) -> None:
    for player in state.all_players():
        player.pos[0] = float(np.clip(player.pos[0], 0.0, PITCH_LENGTH))
        player.pos[1] = float(np.clip(player.pos[1], 0.0, PITCH_WIDTH))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _give_possession(state: GameState, ball: Ball, player: Player | None) -> None:
    if player is None:
        return
    for p in state.all_players():
        p.has_ball = False
    player.has_ball = True
    ball.owner_id = player.player_id
    ball.state = "stationary"
    ball.velocity = np.zeros(2)
    ball.ground()
    ball.is_cross = False
    # A clean possession -- restarts, keeper gathers, a trapped ball. Only the
    # open-play capture of a fast ball sets a pending first touch, after this.
    player.first_touch_speed = 0.0
    player.first_touch_velocity = None
    player.first_touch_height = 0.0


def _nearest_player(state: GameState, pos: np.ndarray, team_id: int | None) -> Player | None:
    candidates = (
        state.home_team.players if team_id == state.home_team.team_id
        else state.away_team.players if team_id == state.away_team.team_id
        else state.all_players()
    )
    best, best_d = None, float("inf")
    for p in candidates:
        d = float(np.linalg.norm(p.pos - pos))
        if d < best_d:
            best, best_d = p, d
    return best


def _find_gk(state: GameState, team_id: int) -> Player | None:
    team = state.get_team(team_id)
    for p in team.players:
        if p.position == "GK":
            return p
    return None


def _find_position(state: GameState, team_id: int, position: str) -> Player | None:
    team = state.get_team(team_id)
    for p in team.players:
        if p.position == position:
            return p
    return None


def _team_defending_x(state: GameState, goal_x: float):
    """Return the team whose goal is at goal_x."""
    # direction=+1 attacks toward x=105, so defends x=0
    # direction=-1 attacks toward x=0,   so defends x=105
    for team in (state.home_team, state.away_team):
        own_goal_x = LEFT_GOAL_X if team.direction == 1 else RIGHT_GOAL_X
        if abs(own_goal_x - goal_x) < 1.0:
            return team
    return state.home_team


def _other_team_id(state: GameState, team_id: int | None) -> int:
    if team_id is None:
        # Unknown last toucher: a coin flip, not a free restart for the home
        # side. This fires at kick-off and after any untracked touch.
        return (
            state.home_team.team_id
            if state.rng.random() < 0.5
            else state.away_team.team_id
        )
    if team_id == state.home_team.team_id:
        return state.away_team.team_id
    return state.home_team.team_id


def _record_goal(state: GameState, conceding_goal_x: float) -> None:
    """Credit a goal to the team attacking the goal that was just scored in."""
    conceding_team = _team_defending_x(state, conceding_goal_x)
    scoring_team_id = _other_team_id(state, conceding_team.team_id)

    if scoring_team_id == state.home_team.team_id:
        state.home_score += 1
    else:
        state.away_score += 1
    state.stats.side(state, scoring_team_id).goals += 1

    scorer = state.ball.last_toucher_team_id
    scorer_name = None
    # Credit the goal to whoever last played the ball, which is the shooter in
    # every normal case. Only credited when he is on the SCORING side, so an
    # own goal is not recorded as a striker's finish.
    # A shot in flight has no owner, so prefer the shooter recorded when it was
    # struck; fall back to the owner for the rare goal that is not a shot (a
    # deflection off a carrier, say).
    shooter_id = state.ball.last_shooter_id
    if shooter_id is None:
        shooter_id = state.ball.owner_id
    if shooter_id is not None:
        try:
            shooter = state.get_player(shooter_id)
        except KeyError:
            shooter = None
        if shooter is not None:
            scorer_name = shooter.name
            # Only credited when the finisher is on the side that scored, so an
            # own goal is never recorded as a striker's finish.
            if shooter.team_id == scoring_team_id:
                state.stats.player(state, shooter).goals += 1

    scoring_name = state.get_team(scoring_team_id).name
    own_goal = scorer is not None and scorer != scoring_team_id
    detail = f"GOAL! {scoring_name} {state.home_score}-{state.away_score}"
    if own_goal:
        detail += " (own goal)"
    state.add_event("goal", scoring_team_id, scorer_name, detail)


def keeper_reach(keeper: Player, height: float) -> float:
    from .decisions.interception import keeper_reach as reach
    return reach(keeper, height)


def control_radius_of(player: Player) -> float:
    """The keeper's reach, shared with the interception model so a ball he can
    catch is exactly the ball he is modelled as able to reach."""
    from .decisions.interception import control_radius

    return control_radius(player)


def _restation_team(team) -> None:
    """Send every player on a team back to their formation station."""
    for player in team.players:
        player.pos = player.assigned_pos.copy()
        player.velocity = np.zeros(2)
        # The sticky bonus would otherwise pull them straight back out.
        player.last_off_ball_target = None


def _restation_all(state: GameState) -> None:
    _restation_team(state.home_team)
    _restation_team(state.away_team)
    # ...then hold each side's attackers onside against the OTHER side's newly
    # restationed line. Stations can sit beyond it -- at depth 2.0 both
    # centre-forwards were stationed 14.9 m behind the opposition defence -- and
    # teleporting to them at every goal kick and keeper gather left strikers
    # offside for the whole restart, which the off-ball clamp could never fix
    # because nobody was deciding anything during the pause.
    #
    # Against the line only, not the ball: at every restart that restations,
    # the ball is at or behind halfway, and offside_line never sits behind it.
    from .offside import offside_line

    for team in (state.home_team, state.away_team):
        line = offside_line(state, team)
        for player in team.players:
            if (float(player.pos[0]) - line) * team.direction > 0.0:
                player.pos = np.array([line, float(player.pos[1])])


def in_penalty_area(pos: np.ndarray, goal_x: float) -> bool:
    """Is `pos` inside the penalty area belonging to the goal at `goal_x`?"""
    depth = abs(float(pos[0]) - goal_x)
    return (
        depth <= PENALTY_AREA_DEPTH
        and PENALTY_AREA_TOP <= float(pos[1]) <= PENALTY_AREA_BOTTOM
    )


def clear_opponents_from_box(state: GameState, keeper_team_id: int, goal_x: float) -> None:
    """Push the attacking side out of the box while the keeper has the ball.

    At a goal kick only the DEFENDING team was being restationed, so the
    attackers stayed exactly where the move had left them -- camped on the
    keeper. Measured on a forced goal kick: four attackers inside the area, the
    nearest 0.5 m from the keeper, who simply took the ball off him and scored.

    Players are moved to just outside the area, the same teleport the other
    restarts already use; the restart pause is what makes it read as a reset.
    """
    for player in state.all_players():
        if player.team_id == keeper_team_id:
            continue
        if not in_penalty_area(player.pos, goal_x):
            continue

        # Push out the way that moves them least: through the D, or around the
        # side the player is already closer to.
        edge_x = (
            goal_x + PENALTY_AREA_DEPTH + BOX_CLEARANCE
            if goal_x < PITCH_LENGTH / 2.0
            else goal_x - PENALTY_AREA_DEPTH - BOX_CLEARANCE
        )
        out_x = abs(float(player.pos[0]) - edge_x)
        top_gap = abs(float(player.pos[1]) - (PENALTY_AREA_TOP - BOX_CLEARANCE))
        bottom_gap = abs(float(player.pos[1]) - (PENALTY_AREA_BOTTOM + BOX_CLEARANCE))

        if out_x <= min(top_gap, bottom_gap):
            player.pos = np.array([edge_x, float(player.pos[1])])
        elif top_gap <= bottom_gap:
            player.pos = np.array([float(player.pos[0]), PENALTY_AREA_TOP - BOX_CLEARANCE])
        else:
            player.pos = np.array([float(player.pos[0]), PENALTY_AREA_BOTTOM + BOX_CLEARANCE])

        player.pos = np.array([
            float(np.clip(player.pos[0], 0.0, PITCH_LENGTH)),
            float(np.clip(player.pos[1], 0.0, PITCH_WIDTH)),
        ])
        player.velocity = np.zeros(2)
        # Otherwise the sticky bonus pulls them straight back in.
        player.last_off_ball_target = None
