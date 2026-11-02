"""
Playmaking: the ball owner picks a spot and sends people to it.

The runway passes built from a teammate's current velocity can only exploit a
run that is already happening. This inverts the initiative: the carrier finds
the most dangerous spot he can actually hit, tells the best-placed teammates to
go there, and plays the ball into it.

The spot is chosen with the same P x V the pass model uses, so a signalled ball
is not a special case -- it is a pass candidate whose receiver has been told to
show up.
"""
from __future__ import annotations

import numpy as np

from ..ball_physics import nominal_pass_speed, travel_ticks
from ..constants import (
    BALL_FRICTION_COEFF,
    PASS_SPEED_HIGH,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SIGNAL_ARC_COUNT,
    SIGNAL_MAX_RECIPIENTS,
    SIGNAL_MIN_UTILITY,
    SIGNAL_RACE_MARGIN_TICKS,
    ARRIVAL_TOLERANCE,
    PROMPTNESS_TAU,
    SIGNAL_RADII,
    SUPERIORITY_MAX,
    SUPERIORITY_MIN,
    SUPERIORITY_RADIUS,
    SUPERIORITY_WEIGHT,
    TOUCHLINE_MARGIN,
)
from .. import kinematics
from ..geometry import distance
from ..kinematics import reach_ticks
from ..offside import is_offside_position
from ..models import GameState, Player
from .interception import route_survival
from .pursuit import space_quality
from .value import position_value


def _times(players: "list[Player]", spot: np.ndarray) -> list[float]:
    """Ticks for each player to reach `spot`: from their actual motion under
    inertia, else the old straight line at top speed."""
    if not players:
        return []
    if kinematics.INERTIA:
        P = np.array([p.pos for p in players]); V = np.array([p.velocity for p in players])
        S = np.array([p.speed for p in players])
        return [float(t) for t in reach_ticks(P, V, S, spot[None, :])]
    return [distance(p.pos, spot) / max(p.speed, 1e-9) for p in players]


def _candidate_spots(player: Player, attacking_goal_x: float) -> list[np.ndarray]:
    """Arcs of space ahead of the carrier, fanned toward the opponent goal."""
    forward = 1.0 if attacking_goal_x > PITCH_LENGTH / 2.0 else -1.0
    spots: list[np.ndarray] = []
    for radius in SIGNAL_RADII:
        for i in range(SIGNAL_ARC_COUNT):
            # Fan across +/-70 degrees of straight ahead.
            frac = (i / (SIGNAL_ARC_COUNT - 1)) * 2.0 - 1.0
            angle = frac * np.deg2rad(70.0)
            dx = forward * np.cos(angle) * radius
            dy = np.sin(angle) * radius
            spot = np.array([player.pos[0] + dx, player.pos[1] + dy])
            # Reject spots off the pitch rather than clamping them onto it.
            # Clamping pinned called runs to y=0.0 / y=68.0 exactly, and the
            # ball simply rolled over the touchline -- this was the single
            # biggest source of throw-ins and it killed attacks.
            if _inside_playable_area(spot):
                spots.append(spot)
    return spots


def _inside_playable_area(point: np.ndarray) -> bool:
    """Leave a margin: the ball overruns its target, so a pass aimed at the
    line itself goes out."""
    return (
        TOUCHLINE_MARGIN <= float(point[0]) <= PITCH_LENGTH - TOUCHLINE_MARGIN
        and TOUCHLINE_MARGIN <= float(point[1]) <= PITCH_WIDTH - TOUCHLINE_MARGIN
    )


def find_signal_target(
    player: Player,
    state: GameState,
) -> tuple[np.ndarray, list[int]] | None:
    """Best spot to play into, plus the teammates who should run onto it.

    Returns None when no spot is worth calling for -- the carrier then just
    picks from the ordinary candidate list.
    """
    team = state.get_team(player.team_id)
    attacking_goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    opponents = state.opponents_of(player.team_id)
    # Only an onside teammate can be sent into space: the call is a pass, and a
    # runner who is offside when it is played gives away a free kick however
    # well the spot was chosen.
    teammates = [
        t for t in state.teammates_of(player.player_id)
        if t.position != "GK"
        and not is_offside_position(t, state, team, ball_x=float(player.pos[0]))
    ]
    if not teammates:
        return None

    best_spot = None
    best_utility = SIGNAL_MIN_UTILITY
    best_runners: list[int] = []

    for spot in _candidate_spots(player, attacking_goal_x):
        ball_dist = distance(player.pos, spot)
        t_ball = travel_ticks(
            nominal_pass_speed(BALL_FRICTION_COEFF, ball_dist),
            BALL_FRICTION_COEFF, ball_dist,
        )
        if t_ball == float("inf"):
            continue

        # Can anyone actually get there around the time the ball does?
        runners = sorted(zip(_times(teammates, spot), (t.player_id for t in teammates)))
        arriving = [pid for t_run, pid in runners if t_run <= t_ball * ARRIVAL_TOLERANCE]
        if not arriving:
            continue

        # The spot must be OURS to reach: our best runner has to beat every
        # opponent to it, with a margin, or the ball is simply being played to
        # whoever gets there first -- and that was not us.
        #
        # A hard gate, deliberately, not another factor in the product below.
        # As a multiplier a high position_value near their goal would buy back
        # an unwinnable race, which is exactly what the old scoring did: nothing
        # here compared our runner to the opposition at all. promptness races
        # him against the BALL, and space_quality only measures how far the
        # nearest opponent stands from the spot at this instant, not who arrives
        # first. Measured consequence: the runner reached the called spot 19-30%
        # of the time, and once signals persisted long enough for him to
        # genuinely commit, called runs held the ball 12-19% against 45-53% for
        # an ordinary pass. He was being sent to lose a race.
        best_run_time = runners[0][0]
        best_opp_time = min(_times(opponents, spot), default=float("inf"))
        if best_run_time + SIGNAL_RACE_MARGIN_TICKS >= best_opp_time:
            continue

        # How comfortably the best runner gets there, not merely whether he
        # scrapes inside the cutoff. A hard yes/no scored a runner 8 m short
        # exactly like one already standing on the spot.
        promptness = 1.0 / (1.0 + max(0.0, best_run_time - t_ball) / PROMPTNESS_TAU)

        # Local numerical superiority. If our people outnumber theirs around
        # the spot, the ball is far likelier to stay ours even if the first
        # touch is contested -- this is what makes a 3-v-1 worth playing into,
        # and nothing in the model rewarded it before.
        superiority = _local_superiority(spot, teammates, opponents)

        p_arrives = route_survival(
            player.pos, spot,
            nominal_pass_speed(BALL_FRICTION_COEFF, ball_dist),
            BALL_FRICTION_COEFF, opponents
        )
        value = position_value(spot, attacking_goal_x)
        space = space_quality(spot, opponents)
        utility = p_arrives * value * space * promptness * superiority

        if utility > best_utility:
            best_utility = utility
            best_spot = spot
            best_runners = arriving[:SIGNAL_MAX_RECIPIENTS]

    if best_spot is None:
        return None
    return best_spot, best_runners


def _local_superiority(
    spot: np.ndarray,
    teammates: list[Player],
    opponents: list[Player],
) -> float:
    """Numerical advantage around `spot`, as a multiplier near 1.0.

    Counts bodies within SUPERIORITY_RADIUS. Being 3-v-1 in the box is the
    whole reason to play a ball there; being 1-v-3 is why the same ball is a
    giveaway. Deliberately gentle -- this is a tilt on an already-meaningful
    utility, not the thing doing the deciding.
    """
    mine = sum(1 for t in teammates if distance(t.pos, spot) <= SUPERIORITY_RADIUS)
    theirs = sum(1 for o in opponents if distance(o.pos, spot) <= SUPERIORITY_RADIUS)
    advantage = mine - theirs
    return float(
        min(
            SUPERIORITY_MAX,
            max(SUPERIORITY_MIN, 1.0 + SUPERIORITY_WEIGHT * advantage),
        )
    )


def choose_attack_runners(
    carrier: "Player", state: "GameState"
) -> tuple[int, ...]:
    """Which teammates the carrier calls forward to support the attack.

    The attack_call parameter is the CARRIER's willingness to summon runs, not
    a runner's own eagerness -- one player decides, several respond. That is
    what makes this expressible in the unit-based model: no global "we are
    attacking" state, just a decision by whoever has the ball.

    Eligible teammates are within earshot and still behind the ball, since a
    player already ahead of it is in the attack zone. They are ranked by how
    useful the run would be -- upfield room available, times space where they
    would arrive -- and the parameter sets how many are called.
    """
    from ..constants import (
        ATTACK_CALL_BASE_RUNNERS,
        ATTACK_CALL_MIN_ROOM,
        ATTACK_CALL_RADIUS,
        ATTACK_CALL_SATED_SAFETY,
        ATTACK_RUN_RANGE,
        PITCH_LENGTH,
    )
    from .pursuit import best_pass_safety
    from .tunables import attack_call_bias

    from ..signals import JOIN
    if JOIN:
        return _choose_joiners(carrier, state)

    team = state.get_team(carrier.team_id)
    call = attack_call_bias(carrier, team)
    attacking_goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    opponents = state.opponents_of(carrier.team_id)
    ball = state.ball

    # A carrier with safe outlets everywhere needs nobody; a pressed one needs
    # bodies ahead of him. The slider still buys extra runners on top, so a
    # high setting puts men in the box even when the ball is comfortable.
    sated = min(1.0, best_pass_safety(state, carrier) / ATTACK_CALL_SATED_SAFETY)
    wanted = ATTACK_CALL_BASE_RUNNERS * call * (1.0 - 0.5 * sated)
    count = int(wanted)
    # Keep the fractional part meaningful rather than truncating it away: at
    # 0.5x the carrier should still sometimes call a single runner.
    if state.rng.random() < wanted - count:
        count += 1
    if count <= 0:
        return ()

    ranked: list[tuple[float, int]] = []
    for mate in state.teammates_of(carrier.player_id):
        if mate.position == "GK":
            continue
        if distance(mate.pos, carrier.pos) > ATTACK_CALL_RADIUS:
            continue
        # Already ahead of the ball: he is the attack, not support for it.
        ahead = (float(mate.pos[0]) - float(ball.pos[0])) * team.direction
        if ahead > 0.0:
            continue
        room = abs(attacking_goal_x - float(mate.pos[0]))
        if room < ATTACK_CALL_MIN_ROOM:
            continue
        # Where he would be heading, judged for space so we do not call someone
        # into a crowd.
        target = mate.pos + np.array([team.direction * ATTACK_RUN_RANGE, 0.0])
        ranked.append((room * space_quality(target, opponents), mate.player_id))

    ranked.sort(key=lambda r: -r[0])
    return tuple(pid for _, pid in ranked[:count])


def _choose_joiners(carrier: "Player", state: "GameState"):
    """Attack call: the carrier judges his teammates' runs and sends the best
    (decisions D-13, D-14; see JOIN_* in constants).

    Every JOIN_REJUDGE_TICKS he judges each outfield teammate's possible runs
    (_run_spots):

        P(first) = 1 / (1 + exp((his ticks - first opponent's ticks) / JOIN_RACE_TAU))
        worth    = lane from the ball now x position value x space   (at the spot, and where he is)
        score    = worth(spot) x P(first) x (1 - risk) ** (JOIN_RISK_EXPONENT / Attack call)
                   / (1 + ticks / JOIN_REACH_TAU)
        only if worth(spot) x P(first) - worth(here) > JOIN_MIN_GAIN

    -- how good a place the run puts him in, not how much better than where he
    is: ranked by the gain, a deep full-back up an empty flank always won
    (about 90% of runs) over a midfielder already in a decent spot.

    and sends the best, up to int(JOIN_PER_SETTING x his Attack call + 0.5),
    one at a time, recounting the cover each time a defender goes. A current
    run keeps (1 + JOIN_STICKY) on its score and its spot. Returns a
    signals.JoinPlan.
    """
    from ..constants import (
        BALL_FRICTION_COEFF,
        JOIN_MIN_GAIN,
        JOIN_PER_SETTING,
        JOIN_REACH_TAU,
        JOIN_REJUDGE_TICKS,
        JOIN_RISK_EXPONENT,
        JOIN_STICKY,
    )
    from ..signals import JoinPlan
    from .interception import route_survival_many
    from .tunables import attack_call_bias

    if state.tick % JOIN_REJUDGE_TICKS:
        return JoinPlan(None)
    team = state.get_team(carrier.team_id)
    setting = attack_call_bias(carrier, team)
    wanted = int(JOIN_PER_SETTING * setting + 0.5)
    goal_x = PITCH_LENGTH if team.direction == 1 else 0.0
    opponents = state.opponents_of(carrier.team_id)
    ball = state.ball.pos
    mates = [m for m in state.teammates_of(carrier.player_id) if m.position != "GK"]
    if not mates or wanted <= 0:
        return JoinPlan(())

    # Every teammate's runs, judged in one batch of lanes. Lanes are judged
    # NOW, not at his arrival: projecting the defenders along their momentum
    # for the 2-3 s a run takes closed every lane (all 0.00), so no run ever
    # gained. Who gets there first is the race below.
    from ..constants import JOIN_MIN_RACE, JOIN_RACE_TAU
    outfield = [o for o in opponents if o.speed > 1e-9]
    OP = np.array([o.pos for o in outfield]); OV = np.array([o.velocity for o in outfield])
    OS = np.array([o.speed for o in outfield])
    rows = []                                   # (mate, spots, ticks, P(first))
    for m in mates:
        if state.tick < m.join_cooldown_until:
            continue
        spots = _run_spots(m, state, team)
        running = m.attack_call is not None and m.attack_call.spot is not None
        if running:
            spots.append(m.attack_call.spot.copy())
        if not spots:
            continue
        pts = np.array(spots)
        if kinematics.INERTIA:
            ticks = np.asarray(reach_ticks(m.pos, m.velocity, max(m.speed, 1e-9), pts), dtype=float)
            t_opp = (np.asarray(reach_ticks(OP[None], OV[None], OS[None], pts[:, None]), dtype=float).min(axis=1)
                     if len(outfield) else np.full(len(pts), np.inf))
        else:
            ticks = np.hypot(pts[:, 0] - m.pos[0], pts[:, 1] - m.pos[1]) / max(m.speed, 1e-9)
            t_opp = (np.min(np.hypot(pts[:, None, 0] - OP[None, :, 0], pts[:, None, 1] - OP[None, :, 1])
                            / OS[None, :], axis=1) if len(outfield) else np.full(len(pts), np.inf))
        first = 1.0 / (1.0 + np.exp(np.clip((ticks - t_opp) / JOIN_RACE_TAU, -50.0, 50.0)))
        ours = first >= JOIN_MIN_RACE
        if running:
            ours[-1] = True                     # his current spot stays judged
        if not ours.any():
            continue
        rows.append((m, pts[ours], ticks[ours], first[ours]))
    if not rows:
        return JoinPlan(())
    here = np.array([m.pos for m, _, _, _ in rows])
    allpts = np.concatenate([here] + [pts for _, pts, _, _ in rows])
    gaps = np.hypot(allpts[:, 0] - ball[0], allpts[:, 1] - ball[1])
    paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(g)) for g in gaps])
    lanes = route_survival_many(ball, allpts, paces, BALL_FRICTION_COEFF, opponents)
    worth = lanes * np.array([position_value(p, goal_x) * space_quality(p, opponents) for p in allpts])

    best = {}                                   # player id -> (score before risk, spot)
    at = len(rows)
    for r, (m, pts, ticks, first) in enumerate(rows):
        now = float(worth[r])
        there = worth[at:at + len(pts)] * first
        gain = there - now
        score = np.where(gain > JOIN_MIN_GAIN, there, -1.0) / (1.0 + ticks / JOIN_REACH_TAU)
        running = m.attack_call is not None and m.attack_call.spot is not None
        if running and np.allclose(pts[-1], m.attack_call.spot):
            score[-1] *= 1.0 + JOIN_STICKY      # his current spot
        k = int(np.argmax(score))
        if gain[k] > JOIN_MIN_GAIN:
            s = float(score[k]) * (1.0 + JOIN_STICKY if running else 1.0)
            best[m.player_id] = (s, pts[k].copy(), m)
        at += len(pts)

    # Send the best, one at a time: each man who goes leaves less cover.
    behind = {m.player_id for m in mates}       # who is staying back
    chosen = []
    exponent = JOIN_RISK_EXPONENT / setting
    while best and len(chosen) < wanted:
        def total(item):
            s, _, m = item
            return s * (1.0 - _risk(m, behind, state, team)) ** exponent
        pid = max(best, key=lambda p: (total(best[p]), -p))
        if total(best[pid]) <= 0.0:
            break
        chosen.append((pid, best[pid][1]))
        behind.discard(pid)
        del best[pid]
    return JoinPlan(tuple(chosen))


def _run_spots(mate, state, team) -> list:
    """Where a teammate could run: JOIN_RUN_LENGTHS m from where he is, at
    JOIN_RUN_ANGLES off straight at their goal, plus a run in behind to the
    offside line in his channel when that is 5 m or more ahead of him; each
    held onside and 3 m inside the lines."""
    from ..constants import JOIN_RUN_ANGLES, JOIN_RUN_LENGTHS, PITCH_WIDTH
    from ..offside import onside_limit

    line = onside_limit(state, team)
    d = team.direction
    out = []

    def keep(x, y):
        if (x - line) * d > 0.0:
            x = line
        x = min(max(x, TOUCHLINE_MARGIN), PITCH_LENGTH - TOUCHLINE_MARGIN)
        y = min(max(y, TOUCHLINE_MARGIN), PITCH_WIDTH - TOUCHLINE_MARGIN)
        if (x - float(mate.pos[0])) * d >= 2.0:      # a run goes forward
            out.append(np.array([x, y]))

    for angle in JOIN_RUN_ANGLES:
        a = np.radians(angle)
        for length in JOIN_RUN_LENGTHS:
            keep(float(mate.pos[0]) + d * length * np.cos(a), float(mate.pos[1]) + length * np.sin(a))
    if (line - float(mate.pos[0])) * d >= 5.0:
        keep(line, float(mate.pos[1]))
    return out


def _risk(mate, behind: set, state, team) -> float:
    """What calling him costs at the back, 0 to 1: danger x shortage.
    danger = min(1, threat where he stands / threat JOIN_DANGER_REF_M from our
    goal); shortage = the larger of 1 - teammates staying deeper than him /
    JOIN_REST_DEFENCE, and 1 - those of them within JOIN_COVER_RADIUS of him /
    JOIN_LOCAL_COVER (floored at 0; the local part only in our half).
    `behind`: the outfield teammates not sent forward."""
    from ..constants import (
        JOIN_COVER_RADIUS,
        JOIN_DANGER_REF_M,
        JOIN_LOCAL_COVER,
        JOIN_REST_DEFENCE,
        PITCH_WIDTH,
    )
    from .value import threat

    own_goal_x = 0.0 if team.direction == 1 else PITCH_LENGTH
    ref = threat(np.array([own_goal_x + team.direction * JOIN_DANGER_REF_M, PITCH_WIDTH / 2.0]), own_goal_x)
    danger = min(1.0, threat(mate.pos, own_goal_x) / max(ref, 1e-9))
    mine = (float(mate.pos[0]) - own_goal_x) * team.direction
    others = [o for o in (state.get_player(pid) for pid in behind if pid != mate.player_id)
              if (float(o.pos[0]) - own_goal_x) * team.direction < mine]
    shortage = max(0.0, 1.0 - len(others) / JOIN_REST_DEFENCE)
    if mine < PITCH_LENGTH / 2.0:
        # Only in our half does the area he leaves need covering: a forward
        # with nobody near him has no flank to leave open.
        near = sum(1 for o in others if distance(o.pos, mate.pos) <= JOIN_COVER_RADIUS)
        shortage = max(shortage, 1.0 - near / JOIN_LOCAL_COVER)
    return danger * shortage
