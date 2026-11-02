"""
Off-ball model: the speed/safety trade-off and marginal contribution.

The two behaviours that matter most here are the ones that cannot be produced
by a rule cascade without a headcount hack:
  - a striker does not track back when teammates already cover the ball
  - somebody always goes, even when several defenders are perfectly symmetric
"""
import math

import numpy as np
import pytest

from app.engine.constants import PLAYER_BASE_SPEED
from app.engine.decisions.marginal import marginal_contribution, player_risk_at
from app.engine.decisions.pursuit import (
    aim_point,
    compute_lambda,
    prediction_confidence,
)
from app.engine.factory import build_test_state
from tests.helpers import midfielder, striker
from app.engine.models import Player


def make_player(x, y, pid=1, team=1, speed=PLAYER_BASE_SPEED, position="CM"):
    return Player(
        player_id=pid, team_id=team, name=f"P{pid}", position=position,
        tactical="normal", pos=np.array([x, y], dtype=float),
        velocity=np.zeros(2), speed=speed, strength=70.0, intelligence=70.0,
        passing=70.0, goalkeeping=40.0,
        assigned_pos=np.array([x, y], dtype=float), marking_target_id=None,
        has_ball=False, runway_signal=None,
    )


# ── lambda: the speed/safety trade-off ───────────────────────────────────────

def test_lambda_bounded_for_all_inputs():
    for conf in np.linspace(0.0, 1.0, 11):
        for cover in np.linspace(0.0, 1.0, 11):
            for outlets in np.linspace(0.0, 1.0, 11):
                lam = compute_lambda(float(conf), float(cover), float(outlets))
                assert 0.0 <= lam <= 1.0 and math.isfinite(lam)


def test_lambda_rises_with_teammate_cover():
    """The key term: if others are already pressing, prioritise positioning
    over speed. Same carrier, same distance -- only the cover changes."""
    conf = prediction_confidence(0.30, 1.0, 15.0)
    alone = compute_lambda(conf, cover=0.0, outlets=0.0)
    covered = compute_lambda(conf, cover=0.9, outlets=0.0)
    assert covered > alone * 1.8
    assert alone == pytest.approx(0.29, abs=0.05)
    assert covered == pytest.approx(0.67, abs=0.05)


def test_lambda_falls_with_opponent_outlets():
    """Open passing options mean the carry trajectory probably will not hold,
    so committing to an interception point is a poor bet."""
    conf = prediction_confidence(0.30, 1.0, 15.0)
    assert compute_lambda(conf, 0.0, outlets=0.9) < compute_lambda(conf, 0.0, outlets=0.0)


def test_lambda_falls_for_unpredictable_carrier():
    """A stationary or weaving carrier tells you nothing -- close him down."""
    fast_straight = prediction_confidence(0.30, 1.0, 15.0)
    stationary = prediction_confidence(0.01, 0.03, 15.0)
    weaving = prediction_confidence(0.30, 0.3, 15.0)
    assert compute_lambda(stationary, 0.0, 0.0) < 0.05
    assert compute_lambda(weaving, 0.0, 0.0) < compute_lambda(fast_straight, 0.0, 0.0)


def test_confidence_decays_with_distance():
    near = prediction_confidence(0.30, 1.0, 15.0)
    far = prediction_confidence(0.30, 1.0, 60.0)
    assert far < near


def test_aim_point_endpoints():
    ball = np.array([50.0, 34.0])
    heading = np.array([1.0, 0.0])
    assert np.allclose(aim_point(ball, heading, 0.30, 20.0, 0.0), ball)
    full = aim_point(ball, heading, 0.30, 20.0, 1.0)
    assert full[0] == pytest.approx(50.0 + 0.30 * 20.0)


def test_aim_point_leads_further_when_slower_to_arrive():
    ball = np.array([50.0, 34.0])
    heading = np.array([1.0, 0.0])
    near = aim_point(ball, heading, 0.30, 15.0, 0.7)
    far = aim_point(ball, heading, 0.30, 45.0, 0.7)
    assert far[0] > near[0]


# ── marginal contribution ────────────────────────────────────────────────────

def test_marginal_equals_factored_identity():
    """(P_with - P_without) == prod(1-r_j) * r_me, exactly. The literal
    subtraction costs a second full team evaluation for the same number."""
    rng = np.random.default_rng(3)
    for _ in range(2000):
        rs = rng.uniform(0, 1, size=rng.integers(0, 6))
        r_me = float(rng.uniform(0, 1))
        without = 1.0 - np.prod(1.0 - rs)
        with_me = 1.0 - np.prod(1.0 - np.append(rs, r_me))
        assert (with_me - without) == pytest.approx(float(np.prod(1.0 - rs)) * r_me, abs=1e-12)


def test_marginal_is_absolute_when_nobody_committed():
    state = build_test_state()
    me = make_player(30.0, 34.0, pid=2)
    point = np.array([32.0, 34.0])
    marginal, cover = marginal_contribution(me, point, 30.0, [], state)
    assert cover == pytest.approx(0.0)
    assert marginal == pytest.approx(player_risk_at(me, point, 30.0))


def test_marginal_decays_as_teammates_commit():
    state = build_test_state()
    me = striker(state)
    point = state.ball.pos.copy()
    committed: list[tuple[int, np.ndarray]] = []
    previous = 1.0
    for mate in state.home_team.players[1:6]:
        mate.pos = point + np.array([1.0, 0.0])
        committed.append((mate.player_id, point.copy()))
        marginal, cover = marginal_contribution(me, point, 30.0, committed, state)
        assert marginal <= previous
        previous = marginal
    assert cover > 0.9


def test_striker_does_not_track_back():
    """With the ball covered by teammates, a distant striker's added value is
    negligible -- so he stays forward. No headcount rule produces this; it
    falls out of prod(1 - r_j) collapsing toward zero."""
    state = build_test_state()
    contest = np.array([25.0, 34.0])

    defenders = state.home_team.players[1:6]
    committed = []
    for i, d in enumerate(defenders):
        d.pos = contest + np.array([float(i) * 0.5, 1.0])
        committed.append((d.player_id, contest.copy()))

    forward = striker(state)
    forward.pos = np.array([75.0, 34.0])
    t_arrive = float(np.hypot(*(contest - forward.pos))) / forward.speed

    covered, _ = marginal_contribution(forward, contest, t_arrive, committed, state)
    assert covered < 0.02

    # Ablation: remove the cover and the same striker becomes worth sending.
    alone, _ = marginal_contribution(forward, contest, t_arrive, [], state)
    assert alone > covered * 5


def test_somebody_always_goes():
    """Four symmetric defenders conditioning on *everyone* would each compute
    the same small marginal and all defer -- nobody goes. Conditioning on
    predecessors only, the first faces an empty set, so he engages."""
    state = build_test_state()
    contest = np.array([30.0, 34.0])
    defenders = state.home_team.players[1:5]
    for d in defenders:
        d.pos = contest + np.array([5.0, 0.0])

    committed: list[tuple[int, np.ndarray]] = []
    marginals = []
    for d in defenders:
        m, _ = marginal_contribution(d, contest, 30.0, committed, state)
        marginals.append(m)
        committed.append((d.player_id, contest.copy()))

    assert marginals[0] > 0.3, "the first player must engage"
    assert marginals == sorted(marginals, reverse=True), "later players defer"
    assert marginals[-1] < marginals[0] / 2


# ── the two-phase guarantee ──────────────────────────────────────────────────

def test_decisions_do_not_mutate_state():
    """All decisions must read one frozen snapshot. The commitment table lives
    in a local dict and must never leak into `state`."""
    from app.engine.decisions.off_ball import off_ball_decision

    state = build_test_state()
    before = [(p.pos.copy(), p.velocity.copy(), p.has_ball) for p in state.all_players()]
    ball_before = (state.ball.pos.copy(), state.ball.velocity.copy(), state.ball.state)

    committed: list[tuple[int, np.ndarray]] = []
    for p in state.all_players():
        if not p.has_ball:
            action = off_ball_decision(p, state, committed)
            committed.append((p.player_id, action.target_pos))

    for p, (pos, vel, had) in zip(state.all_players(), before):
        assert np.allclose(p.pos, pos)
        assert np.allclose(p.velocity, vel)
        assert p.has_ball == had
    assert np.allclose(state.ball.pos, ball_before[0])
    assert np.allclose(state.ball.velocity, ball_before[1])
    assert state.ball.state == ball_before[2]


def test_decision_order_is_deterministic():
    """Order must be a pure function of the snapshot, not of roster list order."""
    from app.engine.game_loop import _decision_order

    state = build_test_state()
    first = [p.player_id for p in _decision_order(state)]

    state.home_team.players.reverse()
    state.away_team.players.reverse()
    state.invalidate_caches()
    second = [p.player_id for p in _decision_order(state)]

    assert first == second


def test_no_role_branching_in_off_ball():
    """Defensive behaviour is a phase, not a role. Role may only enter through
    the context table and the shape leash -- never as control flow."""
    from pathlib import Path

    import app.engine.decisions.off_ball as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("_DEFENDERS", "_MIDFIELDERS", "_FORWARDS"):
        assert banned not in source, f"{banned} reintroduces role branching"
    # The goalkeeper is the one legitimate role check.
    assert source.count('position == "GK"') <= 1


# ── Regressions found while playtesting ──────────────────────────────────────

def test_runway_signal_is_consumed():
    """Runway passes were generated and signals emitted, but the rewrite of
    off_ball.py dropped the code that reads them -- so the receiver never ran
    onto the ball and every through-ball was wasted."""
    from app.engine.decisions.off_ball import off_ball_decision
    from app.engine.signals import RunwaySignal

    state = build_test_state()
    receiver = striker(state)
    target = np.array([80.0, 40.0])
    receiver.runway_signal = RunwaySignal(
        sender_id=midfielder(state).player_id,
        target_pos=target,
        emitted_tick=state.tick,
    )
    action = off_ball_decision(receiver, state, [])
    assert np.allclose(action.target_pos, target)
    assert action.sprint


def test_loose_ball_chase_leads_the_ball():
    """A rolling ball is not a stationary target. Aiming at where it currently
    sits misses by however far it travels while you run."""
    from app.engine.decisions.pursuit import ball_intercept_point
    from app.engine.constants import BALL_FRICTION_COEFF

    chaser = make_player(40.0, 34.0, pid=99)
    ball_pos = np.array([50.0, 34.0])
    rolling = np.array([0.0, 0.6])  # moving across, away from the chaser's line

    point, ticks = ball_intercept_point(ball_pos, rolling, BALL_FRICTION_COEFF, chaser)
    assert ticks > 0
    # The intercept point must be ahead of the ball along its travel direction.
    assert point[1] > ball_pos[1] + 1.0


def test_stationary_ball_is_chased_directly():
    from app.engine.decisions.pursuit import ball_intercept_point
    from app.engine.constants import BALL_FRICTION_COEFF

    chaser = make_player(40.0, 34.0, pid=99)
    ball_pos = np.array([50.0, 34.0])
    point, _ = ball_intercept_point(ball_pos, np.zeros(2), BALL_FRICTION_COEFF, chaser)
    assert np.allclose(point, ball_pos)


def test_nearby_player_engages_a_free_ball():
    """The engagement gate previously used absolute reach probability, which is
    ~0.99 for everyone within a few metres. prod(1 - r_j) then collapsed to zero
    and a player 3.7 m from a free ball stood still because 'someone else has
    it'. Engagement is a race, not a coverage sum."""
    from app.engine.decisions.off_ball import _shape_anchor, off_ball_decision

    state = build_test_state()
    state.ball.owner_id = None
    state.ball.velocity = np.zeros(2)
    state.ball.state = "stationary"
    state.ball.pos = np.array([40.0, 34.0])

    nearest = midfielder(state)
    nearest.pos = np.array([43.0, 34.0])
    for other in state.home_team.players:
        if other is not nearest:
            other.pos = np.array([15.0, 34.0])

    action = off_ball_decision(nearest, state, [])
    anchor = _shape_anchor(nearest, state.ball, state.home_team)
    assert not np.allclose(action.target_pos, anchor), "held shape next to a free ball"
    assert float(np.hypot(*(action.target_pos - state.ball.pos))) < 3.0


# ── Meeting a pass rather than chasing it ───────────────────────────────────

def test_contest_is_judged_on_arrival_time_not_distance():
    """A distance margin called 99% of loose balls contested, which left the
    uncontested logic dead. Players differ in speed, so who actually arrives
    first is the honest test."""
    from app.engine.decisions.off_ball import _ball_is_contested

    state = build_test_state(seed=0)
    receiver = state.home_team.players[5]
    receiver.pos = np.array([50.0, 34.0])
    state.ball.pos = np.array([56.0, 34.0])
    state.ball.owner_id = None

    # Everyone far away: nobody is racing us for it.
    for o in state.away_team.players:
        o.pos = np.array([5.0, 5.0])
    assert not _ball_is_contested(receiver, state)

    # An opponent right on the ball certainly is.
    state.away_team.players[4].pos = np.array([57.0, 34.0])
    assert _ball_is_contested(receiver, state)


def test_a_safe_high_value_ball_may_be_let_to_run():
    """Letting the ball run on is a luxury, allowed only when the chosen spot is
    comfortably ours AND clearly worth more."""
    from app.engine.constants import BALL_FRICTION_COEFF
    from app.engine.decisions.pursuit import ball_intercept_point, best_meeting_point

    state = build_test_state(seed=0)
    receiver = state.home_team.players[9]
    receiver.pos = np.array([70.0, 34.0])
    ball_pos = np.array([72.0, 34.0])
    ball_vel = np.array([0.5, 0.0])  # rolling on toward the goal
    # Nobody anywhere near: the spot down the line is genuinely ours.
    for o in state.away_team.players:
        o.pos = np.array([5.0, 5.0])

    earliest, _ = ball_intercept_point(ball_pos, ball_vel, BALL_FRICTION_COEFF, receiver)
    best, _ = best_meeting_point(
        ball_pos, ball_vel, BALL_FRICTION_COEFF, receiver, 105.0,
        state.away_team.players,
    )
    assert best[0] >= earliest[0] - 1e-9


def test_a_contested_ball_is_taken_at_the_earliest_point():
    """The regression this guards: testing safety at the ball's CURRENT
    position let a receiver wait until tick 120 for a spot an opponent reached
    at tick 20, losing the ball every time."""
    from app.engine.constants import BALL_FRICTION_COEFF
    from app.engine.decisions.pursuit import ball_intercept_point, best_meeting_point

    state = build_test_state(seed=0)
    receiver = state.home_team.players[9]
    receiver.pos = np.array([70.0, 34.0])
    ball_pos = np.array([72.0, 34.0])
    ball_vel = np.array([0.5, 0.0])
    # An opponent camped down the line owns everything further on.
    state.away_team.players[4].pos = np.array([88.0, 34.0])

    earliest = ball_intercept_point(ball_pos, ball_vel, BALL_FRICTION_COEFF, receiver)
    chosen = best_meeting_point(
        ball_pos, ball_vel, BALL_FRICTION_COEFF, receiver, 105.0,
        state.away_team.players,
    )
    assert np.allclose(chosen[0], earliest[0])


def test_waiting_never_concedes_the_chosen_point():
    """Whatever spot is chosen, we must reach it before any opponent."""
    from app.engine.constants import BALL_FRICTION_COEFF
    from app.engine.decisions.pursuit import best_meeting_point

    state = build_test_state(seed=0)
    receiver = state.home_team.players[5]
    opponents = state.away_team.players
    for vel in (np.array([0.6, 0.0]), np.array([0.0, 0.4]), np.array([-0.5, 0.2])):
        point, ticks = best_meeting_point(
            state.ball.pos, vel, BALL_FRICTION_COEFF, receiver, 105.0, opponents
        )
        if ticks <= 0:
            continue
        mine = float(np.hypot(*(point - receiver.pos))) / receiver.speed
        for o in opponents:
            theirs = float(np.hypot(*(point - o.pos))) / o.speed
            assert theirs > mine or mine <= ticks + 1e-6


def test_meeting_point_is_always_reachable():
    from app.engine.constants import BALL_FRICTION_COEFF
    from app.engine.decisions.pursuit import best_meeting_point

    state = build_test_state(seed=0)
    receiver = state.home_team.players[5]
    for vel in (np.array([0.6, 0.0]), np.array([0.0, 0.4]), np.zeros(2)):
        point, ticks = best_meeting_point(
            state.ball.pos, vel, BALL_FRICTION_COEFF, receiver, 105.0,
            state.away_team.players,
        )
        assert np.all(np.isfinite(point))
        if ticks > 0:
            travel = float(np.hypot(*(point - receiver.pos)))
            assert travel <= receiver.speed * ticks + 1e-6
