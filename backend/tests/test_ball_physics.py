"""
Closed-form ball kinematics must agree with the discrete simulator.

The decision layer asks "how long does the ball take to get there?" thousands of
times per tick and cannot afford to step a loop. These tests are what license
that substitution -- if the closed form drifts from physics.compute_ball_update,
every interception probability in the engine is quietly wrong.
"""
import math

import numpy as np
import pytest
from app.engine.ball_physics import speed_for_arrival

from app.engine.ball_physics import carry_distance, travel_distance, travel_ticks
from app.engine.constants import (
    DRIBBLE_FRICTION_COEFF,
    PASS_TICKS_BASE,
    PASS_TICKS_PER_M,
    PASS_TICKS_DRIVEN,
    PASS_SPEED_MAX,
    BALL_FRICTION_COEFF,
    BALL_STOP_THRESHOLD,
    DRIBBLE_FRICTION_COEFF,
    DRIBBLE_SPEED,
    PASS_SPEED_HIGH,
    PASS_SPEED_LOW,
    SHOT_FRICTION_COEFF,
    SHOT_SPEED,
    TICK_RATE,
)
from app.engine.models import Ball
from app.engine.physics import compute_ball_update

# The (speed, friction) pairs the engine actually uses.
REAL_CASES = [
    (PASS_SPEED_LOW, BALL_FRICTION_COEFF),
    (PASS_SPEED_HIGH, BALL_FRICTION_COEFF),
    (SHOT_SPEED, SHOT_FRICTION_COEFF),
    (DRIBBLE_SPEED, DRIBBLE_FRICTION_COEFF),
]


def _simulate(speed: float, friction: float, state: str = "in_pass"):
    """Step the real engine ball update until it stops. Returns (distance, ticks)."""
    ball = Ball(
        pos=np.zeros(2),
        velocity=np.array([speed, 0.0]),
        state=state,
        owner_id=None,
        last_toucher_team_id=None,
    )
    ticks = 0
    while ball.state != "stationary" and ticks < 200_000:
        pos, vel, st, _, _ = compute_ball_update(ball)
        ball.pos, ball.velocity, ball.state = pos, vel, st
        ticks += 1
    return float(ball.pos[0]), ticks


@pytest.mark.parametrize("speed,friction", REAL_CASES)
def test_carry_distance_matches_simulator(speed, friction):
    """carry_distance must equal what the engine's own tick loop produces."""
    state = "in_shot" if friction == SHOT_FRICTION_COEFF else (
        "in_dribble" if friction == DRIBBLE_FRICTION_COEFF else "in_pass"
    )
    sim_dist, _ = _simulate(speed, friction, state)
    # One tick of slack: the sim decays iteratively so float drift decides the
    # boundary tick, and no closed form matches every case. Irrelevant in
    # practice -- carry is now 200-2000 m on a 105 m pitch, so it only ever
    # answers "reachable", and travel_ticks is exact at real pass ranges.
    assert carry_distance(speed, friction) == pytest.approx(sim_dist, abs=0.05)


@pytest.mark.parametrize("speed,friction", REAL_CASES)
def test_travel_ticks_matches_simulator(speed, friction):
    """Arrival time at a range of distances must match the discrete loop within 1%."""
    state = "in_shot" if friction == SHOT_FRICTION_COEFF else (
        "in_dribble" if friction == DRIBBLE_FRICTION_COEFF else "in_pass"
    )
    reach = carry_distance(speed, friction)
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        target = reach * frac
        # Step the sim until it passes `target`.
        ball = Ball(np.zeros(2), np.array([speed, 0.0]), state, None, None)
        ticks = 0
        while float(ball.pos[0]) < target and ball.state != "stationary":
            pos, vel, st, _, _ = compute_ball_update(ball)
            ball.pos, ball.velocity, ball.state = pos, vel, st
            ticks += 1
        assert travel_ticks(speed, friction, target) == pytest.approx(ticks, rel=0.01, abs=1.0)


@pytest.mark.parametrize("speed,friction", REAL_CASES)
def test_round_trip(speed, friction):
    """travel_distance inverts travel_ticks."""
    reach = carry_distance(speed, friction)
    for frac in (0.05, 0.3, 0.6, 0.95):
        d = reach * frac
        assert travel_distance(speed, friction, travel_ticks(speed, friction, d)) == pytest.approx(d, rel=1e-6)


@pytest.mark.parametrize("speed,friction", REAL_CASES)
def test_infinite_beyond_carry_distance(speed, friction):
    """A ball cannot reach further than it carries -- this is what stops the
    decision layer believing in passes that physically die short."""
    reach = carry_distance(speed, friction)
    assert travel_ticks(speed, friction, reach * 0.99) < float("inf")
    assert travel_ticks(speed, friction, reach * 1.01) == float("inf")


def test_travel_ticks_monotonic():
    """Further away is never sooner."""
    prev = -1.0
    for d in np.arange(1.0, 30.0, 0.5):
        t = travel_ticks(PASS_SPEED_LOW, BALL_FRICTION_COEFF, float(d))
        assert t > prev
        prev = t


def test_degenerate_inputs():
    assert travel_ticks(PASS_SPEED_LOW, BALL_FRICTION_COEFF, 0.0) == 0.0
    assert travel_ticks(0.0, BALL_FRICTION_COEFF, 10.0) == float("inf")
    assert carry_distance(0.0, BALL_FRICTION_COEFF) == 0.0
    assert travel_distance(PASS_SPEED_LOW, BALL_FRICTION_COEFF, 0.0) == 0.0
    # A ball already below the stop threshold goes nowhere.
    assert carry_distance(BALL_STOP_THRESHOLD * 0.5, BALL_FRICTION_COEFF) == 0.0


# ── Regression guards on the tuned constants ─────────────────────────────────
# These fail loudly if anyone reintroduces a near-frictionless ball.

def test_pass_arrival_is_realistic():
    """Arrival TIME is the quantity that matters, not carry distance.

    Under real rolling resistance (~0.4 m/s^2) an untouched ball runs for
    hundreds of metres, so carry says nothing about whether a pass is sane --
    the ball stops because someone controls it. Real football: a firm ground
    pass covers 10 m in ~0.6-0.9 s and 20 m in ~1.1-1.5 s.
    """
    for dist, lo, hi in ((10.0, 0.6, 0.9), (20.0, 1.1, 1.5), (30.0, 1.6, 2.2)):
        ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * dist
        speed = speed_for_arrival(BALL_FRICTION_COEFF, dist, ticks)
        seconds = travel_ticks(speed, BALL_FRICTION_COEFF, dist) / TICK_RATE
        assert lo < seconds < hi, f"{dist} m pass arrives in {seconds:.2f}s"


def test_driven_pass_is_much_faster_than_an_ordinary_one():
    """The whole point of driving a ball: buy time against a closing defender."""
    dist = 15.0
    normal = PASS_TICKS_BASE + PASS_TICKS_PER_M * dist
    ordinary = speed_for_arrival(BALL_FRICTION_COEFF, dist, normal)
    driven = speed_for_arrival(BALL_FRICTION_COEFF, dist, normal * PASS_TICKS_DRIVEN)
    assert driven > ordinary * 1.5


def test_shot_arrival_is_realistic():
    """A 20 m shot is well under a second."""
    seconds = travel_ticks(SHOT_SPEED, SHOT_FRICTION_COEFF, 20.0) / TICK_RATE
    assert 0.4 < seconds < 0.9


def test_dribble_carry_is_short():
    assert 1.0 < carry_distance(DRIBBLE_SPEED, DRIBBLE_FRICTION_COEFF) < 3.0


def test_pass_beats_a_shot_over_the_same_ground():
    """A 20 m ground pass ~1.0-1.3 s; a 20 m shot well under 1 s.

    Band tightened when BALL_FRICTION_COEFF rose to 0.040. The old 1.2-1.6 s
    was a consequence of a near-frictionless ball: weighted_speed solves
    backwards from carry distance, so the only way to stop a pass at the
    receiver was to strike it feebly -- a 10 m pass left the foot at 9 m/s and
    took 1.83 s. Higher friction forces a harder strike for the same carry,
    which is the point of the change.
    """
    ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * 20.0
    pass_speed = speed_for_arrival(BALL_FRICTION_COEFF, 20.0, ticks)
    t_pass = travel_ticks(pass_speed, BALL_FRICTION_COEFF, 20.0) / TICK_RATE
    t_shot = travel_ticks(SHOT_SPEED, SHOT_FRICTION_COEFF, 20.0) / TICK_RATE
    assert 0.9 < t_pass < 1.6
    assert 0.5 < t_shot < 0.9
    assert t_shot < t_pass


def test_a_long_pass_is_reachable():
    """A 40 m ground pass must be possible -- it is a normal football action.

    Under the old geometric decay this test asserted the OPPOSITE for
    PASS_SPEED_LOW, because a slow ball genuinely died before 40 m. That was an
    artefact of the decay shape, not football.
    """
    ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * 40.0
    speed = speed_for_arrival(BALL_FRICTION_COEFF, 40.0, ticks)
    assert speed <= PASS_SPEED_MAX
    assert travel_ticks(speed, BALL_FRICTION_COEFF, 40.0) < float("inf")


# ── Weighted passes ──────────────────────────────────────────────────────────

def test_weighted_speed_lands_the_ball_where_intended():
    """A pass struck at a fixed speed carries the same distance whatever it was
    aimed at, so a 10 m ball overshot by 22 m and ran through the receiver to
    whoever stood behind them."""
    from app.engine.ball_physics import weighted_speed

    for dist in (8.0, 12.0, 18.0, 25.0, 32.0):
        for overrun in (2.0, 10.0):
            speed = weighted_speed(BALL_FRICTION_COEFF, dist, overrun, PASS_SPEED_HIGH)
            carry = carry_distance(speed, BALL_FRICTION_COEFF)
            assert carry == pytest.approx(dist + overrun, abs=0.5)


def test_weighted_speed_is_monotonic_in_distance():
    from app.engine.ball_physics import weighted_speed

    previous = 0.0
    for dist in (5.0, 10.0, 15.0, 20.0, 25.0):
        speed = weighted_speed(BALL_FRICTION_COEFF, dist, 5.0, PASS_SPEED_HIGH)
        assert speed > previous
        previous = speed


def test_weighted_speed_never_exceeds_the_maximum():
    """weighted_speed is retained for dribble touches, where the ball really is
    meant to die at a chosen point. It must never invent a speed the player
    does not have.

    It is no longer used for passes: under real rolling resistance a pass never
    stops on its own (200-2000 m of carry), so "weight it to settle at the
    receiver" degenerates -- a ball made to stop at 13 m had to be struck at
    3 m/s. Passes now choose an ARRIVAL TIME instead, because the ball stops
    when someone controls it. See speed_for_arrival and _weight_for_lane.
    """
    from app.engine.ball_physics import weighted_speed

    # A target far beyond what even the hardest strike can reach. It has to be
    # genuinely absurd now: at real friction the hardest ball carries 499 m
    # even with a dribble's deceleration, so 90 m no longer tests the branch.
    speed = weighted_speed(DRIBBLE_FRICTION_COEFF, 900.0, 100.0, PASS_SPEED_MAX)
    assert speed == pytest.approx(PASS_SPEED_MAX)
    # And it is bounded from above everywhere else too.
    assert weighted_speed(DRIBBLE_FRICTION_COEFF, 2.0, 1.0, PASS_SPEED_MAX) <= PASS_SPEED_MAX


def test_flight_time_is_what_the_passer_chooses():
    """The passer picks when the ball ARRIVES, not where it would stop.

    The ball stops when someone controls it -- capture zeroes its velocity --
    so weighting a pass to die at the receiver's feet was never needed for
    reception. It only made passes feeble: 10 m left the foot at 9 m/s and took
    1.83 s, which is why a press won the ball so easily.
    """
    for dist in (5.0, 15.0, 30.0):
        ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * dist
        speed = speed_for_arrival(BALL_FRICTION_COEFF, dist, ticks)
        actual = travel_ticks(speed, BALL_FRICTION_COEFF, dist)
        assert actual == pytest.approx(ticks, rel=0.02)


def test_a_pass_is_never_struck_slower_than_a_roll():
    """PASS_SPEED_MIN is the floor: below it the ball is not a pass at all."""
    for dist in (2.0, 5.0, 50.0):
        ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * dist
        speed = speed_for_arrival(BALL_FRICTION_COEFF, dist, ticks)
        assert speed > 0.0


def test_weighted_speed_bounded_everywhere():
    from app.engine.ball_physics import weighted_speed

    for dist in np.arange(1.0, 60.0, 3.0):
        speed = weighted_speed(BALL_FRICTION_COEFF, float(dist), 5.0, PASS_SPEED_HIGH)
        assert BALL_STOP_THRESHOLD <= speed <= PASS_SPEED_HIGH
