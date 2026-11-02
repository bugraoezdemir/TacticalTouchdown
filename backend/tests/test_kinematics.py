"""
Player inertia: velocity relaxes toward the velocity a player wants, with time
constant PLAYER_TAU (app/engine/kinematics.py).

These pin the law's figures, the exactness of reach_ticks against the
movement it predicts (with the pursuit steering game_loop uses), and the two
steering properties that keep inertia honest: a player stops ON a fixed spot
rather than orbiting it, and follows a MOVING spot without a trailing gap.
"""
import math

import numpy as np
import pytest

from app.engine.constants import PLAYER_TAU, TICK_RATE
from app.engine import kinematics
from app.engine.kinematics import (
    desired_velocity,
    pursuit_velocity,
    reach_distance,
    reach_ticks,
    step_velocity,
)

TOP = 0.20  # m/tick, 6 m/s -- PLAYER_BASE_SPEED


def _speed(v):
    return math.hypot(float(v[0]), float(v[1]))


def test_from_rest_reaches_63_percent_of_top_speed_in_one_tau():
    v = np.zeros(2)
    for _ in range(int(round(PLAYER_TAU))):
        v = step_velocity(v, np.array([TOP, 0.0]), TOP)
    assert 0.60 * TOP <= _speed(v) <= 0.66 * TOP


def test_never_exceeds_top_speed():
    v = np.zeros(2)
    for _ in range(400):
        v = step_velocity(v, np.array([1.0, 0.0]), TOP)
    assert _speed(v) <= TOP + 1e-12


def test_a_reversal_costs_time():
    """Flat out one way, then wanting the other: he must stop first, after
    tau * ln 2, and only then go back -- the price of being wrong-footed."""
    v = np.array([TOP, 0.0])
    ticks = 0
    while float(v[0]) > 0.0:
        v = step_velocity(v, np.array([-TOP, 0.0]), TOP)
        ticks += 1
    assert abs(ticks - PLAYER_TAU * math.log(2.0)) <= 1.0
    assert 0.35 <= ticks / TICK_RATE <= 0.5


def test_arrives_on_a_fixed_spot_without_orbiting_it():
    pos = np.zeros(2)
    v = np.zeros(2)
    target = np.array([10.0, 0.0])
    furthest = 0.0
    for _ in range(300):
        v = step_velocity(v, desired_velocity(pos, target, TOP), TOP)
        pos = pos + v
        furthest = max(furthest, float(pos[0]))
    assert furthest - 10.0 < 0.5          # no real overrun
    assert _speed(v) < 0.01               # settled
    assert abs(float(pos[0]) - 10.0) < 0.1


def test_follows_a_moving_spot_without_an_invented_gap():
    spot_v = np.array([0.6 * TOP, 0.0])

    def run(match_motion):
        spot = np.array([2.0, 0.0])
        pos = np.zeros(2)
        v = spot_v.copy()
        gaps = []
        for _ in range(90):
            tv = spot_v if match_motion else None
            v = step_velocity(v, desired_velocity(pos, spot, TOP, target_velocity=tv), TOP)
            pos = pos + v
            spot = spot + spot_v
            gaps.append(float(spot[0] - pos[0]))
        return gaps

    matched = run(True)
    assert matched[-1] < 0.5               # closes onto the spot and stays
    assert run(False)[-1] > matched[-1]    # without matching it trails


# ── reach_ticks: the prediction every reach estimate uses under inertia ──────

def _simulate_reach(pos, vel, top, point, reach, cap=2000):
    """What game_loop does for a player sent flat out at a point."""
    t = 0
    while math.hypot(*(point - pos)) > reach and t < cap:
        vel = step_velocity(vel, pursuit_velocity(pos, vel, point, top), top)
        pos = pos + vel
        t += 1
    return t


def test_reach_ticks_matches_the_simulated_movement():
    """The licence for using reach_ticks in the decision layer: from moving
    starts in any heading, the player steered by pursuit_velocity gets there
    when reach_ticks says (to within the tick the crossing falls in)."""
    rng = np.random.default_rng(7)
    errors = []
    for _ in range(300):
        top = rng.uniform(0.19, 0.29)
        d = rng.uniform(1.5, 40.0)
        ang = rng.uniform(-math.pi, math.pi)
        vel = np.array([math.cos(ang), math.sin(ang)]) * top * rng.uniform(0.0, 1.0)
        point = np.array([d, 0.0])
        sim = _simulate_reach(np.zeros(2), vel.copy(), top, point, 1.0)
        pred = float(reach_ticks(np.zeros(2), vel, top, point, 1.0))
        errors.append(sim - pred)
    errors = np.array(errors)
    assert np.all(errors >= -1e-6)                 # never sooner than predicted
    assert np.percentile(errors, 95) <= 1.0        # and within the tick


def test_reach_ticks_inverts_reach_distance():
    for ticks in (5.0, 10.0, 30.0, 90.0):
        for v0 in (0.0, 0.5 * TOP, TOP):
            d = reach_distance(TOP, ticks, v0)
            t = float(reach_ticks(np.zeros(2), np.array([v0, 0.0]), TOP, np.array([d, 0.0])))
            assert t == pytest.approx(ticks, abs=1e-2)


def test_moving_away_costs_more_than_standing_still():
    point = np.array([10.0, 0.0])
    still = float(reach_ticks(np.zeros(2), np.zeros(2), TOP, point))
    away = float(reach_ticks(np.zeros(2), np.array([-TOP, 0.0]), TOP, point))
    toward = float(reach_ticks(np.zeros(2), np.array([TOP, 0.0]), TOP, point))
    assert toward < still < away


def test_running_through_a_point_counts_as_reaching_it():
    """Sprinting along a line past a point: he reaches it as he passes, not
    after stopping and coming back."""
    t = float(reach_ticks(np.zeros(2), np.array([TOP, 0.0]), TOP, np.array([2.0, 0.0])))
    assert t == pytest.approx(2.0 / TOP, abs=0.5)


def test_reach_ticks_broadcasts_points_against_players():
    players = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    vels = np.array([[0.1, 0.0], [0.0, -0.15], [-0.2, 0.05]])
    points = np.array([[1.0, 1.0], [20.0, 0.0]])
    table = reach_ticks(players[None, :, :], vels[None, :, :], np.full(3, TOP),
                        points[:, None, :], 1.0)
    assert table.shape == (2, 3)
    for i, pt in enumerate(points):
        for j, p in enumerate(players):
            assert table[i, j] == pytest.approx(float(reach_ticks(p, vels[j], TOP, pt, 1.0)))


def test_single_point_fast_path_equals_the_array_version():
    rng = np.random.default_rng(3)
    for _ in range(500):
        pos = rng.uniform(0.0, 60.0, 2)
        vel = rng.uniform(-0.3, 0.3, 2)
        top = float(rng.uniform(0.1, 0.35))
        point = rng.uniform(0.0, 60.0, 2)
        reach = float(rng.choice([0.0, 0.5, 1.0]))
        fast = float(reach_ticks(pos, vel, top, point, reach))
        slow = float(reach_ticks(pos[None, :], vel[None, :], np.array([top]), point[None, :], reach)[0])
        assert fast == pytest.approx(slow, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("inertia", [False, True])
def test_batched_lanes_equal_the_scalar_ones(monkeypatch, inertia):
    """route_survival_many is the fast path for the off-ball sampler; it must
    give the per-opponent rules' numbers under either movement model. The
    reference is built from opponent_risk directly."""
    from app.engine.decisions.interception import opponent_risk, route_survival_many
    from app.engine.ball_physics import nominal_pass_speed
    from app.engine.constants import BALL_FRICTION_COEFF
    from app.engine.factory import build_test_state
    import app.engine.game_loop as gl

    monkeypatch.setattr(kinematics, "INERTIA", inertia)
    state = build_test_state(seed=3)
    for _ in range(150):           # real positions AND real velocities
        gl.tick(state)
    origin = state.ball.pos.copy()
    opponents = state.away_team.players
    rng = np.random.default_rng(1)
    targets = origin + rng.uniform(-25.0, 25.0, size=(24, 2))
    paces = np.array([nominal_pass_speed(BALL_FRICTION_COEFF, float(np.hypot(*(t - origin))))
                      for t in targets])
    batched = route_survival_many(origin, targets, paces, BALL_FRICTION_COEFF, opponents)
    scalar = [1.0 - max(opponent_risk(o, origin, t, float(s), BALL_FRICTION_COEFF)
                        for o in opponents if o.speed >= 1e-9)
              for t, s in zip(targets, paces)]
    assert np.allclose(batched, scalar, atol=1e-9)


def test_desired_velocity_never_exceeds_the_cap():
    out = desired_velocity(np.zeros(2), np.array([5.0, 0.0]), TOP,
                           target_velocity=np.array([TOP, TOP]))
    assert _speed(out) <= TOP + 1e-12
