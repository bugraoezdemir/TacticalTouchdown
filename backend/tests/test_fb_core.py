"""The Rust solver gives the Python one's answers, on inputs like the game's."""
import numpy as np
import pytest

from app.engine import kinematics as k

fb_core = pytest.importorskip("fb_core")
TOLERANCE = 1e-4     # ticks; the array and scalar Python differ by 2e-5 at the table's edge
k.grip_table()


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0.0, 0.0], [105.0, 68.0], (n, 2))
    point = rng.uniform([0.0, 0.0], [105.0, 68.0], (n, 2))
    near = rng.random(n) < 0.5
    point[near] = pos[near] + rng.normal(0.0, 4.0, (int(near.sum()), 2))
    still = rng.random(n) < 0.1
    speed = rng.uniform(0.0, 0.35, n)
    speed[still] = 0.0
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]
    top = rng.uniform(0.15, 0.35, n)
    reach = rng.choice([0.0, 0.5, 1.0, 2.0], n)
    return pos, vel, top, point, reach


@pytest.fixture
def python_only(monkeypatch):
    def run(fn, *args):
        rust = k._rust
        monkeypatch.setattr(k, "_rust", None)
        try:
            return fn(*args)
        finally:
            monkeypatch.setattr(k, "_rust", rust)
    return run


@pytest.mark.parametrize("grip", [False, True])
def test_arrays_match_python(monkeypatch, python_only, grip):
    monkeypatch.setattr(k, "GRIP", grip)
    args = _inputs(50_000)
    theirs = python_only(k.reach_ticks, *args)
    ours = k.reach_ticks(*args)
    assert np.abs(ours - theirs).max() < TOLERANCE


@pytest.mark.parametrize("grip", [False, True])
def test_single_points_match_python(monkeypatch, python_only, grip):
    monkeypatch.setattr(k, "GRIP", grip)
    pos, vel, top, point, reach = _inputs(2_000, seed=1)
    for i in range(len(pos)):
        args = (pos[i], vel[i], float(top[i]), point[i], float(reach[i]))
        assert abs(float(k.reach_ticks(*args)) - float(python_only(k.reach_ticks, *args))) < TOLERANCE


def test_it_broadcasts_like_numpy(monkeypatch, python_only):
    monkeypatch.setattr(k, "GRIP", True)
    pos, vel, top, point, reach = _inputs(40, seed=2)
    points = point[:7, None, :]
    players = pos[None, :, :]
    theirs = python_only(k.reach_ticks, players, vel[None, :, :], top[None, :], points, 1.5)
    ours = k.reach_ticks(players, vel[None, :, :], top[None, :], points, 1.5)
    assert ours.shape == (7, 40)
    assert np.abs(ours - theirs).max() < TOLERANCE


@pytest.mark.parametrize("grip", [False, True])
def test_lanes_match_python(monkeypatch, python_only, grip):
    from app.engine.constants import BALL_FRICTION_COEFF
    from app.engine.decisions import interception
    from app.engine.factory import build_test_state

    monkeypatch.setattr(k, "INERTIA", True)
    monkeypatch.setattr(k, "GRIP", grip)
    state = build_test_state(seed=4)
    opponents = state.away_team.players
    rng = np.random.default_rng(5)
    for p in opponents:
        p.pos = rng.uniform([0.0, 0.0], [105.0, 68.0])
        p.velocity = rng.normal(0.0, 0.12, 2)
    origin = np.array([40.0, 30.0])
    targets = rng.uniform([0.0, 0.0], [105.0, 68.0], (200, 2))
    speeds = rng.uniform(0.5, 1.6, 200)
    delays = rng.uniform(0.0, 60.0, 200)
    args = (origin, targets, speeds, BALL_FRICTION_COEFF, opponents, delays)
    ours = interception.route_survival_many(*args)
    theirs = python_only(interception.route_survival_many, *args)
    assert np.abs(ours - theirs).max() < TOLERANCE


def test_every_off_ball_spot_matches_python_in_play(monkeypatch):
    from app.engine.decisions import off_ball as ob
    from app.engine.factory import build_test_state
    from app.engine.game_loop import tick

    monkeypatch.setattr(k, "INERTIA", True)
    monkeypatch.setattr(k, "GRIP", True)
    rust = k._rust
    original = ob._in_possession
    compared = []

    def both(player, state, team, committed=None):
        last = player.last_off_ball_target
        ours = original(player, state, team, committed)
        chosen = player.last_off_ball_target
        player.last_off_ball_target = last
        k._rust = None
        theirs = original(player, state, team, committed)
        k._rust = rust
        player.last_off_ball_target = chosen
        compared.append(np.hypot(*(ours.target_pos - theirs.target_pos)))
        return ours

    monkeypatch.setattr(ob, "_in_possession", both)
    state = build_test_state(seed=1)
    for _ in range(120):
        tick(state)
    assert len(compared) > 300
    assert max(compared) < 1e-6
