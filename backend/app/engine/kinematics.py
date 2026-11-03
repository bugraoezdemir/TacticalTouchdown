"""
Player inertia: one law for how a player's velocity can change, and the exact
answer it gives to "how soon can he get there".

The law. Each tick a player's velocity closes a fixed share K of the gap to
the velocity he WANTS (direction and pace, at most his top speed):

    v <- v + K * (desired - v),      K = 1 - exp(-1 / PLAYER_TAU)
    pos <- pos + v

so his velocity relaxes toward what he wants with time constant PLAYER_TAU.
Speeding up, braking and turning are all the same thing -- closing a gap in
velocity -- so each costs time in proportion to how big a change it is: a
strong first step that fades near top speed, a reversal that has to stop him
first, and a defender who must pay the same price to follow a cut.

The prediction. Wanting a constant velocity w for n ticks takes him to

    pos + v0 * S(n) + w * (n - S(n)),      S(n) = r * (1 - r^n) / K,  r = 1 - K

i.e. his momentum carries him to the centre pos + v0 * S(n), and from there he
can reach anything within top * (n - S(n)) by choosing w. So the earliest he
can be within `reach` of a point q is the first n with

    |q - pos - v0 * S(n)| <= top * (n - S(n)) + reach

which reach_ticks solves. It is exact for the movement in game_loop when the
player runs flat out at a constant aim -- and pursuit_velocity is that aim --
so what the decisions assume and what players do cannot drift apart.

Pure functions on numbers; game_loop owns when they are applied.
"""
from __future__ import annotations

import bisect
import math
import os

import numpy as np

from .constants import (
    ARRIVE_TAUS, BRAKE_TAU, CORNER_MIN_SHARE, GRIP_TOP_PER_S, PLAYER_TAU, TICK_RATE,
)

_EPS = 1e-9

# The A/B switch. Lives here, not in game_loop, so the decision modules can
# ask it without importing the loop. Off: movement is the old instant
# velocity and every reach estimate keeps its old formula. On: both follow the
# law above -- they must switch TOGETHER, or decisions plan for movement the
# players cannot make.
INERTIA = os.environ.get("FB_INERTIA", "0") != "0"
# Grip (needs INERTIA): turning and braking are limited by how hard a player
# can push against the ground, so a cut or a stop at a sprint costs far more
# than at a jog. Off: the plain law above.
GRIP = INERTIA and os.environ.get("FB_GRIP", "0") != "0"

_R = math.exp(-1.0 / PLAYER_TAU)     # share of a velocity gap left after a tick
_K = 1.0 - _R                        # share closed per tick
_S_INF = _R / _K                     # S(n) as n -> infinity
# Under grip, braking and turning close their part of the gap at BRAKE_TAU.
_KB_OVER_K = (1.0 - math.exp(-1.0 / BRAKE_TAU)) / _K


def drift(n):
    """S(n): how many ticks' worth of his current velocity his momentum still
    carries him over the next n ticks (n may be fractional or an array)."""
    return _R * (1.0 - np.exp(-np.asarray(n, dtype=float) / PLAYER_TAU)) / _K


def _drift(n: float) -> float:
    return _R * (1.0 - math.exp(-n / PLAYER_TAU)) / _K


def step_velocity(velocity: np.ndarray, desired: np.ndarray, top_speed: float) -> np.ndarray:
    """One tick of `velocity` relaxing toward `desired` (capped at top speed),
    and under GRIP with the turning and braking part of the change held to
    what his grip allows."""
    n = math.hypot(float(desired[0]), float(desired[1]))
    if n > top_speed > 0.0:
        desired = desired * (top_speed / n)
    dv = (desired - velocity) * _K
    if not GRIP:
        return velocity + dv
    return velocity + _grip_limited(velocity, dv, top_speed)


def step_velocity_rows(velocity: np.ndarray, desired: np.ndarray, top_speed: float) -> np.ndarray:
    """step_velocity over rows of (N, 2) arrays (desired already within top)."""
    dv = (desired - velocity) * _K
    if not GRIP:
        return velocity + dv
    return velocity + _grip_limited_arrays(velocity, dv, top_speed)


def _grip(top: float):
    """The most his grip can brake or turn him by in one tick (m/tick)."""
    return top * GRIP_TOP_PER_S / TICK_RATE


def _grip_limited(velocity: np.ndarray, dv: np.ndarray, top: float) -> np.ndarray:
    """`dv` (the plain law's change) with its braking and sideways part made
    quicker -- time constant BRAKE_TAU -- and then held within his grip.
    Pushing on in the direction he runs stays the plain law, which matches
    real sprint starts."""
    vx, vy = float(velocity[0]), float(velocity[1])
    speed = math.hypot(vx, vy)
    if speed < _EPS:
        return dv
    ux, uy = vx / speed, vy / speed
    dx, dy = float(dv[0]), float(dv[1])
    par = dx * ux + dy * uy
    px, py = (dx - par * ux) * _KB_OVER_K, (dy - par * uy) * _KB_OVER_K   # sideways
    brake = min(par, 0.0) * _KB_OVER_K
    push = max(par, 0.0)                         # unlimited
    c = math.sqrt(brake * brake + px * px + py * py)
    limit = _grip(top)
    if c > limit:
        f = limit / c
        brake, px, py = brake * f, px * f, py * f
    along = push + brake
    return np.array([along * ux + px, along * uy + py])


def _grip_limited_arrays(v: np.ndarray, dv: np.ndarray, top: float) -> np.ndarray:
    """_grip_limited over rows of (N, 2) arrays, for the table below."""
    speed = np.hypot(v[:, 0], v[:, 1])
    safe = np.where(speed > _EPS, speed, 1.0)
    u = v / safe[:, None]
    par = (dv * u).sum(axis=1)
    perp = (dv - par[:, None] * u) * _KB_OVER_K
    brake = np.minimum(par, 0.0) * _KB_OVER_K
    push = np.maximum(par, 0.0)
    c = np.sqrt(brake * brake + (perp * perp).sum(axis=1))
    limit = _grip(top)
    f = np.where(c > limit, limit / np.where(c > _EPS, c, 1.0), 1.0)
    out = (push + brake * f)[:, None] * u + perp * f[:, None]
    return np.where((speed > _EPS)[:, None], out, dv)


def desired_velocity(
    pos: np.ndarray,
    target: np.ndarray,
    cap: float,
    arrive: bool = True,
    target_velocity: np.ndarray | None = None,
    arrive_taus: float = ARRIVE_TAUS,
) -> np.ndarray:
    """Head for `target` at up to `cap`; if `arrive`, settle ON it.

    Arriving aims at gap / (arrive_taus x tau): it eases off as the gap
    closes, damped enough at the default that he stops on the spot instead of
    circling it; a presser uses a smaller value and brakes later.
    `target_velocity` is how the target itself is moving -- a marker's spot
    moves with his man -- so he matches that and only arrives on what is left.
    """
    offset = target - pos
    dist = math.hypot(float(offset[0]), float(offset[1]))
    base = np.zeros(2) if target_velocity is None else target_velocity
    if dist < _EPS:
        out = base.copy()
    else:
        speed = min(cap, dist / (arrive_taus * PLAYER_TAU)) if arrive else cap
        out = base + offset / dist * speed
    n = math.hypot(float(out[0]), float(out[1]))
    return out / n * cap if n > cap else out


def pursuit_velocity(
    pos: np.ndarray, vel: np.ndarray, target: np.ndarray, cap: float,
    reach: float = 0.0,
) -> np.ndarray:
    """Flat out for `target`, aimed so as to get there SOONEST: allow for where
    his momentum is already carrying him (the constant aim reach_ticks
    assumes). Aiming straight at the target from a sideways run instead
    curves him round and arrives late -- the gap between what a pass was
    planned on and what the receiver did. `reach`: he only has to get within
    it (it sets how hard he must corner under grip)."""
    n = float(_reach_ticks_linear(pos, vel, cap, target))
    aim = target - (pos + vel * _drift(n))
    d = math.hypot(float(aim[0]), float(aim[1]))
    if d < _EPS:
        aim = target - pos
        d = math.hypot(float(aim[0]), float(aim[1]))
        if d < _EPS:
            return np.zeros(2)
    if GRIP:
        # (Aiming straight at a point ahead, to spend the grip on turning
        # rather than on the braking this aim asks for, was tried: receivers
        # then came late to 34% of passes instead of 20%.)
        cap = min(cap, corner_speed(float(vel[0]), float(vel[1]),
                                    float(target[0] - pos[0]), float(target[1] - pos[1]),
                                    cap, reach))
    return aim / d * cap


def corner_speed(vx: float, vy: float, ox: float, oy: float, top: float,
                 reach: float = 0.0) -> float:
    """Under grip, the fastest he should run at a point `o` ahead of him and
    off to one side, to come within `reach` of it: the speed at which his
    grip can still bend his run round onto it. With the point `ahead` along
    his run and `side` off it (less his reach), the circle that leaves along
    his run and passes through it has radius (ahead^2 + side^2) / (2 side);
    his grip turns him on a circle of radius speed^2 / grip. Any faster and
    he sails past and has to come back -- so he eases off, the way a player
    slows into a turn. A point he will pass within reach of, or one behind
    him (he has to brake anyway), sets no limit."""
    sp = math.hypot(vx, vy)
    if sp < _EPS:
        return top
    ahead = (vx * ox + vy * oy) / sp
    side = abs(vx * oy - vy * ox) / sp - reach
    if ahead <= 0.0 or side <= 0.0:
        return top
    radius = (ahead * ahead + side * side) / (2.0 * side)
    return min(top, max(CORNER_MIN_SHARE * top, math.sqrt(_grip(top) * radius)))


def _corner_speed_arrays(v: np.ndarray, o: np.ndarray, top: float,
                         reach: np.ndarray) -> np.ndarray:
    """corner_speed over rows of (N, 2) arrays, for the table below."""
    sp = np.hypot(v[:, 0], v[:, 1])
    safe = np.maximum(sp, _EPS)
    ahead = (v * o).sum(axis=1) / safe
    side = np.abs(v[:, 0] * o[:, 1] - v[:, 1] * o[:, 0]) / safe - reach
    radius = (ahead * ahead + side * side) / (2.0 * np.maximum(side, _EPS))
    limit = np.clip(np.sqrt(_grip(top) * radius), CORNER_MIN_SHARE * top, top)
    free = (sp < _EPS) | (ahead <= 0.0) | (side <= 0.0)
    return np.where(free, top, limit)


def reach_distance(top: float, ticks: float, v0: float = 0.0) -> float:
    """How far along a line a player moving at `v0` along it covers in `ticks`,
    flat out: momentum v0 * S(n) plus what his effort adds, top * (n - S(n))."""
    s = _drift(max(0.0, ticks))
    return float(v0 * s + max(top, 0.0) * (ticks - s))


# Where the first-crossing search looks: every tick early on, then wider steps.
_GRID = np.unique(np.concatenate([
    np.arange(0.0, 16.0, 1.0),
    np.round(16.0 * 1.2 ** np.arange(0, 25)),
]))
_GRID = _GRID[_GRID <= 600.0]
_GRID_S = drift(_GRID)
_GRID_L = _GRID.tolist()
_GRID_SL = _GRID_S.tolist()

# The solver in Rust (backend/fb_core): the same arithmetic as the Python below,
# which stays as the reference and the fallback. FB_RUST unset: used if built;
# 1: required; 0: off.
_rust = None
if os.environ.get("FB_RUST", "") != "0":
    try:
        # By name: run from backend/, an unbuilt backend/fb_core folder still
        # imports (as an empty namespace package), and must count as not built.
        from fb_core import Reach
    except ImportError:
        if os.environ.get("FB_RUST") == "1":
            raise
    else:
        _rust = Reach(_R, _K, float(PLAYER_TAU), _GRID_L, _GRID_SL)
        from .constants import rust_constants
        _rust.set_constants(rust_constants())
# Halvings of the bracket: within 1/256 tick early on, ~0.1 tick at 100 ticks.
# (An Illinois secant search was tried in their place, 2026-10-27: three steps
# matched this to 1e-5 tick but cost the same -- each step's np.where calls on
# small arrays outweigh the arithmetic -- so this stays.)
_BISECT = 8


def _reach_ticks_linear(
    pos: np.ndarray,
    vel: np.ndarray,
    top: np.ndarray | float,
    point: np.ndarray,
    reach: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Ticks until a player at `pos` moving at `vel` can be within `reach` of
    `point`, flat out at up to `top`. Exact under the law above.

    The FIRST such moment: a player running through the point counts as
    reaching it, even if he would be carried past it. Broadcasts over any
    leading shape -- callers pass (M, 1, 2) points against (1, N, 2) players
    for an (M, N) table.
    """
    if (
        np.ndim(point) == 1 and np.ndim(pos) == 1 and np.ndim(vel) == 1
        and np.ndim(top) == 0 and np.ndim(reach) == 0
    ):
        return np.float64(_reach_ticks_scalar_linear(
            float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1]),
            float(top), float(point[0]), float(point[1]), float(reach),
        ))
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    point = np.asarray(point, dtype=float)
    if _rust is not None:
        return _rust.many(pos, vel, np.asarray(top, dtype=float), point,
                          np.asarray(reach, dtype=float), False)
    off = point - pos
    shape = np.broadcast_shapes(off.shape[:-1], vel.shape[:-1], np.shape(top), np.shape(reach))
    off = np.broadcast_to(off, shape + (2,))
    vel = np.broadcast_to(vel, shape + (2,))
    top = np.maximum(np.broadcast_to(np.asarray(top, dtype=float), shape), _EPS)
    reach = np.broadcast_to(np.asarray(reach, dtype=float), shape)

    # How far the point is still out of reach after n ticks (<= 0: reached),
    # first on the whole grid -- an extra last axis -- then per element.
    gx = off[..., 0, None] - vel[..., 0, None] * _GRID_S
    gy = off[..., 1, None] - vel[..., 1, None] * _GRID_S
    ok = np.hypot(gx, gy) - top[..., None] * (_GRID - _GRID_S) - reach[..., None] <= 0.0
    found = ok.any(axis=-1)
    first = np.argmax(ok, axis=-1)
    hi = _GRID[first]
    lo = _GRID[np.maximum(first - 1, 0)]
    def gap(n):
        s = drift(n)
        return (np.hypot(off[..., 0] - vel[..., 0] * s, off[..., 1] - vel[..., 1] * s)
                - top * (n - s) - reach)

    for _ in range(_BISECT):
        mid = 0.5 * (lo + hi)
        inside = gap(mid) <= 0.0
        hi = np.where(inside, mid, hi)
        lo = np.where(inside, lo, mid)
    # Then one straight-line step between the bracket ends, which the smooth
    # gap makes all but exact.
    g_lo, g_hi = gap(lo), gap(hi)
    span = g_lo - g_hi
    root = np.where(span > _EPS, lo + (hi - lo) * g_lo / np.where(span > _EPS, span, 1.0), hi)
    out = np.where(first == 0, 0.0, np.clip(root, lo, hi))
    # Beyond the grid his momentum is spent: solve the straight-line remainder.
    centre = np.hypot(off[..., 0] - vel[..., 0] * _S_INF, off[..., 1] - vel[..., 1] * _S_INF)
    late = _S_INF + np.maximum(0.0, centre - reach) / top
    return np.where(found, out, np.maximum(late, _GRID[-1]))


def _reach_ticks_scalar_linear(
    px: float, py: float, vx: float, vy: float,
    top: float, qx: float, qy: float, reach: float,
) -> float:
    """reach_ticks for one player and one point, in plain floats (the hot
    case -- numpy's per-call overhead dominated it). Pinned equal to the array
    version in tests/test_kinematics.py."""
    if _rust is not None:
        return _rust.linear(px, py, vx, vy, top, qx, qy, reach)
    top = max(top, _EPS)
    ox, oy = qx - px, qy - py
    if math.hypot(ox, oy) <= reach:
        return 0.0
    def gap(n: float) -> float:
        s = _drift(n)
        return math.hypot(ox - vx * s, oy - vy * s) - top * (n - s) - reach

    prev = 0.0
    for n, s in zip(_GRID_L, _GRID_SL):
        if math.hypot(ox - vx * s, oy - vy * s) - top * (n - s) - reach <= 0.0:
            lo, hi = prev, n
            for _ in range(_BISECT):
                mid = 0.5 * (lo + hi)
                if gap(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            g_lo, g_hi = gap(lo), gap(hi)
            span = g_lo - g_hi
            root = lo + (hi - lo) * g_lo / span if span > _EPS else hi
            return min(max(root, lo), hi)
        prev = n
    centre = math.hypot(ox - vx * _S_INF, oy - vy * _S_INF)
    return max(_S_INF + max(0.0, centre - reach) / top, _GRID_L[-1])


# ── Grip: the prediction ─────────────────────────────────────────────────────
# Under grip the law is no longer linear, so "how soon" has no closed form. The
# answer is the plain law's exact one plus what grip costs, and that extra is a
# function of three numbers only once distances are in units of his top speed:
# how fast he is going (share of top), the angle between his run and the
# target, and how far the target is beyond his reach. It is tabulated once by
# running the real controller -- pursuit_velocity, re-aimed every tick, through
# step_velocity -- with and without grip, so the table holds exactly what grip
# changes in the movement the game plays.
_GT_S = np.linspace(0.0, 1.0, 11)
_GT_TH = np.linspace(0.0, math.pi, 13)
_GT_D = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512.0])
# Reach in top-ticks: 1 m is ~4 for an outfield player. Without it on its
# own axis a wide reach looked like a pin-point target, which is far harder
# to hit when running past.
_GT_R = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 12.0])
_GT_MAX_TICKS = 900
_GT_VERSION = 6
_grip_table: np.ndarray | None = None
_grip_flat: list[float] = []      # the same, flat, for the scalar path
_grip_np_flat = np.zeros(0)
# Offsets of a cell's sixteen corners in the flat table.
_GT_CORNERS = (np.arange(2)[:, None, None, None] * (_GT_TH.size * _GT_D.size * _GT_R.size)
               + np.arange(2)[None, :, None, None] * (_GT_D.size * _GT_R.size)
               + np.arange(2)[None, None, :, None] * _GT_R.size
               + np.arange(2)[None, None, None, :])
_GT_DL, _GT_RL = _GT_D.tolist(), _GT_R.tolist()
_GT_S_STEP = float(_GT_S[1] - _GT_S[0])
_GT_TH_STEP = float(_GT_TH[1] - _GT_TH[0])


def _table_runs(grip: bool) -> np.ndarray:
    """Ticks for every (speed share, angle, distance, reach) cell, top 1."""
    s, th, d, r = np.meshgrid(_GT_S, _GT_TH, _GT_D, _GT_R, indexing="ij")
    s, th, d, r = s.ravel(), th.ravel(), d.ravel(), r.ravel()
    n_runs = s.size
    q = np.stack([np.cos(th), np.sin(th)], axis=1) * (d + r)[:, None]
    pos = np.zeros((n_runs, 2))
    vel = np.stack([s, np.zeros(n_runs)], axis=1)
    gap_prev = d.copy()
    out = np.where(d <= 0.0, 0.0, np.nan)
    live = d > 0.0
    for n in range(1, _GT_MAX_TICKS + 1):
        if not live.any():
            break
        idx = np.nonzero(live)[0]
        p, v, qq, rr = pos[idx], vel[idx], q[idx], r[idx]
        # pursuit_velocity, row by row
        m = _reach_ticks_linear(p, v, 1.0, qq)
        aim = qq - (p + v * drift(m)[:, None])
        norm = np.hypot(aim[:, 0], aim[:, 1])
        direct = qq - p
        bad = norm < _EPS
        aim[bad] = direct[bad]
        norm = np.where(bad, np.hypot(aim[:, 0], aim[:, 1]), norm)
        cap = _corner_speed_arrays(v, direct, 1.0, rr) if grip else np.ones(idx.size)
        desired = aim / np.maximum(norm, _EPS)[:, None] * cap[:, None]
        dv = (desired - v) * _K
        if grip:
            dv = _grip_limited_arrays(v, dv, 1.0)
        v = v + dv
        p = p + v
        vel[idx], pos[idx] = v, p
        gap = np.hypot(qq[:, 0] - p[:, 0], qq[:, 1] - p[:, 1]) - rr
        hit = gap <= 0.0
        if hit.any():
            g0 = gap_prev[idx][hit]
            frac = g0 / np.maximum(g0 - gap[hit], _EPS)
            out[idx[hit]] = n - 1 + np.clip(frac, 0.0, 1.0)
            live[idx[hit]] = False
        gap_prev[idx] = gap
    out = np.where(np.isnan(out), _GT_MAX_TICKS, out)
    return out.reshape(_GT_S.size, _GT_TH.size, _GT_D.size, _GT_R.size)


def _table_path() -> str:
    import hashlib
    import tempfile
    key = repr((_GT_VERSION, PLAYER_TAU, BRAKE_TAU, GRIP_TOP_PER_S, CORNER_MIN_SHARE,
                _GT_S.tolist(), _GT_TH.tolist(), _GT_D.tolist(), _GT_R.tolist()))
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"fb_grip_table_{tag}.npy")


def grip_table() -> np.ndarray:
    """Extra ticks grip costs, indexed [speed share, angle, distance, reach].
    Built once (a few seconds) and kept on disk keyed by the constants."""
    global _grip_table, _grip_flat, _grip_np_flat
    if _grip_table is None:
        path = _table_path()
        try:
            _grip_table = np.load(path)
        except (OSError, ValueError):
            _grip_table = _table_runs(True) - _table_runs(False)
            try:
                tmp = f"{path}.{os.getpid()}.npy"
                np.save(tmp, _grip_table)
                os.replace(tmp, path)
            except OSError:
                pass
        _grip_np_flat = _grip_table.ravel()
        _grip_flat = _grip_np_flat.tolist()
        if _rust is not None:
            _rust.set_table(_grip_np_flat.copy(), _GT_S.size, _GT_TH.size, _GT_S_STEP,
                            _GT_TH_STEP, _GT_DL, _GT_RL)
    return _grip_table


def _axis(grid: np.ndarray, x):
    """Lower index and weight for linear interpolation on `grid` (clamped)."""
    f = np.minimum(np.interp(x, grid, np.arange(grid.size, dtype=float)), grid.size - 1.000001)
    i = f.astype(np.intp)
    return i, f - i


def _grip_extra(s, th, d, r):
    """Interpolated table value over arrays of equal shape: the sixteen
    corners of each cell gathered in one go."""
    grip_table()
    s, th, d, r = np.broadcast_arrays(s, th, d, r)
    shape = s.shape
    ns, nt, nd, nr = _GT_S.size, _GT_TH.size, _GT_D.size, _GT_R.size
    x = np.clip(s.ravel() / _GT_S_STEP, 0.0, ns - 1.000001)
    i = x.astype(np.intp); wi = x - i
    y = np.clip(th.ravel() / _GT_TH_STEP, 0.0, nt - 1.000001)
    j = y.astype(np.intp); wj = y - j
    k, wk = _axis(_GT_D, d.ravel())
    m, wm = _axis(_GT_R, r.ravel())
    base = ((i * nt + j) * nd + k) * nr + m
    w = (np.stack([1.0 - wi, wi])[:, None, None, None]
         * np.stack([1.0 - wj, wj])[None, :, None, None]
         * np.stack([1.0 - wk, wk])[None, None, :, None]
         * np.stack([1.0 - wm, wm])[None, None, None, :])
    vals = _grip_np_flat[base + _GT_CORNERS[..., None]]
    return (vals * w).sum(axis=(0, 1, 2, 3)).reshape(shape)


def _cell(grid: list[float], x: float) -> tuple[int, float]:
    """_axis for one value on a short, non-uniform grid."""
    if x <= grid[0]:
        return 0, 0.0
    last = len(grid) - 2
    if x >= grid[-1]:
        return last, 1.0
    i = bisect.bisect_right(grid, x) - 1
    if i > last:
        i = last
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


def _grip_extra_scalar(s: float, th: float, d: float, r: float) -> float:
    """_grip_extra for one set of floats, in plain Python (the hot case)."""
    if not _grip_flat:
        grip_table()
    t = _grip_flat
    ns, nt, nd, nr = _GT_S.size, _GT_TH.size, _GT_D.size, _GT_R.size
    x = min(max(s / _GT_S_STEP, 0.0), ns - 1.000001)
    i = int(x); wi = x - i
    y = min(max(th / _GT_TH_STEP, 0.0), nt - 1.000001)
    j = int(y); wj = y - j
    k, wk = _cell(_GT_DL, d)
    m, wm = _cell(_GT_RL, r)
    out = 0.0
    for a, wa in ((i, 1.0 - wi), (i + 1, wi)):
        if wa == 0.0:
            continue
        for b, wb in ((j, 1.0 - wj), (j + 1, wj)):
            if wb == 0.0:
                continue
            base = (a * nt + b) * nd
            wab = wa * wb
            for c, wc in ((k, 1.0 - wk), (k + 1, wk)):
                if wc == 0.0:
                    continue
                row = (base + c) * nr
                w = wab * wc
                out += w * ((1.0 - wm) * t[row + m] + wm * t[row + m + 1])
    return out


def _grip_args(px, py, vx, vy, top, qx, qy, reach):
    """(speed share, angle to target, distance beyond reach), top-scaled."""
    top = np.maximum(top, _EPS)
    sp = np.hypot(vx, vy)
    ox, oy = qx - px, qy - py
    dist = np.hypot(ox, oy)
    cos = (vx * ox + vy * oy) / np.maximum(sp * dist, _EPS)
    th = np.where(sp * dist > _EPS, np.arccos(np.clip(cos, -1.0, 1.0)), 0.0)
    return (np.minimum(sp / top, 1.0), th, np.maximum(dist - reach, 0.0) / top,
            reach / top)


def reach_ticks(
    pos: np.ndarray,
    vel: np.ndarray,
    top: np.ndarray | float,
    point: np.ndarray,
    reach: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Ticks until a player at `pos` moving at `vel` can be within `reach` of
    `point`, flat out at up to `top` (see _reach_ticks_linear). Under GRIP,
    plus what grip costs him (the table above)."""
    if not GRIP:
        return _reach_ticks_linear(pos, vel, top, point, reach)
    if (
        np.ndim(point) == 1 and np.ndim(pos) == 1 and np.ndim(vel) == 1
        and np.ndim(top) == 0 and np.ndim(reach) == 0
    ):
        return np.float64(_reach_ticks_scalar(
            float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1]),
            float(top), float(point[0]), float(point[1]), float(reach),
        ))
    if _rust is not None:
        if not _grip_flat:
            grip_table()
        return _rust.many(np.asarray(pos, dtype=float), np.asarray(vel, dtype=float),
                          np.asarray(top, dtype=float), np.asarray(point, dtype=float),
                          np.asarray(reach, dtype=float), True)
    base = _reach_ticks_linear(pos, vel, top, point, reach)
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    point = np.asarray(point, dtype=float)
    extra = _grip_extra(*_grip_args(
        pos[..., 0], pos[..., 1], vel[..., 0], vel[..., 1],
        np.asarray(top, dtype=float), point[..., 0], point[..., 1],
        np.asarray(reach, dtype=float)))
    return np.where(base > 0.0, base + extra, base)


def _reach_ticks_scalar(
    px: float, py: float, vx: float, vy: float,
    top: float, qx: float, qy: float, reach: float,
) -> float:
    """reach_ticks for one player and one point, in plain floats."""
    if _rust is not None and GRIP:
        if not _grip_flat:
            grip_table()
        return _rust.ticks(px, py, vx, vy, top, qx, qy, reach)
    base = _reach_ticks_scalar_linear(px, py, vx, vy, top, qx, qy, reach)
    if not GRIP or base <= 0.0:
        return base
    top = max(top, _EPS)
    sp = math.hypot(vx, vy)
    ox, oy = qx - px, qy - py
    dist = math.hypot(ox, oy)
    if sp * dist > _EPS:
        th = math.acos(max(-1.0, min(1.0, (vx * ox + vy * oy) / (sp * dist))))
    else:
        th = 0.0
    return base + _grip_extra_scalar(min(sp / top, 1.0), th, max(dist - reach, 0.0) / top,
                                     reach / top)
