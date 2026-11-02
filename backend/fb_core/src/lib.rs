//! The engine's hot arithmetic in Rust. Each function mirrors the Python one
//! named in its comment; the Python stays as the reference and the fallback,
//! and tests/test_fb_core.py pins the two together.
use std::collections::HashMap;
use std::sync::OnceLock;

use numpy::ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const EPS: f64 = 1e-9;
const BISECT: usize = 8; // kinematics._BISECT
const TOP_EDGE: f64 = 1.000001; // keeps an index below the last cell

/// kinematics.grip_table, flat, with what is needed to look into it.
struct GripTable {
    flat: Vec<f64>,
    ns: usize,
    nt: usize,
    s_step: f64,
    th_step: f64,
    d: Vec<f64>,
    r: Vec<f64>,
}

impl GripTable {
    /// kinematics._cell: lower index and weight on a short, non-uniform grid.
    #[inline]
    fn cell(grid: &[f64], x: f64) -> (usize, f64) {
        if x <= grid[0] {
            return (0, 0.0);
        }
        let last = grid.len() - 2;
        if x >= grid[last + 1] {
            return (last, 1.0);
        }
        let i = grid.partition_point(|&g| g <= x) - 1;
        let i = i.min(last);
        (i, (x - grid[i]) / (grid[i + 1] - grid[i]))
    }

    /// kinematics._grip_extra_scalar.
    #[inline]
    fn extra(&self, s: f64, th: f64, d: f64, r: f64) -> f64 {
        let (nt, nd, nr) = (self.nt, self.d.len(), self.r.len());
        let x = (s / self.s_step).max(0.0).min(self.ns as f64 - TOP_EDGE);
        let i = x as usize;
        let wi = x - i as f64;
        let y = (th / self.th_step).max(0.0).min(nt as f64 - TOP_EDGE);
        let j = y as usize;
        let wj = y - j as f64;
        let (k, wk) = Self::cell(&self.d, d);
        let (m, wm) = Self::cell(&self.r, r);
        let mut out = 0.0;
        for (a, wa) in [(i, 1.0 - wi), (i + 1, wi)] {
            for (b, wb) in [(j, 1.0 - wj), (j + 1, wj)] {
                let base = (a * nt + b) * nd;
                let wab = wa * wb;
                for (c, wc) in [(k, 1.0 - wk), (k + 1, wk)] {
                    let row = (base + c) * nr;
                    out += wab * wc * ((1.0 - wm) * self.flat[row + m] + wm * self.flat[row + m + 1]);
                }
            }
        }
        out
    }
}

/// The engine's constants the routines below need, handed over by name once so
/// there is one place that holds each value (constants.py).
struct Tuning {
    reaction_ticks: f64,
    risk_tau: f64,
    body_radius: f64,
    min_along: f64,
    blocked_risk: f64,
    presser_range: f64,
    presser_cos: f64,
    presser_side_risk: f64,
    stop_threshold: f64,
    pitch_length: f64,
    goal_center_y: f64,
    goal_top_y: f64,
    goal_bottom_y: f64,
    shot_d0: f64,
    shot_k: f64,
    shot_angle_power: f64,
    value_w_shot: f64,
    progression_power: f64,
    reference_angle: f64,
    receive_floor: f64,
    space_radius: f64,
    compact_reach: f64,
    support_range: f64,
    support_bonus: f64,
    spacing_floor: f64,
    spacing_softness: f64,
    one_two_standoff: f64,
    one_two_penalty: f64,
    one_two_range: f64,
    one_two_pull: f64,
    sticky: f64,
    offside_fade: f64,
    offside_min: f64,
    zone_half_width: f64,
    box_depth: f64,
    box_top: f64,
    box_bottom: f64,
    pass_ticks_base: f64,
    pass_ticks_per_m: f64,
    pass_speed_min: f64,
}

impl Tuning {
    fn from_map(map: &HashMap<String, f64>) -> PyResult<Self> {
        let get = |key: &str| {
            map.get(key)
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("fb_core: no constant {key}")))
        };
        Ok(Tuning {
            reaction_ticks: get("REACTION_TICKS")?,
            risk_tau: get("RISK_TAU")?,
            body_radius: get("BLOCKING_BODY_RADIUS")?,
            min_along: get("BLOCKING_MIN_ALONG")?,
            blocked_risk: get("BLOCKED_LANE_RISK")?,
            presser_range: get("PRESSER_BLOCK_RANGE")?,
            presser_cos: get("PRESSER_BLOCK_COS")?,
            presser_side_risk: get("PRESSER_SIDE_RISK")?,
            stop_threshold: get("BALL_STOP_THRESHOLD")?,
            pitch_length: get("PITCH_LENGTH")?,
            goal_center_y: get("GOAL_CENTER_Y")?,
            goal_top_y: get("GOAL_TOP_Y")?,
            goal_bottom_y: get("GOAL_BOTTOM_Y")?,
            shot_d0: get("SHOT_VALUE_D0")?,
            shot_k: get("SHOT_VALUE_K")?,
            shot_angle_power: get("SHOT_VALUE_ANGLE_POWER")?,
            value_w_shot: get("VALUE_W_SHOT")?,
            progression_power: get("VALUE_PROGRESSION_POWER")?,
            reference_angle: get("REFERENCE_ANGLE")?,
            receive_floor: get("RECEIVABILITY_FLOOR")?,
            space_radius: get("SPACE_RADIUS")?,
            compact_reach: get("COMPACT_REACH")?,
            support_range: get("SUPPORT_RANGE")?,
            support_bonus: get("SUPPORT_MAX_BONUS")?,
            spacing_floor: get("TEAMMATE_SPACING_FLOOR")?,
            spacing_softness: get("SPACING_SOFTNESS")?,
            one_two_standoff: get("ONE_TWO_STANDOFF")?,
            one_two_penalty: get("ONE_TWO_CROWD_PENALTY")?,
            one_two_range: get("ONE_TWO_SUPPORT_RANGE")?,
            one_two_pull: get("ONE_TWO_PULL_BASE")?,
            sticky: get("STICKY_BONUS")?,
            offside_fade: get("OFFSIDE_FADE_DISTANCE")?,
            offside_min: get("OFFSIDE_MIN_FACTOR")?,
            zone_half_width: get("ZONE_CENTRAL_HALF_WIDTH")?,
            box_depth: get("PENALTY_AREA_DEPTH")?,
            box_top: get("PENALTY_AREA_TOP")?,
            box_bottom: get("PENALTY_AREA_BOTTOM")?,
            pass_ticks_base: get("PASS_TICKS_BASE")?,
            pass_ticks_per_m: get("PASS_TICKS_PER_M")?,
            pass_speed_min: get("PASS_SPEED_MIN")?,
        })
    }

    /// interception._travel_ticks_many, one element; None if the ball dies first.
    #[inline]
    fn travel_ticks(&self, speed: f64, friction: f64, dist: f64) -> Option<f64> {
        if dist <= 0.0 {
            return Some(0.0);
        }
        if friction <= EPS {
            return (speed > EPS).then(|| dist / speed);
        }
        let n = (((speed - self.stop_threshold) / friction).floor() + 1.0).min((speed / friction).floor());
        let carry = if speed > self.stop_threshold && n > 0.0 {
            n * speed - friction * n * (n + 1.0) / 2.0
        } else {
            0.0
        };
        let a = friction / 2.0;
        let b = friction / 2.0 - speed;
        let disc = b * b - 4.0 * a * dist;
        (speed >= EPS && dist <= carry && disc >= 0.0).then(|| (-b - disc.max(0.0).sqrt()) / (2.0 * a))
    }

    /// ball_physics.nominal_pass_speed.
    #[inline]
    fn nominal_pass_speed(&self, friction: f64, dist: f64) -> f64 {
        if dist <= 0.0 {
            return self.pass_speed_min;
        }
        let ticks = self.pass_ticks_base + self.pass_ticks_per_m * dist;
        let arrival = if ticks <= 0.0 { 0.0 } else { dist / ticks + friction * (ticks + 1.0) / 2.0 };
        arrival.max(self.pass_speed_min)
    }

    /// value._subtended_goal_angle.
    #[inline]
    fn goal_angle(&self, x: f64, y: f64, goal_x: f64) -> f64 {
        let (tx, ty) = (goal_x - x, self.goal_top_y - y);
        let (bx, by) = (goal_x - x, self.goal_bottom_y - y);
        let n_top = (tx * tx + ty * ty).sqrt();
        let n_bottom = (bx * bx + by * by).sqrt();
        if n_top < EPS || n_bottom < EPS {
            return std::f64::consts::PI;
        }
        ((tx * bx + ty * by) / (n_top * n_bottom)).clamp(-1.0, 1.0).acos()
    }

    /// value.position_value.
    #[inline]
    fn position_value(&self, x: f64, y: f64, goal_x: f64) -> f64 {
        let (dx, dy) = (goal_x - x, self.goal_center_y - y);
        let dist = (dx * dx + dy * dy).sqrt();
        let z = (dist - self.shot_d0) / self.shot_k;
        let base = if z > 700.0 { 0.0 } else { 1.0 / (1.0 + z.exp()) };
        let visibility = (self.goal_angle(x, y, goal_x) / self.reference_angle).min(1.0).powf(self.shot_angle_power);
        let own_goal_x = self.pitch_length - goal_x;
        let progress = ((x - own_goal_x).abs() / self.pitch_length).clamp(0.0, 1.0);
        self.value_w_shot * base * visibility + (1.0 - self.value_w_shot) * progress.powf(self.progression_power)
    }

    /// zones.classify_zone, as an index into the context table.
    #[inline]
    fn zone(&self, x: f64, y: f64, goal_x: f64) -> usize {
        let third = self.pitch_length / 3.0;
        let progress = (x - (self.pitch_length - goal_x)).abs();
        let wing = usize::from((y - self.goal_center_y).abs() > self.zone_half_width);
        if progress >= 2.0 * third {
            wing
        } else if progress >= third {
            2 + wing
        } else {
            4 + wing
        }
    }

    /// off_ball._in_their_box.
    #[inline]
    fn in_their_box(&self, x: f64, y: f64, direction: f64) -> bool {
        let up = if direction == 1.0 { x } else { self.pitch_length - x };
        up >= self.pitch_length - self.box_depth && self.box_top <= y && y <= self.box_bottom
    }
}

/// The reach solver: the constants of kinematics, handed over once.
#[pyclass(frozen)]
struct Reach {
    r: f64,
    k: f64,
    tau: f64,
    grid: Vec<f64>,
    grid_s: Vec<f64>,
    table: OnceLock<GripTable>,
    tuning: OnceLock<Tuning>,
}

impl Reach {
    #[inline]
    fn drift(&self, n: f64) -> f64 {
        self.r * (1.0 - (-n / self.tau).exp()) / self.k
    }

    /// kinematics._reach_ticks_scalar_linear.
    #[inline]
    fn solve_linear(&self, ox: f64, oy: f64, vx: f64, vy: f64, top: f64, reach: f64) -> f64 {
        let top = top.max(EPS);
        if ox.hypot(oy) <= reach {
            return 0.0;
        }
        let gap = |n: f64| {
            let s = self.drift(n);
            (ox - vx * s).hypot(oy - vy * s) - top * (n - s) - reach
        };
        let mut prev = 0.0;
        for (&n, &s) in self.grid.iter().zip(&self.grid_s) {
            if (ox - vx * s).hypot(oy - vy * s) - top * (n - s) - reach <= 0.0 {
                let (mut lo, mut hi) = (prev, n);
                for _ in 0..BISECT {
                    let mid = 0.5 * (lo + hi);
                    if gap(mid) <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let (g_lo, g_hi) = (gap(lo), gap(hi));
                let span = g_lo - g_hi;
                let root = if span > EPS { lo + (hi - lo) * g_lo / span } else { hi };
                return root.max(lo).min(hi);
            }
            prev = n;
        }
        let s_inf = self.r / self.k;
        let centre = (ox - vx * s_inf).hypot(oy - vy * s_inf);
        (s_inf + (centre - reach).max(0.0) / top).max(*self.grid.last().unwrap())
    }

    /// kinematics._reach_ticks_scalar: the plain law plus what grip costs.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn solve_grip(&self, t: &GripTable, ox: f64, oy: f64, vx: f64, vy: f64, top: f64, reach: f64) -> f64 {
        let base = self.solve_linear(ox, oy, vx, vy, top, reach);
        if base <= 0.0 {
            return base;
        }
        let top = top.max(EPS);
        let sp = vx.hypot(vy);
        let dist = ox.hypot(oy);
        let th = if sp * dist > EPS {
            ((vx * ox + vy * oy) / (sp * dist)).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };
        base + t.extra((sp / top).min(1.0), th, (dist - reach).max(0.0) / top, reach / top)
    }

    fn constants(&self) -> PyResult<&Tuning> {
        self.tuning
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("fb_core: constants not set"))
    }

    fn table(&self) -> PyResult<&GripTable> {
        self.table
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("fb_core: grip table not set; call kinematics.grip_table() first"))
    }
}

/// The common shape of arrays broadcast together, numpy's rule.
fn broadcast_shape(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = vec![1usize; rank];
    for shape in shapes {
        let skip = rank - shape.len();
        for (i, &d) in shape.iter().enumerate() {
            let slot = &mut out[skip + i];
            if d != 1 {
                if *slot != 1 && *slot != d {
                    return Err(PyValueError::new_err("fb_core: shapes do not broadcast"));
                }
                *slot = d;
            }
        }
    }
    Ok(out)
}

/// (x, y) of a (..., 2) array, as two views of its leading shape.
fn split_xy<'a>(a: &'a ArrayViewD<'a, f64>) -> PyResult<(ArrayViewD<'a, f64>, ArrayViewD<'a, f64>)> {
    let last = a.ndim().checked_sub(1).filter(|&l| a.shape()[l] == 2);
    let last = last.ok_or_else(|| PyValueError::new_err("fb_core: expected (..., 2)"))?;
    Ok((a.index_axis(Axis(last), 0), a.index_axis(Axis(last), 1)))
}

#[pymethods]
impl Reach {
    #[new]
    fn new(r: f64, k: f64, tau: f64, grid: Vec<f64>, grid_s: Vec<f64>) -> Self {
        Reach { r, k, tau, grid, grid_s, table: OnceLock::new(), tuning: OnceLock::new() }
    }

    /// Hand over the grip table. Later calls keep the first: it is keyed by
    /// constants that do not change while the process runs.
    #[allow(clippy::too_many_arguments)]
    fn set_table(
        &self,
        flat: PyReadonlyArray1<'_, f64>,
        ns: usize,
        nt: usize,
        s_step: f64,
        th_step: f64,
        d: Vec<f64>,
        r: Vec<f64>,
    ) -> PyResult<()> {
        let flat = flat.as_slice()?.to_vec();
        if flat.len() != ns * nt * d.len() * r.len() {
            return Err(PyValueError::new_err("fb_core: grip table has the wrong size"));
        }
        let _ = self.table.set(GripTable { flat, ns, nt, s_step, th_step, d, r });
        Ok(())
    }

    /// One player, one point, the plain law.
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, px: f64, py: f64, vx: f64, vy: f64, top: f64, qx: f64, qy: f64, reach: f64) -> f64 {
        self.solve_linear(qx - px, qy - py, vx, vy, top, reach)
    }

    /// One player, one point, with grip.
    #[allow(clippy::too_many_arguments)]
    fn ticks(&self, px: f64, py: f64, vx: f64, vy: f64, top: f64, qx: f64, qy: f64, reach: f64) -> PyResult<f64> {
        Ok(self.solve_grip(self.table()?, qx - px, qy - py, vx, vy, top, reach))
    }

    /// Hand over the constants. Later calls keep the first.
    fn set_constants(&self, constants: HashMap<String, f64>) -> PyResult<()> {
        let _ = self.tuning.set(Tuning::from_map(&constants)?);
        Ok(())
    }

    /// interception.route_survival_many under inertia: for each target, the
    /// chance a pass gets there past the best-placed opponent. The ball's pace
    /// is the ordinary pass over the distance when `speeds` is empty.
    #[allow(clippy::too_many_arguments)]
    fn lanes<'py>(
        &self,
        py: Python<'py>,
        ox: f64,
        oy: f64,
        targets: PyReadonlyArray2<'py, f64>,
        speeds: PyReadonlyArray1<'py, f64>,
        delays: PyReadonlyArray1<'py, f64>,
        friction: f64,
        pos: PyReadonlyArray2<'py, f64>,
        vel: PyReadonlyArray2<'py, f64>,
        top: PyReadonlyArray1<'py, f64>,
        ctrl: PyReadonlyArray1<'py, f64>,
        grip: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self.constants()?;
        let table = if grip { Some(self.table()?) } else { None };
        let (targets, speeds, delays) = (targets.as_array(), speeds.as_array(), delays.as_array());
        let (pos, vel, top, ctrl) = (pos.as_array(), vel.as_array(), top.as_array(), ctrl.as_array());
        let (m, n) = (targets.nrows(), pos.nrows());
        let mut out = Vec::with_capacity(m);
        for i in 0..m {
            let (sx, sy) = (targets[[i, 0]] - ox, targets[[i, 1]] - oy);
            let seg2 = sx * sx + sy * sy;
            if seg2 < EPS {
                out.push(1.0);
                continue;
            }
            let seg = seg2.sqrt();
            let speed = if speeds.is_empty() { c.nominal_pass_speed(friction, seg) } else { speeds[i] };
            let clamp = |dx: f64, dy: f64| {
                let frac = ((sx * dx + sy * dy) / seg2).clamp(0.0, 1.0);
                (frac, (dx - sx * frac).hypot(dy - sy * frac))
            };
            let mut worst = f64::NEG_INFINITY;
            for j in 0..n {
                let (px, py_, vx, vy) = (pos[[j, 0]], pos[[j, 1]], vel[[j, 0]], vel[[j, 1]]);
                let (dnx, dny) = (px - ox, py_ - oy);
                let (t_now, g_now) = clamp(dnx, dny);
                let (t_mov, g_mov) = clamp(dnx + vx * c.reaction_ticks, dny + vy * c.reaction_ticks);
                let (t, raw_gap) = if g_now < g_mov { (t_now, g_now) } else { (t_mov, g_mov) };
                let dist_along = t * seg;
                let Some(ball) = c.travel_ticks(speed, friction, dist_along) else {
                    worst = worst.max(0.0);
                    continue;
                };
                let t_ball = ball + delays[i];
                let (mx, my) = (ox + sx * t - px, oy + sy * t - py_);
                let t_opp = match table {
                    Some(g) => self.solve_grip(g, mx, my, vx, vy, top[j], ctrl[j]),
                    None => self.solve_linear(mx, my, vx, vy, top[j], ctrl[j]),
                };
                let mut risk = 1.0 / (1.0 + (-(t_ball - t_opp) / c.risk_tau).exp());
                if raw_gap <= c.body_radius && dist_along >= c.min_along && t_ball > 0.0 {
                    risk = risk.max(c.blocked_risk);
                }
                let p_dist = dnx.hypot(dny);
                if p_dist > EPS && p_dist <= c.presser_range {
                    if (dnx * sx + dny * sy) / (p_dist * seg) >= c.presser_cos {
                        risk = risk.max(c.blocked_risk);
                    } else {
                        risk = risk.min(c.presser_side_risk);
                    }
                }
                worst = worst.max(risk);
            }
            out.push(if worst >= 1.0 { 0.0 } else { 1.0 - worst });
        }
        Ok(PyArray1::from_vec(py, out))
    }

    /// off_ball._score_spots: the index of the best candidate spot for a player
    /// off the ball, or -1 if none scored.
    #[allow(clippy::too_many_arguments)]
    fn best_spot(
        &self,
        points: PyReadonlyArray2<'_, f64>,
        lanes: PyReadonlyArray1<'_, f64>,
        opponents: PyReadonlyArray2<'_, f64>,
        mates: PyReadonlyArray2<'_, f64>,
        context: [f64; 6],
        ball: [f64; 2],
        anchor: [f64; 2],
        goal_x: f64,
        direction: f64,
        sigma: f64,
        spacing_radius: f64,
        carrier_pressure: f64,
        box_level: f64,
        support: f64,
        one_two: f64,
        onside: f64,
        last: Option<[f64; 2]>,
    ) -> PyResult<i64> {
        let c = self.constants()?;
        let (points, lanes) = (points.as_array(), lanes.as_array());
        let (opponents, mates) = (opponents.as_array(), mates.as_array());
        let (mut best, mut best_score) = (-1i64, -1.0f64);
        for i in 0..points.nrows() {
            let (x, y) = (points[[i, 0]], points[[i, 1]]);
            let value = c.position_value(x, y, goal_x);
            let receivable = lanes[i].max(c.receive_floor);
            let nearest = (0..opponents.nrows())
                .map(|j| (x - opponents[[j, 0]]).hypot(y - opponents[[j, 1]]))
                .fold(f64::INFINITY, f64::min);
            let mut space = if opponents.nrows() == 0 { 1.0 } else { (nearest / c.space_radius).min(1.0) };
            let to_ball = (x - ball[0]).hypot(y - ball[1]);
            if carrier_pressure > 0.0 {
                let near_ball = 1.0 - (to_ball / c.compact_reach).min(1.0);
                space = space.powf(1.0 - carrier_pressure * near_ball);
            }
            if box_level > 0.0 && c.in_their_box(x, y, direction) {
                space = space.powf(1.0 - box_level);
            }
            let proximity = 1.0 - (to_ball / c.support_range).min(1.0);
            let support_term = 1.0 + c.support_bonus * support * proximity;
            let leash = (x - anchor[0]).hypot(y - anchor[1]) / sigma;
            let mut score = value * receivable * context[c.zone(x, y, goal_x)] * (-(leash * leash)).exp();

            let mut crowded = 1.0f64;
            for j in 0..mates.nrows() {
                let gap = (x - mates[[j, 0]]).hypot(y - mates[[j, 1]]);
                if gap < spacing_radius {
                    crowded = crowded.min(gap / spacing_radius);
                }
            }
            score *= crowded.max(c.spacing_floor).powf(c.spacing_softness);

            if one_two != 0.0 {
                if to_ball < c.one_two_standoff {
                    score *= c.one_two_penalty;
                } else {
                    let near = 1.0 - (to_ball / c.one_two_range).min(1.0);
                    score *= 1.0 + (one_two - 1.0 + c.one_two_pull) * near;
                }
            }
            score *= space * support_term;

            let over = (x - onside) * direction;
            if over > 0.0 {
                let fade = (1.0 - over / c.offside_fade).max(0.0);
                score *= c.offside_min + (1.0 - c.offside_min) * fade;
            }
            if let Some([lx, ly]) = last {
                if (x - lx).hypot(y - ly) < 2.0 {
                    score *= 1.0 + c.sticky;
                }
            }
            if score > best_score {
                best_score = score;
                best = i as i64;
            }
        }
        Ok(best)
    }

    /// Broadcast over any leading shape, like kinematics.reach_ticks.
    #[allow(clippy::too_many_arguments)]
    fn many<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArrayDyn<'py, f64>,
        vel: PyReadonlyArrayDyn<'py, f64>,
        top: PyReadonlyArrayDyn<'py, f64>,
        point: PyReadonlyArrayDyn<'py, f64>,
        reach: PyReadonlyArrayDyn<'py, f64>,
        grip: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let table = if grip { Some(self.table()?) } else { None };
        let (pos, vel, point, top, reach) = (pos.as_array(), vel.as_array(), point.as_array(), top.as_array(), reach.as_array());
        let (px, py_) = split_xy(&pos)?;
        let (vx, vy) = split_xy(&vel)?;
        let (qx, qy) = split_xy(&point)?;
        let shape = broadcast_shape(&[px.shape(), vx.shape(), qx.shape(), top.shape(), reach.shape()])?;
        let dim = IxDyn(&shape);
        let bad = || PyValueError::new_err("fb_core: shapes do not broadcast");
        let px = px.broadcast(dim.clone()).ok_or_else(bad)?;
        let py_ = py_.broadcast(dim.clone()).ok_or_else(bad)?;
        let vx = vx.broadcast(dim.clone()).ok_or_else(bad)?;
        let vy = vy.broadcast(dim.clone()).ok_or_else(bad)?;
        let qx = qx.broadcast(dim.clone()).ok_or_else(bad)?;
        let qy = qy.broadcast(dim.clone()).ok_or_else(bad)?;
        let tops = top.broadcast(dim.clone()).ok_or_else(bad)?;
        let reaches = reach.broadcast(dim.clone()).ok_or_else(bad)?;

        let mut out = Vec::with_capacity(shape.iter().product());
        let it = px
            .iter()
            .zip(py_.iter())
            .zip(vx.iter().zip(vy.iter()))
            .zip(qx.iter().zip(qy.iter()))
            .zip(tops.iter().zip(reaches.iter()));
        for ((((p, q_), (vx, vy)), (qx, qy)), (top, reach)) in it {
            let (ox, oy) = (qx - p, qy - q_);
            out.push(match table {
                Some(t) => self.solve_grip(t, ox, oy, *vx, *vy, *top, *reach),
                None => self.solve_linear(ox, oy, *vx, *vy, *top, *reach),
            });
        }
        let out = ArrayD::from_shape_vec(dim, out).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out.into_pyarray(py))
    }
}

#[pymodule]
fn fb_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reach>()?;
    Ok(())
}
