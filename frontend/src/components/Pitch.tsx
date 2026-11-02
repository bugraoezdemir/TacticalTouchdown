import { useEffect, useRef } from "react";
import type { MutableRefObject } from "react";
import { FRAME_HEAD } from "../hooks/useSimulation";
import type { Playback } from "../hooks/useSimulation";
import type { GameStateDTO } from "../types/simulation";

// Pitch dimensions in metres (engine coordinate system)
const PW = 105;
const PH = 68;
const SCALE = 7.2;
const PAD = 36;
const CANVAS_W = PW * SCALE + PAD * 2;
const CANVAS_H = PH * SCALE + PAD * 2;
const GOAL_HALF = 3.66;
const GOAL_DEPTH = 3;

function mx(x: number) { return PAD + x * SCALE; }
function my(y: number) { return PAD + y * SCALE; }

function drawPitch(ctx: CanvasRenderingContext2D) {
  // Background
  ctx.fillStyle = "#2d8a2d";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Alternating grass stripes
  for (let i = 0; i < 7; i++) {
    ctx.fillStyle = i % 2 === 0 ? "#2d8a2d" : "#319631";
    ctx.fillRect(mx(i * (PW / 7)), my(0), (PW / 7) * SCALE, PH * SCALE);
  }

  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 1.5;

  // Pitch boundary
  ctx.strokeRect(mx(0), my(0), PW * SCALE, PH * SCALE);

  // Halfway line
  ctx.beginPath(); ctx.moveTo(mx(52.5), my(0)); ctx.lineTo(mx(52.5), my(PH)); ctx.stroke();

  // Centre circle (r = 9.15 m)
  ctx.beginPath(); ctx.arc(mx(52.5), my(34), 9.15 * SCALE, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath(); ctx.arc(mx(52.5), my(34), 3, 0, Math.PI * 2); ctx.fill();

  // Penalty areas (16.5 m deep, 40.3 m wide)
  ctx.strokeRect(mx(0), my(34 - 20.15), 16.5 * SCALE, 40.3 * SCALE);
  ctx.strokeRect(mx(PW - 16.5), my(34 - 20.15), 16.5 * SCALE, 40.3 * SCALE);

  // Goal areas (5.5 m deep, 18.3 m wide)
  ctx.strokeRect(mx(0), my(34 - 9.15), 5.5 * SCALE, 18.3 * SCALE);
  ctx.strokeRect(mx(PW - 5.5), my(34 - 9.15), 5.5 * SCALE, 18.3 * SCALE);

  // Goals
  ctx.fillStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 2;
  // Left
  ctx.fillRect(mx(-GOAL_DEPTH), my(34 - GOAL_HALF), GOAL_DEPTH * SCALE, GOAL_HALF * 2 * SCALE);
  ctx.strokeRect(mx(-GOAL_DEPTH), my(34 - GOAL_HALF), GOAL_DEPTH * SCALE, GOAL_HALF * 2 * SCALE);
  // Right
  ctx.fillRect(mx(PW), my(34 - GOAL_HALF), GOAL_DEPTH * SCALE, GOAL_HALF * 2 * SCALE);
  ctx.strokeRect(mx(PW), my(34 - GOAL_HALF), GOAL_DEPTH * SCALE, GOAL_HALF * 2 * SCALE);
}

function drawEntities(ctx: CanvasRenderingContext2D, state: GameStateDTO) {
  const all = [...state.home_team.players, ...state.away_team.players];

  // Movement vectors
  for (const p of all) {
    const [vx, vy] = p.velocity;
    if (Math.hypot(vx, vy) > 0.02) {
      ctx.strokeStyle = "rgba(255,255,255,0.2)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx(p.pos[0]), my(p.pos[1]));
      ctx.lineTo(mx(p.pos[0] + vx * 10), my(p.pos[1] + vy * 10));
      ctx.stroke();
    }
  }

  // Players
  for (const p of all) {
    const isHome = p.team_id === state.home_team.team_id;
    const cx = mx(p.pos[0]);
    const cy = my(p.pos[1]);
    const r = p.has_ball ? 9 : 7;

    // Drop shadow
    ctx.beginPath();
    ctx.arc(cx + 1, cy + 2, r, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fill();

    // Body
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = isHome ? "#42a5f5" : "#ef5350";
    ctx.fill();
    ctx.strokeStyle = isHome ? "#1565c0" : "#b71c1c";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Gold ring for ball carrier
    if (p.has_ball) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 3.5, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffd700";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Position label
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.font = `bold 6.5px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(p.position, cx, cy);
  }

  // Ball
  const [bx, by] = state.ball.pos;
  const [bvx, bvy] = state.ball.velocity;
  const bspeed = Math.hypot(bvx, bvy);
  const gcx = mx(bx);
  const gcy = my(by);
  // In the air: a shadow on the ground, the ball drawn above it and larger.
  const bh = state.ball.height ?? 0;
  const bcx = gcx;
  const bcy = gcy - bh * SCALE * 0.8;
  const brad = 5 * (1 + Math.min(bh, 8) * 0.06);
  if (bh > 0.05) {
    ctx.beginPath();
    ctx.ellipse(gcx, gcy, 4.5, 2.5, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fill();
  }

  // Glow when in flight
  if (state.ball.state !== "stationary" && bspeed > 0.1) {
    const grad = ctx.createRadialGradient(bcx, bcy, 0, bcx, bcy, 20);
    grad.addColorStop(0, "rgba(255,245,80,0.5)");
    grad.addColorStop(1, "rgba(255,245,80,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(bcx, bcy, 20, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.beginPath();
  ctx.arc(bcx, bcy, brad, 0, Math.PI * 2);
  ctx.fillStyle = "#fff9c4";
  ctx.fill();
  ctx.strokeStyle = "#9e9e9e";
  ctx.lineWidth = 1;
  ctx.stroke();
}

/** Further than this in one step is a teleport (a restart, a kick-off), not
 *  movement: drawn where it ended, never slid across the pitch. */
const SNAP_PLAYER_M = 5;
const SNAP_BALL_M = 15;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

/** a -> b, `t` of the way, unless the step is a teleport. */
function lerpAt(a: number[], b: number[], i: number, t: number, snap: number): [number, number] {
  const ax = a[i], ay = a[i + 1], bx = b[i], by = b[i + 1];
  if (Math.hypot(bx - ax, by - ay) > snap) return [bx, by];
  return [lerp(ax, bx, t), lerp(ay, by, t)];
}

/** The state the playhead has reached, with ball and players placed where it
 *  is: between the two received ticks around it. Who has the ball comes from
 *  the frame too; labels and velocities from the state. */
function placed(pb: Playback): GameStateDTO | null {
  const base = pb.shown ?? pb.latest;
  if (!base || pb.frames.length === 0) return base;
  const at = Math.min(Math.max(pb.display - pb.firstTick, 0), pb.frames.length - 1);
  const i = Math.floor(at);
  const t = at - i;
  const a = pb.frames[i];
  const b = pb.frames[Math.min(i + 1, pb.frames.length - 1)];
  const owner = a[3];
  const where = new Map<number, [number, number]>();
  pb.order.forEach((id, k) => where.set(id, lerpAt(a, b, FRAME_HEAD + 2 * k, t, SNAP_PLAYER_M)));
  const move = (team: GameStateDTO["home_team"]) => ({
    ...team,
    players: team.players.map((p) => {
      const pos = where.get(p.player_id);
      return { ...p, pos: pos ?? p.pos, has_ball: p.player_id === owner };
    }),
  });
  const latest = base;
  const ballSnap = Math.hypot(b[0] - a[0], b[1] - a[1]) > SNAP_BALL_M;
  return {
    ...latest,
    home_team: move(latest.home_team),
    away_team: move(latest.away_team),
    ball: {
      ...latest.ball,
      pos: lerpAt(a, b, 0, t, SNAP_BALL_M),
      height: ballSnap ? b[2] : lerp(a[2], b[2], t),
    },
  };
}

interface PitchProps {
  state: GameStateDTO | null;
  /** Every received tick and the playhead (useSimulation). When given, the
   *  pitch draws every display frame from it; otherwise it draws `state`. */
  playback?: MutableRefObject<Playback>;
}

export function Pitch({ state, playback }: PitchProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (playback?.current.latest) return;   // the frame loop below draws
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawPitch(ctx);
    if (state) drawEntities(ctx, state);
  }, [state, playback]);

  useEffect(() => {
    if (!playback) return;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const ctx = canvasRef.current?.getContext("2d");
      const shown = placed(playback.current);
      if (!ctx || !shown) return;
      drawPitch(ctx);
      drawEntities(ctx, shown);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [playback]);

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_W}
      height={CANVAS_H}
      className="pitch-canvas"
      style={{
        borderRadius: 4,
        boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
        // The drawing buffer stays at CANVAS_W x CANVAS_H so pitch geometry is
        // unaffected; only the displayed size adapts. Scaling via CSS keeps the
        // aspect ratio exact, so the pitch never distorts when the sidebars
        // change width.
        maxWidth: "100%",
        height: "auto",
        aspectRatio: `${CANVAS_W} / ${CANVAS_H}`,
      }}
    />
  );
}
