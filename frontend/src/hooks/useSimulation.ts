import { useCallback, useEffect, useRef, useState } from "react";
import type { GameStateDTO, MatchEventDTO, TacticsUpdate, TeamDTO } from "../types/simulation";
import { apiFetch, ENGINE_LABEL } from "../api";

/** Engine ticks per second of match time (TICK_RATE in the backend). */
export const TICKS_PER_SECOND = 30;
/** Most ticks asked for in one request: bounds a single catch-up. */
const MAX_BATCH = 30;
/** How much match the browser keeps in hand beyond the playhead, in seconds
 *  of what is shown. Asking only for the ticks the clock had made due meant
 *  nothing was ever in hand: every slow stretch on the server (on-ball
 *  decisions, ~60-170 ms ticks in clusters) froze the picture -- measured, 76
 *  stalls in 100 s although the engine averages 1.4x real time. With 1.5 s in
 *  hand: one, at the start. */
const BUFFER_SECONDS = 1.5;
/** After running dry, wait for this many ticks before playing on: one short
 *  wait instead of a stutter. Also the wait before kick-off. */
const REBUFFER_TICKS = 8;
/** A stoppage (goal, throw-in, corner...) is announced -- banner, log, score --
 *  this long after the ball is seen to cross the line, the way the whistle
 *  follows it. On the same frame, the message seemed to come first. */
const WHISTLE_DELAY_TICKS = 6;
const STOPPAGES = new Set(["goal", "throw_in", "corner", "goal_kick", "offside"]);
/** At "Max" speed: ticks per request, back to back. */
const MAX_SPEED_BATCH = 8;
/** The scoreboard, stats and sidebars refresh at most this often; the pitch
 *  itself draws every display frame from the playback buffer. Re-rendering
 *  the whole page on every tick was part of what made the match slow. */
const UI_REFRESH_MS = 200;
/** The pitch shows the match this many ticks behind the newest one received,
 *  so there is always a next tick to slide toward... */
const PLAYBACK_LEAD = 1;
/** At "Max", never more than this far behind: past it, playback jumps
 *  forward rather than trail the engine. */
const PLAYBACK_MAX_LAG = 45;
/** Values ahead of the players in a frame: ball x, y, height, owner id (-1
 *  for none), home score, away score. */
export const FRAME_HEAD = 6;

export type SimStatus = "idle" | "running" | "paused";

/** Every tick the server has sent, in order, and where the pitch is playing.
 *
 *  Each frame is [ball x, y, height, owner, home score, away score, then x, y
 *  of each player in `order`]. The pitch plays these back tick by tick at the
 *  real clock rate, sliding between consecutive ticks. Drawing only the last
 *  state of each batch slid the ball in a straight line across the tick a
 *  player took it, so it seemed to turn before it ever reached him.
 *
 *  Everything else -- events, score, stats, clock -- is released when the
 *  playhead reaches it, not when it arrives: taken from the newest tick
 *  received, the goal banner went up before the ball had crossed the line.
 */
export interface Playback {
  frames: number[][];
  firstTick: number;     // tick of frames[0]
  order: number[];       // player ids, as laid out in each frame
  latest: GameStateDTO | null;
  pending: GameStateDTO[];   // received, not yet reached by the playhead
  shown: GameStateDTO | null;   // newest state at or before the playhead
  display: number;       // fractional tick being drawn
}

function emptyPlayback(): Playback {
  return { frames: [], firstTick: 0, order: [], latest: null, pending: [], shown: null, display: 0 };
}

/** `s` with the settings of `src`: each team's tactics and each player's role
 *  and overrides; everything else as it was. */
function withTactics(s: GameStateDTO, src: GameStateDTO): GameStateDTO {
  const team = (t: TeamDTO, from: TeamDTO): TeamDTO => ({
    ...t,
    tactics: from.tactics,
    players: t.players.map((p) => {
      const q = from.players.find((x) => x.player_id === p.player_id);
      if (!q) return p;
      const settings = Object.fromEntries(
        Object.entries(q).filter(([k]) => k.startsWith("bias_") || k === "position" || k === "tactical"),
      );
      return { ...p, ...settings };
    }),
  });
  return { ...s, home_team: team(s.home_team, src.home_team), away_team: team(s.away_team, src.away_team) };
}

/** One frame from a full state: what a response without `frames` carries. */
function frameOf(s: GameStateDTO): { frame: number[]; order: number[] } {
  const players = [...s.home_team.players, ...s.away_team.players];
  const frame = [s.ball.pos[0], s.ball.pos[1], s.ball.height ?? 0,
    s.ball.owner_id ?? -1, s.home_score, s.away_score];
  for (const p of players) frame.push(p.pos[0], p.pos[1]);
  return { frame, order: players.map((p) => p.player_id) };
}

/** `speed` is a multiple of real time (1 = 30 ticks a second); Infinity runs
 *  as fast as the engine can. The clock is real elapsed time, so the label
 *  means what it says: before, each tick waited a fixed delay ON TOP of the
 *  request and the redraw, and "2x" played at about 0.4x real time. */
export function useSimulation(speed: number) {
  const [state, setState] = useState<GameStateDTO | null>(null);
  const [status, setStatus] = useState<SimStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  // Match log, newest first. The engine reports these; the UI no longer has to
  // infer goals from ball position, which missed some and invented others.
  const [log, setLog] = useState<MatchEventDTO[]>([]);
  const [banner, setBanner] = useState<MatchEventDTO | null>(null);
  const [score, setScore] = useState<[number, number]>([0, 0]);

  const playback = useRef<Playback>(emptyPlayback());
  const runningRef = useRef(false);
  const inFlightRef = useRef(false);
  const finishedRef = useRef(false);   // the server has played full time
  const waitingRef = useRef(true);     // holding the playhead until ticks are in hand
  const lastClockRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const lastUiRef = useRef(0);
  const scoreRef = useRef<[number, number]>([0, 0]);
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const seenRef = useRef<Set<string>>(new Set());
  const bannerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** The events of `next` up to tick `upTo` not yet shown. An event carries
   *  the tick it was raised IN, before the engine counted it, so it is first
   *  visible in the frame of tick + 1: the goal at 1277 is the ball crossing
   *  the line in frame 1278. */
  const ingestEvents = useCallback((next: GameStateDTO, upTo: number) => {
    const fresh = next.events.filter((e) => {
      if (e.tick + 1 + (STOPPAGES.has(e.kind) ? WHISTLE_DELAY_TICKS : 0) > upTo) return false;
      const key = `${e.tick}:${e.kind}:${e.detail}`;
      if (seenRef.current.has(key)) return false;
      seenRef.current.add(key);
      return true;
    });
    if (fresh.length === 0) return false;
    setLog((prev) => [...fresh].reverse().concat(prev).slice(0, 30));

    // Surface the most important one: a goal outranks a restart.
    const headline = fresh.find((e) => e.kind === "goal") ?? fresh[fresh.length - 1];
    setBanner(headline);
    if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    bannerTimerRef.current = setTimeout(
      () => setBanner(null),
      headline.kind === "goal" ? 2600 : 1400,
    );
    return true;
  }, []);

  /** Release what the playhead has reached: the score at tick `upTo`, its
   *  events, and the newest state at or before it (throttled, unless
   *  something happened or `force`). */
  const show = useCallback((upTo: number, force = false) => {
    const pb = playback.current;
    if (pb.frames.length) {
      // The score turns over with the goal's announcement, not before it.
      const at = Math.min(Math.max(Math.floor(upTo) - WHISTLE_DELAY_TICKS - pb.firstTick, 0),
                          pb.frames.length - 1);
      const f = pb.frames[at];
      if (f[4] !== scoreRef.current[0] || f[5] !== scoreRef.current[1]) {
        scoreRef.current = [f[4], f[5]];
        setScore([f[4], f[5]]);
      }
    }
    const happened = pb.latest ? ingestEvents(pb.latest, upTo) : false;
    while (pb.pending.length && pb.pending[0].tick <= upTo) pb.shown = pb.pending.shift()!;
    const now = performance.now();
    if (pb.shown && (force || happened || now - lastUiRef.current >= UI_REFRESH_MS)) {
      lastUiRef.current = now;
      setState(pb.shown);
    }
  }, [ingestEvents]);

  /** A new state from the server: its ticks into the playback buffer. It is
   *  shown when the playhead gets there -- at once if it starts a new
   *  buffer. */
  const receive = useCallback((next: GameStateDTO) => {
    const pb = playback.current;
    const batch = next.frames && next.frames.length ? next.frames : [frameOf(next).frame];
    const order = next.frame_order && next.frame_order.length ? next.frame_order : frameOf(next).order;
    const first = next.tick - batch.length + 1;
    const lastHeld = pb.firstTick + pb.frames.length - 1;
    const sameOrder = order.join() === pb.order.join();
    finishedRef.current = next.finished;
    if (pb.frames.length && sameOrder && !next.frames?.length && next.tick <= lastHeld) {
      // A tactics reply: the newest tick, up to BUFFER_SECONDS ahead of the
      // picture, with the new settings. Its settings go into what is shown
      // and what is waiting to be shown, so the sliders answer at once
      // without the rest of the page jumping ahead -- or a waiting state
      // putting the old settings back.
      pb.latest = next;
      pb.pending = pb.pending.map((s) => withTactics(s, next));
      pb.shown = pb.shown ? withTactics(pb.shown, next) : next;
      setState(pb.shown);
      return;
    }
    if (pb.frames.length === 0 || first !== lastHeld + 1 || !sameOrder) {
      // A new match, a reset, a restart of the buffer: start over from this
      // state, shown at once.
      pb.frames = batch.slice();
      pb.firstTick = first;
      pb.order = order;
      pb.display = next.tick;
      pb.latest = next;
      pb.pending = [next];
      show(Infinity, true);   // everything it carries, at once
      return;
    }
    pb.frames.push(...batch);
    // Drop what has been played, keeping a little behind the playhead.
    const drop = Math.floor(pb.display) - pb.firstTick - 2 - WHISTLE_DELAY_TICKS;
    if (drop > 0) {
      pb.frames.splice(0, drop);
      pb.firstTick += drop;
    }
    pb.latest = next;
    pb.pending.push(next);
  }, [show]);

  const requestTicks = useCallback(async (n: number) => {
    inFlightRef.current = true;
    try {
      const res = await apiFetch(`/simulate/tick?n=${n}`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const next: GameStateDTO = await res.json();
      if (!runningRef.current) return;
      // At full time the server freezes the state: receive() marks it, and
      // the loop stops asking and plays out what it holds.
      receive(next);
    } catch (e) {
      setError(String(e));
      runningRef.current = false;
      setStatus("paused");
    } finally {
      inFlightRef.current = false;
    }
  }, [receive]);

  /** The clock: every display frame the playhead moves on at the chosen
   *  speed of real time, and whenever no request is waiting the browser asks
   *  for enough ticks to keep BUFFER_SECONDS in hand ahead of it. */
  const loop = useCallback(() => {
    if (!runningRef.current) return;
    const now = performance.now();
    const dt = now - lastClockRef.current;
    lastClockRef.current = now;
    const s = speedRef.current;
    const finite = Number.isFinite(s);
    const done = finishedRef.current;
    const pb = playback.current;
    const newest = pb.frames.length ? pb.firstTick + pb.frames.length - 1 : 0;
    const ahead = pb.frames.length ? newest - pb.display : 0;
    if (!done && !inFlightRef.current) {
      if (finite) {
        const target = Math.max(REBUFFER_TICKS, BUFFER_SECONDS * TICKS_PER_SECOND * s);
        if (ahead < target) {
          void requestTicks(Math.max(1, Math.min(MAX_BATCH, Math.ceil(target - ahead))));
        }
      } else {
        void requestTicks(MAX_SPEED_BATCH);
      }
    }
    // The playhead moves at the same clock, within the ticks received; at
    // full time it runs to the last one.
    if (pb.frames.length) {
      const lead = done ? 0 : PLAYBACK_LEAD;
      if (waitingRef.current && (done || !finite || ahead >= REBUFFER_TICKS + lead)) {
        waitingRef.current = false;
      }
      if (!waitingRef.current) {
        const step = finite ? (dt / 1000) * TICKS_PER_SECOND * s : newest - pb.display;
        // Never backwards.
        pb.display = Math.max(Math.min(newest - lead, pb.display + step), Math.min(pb.display, newest));
        if (!finite) pb.display = Math.max(pb.display, newest - PLAYBACK_MAX_LAG);
        pb.display = Math.max(pb.display, pb.firstTick);
        // Ran dry: wait for a few ticks rather than stutter tick by tick.
        if (finite && !done && newest - pb.display <= lead + 1e-6) waitingRef.current = true;
      }
      show(pb.display);
      if (done && pb.display >= newest) {
        runningRef.current = false;
        rafRef.current = null;
        show(Infinity, true);   // full time: the last events too
        setStatus("paused");
        return;
      }
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [requestTicks, show]);

  const stopLoop = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }, []);

  /** Forget the match shown: buffer, log, banner, score. */
  const clearShown = useCallback(() => {
    setLog([]);
    setBanner(null);
    seenRef.current.clear();
    playback.current = emptyPlayback();
    finishedRef.current = false;
    scoreRef.current = [0, 0];
    setScore([0, 0]);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    if (status === "idle") {
      // Fresh start: tell server to initialise state.
      try {
        const res = await apiFetch("/simulate/start", { method: "POST" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const initial: GameStateDTO = await res.json();
        receive(initial);
      } catch (e) {
        setError(`Cannot reach ${ENGINE_LABEL}. Is it running?`);
        return;
      }
    }
    runningRef.current = true;
    waitingRef.current = true;
    lastClockRef.current = performance.now();
    setStatus("running");
    rafRef.current = requestAnimationFrame(loop);
  }, [status, loop, receive]);

  // Tactics apply from the next tick; the match is not restarted. The response
  // carries the full state, so the sliders reflect what the engine accepted
  // (it clamps each bias to [0.5, 2.0]).
  const updateTactics = useCallback(async (update: TacticsUpdate) => {
    try {
      const res = await apiFetch("/simulate/tactics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(update),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      receive(await res.json());
    } catch (e) {
      setError(String(e));
    }
  }, [receive]);

  const pause = useCallback(() => {
    stopLoop();
    // Stop where the picture is. The ticks already in hand play on when it
    // resumes; jumping to the newest one skipped up to BUFFER_SECONDS.
    show(playback.current.display, true);
    setStatus("paused");
  }, [stopLoop, show]);

  /** Adopt whatever match the server currently holds, without starting it.
   *
   *  The tournament router loads a fixture into the server's single state
   *  slot; the Match tab then has to show THAT match rather than whatever it
   *  was displaying before. Ticking once is how we read it back -- there is no
   *  GET for the state, and one tick out of 30,000 is not worth an endpoint.
   */
  const loadServerMatch = useCallback(async () => {
    stopLoop();
    setError(null);
    clearShown();
    setStatus("paused");
    try {
      const res = await apiFetch("/simulate/tick", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      receive(await res.json());
    } catch {
      setError(`Cannot reach ${ENGINE_LABEL}. Is it running?`);
    }
  }, [stopLoop, clearShown, receive]);

  const reset = useCallback(async () => {
    stopLoop();
    setError(null);
    clearShown();
    try {
      const res = await apiFetch("/simulate/reset", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      receive(await res.json());
    } catch {
      setState(null);
    }
    setStatus("idle");
  }, [stopLoop, clearShown, receive]);

  useEffect(() => () => {
    stopLoop();
    if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
  }, [stopLoop]);

  return {
    state,
    playback,
    status,
    homeGoals: score[0],
    awayGoals: score[1],
    error,
    log,
    banner,
    start,
    loadServerMatch,
    pause,
    reset,
    updateTactics,
  };
}
