import type { SimReport } from "./engineWorker";

/** Where the game's requests go. Normally the backend on localhost; in the
 *  browser build (`npm run build:browser`) the engine runs in a Web Worker on
 *  the player's own machine, and requests go to it instead, so the hooks need
 *  not know which. */
const API = "http://localhost:8000";
export const IN_BROWSER = import.meta.env.VITE_ENGINE === "browser";
export const ENGINE_LABEL = IN_BROWSER ? "the engine" : `API at ${API}`;

interface Reply {
  id: number;
  status: number;
  text: string;
  /** Requests served and the time they took, for measuring in the console. */
  stats: { requests: number; ms: number };
  /** AI fixtures to run, pickled. */
  specs: Uint8Array[];
  /** Anything the player should be told. */
  notice: string;
}

/** AI matches at once. A matchday has eight, and hardwareConcurrency counts
 *  hyperthreads, so half of it leaves real cores for the live match. */
const SIM_WORKERS = Math.min(8, Math.max(1, Math.floor((navigator.hardwareConcurrency || 2) / 2)));

let worker: Worker | null = null;
let nextId = 0;
const waiting = new Map<number, (reply: Reply) => void>();

function engine(): Worker {
  if (worker === null) {
    worker = new Worker(new URL("./engineWorker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<Reply>) => {
      (globalThis as { engineStats?: unknown }).engineStats = e.data.stats;
      notice = e.data.notice;
      fixtures.push(...e.data.specs);
      nextFixture();
      const resolve = waiting.get(e.data.id);
      waiting.delete(e.data.id);
      resolve?.(e.data);
    };
  }
  return worker;
}

let fixtures: Uint8Array[] = [];
let idle: Worker[] = [];
let sims: Worker[] = [];
let notice = "";

export function engineNotice(): string {
  return notice;
}

/** A new tournament: the old one's AI fixtures are no longer wanted. */
function stopSims() {
  sims.forEach((sim) => sim.terminate());
  sims = [];
  idle = [];
  fixtures = [];
}

function send(sim: Worker, fixture: Uint8Array) {
  sim.postMessage(fixture, [fixture.buffer as ArrayBuffer]);
}

/** Give waiting AI fixtures to idle sim workers, starting more up to the limit. */
function nextFixture() {
  while (fixtures.length > 0 && (idle.length > 0 || sims.length < SIM_WORKERS)) {
    send(idle.pop() ?? startSimWorker(), fixtures.shift()!);
  }
}

function startSimWorker(): Worker {
  const sim = new Worker(new URL("./simWorker.ts", import.meta.url), { type: "module" });
  sims.push(sim);
  sim.onmessage = (e: MessageEvent<SimReport>) => {
    engine().postMessage(e.data);
    if (e.data.sim !== "live") {
      idle.push(sim);
      nextFixture();
    }
  };
  sim.onerror = (e) => {
    engine().postMessage({ sim: "failed", error: e.message } satisfies SimReport);
  };
  return sim;
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  if (!IN_BROWSER) return fetch(`${API}${path}`, init);
  if (path === "/tournament/reset") stopSims();
  // Asked once there is something to keep: browsers may clear storage otherwise.
  if (path === "/tournament/create") void navigator.storage?.persist?.();
  const url = new URL(path, "http://engine");
  const id = nextId++;
  const reply = await new Promise<Reply>((resolve) => {
    waiting.set(id, resolve);
    engine().postMessage({
      id,
      method: init?.method ?? "GET",
      path: url.pathname,
      query: Object.fromEntries(url.searchParams),
      body: init?.body ?? null,
    });
  });
  return new Response(reply.text, {
    status: reply.status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Settles when the engine can answer. Asking now starts loading it before it
 *  is needed, so the first press of Start need not wait for all of it. */
export const engineLoaded: Promise<unknown> = IN_BROWSER
  ? apiFetch("/tournament/state")
  : Promise.resolve();
