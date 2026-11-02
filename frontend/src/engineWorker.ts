/// <reference lib="webworker" />
/** The live match, and the season's record, on their own thread so a slow tick
 *  never holds up the picture. Requests from the page arrive here; AI fixtures
 *  go back to the page, which runs them in simWorker, and their scores and
 *  results come back as messages with a `sim` field. */
import { loadEngine } from "./engineLoad";
import type { PyReply } from "./engineLoad";

interface Request {
  id: number;
  method: string;
  path: string;
  query: Record<string, string>;
  body: string | null;
}

export type SimReport =
  | { sim: "live"; home: number; away: number; homeGoals: number; awayGoals: number; tick: number }
  | { sim: "done"; result: Uint8Array }
  | { sim: "failed"; error: string };

/** What the player should be told, once the engine is up. */
let notice = "";

async function start() {
  const py = await loadEngine(["numpy", "fastapi"]);
  const browser = py.pyimport("app.browser");
  browser.install();
  try {
    const saved = await readSave();
    if (saved) browser.restore(saved);
  } catch (err) {
    console.error(err);
    notice = "Your saved tournament could not be loaded.";
  }
  return { py, browser };
}

const ready = start();
const stats = { requests: 0, ms: 0 };

function openSaves(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("football", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("saves");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readSave(): Promise<Uint8Array | undefined> {
  const db = await openSaves();
  return new Promise((resolve, reject) => {
    const request = db.transaction("saves").objectStore("saves").get("tournament");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function writeSave(data: Uint8Array): Promise<void> {
  const db = await openSaves();
  return new Promise((resolve, reject) => {
    const saves = db.transaction("saves", "readwrite").objectStore("saves");
    const request = data.length > 0 ? saves.put(data, "tournament") : saves.delete("tournament");
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// One at a time and in order, so an older season never lands on a newer one.
let writes: Promise<void> = Promise.resolve();

function save(browser: Awaited<typeof ready>["browser"]) {
  const reply = browser.take_save() as PyReply | undefined;
  if (reply === undefined) return;
  const data = reply.toJs() as unknown as Uint8Array;
  reply.destroy();
  writes = writes.then(() => writeSave(data)).catch((err) => {
    console.error(err);
    notice = "This tournament cannot be saved in this browser.";
  });
}

async function report(message: SimReport) {
  const { browser } = await ready;
  if (message.sim === "live") {
    browser.sim_live(message.home, message.away, message.homeGoals, message.awayGoals, message.tick);
  } else if (message.sim === "done") {
    browser.sim_result(message.result);
  } else {
    browser.sim_failed(message.error);
  }
  save(browser);
}

async function serve({ id, method, path, query, body }: Request) {
  try {
    const { py, browser } = await ready;
    const started = performance.now();
    const args = py.toPy(query);
    const reply = browser.handle(method, path, args, body);
    const [status, text] = reply.toJs();
    reply.destroy();
    args.destroy();
    const queued = browser.take_specs();
    const specs = Array.from(queued.toJs() as Uint8Array[]);
    queued.destroy();
    save(browser);
    stats.requests += 1;
    stats.ms += performance.now() - started;
    self.postMessage({ id, status, text, stats, specs, notice }, specs.map((s) => s.buffer as ArrayBuffer));
  } catch (err) {
    console.error(err);
    self.postMessage({ id, status: 500, text: JSON.stringify({ detail: String(err) }), stats, specs: [], notice });
  }
}

self.onmessage = (e: MessageEvent<Request | SimReport>) => {
  void ("sim" in e.data ? report(e.data) : serve(e.data));
};
