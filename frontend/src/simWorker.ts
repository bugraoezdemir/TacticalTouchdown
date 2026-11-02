/// <reference lib="webworker" />
/** One AI fixture at a time, flat out, on its own thread: the browser's stand-in
 *  for the server's worker processes. */
import { loadEngine } from "./engineLoad";

const ready = loadEngine(["numpy"]).then((py) => py.pyimport("app.browser_sim"));

function live(home: number, away: number, homeGoals: number, awayGoals: number, tick: number) {
  self.postMessage({ sim: "live", home, away, homeGoals, awayGoals, tick });
}

self.onmessage = async (e: MessageEvent<Uint8Array>) => {
  try {
    const sim = await ready;
    const reply = sim.run(e.data, live);
    const result = reply.toJs() as unknown as Uint8Array;
    reply.destroy();
    self.postMessage({ sim: "done", result }, [result.buffer as ArrayBuffer]);
  } catch (err) {
    console.error(err);
    self.postMessage({ sim: "failed", error: String(err) });
  }
};
