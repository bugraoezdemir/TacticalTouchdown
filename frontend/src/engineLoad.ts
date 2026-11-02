/// <reference lib="webworker" />
/** Python and the engine, for the Web Workers. Files come from public/engine,
 *  gathered by scripts/engine_assets.py. */
const base = new URL(`${import.meta.env.BASE_URL}engine/`, self.location.origin).href;

interface Manifest {
  wheel: string;
  table: string;
}

export interface Pyodide {
  toPy(value: unknown): { destroy(): void };
  pyimport(name: string): Record<string, (...args: unknown[]) => PyReply>;
}

export interface PyReply {
  toJs(): [number, string] & Uint8Array[];
  destroy(): void;
}

export async function loadEngine(packages: string[]) {
  const { loadPyodide } = await import(/* @vite-ignore */ `${base}pyodide.mjs`);
  const py = await loadPyodide({
    indexURL: base,
    env: { FB_INERTIA: "1", FB_GRIP: "1", FB_RUST: "1", HOME: "/home/pyodide" },
  });
  const manifest: Manifest = await (await fetch(`${base}manifest.json`)).json();
  await py.loadPackage([...packages, `${base}${manifest.wheel}`]);
  const load = async (name: string) => (await fetch(`${base}${name}`)).arrayBuffer();
  py.unpackArchive(await load("app.zip"), "zip", { extractDir: "/backend" });
  py.FS.writeFile(`/tmp/${manifest.table}`, new Uint8Array(await load("grip_table.npy")));
  py.runPython("import sys; sys.path.insert(0, '/backend')");
  return py as Pyodide;
}
