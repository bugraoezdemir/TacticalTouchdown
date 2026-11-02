"""Gather what the browser build needs into frontend/public/engine: Pyodide, the
packages the engine imports, the fb_core wheel, the engine's source and the grip
table. Run with the backend venv, from anywhere. Downloads are cached there and
checked against Pyodide's lock file. The wheel comes from
backend/fb_core/build_pyodide.sh, copied to backend/fb_core/dist_pyodide.
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PYODIDE = "314.0.7"      # as in build_pyodide.sh
CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE}/full/"
CORE = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
NEEDED = ["numpy", "fastapi"]

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "engine"
BACKEND = ROOT / "backend"


def fetch(name: str, sha256: str | None = None) -> None:
    path = OUT / name
    if not path.exists():
        print("download", name)
        urllib.request.urlretrieve(CDN + name, path)
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        path.unlink()
        sys.exit(f"{name}: checksum does not match Pyodide's lock file")


def packages(lock: dict) -> dict:
    """The lock entries of NEEDED and everything they depend on."""
    by_name = {name.lower().replace("_", "-"): entry for name, entry in lock["packages"].items()}
    seen, todo = {}, list(NEEDED)
    while todo:
        name = todo.pop().lower().replace("_", "-")
        if name not in seen:
            seen[name] = by_name[name]
            todo += seen[name].get("depends", [])
    return seen


def engine_source() -> None:
    with zipfile.ZipFile(OUT / "app.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for folder, dirs, files in os.walk(BACKEND / "app"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if name.endswith(".py"):
                    path = Path(folder) / name
                    z.write(path, path.relative_to(BACKEND))


def grip_table() -> str:
    """The grip table, ready made: building it in the browser would take far
    longer than downloading it. Returns the name kinematics looks for."""
    os.environ.update(FB_INERTIA="1", FB_GRIP="1", FB_RUST="0")
    sys.path.insert(0, str(BACKEND))
    from app.engine import kinematics

    kinematics.grip_table()
    path = Path(kinematics._table_path())
    shutil.copy(path, OUT / "grip_table.npy")
    return path.name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in CORE:
        fetch(name)
    lock = json.loads((OUT / "pyodide-lock.json").read_text(encoding="utf-8"))
    for entry in packages(lock).values():
        fetch(entry["file_name"], entry["sha256"])
    wheels = sorted((BACKEND / "fb_core" / "dist_pyodide").glob("*.whl"))
    if not wheels:
        sys.exit("no fb_core wheel: run backend/fb_core/build_pyodide.sh and copy it to dist_pyodide")
    shutil.copy(wheels[-1], OUT / wheels[-1].name)
    engine_source()
    manifest = {"wheel": wheels[-1].name, "table": grip_table()}
    (OUT / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"{OUT}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
