"""Start the game: backend (FastAPI) + frontend (Vite), then open the browser.

    start_game.bat          (double-click), or
    backend\\venv\\Scripts\\python.exe run_game.py [--no-browser]

Stops both servers on Ctrl+C, when this window is closed, or as soon as either
one exits on its own -- a game with only half its servers running is not a
game. Standard library only, so it runs on the backend's own venv.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
BACKEND_PORT = 8000   # the frontend calls http://localhost:8000 (src/hooks)
FRONTEND_PORT = 5173
STARTUP_TIMEOUT_S = 60.0


def port_busy(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex((host, port)) == 0:
                return True
    return False


def backend_python() -> str:
    venv = BACKEND / "venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    return str(venv) if venv.exists() else sys.executable


def stop(proc: subprocess.Popen | None) -> None:
    """Stop a server AND its children (npm starts node, which starts vite)."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def fail(message: str) -> None:
    print(f"\n[run_game] {message}\n", flush=True)
    # Keep a double-clicked window open long enough to read why it failed.
    if os.name == "nt" and sys.stdin is not None and sys.stdin.isatty():
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
    sys.exit(1)


def main() -> None:
    npm = shutil.which("npm")
    if npm is None:
        fail("npm was not found. Install Node.js (https://nodejs.org) and try again.")
    for port, what in ((BACKEND_PORT, "backend"), (FRONTEND_PORT, "frontend")):
        if port_busy(port):
            fail(f"Port {port} ({what}) is already in use -- is the game (or another "
                 f"server) already running? Close it and start again.")

    if not (FRONTEND / "node_modules").exists():
        print("[run_game] Installing frontend dependencies (first run only)...")
        if subprocess.run([npm, "install"], cwd=FRONTEND).returncode != 0:
            fail("npm install failed; see the messages above.")

    backend = frontend = None
    try:
        print(f"[run_game] Starting backend on http://localhost:{BACKEND_PORT} ...")
        backend = subprocess.Popen(
            [backend_python(), "-m", "uvicorn", "app.main:app", "--port", str(BACKEND_PORT)],
            cwd=BACKEND,
        )
        print(f"[run_game] Starting frontend on http://localhost:{FRONTEND_PORT} ...")
        frontend = subprocess.Popen(
            [npm, "run", "dev", "--", "--port", str(FRONTEND_PORT), "--strictPort"],
            cwd=FRONTEND,
        )

        deadline = time.time() + STARTUP_TIMEOUT_S
        while not (port_busy(BACKEND_PORT) and port_busy(FRONTEND_PORT)):
            if backend.poll() is not None or frontend.poll() is not None:
                fail("A server stopped while starting; see the messages above.")
            if time.time() > deadline:
                fail(f"The servers did not come up within {STARTUP_TIMEOUT_S:.0f} s.")
            time.sleep(0.5)

        url = f"http://localhost:{FRONTEND_PORT}"
        print(f"\n[run_game] The game is running at {url}")
        print("[run_game] Press Ctrl+C (or close this window) to stop it.\n")
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)

        while backend.poll() is None and frontend.poll() is None:
            time.sleep(1.0)
        which = "backend" if backend.poll() is not None else "frontend"
        print(f"\n[run_game] The {which} stopped; shutting the other one down.")
    except KeyboardInterrupt:
        print("\n[run_game] Stopping...")
    finally:
        stop(frontend)
        stop(backend)


if __name__ == "__main__":
    main()
