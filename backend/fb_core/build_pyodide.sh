#!/bin/bash
# Build the fb_core wheel for Pyodide. Run in WSL or Linux from this folder, with
# make and a C compiler on PATH. Everything goes under ~/pyo-build; the wheel ends
# up in ~/pyo-build/crate/dist. PYODIDE must match the Pyodide the page loads:
# 314.0.7 needs Rust 1.93.0 and Emscripten 5.0.3 (`pyodide config get`).
set -eu
PYODIDE=314.0.7
RUST=1.93.0
B=~/pyo-build; mkdir -p $B
HERE=$(pwd)
cd $B
export PATH=~/.local/bin:~/.cargo/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d venv ] || uv venv --python 3.14 venv
. venv/bin/activate
uv pip install -q pyodide-build maturin pip
pyodide xbuildenv install $PYODIDE
command -v rustup >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --no-modify-path
rustup toolchain install $RUST --profile minimal -t wasm32-unknown-emscripten
rustup default $RUST
pyodide xbuildenv install-emscripten
EMSDK_DIR=$(ls -d ~/.cache/pyodide-build/.pyodide-xbuildenv-*/$PYODIDE/emsdk | head -1)
. "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
rm -rf crate; mkdir crate
cp -r "$HERE"/Cargo.toml "$HERE"/Cargo.lock "$HERE"/pyproject.toml "$HERE"/src crate/
cd crate
pyodide build . -o dist
ls -la dist
