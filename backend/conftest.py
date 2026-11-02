"""Put the backend root on sys.path so `import app...` works from any cwd.

The debug_*.py scripts rely on `sys.path.insert(0, ".")`, which only works when
run from backend/. pytest may collect from anywhere, so anchor to this file.
"""
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).parent.resolve())
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
