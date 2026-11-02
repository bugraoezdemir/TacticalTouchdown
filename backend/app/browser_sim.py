"""One AI fixture in a Web Worker (frontend/src/simWorker.ts): the same simulate()
the server's worker processes run. Fixtures and results cross between workers
pickled, as they cross between processes on the server."""
from __future__ import annotations

import pickle
from collections.abc import Callable

from .engine.headless import MatchSpec, simulate


class _LiveScores(dict):
    """Passes each running score on, where the server writes it to a shared dict."""

    def __init__(self, post: Callable[..., None]) -> None:
        super().__init__()
        self.post = post

    def __setitem__(self, key: tuple[int, int], value: tuple[int, int, int]) -> None:
        self.post(key[0], key[1], value[0], value[1], value[2])


def run(data, post: Callable[..., None]) -> bytes:
    spec: MatchSpec = pickle.loads(data.to_bytes() if hasattr(data, "to_bytes") else data)
    return pickle.dumps(simulate(spec, live=_LiveScores(post), reactive=True))
