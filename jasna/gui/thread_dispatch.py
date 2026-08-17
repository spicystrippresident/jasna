"""Thread-safe delivery of background work to the Tk main thread."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True)
class GuiCall:
    """A callback that must be invoked by the GUI thread."""

    callback: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class GuiThreadDispatcher:
    """Queue callbacks without making any Tk calls from producer threads."""

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._lock = threading.Lock()
        self._calls: deque[GuiCall] = deque()
        self._closed = False

    def post(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Queue one callback, returning ``False`` after shutdown starts."""

        with self._lock:
            if self._closed:
                return False
            self._calls.append(GuiCall(callback, args, kwargs))
            return True

    def take(self, limit: int = 128) -> tuple[GuiCall, ...]:
        """Take up to ``limit`` callbacks on the thread that created the queue."""

        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("GUI callbacks may only be drained by the GUI thread")
        if limit <= 0:
            return ()
        with self._lock:
            count = min(limit, len(self._calls))
            return tuple(self._calls.popleft() for _ in range(count))

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._calls)

    def close(self) -> None:
        """Reject future callbacks and discard callbacks queued during shutdown."""

        with self._lock:
            self._closed = True
            self._calls.clear()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed
