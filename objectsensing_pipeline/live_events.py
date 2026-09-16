'Optional JSONL channel for observing the pipeline from another interface.'
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class LiveEventWriter:
    'Write compact events, one JSON object per line, with immediate flushing.'

    def __init__(self, path=None):
        self.path = Path(path).resolve() if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    @property
    def enabled(self):
        return self.path is not None

    def emit(self, kind: str, data=None):
        if not self.path:
            return
        event = {
            "type": str(kind),
            "timestamp": time.time(),
            "data": data or {},
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()


def emit_progress_event(progress, kind: str, data=None):
    """Publish when the log callback exposes a rich ``event`` channel."""
    emit = getattr(progress, "event", None)
    if callable(emit):
        emit(kind, data or {})


def has_progress_events(progress):
    return callable(getattr(progress, "event", None))
