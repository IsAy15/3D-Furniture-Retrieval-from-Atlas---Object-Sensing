"""Shared limits for nested scientific-library parallelism."""

from __future__ import annotations

import os


_THREADPOOL_LIMITER = None


def kdtree_workers() -> int:
    """Return the SciPy worker count selected for the current process.

    The main process keeps SciPy's all-core default. Process-pool initializers
    set ``OBJECTSENSING_KDTREE_WORKERS=1`` so each child stays single-threaded
    instead of multiplying the requested ``--jobs`` by the machine core count.
    """
    try:
        return max(1, int(os.environ["OBJECTSENSING_KDTREE_WORKERS"]))
    except (KeyError, TypeError, ValueError):
        return -1


def configure_process_worker() -> None:
    """Bound nested KD-tree and BLAS pools inside a process-pool worker."""
    os.environ.setdefault("OBJECTSENSING_KDTREE_WORKERS", "1")
    workers = kdtree_workers()

    global _THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=workers)
    except (ImportError, RuntimeError):
        _THREADPOOL_LIMITER = None
