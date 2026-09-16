from types import SimpleNamespace

import numpy as np

from parallelism import kdtree_workers
from transforms import GroundTransform
from verify import icp_refine


def test_kdtree_workers_keeps_all_cores_in_main_process(monkeypatch):
    monkeypatch.delenv("OBJECTSENSING_KDTREE_WORKERS", raising=False)
    assert kdtree_workers() == -1


def test_kdtree_workers_respects_process_limit(monkeypatch):
    monkeypatch.setenv("OBJECTSENSING_KDTREE_WORKERS", "1")
    assert kdtree_workers() == 1


def test_kdtree_workers_sanitizes_invalid_values(monkeypatch):
    monkeypatch.setenv("OBJECTSENSING_KDTREE_WORKERS", "invalid")
    assert kdtree_workers() == -1

    monkeypatch.setenv("OBJECTSENSING_KDTREE_WORKERS", "0")
    assert kdtree_workers() == 1


def test_icp_uses_the_process_kdtree_limit(monkeypatch):
    monkeypatch.setenv("OBJECTSENSING_KDTREE_WORKERS", "1")

    class RecordingTree:
        workers = None

        def query(self, points, workers):
            self.workers = workers
            return np.ones(len(points)), np.zeros(len(points), dtype=int)

    tree = RecordingTree()
    scan = SimpleNamespace(tree=tree, points=np.zeros((30, 3)))
    up = np.array([0.0, 1.0, 0.0])
    transform = GroundTransform(0.0, 1.0, np.zeros(3), up)

    icp_refine(np.zeros((30, 3)), scan, transform, up, iters=1)

    assert tree.workers == 1
