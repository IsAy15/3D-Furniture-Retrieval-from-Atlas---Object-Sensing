"""Valide la distance descripteur vectorisée sur toutes the rotations à la fois
(doit donner exactement the mêmes valeurs que la version par-rotation)."""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from config import DescriptorConfig  # noqa: E402
import descriptors as D  # noqa: E402


def corner_points(n=4000, e=0.12, seed=0):
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.uniform(0, e, n), np.zeros(n), rng.uniform(-e, e, n)])
    b = np.column_stack([np.zeros(n), rng.uniform(0, e, n), rng.uniform(-e, e, n)])
    return np.vstack([a, b])


def test_batch_equals_loop():
    up = np.array([0.0, 0.0, 1.0])
    cfg = DescriptorConfig(udf_grid_res=16, udf_extent=0.12, distance_exponent=4.0)
    tree = cKDTree(corner_points())
    md = D.build_model_descriptor(tree, np.zeros(3), cfg)
    sd = D.LocalDescriptor("scan", 16, 0.12, np.zeros(3),
                           occupied_offsets=corner_points(600, seed=3), n_occupied=600)
    thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    ref = np.array([D.descriptor_distance(md, sd, cfg, th, up) for th in thetas])
    batch = D.descriptor_distance_batch(md, sd, cfg, thetas, up)
    print(f"max ecart batch vs boucle = {np.abs(ref - batch).max():.3e}")
    print(f"min boucle={ref.min():.3e} @theta={thetas[ref.argmin()]:.2f} ; "
          f"min batch={batch.min():.3e} @theta={thetas[batch.argmin()]:.2f}")
    assert np.allclose(ref, batch, rtol=1e-5, atol=1e-12)
    assert ref.argmin() == batch.argmin()


if __name__ == "__main__":
    test_batch_equals_loop()
    print("OK batched descriptor distance")
