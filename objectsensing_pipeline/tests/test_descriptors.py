"""Test des descripteurs locaux + distance (eq.3) avec gestion de la rotation."""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DescriptorConfig  # noqa: E402
import descriptors as D  # noqa: E402


def corner_points(n=4000, e=0.12, seed=0):
    """Coin droit : plan y=0 (x>=0) et plan x=0 (y>=0), arête le long de z."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.uniform(0, e, n), np.zeros(n), rng.uniform(-e, e, n)])
    b = np.column_stack([np.zeros(n), rng.uniform(0, e, n), rng.uniform(-e, e, n)])
    return np.vstack([a, b])


def rot_z(p, deg):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return p @ R.T


def make_scan_desc(offsets, res=16, extent=0.12):
    return D.LocalDescriptor("scan", res, extent, np.zeros(3),
                             occupied_offsets=offsets, n_occupied=len(offsets))


def test_descriptor_rotation():
    up = np.array([0.0, 0.0, 1.0])
    cfg = DescriptorConfig(udf_grid_res=16, udf_extent=0.12, distance_exponent=4.0)
    model = corner_points()
    tree = cKDTree(model)
    md = D.build_model_descriptor(tree, np.zeros(3), cfg)
    print(f"UDF modele: occupied={md.n_occupied}/{md.n_total}")

    # scan = même coin (sous-échantillonné)
    occ_same = corner_points(800, seed=1)
    occ_rot = rot_z(occ_same, 90)            # coin tourné de 90° autour de z
    plane = np.column_stack([np.random.default_rng(2).uniform(-0.12, 0.12, (800, 2)),
                             np.zeros(800)])  # plan z=0

    d_same_0 = D.descriptor_distance(md, make_scan_desc(occ_same), cfg, 0.0, up)
    d_rot_0 = D.descriptor_distance(md, make_scan_desc(occ_rot), cfg, 0.0, up)
    d_rot_back = D.descriptor_distance(md, make_scan_desc(occ_rot), cfg, np.radians(-90), up)
    d_plane = D.descriptor_distance(md, make_scan_desc(plane), cfg, 0.0, up)

    print(f"d(coin meme, th=0)      = {d_same_0:.4g}")
    print(f"d(coin tourne, th=0)    = {d_rot_0:.4g}")
    print(f"d(coin tourne, th=-90)  = {d_rot_back:.4g}")
    print(f"d(plan, th=0)           = {d_plane:.4g}")

    assert d_same_0 < d_rot_0, "le coin tourne devrait moins bien matcher a th=0"
    assert d_rot_back < d_rot_0 * 0.5, "la rotation -90 doit re-aligner le coin"
    assert d_plane > d_same_0, "un plan ne doit pas matcher un coin aussi bien"


def test_udf_lookup_interpolates_closed_grid_faces_and_corners():
    res, extent = 5, 0.1
    offsets = D._grid_offsets(res, extent)
    # A linear field is exactly reproduced by trilinear interpolation, including
    # the last sample on every axis. The grid itself includes both endpoints.
    values = 0.3 + offsets @ np.array([0.2, 0.4, 0.6])
    model = D.LocalDescriptor("model", res, extent, np.zeros(3),
                              udf=values.reshape(res, res, res).astype(np.float32))
    probes = np.array([
        [extent, 0.025, -0.025], [0.025, extent, -0.025],
        [0.025, -0.025, extent], [extent, extent, extent],
        [-extent, -extent, -extent], [0.0, 0.0, 0.0],
    ])
    expected = 0.3 + probes @ np.array([0.2, 0.4, 0.6])

    assert np.allclose(D._udf_lookup(model, probes), expected, atol=1e-7)
    assert D._udf_lookup(model, np.array([[extent + 0.001, 0.0, 0.0]]))[0] == np.float32(2 * extent)


def test_identity_descriptor_does_not_penalize_occupied_upper_faces():
    cfg = DescriptorConfig(udf_grid_res=5, udf_extent=0.1)
    offsets = D._grid_offsets(5, 0.1)
    model = D.LocalDescriptor("model", 5, 0.1, np.zeros(3),
                              udf=np.zeros((5, 5, 5), np.float32))
    scan = make_scan_desc(offsets, res=5, extent=0.1)

    assert D.descriptor_distance(model, scan, cfg, 0.0, np.array([0., 1., 0.])) == 0.0


def test_descriptor_distance_uses_explicit_metric_unit():
    cfg = DescriptorConfig(
        udf_grid_res=3, udf_extent=0.03,
        distance_unit=0.01, distance_exponent=4.0,
    )
    model = D.LocalDescriptor(
        "model", 3, 0.03, np.zeros(3),
        udf=np.full((3, 3, 3), 0.02, np.float32),
    )
    scan = make_scan_desc(np.array([[0.0, 0.0, 0.0]]), res=3, extent=0.03)

    distance = D.descriptor_distance(
        model, scan, cfg, 0.0, np.array([0.0, 0.0, 1.0]),
    )

    assert np.isclose(distance, 16.0), "(2 cm / 1 cm)^4 doit valoir 16"


def test_descriptor_distance_defaults_to_udf_cell_unit():
    cfg = DescriptorConfig(
        udf_grid_res=4, udf_extent=0.03,
        distance_unit=0.0, distance_exponent=4.0,
    )
    model = D.LocalDescriptor(
        "model", 4, 0.03, np.zeros(3),
        udf=np.full((4, 4, 4), 0.03, np.float32),
    )
    scan = make_scan_desc(np.array([[0.0, 0.0, 0.0]]), res=4, extent=0.03)

    assert np.isclose(D.effective_distance_unit(cfg), 0.015)
    assert np.isclose(
        D.descriptor_distance(model, scan, cfg, 0.0, np.array([0.0, 0.0, 1.0])),
        16.0,
    )


def test_scan_descriptor_uses_visibility_and_seed_fill():
    class VisibilityVolume:
        def sample_visibility(self, positions):
            labels = np.zeros(len(positions), np.uint8)
            # Support central connecté (x=0 puis x=1).
            central = (
                np.isin(np.rint(positions[:, 0]).astype(int), [0, 1])
                & (np.abs(positions[:, 1]) < 0.1)
                & (np.abs(positions[:, 2]) < 0.1)
            )
            # Surface voisine séparée par une cellule inconnue.
            neighbor = (
                np.isclose(positions[:, 0], 3.0)
                & (np.abs(positions[:, 1]) < 0.1)
                & (np.abs(positions[:, 2]) < 0.1)
            )
            free = np.isclose(positions[:, 0], -3.0)
            labels[free] = D.VIS_FREE
            labels[central | neighbor] = D.VIS_OCCUPIED
            return labels

    cfg = DescriptorConfig(occ_grid_res=7, occ_extent=3.0, max_occupied=0)
    desc = D.build_scan_descriptor(VisibilityVolume(), np.zeros(3), cfg)

    assert desc.n_occupied == 2
    assert np.any(desc.occ[0] == D.EMPTY)
    assert desc.occ[-1, 3, 3] == D.UNKNOWN


def test_scan_descriptor_removes_unconfirmed_depth_hole_boundaries():
    class BoundaryVolume:
        def sample_visibility_details(self, positions):
            labels = np.full(len(positions), D.VIS_FREE, np.uint8)
            occupied = (
                np.isclose(positions[:, 1], 0.0)
                & np.isclose(positions[:, 2], 0.0)
                & np.isin(np.rint(positions[:, 0]).astype(int), [0, 1])
            )
            boundary = occupied & np.isclose(positions[:, 0], 0.0)
            labels[occupied] = D.VIS_OCCUPIED
            return labels, boundary

        def sample_visibility(self, positions):
            return self.sample_visibility_details(positions)[0]

    cfg = DescriptorConfig(
        occ_grid_res=3, occ_extent=1.0, max_occupied=0,
        filter_hole_boundaries=True,
    )
    desc = D.build_scan_descriptor(BoundaryVolume(), np.zeros(3), cfg)

    assert desc.n_hole_boundary_removed == 1
    assert desc.n_occupied == 1
    assert desc.occ[1, 1, 1] == D.UNKNOWN
    assert desc.occ[2, 1, 1] == D.OCCUPIED


if __name__ == "__main__":
    test_descriptor_rotation()
    print("OK descriptors")
