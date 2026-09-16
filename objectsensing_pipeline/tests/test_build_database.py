"""Vérifie build_database sur une disposition HuggingFace (.zip par synset)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from config import PipelineConfig  # noqa: E402
from database import build_database, cluster_descriptors  # noqa: E402


def test_build_from_zip(shapenet_zip_root):
    root = shapenet_zip_root
    cfg = PipelineConfig()
    cfg.keypoint.neighbor_radius = 0.07
    cfg.keypoint.nms_radius = 0.12
    cfg.keypoint.dedup_radius = 0.08
    cfg.keypoint.curvature_threshold = 0.03
    msgs = []
    db = build_database(root, cfg, synsets=["03001627"], max_per_synset=2,
                        n_points=4000, progress=msgs.append)
    for m in msgs:
        print(m)
    print(f"modeles dans la base: {len(db.models)}")
    assert len(db.models) >= 1, "build_database n'a loaded none modèle depuis le .zip"
    m0 = db.models[0]
    print(f"  -> {m0.name}: {m0.cloud.size} pts, {m0.keypoints.size} key points, synset {m0.synset}")
    assert m0.keypoints.size > 0
    cluster_descriptors(db)
    assert db.cluster_reps is not None
    assert len(db.cluster_reps) > 0


if __name__ == "__main__":
    test_build_from_zip()
    print("OK build_database (zip HuggingFace)")
