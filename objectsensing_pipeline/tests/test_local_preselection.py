import json
from types import SimpleNamespace

import numpy as np
import pytest

from config import PipelineConfig
from descriptors import LocalDescriptor, UNKNOWN, OCCUPIED
from local_preselection import (
    _query_vectors, _shard_costs, rank_local_groups,
    supplement_pool, supplement_from_database,
    source_fingerprint,
)


def test_supplement_preserves_original_candidates_and_global_budget():
    pool, groups, added = supplement_pool(
        [8, 2, 4], [[8, 2], [4, 2]], [[8, 9, 10], [11, 9, 12]], 3,
    )
    assert pool[:3] == [8, 2, 4]
    assert added == [11, 9, 10]
    assert len(pool) == len(set(pool)) == 6
    assert 11 in groups[1] and 10 in groups[0]
    assert supplement_pool([8], [[8]], [[9]], 0)[0] == [8]


def test_missing_or_incomplete_index_preserves_legacy_pool(tmp_path):
    cfg = PipelineConfig()
    result = supplement_from_database(tmp_path, ["model"], [[object()]], cfg, [0], [[0]])
    assert result[:2] == ([0], [[0]])
    assert result[2]["available"] is False
    root = tmp_path / "local_candidate_index"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"names": ["model"], "resolution": 6}))
    result = supplement_from_database(tmp_path, ["model"], [[object()]], cfg, [0], [[0]])
    assert result[:2] == ([0], [[0]])
    assert "incomplete" in result[2]["warning"]
    with pytest.raises(ValueError, match="does not match"):
        rank_local_groups(root, ["different model"], [], cfg)


def test_added_candidates_keep_the_group_that_actually_proposed_them():
    _, groups, added = supplement_pool(
        [0], [[0, 2], [0, 1]], [[1, 7], [2, 1]], 2,
    )
    assert added == [1, 2]
    assert groups == [[1, 0, 2], [2, 0, 1]]


def test_changed_model_database_disables_a_stale_local_index(tmp_path, monkeypatch):
    import db_store
    model = tmp_path / "model.pkl"
    model.write_bytes(b"original")
    monkeypatch.setattr(db_store, "model_files", lambda _: [model])
    original = source_fingerprint(tmp_path)
    root = tmp_path / "local_candidate_index"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "names": ["model"], "resolution": 6, "source_fingerprint": original,
    }))
    model.write_bytes(b"changed model geometry")
    assert source_fingerprint(tmp_path) != original
    pool, groups, metadata = supplement_from_database(
        tmp_path, ["model"], [[object()]], PipelineConfig(), [0], [[0]],
    )
    assert (pool, groups) == ([0], [[0]])
    assert "source changed" in metadata["warning"]


def test_unknown_cells_do_not_vote_as_free_space_and_yaw_is_searched():
    states = np.full((16, 16, 16), UNKNOWN, dtype=np.int8)
    states[12, 8, 8] = OCCUPIED
    descriptor = LocalDescriptor(
        "scan", 16, .12, np.zeros(3), occ=states,
        n_occupied=1, n_unknown=4095, n_total=4096,
    )
    query = _query_vectors([SimpleNamespace(
        descriptor=descriptor, height=.5, response=1.,
    )])
    assert not query["free"].any()
    axis = np.linspace(-.12, .12, 6)
    _, _, z = np.meshgrid(axis, axis, axis, indexing="ij")
    # The observed X-offset fits this Z-plane only after a yaw rotation.
    model = np.abs(z - .072).reshape(-1)
    data = {
        "udf": np.vstack([model, np.full(216, .1)]),
        "height": np.array([.5, .5]), "occupancy": np.ones(2),
        "model": np.array([0, 1]),
    }
    ids, costs = _shard_costs(data, query, PipelineConfig().matching)
    assert ids.tolist() == [0, 1]
    assert costs[0][0, 0] < .001
    assert costs[0][1, 0] > .09
    np.testing.assert_allclose(costs[0], costs[1])
