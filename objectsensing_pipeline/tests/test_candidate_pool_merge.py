from candidate_index import merge_candidate_pools


def test_global_reserve_does_not_erase_group_candidates():
    groups = [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    result = merge_candidate_pools(list(range(30)), groups, list(range(100, 120)), 10, .7, 2)
    assert set([0, 1, 10, 11, 20, 21]).issubset(result)
    assert len(result) == len(set(result)) == 10
    assert result[-4:] == [100, 101, 102, 103]


def test_merge_handles_overlapping_groups_and_small_budget():
    assert merge_candidate_pools([1, 2], [[1, 2], [1, 3]], [1, 4], 3, .7, 20) == [1, 4, 2]
    assert merge_candidate_pools([1], [], [2], 0) == []
