from eval.audio import compute_chunk_ranges


def test_short_audio_is_single_chunk():
    assert compute_chunk_ranges(120.0, [], max_sec=600.0) == [(0.0, 120.0)]


def test_long_audio_splits_under_max():
    ranges = compute_chunk_ranges(1500.0, [], max_sec=600.0, overlap=0.0)
    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == 1500.0
    assert all((b - a) <= 600.0 + 1e-6 for a, b in ranges)
    # contiguous when no overlap
    assert all(abs(ranges[i][1] - ranges[i + 1][0]) < 1e-6 for i in range(len(ranges) - 1))


def test_cut_snaps_to_nearest_silence():
    # target boundary near 600; silence at 590 should be preferred over a hard 600 cut
    ranges = compute_chunk_ranges(1000.0, [590.0], max_sec=600.0, overlap=0.0)
    assert ranges[0] == (0.0, 590.0)


def test_overlap_applied_between_chunks():
    ranges = compute_chunk_ranges(1200.0, [], max_sec=600.0, overlap=2.0)
    # second chunk starts 2s before first chunk ends
    assert abs(ranges[1][0] - (ranges[0][1] - 2.0)) < 1e-6
