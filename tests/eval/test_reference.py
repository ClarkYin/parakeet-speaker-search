import json
from eval.models import Word, TranscriptResult, Reference, Window
from eval.reference import (
    sample_windows, words_in_window, rover, build_reference, save_reference, load_reference,
)

def test_sample_windows_count_and_bounds():
    wins = sample_windows(1000.0, [], k=5, length=100.0)
    assert len(wins) == 5
    assert all(0.0 <= a < b <= 1000.0 for a, b in wins)
    # non-overlapping, ordered
    assert all(wins[i][1] <= wins[i + 1][0] for i in range(len(wins) - 1))

def test_words_in_window_filters_by_midpoint():
    tr = TranscriptResult("e", "", [Word("a", 0, 2), Word("b", 10, 12), Word("c", 20, 22)])
    got = words_in_window(tr, 5.0, 15.0)
    assert [w.text for w in got] == ["b"]

def test_rover_majority_vote():
    a = [Word("the", 0, 1), Word("cat", 1, 2)]
    b = [Word("the", 0, 1), Word("cat", 1, 2)]
    c = [Word("the", 0, 1), Word("bat", 1, 2)]
    consensus = rover([a, b, c])
    assert [w.text for w in consensus] == ["the", "cat"]


def _wl(s):
    return [Word(w, i, i + 1) for i, w in enumerate(s.split())]


def test_rover_votes_across_differing_lengths_not_longest_wins():
    # Real ASR transcripts never share a length. The old positional-modal
    # implementation just picked the first list; the alignment-based one must
    # align and take the majority word even when lengths all differ.
    a = _wl("the quick brown fox jumps")        # 5 words, the outlier ('quick')
    b = _wl("the slow brown fox")               # 4 words
    c = _wl("a the slow brown fox today")       # 6 words
    consensus = " ".join(w.text for w in rover([a, b, c]))
    assert "slow" in consensus
    assert "quick" not in consensus

def test_build_and_roundtrip_reference(tmp_path):
    tr = TranscriptResult("e", "", [Word("hello", 1.0, 1.5, "SPEAKER_00")])
    ref = build_reference("aud", [tr], [(0.0, 10.0)])
    assert isinstance(ref, Reference)
    assert ref.windows[0].corrected is False
    p = tmp_path / "ref.json"
    save_reference(ref, str(p))
    assert load_reference(str(p)) == ref
