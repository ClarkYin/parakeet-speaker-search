from app.merger import merge

SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
    {"speaker": "SPEAKER_01", "start": 5.2, "end": 10.0},
]

WORDS = [
    {"word": "Hello", "start": 0.0, "end": 0.3},
    {"word": "everyone", "start": 0.4, "end": 0.9},
    {"word": "thanks", "start": 5.3, "end": 5.7},
    {"word": "for", "start": 5.8, "end": 6.0},
    {"word": "joining", "start": 6.1, "end": 6.5},
]

def test_merge_groups_words_by_speaker():
    result = merge(WORDS, SEGMENTS)
    assert len(result) == 2
    assert result[0]["speaker_label"] == "SPEAKER_00"
    assert result[0]["text"] == "Hello everyone"
    assert result[1]["speaker_label"] == "SPEAKER_01"
    assert result[1]["text"] == "thanks for joining"

def test_merge_captures_timestamps():
    result = merge(WORDS, SEGMENTS)
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 0.9
    assert result[1]["start_sec"] == 5.3
    assert result[1]["end_sec"] == 6.5

def test_merge_word_outside_all_segments_assigned_nearest():
    words = [{"word": "ghost", "start": 12.0, "end": 12.5}]
    result = merge(words, SEGMENTS)
    # No segment contains the word, so it falls back to the nearest segment
    # (SPEAKER_01 ends at 10.0, closer than SPEAKER_00 ending at 5.0).
    assert result[0]["speaker_label"] == "SPEAKER_01"

def test_merge_empty_words_returns_empty():
    assert merge([], SEGMENTS) == []

def test_merge_empty_segments_labels_all_unknown():
    result = merge(WORDS, [])
    assert all(u["speaker_label"] == "UNKNOWN" for u in result)
