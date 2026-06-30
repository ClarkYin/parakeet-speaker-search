from eval.models import Word, Segment
from eval.metrics import wer, cer, cpwer, der, words_by_speaker, pairwise_wer

def test_wer_identical_is_zero():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0

def test_wer_one_substitution_in_four():
    assert abs(wer("the quick brown fox", "the quick brown dog") - 0.25) < 1e-9

def test_wer_ignores_casing_and_punctuation():
    assert wer("Hello, world.", "hello world") == 0.0

def test_cer_detects_char_error():
    assert cer("cat", "car") > 0.0

def test_words_by_speaker_groups_in_time_order():
    words = [Word("a", 0, 1, "SPEAKER_00"), Word("b", 2, 3, "SPEAKER_01"), Word("c", 4, 5, "SPEAKER_00")]
    grouped = words_by_speaker(words)
    assert grouped["SPEAKER_00"] == "a c"
    assert grouped["SPEAKER_01"] == "b"

def test_cpwer_perfect_attribution_is_zero():
    ref = {"SPEAKER_00": "hello there", "SPEAKER_01": "general kenobi"}
    hyp = {"SPEAKER_00": "hello there", "SPEAKER_01": "general kenobi"}
    assert cpwer(ref, hyp) == 0.0

def test_cpwer_penalizes_wrong_speaker():
    ref = {"SPEAKER_00": "hello there", "SPEAKER_01": "general kenobi"}
    hyp = {"SPEAKER_00": "hello there general kenobi", "SPEAKER_01": ""}
    assert cpwer(ref, hyp) > 0.0

def test_der_identical_is_zero():
    segs = [Segment("SPEAKER_00", 0.0, 5.0), Segment("SPEAKER_01", 5.0, 10.0)]
    assert der(segs, segs) == 0.0

def test_pairwise_wer_matrix():
    named = {
        "a": "the quick brown fox",
        "b": "the quick brown fox",
        "c": "the quick brown dog",
    }
    m = pairwise_wer(named)
    assert m[("a", "b")] == 0.0          # identical
    assert abs(m[("a", "c")] - 0.25) < 1e-9  # one of four words differs
    assert ("a", "a") not in m           # no self-comparison
    assert m[("a", "c")] == m[("c", "a")]  # symmetric
