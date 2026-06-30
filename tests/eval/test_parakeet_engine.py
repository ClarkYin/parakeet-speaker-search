from unittest.mock import patch, MagicMock
from eval.engines.parakeet_mlx import ParakeetMLX


def _tok(text, start, end):
    m = MagicMock(); m.text = text; m.start = start; m.end = end
    return m


def test_parakeet_regroups_subword_tokens_into_words():
    # Parakeet TDT emits sub-word tokens; a leading space marks a word boundary.
    toks = [_tok(" G", 0.0, 0.1), _tok("ood", 0.1, 0.2),
            _tok(" m", 0.3, 0.4), _tok("or", 0.4, 0.5), _tok("ning", 0.5, 0.6),
            _tok(" every", 0.7, 0.8), _tok("one", 0.8, 0.9), _tok(".", 0.9, 0.95)]
    sentence = MagicMock(); sentence.tokens = toks
    result = MagicMock(); result.text = "Good morning everyone."; result.sentences = [sentence]
    model = MagicMock(); model.transcribe.return_value = result
    with patch("eval.engines.parakeet_mlx._get_model", return_value=model):
        out = ParakeetMLX()._transcribe_chunk("/tmp/c.wav", offset=0.0)
    # 8 sub-word tokens collapse to 3 real words with correct boundaries/timestamps
    assert [w.text for w in out.words] == ["Good", "morning", "everyone."]
    assert out.words[0].start == 0.0
    assert out.words[1].start == 0.3
    assert out.words[2].end == 0.95


def test_parakeet_maps_tokens_to_words():
    tok = MagicMock(); tok.text = "hello"; tok.start = 0.0; tok.end = 0.4
    sentence = MagicMock(); sentence.tokens = [tok]
    result = MagicMock(); result.text = "hello"; result.sentences = [sentence]
    model = MagicMock(); model.transcribe.return_value = result

    with patch("eval.engines.parakeet_mlx._get_model", return_value=model):
        out = ParakeetMLX()._transcribe_chunk("/tmp/c.wav", offset=0.0)

    assert out.text == "hello"
    assert out.words[0].text == "hello"
    assert out.words[0].end == 0.4

def test_parakeet_is_not_a_diarizer():
    assert ParakeetMLX().diarizes is False
