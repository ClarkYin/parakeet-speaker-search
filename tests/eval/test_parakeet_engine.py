from unittest.mock import patch, MagicMock
from eval.engines.parakeet_mlx import ParakeetMLX

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
