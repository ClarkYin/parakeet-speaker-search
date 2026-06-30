from __future__ import annotations
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word

_model = None
_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


def _get_model():
    global _model
    if _model is None:
        from parakeet_mlx import from_pretrained
        _model = from_pretrained(_MODEL_ID)
    return _model


class ParakeetMLX(Engine):
    id = "parakeet/tdt-0.6b-v2"
    needs_keys: list[str] = []
    diarizes = False
    max_chunk_sec = 1200.0

    def _check_deps(self) -> None:
        import parakeet_mlx  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        result = _get_model().transcribe(wav_path)
        return TranscriptResult(self.id, result.text.strip(), _tokens_to_words(result))


def _tokens_to_words(result) -> list[Word]:
    """Regroup Parakeet's sub-word tokens into real words.

    TDT tokens are sub-word pieces (e.g. ' G', 'ood', ' m', 'or', 'ning'); a
    leading space marks the start of a new word. We accumulate pieces until the
    next boundary, taking the word's start from its first token and end from its
    last so word-level timestamps stay correct for windowing/attribution.
    """
    words: list[Word] = []
    cur, start, end = "", None, None
    for sentence in getattr(result, "sentences", []):
        for tok in getattr(sentence, "tokens", []):
            text = tok.text
            if start is None or text.startswith(" "):
                if cur.strip():
                    words.append(Word(cur.strip(), start, end))
                cur, start, end = text, float(tok.start), float(tok.end)
            else:
                cur += text
                end = float(tok.end)
    if cur.strip():
        words.append(Word(cur.strip(), start, end))
    return words


register(ParakeetMLX())
