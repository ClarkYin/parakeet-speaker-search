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
        words: list[Word] = []
        for sentence in getattr(result, "sentences", []):
            for tok in getattr(sentence, "tokens", []):
                words.append(Word(tok.text.strip(), float(tok.start), float(tok.end)))
        return TranscriptResult(self.id, result.text.strip(), words)


register(ParakeetMLX())
