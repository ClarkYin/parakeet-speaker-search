from __future__ import annotations
import os
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


class OpenAITranscribe(Engine):
    id = "openai/gpt-4o-transcribe"
    needs_keys = ["OPENAI_API_KEY"]
    diarizes = False
    max_bytes = 24_000_000
    max_chunk_sec = 600.0

    def _check_deps(self) -> None:
        import openai  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        with open(wav_path, "rb") as f:
            resp = _get_client().audio.transcriptions.create(
                file=f, model="gpt-4o-transcribe",
                response_format="verbose_json", timestamp_granularities=["word"],
            )
        words = [Word(w.word, w.start, w.end) for w in (getattr(resp, "words", None) or [])]
        return TranscriptResult(self.id, resp.text, words)


register(OpenAITranscribe())
