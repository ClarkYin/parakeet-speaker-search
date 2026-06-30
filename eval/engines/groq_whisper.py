from __future__ import annotations
import os
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


class GroqWhisper(Engine):
    needs_keys = ["GROQ_API_KEY"]
    diarizes = False
    max_bytes = 24_000_000
    max_chunk_sec = 600.0

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.id = f"groq/{model_name}"

    def _check_deps(self) -> None:
        import groq  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        with open(wav_path, "rb") as f:
            resp = _get_client().audio.transcriptions.create(
                file=(wav_path.split("/")[-1], f),
                model=self.model_name,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        def _w(w):
            if isinstance(w, dict):
                return Word(w["word"], w["start"], w["end"])
            return Word(w.word, w.start, w.end)

        return TranscriptResult(self.id, resp.text, [_w(w) for w in (resp.words or [])])


register(GroqWhisper("whisper-large-v3"))
register(GroqWhisper("whisper-large-v3-turbo"))
