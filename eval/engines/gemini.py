from __future__ import annotations
import os
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word

_PROMPT = (
    "Transcribe this audio verbatim. Return plain text only, no timestamps, "
    "no speaker labels, no commentary."
)


class Gemini(Engine):
    id = "google/gemini-2.5-flash"
    needs_keys = ["GOOGLE_API_KEY"]
    diarizes = False  # text-only path; speakers attached via pyannote downstream

    def _check_deps(self) -> None:
        import google.genai  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        uploaded = client.files.upload(file=wav_path)
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=[_PROMPT, uploaded],
        )
        text = (resp.text or "").strip()
        # no word timestamps from this path; emit a single span over the chunk
        words = [Word(t, 0.0, 0.0) for t in text.split()]
        return TranscriptResult(self.id, text, words)


register(Gemini())
