from __future__ import annotations
import os
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word, Segment
from eval.engines.assemblyai import _segments, _spk


class ElevenLabsScribe(Engine):
    id = "elevenlabs/scribe-v1"
    needs_keys = ["ELEVENLABS_API_KEY"]
    diarizes = True

    def _check_deps(self) -> None:
        import elevenlabs  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        with open(wav_path, "rb") as f:
            r = client.speech_to_text.convert(model_id="scribe_v1", file=f, diarize=True)
        words = [Word(w.text, float(w.start), float(w.end), _spk(getattr(w, "speaker_id", None)))
                 for w in getattr(r, "words", []) if getattr(w, "type", "word") == "word"]
        return TranscriptResult(self.id, getattr(r, "text", ""), words, speakers=_segments(words))


register(ElevenLabsScribe())
