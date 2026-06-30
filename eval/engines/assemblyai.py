from __future__ import annotations
import os
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word, Segment


def _spk(label) -> str:
    if label is None:
        return "UNKNOWN"
    s = str(label)
    return f"SPEAKER_{ord(s.upper()) - 65:02d}" if s.isalpha() and len(s) == 1 else f"SPEAKER_{s}"


class AssemblyAI(Engine):
    id = "assemblyai/universal"
    needs_keys = ["ASSEMBLYAI_API_KEY"]
    diarizes = True

    def _check_deps(self) -> None:
        import assemblyai  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        import assemblyai as aai
        aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
        cfg = aai.TranscriptionConfig(speaker_labels=True)
        t = aai.Transcriber().transcribe(wav_path, config=cfg)
        words = [Word(w.text, w.start / 1000.0, w.end / 1000.0, _spk(getattr(w, "speaker", None)))
                 for w in (t.words or [])]
        return TranscriptResult(self.id, t.text or "", words, speakers=_segments(words))


def _segments(words: list[Word]) -> list[Segment]:
    segs: list[Segment] = []
    for w in words:
        if segs and segs[-1].speaker == w.speaker:
            segs[-1].end = w.end
        else:
            segs.append(Segment(w.speaker or "UNKNOWN", w.start, w.end))
    return segs


register(AssemblyAI())
