from __future__ import annotations
import os
from eval.engines.base import Engine, register
from eval.models import TranscriptResult, Word, Segment

_client = None


def _get_client():
    global _client
    if _client is None:
        from deepgram import DeepgramClient
        _client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
    return _client


def _spk(n) -> str:
    try:
        return f"SPEAKER_{int(n):02d}"
    except (TypeError, ValueError):
        return "UNKNOWN"


class Deepgram(Engine):
    id = "deepgram/nova-3"
    needs_keys = ["DEEPGRAM_API_KEY"]
    diarizes = True
    max_bytes = None
    max_chunk_sec = None

    def _check_deps(self) -> None:
        import deepgram  # noqa: F401

    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult:
        with open(wav_path, "rb") as f:
            data = f.read()
        resp = _get_client().listen.v1.media.transcribe_file(
            request=data, model="nova-3", smart_format=True, punctuate=True, diarize=True,
        )
        alt = resp.results.channels[0].alternatives[0]
        words: list[Word] = []
        for w in (alt.words or []):
            text = w.word if hasattr(w, "word") else w.get("word", "")
            start = w.start if hasattr(w, "start") else w.get("start", 0.0)
            end = w.end if hasattr(w, "end") else w.get("end", 0.0)
            spk = w.speaker if hasattr(w, "speaker") else w.get("speaker")
            words.append(Word(text, start, end, _spk(spk)))
        segments = _segments_from_words(words)
        return TranscriptResult(self.id, alt.transcript, words, speakers=segments)


def _segments_from_words(words: list[Word]) -> list[Segment]:
    segments: list[Segment] = []
    for w in words:
        if segments and segments[-1].speaker == w.speaker:
            segments[-1].end = w.end
        else:
            segments.append(Segment(w.speaker or "UNKNOWN", w.start, w.end))
    return segments


register(Deepgram())
