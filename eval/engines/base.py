from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import replace
from eval.models import NormalizedAudio, TranscriptResult, Word
from eval.audio import chunk_audio

_REGISTRY: dict[str, "Engine"] = {}


class Engine(ABC):
    id: str = ""
    needs_keys: list[str] = []
    diarizes: bool = False
    max_chunk_sec: float | None = None
    max_bytes: int | None = None

    def available(self) -> bool:
        try:
            self._check_deps()
        except Exception:
            return False
        return all(os.environ.get(k) for k in self.needs_keys)

    def _check_deps(self) -> None:
        """Override to import-probe SDKs; raise if missing."""
        return None

    @abstractmethod
    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult: ...

    def transcribe(self, audio: NormalizedAudio, work_dir: str) -> TranscriptResult:
        chunks = chunk_audio(audio, work_dir, self.max_chunk_sec, self.max_bytes)
        words: list[Word] = []
        texts: list[str] = []
        prev_end = 0.0
        for ch in chunks:
            part = self._transcribe_chunk(ch.path, ch.start)
            for w in part.words:
                shifted = replace(w, start=w.start + ch.start, end=w.end + ch.start)
                if shifted.start < prev_end - 1e-6:
                    continue  # inside previous chunk's overlap; drop the duplicate
                words.append(shifted)
            texts.append(part.text)
            prev_end = ch.end
        return TranscriptResult(self.id, " ".join(t.strip() for t in texts).strip(), words,
                                meta={"chunked": len(chunks) > 1})


def register(engine: "Engine") -> "Engine":
    _REGISTRY[engine.id] = engine
    return engine


def get_registry() -> dict[str, "Engine"]:
    return _REGISTRY


def available_engines() -> list["Engine"]:
    return [e for e in _REGISTRY.values() if e.available()]
