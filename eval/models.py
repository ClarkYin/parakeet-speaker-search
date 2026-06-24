from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": _r(self.start), "end": _r(self.end), "speaker": self.speaker}

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(text=d["text"], start=d["start"], end=d["end"], speaker=d.get("speaker"))


@dataclass
class Segment:
    speaker: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "start": _r(self.start), "end": _r(self.end)}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(speaker=d["speaker"], start=d["start"], end=d["end"])


@dataclass
class TranscriptResult:
    engine_id: str
    text: str
    words: list[Word]
    speakers: list[Segment] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "speakers": None if self.speakers is None else [s.to_dict() for s in self.speakers],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptResult":
        sp = d.get("speakers")
        return cls(
            engine_id=d["engine_id"],
            text=d["text"],
            words=[Word.from_dict(w) for w in d["words"]],
            speakers=None if sp is None else [Segment.from_dict(s) for s in sp],
            meta=d.get("meta", {}),
        )


@dataclass
class NormalizedAudio:
    wav_path: str
    duration: float

    def to_dict(self) -> dict[str, Any]:
        return {"wav_path": self.wav_path, "duration": self.duration}

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedAudio":
        return cls(wav_path=d["wav_path"], duration=d["duration"])


@dataclass
class Chunk:
    path: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "start": _r(self.start), "end": _r(self.end)}

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(path=d["path"], start=d["start"], end=d["end"])


@dataclass
class Window:
    start: float
    end: float
    words: list[Word]
    corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _r(self.start),
            "end": _r(self.end),
            "words": [w.to_dict() for w in self.words],
            "corrected": self.corrected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        return cls(
            start=d["start"], end=d["end"],
            words=[Word.from_dict(w) for w in d["words"]],
            corrected=d.get("corrected", False),
        )


@dataclass
class Reference:
    audio_id: str
    windows: list[Window]

    def to_dict(self) -> dict[str, Any]:
        return {"audio_id": self.audio_id, "windows": [w.to_dict() for w in self.windows]}

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        return cls(audio_id=d["audio_id"], windows=[Window.from_dict(w) for w in d["windows"]])


@dataclass
class EngineScore:
    engine_id: str
    cpwer: float | None = None
    wer: float | None = None
    cer: float | None = None
    der: float | None = None
    speaker_count_err: int | None = None
    rtf: float | None = None
    cost_est: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "cpwer": _r(self.cpwer),
            "wer": _r(self.wer),
            "cer": _r(self.cer),
            "der": _r(self.der),
            "speaker_count_err": self.speaker_count_err,
            "rtf": _r(self.rtf),
            "cost_est": _r(self.cost_est),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EngineScore":
        return cls(**d)
