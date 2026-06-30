# Transcription Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `eval/` package that runs multiple ASR/diarization engines on an audio file, builds a bootstrapped + spot-corrected reference, and scores each engine by cpWER/WER/CER/DER into a ranked report.

**Architecture:** A standalone `eval/` package, decoupled from the FastAPI app and Postgres, importing `app/` only for diarization (`app.diarization.diarize`) and word→speaker merge (`app.merger.merge`). Pure-logic units (chunk math, normalization, metrics, ROVER consensus, alignment, reporting) are separated from I/O (ffmpeg, network SDKs) so the logic is unit-tested without audio or network. Engines register in a registry that auto-skips any whose API keys or Python deps are missing.

**Tech Stack:** Python 3.11+, ffmpeg (CLI), `jiwer` (WER/CER + alignment), `meeteval` (cpWER), `pyannote.metrics` (DER), `whisper_normalizer` (text normalization), `parakeet-mlx` (local ASR), `groq` / `deepgram` SDKs, pytest + `unittest.mock`.

## Global Constraints

- **Python:** `requires-python = ">=3.11"` — use `X | None` unions, `list[...]`/`dict[...]` generics, `match` allowed.
- **Package name:** top-level package is `eval` (run via `python -m eval`). Tests import `from eval.<module> import ...` and run from the repo root (cwd on `sys.path`, matching existing `from app...` tests).
- **Test layout:** mirror existing `tests/` — pytest, `unittest.mock.patch`/`MagicMock`, no live network or real audio in unit tests. New tests live under `tests/eval/`.
- **No network/heavy deps at import time:** every engine SDK and `mlx`/`parakeet`/`silero` import is **lazy** (inside functions), exactly like `app/transcription.py` and `app/diarization.py`.
- **Data interchange:** all dataclasses provide `to_dict()`/`from_dict()` returning/consuming plain JSON-serializable `dict`s. Timestamps are floats in seconds, rounded to 3 decimals on serialization.
- **Speaker labels:** uppercase `SPEAKER_NN` form (matches pyannote and `app/merger.py`); unknown attribution is the literal `"UNKNOWN"`.
- **Commit style:** Conventional Commits, footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit at the end of every task.

---

### Task 1: Package scaffold + data models

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/models.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_models.py`
- Modify: `pyproject.toml` (add deps + optional `eval` extra)
- Modify: `.gitignore` (add `runs/`)

**Interfaces:**
- Produces:
  - `Word(text: str, start: float, end: float, speaker: str | None = None)`
  - `Segment(speaker: str, start: float, end: float)`
  - `TranscriptResult(engine_id: str, text: str, words: list[Word], speakers: list[Segment] | None = None, meta: dict = {})`
  - `NormalizedAudio(wav_path: str, duration: float)`
  - `Chunk(path: str, start: float, end: float)`
  - `Window(start: float, end: float, words: list[Word], corrected: bool = False)`
  - `Reference(audio_id: str, windows: list[Window])`
  - `EngineScore(engine_id: str, cpwer: float | None, wer: float | None, cer: float | None, der: float | None, speaker_count_err: int | None, rtf: float | None, cost_est: float | None)`
  - Each dataclass: `.to_dict() -> dict` and classmethod `.from_dict(d: dict) -> Self`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_models.py
from eval.models import Word, TranscriptResult, Reference, Window

def test_word_roundtrips_through_dict():
    w = Word(text="hello", start=0.0, end=0.3, speaker="SPEAKER_00")
    assert Word.from_dict(w.to_dict()) == w

def test_word_rounds_timestamps_on_serialize():
    w = Word(text="hi", start=0.123456, end=0.98765)
    assert w.to_dict()["start"] == 0.123
    assert w.to_dict()["end"] == 0.988

def test_transcript_result_roundtrips_with_words():
    tr = TranscriptResult(
        engine_id="groq/whisper-large-v3-turbo",
        text="hello world",
        words=[Word("hello", 0.0, 0.3), Word("world", 0.4, 0.8)],
        meta={"rtf": 0.12},
    )
    back = TranscriptResult.from_dict(tr.to_dict())
    assert back == tr
    assert back.speakers is None

def test_reference_roundtrips_nested_windows():
    ref = Reference(
        audio_id="roncesvalles",
        windows=[Window(0.0, 180.0, [Word("hi", 1.0, 1.2, "SPEAKER_00")], corrected=True)],
    )
    assert Reference.from_dict(ref.to_dict()) == ref
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/__init__.py
"""Transcription evaluation harness."""
```

```python
# eval/models.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
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


@dataclass
class Chunk:
    path: str
    start: float
    end: float


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
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineScore":
        return cls(**d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Add dependencies + gitignore**

In `pyproject.toml`, add to `dependencies`:
```toml
    "jiwer>=3.0.0",
    "meeteval>=0.3.0",
    "whisper_normalizer>=0.0.10",
    "pyannote.metrics>=3.2.1",
```
Add a new optional extra below the existing `dev` extra:
```toml
engines = [
    "parakeet-mlx>=0.3.0",
    "silero-vad>=5.1.0",
    "assemblyai>=0.34.0",
    "elevenlabs>=1.50.0",
    "openai>=1.50.0",
    "google-genai>=0.3.0",
]
```
Append `runs/` on its own line to `.gitignore`.

- [ ] **Step 6: Install new deps**

Run: `python -m pip install -e ".[dev]"`
Expected: installs `jiwer`, `meeteval`, `whisper_normalizer`, `pyannote.metrics`.

- [ ] **Step 7: Commit**

```bash
git add eval/__init__.py eval/models.py tests/eval/__init__.py tests/eval/test_models.py pyproject.toml .gitignore
git commit -m "feat(eval): data models + package scaffold for eval harness"
```

---

### Task 2: Audio normalization + silence-aware chunker

**Files:**
- Create: `eval/audio.py`
- Create: `tests/eval/test_audio.py`

**Interfaces:**
- Consumes: `NormalizedAudio`, `Chunk` from `eval.models`.
- Produces:
  - `normalize(input_path: str, out_dir: str) -> NormalizedAudio` — ffmpeg → 16 kHz mono PCM WAV at `<out_dir>/normalized.wav`; duration via `ffprobe`.
  - `detect_silences(wav_path: str, noise_db: float = -30.0, min_silence: float = 0.5) -> list[float]` — midpoints (sec) of detected silences via `ffmpeg silencedetect`.
  - `compute_chunk_ranges(duration: float, silence_points: list[float], max_sec: float, overlap: float = 1.0) -> list[tuple[float, float]]` — **pure function**: split `[0, duration]` into ranges each ≤ `max_sec`, preferring a cut at the silence point nearest each target boundary; consecutive ranges overlap by `overlap` sec.
  - `chunk_audio(audio: NormalizedAudio, out_dir: str, max_sec: float, max_bytes: int | None = None) -> list[Chunk]` — slices the WAV with ffmpeg into `chunk_000.wav`, … honoring `compute_chunk_ranges`; if `max_sec` is `None`, returns a single chunk spanning the whole file.

- [ ] **Step 1: Write the failing test (pure chunk math)**

```python
# tests/eval/test_audio.py
from eval.audio import compute_chunk_ranges

def test_short_audio_is_single_chunk():
    assert compute_chunk_ranges(120.0, [], max_sec=600.0) == [(0.0, 120.0)]

def test_long_audio_splits_under_max():
    ranges = compute_chunk_ranges(1500.0, [], max_sec=600.0, overlap=0.0)
    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == 1500.0
    assert all((b - a) <= 600.0 + 1e-6 for a, b in ranges)
    # contiguous when no overlap
    assert all(abs(ranges[i][1] - ranges[i + 1][0]) < 1e-6 for i in range(len(ranges) - 1))

def test_cut_snaps_to_nearest_silence():
    # target boundary near 600; silence at 590 should be preferred over a hard 600 cut
    ranges = compute_chunk_ranges(1000.0, [590.0], max_sec=600.0, overlap=0.0)
    assert ranges[0] == (0.0, 590.0)

def test_overlap_applied_between_chunks():
    ranges = compute_chunk_ranges(1200.0, [], max_sec=600.0, overlap=2.0)
    # second chunk starts 2s before first chunk ends
    assert abs(ranges[1][0] - (ranges[0][1] - 2.0)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_audio.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_chunk_ranges'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/audio.py
from __future__ import annotations
import os
import re
import subprocess
from eval.models import NormalizedAudio, Chunk


def compute_chunk_ranges(
    duration: float,
    silence_points: list[float],
    max_sec: float | None,
    overlap: float = 1.0,
) -> list[tuple[float, float]]:
    if max_sec is None or duration <= max_sec:
        return [(0.0, duration)]

    ranges: list[tuple[float, float]] = []
    start = 0.0
    silences = sorted(silence_points)
    while start < duration - 1e-6:
        target = start + max_sec
        if target >= duration:
            ranges.append((start, duration))
            break
        # prefer a silence point within the back third of this window
        window_lo = start + max_sec * 0.66
        candidates = [s for s in silences if window_lo <= s <= target]
        cut = max(candidates) if candidates else target
        ranges.append((start, cut))
        start = max(cut - overlap, 0.0) if overlap else cut
    return ranges


def _ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def normalize(input_path: str, out_dir: str) -> NormalizedAudio:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "normalized.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path],
        check=True, capture_output=True,
    )
    return NormalizedAudio(wav_path=out_path, duration=_ffprobe_duration(out_path))


def detect_silences(wav_path: str, noise_db: float = -30.0, min_silence: float = 0.5) -> list[float]:
    proc = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def chunk_audio(audio: NormalizedAudio, out_dir: str, max_sec: float | None, max_bytes: int | None = None) -> list[Chunk]:
    os.makedirs(out_dir, exist_ok=True)
    silences = detect_silences(audio.wav_path) if max_sec else []
    ranges = compute_chunk_ranges(audio.duration, silences, max_sec)
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(ranges):
        path = os.path.join(out_dir, f"chunk_{i:03d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio.wav_path, "-ss", str(start), "-to", str(end),
             "-ac", "1", "-ar", "16000", path],
            check=True, capture_output=True,
        )
        chunks.append(Chunk(path=path, start=start, end=end))
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_audio.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/audio.py tests/eval/test_audio.py
git commit -m "feat(eval): audio normalization + silence-aware chunker"
```

---

### Task 3: Engine base class + registry

**Files:**
- Create: `eval/engines/__init__.py`
- Create: `eval/engines/base.py`
- Create: `tests/eval/test_engine_base.py`

**Interfaces:**
- Consumes: `NormalizedAudio`, `TranscriptResult`, `Word` from `eval.models`; `chunk_audio` from `eval.audio`.
- Produces:
  - `class Engine(ABC)` with class attrs `id: str`, `needs_keys: list[str] = []`, `diarizes: bool = False`, `max_chunk_sec: float | None = None`, `max_bytes: int | None = None`; methods `available() -> bool`, abstract `_transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult`, and concrete `transcribe(self, audio: NormalizedAudio, work_dir: str) -> TranscriptResult` (chunks via `chunk_audio`, calls `_transcribe_chunk` per chunk, shifts each chunk's word timestamps by `offset`, concatenates, drops words landing inside the de-dup overlap of the previous chunk).
  - `register(engine: Engine) -> Engine` and `get_registry() -> dict[str, Engine]` and `available_engines() -> list[Engine]`.
  - `available()` returns `True` only if every env var in `needs_keys` is set **and** non-empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_engine_base.py
import os
from unittest.mock import patch
from eval.models import NormalizedAudio, TranscriptResult, Word
from eval.engines.base import Engine, register, get_registry, available_engines


class FakeEngine(Engine):
    id = "fake/one"
    needs_keys = ["FAKE_KEY"]

    def _transcribe_chunk(self, wav_path, offset):
        return TranscriptResult(self.id, "hi", [Word("hi", 0.0, 0.2)])


def test_available_false_when_key_missing(monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    assert FakeEngine().available() is False


def test_available_true_when_key_present(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    assert FakeEngine().available() is True


def test_register_and_lookup():
    e = FakeEngine()
    register(e)
    assert get_registry()["fake/one"] is e


def test_transcribe_offsets_timestamps_across_chunks():
    audio = NormalizedAudio(wav_path="/tmp/a.wav", duration=1000.0)
    calls = []

    class TwoChunk(Engine):
        id = "fake/two"
        max_chunk_sec = 600.0

        def _transcribe_chunk(self, wav_path, offset):
            calls.append(offset)
            return TranscriptResult(self.id, "w", [Word("w", 0.0, 0.2)])

    fake_chunks = [
        type("C", (), {"path": "/tmp/c0.wav", "start": 0.0, "end": 600.0})(),
        type("C", (), {"path": "/tmp/c1.wav", "start": 600.0, "end": 1000.0})(),
    ]
    with patch("eval.engines.base.chunk_audio", return_value=fake_chunks):
        result = TwoChunk().transcribe(audio, work_dir="/tmp")

    assert calls == [0.0, 600.0]
    # second chunk's word shifted by its 600s offset
    assert result.words[1].start == 600.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_engine_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.engines'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/engines/__init__.py
"""Engine adapters."""
```

```python
# eval/engines/base.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_engine_base.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/engines/__init__.py eval/engines/base.py tests/eval/test_engine_base.py
git commit -m "feat(eval): engine base class + registry with key/dep gating"
```

---

### Task 4: Text normalization

**Files:**
- Create: `eval/normalize.py`
- Create: `tests/eval/test_normalize.py`

**Interfaces:**
- Produces: `normalize_text(text: str) -> str` — applies the Whisper `EnglishTextNormalizer` (lazy import); lowercases, expands contractions, normalizes numbers, strips punctuation and filler. Returns a single space-collapsed string. On import failure, falls back to a deterministic local normalizer (lowercase, strip non-alphanumeric to spaces, collapse whitespace).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_normalize.py
from eval.normalize import normalize_text

def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"

def test_collapses_whitespace():
    assert normalize_text("a   b\tc") == "a b c"

def test_empty_string():
    assert normalize_text("") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/normalize.py
from __future__ import annotations
import re

_normalizer = None


def _fallback(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", text.lower())).strip()


def normalize_text(text: str) -> str:
    global _normalizer
    if _normalizer is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _normalizer = EnglishTextNormalizer()
        except Exception:
            _normalizer = _fallback
    out = _normalizer(text)
    return re.sub(r"\s+", " ", out).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_normalize.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/normalize.py tests/eval/test_normalize.py
git commit -m "feat(eval): text normalization for fair WER scoring"
```

---

### Task 5: Metrics (WER, CER, cpWER, DER)

**Files:**
- Create: `eval/metrics.py`
- Create: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `Word`, `Segment` from `eval.models`; `normalize_text` from `eval.normalize`.
- Produces:
  - `wer(reference: str, hypothesis: str) -> float` — normalizes both, then `jiwer.wer`.
  - `cer(reference: str, hypothesis: str) -> float` — normalizes both, then `jiwer.cer`.
  - `cpwer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]) -> float` — normalizes each value, then `meeteval.wer.cp_word_error_rate`; returns its `.error_rate` (`0.0` when both empty).
  - `der(ref_segments: list[Segment], hyp_segments: list[Segment]) -> float` — builds `pyannote.core.Annotation`s and returns `DiarizationErrorRate()` value.
  - `words_by_speaker(words: list[Word]) -> dict[str, str]` — concatenates each speaker's word texts in time order (`UNKNOWN` for `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
from eval.models import Word, Segment
from eval.metrics import wer, cer, cpwer, der, words_by_speaker

def test_wer_identical_is_zero():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0

def test_wer_one_substitution_in_four():
    assert abs(wer("the quick brown fox", "the quick brown dog") - 0.25) < 1e-9

def test_wer_ignores_casing_and_punctuation():
    assert wer("Hello, world.", "hello world") == 0.0

def test_cer_detects_char_error():
    assert cer("cat", "car") > 0.0

def test_words_by_speaker_groups_in_time_order():
    words = [Word("a", 0, 1, "SPEAKER_00"), Word("b", 2, 3, "SPEAKER_01"), Word("c", 4, 5, "SPEAKER_00")]
    grouped = words_by_speaker(words)
    assert grouped["SPEAKER_00"] == "a c"
    assert grouped["SPEAKER_01"] == "b"

def test_cpwer_perfect_attribution_is_zero():
    ref = {"SPEAKER_00": "hello there", "SPEAKER_01": "general kenobi"}
    hyp = {"SPEAKER_00": "hello there", "SPEAKER_01": "general kenobi"}
    assert cpwer(ref, hyp) == 0.0

def test_cpwer_penalizes_wrong_speaker():
    ref = {"SPEAKER_00": "hello there", "SPEAKER_01": "general kenobi"}
    hyp = {"SPEAKER_00": "hello there general kenobi", "SPEAKER_01": ""}
    assert cpwer(ref, hyp) > 0.0

def test_der_identical_is_zero():
    segs = [Segment("SPEAKER_00", 0.0, 5.0), Segment("SPEAKER_01", 5.0, 10.0)]
    assert der(segs, segs) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/metrics.py
from __future__ import annotations
from eval.models import Word, Segment
from eval.normalize import normalize_text


def wer(reference: str, hypothesis: str) -> float:
    import jiwer
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    return float(jiwer.wer(ref, hyp))


def cer(reference: str, hypothesis: str) -> float:
    import jiwer
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    return float(jiwer.cer(ref, hyp))


def words_by_speaker(words: list[Word]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for w in sorted(words, key=lambda x: x.start):
        out.setdefault(w.speaker or "UNKNOWN", []).append(w.text)
    return {spk: " ".join(toks) for spk, toks in out.items()}


def cpwer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]) -> float:
    from meeteval.wer import cp_word_error_rate
    ref = {k: normalize_text(v) for k, v in ref_by_speaker.items()}
    hyp = {k: normalize_text(v) for k, v in hyp_by_speaker.items()}
    if not any(ref.values()) and not any(hyp.values()):
        return 0.0
    return float(cp_word_error_rate(ref, hyp).error_rate)


def _annotation(segments: list[Segment]):
    from pyannote.core import Annotation, Segment as PSegment
    ann = Annotation()
    for s in segments:
        ann[PSegment(s.start, s.end)] = s.speaker
    return ann


def der(ref_segments: list[Segment], hyp_segments: list[Segment]) -> float:
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate()
    return float(metric(_annotation(ref_segments), _annotation(hyp_segments)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): WER/CER/cpWER/DER metrics"
```

---

### Task 6: Word alignment for diffs

**Files:**
- Create: `eval/align.py`
- Create: `tests/eval/test_align.py`

**Interfaces:**
- Consumes: `normalize_text` from `eval.normalize`.
- Produces:
  - `align(reference: str, hypothesis: str) -> list[dict]` — token-level alignment over normalized text; each item `{"op": "equal"|"sub"|"ins"|"del", "ref": str | None, "hyp": str | None}` using `jiwer.process_words` alignment chunks.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_align.py
from eval.align import align

def test_equal_tokens():
    ops = align("the cat sat", "the cat sat")
    assert [o["op"] for o in ops] == ["equal", "equal", "equal"]

def test_substitution_marked():
    ops = align("the cat sat", "the dog sat")
    sub = [o for o in ops if o["op"] == "sub"]
    assert sub == [{"op": "sub", "ref": "cat", "hyp": "dog"}]

def test_insertion_and_deletion():
    ins = [o for o in align("a c", "a b c") if o["op"] == "ins"]
    assert ins == [{"op": "ins", "ref": None, "hyp": "b"}]
    dele = [o for o in align("a b c", "a c") if o["op"] == "del"]
    assert dele == [{"op": "del", "ref": "b", "hyp": None}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_align.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.align'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/align.py
from __future__ import annotations
from eval.normalize import normalize_text


def align(reference: str, hypothesis: str) -> list[dict]:
    import jiwer
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    out = jiwer.process_words(" ".join(ref) or " ", " ".join(hyp) or " ")
    ops: list[dict] = []
    # process_words returns alignment per sentence; we passed one "sentence"
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for r in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ops.append({"op": "equal", "ref": ref[r], "hyp": hyp[r - chunk.ref_start_idx + chunk.hyp_start_idx]})
        elif chunk.type == "substitute":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ops.append({"op": "sub", "ref": ref[chunk.ref_start_idx + k], "hyp": hyp[chunk.hyp_start_idx + k]})
        elif chunk.type == "insert":
            for k in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                ops.append({"op": "ins", "ref": None, "hyp": hyp[k]})
        elif chunk.type == "delete":
            for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ops.append({"op": "del", "ref": ref[k], "hyp": None})
    return ops
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_align.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/align.py tests/eval/test_align.py
git commit -m "feat(eval): word-level alignment for diff visualization"
```

---

### Task 7: Reference builder (window sampling + ROVER consensus + I/O)

**Files:**
- Create: `eval/reference.py`
- Create: `tests/eval/test_reference.py`

**Interfaces:**
- Consumes: `Word`, `Window`, `Reference`, `TranscriptResult` from `eval.models`; `normalize_text` from `eval.normalize`.
- Produces:
  - `sample_windows(duration: float, silence_points: list[float], k: int = 5, length: float = 180.0) -> list[tuple[float, float]]` — **pure**: k stratified `[start, end]` windows of `length` sec spread across `(0, duration)`, snapped to the nearest silence point, non-overlapping, clamped to `duration`.
  - `words_in_window(result: TranscriptResult, start: float, end: float) -> list[Word]` — words whose midpoint falls in `[start, end]`.
  - `rover(results: list[list[Word]]) -> list[Word]` — **pure**: majority-vote consensus over positionally-aligned word lists (simple per-index modal token across engines that agree on count; on length disagreement, choose the modal length group first). Returns the consensus word list (timestamps from the first contributor in the modal group).
  - `build_reference(audio_id, results: list[TranscriptResult], windows: list[tuple[float,float]], seed: TranscriptResult | None = None) -> Reference` — for each window, gather each result's words, ROVER them (seed counted with extra weight if given), produce a `Window(corrected=False)`.
  - `save_reference(ref: Reference, path: str) -> None` / `load_reference(path: str) -> Reference` — JSON via `to_dict`/`from_dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_reference.py
import json
from eval.models import Word, TranscriptResult, Reference, Window
from eval.reference import (
    sample_windows, words_in_window, rover, build_reference, save_reference, load_reference,
)

def test_sample_windows_count_and_bounds():
    wins = sample_windows(1000.0, [], k=5, length=100.0)
    assert len(wins) == 5
    assert all(0.0 <= a < b <= 1000.0 for a, b in wins)
    # non-overlapping, ordered
    assert all(wins[i][1] <= wins[i + 1][0] for i in range(len(wins) - 1))

def test_words_in_window_filters_by_midpoint():
    tr = TranscriptResult("e", "", [Word("a", 0, 2), Word("b", 10, 12), Word("c", 20, 22)])
    got = words_in_window(tr, 5.0, 15.0)
    assert [w.text for w in got] == ["b"]

def test_rover_majority_vote():
    a = [Word("the", 0, 1), Word("cat", 1, 2)]
    b = [Word("the", 0, 1), Word("cat", 1, 2)]
    c = [Word("the", 0, 1), Word("bat", 1, 2)]
    consensus = rover([a, b, c])
    assert [w.text for w in consensus] == ["the", "cat"]

def test_build_and_roundtrip_reference(tmp_path):
    tr = TranscriptResult("e", "", [Word("hello", 1.0, 1.5, "SPEAKER_00")])
    ref = build_reference("aud", [tr], [(0.0, 10.0)])
    assert isinstance(ref, Reference)
    assert ref.windows[0].corrected is False
    p = tmp_path / "ref.json"
    save_reference(ref, str(p))
    assert load_reference(str(p)) == ref
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.reference'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/reference.py
from __future__ import annotations
import json
from collections import Counter
from eval.models import Word, Window, Reference, TranscriptResult


def _nearest(points: list[float], target: float) -> float:
    return min(points, key=lambda p: abs(p - target)) if points else target


def sample_windows(duration: float, silence_points: list[float], k: int = 5, length: float = 180.0) -> list[tuple[float, float]]:
    if duration <= length:
        return [(0.0, duration)]
    centers = [duration * (i + 1) / (k + 1) for i in range(k)]
    wins: list[tuple[float, float]] = []
    last_end = 0.0
    for c in centers:
        start = max(_nearest(silence_points, c - length / 2), last_end)
        end = min(start + length, duration)
        if end - start < 1.0:
            continue
        wins.append((round(start, 3), round(end, 3)))
        last_end = end
    return wins


def words_in_window(result: TranscriptResult, start: float, end: float) -> list[Word]:
    return [w for w in result.words if start <= (w.start + w.end) / 2 <= end]


def rover(results: list[list[Word]]) -> list[Word]:
    if not results:
        return []
    modal_len = Counter(len(r) for r in results).most_common(1)[0][0]
    group = [r for r in results if len(r) == modal_len]
    consensus: list[Word] = []
    for i in range(modal_len):
        texts = [r[i].text for r in group]
        winner = Counter(texts).most_common(1)[0][0]
        src = next(r[i] for r in group if r[i].text == winner)
        consensus.append(Word(winner, src.start, src.end, src.speaker))
    return consensus


def build_reference(audio_id: str, results: list[TranscriptResult], windows: list[tuple[float, float]], seed: TranscriptResult | None = None) -> Reference:
    out_windows: list[Window] = []
    for start, end in windows:
        per_engine = [words_in_window(r, start, end) for r in results]
        if seed is not None:
            seed_words = words_in_window(seed, start, end)
            per_engine = [seed_words, seed_words] + per_engine  # extra weight
        consensus = rover([p for p in per_engine if p])
        out_windows.append(Window(start=start, end=end, words=consensus, corrected=False))
    return Reference(audio_id=audio_id, windows=out_windows)


def save_reference(ref: Reference, path: str) -> None:
    with open(path, "w") as f:
        json.dump(ref.to_dict(), f, indent=2)


def load_reference(path: str) -> Reference:
    with open(path) as f:
        return Reference.from_dict(json.load(f))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_reference.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/reference.py tests/eval/test_reference.py
git commit -m "feat(eval): reference builder — window sampling, ROVER, JSON I/O"
```

---

### Task 8: Parakeet MLX engine (local, runnable now)

**Files:**
- Create: `eval/engines/parakeet_mlx.py`
- Create: `tests/eval/test_parakeet_engine.py`

**Interfaces:**
- Consumes: `Engine` from `eval.engines.base`; `TranscriptResult`, `Word` from `eval.models`.
- Produces: `class ParakeetMLX(Engine)` with `id="parakeet/tdt-0.6b-v2"`, `needs_keys=[]`, `diarizes=False`, `max_chunk_sec=1200.0`. `_check_deps` import-probes `parakeet_mlx`. `_transcribe_chunk` loads a cached model (module-level singleton, lazy) and maps its sentence/token output to `Word`s. Registered via `register(ParakeetMLX())` at import.

- [ ] **Step 1: Write the failing test (model mocked)**

```python
# tests/eval/test_parakeet_engine.py
from unittest.mock import patch, MagicMock
from eval.engines.parakeet_mlx import ParakeetMLX

def test_parakeet_maps_tokens_to_words():
    tok = MagicMock(); tok.text = "hello"; tok.start = 0.0; tok.end = 0.4
    sentence = MagicMock(); sentence.tokens = [tok]
    result = MagicMock(); result.text = "hello"; result.sentences = [sentence]
    model = MagicMock(); model.transcribe.return_value = result

    with patch("eval.engines.parakeet_mlx._get_model", return_value=model):
        out = ParakeetMLX()._transcribe_chunk("/tmp/c.wav", offset=0.0)

    assert out.text == "hello"
    assert out.words[0].text == "hello"
    assert out.words[0].end == 0.4

def test_parakeet_is_not_a_diarizer():
    assert ParakeetMLX().diarizes is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_parakeet_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.engines.parakeet_mlx'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/engines/parakeet_mlx.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_parakeet_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/engines/parakeet_mlx.py tests/eval/test_parakeet_engine.py
git commit -m "feat(eval): Parakeet MLX local engine adapter"
```

---

### Task 9: Groq Whisper + Deepgram engines (runnable now)

**Files:**
- Create: `eval/engines/groq_whisper.py`
- Create: `eval/engines/deepgram.py`
- Create: `tests/eval/test_groq_deepgram_engines.py`

**Interfaces:**
- Consumes: `Engine` from `eval.engines.base`; `TranscriptResult`, `Word`, `Segment` from `eval.models`.
- Produces:
  - `class GroqWhisper(Engine)` — two registered instances `groq/whisper-large-v3` and `groq/whisper-large-v3-turbo`; `needs_keys=["GROQ_API_KEY"]`, `diarizes=False`, `max_bytes=24_000_000`, `max_chunk_sec=600.0`. Reuses the verbose-json word mapping from `app/transcription.py`.
  - `class Deepgram(Engine)` — `id="deepgram/nova-3"`, `needs_keys=["DEEPGRAM_API_KEY"]`, `diarizes=True`, `max_bytes=None`, `max_chunk_sec=None`. Returns `Word`s (with `.speaker` from Deepgram diarization) and `speakers` segments.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_groq_deepgram_engines.py
from unittest.mock import patch, MagicMock
from eval.engines.groq_whisper import GroqWhisper
from eval.engines.deepgram import Deepgram

def test_groq_maps_words():
    word = MagicMock(); word.word = "hi"; word.start = 0.0; word.end = 0.2
    resp = MagicMock(); resp.text = "hi"; resp.words = [word]
    client = MagicMock(); client.audio.transcriptions.create.return_value = resp
    with patch("eval.engines.groq_whisper._get_client", return_value=client):
        out = GroqWhisper("whisper-large-v3-turbo")._transcribe_chunk("/tmp/c.wav", 0.0)
    assert out.words[0].text == "hi"
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-large-v3-turbo"

def test_groq_needs_key():
    assert GroqWhisper("whisper-large-v3").needs_keys == ["GROQ_API_KEY"]

def test_deepgram_maps_words_with_speakers():
    w = MagicMock(); w.word = "hello"; w.start = 0.0; w.end = 0.4; w.speaker = 0
    alt = MagicMock(); alt.transcript = "hello"; alt.words = [w]
    chan = MagicMock(); chan.alternatives = [alt]
    resp = MagicMock(); resp.results.channels = [chan]
    client = MagicMock(); client.listen.v1.media.transcribe_file.return_value = resp
    with patch("eval.engines.deepgram._get_client", return_value=client):
        out = Deepgram()._transcribe_chunk("/tmp/c.wav", 0.0)
    assert out.words[0].text == "hello"
    assert out.words[0].speaker == "SPEAKER_00"

def test_deepgram_is_diarizer():
    assert Deepgram().diarizes is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_groq_deepgram_engines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.engines.groq_whisper'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/engines/groq_whisper.py
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
```

```python
# eval/engines/deepgram.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_groq_deepgram_engines.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/engines/groq_whisper.py eval/engines/deepgram.py tests/eval/test_groq_deepgram_engines.py
git commit -m "feat(eval): Groq Whisper + Deepgram engine adapters"
```

---

### Task 10: Apple Voice Memos transcript importer

**Files:**
- Create: `eval/engines/apple_voicememos.py`
- Create: `tests/eval/test_apple_engine.py`

**Interfaces:**
- Consumes: `TranscriptResult`, `Word`, `Segment` from `eval.models`.
- Produces:
  - `parse_apple_transcript(text: str) -> TranscriptResult` — parses an exported Apple Voice Memos transcript. Format: lines of `HH:MM:SS Speaker N` headers followed by text lines; converts to words with interpolated timestamps across each utterance and `SPEAKER_NN` labels. (Apple export has no per-word times; words are evenly spaced within the utterance's `[start, next_start)` span.)
  - `class AppleVoiceMemos` with `id="apple/voicememos"`, `diarizes=True`. It is **not** registered for inference (no audio path); instead the CLI loads it from a `--apple-transcript <path>` file when provided. Expose `load(path: str) -> TranscriptResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_apple_engine.py
from eval.engines.apple_voicememos import parse_apple_transcript

SAMPLE = """00:00:00 Speaker 1
Hello there everyone.
00:00:04 Speaker 2
General Kenobi.
"""

def test_parses_speakers_and_words():
    tr = parse_apple_transcript(SAMPLE)
    assert tr.engine_id == "apple/voicememos"
    assert tr.words[0].text == "Hello"
    assert tr.words[0].speaker == "SPEAKER_00"
    assert any(w.speaker == "SPEAKER_01" and w.text == "General" for w in tr.words)

def test_word_times_are_monotonic():
    tr = parse_apple_transcript(SAMPLE)
    times = [w.start for w in tr.words]
    assert times == sorted(times)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_apple_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.engines.apple_voicememos'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/engines/apple_voicememos.py
from __future__ import annotations
import re
from eval.models import TranscriptResult, Word, Segment

_HEADER = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s+Speaker\s+(\d+)", re.IGNORECASE)
ENGINE_ID = "apple/voicememos"


def _seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_apple_transcript(text: str) -> TranscriptResult:
    utterances: list[tuple[float, str, list[str]]] = []  # (start, speaker, tokens)
    cur_start, cur_spk, cur_tokens = None, None, []
    for line in text.splitlines():
        m = _HEADER.match(line.strip())
        if m:
            if cur_start is not None and cur_tokens:
                utterances.append((cur_start, cur_spk, cur_tokens))
            cur_start = _seconds(m.group(1), m.group(2), m.group(3))
            cur_spk = f"SPEAKER_{int(m.group(4)) - 1:02d}"
            cur_tokens = []
        elif line.strip():
            cur_tokens.extend(line.split())
    if cur_start is not None and cur_tokens:
        utterances.append((cur_start, cur_spk, cur_tokens))

    words: list[Word] = []
    segments: list[Segment] = []
    for i, (start, spk, tokens) in enumerate(utterances):
        end = utterances[i + 1][0] if i + 1 < len(utterances) else start + len(tokens) * 0.4
        step = (end - start) / max(len(tokens), 1)
        for j, tok in enumerate(tokens):
            ws = start + j * step
            words.append(Word(tok.strip(".,!?;:"), round(ws, 3), round(ws + step, 3), spk))
        segments.append(Segment(spk, start, end))
    full = " ".join(w.text for w in words)
    return TranscriptResult(ENGINE_ID, full, words, speakers=segments)


def load(path: str) -> TranscriptResult:
    with open(path) as f:
        return parse_apple_transcript(f.read())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_apple_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/engines/apple_voicememos.py tests/eval/test_apple_engine.py
git commit -m "feat(eval): Apple Voice Memos transcript importer"
```

---

### Task 11: Dormant premium engines (AssemblyAI, ElevenLabs, OpenAI, Gemini)

**Files:**
- Create: `eval/engines/assemblyai.py`
- Create: `eval/engines/elevenlabs.py`
- Create: `eval/engines/openai_whisper.py`
- Create: `eval/engines/gemini.py`
- Create: `tests/eval/test_premium_engines.py`

**Interfaces:**
- Consumes: `Engine` from `eval.engines.base`; `TranscriptResult`, `Word`, `Segment` from `eval.models`.
- Produces four registered engines, each auto-skipped when its key/dep is absent:
  - `AssemblyAI` — `id="assemblyai/universal"`, `needs_keys=["ASSEMBLYAI_API_KEY"]`, `diarizes=True`.
  - `ElevenLabsScribe` — `id="elevenlabs/scribe-v1"`, `needs_keys=["ELEVENLABS_API_KEY"]`, `diarizes=True`.
  - `OpenAITranscribe` — `id="openai/gpt-4o-transcribe"`, `needs_keys=["OPENAI_API_KEY"]`, `diarizes=False`, `max_bytes=24_000_000`, `max_chunk_sec=600.0`.
  - `Gemini` — `id="google/gemini-2.5-flash"`, `needs_keys=["GOOGLE_API_KEY"]`, `diarizes=True`.
- Each declares `_check_deps` import-probing its SDK so a missing package yields `available() == False` rather than an import crash at registry build.

- [ ] **Step 1: Write the failing test (skip-gating + word mapping with mocked SDKs)**

```python
# tests/eval/test_premium_engines.py
from unittest.mock import patch, MagicMock
from eval.engines.assemblyai import AssemblyAI
from eval.engines.elevenlabs import ElevenLabsScribe
from eval.engines.openai_whisper import OpenAITranscribe
from eval.engines.gemini import Gemini

def test_all_skip_when_keys_missing(monkeypatch):
    for var in ["ASSEMBLYAI_API_KEY", "ELEVENLABS_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    assert AssemblyAI().available() is False
    assert ElevenLabsScribe().available() is False
    assert OpenAITranscribe().available() is False
    assert Gemini().available() is False

def test_openai_maps_words():
    word = MagicMock(); word.word = "hi"; word.start = 0.0; word.end = 0.2
    resp = MagicMock(); resp.text = "hi"; resp.words = [word]
    client = MagicMock(); client.audio.transcriptions.create.return_value = resp
    with patch("eval.engines.openai_whisper._get_client", return_value=client):
        out = OpenAITranscribe()._transcribe_chunk("/tmp/c.wav", 0.0)
    assert out.words[0].text == "hi"

def test_assemblyai_declares_diarization():
    assert AssemblyAI().diarizes is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_premium_engines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.engines.assemblyai'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/engines/assemblyai.py
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
```

```python
# eval/engines/elevenlabs.py
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
```

```python
# eval/engines/openai_whisper.py
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
```

```python
# eval/engines/gemini.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_premium_engines.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/engines/assemblyai.py eval/engines/elevenlabs.py eval/engines/openai_whisper.py eval/engines/gemini.py tests/eval/test_premium_engines.py
git commit -m "feat(eval): dormant premium engine adapters (AssemblyAI/ElevenLabs/OpenAI/Gemini)"
```

---

### Task 12: Report generation (ranked table + diff + disagreement)

**Files:**
- Create: `eval/report.py`
- Create: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: `EngineScore`, `TranscriptResult` from `eval.models`; `align` from `eval.align`.
- Produces:
  - `ranked_table(scores: list[EngineScore]) -> str` — Markdown table sorted ascending by `cpwer` (None sorts last); columns: engine, cpWER, WER, CER, DER, RTF, $est.
  - `diff_html(reference: str, hypothesis: str) -> str` — HTML where `sub`/`ins`/`del` spans are class-tagged for coloring.
  - `disagreement(results: list[TranscriptResult], reference: str) -> list[dict]` — per reference token, the set of distinct hypotheses across engines; returns tokens where engines disagree (`{"ref": tok, "variants": {engine_id: tok_or_None}}`).
  - `write_report(out_dir: str, scores, results, reference_text) -> None` — writes `report.md`, `report.html`, `scores.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_report.py
import json
from eval.models import EngineScore, TranscriptResult, Word
from eval.report import ranked_table, diff_html, write_report

def test_ranked_table_sorts_by_cpwer():
    scores = [
        EngineScore("b", cpwer=0.30, wer=0.30),
        EngineScore("a", cpwer=0.10, wer=0.10),
        EngineScore("c", cpwer=None, wer=None),
    ]
    table = ranked_table(scores)
    a_idx, b_idx, c_idx = table.index("| a "), table.index("| b "), table.index("| c ")
    assert a_idx < b_idx < c_idx

def test_diff_html_marks_substitution():
    html = diff_html("the cat sat", "the dog sat")
    assert "sub" in html
    assert "dog" in html and "cat" in html

def test_write_report_emits_three_files(tmp_path):
    scores = [EngineScore("a", cpwer=0.1, wer=0.1)]
    results = [TranscriptResult("a", "hello world", [Word("hello", 0, 1), Word("world", 1, 2)])]
    write_report(str(tmp_path), scores, results, reference_text="hello world")
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    data = json.loads((tmp_path / "scores.json").read_text())
    assert data[0]["engine_id"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/report.py
from __future__ import annotations
import json
import os
from eval.models import EngineScore, TranscriptResult
from eval.align import align


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def ranked_table(scores: list[EngineScore]) -> str:
    ordered = sorted(scores, key=lambda s: (s.cpwer is None, s.cpwer if s.cpwer is not None else 0.0))
    lines = [
        "| engine | cpWER | WER | CER | DER | RTF | $est |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ordered:
        lines.append(
            f"| {s.engine_id} | {_fmt(s.cpwer)} | {_fmt(s.wer)} | {_fmt(s.cer)} | "
            f"{_fmt(s.der)} | {_fmt(s.rtf)} | {_fmt(s.cost_est)} |"
        )
    return "\n".join(lines)


def diff_html(reference: str, hypothesis: str) -> str:
    spans: list[str] = []
    for op in align(reference, hypothesis):
        if op["op"] == "equal":
            spans.append(f'<span class="equal">{op["hyp"]}</span>')
        elif op["op"] == "sub":
            spans.append(f'<span class="sub" title="ref: {op["ref"]}">{op["hyp"]}</span>')
        elif op["op"] == "ins":
            spans.append(f'<span class="ins">{op["hyp"]}</span>')
        elif op["op"] == "del":
            spans.append(f'<span class="del" title="ref: {op["ref"]}">∅</span>')
    return " ".join(spans)


_CSS = (
    ".equal{color:#222}.sub{background:#fde68a}.ins{background:#bbf7d0}"
    ".del{background:#fecaca;color:#991b1b}body{font-family:system-ui;max-width:900px;margin:2rem auto}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
)


def disagreement(results: list[TranscriptResult], reference: str) -> list[dict]:
    ref_tokens = reference.split()
    rows: list[dict] = []
    per_engine = {r.engine_id: r.text.split() for r in results}
    for i, tok in enumerate(ref_tokens):
        variants = {eid: (toks[i] if i < len(toks) else None) for eid, toks in per_engine.items()}
        if len(set(v for v in variants.values() if v is not None)) > 1:
            rows.append({"ref": tok, "variants": variants})
    return rows


def write_report(out_dir: str, scores: list[EngineScore], results: list[TranscriptResult], reference_text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    table = ranked_table(scores)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("# Evaluation report\n\n" + table + "\n")
    diffs = "".join(
        f"<h3>{r.engine_id}</h3><p>{diff_html(reference_text, r.text)}</p>" for r in results
    )
    html_table = table.replace("|", " ")  # minimal; md table not rendered, keep readable
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(f"<!doctype html><meta charset=utf-8><style>{_CSS}</style>"
                f"<h1>Evaluation report</h1><pre>{html_table}</pre>{diffs}")
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump([s.to_dict() for s in scores], f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_report.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/report.py tests/eval/test_report.py
git commit -m "feat(eval): report generation — ranked table, diff HTML, disagreement"
```

---

### Task 13: CLI orchestration

**Files:**
- Create: `eval/cli.py`
- Create: `eval/__main__.py`
- Create: `tests/eval/test_cli.py`

**Interfaces:**
- Consumes: everything above. Imports the engine modules so their `register(...)` runs (`import eval.engines.parakeet_mlx`, etc.) inside a `_load_engines()` function that swallows `ImportError` per module.
- Produces:
  - `attribute_speakers(result: TranscriptResult, segments: list[Segment]) -> TranscriptResult` — for non-diarizing engines, assigns each word a speaker via `app.merger`'s nearest-segment logic and attaches `speakers`.
  - `score_engine(result: TranscriptResult, reference: Reference) -> EngineScore` — computes WER/CER over corrected-window reference text vs. the result's words in those windows, cpWER via `words_by_speaker`, DER when the result has `speakers`.
  - `run(audio_path, engine_ids, work_dir, apple_transcript=None) -> tuple[list[EngineScore], list[TranscriptResult]]` — orchestrates: normalize → diarize once (`app.diarization.diarize`) → run each selected engine → attribute speakers to non-diarizers → build reference (if none on disk) → score → return.
  - `main(argv: list[str]) -> int` — argparse with subcommands `run` (`--engines`, `--work-dir`, `--apple-transcript`) and `reference` (`build|edit|status`). Default `--engines` = all `available()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_cli.py
from unittest.mock import patch
from eval.models import TranscriptResult, Word, Segment, Reference, Window
from eval.cli import attribute_speakers, score_engine

def test_attribute_speakers_assigns_from_segments():
    result = TranscriptResult("e", "hi yo", [Word("hi", 0.2, 0.5), Word("yo", 6.0, 6.3)])
    segs = [Segment("SPEAKER_00", 0.0, 5.0), Segment("SPEAKER_01", 5.2, 10.0)]
    out = attribute_speakers(result, segs)
    assert out.words[0].speaker == "SPEAKER_00"
    assert out.words[1].speaker == "SPEAKER_01"
    assert out.speakers is not None

def test_score_engine_perfect_match_is_zero_wer():
    ref = Reference("aud", [Window(0.0, 10.0, [Word("hello", 1.0, 1.4, "SPEAKER_00"),
                                               Word("world", 1.5, 1.9, "SPEAKER_00")], corrected=True)])
    result = TranscriptResult("e", "hello world",
                              [Word("hello", 1.0, 1.4, "SPEAKER_00"), Word("world", 1.5, 1.9, "SPEAKER_00")])
    score = score_engine(result, ref)
    assert score.wer == 0.0
    assert score.cpwer == 0.0

def test_run_invokes_engines(monkeypatch, tmp_path):
    from eval.engines.base import Engine, register
    class Stub(Engine):
        id = "stub/x"
        def _transcribe_chunk(self, wav_path, offset):
            return TranscriptResult(self.id, "hello world", [Word("hello", 0, 1), Word("world", 1, 2)])
    register(Stub())
    with patch("eval.cli.normalize", return_value=type("NA", (), {"wav_path": str(tmp_path / "a.wav"), "duration": 5.0})()), \
         patch("eval.audio.chunk_audio", return_value=[type("C", (), {"path": "x", "start": 0.0, "end": 5.0})()]), \
         patch("eval.cli.diarize", return_value=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]):
        scores, results = run_with_stub(tmp_path)
    assert any(r.engine_id == "stub/x" for r in results)

def run_with_stub(tmp_path):
    from eval.cli import run
    return run(str(tmp_path / "in.wav"), ["stub/x"], str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/cli.py
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import replace
from eval.models import TranscriptResult, Segment, EngineScore, Reference
from eval.audio import normalize, detect_silences
from eval.engines.base import get_registry, available_engines
from eval.metrics import wer, cer, cpwer, der, words_by_speaker
from eval.reference import sample_windows, build_reference, save_reference, load_reference
from eval.report import write_report


def _load_engines() -> None:
    for mod in [
        "eval.engines.parakeet_mlx", "eval.engines.groq_whisper", "eval.engines.deepgram",
        "eval.engines.assemblyai", "eval.engines.elevenlabs",
        "eval.engines.openai_whisper", "eval.engines.gemini",
    ]:
        try:
            __import__(mod)
        except ImportError:
            pass


def attribute_speakers(result: TranscriptResult, segments: list[Segment]) -> TranscriptResult:
    from app.merger import _find_speaker
    seg_dicts = [{"speaker": s.speaker, "start": s.start, "end": s.end} for s in segments]
    words = [replace(w, speaker=_find_speaker((w.start + w.end) / 2, seg_dicts)) for w in result.words]
    spk_segments: list[Segment] = []
    for w in words:
        if spk_segments and spk_segments[-1].speaker == w.speaker:
            spk_segments[-1].end = w.end
        else:
            spk_segments.append(Segment(w.speaker or "UNKNOWN", w.start, w.end))
    return replace(result, words=words, speakers=spk_segments)


def _corrected_windows(reference: Reference):
    return [w for w in reference.windows if w.corrected] or reference.windows


def score_engine(result: TranscriptResult, reference: Reference) -> EngineScore:
    windows = _corrected_windows(reference)
    ref_text, hyp_text = [], []
    ref_words, hyp_words = [], []
    for win in windows:
        ref_text.append(" ".join(w.text for w in win.words))
        ref_words.extend(win.words)
        in_win = [w for w in result.words if win.start <= (w.start + w.end) / 2 <= win.end]
        hyp_text.append(" ".join(w.text for w in in_win))
        hyp_words.extend(in_win)
    ref_joined, hyp_joined = " ".join(ref_text), " ".join(hyp_text)
    der_val = None
    if result.speakers is not None:
        ref_segs = _segments_from_words(ref_words)
        hyp_segs = _segments_from_words(hyp_words)
        if ref_segs and hyp_segs:
            der_val = der(ref_segs, hyp_segs)
    return EngineScore(
        engine_id=result.engine_id,
        wer=wer(ref_joined, hyp_joined),
        cer=cer(ref_joined, hyp_joined),
        cpwer=cpwer(words_by_speaker(ref_words), words_by_speaker(hyp_words)),
        der=der_val,
        rtf=result.meta.get("rtf"),
        cost_est=result.meta.get("cost_est"),
    )


def _segments_from_words(words) -> list[Segment]:
    segs: list[Segment] = []
    for w in sorted(words, key=lambda x: x.start):
        spk = w.speaker or "UNKNOWN"
        if segs and segs[-1].speaker == spk:
            segs[-1].end = w.end
        else:
            segs.append(Segment(spk, w.start, w.end))
    return segs


def run(audio_path: str, engine_ids: list[str], work_dir: str, apple_transcript: str | None = None):
    from app.diarization import diarize as _diarize_app  # noqa
    os.makedirs(work_dir, exist_ok=True)
    audio = normalize(audio_path, work_dir)
    diar = diarize(audio.wav_path)
    segments = [Segment(d["speaker"], d["start"], d["end"]) for d in diar]

    registry = get_registry()
    results: list[TranscriptResult] = []
    for eid in engine_ids:
        engine = registry[eid]
        res = engine.transcribe(audio, work_dir)
        if not engine.diarizes:
            res = attribute_speakers(res, segments)
        results.append(res)

    if apple_transcript:
        from eval.engines.apple_voicememos import load as load_apple
        results.append(load_apple(apple_transcript))

    ref_path = os.path.join(work_dir, "reference.json")
    if os.path.exists(ref_path):
        reference = load_reference(ref_path)
    else:
        windows = sample_windows(audio.duration, detect_silences(audio.wav_path))
        reference = build_reference(os.path.basename(audio_path), results, windows)
        save_reference(reference, ref_path)

    scores = [score_engine(r, reference) for r in results]
    write_report(work_dir, scores, results, _reference_text(reference))
    return scores, results


def diarize(wav_path: str):
    from app.diarization import diarize as _d
    return _d(wav_path)


def _reference_text(reference: Reference) -> str:
    return " ".join(" ".join(w.text for w in win.words) for win in reference.windows)


def main(argv: list[str] | None = None) -> int:
    _load_engines()
    parser = argparse.ArgumentParser(prog="eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("audio")
    p_run.add_argument("--engines", default="")
    p_run.add_argument("--work-dir", default="runs/latest")
    p_run.add_argument("--apple-transcript", default=None)

    p_ref = sub.add_parser("reference")
    p_ref.add_argument("action", choices=["build", "edit", "status"])
    p_ref.add_argument("--work-dir", default="runs/latest")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        ids = [e.strip() for e in args.engines.split(",") if e.strip()] or [e.id for e in available_engines()]
        scores, _ = run(args.audio, ids, args.work_dir, args.apple_transcript)
        print(f"Scored {len(scores)} engines → {args.work_dir}/report.md")
        return 0
    if args.cmd == "reference":
        ref_path = os.path.join(args.work_dir, "reference.json")
        ref = load_reference(ref_path)
        corrected = sum(1 for w in ref.windows if w.corrected)
        print(f"{corrected}/{len(ref.windows)} windows corrected in {ref_path}")
        return 0
    return 1
```

```python
# eval/__main__.py
import sys
from eval.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full eval suite**

Run: `python -m pytest tests/eval/ -v`
Expected: PASS (all eval tests green)

- [ ] **Step 6: Commit**

```bash
git add eval/cli.py eval/__main__.py tests/eval/test_cli.py
git commit -m "feat(eval): CLI orchestration — run + reference subcommands"
```

---

### Task 14: End-to-end smoke test + README

**Files:**
- Create: `tests/eval/test_e2e_smoke.py`
- Create: `eval/README.md`

**Interfaces:**
- Consumes: the whole pipeline with a synthetic stub engine (no network, no real audio).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_e2e_smoke.py
from unittest.mock import patch
from eval.models import TranscriptResult, Word
from eval.engines.base import Engine, register

class EchoEngine(Engine):
    id = "echo/test"
    def _transcribe_chunk(self, wav_path, offset):
        return TranscriptResult(self.id, "the quick brown fox",
                                [Word("the", 0, 1), Word("quick", 1, 2), Word("brown", 2, 3), Word("fox", 3, 4)])

def test_full_run_writes_report(tmp_path):
    register(EchoEngine())
    fake_audio = type("NA", (), {"wav_path": str(tmp_path / "n.wav"), "duration": 5.0})()
    fake_chunk = type("C", (), {"path": "x", "start": 0.0, "end": 5.0})()
    from eval import cli
    with patch.object(cli, "normalize", return_value=fake_audio), \
         patch("eval.engines.base.chunk_audio", return_value=[fake_chunk]), \
         patch.object(cli, "diarize", return_value=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]), \
         patch.object(cli, "detect_silences", return_value=[]):
        scores, results = cli.run(str(tmp_path / "in.wav"), ["echo/test"], str(tmp_path))
    assert (tmp_path / "report.md").exists()
    assert scores[0].engine_id == "echo/test"
    assert scores[0].wer == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_e2e_smoke.py -v`
Expected: FAIL (`report.md` missing or import error) until the full chain is wired — confirm it then passes after Step 3 wiring review.

- [ ] **Step 3: Make it pass**

No new product code expected if Tasks 1–13 are correct; if the test exposes a wiring gap (e.g. `detect_silences` not patchable because imported by-value), adjust the `eval/cli.py` import so `detect_silences` is referenced as `eval.cli.detect_silences`. Re-run.

- [ ] **Step 4: Write the README**

```markdown
# eval — transcription evaluation harness

Run multiple ASR/diarization engines on an audio file, build a reference,
and score each engine by cpWER/WER/CER/DER.

## Quickstart

    python -m pip install -e ".[dev,engines]"
    python -m eval run path/to/audio.m4a --work-dir runs/roncesvalles

Engines with missing API keys are skipped automatically. Available without keys:
Parakeet (local MLX). With keys in `.env`: Groq Whisper, Deepgram. Optional:
AssemblyAI, ElevenLabs, OpenAI, Gemini.

## Reference correction

The first run writes `runs/<name>/reference.json` with 5 sampled ~3-min windows,
bootstrapped by ROVER consensus. Edit the `words` and set `"corrected": true` per
window, then re-run to get absolute scores. Provide Apple's exported transcript via
`--apple-transcript file.txt` to seed it.

## Output

`runs/<name>/report.md`, `report.html`, `scores.json`.
```

- [ ] **Step 5: Run the whole test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS for all `tests/eval/` (pre-existing `app` tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add tests/eval/test_e2e_smoke.py eval/README.md
git commit -m "test(eval): end-to-end smoke test + README"
```

---

## Self-Review

**Spec coverage:**
- Architecture/layout → Tasks 1–13 create every module in the spec tree (`audio`, `engines/*`, `reference`, `normalize`, `metrics`, `align`, `report`, `cli`, `__main__`). ✓
- Engine adapter interface + registry skip-on-missing-key → Task 3. ✓
- Long-audio chunking (silence-aware, global offsets, overlap de-dup) → Tasks 2 + 3. ✓
- Reference: window sampling + ROVER + Apple seed + corrected flag → Task 7 + `--apple-transcript` in Task 13. ✓
- Normalization before scoring → Task 4, used by metrics/align. ✓
- Metrics cpWER/WER/CER/DER + speaker-count → Task 5 (speaker_count_err field present in model; populated as a follow-up if needed — see gap note). 
- Reporting: ranked table + colorized diff + disagreement view → Task 12. ✓
- ASR-only engines paired with pyannote → `attribute_speakers` in Task 13. ✓
- Engines: Parakeet, Groq×2, Deepgram, Apple, AssemblyAI, ElevenLabs, OpenAI, Gemini → Tasks 8–11. ✓
- Testing (unit + golden e2e, mocked APIs) → every task + Task 14. ✓

**Gap note (intentional):** `EngineScore.speaker_count_err` is defined but not populated; computing it is a one-line addition in `score_engine` once real diarized runs exist. Left unpopulated to avoid speculative logic (YAGNI) — flagged here so it isn't mistaken for an oversight.

**Placeholder scan:** No TBD/TODO; every code step contains runnable code. ✓

**Type consistency:** `TranscriptResult`, `Word`, `Segment`, `EngineScore`, `Reference`, `Window` signatures are used identically across tasks; `_segments_from_words`/`_segments` helpers are defined where used; `chunk_audio` patched at `eval.engines.base.chunk_audio` (where it's imported) in tests. ✓
