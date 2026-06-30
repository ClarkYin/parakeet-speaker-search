# Transcription Evaluation Harness — Design

**Date:** 2026-06-23
**Project:** parakeet-speaker-search
**Status:** Approved design, ready for implementation planning

## Problem

`parakeet-speaker-search` transcribes audio, diarizes speakers, merges the two,
embeds utterances, and makes them searchable. Today there is **no way to tell how
good the transcripts are**. Different engines (Groq Whisper `large-v3` /
`large-v3-turbo`, Deepgram `nova-3`) produce visibly different output on the same
audio, and nothing measures which is closer to the truth or where they diverge.

Two concrete gaps surfaced while investigating a real 2.5-hour recording
("Roncesvalles Ave", 9,287 s, stereo AAC + a 4-channel spatial track + a data
track):

1. **No long-audio handling.** A 2.5 h mono 16 kHz WAV is ~280 MB — far over
   Groq's ~25–40 MB request cap and Whisper's 30 s window. The pipeline has no
   chunker, so long recordings fail or drift, and each engine's internal overflow
   handling diverges. This is a large part of why "all the transcripts are
   different."
2. **No evaluation.** There are no accuracy metrics (WER/CER/DER), so quality can
   only be eyeballed.

Note: despite the project name, **NVIDIA Parakeet is not wired in** — it is
currently #1 on the HuggingFace Open-ASR leaderboard and runs locally and free on
this machine's Apple Silicon via MLX.

## Goal

Build a **reusable transcription-evaluation harness**: point it at any audio file
and a set of engines; it runs them all, builds a trustworthy reference ("gold
standard"), scores every engine against it, and emits a ranked comparison report
plus a per-engine diff. The "Roncesvalles Ave" recording is the first test case.

The harness is the deliverable. Fixing the production pipeline (wiring in Parakeet,
adding chunking, improving the merge) is **out of scope** here — but the chunker
and adapters the harness produces are exactly the pieces the pipeline will adopt
later, so this doubles as the road to "improve it."

### Decisions locked in

- **Reference strategy: bootstrap + spot-correct ("Both").** Build a candidate
  reference from a consensus of the strongest engines, seeded by Apple's own
  transcript, then have a human spot-correct it. Absolute scores come from the
  corrected portions; relative cross-engine agreement is computed everywhere.
- **Primary metric: cpWER** (concatenated, speaker-attributed WER) — it penalizes
  both wrong words and words attributed to the wrong speaker, the honest single
  number for a speaker-search product. Plain WER, CER, and DER are reported
  alongside.
- **Engine set: pluggable, run-what's-configured.** Wire adapters for Parakeet
  (MLX, local), Groq Whisper, Deepgram, Apple Voice Memos (import), AssemblyAI,
  ElevenLabs Scribe, OpenAI, Google Gemini/Chirp. The registry **auto-skips any
  engine whose API keys or deps are missing** (currently available: Groq,
  Deepgram, HF).
- **Reference windows: 5 × ~3 min, stratified** across the recording, snapped to
  silence boundaries. Full hand-correction of 2.5 h is impractical; sampled
  windows give rigorous absolute scores at a realistic correction cost.
- **ASR-only engines run through pyannote** (the existing `app/diarization.py` +
  merge) so every engine yields a *speaker-attributed* hypothesis. cpWER therefore
  compares full **ASR + diarization pipelines**, and reveals which "ASR + pyannote"
  combination wins.

## Architecture

A standalone `eval/` package in the repo, decoupled from the FastAPI app and
Postgres. It may import `app/` adapters (`app/transcription.py`,
`app/diarization.py`, `app/merger.py`) where they already exist.

```
eval/
  __main__.py              # `python -m eval ...`
  cli.py                   # arg parsing + orchestration
  audio.py                 # normalize → 16 kHz mono WAV; VAD/silence chunker
  engines/
    base.py                # Engine ABC + TranscriptResult + registry
    parakeet_mlx.py
    groq_whisper.py
    deepgram.py
    apple_voicememos.py    # imports an exported Apple transcript (no inference)
    assemblyai.py
    elevenlabs.py
    openai.py
    gemini.py
  reference.py             # window sampling + ROVER consensus + load/save/edit
  normalize.py             # text normalizer (Whisper EnglishTextNormalizer)
  metrics.py               # WER/CER (jiwer), cpWER (meeteval), DER (pyannote.metrics)
  align.py                 # word alignment for diffs
  report.py                # ranked table + colorized diff + disagreement view
runs/                      # gitignored: per-audio references, scores, reports
```

### Data model (dataclasses, JSON on disk)

- `Word { text, start, end, speaker? }`
- `Segment { speaker, start, end }`
- `TranscriptResult { engine_id, text, words[Word], speakers?[Segment],
   meta{ rtf, cost_est, model, chunked } }`
- `Window { start, end, words[Word w/ speaker], corrected: bool }`
- `Reference { audio_id, windows[Window] }`
- `EngineScore { engine_id, cpwer, wer, cer, der?, speaker_count_err,
   rtf, cost_est }`

## Components

### 1. Engine adapter interface (`engines/base.py`)

```python
class Engine(ABC):
    id: str
    needs_keys: list[str]       # env vars required; missing → skipped
    diarizes: bool              # produces native speaker labels
    max_chunk_sec: float | None # None = local, no chunking needed
    max_bytes: int | None

    def available(self) -> bool      # keys present AND deps importable
    @abstractmethod
    def _transcribe_chunk(self, wav_path: str, offset: float) -> TranscriptResult
    def transcribe(self, audio: NormalizedAudio) -> TranscriptResult
        # default: split via audio.chunks honoring limits, restitch on global timeline
```

A module-level registry collects engines; `--engines` selects by id, default = all
`available()`. Native diarizers use their own speaker labels. ASR-only engines are
paired with `app/diarization.py` + `app/merger.py` to produce speaker-attributed
output. The Apple adapter imports an exported transcript rather than running
inference; it serves as both a ranked entry and a reference seed.

### 2. Audio normalization + chunking (`audio.py`)

- `normalize(input) -> NormalizedAudio`: ffmpeg → 16 kHz mono PCM WAV, downmixing
  the stereo AAC stream. The 4-channel spatial and data tracks are ignored for ASR
  (spatial audio is a noted future lever for diarization).
- `chunks(max_sec, max_bytes) -> list[Chunk{path, start, end}]`: silence/VAD-aware
  splitting (silero-vad, or ffmpeg `silencedetect` as a fallback) so cuts land in
  pauses and never split a word. Each chunk respects per-engine caps with a small
  overlap; every word carries an absolute global timestamp on restitch, and the
  overlap region is de-duplicated.

This chunker is the reusable artifact the production pipeline currently lacks.

### 3. Reference / gold-standard builder (`reference.py`)

- **Window sampling:** K windows (default 5 × ~180 s) stratified across the file
  (early / quarter / mid / three-quarter / late), snapped to silence boundaries.
- **Bootstrap:** per window, ROVER-style word-level consensus across the strongest
  available engines (weighted by known accuracy), seeded by the Apple transcript if
  present → candidate transcript + speaker attribution from the best diarizer.
- **Human spot-correction:** candidate written to `reference.json` plus a readable
  per-window `.txt`; the user edits it; a `corrected` flag gates absolute metrics.
  CLI: `eval reference build | edit | status`.
- Absolute cpWER/WER/DER are computed only on corrected windows; uncorrected
  windows are reported as provisional. Whole-file relative agreement is computed
  regardless.

### 4. Normalization (`normalize.py`)

Apply a standard text normalizer (Whisper `EnglishTextNormalizer`) identically to
reference and hypothesis before scoring: lowercase, expand contractions, normalize
numbers, strip punctuation and fillers, unify spelling. This separates genuine
divergence from casing/punctuation noise — the bulk of why raw transcripts "look
different."

### 5. Metrics (`metrics.py`)

- **cpWER** (primary) via `meeteval` — concatenated minimum-permutation,
  speaker-attributed WER.
- **WER + CER** via `jiwer`.
- **DER** via `pyannote.metrics` for diarizing engines and ASR+pyannote pipelines.
- Plus speed (RTF), estimated cost per engine, and speaker-count error.

### 6. Alignment + reporting (`align.py`, `report.py`)

- Word-level alignment of each hypothesis to the reference (substitutions /
  insertions / deletions; speaker confusions marked).
- **Ranked table** (sorted by cpWER): engine | cpWER | WER | CER | DER | speakers |
  RTF | $est.
- **Per-engine colorized diff** vs. reference (HTML + terminal).
- **Cross-engine disagreement view** highlighting where engines diverge most —
  surfacing hard spots such as proper nouns ("Roncesvalles"), overlaps, and
  cross-talk.
- Output to `runs/<audio>/report.html`, `report.md`, and `scores.json`.

## Error handling

- Missing keys/deps → engine skipped with a logged note, not a failure.
- An engine erroring on a chunk → that engine marked failed for the run; others
  continue; the report shows partial results.
- ffmpeg / normalization failure → hard error (nothing downstream can run).
- Uncorrected reference windows → absolute metrics flagged provisional, not blocked.

## Testing

- Unit tests with tiny fixtures: chunker boundaries/offsets/restitch, normalizer,
  metrics on known WER pairs, reference load/save, registry skip-on-missing-key.
- One golden mini-audio (a few sentences, known transcript) end-to-end through the
  local Parakeet engine — validates the whole flow with no network.
- API engine adapters mocked; no live calls in tests.
- Follows the existing `tests/` pytest layout.

## Dependencies (added, lazy-imported per engine)

`jiwer`, `meeteval`, `mlx` + `parakeet-mlx`, a Whisper text normalizer,
`silero-vad`; optional `assemblyai`, `elevenlabs`, `openai`, `google-genai`.

## Prerequisite (not a design blocker)

The `.qta` shared into VoiceMemos' temp was an ephemeral item-provider copy and has
already been garbage-collected; the VoiceMemos library itself is TCC-protected. To
run the actual benchmark, re-export "Roncesvalles Ave" from Voice Memos to
`/Users/clarkyin/Development/parakeet-speaker-search/audio/` (and ideally export
Apple's transcript for the reference seed). The harness is built and tested
independently of this.

## Out of scope

- Modifying the production ingest/search pipeline.
- A web UI for evaluation (this is CLI + file output).
- Full hand-transcription of the entire 2.5 h recording.
