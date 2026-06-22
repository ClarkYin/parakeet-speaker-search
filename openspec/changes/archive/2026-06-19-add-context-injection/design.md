## Context

The current pipeline (`app/ingest.py: run_pipeline`) extracts audio, then runs transcription and diarization in parallel threads, merges words to speakers, and marks the file `ready`. Transcription sends no domain hints, so Whisper guesses unfamiliar names/jargon/acronyms wrong. Whisper accepts a free-text `prompt` (a.k.a. `initial_prompt`) that biases decoding toward expected vocabulary; Groq's API exposes this. We have working Groq (Whisper + Llama chat) and Deepgram providers, a Postgres `files`/`utterances` schema, and a FastAPI upload→status→transcript flow. The constraint from brainstorming: the user wants to **see and edit** the inferred context before the full pass runs, not have it applied blindly.

## Goals / Non-Goals

**Goals:**
- Improve word accuracy by steering the full transcription with an approved context string.
- Infer that context automatically from a cheap sample, then gate on human approval.
- Reuse existing infrastructure (Groq key, Postgres, FastAPI background tasks) without adding new secrets.

**Non-Goals:**
- Solving Groq's 25 MB upload limit for very long files (pre-existing, unchanged here).
- Per-speaker or per-segment context, glossaries, or persistent vocabulary across files.
- A frontend UI for approval — the approval surface is the two HTTP endpoints; any client (curl, Swagger UI) can drive it.
- Deepgram keyterm prompting (left as a future enhancement; Deepgram ignores context in v1).

## Decisions

**1. Two background stages with an approval gate, not one config flag.**
Upload schedules Stage 1 only. Stage 1 transcribes the sample, infers context, sets `awaiting_approval`, and stops. The approve endpoint schedules Stage 2 (full transcription + diarization + merge → `ready`). *Alternative considered:* a single pipeline that auto-applies inferred context with no gate — rejected because the user explicitly wants to review/correct the guess, which is also where most of the accuracy win comes from.

**2. Infer context with Groq Llama, not a new Anthropic dependency.**
Stage 1 calls Groq chat completions (`llama-3.3-70b-versatile`) with the rough transcript and a fixed system prompt asking for a short keyword-style hint — the topic plus any likely names/jargon/acronyms, NOT a narrative description. (Testing showed narrative, sentence-shaped prompts get echoed verbatim into the transcript and cause large content loss; a short keyword phrase steers without leaking.) *Why:* we already have a working Groq key; this adds no new secret, is fast, and a short summary is well within a small open model's ability. *Alternative considered:* Anthropic Claude — higher quality but introduces a new API key, dependency, and cost. The inference provider/model is isolated in one module (`app/context_inference.py`) so swapping to Claude later is a one-file change.

**3. Stage 1 always uses the cheapest/fastest transcription, on a 60 s slice.**
Regardless of the model chosen for the full pass, the rough pass uses `groq/whisper-large-v3-turbo` on the first 60 s (via an ffmpeg `-t 60` trim). The rough text only needs to reveal topic and vocabulary, not be accurate, so paying for the full model here is wasted. Files shorter than 60 s use the whole clip.

**4. Audio is extracted once and reused.**
`extract_audio` already produces a deterministic WAV path from the input path. Stage 1 extracts the full WAV (idempotent) and derives a 60 s slice from it for the rough pass. Stage 2 reuses the full WAV by the same deterministic path, re-extracting only if it is missing. This avoids decoding long files twice.

**5. Context and the chosen model are persisted on the `files` row.**
A nullable `files.context TEXT` column holds the inferred/approved context: Stage 1 writes the inferred value; approval overwrites it with the user's value (or leaves it). Because approval is a separate request from upload, the model picked at upload is no longer available in memory, so a `files.model TEXT` column stores it at upload for Stage 2 to read back. Both columns are nullable, so existing rows and the single-stage path remain valid.

**6. The approve transition is guarded against double-start.**
Approval performs a conditional update — `UPDATE files SET status='processing', context=:ctx WHERE id=:id AND status='awaiting_approval'`. If zero rows change, the file was not awaiting approval and the endpoint returns `409` without scheduling Stage 2. This makes repeated/concurrent approve calls safe.

**7. `transcribe()` gains an optional `context` argument.**
Signature becomes `transcribe(audio_path, model=..., context: str | None = None)`. Groq passes a non-empty `context` as the `prompt` parameter; Deepgram ignores it in v1 (never forwards an unsupported field). Empty/None context preserves today's behavior exactly.

## Risks / Trade-offs

- **Inferred context is wrong or generic** → The approval gate exists precisely for this; the user edits or clears it. Empty context falls back to current (unsteered) behavior, so the feature can never make accuracy worse than today.
- **Bad/biasing prompt degrades transcription** (Whisper echoes prompt words and can drop content) → Keep the inferred context short and keyword-style (topic + names/jargon), never narrative sentences; observed in testing that a paragraph-style hint was repeated into the output and halved the transcript. The user can also trim it at the approval gate.
- **Extra latency: upload no longer yields a transcript in one step** → Acceptable and intended; Stage 1 is fast (60 s slice + one small LLM call) and the gate is the point. Clients poll `GET /files/{id}/context` for `awaiting_approval`.
- **Groq Llama inference call fails** → Non-fatal: status still becomes `awaiting_approval` with empty context so the user can type their own and proceed.
- **Long files still exceed Groq's 25 MB limit on the full pass** → Unchanged pre-existing limitation; out of scope.

## Migration Plan

- Add `context TEXT` and `model TEXT` to `scripts/init_db.sql` for fresh databases.
- For the existing database, run `ALTER TABLE files ADD COLUMN IF NOT EXISTS context TEXT;` and `ALTER TABLE files ADD COLUMN IF NOT EXISTS model TEXT;` (idempotent; existing rows get NULL).
- No data backfill needed. Rollback is dropping the column and reverting code; in-flight files in `awaiting_approval` would need a manual re-upload, which is acceptable for this single-user system.

## Open Questions

- Inference model default is `llama-3.3-70b-versatile`; confirm during spec review or switch to Anthropic Claude if higher-quality summaries are wanted.
- Should the sample window (60 s) be configurable? Defaulting to a constant for v1; promote to a setting only if needed.
