## Why

Transcription randomly misrecognizes domain-specific words (names, jargon, acronyms) because Whisper's decoder has no idea what the audio is about. Whisper accepts an `initial_prompt` that biases the decoder toward expected vocabulary, but we currently send nothing. We can close most of the accuracy gap by automatically inferring what the recording is about and feeding that context back into transcription — with a human approving the guess before the expensive full pass runs.

## What Changes

- Split ingestion into a **two-stage pipeline**:
  - **Stage 1 (context inference)** — on upload, quickly transcribe the first ~60 seconds, ask an LLM to infer a 1–2 sentence context description, store it, and pause at a new `awaiting_approval` status.
  - **Stage 2 (full transcription)** — after the user approves (optionally editing the context), run the full transcription with the approved context as Whisper's `initial_prompt`, then diarize and merge as today.
- Add two endpoints:
  - `GET /files/{id}/context` — returns the inferred context and current status.
  - `POST /files/{id}/context/approve` — accepts an optional context override and starts Stage 2.
- Add a `context TEXT` column to the `files` table to persist the inferred/approved context, and a `model TEXT` column to remember the model selected at upload so Stage 2 can reuse it.
- New status flow: `processing → awaiting_approval → processing → ready` (plus `failed`).
- Stage 1 always uses the fastest model (`groq/whisper-large-v3-turbo`) for the rough pass regardless of the model selected for the full transcription.

## Capabilities

### New Capabilities
- `context-aware-transcription`: Inferring a recording's context from a short sample, surfacing it for human approval, and steering the full transcription with the approved context to improve word accuracy.

### Modified Capabilities
<!-- None — openspec/specs/ is empty; all behavior here is new. -->

## Impact

- **Code**: `app/ingest.py` (split into stage 1 / stage 2 functions), `app/transcription.py` (thread an optional `prompt`/context through to Groq Whisper), `app/routes/files.py` (new endpoints + revised upload flow), `app/storage.py` (persist/read `context` and `model`), new `app/context_inference.py` (LLM call), `scripts/init_db.sql` (add `context` and `model` columns).
- **APIs**: New `GET`/`POST` context endpoints; upload response/status semantics gain `awaiting_approval`.
- **Dependencies**: An LLM chat-completion call for inference. Reusing the existing Groq key (e.g. a Llama model) avoids adding a new provider/secret; the provider choice is decided in `design.md`.
- **Providers**: Context injection maps directly onto Groq Whisper's `prompt` parameter. Deepgram has no equivalent free-text prompt, so for Deepgram the context is mapped to keyterms or skipped — covered in `design.md`.
