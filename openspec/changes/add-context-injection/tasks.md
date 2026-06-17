## 1. Data model & configuration

- [x] 1.1 Add `context TEXT` and `model TEXT` to the `files` table definition in `scripts/init_db.sql`
- [x] 1.2 Apply `ALTER TABLE files ADD COLUMN IF NOT EXISTS context TEXT;` and `ALTER TABLE files ADD COLUMN IF NOT EXISTS model TEXT;` to the running database
- [x] 1.3 Add `inference_model: str = "llama-3.3-70b-versatile"` setting to `app/config.py` (reuses the existing `groq_api_key`)

## 2. Context inference module

- [x] 2.1 Create `app/context_inference.py` with `infer_context(rough_text: str) -> str` that calls Groq chat completions (`settings.inference_model`) with a fixed system prompt asking for a 1–2 sentence description of the recording's topic/setting/likely proper nouns
- [x] 2.2 Make `infer_context` return `""` (not raise) when the Groq call fails, so Stage 1 stays non-fatal
- [x] 2.3 Add `tests/test_context_inference.py` mocking the Groq client to assert the rough text is sent and the trimmed summary is returned, and that an exception yields `""`

## 3. Thread context through transcription

- [x] 3.1 Add optional `context: str | None = None` to `transcribe()` and `_transcribe_groq()` in `app/transcription.py`; when non-empty, pass it as the Groq `prompt` parameter
- [x] 3.2 Leave `_transcribe_deepgram()` ignoring `context` (never forward an unsupported field); accept the argument for signature parity
- [x] 3.3 Add a test asserting a non-empty context is forwarded as `prompt` to the Groq client, and that empty/None sends no `prompt`

## 4. Storage helpers

- [x] 4.1 In `app/storage.py`, add helpers to write the context (`set_file_context`) and read context + status (`get_file_context`)
- [x] 4.2 Add a guarded approve helper: `UPDATE files SET status='processing', context=:ctx WHERE id=:id AND status='awaiting_approval'` returning rows-affected so the caller can detect the no-op case
- [x] 4.3 Add a test for the guarded update returning 0 rows when status is not `awaiting_approval`

## 5. Two-stage ingest pipeline

- [x] 5.1 In `app/ingest.py`, add `extract_audio_slice(audio_path, seconds=60)` using ffmpeg `-t` to produce a short WAV for the rough pass
- [x] 5.2 Add `run_stage1(db, file_id, file_path)`: extract full WAV, make 60 s slice, transcribe slice with `groq/whisper-large-v3-turbo`, call `infer_context`, store context, set status `awaiting_approval`; on error set `failed`
- [x] 5.3 Add `run_stage2(db, file_id, file_path, model, context)`: reuse the deterministic full WAV (re-extract if missing), run full transcription with `context` and diarization in parallel, merge, save utterances, set `ready`; on error set `failed`
- [x] 5.4 Keep the original `run_pipeline` behavior expressible as `run_stage2` with empty context so existing tests/flows still pass
- [x] 5.5 Update/extend ingest tests to cover the stage split and that Stage 1 ends in `awaiting_approval`

## 6. API endpoints

- [x] 6.1 In `app/routes/files.py`, change `/upload` to persist the chosen `model` on the file row (extend `save_file`) and schedule `run_stage1`; the file starts `processing` and becomes `awaiting_approval` after Stage 1
- [x] 6.2 Add `GET /files/{id}/context` returning `{status, context}` (404 if file missing)
- [x] 6.3 Add `POST /files/{id}/context/approve` accepting optional `{context?: str}`; call the guarded approve helper, return `409` if not in `awaiting_approval`, otherwise schedule `run_stage2` with the file's chosen model and approved context
- [x] 6.4 Read the persisted `model` from the file row in the approve handler and pass it into `run_stage2`

## 7. End-to-end verification

- [x] 7.1 Add an integration test: upload → poll status reaches `awaiting_approval` with non-empty context → approve → status reaches `ready` with utterances (Groq/diarization mocked)
- [x] 7.2 Add a test for approve-when-not-awaiting returning `409`
- [x] 7.3 Run the full test suite (`pytest`) and confirm green
- [x] 7.4 Manual smoke test with a real short clip: confirm inferred context appears, edit it, approve, and verify the transcript reflects the steered vocabulary
