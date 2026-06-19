## ADDED Requirements

### Requirement: Automatic context inference on upload

On upload, the system SHALL run a fast rough transcription of the beginning of the recording and use an LLM to infer a short context description, then pause for human approval instead of completing transcription immediately.

#### Scenario: Upload starts inference and pauses for approval
- **WHEN** a file is uploaded
- **THEN** the system transcribes approximately the first 60 seconds using `groq/whisper-large-v3-turbo`
- **AND** passes that rough text to an LLM to produce a short keyword-style context hint (topic plus likely names/jargon, not a narrative description)
- **AND** stores the description on the file and sets the file status to `awaiting_approval`

#### Scenario: Recording shorter than the sample window
- **WHEN** the uploaded recording is shorter than 60 seconds
- **THEN** the system uses the entire recording for the rough pass
- **AND** still produces a context description and sets status to `awaiting_approval`

#### Scenario: Inference failure is non-fatal
- **WHEN** the LLM context inference call fails
- **THEN** the file status still becomes `awaiting_approval`
- **AND** the stored context is empty so the user can supply their own

### Requirement: Context review and approval

The system SHALL expose the inferred context for review and SHALL let the user approve it unchanged or replace it with their own text before the full transcription runs.

#### Scenario: Read the inferred context
- **WHEN** a client requests `GET /files/{id}/context`
- **THEN** the system returns the current status and the stored context description

#### Scenario: Approve using the inferred context
- **WHEN** a client calls `POST /files/{id}/context/approve` with no context override
- **THEN** the stored inferred context is used for the full transcription
- **AND** the file status returns to `processing`

#### Scenario: Approve with an overridden context
- **WHEN** a client calls `POST /files/{id}/context/approve` with a `context` value
- **THEN** the provided value replaces the stored context and is used for the full transcription

#### Scenario: Approve with empty context skips steering
- **WHEN** a client approves with an empty context
- **THEN** the full transcription runs with no `initial_prompt`

#### Scenario: Approval is rejected outside the awaiting state
- **WHEN** a client calls the approve endpoint for a file that is not in `awaiting_approval`
- **THEN** the system rejects the request with a 409 conflict and does not start a second transcription

### Requirement: Context-steered full transcription

After approval, the full transcription SHALL run with the approved context steering the decoder, then diarization and merging SHALL complete the pipeline as in the single-stage flow.

#### Scenario: Groq Whisper receives the context as initial_prompt
- **WHEN** the full transcription runs for a Groq Whisper model with a non-empty approved context
- **THEN** the context is passed as the Whisper `prompt` parameter

#### Scenario: Full transcription uses the model chosen at upload
- **WHEN** the full transcription runs after approval
- **THEN** it uses the transcription model that was selected when the file was uploaded, not a hardcoded default

#### Scenario: Pipeline completes to ready
- **WHEN** the full transcription, diarization, and merge succeed
- **THEN** utterances are saved, speaker count is recorded, and the file status becomes `ready`

#### Scenario: Deepgram has no free-text prompt
- **WHEN** the full transcription runs for a Deepgram model
- **THEN** the approved context does not break transcription
- **AND** the context is applied as keyterms or ignored, never sent as an unsupported parameter

### Requirement: Context persistence and status lifecycle

The system SHALL persist the context on the file record and SHALL move each file through the status lifecycle `processing → awaiting_approval → processing → ready`, with `failed` reachable from any stage on error.

#### Scenario: Context is persisted and retrievable
- **WHEN** a context description is inferred or approved
- **THEN** it is stored in the `context` column of the `files` table
- **AND** is returned by `GET /files/{id}/context`

#### Scenario: Failure during either stage marks the file failed
- **WHEN** an unrecoverable error occurs during the rough pass or the full transcription
- **THEN** the file status becomes `failed` with an error message recorded
