import os
import pytest
from unittest.mock import patch
from sqlalchemy import text

from app.database import SessionLocal
from app.storage import save_file, get_file_context, approve_context

MOCK_TRANSCRIPT = {
    "text": "Hello everyone thanks for joining the Apollo budget review",
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.3},
        {"word": "everyone", "start": 0.4, "end": 0.9},
        {"word": "thanks", "start": 5.3, "end": 5.7},
        {"word": "for", "start": 5.8, "end": 6.0},
        {"word": "joining", "start": 6.1, "end": 6.5},
    ],
}
MOCK_SEGMENTS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
    {"speaker": "SPEAKER_01", "start": 5.2, "end": 10.0},
]


@pytest.fixture
def db():
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres not available")
    yield session
    session.close()


def test_full_context_injection_flow(db):
    from app import ingest

    file_id = save_file(db, filename="integration_test.mp4", model="groq/whisper-large-v3")
    try:
        # --- Stage 1: rough pass + inferred context, ends awaiting_approval ---
        with patch.object(ingest, "extract_audio", return_value="/tmp/it_audio.wav"), \
             patch.object(ingest, "extract_audio_slice", return_value="/tmp/it_slice.wav"), \
             patch.object(ingest, "transcribe", return_value={"text": "rough words apollo", "words": []}), \
             patch.object(ingest, "infer_context", return_value="An Apollo project budget review meeting."):
            ingest.run_stage1(db, file_id, "/tmp/integration_test.mp4")

        ctx = get_file_context(db, file_id)
        assert ctx["status"] == "awaiting_approval"
        assert ctx["context"] == "An Apollo project budget review meeting."

        # --- Approve (guarded) flips to processing ---
        affected = approve_context(db, file_id, ctx["context"])
        assert affected == 1
        assert get_file_context(db, file_id)["status"] == "processing"

        # --- Stage 2: full transcription steered by context, ends ready w/ utterances ---
        with patch.object(ingest, "extract_audio", return_value="/tmp/it_audio.wav"), \
             patch("app.ingest.os.path.exists", return_value=False), \
             patch.object(ingest, "transcribe", return_value=MOCK_TRANSCRIPT) as mock_tx, \
             patch.object(ingest, "diarize", return_value=MOCK_SEGMENTS), \
             patch("app.storage._embed", return_value=[[0.0] * 384, [0.0] * 384]):
            ingest.run_stage2(db, file_id, "/tmp/integration_test.mp4",
                              model="groq/whisper-large-v3", context=ctx["context"])
            # context must be forwarded to transcription
            assert mock_tx.call_args.kwargs["context"] == "An Apollo project budget review meeting."

        # final state
        row = db.execute(text("SELECT status FROM files WHERE id = :id"), {"id": file_id}).mappings().fetchone()
        assert row["status"] == "ready"
        n = db.execute(text("SELECT COUNT(*) FROM utterances WHERE file_id = :id"), {"id": file_id}).scalar()
        assert n >= 1
    finally:
        db.execute(text("DELETE FROM files WHERE id = :id"), {"id": file_id})
        db.commit()
