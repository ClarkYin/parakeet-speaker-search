import pytest
from unittest.mock import patch, MagicMock
from app.ingest import run_pipeline

MOCK_TRANSCRIPT = {
    "text": "Hello everyone thanks for joining",
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

def test_run_pipeline_calls_all_stages():
    mock_db = MagicMock()

    with patch("app.ingest.extract_audio", return_value="/tmp/audio.wav") as mock_extract, \
         patch("app.ingest.transcribe", return_value=MOCK_TRANSCRIPT) as mock_transcribe, \
         patch("app.ingest.diarize", return_value=MOCK_SEGMENTS) as mock_diarize, \
         patch("app.ingest.save_utterances") as mock_save, \
         patch("app.ingest.update_file_status") as mock_status:

        run_pipeline(db=mock_db, file_id="f-123", file_path="/uploads/meeting.mp4")

    mock_extract.assert_called_once_with("/uploads/meeting.mp4")
    mock_transcribe.assert_called_once_with("/tmp/audio.wav", model="groq/whisper-large-v3-turbo")
    mock_diarize.assert_called_once_with("/tmp/audio.wav")
    mock_save.assert_called_once()
    mock_status.assert_called_with(mock_db, "f-123", "ready", speaker_count=2)

def test_run_pipeline_marks_failed_on_error():
    mock_db = MagicMock()

    with patch("app.ingest.extract_audio", side_effect=RuntimeError("ffmpeg not found")), \
         patch("app.ingest.update_file_status") as mock_status:

        run_pipeline(db=mock_db, file_id="f-123", file_path="/uploads/meeting.mp4")

    call_kwargs = mock_status.call_args.kwargs
    assert call_kwargs["status"] == "failed"
    assert "ffmpeg not found" in call_kwargs["error_message"]
