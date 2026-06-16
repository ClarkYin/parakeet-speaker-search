import pytest
from unittest.mock import patch, MagicMock, call
from app.storage import save_file, save_utterances

def test_save_file_inserts_record():
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchone.return_value = ("test-uuid",)

    file_id = save_file(mock_db, filename="meeting.mp4", duration_sec=120.0)

    assert file_id == "test-uuid"
    assert mock_db.execute.called
    assert mock_db.commit.called

def test_save_utterances_inserts_with_embeddings():
    mock_db = MagicMock()
    utterances = [
        {"speaker_label": "SPEAKER_00", "start_sec": 0.0, "end_sec": 5.0, "text": "Hello world"},
        {"speaker_label": "SPEAKER_01", "start_sec": 5.5, "end_sec": 10.0, "text": "Hi there"},
    ]
    mock_embeddings = [[0.1] * 384, [0.2] * 384]

    with patch("app.storage._embed", return_value=mock_embeddings):
        save_utterances(mock_db, file_id="file-123", utterances=utterances)

    assert mock_db.execute.call_count == 2
    assert mock_db.commit.called

def test_save_utterances_empty_list_does_nothing():
    mock_db = MagicMock()
    with patch("app.storage._embed", return_value=[]):
        save_utterances(mock_db, file_id="file-123", utterances=[])
    mock_db.execute.assert_not_called()
