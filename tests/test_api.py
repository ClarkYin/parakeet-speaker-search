import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

def test_upload_returns_file_id_and_processing_status():
    from app.main import app
    client = TestClient(app)

    with patch("app.routes.files.save_file", return_value="file-abc"), \
         patch("app.routes.files.run_pipeline"), \
         patch("app.routes.files.get_db", return_value=iter([MagicMock()])):

        response = client.post(
            "/files/upload",
            files={"file": ("test.wav", b"fake-audio-data", "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["file_id"] == "file-abc"
    assert data["status"] == "processing"

def test_get_file_status_returns_ready():
    from app.main import app
    client = TestClient(app)

    mock_row = {"id": "file-abc", "status": "ready", "speaker_count": 2, "duration_sec": 120.0}
    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchone.return_value = mock_row

    mock_db_instance = MagicMock()
    mock_db_instance.execute.return_value = mock_result

    with patch("app.routes.files.get_db", return_value=iter([mock_db_instance])):
        response = client.get("/files/file-abc/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_search_keyword_mode():
    from app.main import app
    client = TestClient(app)

    mock_rows = [
        {"id": "u-1", "file_id": "f-1", "speaker_label": "SPEAKER_00",
         "start_sec": 1.0, "end_sec": 3.0, "text": "the deadline is tomorrow", "score": 0.9}
    ]

    with patch("app.routes.search.keyword_search", return_value=mock_rows), \
         patch("app.routes.search.get_db", return_value=iter([MagicMock()])):
        response = client.post("/search", json={"query": "deadline", "mode": "keyword"})

    assert response.status_code == 200

def test_upload_rejects_empty_file():
    from app.main import app
    client = TestClient(app)

    response = client.post(
        "/files/upload",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
