import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

def test_upload_returns_file_id_and_processing_status():
    from app.main import app
    client = TestClient(app)

    with patch("app.routes.files.save_file", return_value="file-abc"), \
         patch("app.routes.files.run_stage1"), \
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

def test_get_context_returns_status_and_context():
    from app.main import app
    client = TestClient(app)

    expected = {"status": "awaiting_approval", "context": "a budget meeting"}

    with patch("app.routes.files.get_file_context", return_value=expected), \
         patch("app.routes.files.get_db", return_value=iter([MagicMock()])):
        response = client.get("/files/abc/context")

    assert response.status_code == 200
    assert response.json() == expected

def test_get_context_404_when_missing():
    from app.main import app
    client = TestClient(app)

    with patch("app.routes.files.get_file_context", return_value=None), \
         patch("app.routes.files.get_db", return_value=iter([MagicMock()])):
        response = client.get("/files/abc/context")

    assert response.status_code == 404

def test_approve_schedules_stage2():
    from app.main import app
    client = TestClient(app)

    mock_db = MagicMock()
    mock_db.execute.return_value.mappings.return_value.fetchone.return_value = {
        "filename": "m.mp4",
        "model": "deepgram/nova-3",
    }

    mock_run_stage2 = MagicMock()

    with patch("app.routes.files.get_file_context", return_value={"status": "awaiting_approval", "context": "old ctx"}), \
         patch("app.routes.files.approve_context", return_value=1), \
         patch("app.routes.files.run_stage2", mock_run_stage2), \
         patch("app.routes.files.get_db", return_value=iter([mock_db])):
        response = client.post("/files/abc/context/approve", json={"context": "new ctx"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    # TestClient runs background tasks synchronously after the response
    mock_run_stage2.assert_called_once()
    call_kwargs = mock_run_stage2.call_args
    assert call_kwargs.kwargs.get("context") == "new ctx" or call_kwargs.args[-1] == "new ctx"
    assert call_kwargs.kwargs.get("model") == "deepgram/nova-3" or "deepgram/nova-3" in str(call_kwargs)

def test_approve_returns_409_when_not_awaiting():
    from app.main import app
    client = TestClient(app)

    with patch("app.routes.files.get_file_context", return_value={"status": "ready", "context": "x"}), \
         patch("app.routes.files.approve_context", return_value=0), \
         patch("app.routes.files.get_db", return_value=iter([MagicMock()])):
        response = client.post("/files/abc/context/approve", json={"context": "new ctx"})

    assert response.status_code == 409
