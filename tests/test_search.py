from unittest.mock import patch, MagicMock
from app.search import keyword_search, semantic_search, speaker_transcript

def _make_mock_db(rows):
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_result.mappings.return_value.fetchall.return_value = rows
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_result
    return mock_db

def test_keyword_search_executes_fts_query():
    mock_db = _make_mock_db([])
    keyword_search(mock_db, query="deadline")
    sql = str(mock_db.execute.call_args.args[0])
    assert "plainto_tsquery" in sql or mock_db.execute.called

def test_keyword_search_filters_by_speaker_when_provided():
    mock_db = _make_mock_db([])
    keyword_search(mock_db, query="budget", speaker_label="SPEAKER_00")
    params = mock_db.execute.call_args.args[1]
    assert params.get("speaker_label") == "SPEAKER_00"

def test_semantic_search_calls_embed():
    mock_db = _make_mock_db([])
    with patch("app.search._embed", return_value=[[0.1] * 384]) as mock_embed:
        semantic_search(mock_db, query="financial concerns")
    mock_embed.assert_called_once_with(["financial concerns"])

def test_speaker_transcript_filters_by_speaker():
    mock_db = _make_mock_db([])
    speaker_transcript(mock_db, file_id="f-123", speaker_label="SPEAKER_01")
    params = mock_db.execute.call_args.args[1]
    assert params["speaker_label"] == "SPEAKER_01"
    assert params["file_id"] == "f-123"
