from unittest.mock import patch, MagicMock
from app.diarization import diarize

def _make_mock_pipeline(segments):
    """segments: list of (start, end, speaker) tuples"""
    mock_turn = lambda start, end: type("Turn", (), {"start": start, "end": end})()

    annotation = MagicMock()
    annotation.itertracks.return_value = [
        (mock_turn(s, e), None, sp) for s, e, sp in segments
    ]

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = annotation
    return mock_pipeline

def test_diarize_returns_speaker_segments():
    mock_pipeline = _make_mock_pipeline([
        (0.0, 5.0, "SPEAKER_00"),
        (5.2, 10.0, "SPEAKER_01"),
    ])

    with patch("app.diarization._get_pipeline", return_value=mock_pipeline):
        result = diarize("/tmp/audio.wav")

    assert len(result) == 2
    assert result[0] == {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}
    assert result[1] == {"speaker": "SPEAKER_01", "start": 5.2, "end": 10.0}

def test_diarize_empty_audio_returns_empty_list():
    mock_pipeline = _make_mock_pipeline([])

    with patch("app.diarization._get_pipeline", return_value=mock_pipeline):
        result = diarize("/tmp/empty.wav")

    assert result == []
