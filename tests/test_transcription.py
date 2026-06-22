from unittest.mock import patch, MagicMock
from app.transcription import transcribe

def test_transcribe_returns_word_timestamps():
    mock_word = MagicMock()
    mock_word.word = "hello"
    mock_word.start = 0.0
    mock_word.end = 0.3

    mock_response = MagicMock()
    mock_response.text = "hello world"
    mock_response.words = [mock_word]

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("app.transcription._get_groq_client", return_value=mock_client):
        result = transcribe("/tmp/audio.wav")

    assert result["text"] == "hello world"
    assert len(result["words"]) == 1
    assert result["words"][0] == {"word": "hello", "start": 0.0, "end": 0.3}

def test_transcribe_passes_correct_model():
    mock_word = MagicMock()
    mock_word.word = "hi"
    mock_word.start = 0.0
    mock_word.end = 0.2

    mock_response = MagicMock()
    mock_response.text = "hi"
    mock_response.words = [mock_word]

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("app.transcription._get_groq_client", return_value=mock_client):
        transcribe("/tmp/audio.wav")

    call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == "whisper-large-v3-turbo"
    assert call_kwargs["response_format"] == "verbose_json"
    assert "word" in call_kwargs["timestamp_granularities"]


def _mock_groq_client():
    mock_word = MagicMock()
    mock_word.word = "hi"
    mock_word.start = 0.0
    mock_word.end = 0.2
    mock_response = MagicMock()
    mock_response.text = "hi"
    mock_response.words = [mock_word]
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response
    return mock_client


def test_transcribe_forwards_context_as_prompt():
    mock_client = _mock_groq_client()
    with patch("app.transcription._get_groq_client", return_value=mock_client):
        transcribe("/tmp/audio.wav", context="meeting about the Apollo budget")

    call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["prompt"] == "meeting about the Apollo budget"


def test_transcribe_no_context_sends_no_prompt():
    # Default (no context) and explicit empty string must both omit `prompt`.
    for ctx in (None, ""):
        mock_client = _mock_groq_client()
        with patch("app.transcription._get_groq_client", return_value=mock_client):
            if ctx is None:
                transcribe("/tmp/audio.wav")
            else:
                transcribe("/tmp/audio.wav", context=ctx)
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert "prompt" not in call_kwargs
