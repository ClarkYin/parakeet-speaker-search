from unittest.mock import patch, MagicMock
from eval.engines.assemblyai import AssemblyAI
from eval.engines.elevenlabs import ElevenLabsScribe
from eval.engines.openai_whisper import OpenAITranscribe
from eval.engines.gemini import Gemini

def test_all_skip_when_keys_missing(monkeypatch):
    for var in ["ASSEMBLYAI_API_KEY", "ELEVENLABS_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    assert AssemblyAI().available() is False
    assert ElevenLabsScribe().available() is False
    assert OpenAITranscribe().available() is False
    assert Gemini().available() is False

def test_openai_maps_words(tmp_path):
    wav = tmp_path / "c.wav"; wav.write_bytes(b"RIFF")
    word = MagicMock(); word.word = "hi"; word.start = 0.0; word.end = 0.2
    resp = MagicMock(); resp.text = "hi"; resp.words = [word]
    client = MagicMock(); client.audio.transcriptions.create.return_value = resp
    with patch("eval.engines.openai_whisper._get_client", return_value=client):
        out = OpenAITranscribe()._transcribe_chunk(str(wav), 0.0)
    assert out.words[0].text == "hi"

def test_assemblyai_declares_diarization():
    assert AssemblyAI().diarizes is True
