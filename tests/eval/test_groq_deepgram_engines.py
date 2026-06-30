from unittest.mock import patch, MagicMock
from eval.engines.groq_whisper import GroqWhisper
from eval.engines.deepgram import Deepgram

def test_groq_maps_words(tmp_path):
    wav = tmp_path / "c.wav"; wav.write_bytes(b"RIFF")
    word = MagicMock(); word.word = "hi"; word.start = 0.0; word.end = 0.2
    resp = MagicMock(); resp.text = "hi"; resp.words = [word]
    client = MagicMock(); client.audio.transcriptions.create.return_value = resp
    with patch("eval.engines.groq_whisper._get_client", return_value=client):
        out = GroqWhisper("whisper-large-v3-turbo")._transcribe_chunk(str(wav), 0.0)
    assert out.words[0].text == "hi"
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-large-v3-turbo"

def test_groq_needs_key():
    assert GroqWhisper("whisper-large-v3").needs_keys == ["GROQ_API_KEY"]

def test_deepgram_maps_words_with_speakers(tmp_path):
    wav = tmp_path / "c.wav"; wav.write_bytes(b"RIFF")
    w = MagicMock(); w.word = "hello"; w.start = 0.0; w.end = 0.4; w.speaker = 0
    alt = MagicMock(); alt.transcript = "hello"; alt.words = [w]
    chan = MagicMock(); chan.alternatives = [alt]
    resp = MagicMock(); resp.results.channels = [chan]
    client = MagicMock(); client.listen.v1.media.transcribe_file.return_value = resp
    with patch("eval.engines.deepgram._get_client", return_value=client):
        out = Deepgram()._transcribe_chunk(str(wav), 0.0)
    assert out.words[0].text == "hello"
    assert out.words[0].speaker == "SPEAKER_00"

def test_deepgram_is_diarizer():
    assert Deepgram().diarizes is True
