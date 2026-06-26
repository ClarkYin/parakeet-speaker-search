import os
from unittest.mock import patch
from eval.models import NormalizedAudio, TranscriptResult, Word
from eval.engines.base import Engine, register, get_registry, available_engines


class FakeEngine(Engine):
    id = "fake/one"
    needs_keys = ["FAKE_KEY"]

    def _transcribe_chunk(self, wav_path, offset):
        return TranscriptResult(self.id, "hi", [Word("hi", 0.0, 0.2)])


def test_available_false_when_key_missing(monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    assert FakeEngine().available() is False


def test_available_true_when_key_present(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    assert FakeEngine().available() is True


def test_register_and_lookup():
    e = FakeEngine()
    register(e)
    assert get_registry()["fake/one"] is e


def test_transcribe_offsets_timestamps_across_chunks():
    audio = NormalizedAudio(wav_path="/tmp/a.wav", duration=1000.0)
    calls = []

    class TwoChunk(Engine):
        id = "fake/two"
        max_chunk_sec = 600.0

        def _transcribe_chunk(self, wav_path, offset):
            calls.append(offset)
            return TranscriptResult(self.id, "w", [Word("w", 0.0, 0.2)])

    fake_chunks = [
        type("C", (), {"path": "/tmp/c0.wav", "start": 0.0, "end": 600.0})(),
        type("C", (), {"path": "/tmp/c1.wav", "start": 600.0, "end": 1000.0})(),
    ]
    with patch("eval.engines.base.chunk_audio", return_value=fake_chunks):
        result = TwoChunk().transcribe(audio, work_dir="/tmp")

    assert calls == [0.0, 600.0]
    # second chunk's word shifted by its 600s offset
    assert result.words[1].start == 600.0
