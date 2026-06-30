from unittest.mock import patch
from eval.models import TranscriptResult, Word, Segment, Reference, Window
from eval.cli import attribute_speakers, score_engine, run

def test_attribute_speakers_assigns_from_segments():
    result = TranscriptResult("e", "hi yo", [Word("hi", 0.2, 0.5), Word("yo", 6.0, 6.3)])
    segs = [Segment("SPEAKER_00", 0.0, 5.0), Segment("SPEAKER_01", 5.2, 10.0)]
    out = attribute_speakers(result, segs)
    assert out.words[0].speaker == "SPEAKER_00"
    assert out.words[1].speaker == "SPEAKER_01"
    assert out.speakers is not None

def test_score_engine_perfect_match_is_zero_wer():
    ref = Reference("aud", [Window(0.0, 10.0, [Word("hello", 1.0, 1.4, "SPEAKER_00"),
                                               Word("world", 1.5, 1.9, "SPEAKER_00")], corrected=True)])
    result = TranscriptResult("e", "hello world",
                              [Word("hello", 1.0, 1.4, "SPEAKER_00"), Word("world", 1.5, 1.9, "SPEAKER_00")])
    score = score_engine(result, ref)
    assert score.wer == 0.0
    assert score.cpwer == 0.0

def test_run_invokes_engines(tmp_path):
    from eval.engines.base import Engine, register
    class Stub(Engine):
        id = "stub/x"
        def _transcribe_chunk(self, wav_path, offset):
            return TranscriptResult(self.id, "hello world", [Word("hello", 0, 1), Word("world", 1, 2)])
    register(Stub())
    fake_audio = type("NA", (), {"wav_path": str(tmp_path / "a.wav"), "duration": 5.0})()
    fake_chunk = type("C", (), {"path": "x", "start": 0.0, "end": 5.0})()
    with patch("eval.cli.normalize", return_value=fake_audio), \
         patch("eval.engines.base.chunk_audio", return_value=[fake_chunk]), \
         patch("eval.cli.detect_silences", return_value=[]), \
         patch("eval.cli.diarize", return_value=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]):
        scores, results = run(str(tmp_path / "in.wav"), ["stub/x"], str(tmp_path))
    assert any(r.engine_id == "stub/x" for r in results)
