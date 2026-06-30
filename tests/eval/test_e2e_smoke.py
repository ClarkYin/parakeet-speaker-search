from unittest.mock import patch
from eval.models import TranscriptResult, Word
from eval.engines.base import Engine, register

class EchoEngine(Engine):
    id = "echo/test"
    def _transcribe_chunk(self, wav_path, offset):
        return TranscriptResult(self.id, "the quick brown fox",
                                [Word("the", 0, 1), Word("quick", 1, 2), Word("brown", 2, 3), Word("fox", 3, 4)])

def test_full_run_writes_report(tmp_path):
    register(EchoEngine())
    fake_audio = type("NA", (), {"wav_path": str(tmp_path / "n.wav"), "duration": 5.0})()
    fake_chunk = type("C", (), {"path": "x", "start": 0.0, "end": 5.0})()
    from eval import cli
    with patch.object(cli, "normalize", return_value=fake_audio), \
         patch("eval.engines.base.chunk_audio", return_value=[fake_chunk]), \
         patch.object(cli, "diarize", return_value=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]), \
         patch.object(cli, "detect_silences", return_value=[]):
        scores, results = cli.run(str(tmp_path / "in.wav"), ["echo/test"], str(tmp_path))
    assert (tmp_path / "report.md").exists()
    assert scores[0].engine_id == "echo/test"
    assert scores[0].wer == 0.0
