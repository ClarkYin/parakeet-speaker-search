import pathlib
import pytest

@pytest.fixture(autouse=True)
def tmp_audio_wav(tmp_path):
    """Create a dummy /tmp/audio.wav so transcription tests can open the file."""
    dummy = pathlib.Path("/tmp/audio.wav")
    dummy.touch()
    yield dummy
