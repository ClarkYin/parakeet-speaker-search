import subprocess
import threading
from app.transcription import transcribe
from app.diarization import diarize
from app.merger import merge
from app.storage import save_utterances, update_file_status

def extract_audio(file_path: str) -> str:
    """Extracts audio from video/audio file to mono 16kHz WAV using ffmpeg."""
    out_path = file_path.rsplit(".", 1)[0] + "_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", out_path],
        check=True,
        capture_output=True,
    )
    return out_path

def run_pipeline(db, file_id: str, file_path: str, model: str = "groq/whisper-large-v3-turbo"):
    try:
        audio_path = extract_audio(file_path)

        transcript_result: dict = {}
        segments_result: list = []
        errors: list = []

        def do_transcribe():
            try:
                transcript_result.update(transcribe(audio_path, model=model))
            except Exception as e:
                errors.append(e)

        def do_diarize():
            try:
                segments_result.extend(diarize(audio_path))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_transcribe)
        t2 = threading.Thread(target=do_diarize)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if errors:
            raise errors[0]

        utterances = merge(transcript_result.get("words", []), segments_result)
        save_utterances(db, file_id=file_id, utterances=utterances)

        speaker_count = len({u["speaker_label"] for u in utterances})
        update_file_status(db, file_id, "ready", speaker_count=speaker_count)

    except Exception as exc:
        update_file_status(db, file_id, status="failed", error_message=str(exc))
