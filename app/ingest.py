import os
import subprocess
import threading
from app.transcription import transcribe
from app.diarization import diarize
from app.merger import merge
from app.storage import save_utterances, update_file_status, set_file_context
from app.context_inference import infer_context

def extract_audio(file_path: str) -> str:
    """Extracts audio from video/audio file to mono 16kHz WAV using ffmpeg."""
    out_path = file_path.rsplit(".", 1)[0] + "_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", out_path],
        check=True,
        capture_output=True,
    )
    return out_path

def extract_audio_slice(audio_path: str, seconds: int = 60) -> str:
    """Extracts a short slice from an existing audio WAV file using ffmpeg."""
    out_path = audio_path.rsplit(".", 1)[0] + "_slice.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-t", str(seconds), "-ac", "1", "-ar", "16000", out_path],
        check=True,
        capture_output=True,
    )
    return out_path

def run_stage1(db, file_id: str, file_path: str):
    """Stage 1: rough-transcribe first 60s, infer context, store it, set status awaiting_approval."""
    try:
        audio_path = extract_audio(file_path)
        slice_path = extract_audio_slice(audio_path, 60)
        rough = transcribe(slice_path, model="groq/whisper-large-v3-turbo")
        context = infer_context(rough.get("text", ""))
        set_file_context(db, file_id, context)
        update_file_status(db, file_id, "awaiting_approval")
    except Exception as exc:
        update_file_status(db, file_id, status="failed", error_message=str(exc))

def run_stage2(db, file_id: str, file_path: str, model: str, context: str):
    """Stage 2: full transcription steered by approved context, plus diarization, merge, save."""
    try:
        expected_audio_path = file_path.rsplit(".", 1)[0] + "_audio.wav"
        if os.path.exists(expected_audio_path):
            audio_path = expected_audio_path
        else:
            audio_path = extract_audio(file_path)

        transcript_result: dict = {}
        segments_result: list = []
        errors: list = []

        def do_transcribe():
            try:
                transcript_result.update(transcribe(audio_path, model=model, context=context))
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

def run_pipeline(db, file_id: str, file_path: str, model: str = "groq/whisper-large-v3-turbo"):
    """Backward-compatible wrapper: delegates to run_stage2 with empty context."""
    run_stage2(db, file_id, file_path, model=model, context="")
