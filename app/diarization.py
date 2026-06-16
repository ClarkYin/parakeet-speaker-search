from app.config import settings

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline  # lazy import — heavy dependency
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
    return _pipeline

def diarize(audio_path: str) -> list[dict]:
    """
    Returns [{"speaker": str, "start": float, "end": float}]
    """
    pipeline = _get_pipeline()
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })
    return segments
