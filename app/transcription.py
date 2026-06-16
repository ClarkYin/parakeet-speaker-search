from groq import Groq
from app.config import settings

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client

def transcribe(audio_path: str) -> dict:
    """
    Returns {"text": str, "words": [{"word": str, "start": float, "end": float}]}
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.split("/")[-1], f),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    words = [
        {"word": w.word, "start": w.start, "end": w.end}
        for w in (response.words or [])
    ]
    return {"text": response.text, "words": words}
