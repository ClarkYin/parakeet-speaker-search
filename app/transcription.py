from app.config import settings

_groq_client = None
_deepgram_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _get_deepgram_client():
    global _deepgram_client
    if _deepgram_client is None:
        from deepgram import DeepgramClient
        _deepgram_client = DeepgramClient(settings.deepgram_api_key)
    return _deepgram_client


def _transcribe_groq(audio_path: str, model: str) -> dict:
    client = _get_groq_client()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.split("/")[-1], f),
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    def _w(w):
        if isinstance(w, dict):
            return {"word": w["word"], "start": w["start"], "end": w["end"]}
        return {"word": w.word, "start": w.start, "end": w.end}
    return {"text": response.text, "words": [_w(w) for w in (response.words or [])]}


def _transcribe_deepgram(audio_path: str, model: str) -> dict:
    client = _get_deepgram_client()
    from deepgram import PrerecordedOptions
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    response = client.listen.rest.v("1").transcribe_file(
        {"buffer": audio_data, "mimetype": "audio/wav"},
        PrerecordedOptions(model=model, smart_format=True, punctuate=True, words=True),
    )
    alt = response.results.channels[0].alternatives[0]
    words = [{"word": w.word, "start": w.start, "end": w.end} for w in (alt.words or [])]
    return {"text": alt.transcript, "words": words}


def transcribe(audio_path: str, model: str = "groq/whisper-large-v3-turbo") -> dict:
    """
    model format: "provider/model-name"
    Supported: groq/whisper-large-v3-turbo, groq/whisper-large-v3, deepgram/nova-3
    Returns {"text": str, "words": [{"word": str, "start": float, "end": float}]}
    """
    provider, model_name = model.split("/", 1)
    if provider == "groq":
        return _transcribe_groq(audio_path, model_name)
    elif provider == "deepgram":
        return _transcribe_deepgram(audio_path, model_name)
    raise ValueError(f"Unknown provider: {provider!r}. Use 'groq' or 'deepgram'.")
