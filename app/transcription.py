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
        _deepgram_client = DeepgramClient(api_key=settings.deepgram_api_key)
    return _deepgram_client


def _transcribe_groq(audio_path: str, model: str, context: str | None = None) -> dict:
    client = _get_groq_client()
    with open(audio_path, "rb") as f:
        kwargs = dict(
            file=(audio_path.split("/")[-1], f),
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        if context:
            kwargs["prompt"] = context
        response = client.audio.transcriptions.create(**kwargs)
    def _w(w):
        if isinstance(w, dict):
            return {"word": w["word"], "start": w["start"], "end": w["end"]}
        return {"word": w.word, "start": w.start, "end": w.end}
    return {"text": response.text, "words": [_w(w) for w in (response.words or [])]}


def _transcribe_deepgram(audio_path: str, model: str, context: str | None = None) -> dict:
    # `context` is accepted for signature parity but intentionally ignored:
    # Deepgram has no free-text prompt equivalent, so we never forward it.
    client = _get_deepgram_client()
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    response = client.listen.v1.media.transcribe_file(
        request=audio_data,
        model=model,
        smart_format=True,
        punctuate=True,
    )
    alt = response.results.channels[0].alternatives[0]
    words = []
    for w in (alt.words or []):
        word_text = w.word if hasattr(w, 'word') else w.get('word', '')
        start = w.start if hasattr(w, 'start') else w.get('start', 0)
        end = w.end if hasattr(w, 'end') else w.get('end', 0)
        words.append({"word": word_text, "start": start, "end": end})
    return {"text": alt.transcript, "words": words}


def transcribe(audio_path: str, model: str = "groq/whisper-large-v3-turbo",
               context: str | None = None) -> dict:
    """
    model format: "provider/model-name"
    Supported: groq/whisper-large-v3-turbo, groq/whisper-large-v3, deepgram/nova-3
    context: optional free-text description of the recording. For Groq Whisper it
        is passed as the `prompt` parameter to steer the decoder toward expected
        vocabulary; for Deepgram it is ignored. Empty/None preserves default behavior.
    Returns {"text": str, "words": [{"word": str, "start": float, "end": float}]}
    """
    provider, model_name = model.split("/", 1)
    if provider == "groq":
        return _transcribe_groq(audio_path, model_name, context=context)
    elif provider == "deepgram":
        return _transcribe_deepgram(audio_path, model_name, context=context)
    raise ValueError(f"Unknown provider: {provider!r}. Use 'groq' or 'deepgram'.")
