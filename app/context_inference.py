from app.config import settings

SYSTEM_PROMPT = (
    "You write a short transcription hint for a speech-to-text model. "
    "From the rough transcript, output only the subject matter plus any specific "
    "names, places, organizations, acronyms, or unusual/technical terms likely to "
    "be spoken. Format as a brief topic phrase or comma-separated keywords, at most "
    "about 25 words. Do NOT write full sentences. Do NOT describe the speaker, their "
    "tone, mood, or style. No commentary, preamble, quotes, or markdown. If nothing "
    "distinctive stands out, give just a short topic phrase."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def infer_context(rough_text: str) -> str:
    """Return a short, keyword-style context hint inferred from rough_text.

    The hint names the topic plus likely proper nouns/jargon (no narrative
    sentences) so it can steer a speech-to-text decoder without being echoed
    into the transcript. Returns "" (empty string) without raising on any
    failure, including empty/whitespace input or any exception from the Groq API.
    """
    if not rough_text or not rough_text.strip():
        return ""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.inference_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rough_text},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        # Collapse any whitespace/newlines to single spaces (and strip).
        return " ".join(content.split())
    except Exception:
        return ""
