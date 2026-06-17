from app.config import settings

SYSTEM_PROMPT = (
    "Output ONLY a 1-2 sentence description of what the recording is about: "
    "its topic, setting, and any names, jargon, or acronyms likely to appear. "
    "Be descriptive, not a transcript. No preamble, no quotes, no markdown."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def infer_context(rough_text: str) -> str:
    """Return a 1-2 sentence context description inferred from rough_text.

    Returns "" (empty string) without raising on any failure, including an
    empty/whitespace-only input or any exception from the Groq API.
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
            temperature=0.2,
            max_tokens=120,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()
    except Exception:
        return ""
