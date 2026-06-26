from __future__ import annotations
import re

_normalizer = None


def _fallback(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", text.lower())).strip()


def normalize_text(text: str) -> str:
    global _normalizer
    if _normalizer is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _normalizer = EnglishTextNormalizer()
        except Exception:
            _normalizer = _fallback
    out = _normalizer(text)
    return re.sub(r"\s+", " ", out).strip()
