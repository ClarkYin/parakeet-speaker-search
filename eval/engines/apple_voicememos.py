from __future__ import annotations
import re
from eval.models import TranscriptResult, Word, Segment

_HEADER = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s+Speaker\s+(\d+)", re.IGNORECASE)
ENGINE_ID = "apple/voicememos"


def _seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_apple_transcript(text: str) -> TranscriptResult:
    utterances: list[tuple[float, str, list[str]]] = []  # (start, speaker, tokens)
    cur_start, cur_spk, cur_tokens = None, None, []
    for line in text.splitlines():
        m = _HEADER.match(line.strip())
        if m:
            if cur_start is not None and cur_tokens:
                utterances.append((cur_start, cur_spk, cur_tokens))
            cur_start = _seconds(m.group(1), m.group(2), m.group(3))
            cur_spk = f"SPEAKER_{int(m.group(4)) - 1:02d}"
            cur_tokens = []
        elif line.strip():
            cur_tokens.extend(line.split())
    if cur_start is not None and cur_tokens:
        utterances.append((cur_start, cur_spk, cur_tokens))

    words: list[Word] = []
    segments: list[Segment] = []
    for i, (start, spk, tokens) in enumerate(utterances):
        end = utterances[i + 1][0] if i + 1 < len(utterances) else start + len(tokens) * 0.4
        step = (end - start) / max(len(tokens), 1)
        for j, tok in enumerate(tokens):
            ws = start + j * step
            words.append(Word(tok.strip(".,!?;:"), round(ws, 3), round(ws + step, 3), spk))
        segments.append(Segment(spk, start, end))
    full = " ".join(w.text for w in words)
    return TranscriptResult(ENGINE_ID, full, words, speakers=segments)


def load(path: str) -> TranscriptResult:
    with open(path) as f:
        return parse_apple_transcript(f.read())
