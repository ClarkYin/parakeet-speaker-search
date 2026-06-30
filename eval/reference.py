from __future__ import annotations
import difflib
import json
from collections import Counter
from eval.models import Word, Window, Reference, TranscriptResult
from eval.normalize import normalize_text


def _nearest(points: list[float], target: float) -> float:
    return min(points, key=lambda p: abs(p - target)) if points else target


def sample_windows(duration: float, silence_points: list[float], k: int = 5, length: float = 180.0) -> list[tuple[float, float]]:
    if duration <= length:
        return [(0.0, duration)]
    centers = [duration * (i + 1) / (k + 1) for i in range(k)]
    wins: list[tuple[float, float]] = []
    last_end = 0.0
    for c in centers:
        start = max(_nearest(silence_points, c - length / 2), last_end)
        end = min(start + length, duration)
        if end - start < 1.0:
            continue
        wins.append((round(start, 3), round(end, 3)))
        last_end = end
    return wins


def words_in_window(result: TranscriptResult, start: float, end: float) -> list[Word]:
    return [w for w in result.words if start <= (w.start + w.end) / 2 <= end]


def rover(results: list[list[Word]]) -> list[Word]:
    """Align transcripts to a central pivot and take a per-position majority vote.

    Real ASR transcripts never share a word count, so a positional vote across
    equal-length lists degenerates to "pick one transcript". Instead we choose
    the medoid (the transcript with the least total word-error distance to the
    others) as the pivot, align every other transcript onto it, and vote per
    pivot position. The result corrects pivot words wherever a majority of the
    aligned transcripts agree on a different word — and is never just the
    longest input.
    """
    import jiwer

    groups = [r for r in results if r]
    if not groups:
        return []
    if len(groups) == 1:
        return list(groups[0])

    texts = [normalize_text(" ".join(w.text for w in r)) or " " for r in groups]
    pivot_idx = min(
        range(len(groups)),
        key=lambda i: sum(jiwer.wer(texts[i], texts[j]) for j in range(len(groups)) if j != i),
    )
    pivot = groups[pivot_idx]
    pivot_words = [w.text for w in pivot]
    votes: list[list[str]] = [[w] for w in pivot_words]

    for k, other in enumerate(groups):
        if k == pivot_idx:
            continue
        other_words = [w.text for w in other]
        matcher = difflib.SequenceMatcher(a=pivot_words, b=other_words, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("equal", "replace"):
                for off in range(min(i2 - i1, j2 - j1)):
                    votes[i1 + off].append(other_words[j1 + off])

    consensus: list[Word] = []
    for i, candidates in enumerate(votes):
        winner = Counter(candidates).most_common(1)[0][0]
        consensus.append(Word(winner, pivot[i].start, pivot[i].end, pivot[i].speaker))
    return consensus


def build_reference(audio_id: str, results: list[TranscriptResult], windows: list[tuple[float, float]], seed: TranscriptResult | None = None) -> Reference:
    out_windows: list[Window] = []
    for start, end in windows:
        per_engine = [words_in_window(r, start, end) for r in results]
        if seed is not None:
            seed_words = words_in_window(seed, start, end)
            per_engine = [seed_words, seed_words] + per_engine  # extra weight
        consensus = rover([p for p in per_engine if p])
        out_windows.append(Window(start=start, end=end, words=consensus, corrected=False))
    return Reference(audio_id=audio_id, windows=out_windows)


def save_reference(ref: Reference, path: str) -> None:
    with open(path, "w") as f:
        json.dump(ref.to_dict(), f, indent=2)


def load_reference(path: str) -> Reference:
    with open(path) as f:
        return Reference.from_dict(json.load(f))
