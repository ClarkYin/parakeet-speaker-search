from __future__ import annotations
import json
from collections import Counter
from eval.models import Word, Window, Reference, TranscriptResult


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
    if not results:
        return []
    modal_len = Counter(len(r) for r in results).most_common(1)[0][0]
    group = [r for r in results if len(r) == modal_len]
    consensus: list[Word] = []
    for i in range(modal_len):
        texts = [r[i].text for r in group]
        winner = Counter(texts).most_common(1)[0][0]
        src = next(r[i] for r in group if r[i].text == winner)
        consensus.append(Word(winner, src.start, src.end, src.speaker))
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
