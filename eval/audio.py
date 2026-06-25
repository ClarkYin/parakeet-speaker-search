from __future__ import annotations
import os
import re
import subprocess
from eval.models import NormalizedAudio, Chunk


def compute_chunk_ranges(
    duration: float,
    silence_points: list[float],
    max_sec: float | None,
    overlap: float = 1.0,
) -> list[tuple[float, float]]:
    if max_sec is None or duration <= max_sec:
        return [(0.0, duration)]

    ranges: list[tuple[float, float]] = []
    start = 0.0
    silences = sorted(silence_points)
    while start < duration - 1e-6:
        target = start + max_sec
        if target >= duration:
            ranges.append((start, duration))
            break
        # prefer a silence point within the back third of this window
        window_lo = start + max_sec * 0.66
        candidates = [s for s in silences if window_lo <= s <= target]
        cut = max(candidates) if candidates else target
        ranges.append((start, cut))
        start = max(cut - overlap, 0.0) if overlap else cut
    return ranges


def _ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def normalize(input_path: str, out_dir: str) -> NormalizedAudio:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "normalized.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path],
        check=True, capture_output=True,
    )
    return NormalizedAudio(wav_path=out_path, duration=_ffprobe_duration(out_path))


def detect_silences(wav_path: str, noise_db: float = -30.0, min_silence: float = 0.5) -> list[float]:
    proc = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def chunk_audio(audio: NormalizedAudio, out_dir: str, max_sec: float | None, max_bytes: int | None = None) -> list[Chunk]:
    os.makedirs(out_dir, exist_ok=True)
    silences = detect_silences(audio.wav_path) if max_sec else []
    ranges = compute_chunk_ranges(audio.duration, silences, max_sec)
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(ranges):
        path = os.path.join(out_dir, f"chunk_{i:03d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio.wav_path, "-ss", str(start), "-to", str(end),
             "-ac", "1", "-ar", "16000", path],
            check=True, capture_output=True,
        )
        chunks.append(Chunk(path=path, start=start, end=end))
    return chunks
