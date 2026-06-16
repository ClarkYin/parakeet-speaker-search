def _find_speaker(word_mid: float, segments: list) -> str:
    # First try exact containment
    for seg in segments:
        if seg["start"] <= word_mid <= seg["end"]:
            return seg["speaker"]

    # Fall back to nearest segment by distance from word midpoint
    if not segments:
        return "UNKNOWN"
    nearest = min(segments, key=lambda s: min(abs(word_mid - s["start"]), abs(word_mid - s["end"])))
    return nearest["speaker"]


def merge(words: list, segments: list) -> list:
    """
    words:    [{"word": str, "start": float, "end": float}]
    segments: [{"speaker": str, "start": float, "end": float}]
    returns:  [{"speaker_label": str, "start_sec": float, "end_sec": float, "text": str}]
    """
    if not words:
        return []

    utterances = []
    current_speaker = None
    current_words: list = []
    current_start: float = 0.0

    for w in words:
        mid = (w["start"] + w["end"]) / 2
        speaker = _find_speaker(mid, segments)

        if speaker != current_speaker:
            if current_words:
                utterances.append({
                    "speaker_label": current_speaker,
                    "start_sec": current_start,
                    "end_sec": current_words[-1]["end"],
                    "text": " ".join(x["word"] for x in current_words),
                })
            current_speaker = speaker
            current_words = [w]
            current_start = w["start"]
        else:
            current_words.append(w)

    if current_words:
        utterances.append({
            "speaker_label": current_speaker,
            "start_sec": current_start,
            "end_sec": current_words[-1]["end"],
            "text": " ".join(x["word"] for x in current_words),
        })

    return utterances
