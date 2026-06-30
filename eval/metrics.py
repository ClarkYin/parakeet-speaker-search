from __future__ import annotations
from eval.models import Word, Segment
from eval.normalize import normalize_text


def wer(reference: str, hypothesis: str) -> float:
    import jiwer
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    return float(jiwer.wer(ref, hyp))


def cer(reference: str, hypothesis: str) -> float:
    import jiwer
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    return float(jiwer.cer(ref, hyp))


def pairwise_wer(named: dict[str, str]) -> dict[tuple[str, str], float]:
    """WER of every transcript against every other (ref=first, hyp=second).

    Needs no reference, so it measures genuine engine-to-engine divergence
    without the circularity of scoring engines against a consensus built from
    those same engines. The honest relative-ranking view.
    """
    ids = list(named)
    return {
        (a, b): wer(named[a], named[b])
        for a in ids for b in ids if a != b
    }


def words_by_speaker(words: list[Word]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for w in sorted(words, key=lambda x: x.start):
        out.setdefault(w.speaker or "UNKNOWN", []).append(w.text)
    return {spk: " ".join(toks) for spk, toks in out.items()}


def cpwer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]) -> float:
    from meeteval.wer import cp_word_error_rate
    ref = {k: normalize_text(v) for k, v in ref_by_speaker.items()}
    hyp = {k: normalize_text(v) for k, v in hyp_by_speaker.items()}
    if not any(ref.values()) and not any(hyp.values()):
        return 0.0
    return float(cp_word_error_rate(ref, hyp).error_rate)


def _annotation(segments: list[Segment]):
    from pyannote.core import Annotation, Segment as PSegment
    ann = Annotation()
    for s in segments:
        ann[PSegment(s.start, s.end)] = s.speaker
    return ann


def der(ref_segments: list[Segment], hyp_segments: list[Segment]) -> float:
    import warnings
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate()
    with warnings.catch_warnings():
        # pyannote approximates the evaluation region (uem) from the segment
        # extents when none is supplied; that approximation is exactly what we
        # want here, so silence the advisory to keep output pristine.
        warnings.filterwarnings("ignore", message=".*uem.*approximated.*")
        return float(metric(_annotation(ref_segments), _annotation(hyp_segments)))
