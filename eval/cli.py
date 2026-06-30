from __future__ import annotations
import argparse
import os
import sys
from dataclasses import replace
from eval.models import TranscriptResult, Segment, EngineScore, Reference
from eval.audio import normalize, detect_silences
from eval.engines.base import get_registry, available_engines
from eval.metrics import wer, cer, cpwer, der, words_by_speaker
from eval.reference import sample_windows, build_reference, save_reference, load_reference
from eval.report import write_report


def _load_engines() -> None:
    for mod in [
        "eval.engines.parakeet_mlx", "eval.engines.groq_whisper", "eval.engines.deepgram",
        "eval.engines.assemblyai", "eval.engines.elevenlabs",
        "eval.engines.openai_whisper", "eval.engines.gemini",
    ]:
        try:
            __import__(mod)
        except ImportError:
            pass


def attribute_speakers(result: TranscriptResult, segments: list[Segment]) -> TranscriptResult:
    from app.merger import _find_speaker
    seg_dicts = [{"speaker": s.speaker, "start": s.start, "end": s.end} for s in segments]
    words = [replace(w, speaker=_find_speaker((w.start + w.end) / 2, seg_dicts)) for w in result.words]
    spk_segments: list[Segment] = []
    for w in words:
        if spk_segments and spk_segments[-1].speaker == w.speaker:
            spk_segments[-1].end = w.end
        else:
            spk_segments.append(Segment(w.speaker or "UNKNOWN", w.start, w.end))
    return replace(result, words=words, speakers=spk_segments)


def _corrected_windows(reference: Reference):
    return [w for w in reference.windows if w.corrected] or reference.windows


def score_engine(result: TranscriptResult, reference: Reference) -> EngineScore:
    windows = _corrected_windows(reference)
    ref_text, hyp_text = [], []
    ref_words, hyp_words = [], []
    for win in windows:
        ref_text.append(" ".join(w.text for w in win.words))
        ref_words.extend(win.words)
        in_win = [w for w in result.words if win.start <= (w.start + w.end) / 2 <= win.end]
        hyp_text.append(" ".join(w.text for w in in_win))
        hyp_words.extend(in_win)
    ref_joined, hyp_joined = " ".join(ref_text), " ".join(hyp_text)
    der_val = None
    if result.speakers is not None:
        ref_segs = _segments_from_words(ref_words)
        hyp_segs = _segments_from_words(hyp_words)
        if ref_segs and hyp_segs:
            der_val = der(ref_segs, hyp_segs)
    return EngineScore(
        engine_id=result.engine_id,
        wer=wer(ref_joined, hyp_joined),
        cer=cer(ref_joined, hyp_joined),
        cpwer=cpwer(words_by_speaker(ref_words), words_by_speaker(hyp_words)),
        der=der_val,
        rtf=result.meta.get("rtf"),
        cost_est=result.meta.get("cost_est"),
    )


def _segments_from_words(words) -> list[Segment]:
    segs: list[Segment] = []
    for w in sorted(words, key=lambda x: x.start):
        spk = w.speaker or "UNKNOWN"
        if segs and segs[-1].speaker == spk:
            segs[-1].end = w.end
        else:
            segs.append(Segment(spk, w.start, w.end))
    return segs


def run(audio_path: str, engine_ids: list[str], work_dir: str, apple_transcript: str | None = None):
    os.makedirs(work_dir, exist_ok=True)
    audio = normalize(audio_path, work_dir)
    diar = diarize(audio.wav_path)
    segments = [Segment(d["speaker"], d["start"], d["end"]) for d in diar]

    registry = get_registry()
    results: list[TranscriptResult] = []
    for eid in engine_ids:
        engine = registry[eid]
        res = engine.transcribe(audio, work_dir)
        if not engine.diarizes:
            res = attribute_speakers(res, segments)
        results.append(res)

    if apple_transcript:
        from eval.engines.apple_voicememos import load as load_apple
        results.append(load_apple(apple_transcript))

    ref_path = os.path.join(work_dir, "reference.json")
    if os.path.exists(ref_path):
        reference = load_reference(ref_path)
    else:
        windows = sample_windows(audio.duration, detect_silences(audio.wav_path))
        reference = build_reference(os.path.basename(audio_path), results, windows)
        save_reference(reference, ref_path)

    scores = [score_engine(r, reference) for r in results]
    write_report(work_dir, scores, results, _reference_text(reference))
    return scores, results


def diarize(wav_path: str):
    from app.diarization import diarize as _d
    return _d(wav_path)


def _reference_text(reference: Reference) -> str:
    return " ".join(" ".join(w.text for w in win.words) for win in reference.windows)


def main(argv: list[str] | None = None) -> int:
    _load_engines()
    parser = argparse.ArgumentParser(prog="eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("audio")
    p_run.add_argument("--engines", default="")
    p_run.add_argument("--work-dir", default="runs/latest")
    p_run.add_argument("--apple-transcript", default=None)

    p_ref = sub.add_parser("reference")
    p_ref.add_argument("action", choices=["build", "edit", "status"])
    p_ref.add_argument("--work-dir", default="runs/latest")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        ids = [e.strip() for e in args.engines.split(",") if e.strip()] or [e.id for e in available_engines()]
        scores, _ = run(args.audio, ids, args.work_dir, args.apple_transcript)
        print(f"Scored {len(scores)} engines → {args.work_dir}/report.md")
        return 0
    if args.cmd == "reference":
        ref_path = os.path.join(args.work_dir, "reference.json")
        ref = load_reference(ref_path)
        corrected = sum(1 for w in ref.windows if w.corrected)
        print(f"{corrected}/{len(ref.windows)} windows corrected in {ref_path}")
        return 0
    return 1
