from __future__ import annotations
import json
import os
from eval.models import EngineScore, TranscriptResult
from eval.align import align


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def ranked_table(scores: list[EngineScore]) -> str:
    ordered = sorted(scores, key=lambda s: (s.cpwer is None, s.cpwer if s.cpwer is not None else 0.0))
    lines = [
        "| engine | cpWER | WER | CER | DER | RTF | $est |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ordered:
        lines.append(
            f"| {s.engine_id} | {_fmt(s.cpwer)} | {_fmt(s.wer)} | {_fmt(s.cer)} | "
            f"{_fmt(s.der)} | {_fmt(s.rtf)} | {_fmt(s.cost_est)} |"
        )
    return "\n".join(lines)


def diff_html(reference: str, hypothesis: str) -> str:
    spans: list[str] = []
    for op in align(reference, hypothesis):
        if op["op"] == "equal":
            spans.append(f'<span class="equal">{op["hyp"]}</span>')
        elif op["op"] == "sub":
            spans.append(f'<span class="sub" title="ref: {op["ref"]}">{op["hyp"]}</span>')
        elif op["op"] == "ins":
            spans.append(f'<span class="ins">{op["hyp"]}</span>')
        elif op["op"] == "del":
            spans.append(f'<span class="del" title="ref: {op["ref"]}">∅</span>')
    return " ".join(spans)


_CSS = (
    ".equal{color:#222}.sub{background:#fde68a}.ins{background:#bbf7d0}"
    ".del{background:#fecaca;color:#991b1b}body{font-family:system-ui;max-width:900px;margin:2rem auto}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
)


def disagreement(results: list[TranscriptResult], reference: str) -> list[dict]:
    ref_tokens = reference.split()
    rows: list[dict] = []
    per_engine = {r.engine_id: r.text.split() for r in results}
    for i, tok in enumerate(ref_tokens):
        variants = {eid: (toks[i] if i < len(toks) else None) for eid, toks in per_engine.items()}
        if len(set(v for v in variants.values() if v is not None)) > 1:
            rows.append({"ref": tok, "variants": variants})
    return rows


def write_report(out_dir: str, scores: list[EngineScore], results: list[TranscriptResult], reference_text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    table = ranked_table(scores)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("# Evaluation report\n\n" + table + "\n")
    diffs = "".join(
        f"<h3>{r.engine_id}</h3><p>{diff_html(reference_text, r.text)}</p>" for r in results
    )
    html_table = table.replace("|", " ")  # minimal; md table not rendered, keep readable
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(f"<!doctype html><meta charset=utf-8><style>{_CSS}</style>"
                f"<h1>Evaluation report</h1><pre>{html_table}</pre>{diffs}")
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump([s.to_dict() for s in scores], f, indent=2)
