from __future__ import annotations
from eval.normalize import normalize_text


def align(reference: str, hypothesis: str) -> list[dict]:
    import jiwer
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    out = jiwer.process_words(" ".join(ref) or " ", " ".join(hyp) or " ")
    ops: list[dict] = []
    # process_words returns alignment per sentence; we passed one "sentence"
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for r in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ops.append({"op": "equal", "ref": ref[r], "hyp": hyp[r - chunk.ref_start_idx + chunk.hyp_start_idx]})
        elif chunk.type == "substitute":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ops.append({"op": "sub", "ref": ref[chunk.ref_start_idx + k], "hyp": hyp[chunk.hyp_start_idx + k]})
        elif chunk.type == "insert":
            for k in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                ops.append({"op": "ins", "ref": None, "hyp": hyp[k]})
        elif chunk.type == "delete":
            for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ops.append({"op": "del", "ref": ref[k], "hyp": None})
    return ops
