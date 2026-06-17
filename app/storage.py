from sqlalchemy import text
from typing import Optional

_model = None

def _embed(texts: list) -> list:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(texts).tolist()

def save_file(db, filename: str, duration_sec: Optional[float] = None) -> str:
    row = db.execute(
        text("""
            INSERT INTO files (filename, duration_sec)
            VALUES (:filename, :duration_sec)
            RETURNING id
        """),
        {"filename": filename, "duration_sec": duration_sec},
    ).fetchone()
    db.commit()
    return str(row[0])

def update_file_status(
    db,
    file_id: str,
    status: str,
    speaker_count: Optional[int] = None,
    error_message: Optional[str] = None,
):
    db.execute(
        text("""
            UPDATE files
            SET status = :status,
                speaker_count = :speaker_count,
                error_message = :error_message
            WHERE id = :file_id
        """),
        {"status": status, "speaker_count": speaker_count,
         "error_message": error_message, "file_id": file_id},
    )
    db.commit()

def set_file_context(db, file_id: str, context: str) -> None:
    """Write the context column for a file."""
    db.execute(
        text("UPDATE files SET context = :context WHERE id = :file_id"),
        {"context": context, "file_id": file_id},
    )
    db.commit()


def get_file_context(db, file_id: str) -> Optional[dict]:
    """Return the status and context for a file, or None if not found."""
    row = db.execute(
        text("SELECT status, context FROM files WHERE id = :file_id"),
        {"file_id": file_id},
    ).mappings().fetchone()
    if row is None:
        return None
    return {"status": row["status"], "context": row["context"]}


def approve_context(db, file_id: str, context: str) -> int:
    """Atomically flip status to processing and set context; returns rows affected."""
    result = db.execute(
        text(
            "UPDATE files SET status='processing', context=:ctx"
            " WHERE id=:id AND status='awaiting_approval'"
        ),
        {"ctx": context, "id": file_id},
    )
    db.commit()
    return result.rowcount


def save_utterances(db, file_id: str, utterances: list):
    if not utterances:
        return
    texts = [u["text"] for u in utterances]
    embeddings = _embed(texts)
    for u, emb in zip(utterances, embeddings):
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"
        db.execute(
            text(f"""
                INSERT INTO utterances
                    (file_id, speaker_label, start_sec, end_sec, text, embedding)
                VALUES
                    (:file_id, :speaker_label, :start_sec, :end_sec, :text, '{emb_str}'::vector)
            """),
            {
                "file_id": file_id,
                "speaker_label": u["speaker_label"],
                "start_sec": u["start_sec"],
                "end_sec": u["end_sec"],
                "text": u["text"],
            },
        )
    db.commit()
