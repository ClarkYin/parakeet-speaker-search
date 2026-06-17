import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db
from app.storage import save_file, get_file_context, approve_context
from app.ingest import run_stage1, run_stage2

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_db():
    """Thin wrapper so that patching `app.routes.files.get_db` is picked up at call time."""
    yield from get_db()


class ContextApproval(BaseModel):
    context: Optional[str] = None


@router.post("/upload")
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default="groq/whisper-large-v3-turbo"),
    db: Session = Depends(_get_db),
):
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    file_id = save_file(db, filename=file.filename, model=model)
    background_tasks.add_task(run_stage1, db=db, file_id=file_id, file_path=file_path)

    return {"file_id": file_id, "status": "processing", "model": model}


@router.get("/{file_id}/status")
def get_file_status(file_id: str, db: Session = Depends(_get_db)):
    row = db.execute(
        text("SELECT id, status, speaker_count, duration_sec FROM files WHERE id = :id"),
        {"id": file_id},
    ).mappings().fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    return dict(row)


@router.get("/{file_id}/context")
def get_context(file_id: str, db: Session = Depends(_get_db)):
    row = get_file_context(db, file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    return row


@router.post("/{file_id}/context/approve")
def approve_file_context(
    file_id: str,
    background_tasks: BackgroundTasks,
    payload: Optional[ContextApproval] = None,
    db: Session = Depends(_get_db),
):
    row = get_file_context(db, file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    override = payload.context if payload is not None else None
    approved = override if override is not None else (row["context"] or "")
    affected = approve_context(db, file_id, approved)
    if affected == 0:
        raise HTTPException(status_code=409, detail="File is not awaiting approval")
    # read filename + model to rebuild the path and reuse the chosen model
    meta = db.execute(
        text("SELECT filename, model FROM files WHERE id = :id"),
        {"id": file_id},
    ).mappings().fetchone()
    file_path = os.path.join(UPLOAD_DIR, meta["filename"])
    model = meta["model"] or "groq/whisper-large-v3-turbo"
    background_tasks.add_task(run_stage2, db=db, file_id=file_id, file_path=file_path, model=model, context=approved)
    return {"file_id": file_id, "status": "processing"}


@router.get("/{file_id}/transcript")
def get_transcript(file_id: str, db: Session = Depends(_get_db)):
    rows = db.execute(
        text("""
            SELECT speaker_label, start_sec, end_sec, text
            FROM utterances WHERE file_id = :id ORDER BY start_sec
        """),
        {"id": file_id},
    ).mappings().fetchall()
    return [dict(r) for r in rows]


@router.get("/{file_id}/speakers")
def get_speakers(file_id: str, db: Session = Depends(_get_db)):
    rows = db.execute(
        text("""
            SELECT speaker_label,
                   SUM(end_sec - start_sec) AS total_sec,
                   COUNT(*) AS utterance_count
            FROM utterances WHERE file_id = :id
            GROUP BY speaker_label ORDER BY total_sec DESC
        """),
        {"id": file_id},
    ).mappings().fetchall()
    return [dict(r) for r in rows]
