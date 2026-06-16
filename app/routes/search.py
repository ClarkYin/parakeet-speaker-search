from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.search import keyword_search, semantic_search, combined_search, speaker_transcript

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    mode: str = "both"
    file_id: str = None
    speaker_label: str = None

@router.post("/search")
def search(req: SearchRequest, db: Session = Depends(get_db)):
    if req.mode == "keyword":
        results = keyword_search(db, req.query, req.file_id, req.speaker_label)
    elif req.mode == "semantic":
        results = semantic_search(db, req.query, req.file_id, req.speaker_label)
    elif req.mode == "both":
        results = combined_search(db, req.query, req.file_id, req.speaker_label)
    else:
        raise HTTPException(status_code=400, detail="mode must be keyword, semantic, or both")
    return [dict(r) for r in results]

@router.get("/speakers/{speaker_label}/transcript")
def get_speaker_transcript(
    speaker_label: str,
    file_id: str,
    db: Session = Depends(get_db),
):
    rows = speaker_transcript(db, file_id=file_id, speaker_label=speaker_label)
    return [dict(r) for r in rows]
