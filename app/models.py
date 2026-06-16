from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class FileRecord:
    id: str
    filename: str
    status: str
    duration_sec: Optional[float] = None
    speaker_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None

@dataclass
class Utterance:
    id: str
    file_id: str
    speaker_label: str
    start_sec: float
    end_sec: float
    text: str
    embedding: Optional[list[float]] = None  # 384-dim vector, DB-only for text_tsv
