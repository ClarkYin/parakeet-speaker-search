from sqlalchemy import text
from app.storage import _embed

def keyword_search(
    db,
    query: str,
    file_id: str = None,
    speaker_label: str = None,
    limit: int = 50,
) -> list:
    sql = """
        SELECT id, file_id, speaker_label, start_sec, end_sec, text,
               ts_rank(text_tsv, plainto_tsquery('english', :query)) AS score
        FROM utterances
        WHERE text_tsv @@ plainto_tsquery('english', :query)
    """
    params = {"query": query}
    if file_id:
        sql += " AND file_id = :file_id"
        params["file_id"] = file_id
    if speaker_label:
        sql += " AND speaker_label = :speaker_label"
        params["speaker_label"] = speaker_label
    sql += " ORDER BY score DESC LIMIT :limit"
    params["limit"] = limit
    return db.execute(text(sql), params).mappings().fetchall()

def semantic_search(
    db,
    query: str,
    file_id: str = None,
    speaker_label: str = None,
    limit: int = 20,
) -> list:
    embedding = _embed([query])[0]
    emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
    sql = """
        SELECT id, file_id, speaker_label, start_sec, end_sec, text,
               1 - (embedding <=> :emb::vector) AS score
        FROM utterances
        WHERE embedding IS NOT NULL
    """
    params = {"emb": emb_str}
    if file_id:
        sql += " AND file_id = :file_id"
        params["file_id"] = file_id
    if speaker_label:
        sql += " AND speaker_label = :speaker_label"
        params["speaker_label"] = speaker_label
    sql += " ORDER BY score DESC LIMIT :limit"
    params["limit"] = limit
    return db.execute(text(sql), params).mappings().fetchall()

def combined_search(db, query: str, file_id=None, speaker_label=None) -> list:
    kw = keyword_search(db, query, file_id, speaker_label, limit=30)
    sem = semantic_search(db, query, file_id, speaker_label, limit=20)
    seen = set()
    merged = []
    for row in list(kw) + list(sem):
        uid = row["id"]
        if uid not in seen:
            seen.add(uid)
            merged.append(dict(row))
    return merged

def speaker_transcript(db, file_id: str, speaker_label: str) -> list:
    sql = """
        SELECT id, file_id, speaker_label, start_sec, end_sec, text
        FROM utterances
        WHERE file_id = :file_id AND speaker_label = :speaker_label
        ORDER BY start_sec
    """
    return db.execute(
        text(sql), {"file_id": file_id, "speaker_label": speaker_label}
    ).mappings().fetchall()
