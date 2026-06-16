CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS files (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    duration_sec FLOAT,
    speaker_count INTEGER,
    status      TEXT NOT NULL DEFAULT 'processing',
    error_message TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS utterances (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id      UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    speaker_label TEXT NOT NULL,
    start_sec    FLOAT NOT NULL,
    end_sec      FLOAT NOT NULL,
    text         TEXT NOT NULL,
    text_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding    VECTOR(384)
);

CREATE INDEX IF NOT EXISTS utterances_file_id_idx ON utterances(file_id);
CREATE INDEX IF NOT EXISTS utterances_speaker_idx ON utterances(speaker_label);
CREATE INDEX IF NOT EXISTS utterances_fts_idx ON utterances USING GIN(text_tsv);
CREATE INDEX IF NOT EXISTS utterances_embedding_idx
    ON utterances USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
