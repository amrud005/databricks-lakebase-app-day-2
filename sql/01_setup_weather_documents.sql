-- weather_documents: raw NWS alerts + forecast narratives
-- Run manually in Lakebase if you prefer SQL over lakebase.ensure_weather_tables().

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    source_type TEXT NOT NULL,
    headline TEXT,
    event TEXT,
    narrative_text TEXT NOT NULL,
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
ON weather_documents (source_type);

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
