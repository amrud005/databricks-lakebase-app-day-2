# SQL setup for the weather vector pipeline

Run these in Lakebase **or** let `lakebase.ensure_weather_tables()` create them
automatically on first `/weather/sync` or `/weather/search`.

1. `01_setup_weather_documents.sql` — raw NWS documents
2. `02_setup_weather_embeddings.sql` — chunk embeddings + HNSW index (`vector(384)`)

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
