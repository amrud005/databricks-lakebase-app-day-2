## 1. Data source

I used the National Weather Service API (https://api.weather.gov).

Why:

- No API key, so the Databricks App does not need extra secrets for weather
- Alerts and forecasts both return real narrative text, which is what we embed
- Rate limits are fine for a small city list during demos

What I pull:

- Active alerts by state — description and instruction fields
- Multi-day forecast periods via grid point — detailedForecast text

Cities resolve through a fixed city to lat/lon map in weather_client.py. I did
not add a geocoding API so the work stays on harvest, embed, and search inside
Databricks and Lakebase.

## 2. Schema decisions

### weather_documents

- id — stable key; alerts use alert: plus the NWS id, forecasts use a hash of
  location, period start, and name
- location — canonical city string
- source_type — alert or forecast
- headline / event — short labels for the UI and search results
- narrative_text — free text that gets chunked and embedded
- issued_at / effective_at — timestamps from NWS when present
- payload — raw JSON kept for debugging
- synced_at — when the row was upserted

Sync upserts on id, so re-running sync from the App does not create duplicate
rows.

### weather_embeddings

- id — document_id, chunk marker, and chunk index
- document_id — foreign key to weather_documents.id
- chunk_index / chunk_text — sliding-window pieces of narrative_text
- embedding — vector with 384 dimensions
- model_name — which model produced the vector
- created_at — insert time

Chunking:

- CHUNK_SIZE = 800
- CHUNK_OVERLAP = 100

Most NWS text fits in one chunk. Overlap mainly helps when alert description
and instruction are joined into a longer blob.

Embedding model:

- sentence-transformers/all-MiniLM-L6-v2
- 384 dimensions (same idea as the original news pipeline)

Index:

- HNSW on embedding with cosine ops so search can use pgvector distance

The ingest script writes with psycopg2 and casts embeddings to vector. No Spark
JDBC against Lakebase.

## 3. Sync → embed → search 

1. Deploy the Databricks App from this workspace folder using app.yaml.
   Lakebase URL comes from secret database/lakebase-url.
2. Open the App URL, select cities, and sync.
   That calls POST /weather/sync and upserts into weather_documents.
3. In the workspace, run notebooks/ingest_weather_embeddings.py as a notebook
   or job. It reads unembedded documents from Lakebase, chunks, embeds, and
   upserts into weather_embeddings.
4. Back in the App, ask a question on the App URL.
   Results are ranked by vector similarity.


## 4. Limitations

- Only the fixed city list works; unknown cities return 400
- Alerts are fetched by state, so two cities in the same state can share alert
  docs (ids still dedupe them)
- Embedding is a separate workspace step after App sync; it is not automatic on
  /weather/sync
- First search in the App loads the model and is slower; later searches are fine
- No scheduled Workflow yet to re-sync alerts on a timer
- No LLM summary on top of the retrieved chunks
- Search returns source_type but does not filter by it yet
