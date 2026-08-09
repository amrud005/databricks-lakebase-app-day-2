## Weather Intelligence (Lakebase + pgvector)

Databricks App that pulls free-text weather alerts and forecasts from the
National Weather Service, stores them in Lakebase (Postgres), embeds the text
with sentence-transformers, and serves semantic search through Flask plus a
simple UI.

## What it does

- Harvests NWS alerts and forecast narratives for a fixed list of cities
- Upserts documents into weather_documents in Lakebase
- Chunks and embeds narrative text into weather_embeddings (384-dim vectors)
- Serves semantic search with pgvector cosine distance

## Databricks setup

1. Sync this repo folder into your Databricks workspace
2. Store the Lakebase connection URL as secret database/lakebase-url
3. Create / open a Databricks App pointed at this folder 
4. Deploy the App from the Apps UI

## How to use it on Databricks

1. Open the deployed App URL
2. Pick cities and sync (writes into weather_documents)
3. In a workspace notebook or job, run notebooks/ingest_weather_embeddings.py
   so rows land in weather_embeddings
4. Back in the App UI, ask a weather question

## API (on the App URL)

- GET /healthz — health check
- GET / — ask-questions UI
- GET /weather/locations — cities the client knows about
- POST /weather/sync — JSON body with locations list and optional limit
- POST /weather/search — JSON body with query and optional top_k

Example sync body: locations like Chicago, IL and Austin, TX, limit 50.

Example search body: query like flash flood risk this weekend, top_k 5.

## Lakebase tables

- weather_documents — raw alerts and forecasts
- weather_embeddings — chunk text plus 384-dim vectors, HNSW index

Tables are created on first sync/search via lakebase.ensure_weather_tables(),
or you can run the SQL under sql/ in Lakebase.

## Main files

- weather_client.py — NWS client and document normalization
- lakebase.py — connection helper and DDL
- app.py — Flask routes and UI
- app.yaml — Databricks App command and env
- notebooks/ingest_weather_embeddings.py — chunk, embed, upsert
- templates/index.html — search UI
