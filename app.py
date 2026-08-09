"""
Weather Intelligence Databricks App:
- Harvests NWS alerts + forecast narratives into Lakebase
- Serves semantic search over weather_embeddings (pgvector)
- Simple UI to ask weather questions

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

from __future__ import annotations

import json
import logging
import os

from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient, list_known_locations, resolve_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

DOCUMENTS_TABLE = lakebase.WEATHER_DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.WEATHER_EMBEDDINGS_TABLE
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once (module-level singleton), not per request."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL_NAME)
        cache = os.environ.get("HF_HOME", "/tmp/.cache/huggingface")
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache)
        logger.info("Embedding model ready")
    return _embedding_model


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure unhandled errors return JSON so the UI's resp.json() never chokes."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """UI: ask weather questions + optionally trigger a sync."""
    return render_template(
        "index.html",
        known_locations=list_known_locations(),
    )


@app.route("/weather/locations")
def weather_locations():
    """List cities supported by the fixed city→lat/lon map."""
    return jsonify({"locations": list_known_locations()})


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """
    Harvest NWS alerts + forecasts for the given locations and upsert into
    weather_documents.

    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    """
    lakebase.ensure_weather_tables()

    body = request.json if request.is_json else {}
    locations = body.get("locations")
    if not locations or not isinstance(locations, list):
        return jsonify(
            {
                "error": "Request body must include a non-empty 'locations' list "
                f"(known: {list_known_locations()})"
            }
        ), 400

    cleaned: list[str] = []
    unknown: list[str] = []
    for loc in locations:
        if not isinstance(loc, str) or not loc.strip():
            continue
        try:
            canonical, _, _, _ = resolve_location(loc)
            cleaned.append(canonical)
        except ValueError:
            unknown.append(loc)

    if not cleaned:
        return jsonify(
            {
                "error": "No valid locations provided",
                "unknown": unknown,
                "known": list_known_locations(),
            }
        ), 400

    limit = int(body.get("limit", 50))
    limit = max(1, min(limit, 200))

    client = WeatherClient()
    docs = client.harvest_locations(cleaned, limit=limit)
    synced = _upsert_weather_documents(docs)

    return jsonify(
        {
            "synced": synced,
            "locations": cleaned,
            "unknown": unknown,
            "document_ids": [d["id"] for d in docs],
        }
    )


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    Semantic search over weather_embeddings.

    Body: {"query": "flash flood risk this weekend", "top_k": 5}
    """
    lakebase.ensure_weather_tables()

    body = request.json if request.is_json else {}
    query = body.get("query") if isinstance(body, dict) else None
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing or empty 'query' string"}), 400
    query = query.strip()

    top_k = body.get("top_k", 5)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "'top_k' must be an integer"}), 400
    top_k = max(1, min(top_k, 20))

    # Empty table edge case
    count_rows = lakebase.run_query(
        f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE}"
    )
    if not count_rows or int(count_rows[0]["n"]) == 0:
        return jsonify(
            {
                "query": query,
                "top_k": top_k,
                "results": [],
                "message": "No weather data is ready to search yet. "
                "Refresh cities first, then try again.",
            }
        )

    model = get_embedding_model()
    vector = model.encode(query).tolist()
    vector_literal = "[" + ",".join(str(float(x)) for x in vector) + "]"

    rows = lakebase.run_query(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.event,
            d.narrative_text,
            e.chunk_index,
            e.chunk_text,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (vector_literal, vector_literal, top_k),
    )

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "location": row["location"],
                "source_type": row["source_type"],
                "headline": row["headline"],
                "event": row["event"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "similarity": float(row["similarity"]) if row["similarity"] is not None else None,
            }
        )

    return jsonify({"query": query, "top_k": top_k, "results": results})


def _upsert_weather_documents(docs: list[dict]) -> int:
    """Upsert normalized weather documents into Lakebase."""
    if not docs:
        return 0

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in docs:
                cur.execute(
                    f"""
                    INSERT INTO {DOCUMENTS_TABLE} (
                        id, location, source_type, headline, event,
                        narrative_text, issued_at, effective_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        location = EXCLUDED.location,
                        source_type = EXCLUDED.source_type,
                        headline = EXCLUDED.headline,
                        event = EXCLUDED.event,
                        narrative_text = EXCLUDED.narrative_text,
                        issued_at = EXCLUDED.issued_at,
                        effective_at = EXCLUDED.effective_at,
                        payload = EXCLUDED.payload,
                        synced_at = EXCLUDED.synced_at
                    """,
                    (
                        doc["id"],
                        doc["location"],
                        doc["source_type"],
                        doc.get("headline"),
                        doc.get("event"),
                        doc["narrative_text"],
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        json.dumps(doc.get("payload") or {}),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)
