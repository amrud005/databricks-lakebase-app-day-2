"""
Ingest weather document embeddings into Lakebase via psycopg2.

Reads rows from weather_documents, chunks narrative_text, embeds with
sentence-transformers/all-MiniLM-L6-v2 (384-dim), and upserts into
weather_embeddings using execute_values + %s::vector casts.

Run (local, with LAKEBASE_URL in .env):
    python notebooks/ingest_weather_embeddings.py

Or as a Databricks notebook / job — uses lakebase.get_connection()
(env LAKEBASE_URL or secret database/lakebase-url).
"""

from __future__ import annotations

import logging
import os
import sys

# Allow importing lakebase when run as a script from repo root or notebooks/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

import lakebase

load_dotenv(os.path.join(_ROOT, ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-weather-embeddings")

DOCUMENTS_TABLE = lakebase.WEATHER_DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.WEATHER_EMBEDDINGS_TABLE
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunks. Short NWS text usually yields a single chunk."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def fetch_documents() -> list[dict]:
    return lakebase.run_query(
        f"""
        SELECT id, narrative_text
        FROM {DOCUMENTS_TABLE}
        WHERE narrative_text IS NOT NULL
          AND TRIM(narrative_text) <> ''
        ORDER BY synced_at DESC
        """
    )


def already_embedded_document_ids() -> set[str]:
    rows = lakebase.run_query(
        f"SELECT DISTINCT document_id FROM {EMBEDDINGS_TABLE}"
    )
    return {r["document_id"] for r in rows}


def build_chunk_rows(docs: list[dict], skip_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for doc in docs:
        doc_id = doc["id"]
        if doc_id in skip_ids:
            continue
        for idx, chunk in enumerate(chunk_text(doc["narrative_text"])):
            rows.append(
                {
                    "id": f"{doc_id}::chunk::{idx}",
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                }
            )
    return rows


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts = [c["chunk_text"] for c in chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = model.encode(batch, show_progress_bar=False)
        vectors.extend(v.tolist() for v in encoded)
        logger.info("Embedded %s / %s chunks", min(i + BATCH_SIZE, len(texts)), len(texts))

    out = []
    for chunk, vector in zip(chunks, vectors):
        out.append({**chunk, "embedding": vector, "model_name": EMBEDDING_MODEL})
    return out


def upsert_embeddings(rows: list[dict]) -> int:
    if not rows:
        return 0

    insert_data = [
        (
            r["id"],
            r["document_id"],
            r["chunk_index"],
            r["chunk_text"],
            "[" + ",".join(str(float(x)) for x in r["embedding"]) + "]",
            r["model_name"],
        )
        for r in rows
    ]

    sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = now()
    """
    template = "(%s, %s, %s, %s, %s::vector, %s, now())"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, insert_data, template=template, page_size=100)
            conn.commit()
            return len(insert_data)


def main() -> None:
    logger.info("Ensuring weather tables exist...")
    lakebase.ensure_weather_tables()

    docs = fetch_documents()
    logger.info("Loaded %s documents from %s", len(docs), DOCUMENTS_TABLE)
    if not docs:
        logger.warning("No documents to embed. Run POST /weather/sync first.")
        return

    skip = already_embedded_document_ids()
    logger.info("%s documents already have embeddings (will skip)", len(skip))

    chunks = build_chunk_rows(docs, skip)
    logger.info("%s new chunks to embed (chunk_size=%s, overlap=%s)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        logger.info("Nothing new to embed.")
        return

    cache = os.environ.get("HF_HOME", "/tmp/.cache/huggingface")
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache)
    logger.info("Loading model %s ...", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=cache)

    embedded = embed_chunks(chunks, model)
    written = upsert_embeddings(embedded)
    logger.info("Wrote %s embeddings into %s", written, EMBEDDINGS_TABLE)


if __name__ == "__main__":
    main()
