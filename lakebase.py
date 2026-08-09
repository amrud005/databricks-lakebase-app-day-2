"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password.

Resolution order:
1. LAKEBASE_URL env var (local / .env)
2. Databricks secret scope (database/lakebase-url by default)
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

WEATHER_DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
WEATHER_EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))


def _lakebase_url() -> str:
    """Resolve the Lakebase connection URL from env or Databricks secrets."""
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url

    w = WorkspaceClient()
    secret = w.secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def ensure_weather_tables(embedding_dim: int | None = None) -> None:
    """
    Create weather_documents + weather_embeddings (pgvector) if missing.
    Safe to call on every sync/search request.
    """
    dim = embedding_dim or EMBEDDING_DIM
    docs = WEATHER_DOCUMENTS_TABLE
    emb = WEATHER_EMBEDDINGS_TABLE

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {docs} (
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
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{docs}_location ON {docs} (location)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{docs}_source_type ON {docs} (source_type)"
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {emb} (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES {docs}(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (document_id, chunk_index)
                )
                """
            )
            # HNSW for cosine similarity (<=>)
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{emb}_embedding_hnsw
                ON {emb}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.commit()
