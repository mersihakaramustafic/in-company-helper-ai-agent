import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg_pool import AsyncConnectionPool

# Supabase's transaction pooler (port 6543) doesn't support prepared statements.
_CONN_KWARGS = {"prepare_threshold": None}


async def _configure_async(conn) -> None:
    await register_vector_async(conn)


# Used by the API. Opened/closed by the FastAPI lifespan in api.py.
async_pool = AsyncConnectionPool(
    os.environ["DATABASE_URL"],
    min_size=1,
    max_size=10,
    open=False,
    configure=_configure_async,
    kwargs=_CONN_KWARGS,
)


def _connect() -> psycopg.Connection:
    """Sync connection for the ingestion pipeline. Use as `with _connect() as conn:`,
    which commits on success and closes the connection."""
    conn = psycopg.connect(os.environ["DATABASE_URL"], **_CONN_KWARGS)
    register_vector(conn)
    return conn


def get_existing_chunk_hashes(page_id: str) -> set:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT content_hash FROM documents WHERE page_id = %s",
            (page_id,),
        ).fetchall()
        return {row[0] for row in rows if row[0]}


def sync_page_chunks(
    page_id: str,
    valid_hashes: List[str],
    new_rows: List[Dict[str, Any]],
    last_updated: str,
) -> None:
    """
    Brings stored chunks for a page in line with its current content.

    valid_hashes: content_hash of every chunk the page currently has.
      Any stored row whose hash isn't in this list is stale and gets deleted.
    new_rows: only the chunks that are new/changed and need inserting
      (must include embedding, content_hash, chunk_index, content, page_title,
      source_url, connector_type).
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM documents WHERE page_id = %s "
            "AND (content_hash IS NULL OR NOT (content_hash = ANY(%s::text[])))",
            (page_id, valid_hashes),
        )

        if new_rows:
            rows = [
                (
                    str(uuid.uuid4()),
                    c["page_id"],
                    c["page_title"],
                    c["source_url"],
                    c["connector_type"],
                    c["chunk_index"],
                    c["content"],
                    c["content_hash"],
                    c["embedding"],
                    c["last_updated"],
                )
                for c in new_rows
            ]

            cur.executemany(
                """
                INSERT INTO documents
                    (id, page_id, page_title, source_url, connector_type,
                     chunk_index, content, content_hash, embedding, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                """,
                rows,
            )

        cur.execute(
            "UPDATE documents SET last_updated = %s WHERE page_id = %s",
            (last_updated, page_id),
        )


def page_has_changed(page_id: str, last_updated: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT last_updated FROM documents WHERE page_id = %s LIMIT 1",
            (page_id,),
        ).fetchone()
        if not row:
            return True
        return row[0] != datetime.fromisoformat(last_updated.replace("Z", "+00:00"))


async def asearch(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Cosine similarity search over the shared connection pool. Returns chunks with metadata."""
    async with async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT page_title, source_url, connector_type, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in await cur.fetchall()]
