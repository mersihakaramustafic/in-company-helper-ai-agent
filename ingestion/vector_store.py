import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


def _connect():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def get_existing_chunk_hashes(page_id: str) -> set:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM documents WHERE page_id = %s",
                (page_id,),
            )
            return {row[0] for row in cur.fetchall() if row[0]}
    finally:
        conn.close()


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
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE page_id = %s "
                "AND (content_hash IS NULL OR NOT (content_hash = ANY(%s)))",
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

                execute_values(
                    cur,
                    """
                    INSERT INTO documents
                        (id, page_id, page_title, source_url, connector_type,
                         chunk_index, content, content_hash, embedding, last_updated)
                    VALUES %s
                    """,
                    rows,
                )

            cur.execute(
                "UPDATE documents SET last_updated = %s WHERE page_id = %s",
                (last_updated, page_id),
            )
        conn.commit()
    finally:
        conn.close()


def page_has_changed(page_id: str, last_updated: str) -> bool:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_updated FROM documents WHERE page_id = %s LIMIT 1",
                (page_id,),
            )
            row = cur.fetchone()
            if not row:
                return True
            return row[0] != datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
    finally:
        conn.close()


def search(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Cosine similarity search. Returns chunks with metadata."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
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
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
