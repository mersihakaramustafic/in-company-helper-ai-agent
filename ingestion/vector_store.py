import os
import uuid
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


def _connect():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Each chunk dict must have:
      page_id, page_title, source_url, connector_type,
      chunk_index, content, embedding, last_updated
    Deletes existing chunks for the given pages before inserting.
    """
    if not chunks:
        return

    conn = _connect()
    try:
        with conn.cursor() as cur:
            page_ids = list({c["page_id"] for c in chunks})
            cur.execute(
                "DELETE FROM documents WHERE page_id = ANY(%s)", (page_ids,)
            )

            rows = [
                (
                    str(uuid.uuid4()),
                    c["page_id"],
                    c["page_title"],
                    c["source_url"],
                    c["connector_type"],
                    c["chunk_index"],
                    c["content"],
                    c["embedding"],
                    c["last_updated"],
                )
                for c in chunks
            ]

            execute_values(
                cur,
                """
                INSERT INTO documents
                    (id, page_id, page_title, source_url, connector_type,
                     chunk_index, content, embedding, last_updated)
                VALUES %s
                """,
                rows,
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
            return row[0].isoformat() != last_updated.replace("Z", "+00:00")
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
