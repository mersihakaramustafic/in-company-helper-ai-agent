"""
Run ingestion: python -m ingestion.pipeline
Syncs all Notion pages → chunks → embeddings → pgvector.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from ingestion.notion_connector import (
    get_all_pages,
    get_page_title,
    get_page_url,
    get_page_content,
    get_page_last_updated,
)
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_texts
from ingestion.vector_store import upsert_chunks

EMBED_BATCH_SIZE = 50


def run() -> None:
    print("Fetching Notion pages...")
    pages = get_all_pages()
    print(f"Found {len(pages)} pages\n")

    all_chunks = []

    for i, page in enumerate(pages, 1):
        page_id = page["id"]
        title = get_page_title(page)
        url = get_page_url(page)
        last_updated = get_page_last_updated(page)

        print(f"[{i}/{len(pages)}] {title}")

        try:
            content = get_page_content(page_id)
        except Exception as exc:
            print(f"  Skipped — error fetching content: {exc}")
            continue

        if not content.strip():
            print("  Skipped — empty")
            continue

        chunks = chunk_text(content)
        print(f"  {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "page_id": page_id,
                    "page_title": title,
                    "source_url": url,
                    "connector_type": "notion",
                    "chunk_index": idx,
                    "content": chunk,
                    "last_updated": last_updated,
                }
            )

    if not all_chunks:
        print("\nNo chunks to store.")
        return

    print(f"\nEmbedding and storing {len(all_chunks)} chunks...")

    for start in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[start : start + EMBED_BATCH_SIZE]
        embeddings = embed_texts([c["content"] for c in batch])
        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb
        upsert_chunks(batch)
        end = min(start + EMBED_BATCH_SIZE, len(all_chunks))
        print(f"  Stored {end}/{len(all_chunks)}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    run()
