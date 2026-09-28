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
from ingestion.chunker import chunk_text, hash_chunk
from ingestion.embedder import embed_texts
from ingestion.vector_store import (
    page_has_changed,
    get_existing_chunk_hashes,
    sync_page_chunks,
)

EMBED_BATCH_SIZE = 50


def run() -> None:
    print("Fetching Notion pages...")
    pages = get_all_pages()
    print(f"Found {len(pages)} pages\n")

    to_embed = []  # chunks (dicts) that need a fresh embedding
    page_sync_info = {}  # page_id -> {valid_hashes, rows, last_updated}
    skipped = 0

    for i, page in enumerate(pages, 1):
        page_id = page["id"]
        last_updated = get_page_last_updated(page)

        if not page_has_changed(page_id, last_updated):
            skipped += 1
            continue

        title = get_page_title(page)
        url = get_page_url(page)
        print(f"[{i}/{len(pages)}] {title}")

        try:
            content = get_page_content(page_id)
        except Exception as exc:
            print(f"  Skipped — error fetching content: {exc}")
            continue

        chunks = chunk_text(content)
        existing_hashes = get_existing_chunk_hashes(page_id)

        rows = []
        new_count = 0
        for idx, chunk in enumerate(chunks):
            content_hash = hash_chunk(chunk)
            row = {
                "page_id": page_id,
                "page_title": title,
                "source_url": url,
                "connector_type": "notion",
                "chunk_index": idx,
                "content": chunk,
                "content_hash": content_hash,
                "last_updated": last_updated,
            }
            rows.append(row)
            if content_hash not in existing_hashes:
                to_embed.append(row)
                new_count += 1

        page_sync_info[page_id] = {
            "valid_hashes": [r["content_hash"] for r in rows],
            "rows": rows,
            "last_updated": last_updated,
        }
        print(f"  {len(chunks)} chunks, {new_count} changed")

    if skipped:
        print(f"\nSkipped {skipped} unchanged page(s).")

    if not page_sync_info:
        print("\nNothing to sync.")
        return

    if to_embed:
        print(f"\nEmbedding {len(to_embed)} changed chunk(s)...")
        for start in range(0, len(to_embed), EMBED_BATCH_SIZE):
            batch = to_embed[start : start + EMBED_BATCH_SIZE]
            embeddings = embed_texts([c["content"] for c in batch])
            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb
            end = min(start + EMBED_BATCH_SIZE, len(to_embed))
            print(f"  Embedded {end}/{len(to_embed)}")

    print(f"\nSyncing {len(page_sync_info)} page(s) to the database...")
    for page_id, info in page_sync_info.items():
        new_rows = [r for r in info["rows"] if "embedding" in r]
        sync_page_chunks(page_id, info["valid_hashes"], new_rows, info["last_updated"])

    print("\nIngestion complete.")


if __name__ == "__main__":
    run()
