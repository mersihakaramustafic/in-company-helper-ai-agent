from dotenv import load_dotenv
load_dotenv()
from ingestion.embedder import embed_texts
from ingestion.vector_store import search

embedding = embed_texts(["what topics do we have in Notion?"])[0]
print("Embedding length:", len(embedding))
chunks = search(embedding, limit=5)
print("Chunks found:", len(chunks))
for c in chunks:
    print(" -", c.get("page_title"), c.get("score"))
