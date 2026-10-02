from dotenv import load_dotenv
load_dotenv()
from ingestion.embedder import embed_texts
from ingestion.vector_store import _connect

conn = _connect()
cur = conn.cursor()

# Force sequential scan (bypass ivfflat index)
cur.execute("SET enable_indexscan = off")

embedding = embed_texts(["vacation policy"])[0]
cur.execute(
    "SELECT page_title, 1 - (embedding <=> %s::vector) AS score FROM documents ORDER BY embedding <=> %s::vector LIMIT 5",
    (embedding, embedding)
)
for row in cur.fetchall():
    print(row)

conn.close()
