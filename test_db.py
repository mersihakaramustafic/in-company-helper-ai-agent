from dotenv import load_dotenv
load_dotenv()
import psycopg2
import os
from pgvector.psycopg2 import register_vector
from ingestion.embedder import embed_texts

conn = psycopg2.connect(os.environ["DATABASE_URL"])
register_vector(conn)
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
