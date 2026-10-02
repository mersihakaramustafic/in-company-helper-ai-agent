import asyncio
from dotenv import load_dotenv
load_dotenv()
from ingestion.embedder import aembed_texts
from ingestion.vector_store import async_pool, asearch


async def main():
    async with async_pool:
        embedding = (await aembed_texts(["what topics do we have in Notion?"]))[0]
        print("Embedding length:", len(embedding))
        chunks = await asearch(embedding, limit=5)
        print("Chunks found:", len(chunks))
        for c in chunks:
            print(" -", c.get("page_title"), c.get("score"))


asyncio.run(main())
