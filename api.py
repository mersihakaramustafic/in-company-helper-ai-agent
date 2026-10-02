import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest
from agent.graph import graph
from ingestion.vector_store import async_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await async_pool.open()
    yield
    await async_pool.close()


app = FastAPI(lifespan=lifespan)


async def _stream_chat(message: str):
    inputs = {"query": message, "chunks": [], "answer": "", "sources": [], "error": None}
    async for mode, payload in graph.astream(inputs, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield f"event: token\ndata: {json.dumps(payload)}\n\n"
        elif mode == "values" and payload["answer"]:
            done = {"sources": payload["sources"], "error": payload.get("error")}
            yield f"event: done\ndata: {json.dumps(done)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(_stream_chat(request.message), media_type="text/event-stream")