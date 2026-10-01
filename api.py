import json

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest
from agent.graph import graph

app = FastAPI()


async def _stream_chat(message: str):
    inputs = {"query": message, "chunks": [], "answer": "", "error": None}
    async for mode, payload in graph.astream(inputs, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield f"event: token\ndata: {json.dumps(payload)}\n\n"
        elif mode == "values" and payload["answer"]:
            sources = [
                {"title": c["page_title"], "url": c["source_url"], "score": c["score"]}
                for c in payload["chunks"]
            ]
            done = {"sources": sources, "error": payload.get("error")}
            yield f"event: done\ndata: {json.dumps(done)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(_stream_chat(request.message), media_type="text/event-stream")