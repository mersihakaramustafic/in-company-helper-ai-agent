import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langfuse import get_client, observe, propagate_attributes
from schemas.chat import ChatRequest, FeedbackRequest
from agent.graph import agent_config, graph, initial_state
from agent.prompts import get_answer_prompt
from ingestion.vector_store import async_pool

langfuse = get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await async_pool.open()
    get_answer_prompt()  # warm the prompt cache so the first request doesn't block on a fetch
    yield
    await async_pool.close()
    langfuse.flush()


app = FastAPI(lifespan=lifespan)


# Root span of the trace; retrieve/generate spans and the OpenAI generations nest under it.
@observe(name="chat", as_type="agent")
async def _stream_chat(message: str):
    with propagate_attributes(trace_name="chat", metadata=agent_config(), tags=["chat"]):
        trace_id = langfuse.get_current_trace_id()
        async for mode, payload in graph.astream(initial_state(message), stream_mode=["custom", "values"]):
            if mode == "custom":
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"
            elif mode == "values" and payload["answer"]:
                done = {"sources": payload["sources"], "error": payload.get("error"), "trace_id": trace_id}
                yield f"event: done\ndata: {json.dumps(done)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(_stream_chat(request.message), media_type="text/event-stream")


@app.post("/feedback", status_code=204)
def feedback(request: FeedbackRequest):
    langfuse.create_score(
        name="user_feedback",
        value=1 if request.helpful else 0,
        data_type="BOOLEAN",
        trace_id=request.trace_id,
        comment=request.comment,
    )
