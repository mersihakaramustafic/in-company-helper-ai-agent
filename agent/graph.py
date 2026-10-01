import asyncio
import os
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from ingestion.embedder import embed_texts
from ingestion.vector_store import search

TOP_K = 5
MODEL = "gpt-4o-mini"

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class AgentState(TypedDict):
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    error: Optional[str]


async def retrieve(state: AgentState) -> Dict[str, Any]:
    try:
        embedding = await asyncio.to_thread(embed_texts, [state["query"]])
        chunks = await asyncio.to_thread(search, embedding[0], TOP_K)
        return {"chunks": chunks}
    except Exception as exc:
        print(f"[retrieve] failed: {exc}")
        return {"chunks": [], "error": str(exc)}


async def generate(state: AgentState) -> Dict[str, Any]:
    writer = get_stream_writer()

    if state.get("error"):
        writer("I couldn't search company documents right now. Please try again shortly.")
        return {"answer": "I couldn't search company documents right now. Please try again shortly."}

    chunks = state["chunks"]
    context = "\n\n".join(
        f"[{c['page_title']}]\n{c['content']}" for c in chunks
    )
    prompt = (
        f"Use the following excerpts from company documentation to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {state['query']}"
    )

    def _call():
        stream = _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        pieces = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                pieces.append(delta)
                writer(delta)
        return "".join(pieces)

    try:
        answer = await asyncio.to_thread(_call)
        return {"answer": answer}
    except Exception as exc:
        print(f"[generate] failed: {exc}")
        writer("I couldn't generate an answer right now. Please try again shortly.")
        return {"answer": "I couldn't generate an answer right now. Please try again shortly.", "error": str(exc)}


def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


graph = _build_graph()
