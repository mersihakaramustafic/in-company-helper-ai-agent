import asyncio
import os
import re
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from ingestion.embedder import embed_texts
from ingestion.vector_store import search

TOP_K = 5
# Cosine similarity below this is treated as unrelated (text-embedding-3-small
# scores on-topic chunks ~0.4+ and unrelated ones ~0.1-0.25).
MIN_SCORE = 0.3
MODEL = "gpt-4o-mini"
NO_RESULTS_ANSWER = "I couldn't find anything about that in the company documentation."
SYSTEM_PROMPT = (
    "You are a helpful assistant for company employees. Answer the question using "
    "only the documentation excerpts provided. Use information that is relevant to "
    "the question even if it doesn't match the wording exactly. Never add facts that "
    "aren't in the excerpts; if they contain nothing relevant, say you couldn't find "
    "it in the documentation. Be concise. Each excerpt is numbered; after every "
    "fact, cite the excerpt it came from like [1]. Cite multiple excerpts as [1][2]. "
    "Don't cite anything when you couldn't find the answer."
)
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class AgentState(TypedDict):
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


async def retrieve(state: AgentState) -> Dict[str, Any]:
    try:
        embedding = await asyncio.to_thread(embed_texts, [state["query"]])
        chunks = await asyncio.to_thread(search, embedding[0], TOP_K)
        return {"chunks": [c for c in chunks if c["score"] >= MIN_SCORE]}
    except Exception as exc:
        print(f"[retrieve] failed: {exc}")
        return {"chunks": [], "error": str(exc)}


async def generate(state: AgentState) -> Dict[str, Any]:
    writer = get_stream_writer()

    if state.get("error"):
        writer("I couldn't search company documents right now. Please try again shortly.")
        return {"answer": "I couldn't search company documents right now. Please try again shortly."}

    chunks = state["chunks"]
    if not chunks:
        writer(NO_RESULTS_ANSWER)
        return {"answer": NO_RESULTS_ANSWER}

    # One numbered entry per page, so citation numbers map directly to sources.
    pages: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        page = pages.setdefault(c["source_url"], {
            "id": len(pages) + 1,
            "title": c["page_title"],
            "url": c["source_url"],
            "score": c["score"],
            "contents": [],
        })
        page["score"] = max(page["score"], c["score"])
        page["contents"].append(c["content"])

    context = "\n\n".join(
        f"[{p['id']}] {p['title']}\n" + "\n\n".join(p["contents"])
        for p in pages.values()
    )
    prompt = (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"Question: {state['query']}"
    )

    def _call():
        stream = _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
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
        return {"answer": answer, "sources": _cited_sources(answer, list(pages.values()))}
    except Exception as exc:
        print(f"[generate] failed: {exc}")
        writer("I couldn't generate an answer right now. Please try again shortly.")
        return {"answer": "I couldn't generate an answer right now. Please try again shortly.", "error": str(exc)}


def _cited_sources(answer: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cited = {
        int(n)
        for group in CITATION_RE.findall(answer)
        for n in group.split(",")
    }
    return [
        {k: p[k] for k in ("id", "title", "url", "score")}
        for p in pages
        if p["id"] in cited
    ]


def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


graph = _build_graph()
