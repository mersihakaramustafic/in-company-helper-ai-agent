import os
import re
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict
from langfuse import observe
from langfuse.openai import AsyncOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from agent.prompts import get_answer_prompt, prompt_version
from ingestion.embedder import aembed_texts
from ingestion.vector_store import asearch

TOP_K = 5
# Cosine similarity below this is treated as unrelated (text-embedding-3-small
# scores on-topic chunks ~0.4+ and unrelated ones ~0.1-0.25).
MIN_SCORE = 0.3
MODEL = "gpt-4o-mini"
NO_RESULTS_ANSWER = "I couldn't find anything about that in the company documentation."
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def agent_config() -> dict:
    """Everything that changes agent behaviour; attached to traces and eval runs."""
    return {
        "model": MODEL,
        "top_k": str(TOP_K),
        "min_score": str(MIN_SCORE),
        "prompt_version": prompt_version(get_answer_prompt()),
    }


# Langfuse drop-in client: logs each call as a generation with model, tokens and cost.
_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


class AgentState(TypedDict):
    query: str
    retrieved: List[Dict[str, Any]]  # top-K from vector search, ranked, before MIN_SCORE
    chunks: List[Dict[str, Any]]  # retrieved chunks that pass MIN_SCORE; sent to the LLM
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


def initial_state(query: str) -> AgentState:
    return {"query": query, "retrieved": [], "chunks": [], "answer": "", "sources": [], "error": None}


def build_context(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Groups chunks into one numbered entry per page, so citation numbers map
    directly to sources. Returns the pages and the excerpt text for the prompt."""
    pages: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        page = pages.setdefault(c["source_url"], {
            "id": len(pages) + 1,
            "page_id": c["page_id"],
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
    return list(pages.values()), context


@observe(name="retrieve", as_type="retriever")
async def retrieve(state: AgentState) -> Dict[str, Any]:
    try:
        embedding = await aembed_texts([state["query"]])
        retrieved = await asearch(embedding[0], TOP_K)
        return {
            "retrieved": retrieved,
            "chunks": [c for c in retrieved if c["score"] >= MIN_SCORE],
        }
    except Exception as exc:
        print(f"[retrieve] failed: {exc}")
        return {"retrieved": [], "chunks": [], "error": str(exc)}


@observe(name="generate")
async def generate(state: AgentState) -> Dict[str, Any]:
    writer = get_stream_writer()

    if state.get("error"):
        writer("I couldn't search company documents right now. Please try again shortly.")
        return {"answer": "I couldn't search company documents right now. Please try again shortly."}

    chunks = state["chunks"]
    if not chunks:
        writer(NO_RESULTS_ANSWER)
        return {"answer": NO_RESULTS_ANSWER}

    pages, context = build_context(chunks)
    prompt = get_answer_prompt()
    messages = prompt.compile(context=context, question=state["query"])

    async def _call():
        stream = await _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            name="answer",
            langfuse_prompt=prompt,  # links the generation to the exact prompt version
        )
        pieces = []
        async for chunk in stream:
            # With include_usage, the final chunk carries token usage and no choices.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                pieces.append(delta)
                writer(delta)
        return "".join(pieces)

    try:
        answer = await _call()
        return {"answer": answer, "sources": _cited_sources(answer, pages)}
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
        {k: p[k] for k in ("id", "page_id", "title", "url", "score")}
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
