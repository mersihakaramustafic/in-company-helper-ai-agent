import asyncio
from typing import Any, Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from ingestion.embedder import embed_texts
from ingestion.vector_store import search

TOP_K = 5


class AgentState(TypedDict):
    query: str
    chunks: List[Dict[str, Any]]


async def retrieve(state: AgentState) -> Dict[str, Any]:
    embedding = await asyncio.to_thread(embed_texts, [state["query"]])
    chunks = await asyncio.to_thread(search, embedding[0], TOP_K)
    return {"chunks": chunks}


def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", END)
    return workflow.compile()


graph = _build_graph()
