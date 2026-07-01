from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from schemas.chat import ChatRequest
from agent.graph import graph

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    result = await graph.ainvoke({
        "query": request.message,
        "chunks": []
    })

    return result