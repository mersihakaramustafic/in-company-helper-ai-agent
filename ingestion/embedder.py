import os
from typing import List
from openai import OpenAI

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    response = _client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in response.data]
