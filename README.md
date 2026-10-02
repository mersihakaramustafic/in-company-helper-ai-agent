# in-company-helper-ai-agent
## Running locally

Requires [uv](https://docs.astral.sh/uv/). Backend (from the repo root):

```bash
uv sync
uv run uvicorn api:app --reload --port 8000
```

Ingest Notion pages into pgvector:

```bash
uv run python -m ingestion.pipeline
```

Add a dependency with `uv add <package>`; it updates `pyproject.toml` and `uv.lock`.

Frontend (in another terminal):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/chat` to the backend on port 8000.
