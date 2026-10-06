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

## Evaluation

Tracing, user feedback and evals go to Langfuse (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` in `.env`).

```bash
uv run python -m evals.run                      # run the agent on the Langfuse dataset and score it
uv run python -m evals.run --ci                 # exit 1 if a threshold is breached
```

The answer prompt is managed in Langfuse (`company-helper-answer`; the agent serves the `production` label). Create it in a new Langfuse project with `uv run python -m agent.prompts`.

The evaluation dataset lives in Langfuse (Datasets → `company-helper-eval`); edit items there.

See [docs/evaluation-framework.md](docs/evaluation-framework.md) for the dataset, metrics, judge and thresholds.
