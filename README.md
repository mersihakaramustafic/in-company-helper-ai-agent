# in-company-helper-ai-agent
## Running locally

Backend (from the repo root):

```bash
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

Frontend (in another terminal):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/chat` to the backend on port 8000.
