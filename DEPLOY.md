# Deploying the RAG Assistant (Streamlit Cloud + Render + Qdrant Cloud)

Streamlit Community Cloud runs **one Python process** with an **ephemeral disk**,
so it can't host the FastAPI backend, Qdrant, and Redis. This guide uses the
**split** architecture:

```
Streamlit Community Cloud            Render (Docker)              Qdrant Cloud
┌───────────────────────┐  HTTPS   ┌────────────────────┐  HTTPS ┌─────────────┐
│  Streamlit UI         │ ───────► │  FastAPI backend   │ ─────► │  Vectors    │
│  (ui/app.py)          │  API_URL │  (Dockerfile.api)  │  +key  │  (managed)  │
└───────────────────────┘          └────────────────────┘        └─────────────┘
```

- **Cache:** no Redis needed — set `CACHE_MODE=memory` (the app also falls back
  to in-memory automatically if Redis is unreachable).
- **CORS:** not a concern — the Streamlit server calls the API server-side
  (httpx), so the browser never calls the API directly.

---

## Prerequisites

- Accounts: [Qdrant Cloud](https://cloud.qdrant.io), [Render](https://render.com),
  [Streamlit Community Cloud](https://share.streamlit.io) (all have free tiers).
- API keys: `VOYAGE_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`
  (and `OPENAI_API_KEY` if you use GPT models).
- A small, **non-confidential** demo corpus (PDF/DOCX/PPTX/XLSX). Do **not**
  upload private customer docs — this repo is public.

---

## 1. Qdrant Cloud (vector DB)

1. Create a free cluster. Copy its **URL** (e.g.
   `https://xxxx.eu-central.aws.cloud.qdrant.io:6333`) and an **API key**.
2. The collection name defaults to `default` (env `QDRANT_COLLECTION`); the
   backend creates it on first ingest.

## 2. Backend on Render

1. Render → **New → Blueprint** → select this repo. It reads [`render.yaml`](render.yaml)
   and provisions a Docker web service from [`Dockerfile.api`](Dockerfile.api).
   (Or **New → Web Service → Docker**, Dockerfile path `./Dockerfile.api`.)
2. Set the secret env vars (marked `sync: false` in the blueprint):
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `VOYAGE_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`
   - `ADMIN_TOKEN` (any value; the UI must use the same)
   - `CACHE_MODE=memory` is preset by the blueprint.
3. Deploy. Health check is `/readyz`. When live, note the URL, e.g.
   `https://rag-assistant-api.onrender.com`. Verify:
   ```bash
   curl https://rag-assistant-api.onrender.com/health
   ```
   Expect `qdrant: "ok"`. (Free instances cold-start after idle — first request
   can take ~50s.)

## 3. Index the demo corpus

Vectors live in Qdrant Cloud (persistent). The simplest way to load docs:

- **Via the UI** (after step 4): open the **Corpus** page and upload files, or
- **Via curl**, one file at a time:
  ```bash
  curl -X POST https://rag-assistant-api.onrender.com/ingest/upload \
    -F "file=@./demo_docs/manual.pdf"
  ```
Confirm with `GET /stats` (`total_chunks` > 0).

## 4. UI on Streamlit Community Cloud

1. New app → this repo → **Main file path:** `ui/app.py`.
2. **Secrets** (App settings → Secrets) — see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example):
   ```toml
   API_URL = "https://rag-assistant-api.onrender.com"
   ADMIN_TOKEN = "same-value-as-render"
   ```
   Streamlit exposes these as env vars, which the UI reads via `os.getenv` — no
   code change needed.
3. Dependencies: Streamlit Cloud installs the repo-root `requirements.txt` (which
   already contains the UI's `httpx`/`pydantic`/`pyyaml`); [`packages.txt`](packages.txt)
   provides `libmagic1` so that install completes cleanly.
4. Deploy, then ask a question in **Chat** — it round-trips to Render → Qdrant
   Cloud and renders the grounding badge, estimated cost, and citations.

---

## Notes & limitations (free tier)

- **Extracted images / source files** are written to the backend's local disk
  (`corpus/`), which is **ephemeral** on Render's free plan — text RAG works
  fully (vectors persist in Qdrant Cloud), but image thumbnails may 404 after a
  cold restart. For a stable public demo, attach a Render **persistent disk**
  mounted at `/app/corpus` and `/app/data`, or re-upload after restarts.
- **Cold starts:** the free Render instance sleeps when idle; the first request
  wakes it (~50s). The UI shows a spinner.
- **Cost:** generation calls cost money (shown per-answer in the UI). Use
  `gemini-2.5-flash-lite` (the default) to keep it cheap.
- **Optional Redis:** to enable the semantic cache, set `REDIS_URL` to an Upstash
  instance and `CACHE_MODE=semantic` instead of `memory`.
