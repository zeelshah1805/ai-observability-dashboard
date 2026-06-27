# Deploying the dashboard

The dashboard self-seeds synthetic demo data on first load, so a fresh
deployment is never blank — visitors immediately see the cost/latency/error
charts and the v1-vs-v2 comparison. No API keys or live LLM calls are needed.

## Option A — Streamlit Community Cloud (recommended, free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Fill in:
   - **Repository:** `zeelshah1805/ai-observability-dashboard`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. (Optional) **Advanced settings → Python version:** 3.12.
5. **Deploy.** First build installs `requirements.txt` (a few minutes); after
   that the URL is live, e.g. `https://<your-app>.streamlit.app`.

Use the **🔄 Regenerate demo data** button in the sidebar to refresh the sample.

## Option B — Render (free web service)

A `render.yaml` is included. Connect the repo at <https://render.com> →
**New → Blueprint**, point it at this repo, and deploy. Note: the free tier
sleeps after ~15 min idle, so the first visit after a nap takes ~30–60s.

## Option C — Docker anywhere

```bash
docker build -t llm-observability .
docker run -p 8501:8501 llm-observability \
  streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

Or run the full stack (API + dashboard) with `docker-compose up --build`.

## Notes

- The trace store is SQLite on the local/ephemeral filesystem. On hosts with a
  non-persistent disk the data resets on restart — that's fine here because the
  dashboard re-seeds automatically.
- To demo against a real provider, set `LLM_PROVIDER` / `LLM_MODEL` /
  `GROQ_API_KEY` (see `.env.example`) and drive traffic through the FastAPI
  service with `scripts/load_gen.py`.
