# 🔭 AI Observability Dashboard

A production-style **observability layer for LLM applications**. A FastAPI
service wraps every model call and records token cost, latency, error rates,
prompt-version performance, and full per-request traces; a Streamlit dashboard
makes it all queryable.

100% free / open-source. Runs locally with **zero setup** against a built-in
mock provider, or against Groq / OpenRouter free tiers / local Ollama.

## Live demo

**▶ [Open the dashboard](https://ai-observability-dashboard.streamlit.app)**

The hosted dashboard self-seeds synthetic telemetry, so it loads straight into
the cost / latency / error views and the v1-vs-v2 prompt comparison — no setup,
no API key. See [DEPLOY.md](DEPLOY.md) for how it's deployed.

## Why I built this

Every LLM feature I worked on had the same blind spot: it would pass review and
then quietly drift — costs creeping up, latency spiking under load, a "small"
prompt tweak silently raising the rate of malformed answers. Ordinary APM
watches status codes and CPU; none of it sees token cost or whether the *output*
was actually any good.

So I built the control plane I wanted: instrument every model call, write a full
trace (cost, latency, tokens, prompt version, status), and make the trends
queryable in one place. The payoff is the prompt-regression workflow below — I
can compare two prompt versions on real telemetry instead of guessing.

---

## Architecture

```
client → FastAPI /chat ──► LLMClient wrapper
                            ├─ tokenizer (cost calc)
                            ├─ timer (latency)
                            ├─ prompt registry (version hash)
                            └─ @traced ──► writes a Trace row ──► SQLite
         FastAPI /metrics ─► Prometheus exposition
                                              │
                                              ▼
                                     Streamlit dashboard
```

The key design idea: a single `trace_call(...)` context manager
([app/tracing.py](app/tracing.py)) captures everything. Application code stays
clean; observability is cross-cutting.

## Quick start (local, no API key)

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -r requirements.txt
cp .env.example .env                                # defaults to provider=mock

# 1) run the API
uvicorn app.main:app --reload

# 2) generate realistic traffic (in another shell)
python scripts/load_gen.py --n 200

# 3) open the dashboard
streamlit run dashboard/app.py
```

API docs at http://localhost:8000/docs, metrics at http://localhost:8000/metrics.

## Using a real provider

Edit `.env`:

```
LLM_PROVIDER=groq            # or openrouter | ollama
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=...             # for groq/openrouter
```

All providers speak the OpenAI-compatible chat/completions API, so the wrapper
is identical across them.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + active provider/model |
| POST | `/chat` | instrumented LLM call → produces a Trace |
| GET | `/traces` | recent traces (filter by model/version/status) |
| GET | `/traces/{id}` | single-trace drill-down |
| GET | `/prompts` | registered prompts + version hashes |
| GET | `/metrics` | Prometheus exposition |

`/chat` accepts either a registered prompt (versioned) or raw messages:

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "prompt_name": "summarize",
  "variables": {"text": "Long text to summarize..."}
}'
```

## Prompt-regression demo

Every prompt template is registered with a content-hash version. The registry
seeds two `summarize` prompts: **v1** (terse, unconstrained) and **v2**
(constrained to a 2-sentence contract). One command runs the same workload
under both and prints a side-by-side table from real trace data:

```bash
python scripts/regression_demo.py --n 150
```

```
Metric           | v1 (baseline) | v2 (improved) | Delta
-----------------+---------------+---------------+------
p95 latency      | 1981 ms       |  735 ms       | -63%
Bad-output rate  | 86.7%         | 10.7%         | -88%
Avg cost / req   | $0.000006     | $0.000005     | -10%
Error rate       | 2.0%          | 1.3%          | -33%
Requests sampled | 150           | 150           | -
```

(Numbers come from the built-in mock provider, which simulates an LLM that
follows an explicit constraint more reliably than a vague one — so v2's tighter
prompt yields shorter, faster, lower-variance output. Against a real provider
the same machinery measures real telemetry.)

The same data is visible live in the dashboard's **Prompt version comparison**
table — the v1/v2 difference is backed by real telemetry, not guesswork.

## Docker

```bash
docker-compose up --build       # API on :8000, dashboard on :8501
```

## Data model

A single `traces` table is the backbone — see [app/models.py](app/models.py)
and [app/db.py](app/db.py). Cost is attributed from token usage × a per-model
price table ([app/pricing.py](app/pricing.py)); for free tiers the cost is a
*simulated* list-price figure, not actually incurred.

## Scope

**In scope:** per-request + aggregate cost/latency/error/prompt-version
tracking, trace inspection, Prometheus metrics.

**Out of scope (next steps):** distributed tracing across services,
multi-tenant auth, real alerting integrations, OpenTelemetry export to
Grafana/Tempo, async write buffering + sampling.

## Features

- Instrumented `LLMClient` — every `/chat` call produces a complete trace
- Token cost attribution from a per-model price table
- Latency percentiles (p50/p95/p99) and time-to-first-token
- Prompt registry with content-hash versioning + v1/v2 comparison
- Reliability: retries with backoff, timeout handling, bad-output detection
- Streamlit dashboard (cost/latency/error views + trace drill-down)
- Prometheus `/metrics` endpoint, Dockerfile + docker-compose
