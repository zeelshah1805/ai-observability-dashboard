"""Streamlit observability dashboard.

Reads the same SQLite trace store the API writes to and renders:
  - KPI tiles (requests, cost, error rate, p95 latency)
  - cost over time, latency percentiles, error breakdown
  - prompt v1-vs-v2 comparison
  - trace drill-down

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the `app` package importable when this file is run directly (e.g. on
# Streamlit Community Cloud, where the entrypoint is dashboard/app.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db as appdb  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.seed import seed_demo  # noqa: E402

DB_PATH = get_settings().db_path

st.set_page_config(page_title="LLM Observability", layout="wide")


@st.cache_data(ttl=5)
def load_traces(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM traces", conn)
        except Exception:
            return pd.DataFrame()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    return df


def pctl(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else 0.0


def _seed(clear: bool = False) -> None:
    with st.spinner("Generating demo telemetry..."):
        seed_demo(n_per_version=150, clear=clear)
    load_traces.clear()


st.title("🔭 LLM Observability Dashboard")

# On a fresh deploy the store is empty — seed it so the page is never blank.
appdb.init_db()
if appdb.count_traces() == 0:
    _seed(clear=False)

df = load_traces(DB_PATH)

with st.sidebar:
    if st.button("🔄 Regenerate demo data", use_container_width=True):
        _seed(clear=True)
        st.rerun()
    st.caption(
        "Demo data is synthetic (no live LLM calls). It mirrors a real workload "
        "comparing an unconstrained prompt (v1) against a constrained one (v2)."
    )

if df.empty:
    st.info("No traces yet — click **Regenerate demo data** in the sidebar.")
    st.stop()

# ---- Filters ---------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    models = sorted(df["model"].dropna().unique())
    versions = sorted(df["prompt_version"].dropna().unique())
    statuses = sorted(df["status"].dropna().unique())

    sel_models = st.multiselect("Model", models, default=models)
    sel_versions = st.multiselect("Prompt version", versions, default=versions)
    sel_status = st.multiselect("Status", statuses, default=statuses)

f = df[
    df["model"].isin(sel_models)
    & (df["prompt_version"].isin(sel_versions) | df["prompt_version"].isna())
    & df["status"].isin(sel_status)
]

# ---- KPI tiles -------------------------------------------------------------
# Latency is independent of output validity: include every completed
# generation (ok + bad_output), exclude only hard errors (partial latency).
completed = f[f["status"].isin(["ok", "bad_output"])]
total = len(f)
# True errors (timeout/rate_limit/error) are distinct from bad_output, which is
# a quality failure — keep them separate so neither tile misleads.
err_rate = (~f["status"].isin(["ok", "bad_output"])).mean() * 100 if total else 0.0
bad_rate = (f["status"] == "bad_output").mean() * 100 if total else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Requests", f"{total:,}")
c2.metric("Total cost", f"${f['cost_usd'].sum():.4f}")
c3.metric("Error rate", f"{err_rate:.1f}%")
c4.metric("Bad-output rate", f"{bad_rate:.1f}%")
c5.metric("p95 latency", f"{pctl(completed['latency_ms'], 0.95):.0f} ms")
c6.metric("Total tokens", f"{int(f['total_tokens'].sum()):,}")

st.divider()

# ---- Time series -----------------------------------------------------------
left, right = st.columns(2)

ts = f.set_index("timestamp").sort_index()
if not ts.empty:
    cost_by_min = ts["cost_usd"].resample("1min").sum().reset_index()
    fig_cost = px.area(
        cost_by_min, x="timestamp", y="cost_usd", title="Cost over time (USD/min)"
    )
    left.plotly_chart(fig_cost, use_container_width=True)

    lat = completed.set_index("timestamp")["latency_ms"].sort_index()
    if not lat.empty:
        roll = lat.resample("1min").agg(
            p50=lambda x: x.quantile(0.5),
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99),
        ).reset_index()
        fig_lat = px.line(
            roll,
            x="timestamp",
            y=["p50", "p95", "p99"],
            title="Latency percentiles (ms)",
        )
        right.plotly_chart(fig_lat, use_container_width=True)

# ---- Status breakdown ------------------------------------------------------
sc = f["status"].value_counts().reset_index()
sc.columns = ["status", "count"]
fig_status = px.bar(sc, x="status", y="count", title="Requests by status", color="status")
st.plotly_chart(fig_status, use_container_width=True)

# ---- Prompt v1 vs v2 comparison --------------------------------------------
st.subheader("Prompt version comparison")
if f["prompt_version"].nunique() >= 1:
    rows = []
    for v, g in f.groupby("prompt_version"):
        g_completed = g[g["status"].isin(["ok", "bad_output"])]
        rows.append(
            {
                "prompt_version": v,
                "requests": len(g),
                "avg_cost_usd": round(g["cost_usd"].mean(), 6),
                "p95_latency_ms": round(pctl(g_completed["latency_ms"], 0.95), 0),
                "bad_output_rate_%": round(
                    (g["status"] == "bad_output").mean() * 100, 1
                ),
                "error_rate_%": round(
                    (~g["status"].isin(["ok", "bad_output"])).mean() * 100, 1
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("Tag requests with prompt versions to compare them here.")

# ---- Trace drill-down ------------------------------------------------------
st.subheader("Recent traces")
display_cols = [
    "timestamp", "id", "model", "prompt_version", "status",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_ms",
]
recent = f.sort_values("timestamp", ascending=False).head(200)
st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

trace_id = st.selectbox("Inspect a trace", [""] + recent["id"].tolist())
if trace_id:
    row = f[f["id"] == trace_id].iloc[0]
    st.json(
        {
            "id": row["id"],
            "model": row["model"],
            "provider": row["provider"],
            "prompt_version": row["prompt_version"],
            "status": row["status"],
            "error_type": row.get("error_type"),
            "retries": int(row["retries"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "cost_usd": float(row["cost_usd"]),
            "latency_ms": int(row["latency_ms"]),
            "ttft_ms": (None if pd.isna(row["ttft_ms"]) else int(row["ttft_ms"])),
            "metadata": json.loads(row["metadata"] or "{}"),
        }
    )
    st.text_area("Input", row.get("input") or "", height=120)
    st.text_area("Output", row.get("output") or "", height=120)
