# streamlit_echo_test.py
import glob
import os
import time
from collections import deque

import pandas as pd
import requests
import streamlit as st

# === Configuration - replace with your webhook(s) ===
N8N_WEBHOOK_URL = "https://fpgconsulting.app.n8n.cloud/webhook-test/echo_agent"
N8N_WRITE_URL = "https://fpgconsulting.app.n8n.cloud/webhook-test/write_agent"  # optional

# === Secret retrieval (Streamlit Secrets preferred) ===
try:
    WEBHOOK_SECRET = st.secrets.get("WEBHOOK_SECRET", None)
except Exception:
    WEBHOOK_SECRET = None

if not WEBHOOK_SECRET:
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "WIBBLE")


def compute_nodes_to_query(df_tax: pd.DataFrame, selected_path_or_name: str, rel_depth: int):
    """
    df_tax: normalized DataFrame with columns like ['node_id','parent_id','level','path','display_name']
    selected_path_or_name: string (either path or display_name)
    rel_depth: int (0 => only selected node, 1 => include immediate children, etc)
    Returns: list of dicts: [{node_id, path, display_name, level, parent_id}, ...]
    """
    if df_tax is None or df_tax.empty:
        return []

    # Prefer exact path match, then exact display_name match, then substring match
    root_row = None
    if "path" in df_tax.columns:
        exact_path = df_tax[df_tax["path"] == selected_path_or_name]
        if not exact_path.empty:
            root_row = exact_path.iloc[0]

    if root_row is None and "display_name" in df_tax.columns:
        exact_name = df_tax[df_tax["display_name"] == selected_path_or_name]
        if not exact_name.empty:
            root_row = exact_name.iloc[0]

    if root_row is None and "display_name" in df_tax.columns:
        matches = df_tax[df_tax["display_name"].str.contains(selected_path_or_name, na=False, case=False)]
        if not matches.empty:
            root_row = matches.iloc[0]

    if root_row is None and "path" in df_tax.columns:
        # fallback: substring match on path
        matches = df_tax[df_tax["path"].str.contains(selected_path_or_name, na=False, case=False)]
        if not matches.empty:
            root_row = matches.iloc[0]

    if root_row is None:
        raise ValueError("Selected node not found in taxonomy dataframe")

    root_id = root_row.get("node_id")
    root_path = root_row.get("path", "")

    # Build parent -> children map if parent_id exists
    children_map = {}
    if "parent_id" in df_tax.columns and "node_id" in df_tax.columns:
        for _, r in df_tax.iterrows():
            pid = r.get("parent_id", None)
            nid = r.get("node_id")
            children_map.setdefault(pid, []).append(nid)
    else:
        children_map = None

    results = []

    if children_map:
        id_to_row = {r["node_id"]: r for _, r in df_tax.to_dict(orient="index").items() if r.get("node_id") is not None}
        q = deque()
        q.append((root_id, 0))
        visited = set()

        while q:
            current_id, depth = q.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            row = df_tax[df_tax["node_id"] == current_id]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append(
                {
                    "node_id": row.get("node_id"),
                    "path": row.get("path"),
                    "display_name": row.get("display_name"),
                    "level": int(row["level"]) if "level" in row and pd.notna(row["level"]) else None,
                    "parent_id": row.get("parent_id"),
                }
            )
            if depth < rel_depth:
                for child_id in children_map.get(current_id, []):
                    q.append((child_id, depth + 1))
    else:
        # Fallback: use path prefix matching and compute relative depth by separators
        prefix = root_path or selected_path_or_name
        matched = df_tax[df_tax["path"].str.startswith(prefix, na=False)]
        root_depth = prefix.count(" > ")

        def rel_d(row_path: str) -> int:
            return row_path.count(" > ") - root_depth

        for _, row in matched.iterrows():
            if rel_d(row.get("path", "")) <= rel_depth:
                results.append(
                    {
                        "node_id": row.get("node_id"),
                        "path": row.get("path"),
                        "display_name": row.get("display_name"),
                        "level": int(row["level"]) if "level" in row and pd.notna(row["level"]) else None,
                        "parent_id": row.get("parent_id"),
                    }
                )

    return results

st.title("n8n Echo / Query Test")

st.markdown("Upload your taxonomy Excel, pick a node, set query options, and POST to n8n.")

# 1) Upload taxonomy Excel and choose column to use as node labels
node_choice = None
df = None

# Auto-detect taxonomy file in repo (GitHub). Patterns are checked in order.
LOCAL_TAXONOMY_FILE = None
patterns = [
    "./Aero and Defence Taxonomy Structure for n8n.xlsx",
    "./Aero and Defence Taxonomy Structure for n8n DATA for charts.xlsx",
    "./taxonomy*.xlsx",
    "./data/*.xlsx",
    "./*.xlsx",
]
for p in patterns:
    matches = glob.glob(p)
    if matches:
        LOCAL_TAXONOMY_FILE = matches[0]
        break

if LOCAL_TAXONOMY_FILE:
    try:
        df = pd.read_excel(LOCAL_TAXONOMY_FILE, sheet_name=0)
        st.info(f"Auto-loaded taxonomy from repo file: {os.path.basename(LOCAL_TAXONOMY_FILE)}")
    except Exception as e:
        st.warning(f"Auto-load failed ({LOCAL_TAXONOMY_FILE}): {e}")
        df = None

uploaded = st.file_uploader("Upload taxonomy Excel (.xlsx) — or leave blank to use auto-detected file", type=["xlsx"])

# Only read the uploaded file if auto-load did not already populate df
if uploaded and df is None:
    try:
        df = pd.read_excel(uploaded, sheet_name=0)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        df = None

if df is not None:
    st.write("Detected columns:", list(df.columns))
    candidate_cols = [
        c
        for c in df.columns
        if c and str(c).lower() in ("hierarchy", "node", "node_path", "name", "title")
    ]
    default_col = candidate_cols[0] if candidate_cols else df.columns[0]
    selected_col = st.selectbox(
        "Which column contains the taxonomy node labels?",
        df.columns,
        index=list(df.columns).index(default_col),
    )
    try:
        sample_vals = df[selected_col].dropna().astype(str).unique().tolist()
    except Exception:
        sample_vals = df[selected_col].dropna().astype(str).head(100).tolist()
    node_choice = st.selectbox("Choose taxonomy node", sample_vals)
else:
    st.info("Auto-detected no taxonomy file — upload an Excel file to select taxonomy nodes.")
    node_choice = st.text_input("Default taxonomy node (used when no Excel uploaded):", value="DEFAULT_NODE")

# === Query options / Prompt editor / Run ===
st.subheader("Query options")

# Hidden defaults used by the workflow
DEFAULT_REQUIRED_FIELDS = ["summary", "sources"]
DEFAULT_QUERY_DEPTH = 3

# 1) Global context (editable, pre-filled)
default_global_context = """We are conducting a systematic, node-by-node analysis of the global Aerospace & Defence industry.
This industry is organized into a 5-tier hierarchical taxonomy:
1. Total — The overall Aerospace & Defence industry.
2. Main Categories — Aerospace and Defence.
3. Sectors — Subdivisions within each Main Category.
4. Sub-Sectors — Granular specializations within each Sector.
5. Sub-Sub-Sectors — Final level nodes, which may contain pure-play or adjacent market players.

Before performing any analysis, read and interpret the Aerospace & Defence taxonomy from this Excel file:
https://raw.githubusercontent.com/Steve-Loncar/echo_test/main/taxonomy_normalized_cleaned.xlsx

Use this taxonomy to understand the hierarchy, relationships, and node structure.
Do not attempt to recreate or re-summarise it — just use it to inform scope and boundaries for the node analyses.

Each node represents a distinct scope of analysis. Nodes are independent — data, players, and commentary
should be limited to the node’s direct scope and immediate children only.

All financials are in USD billions unless otherwise stated.
All CAGR and margin figures are FY23–FY25 unless otherwise noted.

You must respect taxonomy boundaries — do not merge, rename, or infer data from unrelated nodes.
"""
global_context = st.text_area("Global context (applies to all nodes in this run)", value=default_global_context, height=260)

# 2) Prompt editor (default strict JSON prompt)
st.markdown("### Prompt editor")
st.markdown("The prompt below is pre-filled with a strict JSON schema for the LLM. Edit it to refine what you send to the model. Use placeholders: {{display_name}}, {{path}}, {{node_id}}, {{required_fields}}, {{extra_context}}, {{query_depth}}, {{global_context}}")

default_prompt = """You are a strict JSON-outputting market analyst. Return exactly ONE JSON OBJECT and nothing else (no commentary, no explanation, no code fences).

Schema (MUST be followed exactly):

{
  "node_id": string | null,
  "path": string,
  "display_name": string,
  "level": integer | null,
  "requested_fields": array[string],
  "results": {
    "node_financials": {
      "fy23_revenue_usd_bn": number | null,
      "fy24_revenue_usd_bn": number | null,
      "fy25_revenue_usd_bn": number | null,
      "revenue_cagr_pct": number | null,
      "fy23_ebitda_usd_bn": number | null,
      "fy24_ebitda_usd_bn": number | null,
      "fy25_ebitda_usd_bn": number | null,
      "ebitda_cagr_pct": number | null,
      "fy23_ebitda_margin_pct": number | null,
      "fy24_ebitda_margin_pct": number | null,
      "fy25_ebitda_margin_pct": number | null
    },
    "node_players": [
      {
        "rank": number,
        "name": string,
        "country": string | null,
        "type": string | null,
        "fy25_node_revenue_usd_bn": number | null,
        "fy25_node_ebitda_usd_bn": number | null,
        "fy25_node_ebitda_margin_pct": number | null,
        "confidence_score": number,
        "attribution_basis": string
      }
    ],
    "pure_play_estimates": [
      {
        "name": string,
        "country": string | null,
        "type": string | null,
        "proxy_reason": string,
        "fy25_revenue_usd_bn": number | null,
        "fy25_ebitda_usd_bn": number | null,
        "fy25_ebitda_margin_pct": number | null,
        "confidence_score": number
      }
    ],
    "methodology_summary": string | null,
    "financial_commentary": string | null,
    "player_commentary": string | null,
    "taxonomy_reference": string | null,
    "sources": [
      {
        "title": string,
        "url": string | null,
        "publisher": string | null,
        "year": string | null,
        "confidence_score": number
      }
    ]
  },
  "evidence": [
    {
      "evidence_id": string,
      "title": string | null,
      "url": string | null,
      "type": string | null,
      "snippet": string | null,
      "confidence_score": number
    }
  ],
  "assertions": [
    {
      "claim": string,
      "supported_by": ["evidence_id", ...],
      "confidence": number
    }
  ],
  "notes": string | null,
  "confidence": number,
  "timestamp": string
}

Context and placeholders you MUST use:
- Global context: {{global_context}}
- Node display_name: {{display_name}}
- Node path: {{path}}
- Node id: {{node_id}}
- Requested fields: {{required_fields}}
- Extra context: {{extra_context}}
- Query depth: {{query_depth}}
- Retrieved documents (if provided): use only the objects in retrieved_docs; each has id, title, url, snippet, text.

For every node analysis, complete the following structured subtasks inside the "results" object.
All quantitative values must include units (USD billions, %, etc.). 
If data are estimates or derived from proxies, clearly state the basis and confidence in "notes".

### 1. Node Financial Data
Purpose: quantify the commercial scale of the node for cross-node comparison.

Return object `node_financials` with:
{
  "fy23_revenue_usd_bn": number | null,
  "fy24_revenue_usd_bn": number | null,
  "fy25_revenue_usd_bn": number | null,
  "revenue_cagr_pct": number | null,
  "fy23_ebitda_usd_bn": number | null,
  "fy24_ebitda_usd_bn": number | null,
  "fy25_ebitda_usd_bn": number | null,
  "ebitda_cagr_pct": number | null,
  "fy23_ebitda_margin_pct": number | null,
  "fy24_ebitda_margin_pct": number | null,
  "fy25_ebitda_margin_pct": number | null
}
If direct figures are unavailable, infer from segment disclosures, peer averages, or public industry ratios and record the estimation method and assumptions in `notes`.

### 2. Node Player Data
Purpose: identify and profile the main companies that materially participate in this node.

Return top 5 players ranked by FY25 **node-attributable revenue** in object `node_players`, each entry:
{
  "rank": number,
  "name": string,
  "country": string | null,
  "type": string | null,   // prime, tier 1/2/3, specialist, startup, etc.
  "fy25_node_revenue_usd_bn": number | null,
  "fy25_node_ebitda_usd_bn": number | null,
  "fy25_node_ebitda_margin_pct": number | null,
  "confidence_score": number,
  "attribution_basis": string   // e.g. "direct disclosure", "segment revenue", "modelled estimate", "derived from comparable peers"
}
Clarify that all financial metrics represent **only the portion attributable to this node**, not total company results.  
If segment data are disclosed, use those directly.  
If not disclosed, infer using proportionate segment weightings, business mix, or peer benchmarking — and explain how attribution was determined in `attribution_basis` and `notes`.  
Avoid aggregating full company revenues unless the firm is a pure play in this node.

### 3. Node Pure-Play / Proxy Estimates
Purpose: provide cleaner benchmarks when most participants are diversified.

Return up to 3 representative pure-plays or near-adjacent proxies in `pure_play_estimates`, each entry:
{
  "name": string,
  "country": string | null,
  "type": string | null,
  "proxy_reason": string,   // e.g. "sole focus on avionics sensors" or "closest adjacent propulsion OEM"
  "fy25_revenue_usd_bn": number | null,
  "fy25_ebitda_usd_bn": number | null,
  "fy25_ebitda_margin_pct": number | null,
  "confidence_score": number
}
Explain **why** each proxy is relevant — specialization, technology overlap, customer base, or cost structure similarity — and explicitly note when using proxies instead of true participants.

### 4. Methodology Summary
Purpose: ensure transparency and reproducibility of sizing approach.

In `results.methodology_summary`, concisely explain:
- Main data sources (reports, databases, company filings, or inference).  
- How node-level financials were attributed from segment or total-company data.  
- Treatment of diversified companies.  
- Any currency or inflation adjustments or conversion assumptions.
Limit to ≤150 words.

### 5. Evidence Handling
Every numeric or factual statement must be supported by an `evidence` object with a confidence score (0–1). 
Evidence should reference credible financial or industry sources (company reports, market studies, analyst estimates, etc.), not blogs or forums.


Rules (MANDATORY):
1) Output EXACTLY ONE JSON OBJECT conforming to the schema above, nothing else.
2) For every item in requested_fields include a corresponding key in results. If unknown, set scalar -> null, arrays -> [].
3) All financials must be in USD billions unless otherwise stated.
4) CAGR and margin figures must use FY23–FY25 unless otherwise noted.
5) Use retrieved_docs ONLY when supplied. Do not invent URLs. If you assert a source without URL, set url=null and confidence_score <= 0.3.
6) Evidence objects must reference retrieved_docs ids where applicable; if evidence is generated, use a new evidence_id and set confidence_score <= 0.5.
7) All confidence scores must be between 0.0 and 1.0.
8) timestamp MUST be ISO-8601 UTC (e.g., 2025-10-27T15:21:00Z).
9) Keep evidence array <= 10 items, ordered by confidence_score descending.
10) If retrieval is empty/not provided, set overall confidence <= 0.5 and explain in notes: \"No retrieval evidence provided — answers are model-derived and must be verified.\"
11) Include methodology_summary describing how market sizing was derived.
12) Include taxonomy_reference clarifying how this node relates to its parent and siblings.
13) Do not include any text outside the JSON object.
14) Refer to the attached document taxonomy_normalized_cleaned.xlsx for the official 5-tier Aerospace & Defence taxonomy. Use it to verify scope boundaries and hierarchical relationships.

"""

prompt_text = st.text_area("Prompt (editable)", value=default_prompt, height=320)

# 3) Extra context (node/run-level, shorter than global)
extra_context = st.text_area("Extra context / constraints (optional, node-level)", height=100)

# 4) LLM model & simple technical settings (clean, consistent)
st.markdown("### LLM model & settings")
_model_choice = st.selectbox("Model (choose or pick Other to type)", ["sonar (default)", "Other..."], index=0)
if _model_choice == "Other...":
    model_name = st.text_input("Model name", value="sonar")
else:
    model_name = "sonar"

temperature = st.slider("Temperature (determinism)", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                        help="Lower = more deterministic. Use 0 for structured JSON outputs.")
max_tokens = st.number_input("Max tokens", min_value=256, max_value=8000, value=1500,
                             help="Max tokens to request from the model. Increase if you expect long outputs.")

include_retrieval = st.checkbox("Include retrieval (top-k docs) before sending to LLM", value=False,
                                help="If enabled, n8n should run a retriever and include retrieved_docs in the prompt.")
priority = st.selectbox("Priority", ["normal", "high", "low"], index=0,
                        help="Tag the job priority; can be used by downstream schedulers or model routing.")
dry_run = st.checkbox("Dry run (don't persist downstream)", value=False, help="If set, results won't be written to datasets downstream.")

# Run / Preview buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Preview merged prompt"):
        # Simple client-side preview: best-effort replace of placeholders with current inputs
        display_preview = prompt_text
        display_preview = display_preview.replace("{{global_context}}", str(global_context))
        display_preview = display_preview.replace("{{display_name}}", str(node_choice))
        display_preview = display_preview.replace("{{path}}", str(node_choice))
        display_preview = display_preview.replace("{{required_fields}}", str(DEFAULT_REQUIRED_FIELDS))
        display_preview = display_preview.replace("{{extra_context}}", str(extra_context))
        display_preview = display_preview.replace("{{query_depth}}", str(DEFAULT_QUERY_DEPTH))
        st.subheader("Merged prompt preview")
        st.code(display_preview)
with col2:
    if st.button("Run query"):
        if not node_choice:
            st.error("No taxonomy node selected. Upload an Excel and select a node first.")
        else:
            # Resolve actual node fields from the uploaded taxonomy (best-effort)
            node_id_val = None
            node_path = str(node_choice)
            display_name = str(node_choice)
            level_val = None
            parent_id_val = None
            node_row = None

            if df is not None:
                try:
                    # 1) Prefer exact match on the user-selected column (selected_col) if available
                    if 'selected_col' in locals() and selected_col in df.columns:
                        matches = df[df[selected_col].astype(str) == str(node_choice)]
                        if not matches.empty:
                            node_row = matches.iloc[0]

                    # 2) Try common column names for exact matches
                    if node_row is None:
                        for colname in ('path', 'display_name', 'name', 'title'):
                            if colname in df.columns:
                                matches = df[df[colname].astype(str) == str(node_choice)]
                                if not matches.empty:
                                    node_row = matches.iloc[0]
                                    break

                    # 3) Substring fallback on the selected column
                    if node_row is None and 'selected_col' in locals() and selected_col in df.columns:
                        try:
                            matches = df[df[selected_col].astype(str).str.contains(str(node_choice), na=False, case=False)]
                            if not matches.empty:
                                node_row = matches.iloc[0]
                        except Exception:
                            pass

                    # 4) Path-prefix fallback
                    if node_row is None and 'path' in df.columns:
                        matched = df[df['path'].astype(str).str.startswith(str(node_choice))]
                        if not matched.empty:
                            node_row = matched.iloc[0]

                    # 5) Extract fields if we found a row
                    if node_row is not None:
                        node_id_val = node_row.get('node_id') or node_row.get('id') or node_row.get('nodeId') or node_row.get('NodeID')
                        # If no explicit id column, derive a stable id from the dataframe index (best-effort)
                        if not node_id_val:
                            try:
                                idx = int(node_row.name)
                                node_id_val = f"N{idx:05d}"
                            except Exception:
                                node_id_val = None

                        node_path = node_row.get('path') or node_path
                        if 'selected_col' in locals() and selected_col in node_row:
                            display_name = node_row.get('display_name') or node_row.get(selected_col) or display_name
                        else:
                            display_name = node_row.get('display_name') or display_name

                        try:
                            level_val = int(node_row['level']) if 'level' in node_row and pd.notna(node_row['level']) else level_val
                        except Exception:
                            level_val = level_val

                        parent_id_val = node_row.get('parent_id') or node_row.get('parentId') or parent_id_val

                except Exception as e:
                    # Non-fatal: keep defaults and surface a warning for debugging
                    st.warning(f"Node lookup warning: {e}")

            # Final fallback: try deriving id from dataframe index where possible
            if not node_id_val and df is not None and 'selected_col' in locals() and selected_col in df.columns:
                try:
                    idxs = df[df[selected_col].astype(str) == str(node_choice)].index
                    if len(idxs) > 0:
                        node_id_val = f"N{int(idxs[0]):05d}"
                except Exception:
                    node_id_val = None

            # If still unresolved, leave node_id as None (better than a misleading placeholder)
            if not node_id_val:
                node_id_val = None

            single_node = {
                "node_id": node_id_val,
                "path": node_path,
                "display_name": display_name,
                "level": level_val,
                "parent_id": parent_id_val
            }

            # Debug output in the UI so you can confirm what was resolved
            st.write("Resolved node:", single_node)

            payload = {
                "taxonomy_node": node_choice,
                "nodes_to_query": [single_node],
                "rel_depth": 0,
                "query_depth": int(DEFAULT_QUERY_DEPTH),
                "required_fields": DEFAULT_REQUIRED_FIELDS,
                "global_context": global_context,
                "extra_context": extra_context,
                "prompt_text": prompt_text,
                "model_name": model_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "include_retrieval": bool(include_retrieval),
                "priority": priority,
                "dry_run": bool(dry_run),
                "client_timestamp": time.time()
            }

            headers = {"X-Webhook-Secret": WEBHOOK_SECRET, "Content-Type": "application/json"}

            try:
                resp = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=120)
                st.write("Status:", resp.status_code)
                try:
                    j = resp.json()
                    st.subheader("Response (JSON)")
                    st.json(j)
                    if isinstance(j, dict) and "rows" in j and isinstance(j["rows"], list) and len(j["rows"]) > 0:
                        try:
                            df_rows = pd.DataFrame(j["rows"])
                            st.subheader("Rows (table)")
                            st.dataframe(df_rows)
                        except Exception:
                            st.write("Rows present but failed to render as table.")
                except Exception:
                    st.subheader("Raw response")
                    st.text(resp.text)
            except Exception as e:
                st.error(f"Request failed: {e}")