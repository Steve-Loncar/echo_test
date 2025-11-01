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
DEFAULT_REQUIRED_FIELDS = [
    "summary",
    "node_financials",
    "node_players",
    "pure_play_estimates",
    "methodology_summary",
    "financial_commentary",
    "player_commentary",
    "sources"
]
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

Clarifications (MANDATORY):
- You must populate **every field** in the "results" object, even if partial or estimated.
- Do not omit or collapse sub-objects such as `node_financials`, `node_players`, or `pure_play_estimates`.
- For numeric data, include estimates when exact figures are unavailable and describe the estimation logic in `methodology_summary`.
- All player- and company-level metrics must reflect values **attributable to this taxonomy node only** (if disclosed, cite; if inferred, explain how).
- Use `null` for unknown scalars and `[]` for missing arrays.
- Always output one and only one JSON object conforming to the schema above.

"""

prompt_text = st.text_area("Prompt (editable)", value=default_prompt, height=320)

# 3) Extra context (node/run-level, shorter than global)
extra_context = st.text_area("Extra context / constraints (optional, node-level)", height=100)

# 4) LLM model & simple technical settings (clean, consistent)
st.markdown("### ⚙️ LLM Model & Settings — *Control accuracy, depth, and cost*")
st.markdown("""
Choose a **Perplexity Sonar** model variant and adjust parameters to balance speed, analytical depth, and cost.

**Model Options**
- 🧩 **sonar (default)** — Fast, concise, best for summaries or debugging.
- ⚙️ **sonar-pro** — Slower but performs multi-document reasoning for better numeric coherence.
- 🔍 **sonar-deep-research** — Most thorough, cross-validates sources and produces full analytical writeups.

> 💸 *Note:* higher models and larger token counts may cost more per request.  
See [Perplexity API pricing](https://docs.perplexity.ai/docs/pricing) for current rates.
""")

_model_choice = st.selectbox(
    "Select model variant:",
    ["sonar (default)", "sonar-pro", "sonar-deep-research"],
    index=2,
    help="Select a model tuned for your analysis depth. Deep-research is most capable but slower and costlier."
)

# Map to model identifiers
if _model_choice.startswith("sonar-pro"):
    model_name = "sonar-pro"
elif _model_choice.startswith("sonar-deep-research"):
    model_name = "sonar-deep-research"
else:
    model_name = "sonar"

# Default analytical tuning for refined research
DEFAULT_MODEL_NAME = model_name
DEFAULT_TEMPERATURE = 0.35
DEFAULT_MAX_TOKENS = 5000

temperature = st.slider(
    "Temperature (analytical creativity)",
    min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05,
    help="Higher = more flexible reasoning and interpolation. 0.35 is ideal for analytical estimation."
)

if st.session_state["env_mode"] == "live" and temperature > 0.5:
    st.warning("⚠️ You are in LIVE mode with a high temperature — results may vary and cost more.")

max_tokens = st.number_input(
    "Max tokens (response length)",
    min_value=256, max_value=8000, value=DEFAULT_MAX_TOKENS,
    help="Higher allows longer structured analyses and full financial tables. Costs scale with token count."
)

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

st.divider()
st.markdown("## 🔍 Latest Webhook Response")

# When the webhook responds, show structured info
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

# Webhook listener / refresh button
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("When you trigger the workflow, the response will appear here automatically if sent back from n8n.")
with col2:
    if st.button("🔄 Refresh"):
        st.session_state["last_response"] = None

# Simulate fetching (you'll replace this with actual webhook receiver logic later)
if st.session_state["last_response"]:
    data = st.session_state["last_response"]
else:
    st.info("Awaiting response from n8n...")
    data = None

if data:
    st.success("✅ Response received")

    # Top metadata
    st.subheader("Metadata")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Model", data.get("model", "—"))
    meta_cols[1].metric("Tokens", data.get("total_tokens", "—"))
    meta_cols[2].metric("Cost (USD)", f"${data.get('cost_usd', 0):.4f}")
    meta_cols[3].metric("Timestamp", data.get("timestamp", "—"))

    # Citations
    if data.get("citations"):
        with st.expander("📚 Citations"):
            for c in data["citations"]:
                st.markdown(f"- [{c}]({c})")

    # Search results (if present)
    if data.get("search_results"):
        with st.expander("🔎 Search Results"):
            for r in data["search_results"]:
                st.markdown(f"**{r.get('title','')}** — [{r.get('url','')}]({r.get('url','')})")
                st.caption(r.get("snippet", ""))

    # LLM Output
    st.subheader("🧠 Model Output")
    view_mode = st.radio("View as:", ["Parsed JSON", "Raw Text"], horizontal=True)

    if view_mode == "Parsed JSON" and "llm_output_parsed" in data:
        st.json(data["llm_output_parsed"])
    else:
        st.code(data.get("llm_output_raw", ""), language="json")

    # Advanced diagnostics
    with st.expander("⚙️ Advanced Debug Info"):
        st.write(data)

else:
    st.markdown("Once a response arrives from n8n, you'll see model outputs, citations, and costs here.")


# ============================================================
# 🪄 Optional: Webhook Receiver
# ============================================================
# If you plan to make the Streamlit app directly handle POSTs from n8n
# (e.g. in production with public URLs), you'll use st.experimental_connection
# or a small FastAPI wrapper here. For now, the app just displays data
# from st.session_state['last_response'] which n8n can POST into via REST.


# ============================================================
# 8) Webhook Response Viewer & Model Output Comparison
# ============================================================

st.markdown("## 📊 LLM Response Viewer & Comparison")
st.markdown("""
View and compare structured analysis results returned by your n8n/Perplexity webhook.

Paste one or more JSON responses below (from n8n or API test runs) to inspect:
- ✅ Clean formatted view
- 🔍 Model info, cost, and token breakdown
- 📋 Extracted node-level data
- 📈 Side-by-side comparison by model variant
""")

# JSON input area
_response_input = st.text_area(
    "Paste raw JSON response(s):",
    placeholder="Paste the full JSON array returned from n8n...",
    height=250
)

if _response_input.strip():
    import json
    try:
        data = json.loads(_response_input)
        if isinstance(data, dict):
            data = [data]

        st.success(f"✅ Parsed {len(data)} response object(s).")

        for i, item in enumerate(data):
            st.divider()
            st.markdown(f"### 🧩 Response {i+1}")

            body = item.get("body", {})
            model = body.get("model", "unknown")
            usage = body.get("usage", {})
            cost = usage.get("cost", {})
            msg = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            st.markdown(f"**Model:** `{model}` | **Tokens:** {usage.get('total_tokens','?')} | **Est. Cost:** ${cost.get('total_cost','?')}")

            # Show structured JSON content (if valid)
            try:
                structured = json.loads(msg)
                st.json(structured)
            except Exception:
                st.code(msg, language="json")

            # Optional: summary preview
            if usage:
                with st.expander("🔍 Token / Cost Details"):
                    st.write(usage)

            # Optional: evidence preview
            citations = body.get("citations", [])
            if citations:
                with st.expander("📚 Citations"):
                    for c in citations:
                        st.markdown(f"- [{c}]({c})")

        # Compare across models if >1 response
        if len(data) > 1:
            st.divider()
            st.markdown("### ⚖️ Cross-Model Comparison")
            models = [d.get("body", {}).get("model", "unknown") for d in data]
            cols = st.columns(len(data))
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"**{models[i]}**")
                    msg = (
                        data[i].get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                    )
                    st.code(msg[:1500] + "..." if len(msg) > 1500 else msg, language="json")
    except Exception as e:
        st.error(f"❌ Failed to parse JSON: {e}")


# ============================================================
# 9) Live Webhook Fetcher (Auto-Retrieval from n8n)
# ============================================================

import requests

st.divider()
st.markdown("## 🌐 Live Webhook Fetcher")
st.markdown("""
Fetch and display **live responses** directly from your n8n workflow webhook.

Enter the webhook URL once and click *Fetch Latest*.  
The app will call your n8n endpoint, parse the JSON, and render it automatically.

> 💡 *Tip:* Keep your webhook private — only use secure URLs (HTTPS, with auth if enabled).
""")

# Store webhook URL in session so you don't have to retype it
if "webhook_url" not in st.session_state:
    st.session_state["webhook_url"] = ""

webhook_url = st.text_input(
    "Webhook URL",
    value=st.session_state["webhook_url"],
    placeholder="https://fpgconsulting.app.n8n.cloud/webhook/echo_agent",
)

# Save automatically
st.session_state["webhook_url"] = webhook_url.strip()


# ============================================================
# 🔀 Test / Live Mode Toggle
# ============================================================

st.divider()
st.markdown("## 🧪 Environment Mode Switch")
st.markdown("""
Use this toggle to quickly switch between your **Test** and **Live** webhook environments.  

- **Test Mode** → lower cost, sandbox-safe environment  
- **Live Mode** → production endpoint for real research runs  

> ⚠️ Always confirm you're in the right mode before sending expensive API calls.
""")

# Default URLs — replace with your real ones
TEST_WEBHOOK_URL = "https://fpgconsulting.app.n8n.cloud/webhook-test/echo_agent"
LIVE_WEBHOOK_URL = "https://fpgconsulting.app.n8n.cloud/webhook/echo_agent"

if "env_mode" not in st.session_state:
    st.session_state["env_mode"] = "test"

# Toggle buttons
col1, col2 = st.columns([1, 4])
with col1:
    env_choice = st.radio(
        "Mode",
        ["test", "live"],
        horizontal=True,
        index=0 if st.session_state["env_mode"] == "test" else 1,
        help="Switch between sandbox and production webhook targets."
    )

# Update mode and URL
if env_choice != st.session_state["env_mode"]:
    st.session_state["env_mode"] = env_choice
    st.session_state["webhook_url"] = (
        TEST_WEBHOOK_URL if env_choice == "test" else LIVE_WEBHOOK_URL
    )

# Visual cue
mode_color = "orange" if env_choice == "test" else "red"
st.markdown(
    f"<div style='padding:8px;border-radius:8px;background-color:{mode_color};color:white;text-align:center;'>"
    f"🧭 Current Mode: <b>{env_choice.upper()}</b>"
    "</div>",
    unsafe_allow_html=True,
)

if webhook_url.strip():
    if st.button("🔄 Fetch Latest Response from n8n"):
        with st.spinner("Contacting webhook..."):
            try:
                response = requests.get(webhook_url.strip(), timeout=30)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        st.success(f"✅ Webhook responded with {len(data) if isinstance(data, list) else 1} record(s)")
                        
                        # Store for comparison functionality
                        st.session_state["latest_response"] = data
                        
                        # Reuse existing display logic
                        if isinstance(data, dict):
                            data = [data]

                        for i, item in enumerate(data):
                            st.divider()
                            st.markdown(f"### 🧩 Response {i+1}")

                            body = item.get("body", {})
                            model = body.get("model", "unknown")
                            usage = body.get("usage", {})
                            cost = usage.get("cost", {})
                            msg = (
                                body.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )

                            st.markdown(f"**Model:** `{model}` | **Tokens:** {usage.get('total_tokens','?')} | **Est. Cost:** ${cost.get('total_cost','?')}")

                            # Try to display structured JSON if possible
                            try:
                                structured = json.loads(msg)
                                st.json(structured)
                            except Exception:
                                st.code(msg, language="json")

                            citations = body.get("citations", [])
                            if citations:
                                with st.expander("📚 Citations"):
                                    for c in citations:
                                        st.markdown(f"- [{c}]({c})")

                    except Exception as parse_err:
                        st.error(f"⚠️ Could not parse JSON: {parse_err}")
                        st.text(response.text[:1000])
                else:
                    st.error(f"❌ Webhook returned status {response.status_code}")
                    st.text(response.text[:1000])

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error fetching from webhook: {e}")

else:
    st.info("Enter your n8n webhook URL above to fetch live responses automatically.")


# ============================================================
# 🌐 Live Webhook Receiver
# ============================================================

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Config ---
WEBHOOK_PORT = 8502   # you can change this if your main Streamlit is on 8501
WEBHOOK_PATH = "/n8n-webhook"

# Thread-safe storage
shared_data = {"latest": None}

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != WEBHOOK_PATH:
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            import json
            parsed = json.loads(body)
            shared_data["latest"] = parsed

            # Acknowledge
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())


# Start server thread (if not already running)
def start_webhook_server():
    server = HTTPServer(("0.0.0.0", WEBHOOK_PORT), WebhookHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://localhost:{WEBHOOK_PORT}{WEBHOOK_PATH}"

webhook_url = start_webhook_server()

st.divider()
st.markdown("## 🌐 Live Webhook Receiver")
st.info(f"Listening for POSTs at: `{webhook_url}`")

if shared_data["latest"]:
    st.success("✅ Live webhook data received!")
    st.json(shared_data["latest"])
else:
    st.warning("No webhook data yet — waiting for n8n to POST results here.")

st.caption("Use this endpoint as the `POST` target in your n8n Respond to Webhook node.")


# ============================================================
# 10) Compare Previous vs Latest Response
# ============================================================

import difflib

if "last_webhook_response" not in st.session_state:
    st.session_state["last_webhook_response"] = None

st.divider()
st.markdown("## 🧮 Compare Previous vs Latest Response")
st.markdown("""
This section compares your **latest fetched** webhook response with the **previous** one.

Useful when testing prompt refinements, model changes, or retrieval adjustments — to see what actually changed in the structure or output.
""")

if "latest_response" in st.session_state:
    prev = st.session_state.get("last_webhook_response")
    latest = st.session_state["latest_response"]

    if prev is not None:
        prev_str = json.dumps(prev, indent=2, sort_keys=True)
        latest_str = json.dumps(latest, indent=2, sort_keys=True)

        diff_lines = list(
            difflib.unified_diff(
                prev_str.splitlines(),
                latest_str.splitlines(),
                fromfile="Previous",
                tofile="Latest",
                lineterm=""
            )
        )

        if diff_lines:
            st.markdown("### 🔍 Differences Detected")
            st.code("\n".join(diff_lines[:5000]), language="diff")
            st.info("🧠 Tip: '+' means new or changed lines, '-' means removed content.")
        else:
            st.success("✅ No differences detected — output identical to previous run.")

    else:
        st.info("No previous response cached yet — fetch twice to start comparison.")

    # Always store the latest as "last" for next cycle
    st.session_state["last_webhook_response"] = st.session_state["latest_response"]

else:
    st.info("Fetch at least one webhook response to enable comparison.")