# streamlit_echo_test.py
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
uploaded = st.file_uploader("Upload taxonomy Excel (.xlsx)", type=["xlsx"])
node_choice = None
df = None

if uploaded:
    try:
        df = pd.read_excel(uploaded, sheet_name=0)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        df = None

if df is not None:
    st.write("Detected columns:", list(df.columns))
    # try to pick a sensible default column
    candidate_cols = [c for c in df.columns if c and str(c).lower() in ("hierarchy", "node", "node_path", "name", "title")]
    default_col = candidate_cols[0] if candidate_cols else df.columns[0]
    selected_col = st.selectbox("Which column contains the taxonomy node labels?", df.columns, index=list(df.columns).index(default_col))
    # show sample values
    try:
        sample_vals = df[selected_col].dropna().astype(str).unique().tolist()
    except Exception:
        sample_vals = df[selected_col].dropna().astype(str).head(100).tolist()
    node_choice = st.selectbox("Choose taxonomy node", sample_vals)
else:
    st.info("Upload an Excel file to select taxonomy nodes. A default node will be used for testing.")
    node_choice = st.text_input("Default taxonomy node (used when no Excel uploaded):", value="DEFAULT_NODE")

# === Query options / Prompt editor / Run ===
st.subheader("Query options")

# Hidden defaults used by the workflow
DEFAULT_REQUIRED_FIELDS = ["summary", "sources"]
DEFAULT_QUERY_DEPTH = 3

# Extra context box (explicit, separate)
extra_context = st.text_area("Extra context / constraints (optional)", height=100)

# Perplexity model selector (simple + 'Other' option)
st.markdown("### LLM model & settings")
_model_choice = st.selectbox("Perplexity model (choose or pick Other to type)", ["perplexity (default)", "Other..."], index=0)
if _model_choice == "Other...":
    model_name = st.text_input("Perplexity model name", value="perplexity")
else:
    model_name = "perplexity"

temperature = st.slider("Temperature (determinism)", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                        help="Lower = more deterministic. Use 0 for structured JSON outputs.")
max_tokens = st.number_input("Max tokens", min_value=256, max_value=8000, value=1500,
                             help="Max tokens to request from the model. Increase if you expect long outputs.")

include_retrieval = st.checkbox("Include retrieval (top-k docs) before sending to LLM", value=False,
                                help="If enabled, n8n should run a retriever and include retrieved_docs in the prompt.")
priority = st.selectbox("Priority", ["normal", "high", "low"], index=0,
                        help="Tag the job priority; can be used by downstream schedulers or model routing.")

st.markdown("### Prompt editor")
st.markdown("The prompt below is pre-filled with a strict JSON schema for the LLM. Edit it to refine what you send to the model. Use placeholders: {{display_name}}, {{path}}, {{node_id}}, {{required_fields}}, {{extra_context}}, {{query_depth}}")

default_prompt = """You are a strict JSON-outputting market analyst. Return exactly ONE JSON OBJECT and nothing else (no commentary, no explanation, no code fences).

Schema (MUST be followed exactly):
{
  "node_id": string | null,
  "path": string,
  "display_name": string,
  "level": integer | null,
  "requested_fields": array[string],
  "results": object,
  "evidence": [
    {"evidence_id": string, "title": string|null, "url": string|null, "type": string|null, "snippet": string|null, "confidence_score": number}
  ],
  "assertions": [
    {"claim": string, "supported_by": [ "evidence_id", ... ], "confidence": number}
  ],
  "notes": string | null,
  "confidence": number,
  "timestamp": string
}

Context and placeholders you MUST use:
- Node display_name: {{display_name}}
- Node path: {{path}}
- Node id: {{node_id}}
- Requested fields: {{required_fields}}
- Extra context: {{extra_context}}
- Query depth: {{query_depth}}
- Retrieved documents (if provided): use only the objects in retrieved_docs; each has id,title,url,snippet,text.

Rules (MANDATORY):
1) Output EXACTLY ONE JSON OBJECT conforming to the schema above, nothing else.
2) For every item in requested_fields include a corresponding key in results. If unknown, set scalar -> null, arrays -> [].
3) Use retrieved_docs ONLY when supplied. Do not invent URLs. If you assert a source without URL, set url=null and confidence_score <= 0.3.
4) Evidence objects must reference the retrieved_docs ids where applicable; if evidence is generated, use a new evidence_id and set confidence_score <= 0.5.
5) All confidence scores must be between 0.0 and 1.0.
6) timestamp MUST be ISO-8601 UTC (e.g., 2025-10-27T15:21:00Z).
7) Keep evidence array <= 10 items and order by confidence_score descending.
8) If retrieval is empty/not provided, set overall confidence <= 0.5 and explain in notes: "No retrieval evidence provided — answers are model-derived and must be verified."
9) Include units for numeric estimates or companion keys (e.g., market_size_usd + market_size_usd_units).
10) Do not include any text outside the JSON object.

END.
"""

prompt_text = st.text_area("Prompt (editable)", value=default_prompt, height=320)

# Run / Preview buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Preview merged prompt"):
        # Show a simple preview of how placeholders will look when merged with the selected node (best-effort)
        display_preview = default_prompt.replace("{{display_name}}", str(node_choice)).replace("{{path}}", str(node_choice)).replace("{{required_fields}}", str(DEFAULT_REQUIRED_FIELDS)).replace("{{extra_context}}", extra_context).replace("{{query_depth}}", str(DEFAULT_QUERY_DEPTH))
        st.subheader("Merged prompt preview")
        st.code(display_preview)
with col2:
    if st.button("Run query"):
        if not node_choice:
            st.error("No taxonomy node selected. Upload an Excel and select a node first.")
        else:
            single_node = {
                "node_id": "N00001",
                "path": str(node_choice),
                "display_name": str(node_choice),
                "level": None,
                "parent_id": None
            }

            payload = {
                "taxonomy_node": node_choice,
                "nodes_to_query": [single_node],
                "rel_depth": 0,
                "query_depth": int(DEFAULT_QUERY_DEPTH),
                "required_fields": DEFAULT_REQUIRED_FIELDS,
                "extra_context": extra_context,
                "prompt_text": prompt_text,
                "model_name": model_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "include_retrieval": bool(include_retrieval),
                "priority": priority,
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