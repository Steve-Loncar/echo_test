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

# 2) Query options
st.subheader("Query options")
# Simplified UI: remove the search-depth slider and explicit fields selector.
DEFAULT_REQUIRED_FIELDS = ["summary", "sources"]
DEFAULT_QUERY_DEPTH = 3
extra_context = st.text_area("Extra context / constraints (optional)", height=120)

st.markdown("### Prompt editor")
st.markdown("The prompt below is pre-filled from the selected node. Edit it to refine what you send to the LLM.")

default_prompt = f"""You are a strict JSON-outputting market analyst. Return exactly one JSON object and nothing else.
Schema:
{{ "node_id": string|null, "path": string, "display_name": string, "requested_fields": array[string], "results": object, "notes": string|null, "confidence": number, "timestamp": string }}
Context:
- Node display_name: {node_choice}
- Node path: {node_choice}
- Requested fields: {DEFAULT_REQUIRED_FIELDS}
- Extra context: {{extra_context}}
Rules:
- Output exactly one JSON object and nothing else.
- Use null for missing scalars and [] for missing arrays.
"""

prompt_text = st.text_area("Prompt (editable)", value=default_prompt, height=260)

# 3) Buttons to run query or write to dataset
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
            "client_timestamp": time.time()
        }

        headers = {"X-Webhook-Secret": WEBHOOK_SECRET, "Content-Type": "application/json"}

        try:
            resp = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=60)
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

# Optional: write to dataset flow (fires to a second webhook)
if st.button("Write to dataset (example)"):
    if not node_choice:
        st.error("No taxonomy node selected.")
    else:
        write_payload = {"taxonomy_node": node_choice, "data": {"example_field": "value"}, "client_timestamp": time.time()}
        headers = {"X-Webhook-Secret": WEBHOOK_SECRET, "Content-Type": "application/json"}
        try:
            wr = requests.post(N8N_WRITE_URL, json=write_payload, headers=headers, timeout=30)
            st.write("Write status:", wr.status_code)
            try:
                st.json(wr.json())
            except Exception:
                st.text(wr.text)
        except Exception as e:
            st.error(f"Write request failed: {e}")