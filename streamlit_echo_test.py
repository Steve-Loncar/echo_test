# streamlit_echo_test.py
import os
import streamlit as st
import pandas as pd
import requests
import time

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

# === Helper function for taxonomy hierarchy ===

# Helper: compute descendant nodes up to relative depth using a normalized df  
def compute_nodes_to_query(df_tax, selected_path_or_name, rel_depth):  
    """  
    df_tax: normalized DataFrame with at least columns ['node_id','parent_id','level','path','display_name']  
    selected_path_or_name: string (either path or display_name)  
    rel_depth: int (0 => only selected node, 1 => include immediate children, etc)  
    Returns: list of dicts: [{node_id, path, display_name, level, parent_id}, ...]  
    """  
    # try matching by path first, then display_name  
    if 'path' in df_tax.columns and selected_path_or_name in df_tax['path'].values:  
        root = df_tax.loc[df_tax['path'] == selected_path_or_name].iloc[0]  
    elif 'display_name' in df_tax.columns and selected_path_or_name in df_tax['display_name'].values:  
        root = df_tax.loc[df_tax['display_name'] == selected_path_or_name].iloc[0]  
    else:  
        # fallback: pick first row where display_name contains substring  
        matches = df_tax[df_tax['display_name'].str.contains(selected_path_or_name, na=False, case=False)]  
        if len(matches):  
            root = matches.iloc[0]  
        else:  
            raise ValueError("Selected node not found in taxonomy dataframe")  
  
    root_id = root['node_id'] if 'node_id' in root else None  
    root_level = int(root['level']) if 'level' in root and not pd.isna(root['level']) else None  
  
    # Build parent->children map  
    children_map = {}  
    if 'parent_id' in df_tax.columns:  
        for _, r in df_tax.iterrows():  
            pid = r['parent_id']  
            children_map.setdefault(pid, []).append(r['node_id'])  
    else:  
        # fall back: if no parent_id, match by path prefix  
        children_map = None  
  
    # BFS starting at root to collect nodes up to rel_depth  
    results = []  
    if children_map:  
        from collections import deque  
        q = deque()  
        q.append((root['node_id'], 0))  
        visited = set()  
        id_to_row = {r['node_id']: r for _, r in df_tax.iterrows()}  
  
        while q:  
            current_id, depth = q.popleft()  
            if current_id in visited:  
                continue  
            visited.add(current_id)  
            row = id_to_row.get(current_id)  
            if row is None:  
                continue  
            results.append({  
                "node_id": row["node_id"],  
                "path": row.get("path"),  
                "display_name": row.get("display_name"),  
                "level": int(row.get("level")) if pd.notna(row.get("level")) else None,  
                "parent_id": row.get("parent_id")  
            })  
            if depth < rel_depth:  
                for child_id in children_map.get(current_id, []):  
                    q.append((child_id, depth + 1))  
    else:  
        # fallback by path prefix  
        prefix = root['path']  
        matched = df_tax[df_tax['path'].str.startswith(prefix, na=False)]  
        # compute depth filter relative to root (count separators)  
        root_depth = prefix.count(" > ")  
        def rel_d(row):  
            return row['path'].count(" > ") - root_depth  
        matched = matched[matched.apply(lambda r: rel_d(r) <= rel_depth, axis=1)]  
        for _, row in matched.iterrows():  
            results.append({  
                "node_id": row.get("node_id"),  
                "path": row.get("path"),  
                "display_name": row.get("display_name"),  
                "level": int(row.get("level")) if pd.notna(row.get("level")) else None,  
                "parent_id": row.get("parent_id")  
            })  
  
    return results

st.title("n8n Echo / Query Test")

st.markdown("Upload your taxonomy Excel, pick a node, set query options, and POST to n8n.")

#
# 1) Upload taxonomy Excel and choose column to use as node labels
#
uploaded = st.file_uploader("Upload taxonomy Excel (.xlsx)", type=["xlsx"])
node_choice = None

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
        sample_vals = df[selected_col].dropna().astype(str).unique().tolist()
        # If the taxonomy is hierarchical and stored across multiple columns you might want to preprocess externally.
        node_choice = st.selectbox("Choose taxonomy node", sample_vals)
else:
    st.info("Upload an Excel file to select taxonomy nodes. A default node will be used for testing.")
    # Allow a simple editable default when no Excel is provided.
    node_choice = st.text_input("Default taxonomy node (used when no Excel uploaded):", value="DEFAULT_NODE")

#
# 2) Query options
#
st.subheader("Query options")
# query_depth remains for LLM search depth / results (unchanged)
query_depth = st.slider("Search depth / results", min_value=1, max_value=10, value=3)
default_fields = ["summary", "market_size", "players", "sources", "brief", "confidence"]
required_fields = st.multiselect("Fields to request from the LLM", default_fields, default=["summary", "sources"])
extra_context = st.text_area("Extra context / constraints (optional)", height=120)

# New: analysis granularity controls
st.markdown("### Analysis granularity")
gran_choice = st.radio("Choose granularity mode", ["Selected node only", "Include children (relative depth)"], index=1)
if gran_choice == "Include children (relative depth)":
    rel_depth = st.slider("Include children up to depth (0 = only selected node)", min_value=0, max_value=5, value=0)
else:
    rel_depth = 0

#
# 3) Buttons to run query or write to dataset
#
if st.button("Run query"):

    if not node_choice:
        st.error("No taxonomy node selected. Upload an Excel and select a node first.")
    else:
        # Build nodes_to_query (list of dicts). If taxonomy DataFrame present, compute descendants.
        nodes_to_query = []
        # If df exists (uploaded), compute descendants using helper; otherwise use the single node_choice.
        if 'df' in locals() and isinstance(df, pd.DataFrame):
            # We'll call helper function (defined below) to return list of node dicts
            try:
                nodes_to_query = compute_nodes_to_query(df, node_choice, rel_depth)
            except Exception as e:
                st.error(f"Failed to compute node expansion: {e}")
                nodes_to_query = [{"path": node_choice, "display_name": node_choice, "level": None}]
        else:
            nodes_to_query = [{"path": node_choice, "display_name": node_choice, "level": None}]

        # inside the "Run query" button handler (after compute_nodes_to_query)
        payload = {
            "taxonomy_node": node_choice,
            "nodes_to_query": nodes_to_query,        # list of dicts from compute_nodes_to_query(...)
            "rel_depth": int(rel_depth),
            "query_depth": int(query_depth),
            "required_fields": required_fields,
            "extra_context": extra_context,
            "client_timestamp": time.time()
        }

        headers = {
            "X-Webhook-Secret": WEBHOOK_SECRET,
            "Content-Type": "application/json"
        }
      
        try:
            resp = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=60)
            st.write("Status:", resp.status_code)
            try:
                j = resp.json()
                st.subheader("Response (JSON)")
                st.json(j)
                # If LLM returns rows array, show as table
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
        write_payload = {
            "taxonomy_node": node_choice,
            "data": {"example_field": "value"},
            "client_timestamp": time.time()
        }
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