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
query_depth = st.slider("Search depth / results", min_value=1, max_value=10, value=3)
default_fields = ["summary", "market_size", "players", "sources", "brief", "confidence"]
required_fields = st.multiselect("Fields to request from the LLM", default_fields, default=["summary", "sources"])
extra_context = st.text_area("Extra context / constraints (optional)", height=120)

#
# 3) Buttons to run query or write to dataset
#
if st.button("Run query"):

    if not node_choice:
        st.error("No taxonomy node selected. Upload an Excel and select a node first.")
    else:
        payload = {
            "taxonomy_node": node_choice,
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