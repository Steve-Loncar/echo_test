# streamlit_echo_test.py
import os
import streamlit as st
import requests
import json
import time

N8N_WEBHOOK_URL = "https://fpgconsulting.app.n8n.cloud/webhook-test/echo_agent"  # <- replace if needed
# Read secret from Streamlit Secrets (recommended). Fall back to environment var, then to a local default.
try:
    # st.secrets is available only after Streamlit has been imported and initialised
    WEBHOOK_SECRET = st.secrets.get("WEBHOOK_SECRET", None)
except Exception:
    WEBHOOK_SECRET = None

if not WEBHOOK_SECRET:
    # fallback to environment variable (useful for CI or non-Streamlit hosts)
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "WIBBLE")

st.title("n8n Echo Test")

payload = {
    "test": "hello from streamlit",
    "timestamp": time.time()
}

st.write("Webhook URL:", N8N_WEBHOOK_URL)

if st.button("Send test POST to n8n"):
    try:
        headers = {"X-Webhook-Secret": WEBHOOK_SECRET, "Content-Type": "application/json"}
        r = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=15)
        st.write("Status:", r.status_code)
        try:
            st.json(r.json())
        except Exception:
            st.text(r.text)
    except Exception as e:
        st.error(f"Request failed: {e}")