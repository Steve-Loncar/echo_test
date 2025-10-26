# streamlit_echo_test.py
import streamlit as st
import requests
import json
import time

N8N_WEBHOOK_URL = "https://fpgconsulting.app.n8n.cloud/webhook/echo_agent"  # <- replace
WEBHOOK_SECRET = "WIBBLE"  # <- must match the Function node expectedSecret

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
        except:
            st.text(r.text)
    except Exception as e:
        st.error(f"Request failed: {e}")