import streamlit as st
import json
import pandas as pd
import streamlit.components.v1 as components
import subprocess

st.set_page_config(page_title="Digital Public Safety — Fraud Network Intelligence", layout="wide")

st.title("Fraud Network Graph Intelligence")
st.caption("Detects coordinated fraud rings from transaction and device data — no manual rules, no labels required.")

# --- Button to (re)run detection live, for demo drama ---
col1, col2 = st.columns([1, 4])
with col1:
    run_clicked = st.button("Run fraud detection", type="primary")

if run_clicked:
    with st.spinner("Building graph and detecting clusters..."):
        subprocess.run(["python3", "detect_fraud.py"])
        subprocess.run(["python3", "visualize.py"])
    st.success("Detection complete.")

# --- Load results (works whether or not the button was just clicked) ---
try:
    with open("cluster_results.json") as f:
        results = json.load(f)
except FileNotFoundError:
    st.warning("No results yet — click 'Run fraud detection' above.")
    st.stop()

# separate the real fraud clusters from the giant normal-account cluster
suspicious = [r for r in results if r["size"] < 50]
suspicious.sort(key=lambda r: r["internal_volume"], reverse=True)

# --- Top-level metrics row ---
m1, m2, m3 = st.columns(3)
m1.metric("Fraud clusters flagged", len(suspicious))
m2.metric("Accounts flagged", sum(r["size"] for r in suspicious))
m3.metric("Total suspicious volume", f"₹{sum(r['internal_volume'] for r in suspicious):,}")

st.divider()

# --- Two-column layout: graph on the left, case list on the right ---
left, right = st.columns([2, 1])

with left:
    st.subheader("Network view")
    try:
        with open("fraud_graph.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=800, scrolling=True)
    except FileNotFoundError:
        st.info("Graph not generated yet — click 'Run fraud detection' above.")

with right:
    st.subheader("Flagged cases")
    for r in suspicious:
        risk = "High" if r["density"] > 0.7 else "Medium"
        with st.container(border=True):
            st.markdown(f"**Cluster {r['cluster_id']}** — {risk} risk")
            st.write(f"{r['size']} accounts · density {r['density']} · ₹{r['internal_volume']:,} moved internally")
            st.caption(f"Flag reason: shared device/phone across {r['size']} accounts, "
                       f"{int(r['density']*100)}% of possible internal transactions present — "
                       f"consistent with a money-mule ring, not organic activity.")
            with st.expander("View account IDs (evidence)"):
                st.code("\n".join(r["members"]))

st.divider()
st.subheader("All clusters (raw)")
st.dataframe(pd.DataFrame(results)[["cluster_id", "size", "density", "internal_volume", "true_label_majority"]])