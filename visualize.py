"""CLI visualization of the full fraud graph (legacy entrypoint).

Prefer the Streamlit Command Center (app.py) for investigation.
This script remains for offline HTML export of the full network.
"""

import json
from pathlib import Path

from pyvis.network import Network

from src.config import (
    BASE_DIR,
    OUTPUT_DIR,
    CLUSTER_RESULTS_FILE,
    GRAPH_FILE,
    FRAUD_GRAPH_HTML,
)
from src.graph_builder import load_graph
from src.dashboard_data import ROLE_COLORS, RISK_COLORS, strip_forbidden

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    results_path = CLUSTER_RESULTS_FILE
    if not results_path.exists():
        results_path = BASE_DIR / "cluster_results.json"
    graph_path = GRAPH_FILE
    if not graph_path.exists():
        raise SystemExit(f"Graph not found at {GRAPH_FILE}. Run detect_fraud.py first.")

    with open(results_path) as f:
        cluster_results = strip_forbidden(json.load(f))

    G = load_graph(graph_path)

    # Map accounts -> highest-risk case and role
    account_meta = {}
    for cluster in cluster_results:
        role_map = {
            p["account_id"]: p.get("probable_role", "unknown")
            for p in cluster.get("account_profiles", [])
        }
        for member in cluster.get("members", []):
            prev = account_meta.get(member)
            score = float(cluster.get("risk_score", 0) or 0)
            if prev is None or score > prev["risk_score"]:
                account_meta[member] = {
                    "cluster_id": cluster["cluster_id"],
                    "risk_score": score,
                    "risk_level": cluster.get("risk_level", "LOW"),
                    "role": role_map.get(member, "unknown"),
                }

    net = Network(height="800px", width="100%", bgcolor="#0f1419", font_color="white")
    net.barnes_hut()

    for node in G.nodes():
        meta = account_meta.get(node)
        if meta and str(meta["risk_level"]).upper() in {"MEDIUM", "HIGH", "CRITICAL"}:
            color = ROLE_COLORS.get(meta["role"], ROLE_COLORS["unknown"])
            size = 22
            title = (
                f"{node} — CASE-{int(meta['cluster_id']):03d} "
                f"({meta['risk_level']}) · {meta['role']}"
            )
        elif meta:
            color = "#566573"
            size = 12
            title = f"{node} — CASE-{int(meta['cluster_id']):03d} (LOW)"
        else:
            color = "#3d4f5f"
            size = 8
            title = f"{node}"
        net.add_node(node, label="", title=title, color=color, size=size)

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1))

    out = FRAUD_GRAPH_HTML
    net.write_html(str(out), open_browser=False, notebook=False)
    # compatibility copy at repo root
    root_html = BASE_DIR / "fraud_graph.html"
    root_html.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved visualization to {out}")
    print(f"Saved compatibility copy to {root_html}")


if __name__ == "__main__":
    main()
