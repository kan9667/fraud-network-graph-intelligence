"""Reusable cluster / neighborhood visualization for the investigator UI."""

from __future__ import annotations

import tempfile
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from src.dashboard_data import ROLE_COLORS, ROLE_DISPLAY


def _role_for(node, role_by_account):
    return role_by_account.get(node, "unknown")


def build_cluster_network_html(
    G,
    members,
    role_by_account=None,
    height="620px",
    physics=True,
):
    """Build interactive PyVis HTML for a cluster subgraph, colored by role."""
    if role_by_account is None:
        role_by_account = {}
    members = set(members)
    # Include 1-hop external counterparties for context (dimmed)
    nodes = set(members)
    for m in list(members):
        if m in G:
            nodes.update(G.neighbors(m))

    sub = G.subgraph(nodes).copy()
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0f1419",
        font_color="#ecf0f1",
        directed=False,
    )
    if physics:
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=120,
            spring_strength=0.02,
            damping=0.5,
        )
    else:
        net.toggle_physics(False)

    for node in sub.nodes():
        in_cluster = node in members
        role = _role_for(node, role_by_account) if in_cluster else "unknown"
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        if not in_cluster:
            color = "#4a5568"
            size = 12
            title = f"{node} (external counterparty)"
            label = ""
        else:
            size = 22 if role != "unknown" else 16
            role_label = ROLE_DISPLAY.get(role, role)
            title = f"{node}\n{role_label}"
            label = node[-4:] if len(node) > 4 else node

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=2 if in_cluster else 1,
        )

    for u, v, data in sub.edges(data=True):
        weight = data.get("weight", 1)
        vol = data.get("transaction_volume", 0)
        title = f"{u} — {v}"
        if vol:
            title += f"\nVolume: ₹{int(vol):,}"
        if data.get("transaction_count"):
            title += f"\nTx count: {data['transaction_count']}"
        rel = data.get("relationship_types")
        if rel:
            title += f"\nRelations: {', '.join(sorted(rel)) if isinstance(rel, (set, list)) else rel}"
        width = min(8, 1 + float(weight) / 3)
        both_in = u in members and v in members
        edge_color = "#5dade2" if both_in else "#566573"
        net.add_edge(u, v, value=weight, title=title, color=edge_color, width=width)

    return _net_to_html(net)


def build_neighborhood_html(G, account_id, role_by_account=None, height="420px"):
    """Immediate neighborhood of an account."""
    if role_by_account is None:
        role_by_account = {}
    if account_id not in G:
        return "<html><body style='background:#0f1419;color:#aaa;padding:2rem'>Account not in graph.</body></html>"

    nodes = {account_id} | set(G.neighbors(account_id))
    # 2nd hop lightly
    for n in list(G.neighbors(account_id)):
        nodes.update(list(G.neighbors(n))[:8])

    sub = G.subgraph(nodes).copy()
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0f1419",
        font_color="#ecf0f1",
        directed=False,
    )
    net.barnes_hut(gravity=-5000, central_gravity=0.4, spring_length=90)

    for node in sub.nodes():
        role = role_by_account.get(node, "unknown")
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        size = 28 if node == account_id else 16
        border = 3 if node == account_id else 1
        title = f"{node}\n{ROLE_DISPLAY.get(role, role)}"
        if node == account_id:
            title = f"FOCUS: {title}"
        net.add_node(
            node,
            label=node[-4:] if len(str(node)) > 4 else str(node),
            title=title,
            color=color,
            size=size,
            borderWidth=border,
        )

    for u, v, data in sub.edges(data=True):
        net.add_edge(
            u,
            v,
            value=data.get("weight", 1),
            color="#5dade2",
            title=f"weight={data.get('weight', 1)}",
        )

    return _net_to_html(net)


def _net_to_html(net: Network) -> str:
    """Write PyVis network to a temp file and return HTML string."""
    # pyvis write_html needs a path; use temp file
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        path = tmp.name
    try:
        net.write_html(path, open_browser=False, notebook=False)
        return Path(path).read_text(encoding="utf-8")
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def role_legend_markdown() -> str:
    parts = []
    for role, color in ROLE_COLORS.items():
        label = ROLE_DISPLAY.get(role, role)
        parts.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:2px;margin-right:6px;"></span>'
            f'{label}'
        )
    return " &nbsp;&nbsp; ".join(parts)
