import networkx as nx
import pandas as pd
from collections import defaultdict

from src.config import (
    TRANSACTION_EDGE_WEIGHT,
    SHARED_IDENTITY_EDGE_WEIGHT,
)


def build_graph(accounts_df, transactions_df):
    G = nx.Graph()

    optional_acct_cols = ["ip_address", "email", "account_status"]
    for _, row in accounts_df.iterrows():
        attrs = {
            "account_id": row["account_id"],
            "phone": row["phone"],
            "device_id": row["device_id"],
        }
        for col in optional_acct_cols:
            if col in accounts_df.columns and pd.notna(row.get(col)):
                attrs[col] = row[col]
        G.add_node(row["account_id"], **attrs)

    tx_edges = defaultdict(lambda: {
        "transaction_count": 0,
        "transaction_volume": 0,
        "first_transaction": None,
        "last_transaction": None,
    })

    for _, row in transactions_df.iterrows():
        a, b = row["from_account"], row["to_account"]
        key = tuple(sorted([a, b]))
        amt = row["amount"]
        ts = row["timestamp"]

        e = tx_edges[key]
        e["transaction_count"] += 1
        e["transaction_volume"] += amt
        if e["first_transaction"] is None or ts < e["first_transaction"]:
            e["first_transaction"] = ts
        if e["last_transaction"] is None or ts > e["last_transaction"]:
            e["last_transaction"] = ts

    for (a, b), meta in tx_edges.items():
        meta["relationship_types"] = {"transaction"}
        G.add_edge(a, b, **meta)

    by_phone = defaultdict(list)
    by_device = defaultdict(list)
    by_ip = defaultdict(list)
    for _, row in accounts_df.iterrows():
        by_phone[row["phone"]].append(row["account_id"])
        by_device[row["device_id"]].append(row["account_id"])
        if "ip_address" in accounts_df.columns:
            ip = row.get("ip_address")
            if isinstance(ip, str) and ip.strip():
                by_ip[ip].append(row["account_id"])

    for group in by_phone.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                _add_shared_edge(G, a, b, "shared_phone")

    for group in by_device.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                _add_shared_edge(G, a, b, "shared_device")

    for group in by_ip.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                _add_shared_edge(G, a, b, "shared_ip")

    compute_weights(G)

    return G


def _add_shared_edge(G, a, b, rel_type):
    if G.has_edge(a, b):
        types = G[a][b].get("relationship_types", set())
    else:
        types = set()
    types.add(rel_type)
    G.add_edge(a, b, relationship_types=types)


def compute_weights(G):
    for u, v, data in G.edges(data=True):
        weight = 0
        types = data.get("relationship_types", set())

        if "transaction" in types:
            weight += TRANSACTION_EDGE_WEIGHT * data.get("transaction_count", 1)
        if "shared_phone" in types:
            weight += SHARED_IDENTITY_EDGE_WEIGHT
        if "shared_device" in types:
            weight += SHARED_IDENTITY_EDGE_WEIGHT
        if "shared_ip" in types:
            weight += SHARED_IDENTITY_EDGE_WEIGHT

        G[u][v]["weight"] = weight


def save_graph(G, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)


def load_graph(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
