import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


def extract_cluster_features(cluster, G, accounts_df, transactions_df):
    members = set(cluster)
    subgraph = G.subgraph(members)

    features = {}
    features.update(_network_features(subgraph, G))
    features.update(_identity_features(members, accounts_df))
    features.update(_transaction_features(members, transactions_df))
    features.update(_temporal_features(members, transactions_df))
    features.update(_money_flow_features(members, transactions_df))
    features.update(_centrality_features(subgraph))
    return features


def _network_features(subgraph, full_graph):
    n = subgraph.number_of_nodes()
    m = subgraph.number_of_edges()
    density = nx.density(subgraph) if n > 1 else 0.0
    avg_deg = (2 * m) / n if n > 0 else 0.0
    weighted_deg = sum(
        d.get("weight", 1) for _, _, d in subgraph.edges(data=True)
    ) / n if n > 0 else 0.0
    avg_cc = nx.average_clustering(subgraph, weight="weight") if n > 2 else 0.0
    components = nx.number_connected_components(subgraph)

    return {
        "cluster_size": n,
        "network_density": round(density, 4),
        "number_of_edges": m,
        "average_degree": round(avg_deg, 4),
        "weighted_degree": round(weighted_deg, 4),
        "average_clustering_coefficient": round(avg_cc, 4),
        "connected_components": components,
    }


def _identity_features(members, accounts_df):
    cluster_accts = accounts_df[accounts_df["account_id"].isin(members)]

    phone_groups = cluster_accts.groupby("phone")["account_id"].apply(set)
    shared_phone_groups = phone_groups[phone_groups.apply(len) > 1]
    shared_phone_count = int(shared_phone_groups.apply(
        lambda g: len(g) * (len(g) - 1) / 2
    ).sum()) if len(shared_phone_groups) > 0 else 0
    accounts_sharing_phone = int(sum(len(g) for g in shared_phone_groups))

    device_groups = cluster_accts.groupby("device_id")["account_id"].apply(set)
    shared_device_groups = device_groups[device_groups.apply(len) > 1]
    shared_device_count = int(shared_device_groups.apply(
        lambda g: len(g) * (len(g) - 1) / 2
    ).sum()) if len(shared_device_groups) > 0 else 0
    accounts_sharing_device = int(sum(len(g) for g in shared_device_groups))

    total = len(members)
    identity_reuse_ratio = round(
        max(accounts_sharing_phone, accounts_sharing_device) / total
        if total > 0 else 0, 4
    )

    return {
        "shared_phone_count": shared_phone_count,
        "shared_device_count": shared_device_count,
        "accounts_sharing_phone": accounts_sharing_phone,
        "accounts_sharing_device": accounts_sharing_device,
        "identity_reuse_ratio": identity_reuse_ratio,
    }


def _transaction_features(members, transactions_df):
    internal = _internal_tx(members, transactions_df)
    external_from = _external_inflow_tx(members, transactions_df)
    external_to = _external_outflow_tx(members, transactions_df)

    internal_count = len(internal)
    internal_volume = int(internal["amount"].sum()) if internal_count > 0 else 0
    external_inflow = int(external_from["amount"].sum()) if len(external_from) > 0 else 0
    external_outflow = int(external_to["amount"].sum()) if len(external_to) > 0 else 0
    total_volume = internal_volume + external_inflow + external_outflow

    unique_ext = set(external_from["from_account"].tolist() + external_to["to_account"].tolist())
    unique_external_counterparties = len(unique_ext)

    all_cluster_tx = pd.concat([internal, external_from, external_to], ignore_index=True)
    if len(all_cluster_tx) > 0:
        avg_amt = float(all_cluster_tx["amount"].mean())
        med_amt = float(all_cluster_tx["amount"].median())
        max_amt = float(all_cluster_tx["amount"].max())
    else:
        avg_amt = med_amt = max_amt = 0.0

    return {
        "internal_transaction_count": internal_count,
        "internal_transaction_volume": internal_volume,
        "external_inflow": external_inflow,
        "external_outflow": external_outflow,
        "total_transaction_volume": total_volume,
        "unique_external_counterparties": unique_external_counterparties,
        "average_transaction_amount": round(avg_amt, 2),
        "median_transaction_amount": round(med_amt, 2),
        "max_transaction_amount": round(max_amt, 2),
    }


def _temporal_features(members, transactions_df):
    all_tx = _all_cluster_tx(members, transactions_df)
    if len(all_tx) == 0:
        return {
            "first_transaction": None,
            "last_transaction": None,
            "active_duration_hours": 0.0,
            "transactions_per_hour": 0.0,
            "transactions_per_day": 0.0,
            "burstiness_score": 0.0,
        }

    timestamps = all_tx["timestamp"].sort_values()
    first_ts = timestamps.iloc[0]
    last_ts = timestamps.iloc[-1]
    duration_h = (last_ts - first_ts).total_seconds() / 3600.0
    duration_h = max(duration_h, 1e-6)

    n_tx = len(timestamps)
    tx_per_hour = round(n_tx / duration_h, 4)
    tx_per_day = round(n_tx / (duration_h / 24.0), 4)

    if n_tx >= 3:
        gaps = np.diff(timestamps.astype(np.int64) // 10**9)
        mean_gap = gaps.mean() if gaps.sum() > 0 else 1e-6
        std_gap = gaps.std()
        burstiness = round(std_gap / mean_gap, 4) if mean_gap > 0 else 0.0
    else:
        burstiness = 0.0

    return {
        "first_transaction": first_ts.isoformat() if pd.notna(first_ts) else None,
        "last_transaction": last_ts.isoformat() if pd.notna(last_ts) else None,
        "active_duration_hours": round(duration_h, 4),
        "transactions_per_hour": tx_per_hour,
        "transactions_per_day": tx_per_day,
        "burstiness_score": burstiness,
    }


def _money_flow_features(members, transactions_df):
    internal = _internal_tx(members, transactions_df)
    external_from = _external_inflow_tx(members, transactions_df)
    external_to = _external_outflow_tx(members, transactions_df)

    int_vol = int(internal["amount"].sum()) if len(internal) > 0 else 0
    ext_in = int(external_from["amount"].sum()) if len(external_from) > 0 else 0
    ext_out = int(external_to["amount"].sum()) if len(external_to) > 0 else 0
    total = int_vol + ext_in + ext_out

    internal_flow_ratio = round(int_vol / total, 4) if total > 0 else 0.0
    external_inflow_ratio = round(ext_in / total, 4) if total > 0 else 0.0
    external_outflow_ratio = round(ext_out / total, 4) if total > 0 else 0.0
    rapid_forwarding_ratio = round(
        min(ext_in, ext_out) / total if total > 0 else 0.0, 4
    )

    return {
        "internal_flow_ratio": internal_flow_ratio,
        "external_inflow_ratio": external_inflow_ratio,
        "external_outflow_ratio": external_outflow_ratio,
        "rapid_forwarding_ratio": rapid_forwarding_ratio,
    }


def _centrality_features(subgraph):
    n = subgraph.number_of_nodes()
    if n == 0:
        return {
            "average_degree_centrality": 0.0,
            "max_betweenness_centrality": 0.0,
            "number_of_high_centrality_nodes": 0,
        }

    deg_cent = nx.degree_centrality(subgraph)
    avg_deg_cent = round(sum(deg_cent.values()) / n, 4)

    if n > 1:
        components = list(nx.connected_components(subgraph))
        if len(components) == 1:
            betw = nx.betweenness_centrality(subgraph, weight="weight", normalized=True)
        else:
            betw = {}
            for comp in components:
                if len(comp) >= 2:
                    sg = subgraph.subgraph(comp)
                    cb = nx.betweenness_centrality(sg, weight="weight", normalized=True)
                    betw.update(cb)
                else:
                    node = next(iter(comp))
                    betw[node] = 0.0
        max_betw = round(max(betw.values()), 4) if betw else 0.0
        high_cent = sum(1 for v in betw.values() if v > 0.5) if betw else 0
    else:
        max_betw = 0.0
        high_cent = 0

    return {
        "average_degree_centrality": avg_deg_cent,
        "max_betweenness_centrality": max_betw,
        "number_of_high_centrality_nodes": high_cent,
    }


def _internal_tx(members, transactions_df):
    return transactions_df[
        transactions_df["from_account"].isin(members)
        & transactions_df["to_account"].isin(members)
    ]


def _external_inflow_tx(members, transactions_df):
    return transactions_df[
        ~transactions_df["from_account"].isin(members)
        & transactions_df["to_account"].isin(members)
    ]


def _external_outflow_tx(members, transactions_df):
    return transactions_df[
        transactions_df["from_account"].isin(members)
        & ~transactions_df["to_account"].isin(members)
    ]


def _all_cluster_tx(members, transactions_df):
    return transactions_df[
        transactions_df["from_account"].isin(members)
        | transactions_df["to_account"].isin(members)
    ]
