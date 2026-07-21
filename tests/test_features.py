import pandas as pd
import networkx as nx
from src.features import extract_cluster_features


def _make_dense_ring():
    accounts_df = pd.DataFrame({
        "account_id": [f"A{i}" for i in range(5)],
        "phone": ["phone_shared"] * 5,
        "device_id": ["device_shared"] * 5,
        "is_fraud_ring": ["ring_1"] * 5,
    })
    txs = []
    for i in range(5):
        for j in range(i + 1, 5):
            txs.append({
                "from_account": f"A{i}",
                "to_account": f"A{j}",
                "amount": 100000,
                "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
            })
    transactions_df = pd.DataFrame(txs)
    G = nx.Graph()
    for _, r in accounts_df.iterrows():
        G.add_node(r["account_id"], account_id=r["account_id"],
                   phone=r["phone"], device_id=r["device_id"])
    for _, r in transactions_df.iterrows():
        a, b = r["from_account"], r["to_account"]
        if G.has_edge(a, b):
            G[a][b]["transaction_count"] += 1
            G[a][b]["transaction_volume"] += r["amount"]
        else:
            G.add_edge(a, b, transaction_count=1, transaction_volume=r["amount"],
                       relationship_types={"transaction"})
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return accounts_df, transactions_df, G


def _make_normal_cluster():
    accounts_df = pd.DataFrame({
        "account_id": [f"N{i}" for i in range(50)],
        "phone": [f"phone_{i}" for i in range(50)],
        "device_id": [f"device_{i}" for i in range(50)],
        "is_fraud_ring": ["no"] * 50,
    })
    txs = []
    for i in range(20):
        a, b = f"N{i}", f"N{(i+1) % 20}"
        txs.append({
            "from_account": a, "to_account": b,
            "amount": 1000, "timestamp": pd.Timestamp("2026-06-15 12:00:00"),
        })
    transactions_df = pd.DataFrame(txs)
    G = nx.Graph()
    for _, r in accounts_df.iterrows():
        G.add_node(r["account_id"], account_id=r["account_id"],
                   phone=r["phone"], device_id=r["device_id"])
    for _, r in transactions_df.iterrows():
        a, b = r["from_account"], r["to_account"]
        if G.has_edge(a, b):
            G[a][b]["transaction_count"] += 1
            G[a][b]["transaction_volume"] += r["amount"]
        else:
            G.add_edge(a, b, transaction_count=1, transaction_volume=r["amount"],
                       relationship_types={"transaction"})
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return accounts_df, transactions_df, G


class TestExtractClusterFeatures:
    def test_dense_ring_high_identity_reuse(self):
        accounts_df, transactions_df, G = _make_dense_ring()
        members = {"A0", "A1", "A2", "A3", "A4"}
        f = extract_cluster_features(members, G, accounts_df, transactions_df)

        assert f["cluster_size"] == 5
        assert f["network_density"] == 1.0
        assert f["identity_reuse_ratio"] == 1.0
        assert f["shared_phone_count"] > 0
        assert f["shared_device_count"] > 0

    def test_normal_cluster_low_identity_reuse(self):
        accounts_df, transactions_df, G = _make_normal_cluster()
        members = {f"N{i}" for i in range(50)}
        f = extract_cluster_features(members, G, accounts_df, transactions_df)

        assert f["cluster_size"] == 50
        assert f["identity_reuse_ratio"] == 0.0
        assert f["shared_phone_count"] == 0
        assert f["shared_device_count"] == 0

    def test_zero_transactions_no_crash(self):
        accounts_df = pd.DataFrame({
            "account_id": ["X1", "X2", "X3"],
            "phone": ["p1", "p2", "p3"],
            "device_id": ["d1", "d2", "d3"],
            "is_fraud_ring": ["no"] * 3,
        })
        transactions_df = pd.DataFrame(columns=[
            "from_account", "to_account", "amount", "timestamp"
        ])
        G = nx.Graph()
        for _, r in accounts_df.iterrows():
            G.add_node(r["account_id"], account_id=r["account_id"],
                       phone=r["phone"], device_id=r["device_id"])
        members = {"X1", "X2", "X3"}
        f = extract_cluster_features(members, G, accounts_df, transactions_df)

        assert f["internal_transaction_volume"] == 0
        assert f["internal_transaction_count"] == 0
        assert f["total_transaction_volume"] == 0
        assert f["transactions_per_hour"] == 0.0
        assert f["transactions_per_day"] == 0.0

    def test_ground_truth_absent_from_features(self):
        accounts_df, transactions_df, G = _make_dense_ring()
        members = {"A0", "A1", "A2", "A3", "A4"}
        f = extract_cluster_features(members, G, accounts_df, transactions_df)

        assert "is_fraud_ring" not in f
        assert "true_label_majority" not in f
        assert "ring_1" not in str(f)

    def test_empty_cluster_no_crash(self):
        accounts_df = pd.DataFrame(columns=[
            "account_id", "phone", "device_id", "is_fraud_ring"
        ])
        transactions_df = pd.DataFrame(columns=[
            "from_account", "to_account", "amount", "timestamp"
        ])
        G = nx.Graph()
        f = extract_cluster_features(set(), G, accounts_df, transactions_df)

        assert f["cluster_size"] == 0
        assert f["internal_transaction_count"] == 0

    def test_external_flows_detected(self):
        accounts_df = pd.DataFrame({
            "account_id": ["A0", "A1", "B0"],
            "phone": ["pa", "pa", "pb"],
            "device_id": ["da", "da", "db"],
            "is_fraud_ring": ["ring_1", "ring_1", "no"],
        })
        txs = [
            {"from_account": "A0", "to_account": "A1", "amount": 500,
             "timestamp": pd.Timestamp("2026-01-01")},
            {"from_account": "B0", "to_account": "A0", "amount": 300,
             "timestamp": pd.Timestamp("2026-01-02")},
            {"from_account": "A1", "to_account": "B0", "amount": 200,
             "timestamp": pd.Timestamp("2026-01-03")},
        ]
        transactions_df = pd.DataFrame(txs)
        G = nx.Graph()
        for _, r in accounts_df.iterrows():
            G.add_node(r["account_id"], account_id=r["account_id"],
                       phone=r["phone"], device_id=r["device_id"])
        for _, r in transactions_df.iterrows():
            a, b = r["from_account"], r["to_account"]
            key = tuple(sorted([a, b]))
            if G.has_edge(*key):
                G[key[0]][key[1]]["transaction_count"] += 1
                G[key[0]][key[1]]["transaction_volume"] += r["amount"]
            else:
                G.add_edge(key[0], key[1], transaction_count=1,
                           transaction_volume=r["amount"],
                           relationship_types={"transaction"})
        for u, v in G.edges():
            G[u][v]["weight"] = 1

        members = {"A0", "A1"}
        f = extract_cluster_features(members, G, accounts_df, transactions_df)

        assert f["internal_transaction_volume"] == 500
        assert f["external_inflow"] == 300
        assert f["external_outflow"] == 200
        assert f["unique_external_counterparties"] == 1
