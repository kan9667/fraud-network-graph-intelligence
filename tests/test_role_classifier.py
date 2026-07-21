import pandas as pd
import networkx as nx
from src.role_classifier import classify_cluster_roles


def _build_graph(accounts_df, transactions_df, extra_nodes=None):
    G = nx.Graph()
    for _, r in accounts_df.iterrows():
        G.add_node(
            r["account_id"],
            account_id=r["account_id"],
            phone=r["phone"],
            device_id=r["device_id"],
        )
    if extra_nodes:
        for n in extra_nodes:
            G.add_node(n, account_id=n, phone=f"p_{n}", device_id=f"d_{n}")
    for _, r in transactions_df.iterrows():
        a, b = r["from_account"], r["to_account"]
        key = tuple(sorted([a, b]))
        if G.has_edge(*key):
            G[key[0]][key[1]]["transaction_count"] += 1
            G[key[0]][key[1]]["transaction_volume"] += r["amount"]
        else:
            G.add_edge(
                key[0],
                key[1],
                transaction_count=1,
                transaction_volume=r["amount"],
                relationship_types={"transaction"},
            )
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


def _make_mule_scenario():
    """Inbound from several sources, rapid internal forward — mule not cash-out."""
    accounts_df = pd.DataFrame({
        "account_id": ["MULE", "S1", "S2", "S3", "D1", "D2", "D3"],
        "phone": ["p_m"] * 7,
        "device_id": ["d_m"] * 7,
        "is_fraud_ring": ["ring_1"] * 7,
    })
    txs = [
        ["S1", "MULE", 100000, pd.Timestamp("2026-01-01 10:00")],
        ["S2", "MULE", 150000, pd.Timestamp("2026-01-01 10:05")],
        ["S3", "MULE", 120000, pd.Timestamp("2026-01-01 10:10")],
        ["MULE", "D1", 80000, pd.Timestamp("2026-01-01 10:30")],
        ["MULE", "D2", 120000, pd.Timestamp("2026-01-01 10:35")],
        ["MULE", "D3", 140000, pd.Timestamp("2026-01-01 10:40")],
    ]
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(accounts_df, transactions_df)
    return accounts_df, transactions_df, G


def _make_coordinator_scenario():
    accts = {"COORD": "ring_1"}
    for i in range(6):
        accts[f"MEM{i}"] = "ring_1"
    accounts_df = pd.DataFrame({
        "account_id": list(accts.keys()),
        "phone": ["p_coord"] + [f"p{i}" for i in range(6)],
        "device_id": ["d_coord"] + [f"d{i}" for i in range(6)],
        "is_fraud_ring": list(accts.values()),
    })
    txs = []
    for i in range(3):
        txs.append([f"MEM{i}", "COORD", 10000, pd.Timestamp("2026-01-01")])
        txs.append([f"MEM{i}", f"MEM{(i+1)%3}", 10000, pd.Timestamp("2026-01-01")])
    for i in range(3, 6):
        txs.append(["COORD", f"MEM{i}", 2000, pd.Timestamp("2026-01-01")])
        txs.append([f"MEM{i}", f"MEM{3+(i+1)%3}", 10000, pd.Timestamp("2026-01-01")])
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(accounts_df, transactions_df)
    return accounts_df, transactions_df, G


def _make_consolidator_scenario():
    accounts_df = pd.DataFrame({
        "account_id": ["CONSOL", "S1", "S2", "S3", "S4", "DEST"],
        "phone": ["p_con"] + [f"p{i}" for i in range(5)],
        "device_id": ["d_con"] + [f"d{i}" for i in range(5)],
        "is_fraud_ring": ["ring_1"] * 5 + ["no"],
    })
    txs = [
        ["S1", "CONSOL", 100000, pd.Timestamp("2026-01-01")],
        ["S2", "CONSOL", 200000, pd.Timestamp("2026-01-02")],
        ["S3", "CONSOL", 150000, pd.Timestamp("2026-01-03")],
        ["S4", "CONSOL", 250000, pd.Timestamp("2026-01-04")],
        ["CONSOL", "DEST", 600000, pd.Timestamp("2026-01-05")],
    ]
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(accounts_df, transactions_df)
    return accounts_df, transactions_df, G


def _make_cash_out_scenario():
    """Suspicious inbound → large concentrated external outbound."""
    accounts_df = pd.DataFrame({
        "account_id": ["CASHOUT", "FUNDER1", "FUNDER2"],
        "phone": ["p_co"] * 3,
        "device_id": ["d_co"] * 3,
        "is_fraud_ring": ["ring_1", "ring_1", "ring_1"],
    })
    txs = [
        ["FUNDER1", "CASHOUT", 500000, pd.Timestamp("2026-01-01 10:00")],
        ["FUNDER2", "CASHOUT", 600000, pd.Timestamp("2026-01-01 11:00")],
        ["CASHOUT", "EXT_ACCT", 850000, pd.Timestamp("2026-01-01 12:00")],
        ["CASHOUT", "EXT_ACCT2", 100000, pd.Timestamp("2026-01-01 13:00")],
        ["CASHOUT", "EXT_ACCT3", 100000, pd.Timestamp("2026-01-01 14:00")],
        ["FUNDER1", "FUNDER2", 50000, pd.Timestamp("2026-01-02")],
    ]
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(
        accounts_df,
        transactions_df,
        extra_nodes=["EXT_ACCT", "EXT_ACCT2", "EXT_ACCT3"],
    )
    return accounts_df, transactions_df, G


def _make_victim_scenario():
    accounts_df = pd.DataFrame({
        "account_id": ["VICTIM", "MULE1", "MULE2"],
        "phone": ["p_v", "p_m1", "p_m2"],
        "device_id": ["d_v", "d_m1", "d_m2"],
        "is_fraud_ring": ["no", "ring_1", "ring_1"],
    })
    txs = [
        ["VICTIM", "MULE1", 200000, pd.Timestamp("2026-01-01")],
        ["VICTIM", "MULE2", 150000, pd.Timestamp("2026-01-02")],
    ]
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(accounts_df, transactions_df)
    return accounts_df, transactions_df, G


def _make_ambiguous_scenario():
    accounts_df = pd.DataFrame({
        "account_id": ["A", "B", "C"],
        "phone": ["p_a", "p_b", "p_c"],
        "device_id": ["d_a", "d_b", "d_c"],
        "is_fraud_ring": ["no"] * 3,
    })
    txs = [
        ["A", "B", 1000, pd.Timestamp("2026-01-01")],
        ["B", "C", 1500, pd.Timestamp("2026-01-02")],
        ["C", "A", 200, pd.Timestamp("2026-01-03")],
    ]
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(accounts_df, transactions_df)
    return accounts_df, transactions_df, G


def _make_normal_outbound_scenario():
    """Normal accounts with ordinary outbound transfers — must NOT be cash-out."""
    accounts_df = pd.DataFrame({
        "account_id": ["N1", "N2", "N3", "N4"],
        "phone": [f"p{i}" for i in range(4)],
        "device_id": [f"d{i}" for i in range(4)],
        "is_fraud_ring": ["no"] * 4,
    })
    txs = [
        # light internal chatter
        ["N1", "N2", 5000, pd.Timestamp("2026-01-01 10:00")],
        ["N2", "N3", 4000, pd.Timestamp("2026-01-02 11:00")],
        # ordinary external payments (salary/bills style) — not cash-out
        ["N1", "MERCHANT1", 2500, pd.Timestamp("2026-01-03 12:00")],
        ["N2", "MERCHANT2", 3000, pd.Timestamp("2026-01-04 13:00")],
        ["N3", "MERCHANT1", 1500, pd.Timestamp("2026-01-05 14:00")],
        ["N4", "N1", 2000, pd.Timestamp("2026-01-06 15:00")],
        ["N4", "UTILITY", 8000, pd.Timestamp("2026-01-07 16:00")],
    ]
    transactions_df = pd.DataFrame(
        txs, columns=["from_account", "to_account", "amount", "timestamp"]
    )
    G = _build_graph(
        accounts_df,
        transactions_df,
        extra_nodes=["MERCHANT1", "MERCHANT2", "UTILITY"],
    )
    return accounts_df, transactions_df, G


class TestRoleClassifier:
    def test_mule_detection(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = {"MULE", "S1", "S2", "S3", "D1", "D2", "D3"}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        mule_profile = next(p for p in profiles if p["account_id"] == "MULE")
        assert mule_profile["probable_role"] == "probable_mule"
        assert mule_profile["role_confidence"] >= 40

    def test_mule_not_classified_as_cash_out_when_money_stays_internal(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = {"MULE", "S1", "S2", "S3", "D1", "D2", "D3"}
        profiles, _, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        mule = next(p for p in profiles if p["account_id"] == "MULE")
        assert mule["probable_role"] == "probable_mule"
        assert mule["probable_role"] != "probable_cash_out"

    def test_coordinator_detection(self):
        accounts_df, transactions_df, G = _make_coordinator_scenario()
        members = {"COORD"} | {f"MEM{i}" for i in range(6)}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="MEDIUM",
        )
        coord_profile = next(p for p in profiles if p["account_id"] == "COORD")
        assert coord_profile["probable_role"] == "probable_coordinator"
        assert coord_profile["role_confidence"] >= 40

    def test_consolidator_detection(self):
        accounts_df, transactions_df, G = _make_consolidator_scenario()
        members = {"CONSOL", "S1", "S2", "S3", "S4", "DEST"}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        consol_profile = next(p for p in profiles if p["account_id"] == "CONSOL")
        assert consol_profile["probable_role"] == "probable_consolidator"
        assert consol_profile["role_confidence"] >= 40

    def test_cash_out_detection(self):
        accounts_df, transactions_df, G = _make_cash_out_scenario()
        members = {"CASHOUT", "FUNDER1", "FUNDER2"}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        cash_profile = next(p for p in profiles if p["account_id"] == "CASHOUT")
        assert cash_profile["probable_role"] == "probable_cash_out"
        assert cash_profile["role_confidence"] >= 40
        evidence = " ".join(cash_profile["role_evidence"])
        assert "external" in evidence.lower() or "₹" in evidence

    def test_normal_outbound_not_cash_out(self):
        """Ordinary outbound transfers must NOT become probable cash-out."""
        accounts_df, transactions_df, G = _make_normal_outbound_scenario()
        members = {"N1", "N2", "N3", "N4"}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="LOW",
        )
        for p in profiles:
            assert p["probable_role"] != "probable_cash_out", (
                f"{p['account_id']} incorrectly classified as cash-out: "
                f"{p['role_evidence']}"
            )
        # Even if cluster wrongly marked MEDIUM, low internal inbound blocks cash-out
        profiles_med, _, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="MEDIUM",
        )
        for p in profiles_med:
            assert p["probable_role"] != "probable_cash_out", (
                f"{p['account_id']} cash-out under MEDIUM: {p['role_evidence']}"
            )

    def test_victim_detection(self):
        accounts_df, transactions_df, G = _make_victim_scenario()
        members = {"VICTIM", "MULE1", "MULE2"}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        victim_profile = next(p for p in profiles if p["account_id"] == "VICTIM")
        assert victim_profile["probable_role"] == "suspected_victim"
        assert victim_profile["role_confidence"] >= 40

    def test_ambiguous_account_unknown(self):
        accounts_df, transactions_df, G = _make_ambiguous_scenario()
        members = {"A", "B", "C"}
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="LOW",
        )
        for p in profiles:
            if p["probable_role"] != "unknown":
                assert p["role_confidence"] < 60
        # At least some should be unknown given tiny ambiguous flows
        assert summary["unknown"] >= 1 or all(
            p["role_confidence"] < 60 for p in profiles
        )

    def test_confidence_between_0_and_100(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = set(accounts_df["account_id"])
        profiles, _, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        for p in profiles:
            assert 0 <= p["role_confidence"] <= 100

    def test_every_profile_has_evidence(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = set(accounts_df["account_id"])
        profiles, _, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        for p in profiles:
            assert len(p["role_evidence"]) > 0
            for ev in p["role_evidence"]:
                assert isinstance(ev, str)
                assert len(ev) > 10

    def test_no_definitive_claims(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = set(accounts_df["account_id"])
        profiles, _, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        for p in profiles:
            assert "probable" in p["probable_role"] or p["probable_role"] in (
                "suspected_victim", "unknown"
            )

    def test_ground_truth_not_used(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = set(accounts_df["account_id"])
        profiles, summary, network_summary = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        for p in profiles:
            assert "is_fraud_ring" not in p
            assert "true_label_majority" not in p
            assert "ring_1" not in str(p.get("role_evidence", []))
        assert "is_fraud_ring" not in str(summary)

    def test_role_summary_counts(self):
        accounts_df, transactions_df, G = _make_coordinator_scenario()
        members = set(accounts_df["account_id"])
        profiles, summary, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="MEDIUM",
        )
        total = sum(summary.values())
        assert total == len(members)

    def test_evidence_contains_measurable_facts(self):
        accounts_df, transactions_df, G = _make_mule_scenario()
        members = set(accounts_df["account_id"])
        profiles, _, _ = classify_cluster_roles(
            members, G, accounts_df, transactions_df,
            cluster_risk_level="HIGH",
        )
        mule = next(p for p in profiles if p["account_id"] == "MULE")
        all_evidence = " ".join(mule["role_evidence"])
        assert "₹" in all_evidence or any(c.isdigit() for c in all_evidence)

    def test_detection_modules_do_not_load_ground_truth(self):
        from pathlib import Path
        roots = [
            Path("detect_fraud.py"),
            Path("src/loader.py"),
            Path("src/role_classifier.py"),
            Path("src/money_flow.py"),
            Path("src/features.py"),
            Path("src/risk_scorer.py"),
            Path("src/graph_builder.py"),
        ]
        for path in roots:
            text = path.read_text()
            assert "GROUND_TRUTH_FILE" not in text, f"{path} loads GT config"
            assert "load_ground_truth" not in text, f"{path} loads GT helper"
            assert "ground_truth_roles.csv" not in text, f"{path} references GT file"
