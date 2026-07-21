"""UI-independent tests for dashboard data helpers (no Streamlit runtime)."""

from src.dashboard_data import (
    build_case_report,
    case_report_markdown,
    compute_command_metrics,
    load_cluster_results_raw,
    strip_forbidden,
    primary_roles,
)


def test_strip_forbidden_removes_ground_truth():
    payload = {
        "cluster_id": 1,
        "risk_score": 40,
        "is_fraud_ring": "yes",
        "account_profiles": [
            {
                "account_id": "A",
                "probable_role": "probable_mule",
                "ground_truth_role": "mule",
                "fraud_ring_id": "ring_1",
            }
        ],
        "true_label_majority": "fraud",
    }
    clean = strip_forbidden(payload)
    assert "is_fraud_ring" not in clean
    assert "true_label_majority" not in clean
    assert "ground_truth_role" not in clean["account_profiles"][0]
    assert "fraud_ring_id" not in clean["account_profiles"][0]
    assert clean["risk_score"] == 40


def test_case_report_excludes_ground_truth():
    case = {
        "cluster_id": 10,
        "risk_level": "MEDIUM",
        "risk_score": 48,
        "size": 11,
        "members": ["A", "B"],
        "internal_volume": 1000,
        "risk_factors": [{"factor": "identity_reuse"}],
        "role_summary": {"probable_mules": 2, "probable_cash_out": 1},
        "money_flow": {
            "internal_volume": 1000,
            "external_inbound_volume": 0,
            "external_outbound_volume": 0,
            "exit_outbound_volume": 200,
            "estimated_forwarded_volume": 200,
            "estimated_forwarding_ratio": 0.2,
            "rapid_forwarding_events": 3,
        },
        "money_flow_paths": [
            {
                "path": ["A", "B"],
                "total_volume": 500,
                "transaction_count": 2,
                "time_span_hours": 1.0,
            }
        ],
        "account_profiles": [
            {
                "account_id": "A",
                "probable_role": "probable_mule",
                "role_confidence": 80,
                "role_evidence": ["Forwarded funds within network"],
            }
        ],
        "network_structure_summary": "test summary",
        "is_fraud_ring": "should_be_stripped_if_present",
    }
    report = build_case_report(case)
    blob = str(report)
    assert "is_fraud_ring" not in report
    assert "ground_truth" not in blob.lower()
    assert "fraud_ring_id" not in blob
    assert report["case_id"] == "CASE-010"
    md = case_report_markdown(report)
    assert "CASE-010" in md
    assert "ground_truth" not in md.lower()


def test_primary_roles_formatting():
    text = primary_roles({"probable_mules": 2, "probable_cash_out": 1, "unknown": 3})
    assert "2 mule" in text
    assert "cash-out" in text


def test_load_results_and_metrics_smoke():
    results = load_cluster_results_raw()
    assert isinstance(results, list)
    assert len(results) >= 1
    # No ground truth keys at top level of any case
    for r in results:
        assert "is_fraud_ring" not in r
        assert "true_label_majority" not in r
        assert "ground_truth_role" not in r
        assert "fraud_ring_id" not in r

    import pandas as pd
    from src.dashboard_data import load_accounts_safe, load_transactions_safe

    accounts = load_accounts_safe()
    transactions = load_transactions_safe()
    assert "is_fraud_ring" not in accounts.columns
    metrics = compute_command_metrics(results, accounts, transactions)
    assert metrics["total_accounts"] == len(accounts)
    assert metrics["total_transactions"] == len(transactions)
    assert metrics["candidate_networks"] == len(results)
    assert set(metrics["risk_distribution"].keys()) == {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }
