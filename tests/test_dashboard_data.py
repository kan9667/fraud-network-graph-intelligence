"""UI-independent tests for dashboard data helpers (no Streamlit runtime)."""

from pathlib import Path

import pandas as pd

from src.dashboard_data import (
    build_case_report,
    case_report_markdown,
    compute_command_metrics,
    format_inr,
    load_cluster_results_raw,
    select_recommended_case,
    strip_forbidden,
    primary_roles,
)


def test_format_inr_full_amount_no_abbreviation():
    assert format_inr(1_959_542) == "₹1,959,542"
    assert format_inr(0) == "₹0"
    assert format_inr(None) == "₹0"
    assert format_inr(250_438) == "₹250,438"
    # Never abbreviate
    assert "M" not in format_inr(1_959_542)
    assert ".." not in format_inr(1_959_542)


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


def test_case_report_excludes_ground_truth_and_formats_currency():
    case = {
        "cluster_id": 10,
        "risk_level": "MEDIUM",
        "risk_score": 48,
        "size": 11,
        "members": ["ACC0214", "ACC0218"],
        "internal_volume": 1_959_542,
        "risk_factors": [{"factor": "identity_reuse"}],
        "role_summary": {"probable_mules": 2, "probable_cash_out": 1},
        "money_flow": {
            "internal_volume": 1_959_542,
            "external_inbound_volume": 0,
            "external_outbound_volume": 0,
            "exit_outbound_volume": 250_438,
            "estimated_forwarded_volume": 250_438,
            "estimated_forwarding_ratio": 0.2,
            "rapid_forwarding_events": 3,
            "top_external_flows": [],
        },
        "money_flow_paths": [
            {
                "path": ["ACC0214", "ACC0218"],
                "total_volume": 169_655,
                "transaction_count": 2,
                "time_span_hours": 1.0,
            }
        ],
        "account_profiles": [
            {
                "account_id": "ACC0214",
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
    assert report["money_flow_summary"]["internal_volume_display"] == "₹1,959,542"
    assert report["top_money_flow_paths"][0]["path_display"] == "ACC0214 → ACC0218"
    assert "**" not in report["top_money_flow_paths"][0]["path_display"]

    md = case_report_markdown(report)
    assert "CASE-010" in md
    assert "₹1,959,542" in md
    assert "ACC0214 → ACC0218" in md
    assert "ground_truth" not in md.lower()
    assert "is_fraud_ring" not in md
    # Required report sections
    for section in (
        "Risk indicators",
        "Money-flow summary",
        "Rapid-forwarding evidence",
        "Exit / sink evidence",
        "Top money-flow paths",
        "Timeline summary",
        "Account-level evidence",
        "Disclaimer",
    ):
        assert section in md


def test_select_recommended_case_prefers_medium_plus():
    results = [
        {
            "cluster_id": 1,
            "risk_level": "LOW",
            "risk_score": 90,
            "internal_volume": 9_000_000,
            "money_flow": {"exit_outbound_volume": 9_000_000, "rapid_forwarding_events": 99},
            "role_summary": {},
        },
        {
            "cluster_id": 10,
            "risk_level": "MEDIUM",
            "risk_score": 48,
            "internal_volume": 1_959_542,
            "money_flow": {"exit_outbound_volume": 250_438, "rapid_forwarding_events": 11},
            "role_summary": {"probable_mules": 4, "probable_cash_out": 1},
        },
        {
            "cluster_id": 9,
            "risk_level": "MEDIUM",
            "risk_score": 46,
            "internal_volume": 2_815_219,
            "money_flow": {"exit_outbound_volume": 100_000, "rapid_forwarding_events": 5},
            "role_summary": {"probable_mules": 1},
        },
    ]
    rec = select_recommended_case(results)
    assert rec is not None
    assert int(rec["cluster_id"]) == 10  # higher score among MEDIUM+
    assert str(rec["risk_level"]).upper() == "MEDIUM"


def test_primary_roles_formatting():
    text = primary_roles({"probable_mules": 2, "probable_cash_out": 1, "unknown": 3})
    assert "2 mule" in text
    assert "cash-out" in text


def test_load_results_and_metrics_smoke():
    results = load_cluster_results_raw()
    assert isinstance(results, list)
    assert len(results) >= 1
    for r in results:
        assert "is_fraud_ring" not in r
        assert "true_label_majority" not in r
        assert "ground_truth_role" not in r
        assert "fraud_ring_id" not in r

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


def test_money_flow_path_rendering_no_markdown_bold():
    """Account IDs in path display must be plain text."""
    from src.dashboard_data import build_case_report

    case = {
        "cluster_id": 1,
        "risk_level": "MEDIUM",
        "risk_score": 40,
        "size": 2,
        "members": ["ACC0001", "ACC0002"],
        "money_flow": {
            "internal_volume": 100,
            "external_inbound_volume": 0,
            "external_outbound_volume": 0,
            "exit_outbound_volume": 0,
            "estimated_forwarded_volume": 0,
            "estimated_forwarding_ratio": 0,
            "rapid_forwarding_events": 0,
        },
        "money_flow_paths": [
            {"path": ["ACC0001", "ACC0002"], "total_volume": 100, "transaction_count": 1}
        ],
        "account_profiles": [],
        "role_summary": {},
    }
    report = build_case_report(case)
    disp = report["top_money_flow_paths"][0]["path_display"]
    assert disp == "ACC0001 → ACC0002"
    assert "**" not in disp


def test_app_money_flow_paths_use_plain_account_ids():
    text = Path("app.py").read_text()
    assert 'f"**{node}**"' not in text
    assert "use_container_width" not in text
    assert "components.html" not in text
    assert "st.components" not in text


def test_bootstrap_helpers_exist():
    import app as app_mod

    assert hasattr(app_mod, "ensure_pipeline_ready")
    assert hasattr(app_mod, "bootstrap_pipeline_cached")
    assert hasattr(app_mod, "_run_generate_data")
    assert hasattr(app_mod, "_run_detect_fraud")
