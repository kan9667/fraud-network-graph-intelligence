"""Tests for money-flow features, paths, and temporal analysis."""

import pandas as pd
import pytest

from src.money_flow import (
    compute_account_money_flow,
    compute_temporal_flow,
    find_money_flow_paths,
    summarize_cluster_money_flow,
    is_external,
)
from src.config import RAPID_FORWARDING_WINDOW_HOURS


def _tx(rows):
    return pd.DataFrame(
        rows,
        columns=["from_account", "to_account", "amount", "timestamp"],
    )


class TestAccountMoneyFlow:
    def test_internal_vs_external_volumes(self):
        members = {"A", "B", "C"}
        tx = _tx([
            ["A", "B", 100_000, pd.Timestamp("2026-01-01 10:00")],
            ["B", "C", 80_000, pd.Timestamp("2026-01-01 11:00")],
            ["C", "EXT1", 70_000, pd.Timestamp("2026-01-01 12:00")],
            ["EXT2", "A", 50_000, pd.Timestamp("2026-01-01 09:00")],
        ])
        flow_c = compute_account_money_flow("C", members, tx)
        assert flow_c["internal_inbound_volume"] == 80_000
        assert flow_c["external_outbound_volume"] == 70_000
        assert flow_c["internal_outbound_volume"] == 0
        assert flow_c["percentage_outbound_to_external"] == 100.0
        assert flow_c["unique_internal_sources"] == 1
        assert flow_c["unique_external_destinations"] == 1

        flow_a = compute_account_money_flow("A", members, tx)
        assert flow_a["external_inbound_volume"] == 50_000
        assert flow_a["internal_outbound_volume"] == 100_000
        assert flow_a["percentage_inbound_from_cluster"] == 0.0

    def test_forwarding_ratio(self):
        members = {"MULE", "SRC", "DST"}
        tx = _tx([
            ["SRC", "MULE", 200_000, pd.Timestamp("2026-01-01 10:00")],
            ["MULE", "DST", 180_000, pd.Timestamp("2026-01-01 11:00")],
        ])
        flow = compute_account_money_flow("MULE", members, tx)
        assert flow["estimated_forwarded_volume"] == 180_000
        assert flow["estimated_forwarding_ratio"] > 0.5

    def test_inbound_outbound_totals(self):
        members = {"X", "Y"}
        tx = _tx([
            ["Y", "X", 10_000, pd.Timestamp("2026-01-01")],
            ["EXT", "X", 5_000, pd.Timestamp("2026-01-02")],
            ["X", "Y", 3_000, pd.Timestamp("2026-01-03")],
            ["X", "OUT", 4_000, pd.Timestamp("2026-01-04")],
        ])
        flow = compute_account_money_flow("X", members, tx)
        assert flow["inbound_volume"] == 15_000
        assert flow["outbound_volume"] == 7_000
        assert flow["internal_inbound_count"] == 1
        assert flow["external_inbound_count"] == 1
        assert flow["internal_outbound_count"] == 1
        assert flow["external_outbound_count"] == 1


class TestTemporalFlow:
    def test_rapid_forwarding_detection(self):
        members = {"M", "S", "D"}
        tx = _tx([
            ["S", "M", 100_000, pd.Timestamp("2026-01-01 10:00")],
            ["M", "D", 90_000, pd.Timestamp("2026-01-01 10:30")],
            ["S", "M", 50_000, pd.Timestamp("2026-01-02 10:00")],
            ["M", "D", 40_000, pd.Timestamp("2026-01-02 11:00")],
        ])
        temporal = compute_temporal_flow("M", members, tx, window_hours=24)
        assert temporal["rapid_forwarding_events"] >= 2
        assert temporal["median_inbound_to_outbound_delay_hours"] is not None
        assert temporal["median_inbound_to_outbound_delay_hours"] < 2
        assert temporal["min_inbound_to_outbound_delay_hours"] <= temporal[
            "max_inbound_to_outbound_delay_hours"
        ]

    def test_no_rapid_when_delay_exceeds_window(self):
        members = {"M", "S", "D"}
        tx = _tx([
            ["S", "M", 100_000, pd.Timestamp("2026-01-01 10:00")],
            ["M", "D", 90_000, pd.Timestamp("2026-01-05 10:00")],
        ])
        temporal = compute_temporal_flow("M", members, tx, window_hours=24)
        assert temporal["rapid_forwarding_events"] == 0
        assert temporal["median_inbound_to_outbound_delay_hours"] == pytest.approx(
            96.0, abs=0.1
        )

    def test_window_uses_config_default(self):
        assert RAPID_FORWARDING_WINDOW_HOURS == 24


class TestMoneyFlowPaths:
    def test_paths_from_real_transactions(self):
        members = {"A", "B", "C"}
        tx = _tx([
            ["V", "A", 200_000, pd.Timestamp("2026-01-01 09:00")],
            ["A", "B", 180_000, pd.Timestamp("2026-01-01 10:00")],
            ["B", "C", 150_000, pd.Timestamp("2026-01-01 11:00")],
            ["C", "EXT", 140_000, pd.Timestamp("2026-01-01 12:00")],
        ])
        paths = find_money_flow_paths(members, tx, top_n=5)
        assert len(paths) > 0
        for p in paths:
            assert "source_account" in p
            assert "destination_account" in p
            assert "path" in p
            assert "total_volume" in p
            assert "transaction_count" in p
            assert "time_span_hours" in p
            assert len(p["path"]) >= 2
            assert p["total_volume"] > 0

    def test_paths_prefer_high_volume(self):
        members = {"A", "B"}
        tx = _tx([
            ["A", "B", 10_000, pd.Timestamp("2026-01-01")],
            ["A", "EXT", 500_000, pd.Timestamp("2026-01-02")],
        ])
        paths = find_money_flow_paths(members, tx, top_n=2)
        assert paths[0]["total_volume"] >= paths[-1]["total_volume"]


class TestClusterSummary:
    def test_summarize_cluster_money_flow(self):
        members = {"A", "B", "C"}
        tx = _tx([
            ["V", "A", 100_000, pd.Timestamp("2026-01-01 09:00")],
            ["A", "B", 90_000, pd.Timestamp("2026-01-01 10:00")],
            ["B", "C", 80_000, pd.Timestamp("2026-01-01 10:30")],
            ["C", "EXT", 70_000, pd.Timestamp("2026-01-01 11:00")],
        ])
        summary = summarize_cluster_money_flow(members, tx)
        mf = summary["money_flow"]
        assert mf["internal_volume"] == 90_000 + 80_000
        assert mf["external_inbound_volume"] == 100_000
        assert mf["external_outbound_volume"] == 70_000
        assert "top_internal_flows" in mf
        assert "top_external_flows" in mf
        assert "rapid_forwarding_events" in mf
        assert "money_flow_paths" in summary
        assert isinstance(summary["money_flow_paths"], list)

    def test_is_external(self):
        assert is_external("OUT", {"A", "B"}) is True
        assert is_external("A", {"A", "B"}) is False
