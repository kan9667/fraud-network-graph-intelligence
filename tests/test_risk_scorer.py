import pytest
from src.config import RISK_WEIGHTS, RISK_LEVEL_THRESHOLDS
from src.risk_scorer import score_cluster, _assign_risk_level


def _dense_ring_features():
    return {
        "cluster_size": 5,
        "network_density": 1.0,
        "number_of_edges": 10,
        "average_degree": 4.0,
        "weighted_degree": 4.0,
        "average_clustering_coefficient": 1.0,
        "connected_components": 1,
        "shared_phone_count": 10,
        "shared_device_count": 10,
        "accounts_sharing_phone": 5,
        "accounts_sharing_device": 5,
        "identity_reuse_ratio": 1.0,
        "internal_transaction_count": 10,
        "internal_transaction_volume": 1000000,
        "external_inflow": 0,
        "external_outflow": 0,
        "total_transaction_volume": 1000000,
        "unique_external_counterparties": 0,
        "average_transaction_amount": 100000.0,
        "median_transaction_amount": 100000.0,
        "max_transaction_amount": 100000.0,
        "first_transaction": "2026-01-01T10:00:00",
        "last_transaction": "2026-01-01T11:00:00",
        "active_duration_hours": 1.0,
        "transactions_per_hour": 10.0,
        "transactions_per_day": 240.0,
        "burstiness_score": 0.5,
        "internal_flow_ratio": 1.0,
        "external_inflow_ratio": 0.0,
        "external_outflow_ratio": 0.0,
        "rapid_forwarding_ratio": 0.0,
        "average_degree_centrality": 1.0,
        "max_betweenness_centrality": 0.25,
        "number_of_high_centrality_nodes": 0,
    }


def _normal_cluster_features():
    return {
        "cluster_size": 50,
        "network_density": 0.02,
        "number_of_edges": 25,
        "average_degree": 1.0,
        "weighted_degree": 1.0,
        "average_clustering_coefficient": 0.0,
        "connected_components": 5,
        "shared_phone_count": 0,
        "shared_device_count": 0,
        "accounts_sharing_phone": 0,
        "accounts_sharing_device": 0,
        "identity_reuse_ratio": 0.0,
        "internal_transaction_count": 20,
        "internal_transaction_volume": 20000,
        "external_inflow": 5000,
        "external_outflow": 3000,
        "total_transaction_volume": 28000,
        "unique_external_counterparties": 10,
        "average_transaction_amount": 1000.0,
        "median_transaction_amount": 1000.0,
        "max_transaction_amount": 1000.0,
        "first_transaction": "2026-06-01T10:00:00",
        "last_transaction": "2026-06-30T10:00:00",
        "active_duration_hours": 696.0,
        "transactions_per_hour": 0.029,
        "transactions_per_day": 0.69,
        "burstiness_score": 0.3,
        "internal_flow_ratio": 0.714,
        "external_inflow_ratio": 0.179,
        "external_outflow_ratio": 0.107,
        "rapid_forwarding_ratio": 0.107,
        "average_degree_centrality": 0.02,
        "max_betweenness_centrality": 0.1,
        "number_of_high_centrality_nodes": 0,
    }


class TestRiskScorer:
    def test_dense_ring_high_risk(self):
        features = _dense_ring_features()
        risk = score_cluster({"A0", "A1", "A2", "A3", "A4"}, features, None)
        assert risk["risk_score"] >= 60
        assert risk["risk_level"] in ("HIGH", "CRITICAL")

    def test_normal_cluster_low_risk(self):
        features = _normal_cluster_features()
        risk = score_cluster({"N0"}, features, None)
        assert risk["risk_score"] < 30
        assert risk["risk_level"] == "LOW"

    def test_risk_score_between_0_and_100(self):
        features = _dense_ring_features()
        risk = score_cluster({"A0", "A1"}, features, None)
        assert 0 <= risk["risk_score"] <= 100

        features2 = _normal_cluster_features()
        risk2 = score_cluster({"N0"}, features2, None)
        assert 0 <= risk2["risk_score"] <= 100

    def test_risk_level_thresholds(self):
        assert _assign_risk_level(0) == "LOW"
        assert _assign_risk_level(29) == "LOW"
        assert _assign_risk_level(30) == "MEDIUM"
        assert _assign_risk_level(59) == "MEDIUM"
        assert _assign_risk_level(60) == "HIGH"
        assert _assign_risk_level(79) == "HIGH"
        assert _assign_risk_level(80) == "CRITICAL"
        assert _assign_risk_level(100) == "CRITICAL"

    def test_risk_factors_are_present_for_high_risk(self):
        features = _dense_ring_features()
        risk = score_cluster({"A0", "A1"}, features, None)
        assert len(risk["risk_factors"]) > 0
        for rf in risk["risk_factors"]:
            assert "factor" in rf
            assert "score" in rf
            assert "evidence" in rf

    def test_explanation_is_present(self):
        features = _dense_ring_features()
        risk = score_cluster({"A0", "A1"}, features, None)
        assert "explanation" in risk
        assert len(risk["explanation"]) > 0

    def test_risk_weights_sum_to_one(self):
        total = sum(RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_zero_volume_no_division_error(self):
        features = _dense_ring_features()
        features["internal_transaction_volume"] = 0
        features["total_transaction_volume"] = 0
        features["internal_transaction_count"] = 0
        features["internal_flow_ratio"] = 0.0
        risk = score_cluster({"A0", "A1"}, features, None)
        assert 0 <= risk["risk_score"] <= 100

    def test_no_ground_truth_in_risk_output(self):
        features = _dense_ring_features()
        risk = score_cluster({"A0", "A1"}, features, None)
        assert "is_fraud_ring" not in risk
        assert "true_label_majority" not in risk
