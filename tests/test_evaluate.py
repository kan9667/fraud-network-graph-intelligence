"""Tests for offline evaluation module (Phase 7)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import (
    best_cluster_match,
    confusion_counts,
    evaluate_ring_recovery,
    evaluate_roles,
    fraud_accounts_from_gt,
    load_ground_truth,
    precision_recall_f1,
    run_evaluation,
    safe_div,
    write_evaluation_report,
)
from src.config import GROUND_TRUTH_FILE, EVALUATION_REPORT_JSON


class TestSafeMetrics:
    def test_safe_div_zero(self):
        assert safe_div(1, 0) == 0.0
        assert safe_div(0, 0) == 0.0
        assert safe_div(3, 4) == 0.75

    def test_precision_recall_f1_zero_division(self):
        m = precision_recall_f1(0, 0, 0)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_precision_recall_f1_values(self):
        m = precision_recall_f1(tp=2, fp=2, fn=2)
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["f1"] == 0.5

    def test_confusion_counts(self):
        universe = {"a", "b", "c", "d"}
        true_pos = {"a", "b"}
        pred_pos = {"a", "c"}
        c = confusion_counts(true_pos, pred_pos, universe)
        assert c["tp"] == 1
        assert c["fp"] == 1
        assert c["fn"] == 1
        assert c["tn"] == 1


class TestRingMatching:
    def test_best_cluster_match(self):
        gt = {"A1", "A2", "A3", "A4"}
        results = [
            {
                "cluster_id": 1,
                "risk_level": "MEDIUM",
                "risk_score": 40,
                "members": ["A1", "A2", "X"],
            },
            {
                "cluster_id": 2,
                "risk_level": "LOW",
                "risk_score": 10,
                "members": ["A1", "A2", "A3", "A4"],
            },
        ]
        match = best_cluster_match(gt, results)
        # Perfect match on cluster 2 should win by F1 even if LOW
        assert match["cluster_id"] == 2
        assert match["recall"] == 1.0
        assert match["precision"] == 1.0
        assert match["f1"] == 1.0

    def test_missing_ring_handled(self):
        gt_df = pd.DataFrame(
            {
                "account_id": ["A", "B", "C"],
                "fraud_ring_id": ["none", "none", "none"],
                "ground_truth_role": ["normal", "normal", "normal"],
            }
        )
        results = [
            {
                "cluster_id": 0,
                "risk_level": "MEDIUM",
                "risk_score": 50,
                "members": ["A", "B"],
                "account_profiles": [],
            }
        ]
        recovery = evaluate_ring_recovery(gt_df, results)
        assert recovery == {}

    def test_ring_recovery_structure(self):
        gt_df = pd.DataFrame(
            {
                "account_id": ["A1", "A2", "A3", "N1"],
                "fraud_ring_id": ["ring_1", "ring_1", "ring_1", "none"],
                "ground_truth_role": ["mule", "victim", "cash_out", "normal"],
            }
        )
        results = [
            {
                "cluster_id": 7,
                "risk_level": "MEDIUM",
                "risk_score": 45,
                "members": ["A1", "A2", "A3", "N1"],
                "account_profiles": [],
            }
        ]
        recovery = evaluate_ring_recovery(gt_df, results, min_recall=0.4, min_f1=0.3)
        assert "ring_1" in recovery
        assert recovery["ring_1"]["best_matching_cluster"] == 7
        assert recovery["ring_1"]["recall"] == 1.0
        assert recovery["ring_1"]["recovered"] is True


class TestRoleEvaluation:
    def test_unknown_handling(self):
        gt_df = pd.DataFrame(
            {
                "account_id": ["A", "B", "C"],
                "fraud_ring_id": ["ring_1", "ring_1", "none"],
                "ground_truth_role": ["mule", "victim", "normal"],
            }
        )
        results = [
            {
                "cluster_id": 1,
                "risk_level": "MEDIUM",
                "risk_score": 40,
                "members": ["A", "B", "C"],
                "account_profiles": [
                    {
                        "account_id": "A",
                        "probable_role": "probable_mule",
                        "role_confidence": 80,
                        "role_evidence": ["x"],
                    },
                    {
                        "account_id": "B",
                        "probable_role": "unknown",
                        "role_confidence": 0,
                        "role_evidence": ["x"],
                    },
                    {
                        "account_id": "C",
                        "probable_role": "unknown",
                        "role_confidence": 0,
                        "role_evidence": ["x"],
                    },
                ],
            }
        ]
        metrics = evaluate_roles(gt_df, results)
        assert metrics["n_classified_non_unknown"] == 1
        assert 0 < metrics["coverage"] <= 1
        assert "per_role" in metrics
        assert "mule" in metrics["per_role"]
        assert metrics["per_role"]["mule"]["tp"] == 1


class TestGroundTruthIsolation:
    def test_detection_modules_do_not_import_evaluate_gt_loader_in_detect(self):
        text = Path("detect_fraud.py").read_text()
        assert "load_ground_truth" not in text
        assert "from src.evaluate" not in text
        assert "GROUND_TRUTH_FILE" not in text
        assert "ground_truth_roles.csv" not in text

    def test_dashboard_does_not_load_gt_file(self):
        for path in [
            Path("app.py"),
            Path("src/dashboard_data.py"),
            Path("src/role_classifier.py"),
            Path("src/risk_scorer.py"),
            Path("src/loader.py"),
        ]:
            text = path.read_text()
            assert "load_ground_truth" not in text
            assert "GROUND_TRUTH_FILE" not in text
            assert "ground_truth_roles.csv" not in text

    def test_only_evaluate_module_loads_gt_file_reference(self):
        # src/evaluate.py and evaluate.py CLI / config path constant are allowed
        text = Path("src/evaluate.py").read_text()
        assert "GROUND_TRUTH_FILE" in text or "ground_truth_roles" in text


class TestFullEvaluationSmoke:
    @pytest.mark.skipif(
        not GROUND_TRUTH_FILE.exists(),
        reason="ground truth not generated",
    )
    def test_run_evaluation_sections(self, tmp_path):
        if not Path("output/cluster_results.json").exists():
            pytest.skip("detection results missing")
        report = run_evaluation()
        for key in [
            "dataset_summary",
            "account_metrics",
            "ring_recovery",
            "scenario_metrics",
            "role_metrics",
            "risk_score_analysis",
            "signal_ablation",
        ]:
            assert key in report

        assert "medium_plus" in report["account_metrics"]
        assert "high_critical_only" in report["account_metrics"]

        jpath, mpath = write_evaluation_report(
            report,
            json_path=tmp_path / "evaluation_report.json",
            md_path=tmp_path / "evaluation_report.md",
        )
        assert jpath.exists()
        data = json.loads(jpath.read_text())
        assert "ring_recovery" in data
        assert mpath.exists()
        assert "Ring recovery" in mpath.read_text()

    def test_fraud_accounts_from_gt(self):
        gt = pd.DataFrame(
            {
                "account_id": ["A", "B", "C"],
                "fraud_ring_id": ["ring_1", "none", "ring_2"],
                "ground_truth_role": ["mule", "normal", "victim"],
            }
        )
        fraud = fraud_accounts_from_gt(gt)
        assert fraud == {"A", "C"}
