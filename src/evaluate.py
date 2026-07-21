"""Offline unsupervised detection evaluation (Phase 7).

THIS MODULE is the only production-adjacent code allowed to load ground truth.

Ground-truth sources (evaluation only):
  - data/ground_truth_roles.csv
  - accounts.is_fraud_ring (optional cross-check; primary labels come from GT file)

Detection inputs (unsupervised outputs only):
  - output/cluster_results.json

Does NOT modify detection, scoring, or dashboard behavior.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.config import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    GROUND_TRUTH_FILE,
    CLUSTER_RESULTS_FILE,
    ACCOUNTS_FILE,
    TRANSACTIONS_FILE,
    RISK_WEIGHTS,
    RISK_LEVEL_THRESHOLDS,
    RING_RECOVERY_MIN_RECALL,
    RING_RECOVERY_MIN_F1,
    EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS,
    EVAL_SUSPICIOUS_LEVELS_HIGH_PLUS,
    EVALUATION_REPORT_JSON,
    EVALUATION_REPORT_MD,
    EVAL_SCENARIO_META,
    ABLATION_SIGNAL_FAMILIES,
)


# Inferred role label → ground-truth role label
_PRED_ROLE_TO_GT = {
    "probable_mule": "mule",
    "probable_coordinator": "coordinator",
    "probable_consolidator": "consolidator",
    "probable_cash_out": "cash_out",
    "suspected_victim": "victim",
    "unknown": "unknown",
}

_GT_ROLES = ("victim", "mule", "coordinator", "consolidator", "cash_out", "normal")


# ---------------------------------------------------------------------------
# Safe metrics
# ---------------------------------------------------------------------------

def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def precision_recall_f1(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def confusion_counts(y_true_pos, y_pred_pos, universe):
    """Binary detection confusion over a finite universe of account ids."""
    y_true_pos = set(y_true_pos)
    y_pred_pos = set(y_pred_pos)
    universe = set(universe)
    tp = y_true_pos & y_pred_pos
    fp = y_pred_pos - y_true_pos
    fn = y_true_pos - y_pred_pos
    tn = universe - y_true_pos - y_pred_pos
    metrics = precision_recall_f1(len(tp), len(fp), len(fn))
    acc = safe_div(len(tp) + len(tn), len(universe))
    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "tn": len(tn),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": round(acc, 4),
    }


# ---------------------------------------------------------------------------
# Loaders (evaluation only)
# ---------------------------------------------------------------------------

def load_ground_truth(path=None):
    """Load evaluation ground truth. Only evaluation code should call this."""
    if path is None:
        path = GROUND_TRUTH_FILE
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {path}. Run generate_data.py first."
        )
    df = pd.read_csv(path)
    required = {"account_id", "fraud_ring_id", "ground_truth_role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground_truth_roles.csv missing columns: {sorted(missing)}")
    return df


def load_detection_results(path=None):
    if path is None:
        path = CLUSTER_RESULTS_FILE
        if not Path(path).exists():
            path = BASE_DIR / "cluster_results.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Detection results not found at {path}. Run detect_fraud.py first."
        )
    with open(path) as f:
        return json.load(f)


def load_dataset_sizes():
    accounts_path = ACCOUNTS_FILE if ACCOUNTS_FILE.exists() else BASE_DIR / "accounts.csv"
    tx_path = (
        TRANSACTIONS_FILE if TRANSACTIONS_FILE.exists() else BASE_DIR / "transactions.csv"
    )
    n_accounts = len(pd.read_csv(accounts_path)) if Path(accounts_path).exists() else None
    n_tx = len(pd.read_csv(tx_path)) if Path(tx_path).exists() else None
    return n_accounts, n_tx


# ---------------------------------------------------------------------------
# Account ↔ detection mapping
# ---------------------------------------------------------------------------

def account_cluster_assignments(results):
    """Map account_id -> list of clusters containing it (usually one)."""
    mapping = defaultdict(list)
    for r in results:
        for m in r.get("members", []):
            mapping[m].append(r)
    return mapping


def account_risk_from_clusters(results, all_accounts):
    """Assign each account the max risk_score among clusters it belongs to."""
    scores = {a: 0.0 for a in all_accounts}
    levels = {a: "NONE" for a in all_accounts}
    for r in results:
        score = float(r.get("risk_score", 0) or 0)
        level = str(r.get("risk_level", "LOW")).upper()
        for m in r.get("members", []):
            if m not in scores or score >= scores[m]:
                scores[m] = score
                levels[m] = level
    return scores, levels


def predicted_fraud_accounts(results, levels=None):
    """Accounts in clusters whose risk_level is in `levels`."""
    if levels is None:
        levels = EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS
    levels = {str(x).upper() for x in levels}
    flagged = set()
    for r in results:
        if str(r.get("risk_level", "")).upper() in levels:
            flagged.update(r.get("members", []))
    return flagged


def fraud_accounts_from_gt(gt_df):
    """Accounts belonging to any planted fraud ring (fraud_ring_id != none)."""
    mask = gt_df["fraud_ring_id"].astype(str).str.lower() != "none"
    return set(gt_df.loc[mask, "account_id"].astype(str))


def ring_membership(gt_df):
    """fraud_ring_id -> set of account_ids (excludes 'none')."""
    rings = {}
    for ring_id, grp in gt_df.groupby("fraud_ring_id"):
        rid = str(ring_id)
        if rid.lower() == "none":
            continue
        rings[rid] = set(grp["account_id"].astype(str))
    return rings


def predicted_roles_by_account(results):
    """account_id -> (probable_role, confidence) from detection profiles."""
    out = {}
    for r in results:
        for p in r.get("account_profiles", []):
            aid = p.get("account_id")
            if not aid:
                continue
            conf = float(p.get("role_confidence", 0) or 0)
            prev = out.get(aid)
            if prev is None or conf >= prev[1]:
                out[aid] = (p.get("probable_role", "unknown"), conf)
    return out


# ---------------------------------------------------------------------------
# Ring recovery
# ---------------------------------------------------------------------------

def best_cluster_match(gt_accounts, results):
    """Find cluster with highest F1 overlap against a ground-truth ring."""
    if not results:
        return None
    best = None
    for r in results:
        level = str(r.get("risk_level", "LOW")).upper()
        det = set(r.get("members", []))
        inter = gt_accounts & det
        tp = len(inter)
        fp = len(det - gt_accounts)
        fn = len(gt_accounts - det)
        prf = precision_recall_f1(tp, fp, fn)
        jaccard = safe_div(tp, len(gt_accounts | det))
        overlap_pct = round(100.0 * safe_div(tp, len(gt_accounts)), 2)
        candidate = {
            "cluster_id": r.get("cluster_id"),
            "risk_level": level,
            "risk_score": r.get("risk_score"),
            "detected_size": len(det),
            "intersection_size": tp,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "jaccard": round(jaccard, 4),
            "overlap_percentage": overlap_pct,
            "is_suspicious": level in EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS,
        }
        if best is None or candidate["f1"] > best["f1"] or (
            candidate["f1"] == best["f1"] and candidate["recall"] > best["recall"]
        ):
            best = candidate
    return best


def _empty_match():
    return {
        "cluster_id": None,
        "risk_level": None,
        "risk_score": None,
        "detected_size": 0,
        "intersection_size": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "jaccard": 0.0,
        "overlap_percentage": 0.0,
        "is_suspicious": False,
    }


def evaluate_ring_recovery(
    gt_df,
    results,
    min_recall=None,
    min_f1=None,
):
    if min_recall is None:
        min_recall = RING_RECOVERY_MIN_RECALL
    if min_f1 is None:
        min_f1 = RING_RECOVERY_MIN_F1

    rings = ring_membership(gt_df)
    recovery = {}
    for ring_id in sorted(rings.keys()):
        gt_accounts = rings[ring_id]
        suspicious_results = [
            r for r in results
            if str(r.get("risk_level", "")).upper() in EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS
        ]
        # Match among MEDIUM+ first (recovery decision); also keep overall best
        match_susp = best_cluster_match(gt_accounts, suspicious_results)
        match_any = best_cluster_match(gt_accounts, results)
        match = match_susp if match_susp is not None else (match_any or _empty_match())

        recovered = bool(
            match_susp is not None
            and match_susp["recall"] >= min_recall
            and match_susp["f1"] >= min_f1
        )
        meta = EVAL_SCENARIO_META.get(ring_id, {})
        recovery[ring_id] = {
            "title": meta.get("title", ring_id),
            "identity_signal": meta.get("identity_signal", "unknown"),
            "ground_truth_accounts": len(gt_accounts),
            "ground_truth_account_ids": sorted(gt_accounts),
            "best_matching_cluster": match["cluster_id"],
            "detected_accounts": match["detected_size"],
            "intersection": match["intersection_size"],
            "precision": match["precision"],
            "recall": match["recall"],
            "f1": match["f1"],
            "jaccard": match["jaccard"],
            "overlap_percentage": match["overlap_percentage"],
            "best_cluster_risk_level": match["risk_level"],
            "best_cluster_risk_score": match["risk_score"],
            "best_match_any_cluster": (
                None if match_any is None else {
                    "cluster_id": match_any["cluster_id"],
                    "risk_level": match_any["risk_level"],
                    "f1": match_any["f1"],
                    "recall": match_any["recall"],
                }
            ),
            "recovered": recovered,
            "recovery_thresholds": {
                "min_recall": min_recall,
                "min_f1": min_f1,
            },
        }
    return recovery


# ---------------------------------------------------------------------------
# Account-level detection
# ---------------------------------------------------------------------------

def evaluate_account_detection(gt_df, results):
    universe = set(gt_df["account_id"].astype(str))
    fraud = fraud_accounts_from_gt(gt_df)
    normal = universe - fraud

    pred_medium = predicted_fraud_accounts(results, EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS)
    pred_high = predicted_fraud_accounts(results, EVAL_SUSPICIOUS_LEVELS_HIGH_PLUS)

    return {
        "medium_plus": {
            "description": "Predicted fraud = members of MEDIUM/HIGH/CRITICAL clusters",
            "predicted_positive": len(pred_medium),
            **confusion_counts(fraud, pred_medium, universe),
        },
        "high_critical_only": {
            "description": "Predicted fraud = members of HIGH/CRITICAL clusters only",
            "predicted_positive": len(pred_high),
            **confusion_counts(fraud, pred_high, universe),
        },
        "ground_truth_fraud_accounts": len(fraud),
        "ground_truth_normal_accounts": len(normal),
    }


# ---------------------------------------------------------------------------
# Risk score analysis
# ---------------------------------------------------------------------------

def evaluate_risk_scores(gt_df, results):
    all_accounts = set(gt_df["account_id"].astype(str))
    scores, levels = account_risk_from_clusters(results, all_accounts)
    fraud = fraud_accounts_from_gt(gt_df)
    normal = all_accounts - fraud

    def stats(ids):
        vals = [scores[a] for a in ids if a in scores]
        if not vals:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        s = pd.Series(vals, dtype=float)
        return {
            "count": int(len(vals)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        }

    def level_dist(ids):
        c = Counter(levels.get(a, "NONE") for a in ids)
        return dict(sorted(c.items()))

    return {
        "fraud_accounts": {
            **stats(fraud),
            "risk_level_distribution": level_dist(fraud),
        },
        "normal_accounts": {
            **stats(normal),
            "risk_level_distribution": level_dist(normal),
        },
        "note": (
            "Account risk scores are assigned from the unsupervised cluster "
            "risk_score of the community containing the account (0 if unclustered)."
        ),
    }


# ---------------------------------------------------------------------------
# Role classification evaluation
# ---------------------------------------------------------------------------

def evaluate_roles(gt_df, results):
    gt_roles = {
        str(r.account_id): str(r.ground_truth_role)
        for r in gt_df.itertuples(index=False)
    }
    pred = predicted_roles_by_account(results)

    # Align on accounts present in GT
    pairs = []
    for aid, gt_role in gt_roles.items():
        p_role_raw, _conf = pred.get(aid, ("unknown", 0))
        p_role = _PRED_ROLE_TO_GT.get(p_role_raw, "unknown")
        pairs.append((aid, gt_role, p_role, p_role_raw))

    total = len(pairs)
    classified = [(a, g, p, raw) for a, g, p, raw in pairs if p != "unknown"]
    coverage = safe_div(len(classified), total)

    # Accuracy among classified (exclude GT normal optional? include all)
    correct_classified = sum(1 for _, g, p, _ in classified if g == p)
    acc_classified = safe_div(correct_classified, len(classified))

    # Overall: unknown counts as incorrect unless GT is normal and we want
    # unknown≈normal? Spec says overall accuracy including unknown predictions.
    # Treat unknown as correct only if we map it as "no fraud role claim" —
    # standard: unknown is wrong when GT is a specific role; for GT normal,
    # unknown is a reasonable "not assigned fraud role".
    correct_overall = 0
    for _, g, p, _ in pairs:
        if p == "unknown":
            if g == "normal":
                correct_overall += 1
        elif p == g:
            correct_overall += 1
    acc_overall = safe_div(correct_overall, total)

    # Per-role precision/recall/F1 (for fraud roles; among all predictions)
    role_metrics = {}
    labels = ["victim", "mule", "coordinator", "consolidator", "cash_out"]
    for label in labels:
        tp = sum(1 for _, g, p, _ in pairs if g == label and p == label)
        fp = sum(1 for _, g, p, _ in pairs if g != label and p == label)
        fn = sum(1 for _, g, p, _ in pairs if g == label and p != label)
        prf = precision_recall_f1(tp, fp, fn)
        support = sum(1 for _, g, _, _ in pairs if g == label)
        role_metrics[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **prf,
        }

    macro_f1 = round(
        sum(role_metrics[l]["f1"] for l in labels) / len(labels), 4
    ) if labels else 0.0

    # Confusion matrix: rows = GT, cols = predicted
    pred_labels = labels + ["unknown", "normal"]
    # predicted never is "normal" from our mapping; include unknown
    col_labels = labels + ["unknown"]
    matrix = {gt: {pr: 0 for pr in col_labels} for gt in _GT_ROLES}
    for _, g, p, _ in pairs:
        g_key = g if g in matrix else "normal"
        p_key = p if p in col_labels else "unknown"
        matrix[g_key][p_key] = matrix[g_key].get(p_key, 0) + 1

    return {
        "coverage": round(coverage, 4),
        "coverage_percent": round(100 * coverage, 2),
        "n_accounts": total,
        "n_classified_non_unknown": len(classified),
        "accuracy_among_classified": round(acc_classified, 4),
        "accuracy_overall_including_unknown": round(acc_overall, 4),
        "macro_f1": macro_f1,
        "per_role": role_metrics,
        "confusion_matrix": {
            "rows_ground_truth": list(_GT_ROLES),
            "columns_predicted": col_labels,
            "matrix": matrix,
        },
        "note": (
            "Predicted labels mapped from probable_* / suspected_victim. "
            "Unknown predicted vs GT normal counts as overall-correct "
            "(no false fraud-role claim)."
        ),
    }


# ---------------------------------------------------------------------------
# Scenario-level table
# ---------------------------------------------------------------------------

def evaluate_scenarios(ring_recovery):
    rows = []
    for ring_id, info in sorted(ring_recovery.items()):
        rows.append({
            "scenario": ring_id,
            "title": info.get("title"),
            "identity_signal": info.get("identity_signal"),
            "accounts": info.get("ground_truth_accounts"),
            "best_cluster": info.get("best_matching_cluster"),
            "recall": info.get("recall"),
            "precision": info.get("precision"),
            "f1": info.get("f1"),
            "risk_level": info.get("best_cluster_risk_level"),
            "risk_score": info.get("best_cluster_risk_score"),
            "recovered": info.get("recovered"),
            "overlap_percentage": info.get("overlap_percentage"),
        })
    return rows


# ---------------------------------------------------------------------------
# Signal ablation (evaluation-only reweighting of stored feature_scores)
# ---------------------------------------------------------------------------

def _ablated_risk_score(feature_scores, components):
    """Recompute risk score using a subset of feature_scores + renormalized weights."""
    weights = {c: RISK_WEIGHTS[c] for c in components if c in RISK_WEIGHTS}
    if not weights:
        return 0.0
    wsum = sum(weights.values())
    if wsum <= 0:
        return 0.0
    total = 0.0
    for c, w in weights.items():
        total += float(feature_scores.get(c, 0) or 0) * (w / wsum)
    return max(0.0, min(100.0, total))


def _assign_level(score):
    level = "LOW"
    for name, thr in sorted(RISK_LEVEL_THRESHOLDS.items(), key=lambda x: x[1]):
        if score >= thr:
            level = name
    return level


def evaluate_signal_ablation(gt_df, results):
    """Evaluation-only ablation using stored feature_scores (no detector change)."""
    universe = set(gt_df["account_id"].astype(str))
    fraud = fraud_accounts_from_gt(gt_df)
    medium_thr = RISK_LEVEL_THRESHOLDS.get("MEDIUM", 30)

    ablation = {}
    for family, components in ABLATION_SIGNAL_FAMILIES.items():
        # Rebuild pseudo-clusters with ablated scores
        flagged = set()
        cluster_scores = []
        for r in results:
            fs = r.get("feature_scores") or {}
            score = _ablated_risk_score(fs, components)
            level = _assign_level(score)
            cluster_scores.append({
                "cluster_id": r.get("cluster_id"),
                "ablated_score": round(score, 2),
                "ablated_level": level,
                "size": r.get("size"),
            })
            if score >= medium_thr:
                flagged.update(r.get("members", []))

        det = confusion_counts(fraud, flagged, universe)

        # Ring recovery under ablated MEDIUM+ definition
        rings = ring_membership(gt_df)
        ring_f1s = []
        ring_recovered = 0
        for ring_id, gt_accounts in rings.items():
            best_f1 = 0.0
            best_recall = 0.0
            for r in results:
                fs = r.get("feature_scores") or {}
                score = _ablated_risk_score(fs, components)
                if score < medium_thr:
                    continue
                det_set = set(r.get("members", []))
                inter = len(gt_accounts & det_set)
                prf = precision_recall_f1(
                    inter, len(det_set - gt_accounts), len(gt_accounts - det_set)
                )
                if prf["f1"] > best_f1:
                    best_f1 = prf["f1"]
                    best_recall = prf["recall"]
            ring_f1s.append(best_f1)
            if (
                best_recall >= RING_RECOVERY_MIN_RECALL
                and best_f1 >= RING_RECOVERY_MIN_F1
            ):
                ring_recovered += 1

        ablation[family] = {
            "components": components,
            "account_detection_medium_plus": det,
            "mean_best_ring_f1": round(sum(ring_f1s) / len(ring_f1s), 4) if ring_f1s else 0.0,
            "rings_recovered": ring_recovered,
            "rings_total": len(rings),
            "n_flagged_accounts": len(flagged),
            "top_clusters": sorted(
                cluster_scores, key=lambda x: -x["ablated_score"]
            )[:5],
        }

    return {
        "method": (
            "Reweight stored per-cluster feature_scores using RISK_WEIGHTS "
            "restricted to each signal family; MEDIUM threshold applied to "
            "ablated score. Evaluation-only — production detector unchanged."
        ),
        "families": ablation,
    }


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    gt_path=None,
    results_path=None,
    min_recall=None,
    min_f1=None,
):
    gt_df = load_ground_truth(gt_path)
    results = load_detection_results(results_path)
    n_accounts, n_tx = load_dataset_sizes()
    if n_accounts is None:
        n_accounts = len(gt_df)

    fraud = fraud_accounts_from_gt(gt_df)
    rings = ring_membership(gt_df)

    ring_recovery = evaluate_ring_recovery(
        gt_df, results, min_recall=min_recall, min_f1=min_f1
    )
    account_metrics = evaluate_account_detection(gt_df, results)
    risk_analysis = evaluate_risk_scores(gt_df, results)
    role_metrics = evaluate_roles(gt_df, results)
    scenario_metrics = evaluate_scenarios(ring_recovery)
    signal_ablation = evaluate_signal_ablation(gt_df, results)

    medium = account_metrics["medium_plus"]
    report = {
        "dataset_summary": {
            "accounts": n_accounts,
            "transactions": n_tx,
            "fraud_accounts": len(fraud),
            "normal_accounts": n_accounts - len(fraud),
            "fraud_rings": len(rings),
            "candidate_clusters": len(results),
            "medium_plus_clusters": sum(
                1
                for r in results
                if str(r.get("risk_level", "")).upper()
                in EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS
            ),
            "high_critical_clusters": sum(
                1
                for r in results
                if str(r.get("risk_level", "")).upper()
                in EVAL_SUSPICIOUS_LEVELS_HIGH_PLUS
            ),
        },
        "account_metrics": account_metrics,
        "ring_recovery": ring_recovery,
        "scenario_metrics": scenario_metrics,
        "role_metrics": role_metrics,
        "risk_score_analysis": risk_analysis,
        "signal_ablation": signal_ablation,
        "evaluation_config": {
            "ring_recovery_min_recall": min_recall or RING_RECOVERY_MIN_RECALL,
            "ring_recovery_min_f1": min_f1 or RING_RECOVERY_MIN_F1,
            "ground_truth_file": str(GROUND_TRUTH_FILE),
            "detection_results_file": str(
                results_path or CLUSTER_RESULTS_FILE
            ),
            "note": (
                "Ground truth is used only in this evaluation module. "
                "Detection and dashboard remain unsupervised."
            ),
        },
    }
    return report


def write_evaluation_report(report, json_path=None, md_path=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = Path(json_path or EVALUATION_REPORT_JSON)
    md_path = Path(md_path or EVALUATION_REPORT_MD)

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md = render_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(md)

    return json_path, md_path


def render_markdown_report(report) -> str:
    ds = report["dataset_summary"]
    acc = report["account_metrics"]["medium_plus"]
    acc_h = report["account_metrics"]["high_critical_only"]
    role = report["role_metrics"]
    lines = [
        "# Unsupervised Detection Evaluation Report",
        "",
        "> Ground truth used **only** for offline evaluation. "
        "Production detection remains unsupervised.",
        "",
        "## Dataset",
        f"- Accounts: {ds['accounts']}",
        f"- Transactions: {ds['transactions']}",
        f"- Fraud accounts: {ds['fraud_accounts']}",
        f"- Normal accounts: {ds['normal_accounts']}",
        f"- Fraud rings: {ds['fraud_rings']}",
        f"- Candidate clusters: {ds['candidate_clusters']}",
        f"- MEDIUM+ clusters: {ds['medium_plus_clusters']}",
        f"- HIGH/CRITICAL clusters: {ds['high_critical_clusters']}",
        "",
        "## Account-level detection (MEDIUM+)",
        f"- Precision: {acc['precision']}",
        f"- Recall: {acc['recall']}",
        f"- F1: {acc['f1']}",
        f"- Accuracy: {acc['accuracy']}",
        f"- TP/FP/FN/TN: {acc['tp']}/{acc['fp']}/{acc['fn']}/{acc['tn']}",
        "",
        "## Account-level detection (HIGH/CRITICAL only)",
        f"- Precision: {acc_h['precision']}",
        f"- Recall: {acc_h['recall']}",
        f"- F1: {acc_h['f1']}",
        f"- TP/FP/FN/TN: {acc_h['tp']}/{acc_h['fp']}/{acc_h['fn']}/{acc_h['tn']}",
        "",
        "## Ring recovery",
        "",
        "| Scenario | Identity signal | Accounts | Best cluster | Recall | Precision | F1 | Risk | Recovered |",
        "|----------|-----------------|----------|--------------|--------|-----------|----|------|-----------|",
    ]
    for row in report["scenario_metrics"]:
        lines.append(
            f"| {row['scenario']} ({row['title']}) | {row['identity_signal']} | "
            f"{row['accounts']} | {row['best_cluster']} | {row['recall']} | "
            f"{row['precision']} | {row['f1']} | {row['risk_level']} | "
            f"{row['recovered']} |"
        )

    lines += [
        "",
        "## Role inference",
        f"- Coverage (non-unknown): {role['coverage_percent']}%",
        f"- Accuracy among classified: {role['accuracy_among_classified']}",
        f"- Overall accuracy (unknown handling): {role['accuracy_overall_including_unknown']}",
        f"- Macro F1 (fraud roles): {role['macro_f1']}",
        "",
        "### Per-role",
        "",
        "| Role | Support | Precision | Recall | F1 |",
        "|------|---------|-----------|--------|----|",
    ]
    for label, m in role["per_role"].items():
        lines.append(
            f"| {label} | {m['support']} | {m['precision']} | {m['recall']} | {m['f1']} |"
        )

    ra = report["risk_score_analysis"]
    lines += [
        "",
        "## Risk score analysis",
        f"- Fraud mean/median/min/max: "
        f"{ra['fraud_accounts']['mean']} / {ra['fraud_accounts']['median']} / "
        f"{ra['fraud_accounts']['min']} / {ra['fraud_accounts']['max']}",
        f"- Normal mean/median/min/max: "
        f"{ra['normal_accounts']['mean']} / {ra['normal_accounts']['median']} / "
        f"{ra['normal_accounts']['min']} / {ra['normal_accounts']['max']}",
        f"- Fraud level dist: {ra['fraud_accounts']['risk_level_distribution']}",
        f"- Normal level dist: {ra['normal_accounts']['risk_level_distribution']}",
        "",
        "## Signal ablation (evaluation-only)",
        "",
        "| Family | Account F1 | Rings recovered | Mean best ring F1 | Flagged accounts |",
        "|--------|------------|-----------------|-------------------|------------------|",
    ]
    for family, info in report["signal_ablation"]["families"].items():
        det = info["account_detection_medium_plus"]
        lines.append(
            f"| {family} | {det['f1']} | "
            f"{info['rings_recovered']}/{info['rings_total']} | "
            f"{info['mean_best_ring_f1']} | {info['n_flagged_accounts']} |"
        )

    lines += [
        "",
        "---",
        report["evaluation_config"]["note"],
        "",
    ]
    return "\n".join(lines)


def print_concise_report(report):
    ds = report["dataset_summary"]
    acc = report["account_metrics"]["medium_plus"]
    acc_h = report["account_metrics"]["high_critical_only"]
    role = report["role_metrics"]
    print("=" * 70)
    print("UNSUPERVISED DETECTION EVALUATION (offline / ground-truth only here)")
    print("=" * 70)
    print("\nDATASET")
    print(f"  Accounts:        {ds['accounts']}")
    print(f"  Transactions:    {ds['transactions']}")
    print(f"  Fraud accounts:  {ds['fraud_accounts']}")
    print(f"  Normal accounts: {ds['normal_accounts']}")
    print(f"  Fraud rings:     {ds['fraud_rings']}")
    print(f"  MEDIUM+ clusters:{ds['medium_plus_clusters']}")
    print(f"  HIGH/CRITICAL:   {ds['high_critical_clusters']}")

    print("\nACCOUNT DETECTION (MEDIUM+)")
    print(f"  Precision: {acc['precision']:.2%}")
    print(f"  Recall:    {acc['recall']:.2%}")
    print(f"  F1:        {acc['f1']:.2%}")
    print(f"  Accuracy:  {acc['accuracy']:.2%}")
    print(f"  TP={acc['tp']} FP={acc['fp']} FN={acc['fn']} TN={acc['tn']}")

    print("\nACCOUNT DETECTION (HIGH/CRITICAL only)")
    print(f"  Precision: {acc_h['precision']:.2%}")
    print(f"  Recall:    {acc_h['recall']:.2%}")
    print(f"  F1:        {acc_h['f1']:.2%}")
    print(f"  TP={acc_h['tp']} FP={acc_h['fp']} FN={acc_h['fn']} TN={acc_h['tn']}")

    print("\nRING RECOVERY")
    for ring_id, info in sorted(report["ring_recovery"].items()):
        status = "RECOVERED" if info["recovered"] else "missed"
        print(
            f"  {ring_id}: {status} | F1={info['f1']:.2f} "
            f"P={info['precision']:.2f} R={info['recall']:.2f} | "
            f"best=cluster {info['best_matching_cluster']} "
            f"({info['best_cluster_risk_level']}) | "
            f"{info['identity_signal']}"
        )

    print("\nROLE INFERENCE")
    print(f"  Coverage:  {role['coverage_percent']}%")
    print(f"  Accuracy (classified): {role['accuracy_among_classified']:.2%}")
    print(f"  Accuracy (overall):    {role['accuracy_overall_including_unknown']:.2%}")
    print(f"  Macro F1:  {role['macro_f1']:.2%}")

    print("\nSIGNAL ABLATION (account F1 @ MEDIUM ablated threshold)")
    for family, info in report["signal_ablation"]["families"].items():
        f1 = info["account_detection_medium_plus"]["f1"]
        print(
            f"  {family:10s}: F1={f1:.2%} | "
            f"rings {info['rings_recovered']}/{info['rings_total']} | "
            f"mean ring F1={info['mean_best_ring_f1']:.2f}"
        )

    ra = report["risk_score_analysis"]
    print("\nRISK SCORES")
    print(
        f"  Fraud:  mean={ra['fraud_accounts']['mean']:.1f} "
        f"median={ra['fraud_accounts']['median']:.1f} "
        f"range=[{ra['fraud_accounts']['min']:.0f}, {ra['fraud_accounts']['max']:.0f}]"
    )
    print(
        f"  Normal: mean={ra['normal_accounts']['mean']:.1f} "
        f"median={ra['normal_accounts']['median']:.1f} "
        f"range=[{ra['normal_accounts']['min']:.0f}, {ra['normal_accounts']['max']:.0f}]"
    )
    print()


def main():
    report = run_evaluation()
    json_path, md_path = write_evaluation_report(report)
    print_concise_report(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return report


if __name__ == "__main__":
    main()
