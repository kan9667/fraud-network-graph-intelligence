from src.config import RISK_WEIGHTS, RISK_LEVEL_THRESHOLDS


def score_cluster(cluster_members, features, G):
    scores = {}

    scores["network_structure"] = _score_network_structure(features)
    scores["identity_reuse"] = _score_identity_reuse(features)
    scores["transaction_velocity"] = _score_transaction_velocity(features)
    scores["money_flow_concentration"] = _score_money_flow_concentration(features)
    scores["rapid_forwarding"] = _score_rapid_forwarding(features)
    scores["temporal_anomaly"] = _score_temporal_anomaly(features)
    scores["external_counterparty"] = _score_external_counterparty(features)

    weighted_sum = sum(
        scores[component] * RISK_WEIGHTS[component]
        for component in RISK_WEIGHTS
    )
    risk_score = max(0, min(100, round(weighted_sum)))

    risk_level = _assign_risk_level(risk_score)

    risk_factors = _build_risk_factors(scores, features)

    explanation = _build_explanation(risk_score, risk_level, risk_factors, features)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "feature_scores": {k: round(v) for k, v in scores.items()},
        "explanation": explanation,
    }


def _score_network_structure(f):
    density = f.get("network_density", 0)
    clustering = f.get("average_clustering_coefficient", 0)
    avg_deg = f.get("average_degree", 0)

    density_score = min(density * 100, 100)
    clustering_score = min(clustering * 100, 100)
    degree_score = min(avg_deg / 12 * 100, 100)

    return 0.4 * density_score + 0.3 * clustering_score + 0.3 * degree_score


def _score_identity_reuse(f):
    ratio = f.get("identity_reuse_ratio", 0)
    return min(ratio * 100, 100)


def _score_transaction_velocity(f):
    cluster_size = f.get("cluster_size", 1)
    tx_per_day = f.get("transactions_per_day", 0)
    tx_per_hour = f.get("transactions_per_hour", 0)

    if cluster_size == 0:
        return 0

    tx_per_account_per_day = tx_per_day / cluster_size
    tx_per_account_per_hour = tx_per_hour / cluster_size
    velocity = max(tx_per_account_per_day, tx_per_account_per_hour * 24)
    return min(velocity / 5 * 100, 100)


def _score_money_flow_concentration(f):
    ratio = f.get("internal_flow_ratio", 0)
    return min(ratio * 100, 100)


def _score_rapid_forwarding(f):
    ratio = f.get("rapid_forwarding_ratio", 0)
    return min(ratio * 100, 100)


def _score_temporal_anomaly(f):
    burstiness = f.get("burstiness_score", 0)
    return min(burstiness / 5 * 100, 100)


def _score_external_counterparty(f):
    cluster_size = f.get("cluster_size", 1)
    unique_ext = f.get("unique_external_counterparties", 0)
    if cluster_size == 0:
        return 0
    ext_density = unique_ext / cluster_size
    score = max(0, 100 - ext_density * 100)
    return score


def _assign_risk_level(score):
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if score >= RISK_LEVEL_THRESHOLDS[level]:
            return level
    return "LOW"


def _build_risk_factors(scores, features):
    factor_map = {
        "network_structure": {
            "label": "Dense internal network",
            "evidence": _evidence_network_structure,
        },
        "identity_reuse": {
            "label": "High identity reuse",
            "evidence": _evidence_identity_reuse,
        },
        "transaction_velocity": {
            "label": "High transaction velocity",
            "evidence": _evidence_transaction_velocity,
        },
        "money_flow_concentration": {
            "label": "Concentrated money flow",
            "evidence": _evidence_money_flow,
        },
        "rapid_forwarding": {
            "label": "Rapid forwarding pattern",
            "evidence": _evidence_rapid_forwarding,
        },
        "temporal_anomaly": {
            "label": "Irregular timing pattern",
            "evidence": _evidence_temporal,
        },
        "external_counterparty": {
            "label": "Limited external contacts",
            "evidence": _evidence_external,
        },
    }

    factors = []
    for component, info in factor_map.items():
        score = round(scores.get(component, 0))
        if score >= 40:
            factors.append({
                "factor": info["label"],
                "score": score,
                "evidence": info["evidence"](features),
            })

    factors.sort(key=lambda f: f["score"], reverse=True)
    return factors


def _build_explanation(risk_score, risk_level, risk_factors, features):
    n = features.get("cluster_size", 0)
    if risk_level == "LOW":
        return (
            f"This {n}-account network shows no strong fraud signals. "
            "Activity patterns are consistent with normal organic behavior."
        )
    top = risk_factors[:3] if risk_factors else []
    signals = ", ".join(f.lower() for f in [t["factor"] for t in top])
    return (
        f"This {n}-account network is rated {risk_level} risk (score: {risk_score}). "
        f"Key signals: {signals}. "
        f"See risk factors for detailed evidence."
    )


def _evidence_network_structure(f):
    density = f.get("network_density", 0)
    edges = f.get("number_of_edges", 0)
    avg_deg = f.get("average_degree", 0)
    return (
        f"Network density is {density} with {edges} relationships "
        f"and average degree {avg_deg}"
    )


def _evidence_identity_reuse(f):
    phone_accts = f.get("accounts_sharing_phone", 0)
    device_accts = f.get("accounts_sharing_device", 0)
    total = f.get("cluster_size", 0)
    parts = []
    if phone_accts > 0:
        parts.append(f"{phone_accts} accounts share a phone number")
    if device_accts > 0:
        parts.append(f"{device_accts} accounts share a device")
    if not parts:
        return "No identity reuse detected"
    return "; ".join(parts) + f" out of {total} accounts in this cluster"


def _evidence_transaction_velocity(f):
    tx_count = f.get("internal_transaction_count", 0)
    tx_per_day = f.get("transactions_per_day", 0)
    duration = f.get("active_duration_hours", 0)
    return (
        f"{tx_count} transactions in {duration:.1f} hours "
        f"({tx_per_day:.1f} per day)"
    )


def _evidence_money_flow(f):
    int_vol = f.get("internal_transaction_volume", 0)
    total_vol = f.get("total_transaction_volume", 0)
    ratio = f.get("internal_flow_ratio", 0) * 100
    return (
        f"{ratio:.0f}% of total volume (₹{int_vol:,}) flows internally "
        f"within this cluster (total: ₹{total_vol:,})"
    )


def _evidence_rapid_forwarding(f):
    ext_in = f.get("external_inflow", 0)
    ext_out = f.get("external_outflow", 0)
    ratio = f.get("rapid_forwarding_ratio", 0) * 100
    return (
        f"{ratio:.0f}% of total volume involves funds arriving from "
        f"outside (₹{ext_in:,}) and leaving to outside (₹{ext_out:,})"
    )


def _evidence_temporal(f):
    burst = f.get("burstiness_score", 0)
    tx_count = f.get("internal_transaction_count", 0)
    return (
        f"Burstiness score {burst} across {tx_count} transactions, "
        f"indicating {'highly irregular' if burst > 3 else 'moderately irregular' if burst > 1 else 'regular'} timing"
    )


def _evidence_external(f):
    unique_ext = f.get("unique_external_counterparties", 0)
    cluster_size = f.get("cluster_size", 0)
    if unique_ext == 0 and cluster_size < 50:
        return (
            f"No external counterparties detected — cluster is fully isolated, "
            f"consistent with a covert ring"
        )
    return (
        f"Only {unique_ext} unique external counterparties "
        f"for {cluster_size} accounts"
    )
