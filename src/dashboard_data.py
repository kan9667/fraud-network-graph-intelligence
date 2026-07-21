"""Data loading, metrics, and evidence helpers for the investigator dashboard.

Ground-truth fields are never exposed through these APIs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import (
    ACCOUNTS_FILE,
    TRANSACTIONS_FILE,
    CLUSTER_RESULTS_FILE,
    GRAPH_FILE,
    BASE_DIR,
    OUTPUT_DIR,
    DATA_DIR,
    RAPID_FORWARDING_WINDOW_HOURS,
    SUSPICIOUS_RISK_LEVELS,
)
from src.loader import load_accounts, load_transactions
from src.graph_builder import load_graph
from src.money_flow import (
    compute_account_money_flow,
    compute_temporal_flow,
    identify_sink_accounts,
)

# Columns that must never appear in the investigator UI or exports
_FORBIDDEN_KEYS = frozenset({
    "is_fraud_ring",
    "fraud_ring_id",
    "ground_truth_role",
    "true_label_majority",
})

ROLE_DISPLAY = {
    "probable_mule": "Probable mule",
    "probable_coordinator": "Probable coordinator",
    "probable_consolidator": "Probable consolidator",
    "probable_cash_out": "Probable cash-out",
    "suspected_victim": "Suspected victim",
    "unknown": "Unknown / unclassified",
}

ROLE_COLORS = {
    "probable_mule": "#e67e22",
    "probable_coordinator": "#9b59b6",
    "probable_consolidator": "#2980b9",
    "probable_cash_out": "#c0392b",
    "suspected_victim": "#f1c40f",
    "unknown": "#95a5a6",
}

RISK_COLORS = {
    "LOW": "#27ae60",
    "MEDIUM": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#c0392b",
}

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def strip_forbidden(obj):
    """Recursively drop ground-truth keys from dict/list structures."""
    if isinstance(obj, dict):
        return {
            k: strip_forbidden(v)
            for k, v in obj.items()
            if k not in _FORBIDDEN_KEYS
        }
    if isinstance(obj, list):
        return [strip_forbidden(x) for x in obj]
    return obj


def _resolve_cluster_results_path():
    if CLUSTER_RESULTS_FILE.exists():
        return CLUSTER_RESULTS_FILE
    root = BASE_DIR / "cluster_results.json"
    if root.exists():
        return root
    return CLUSTER_RESULTS_FILE


def _resolve_graph_path():
    if GRAPH_FILE.exists():
        return GRAPH_FILE
    return GRAPH_FILE


def load_cluster_results_raw():
    path = _resolve_cluster_results_path()
    with open(path) as f:
        results = json.load(f)
    return strip_forbidden(results)


def load_accounts_safe():
    """Load accounts, dropping any ground-truth columns."""
    path = ACCOUNTS_FILE if ACCOUNTS_FILE.exists() else BASE_DIR / "accounts.csv"
    df = load_accounts(path)
    drop = [c for c in df.columns if c in _FORBIDDEN_KEYS]
    if drop:
        df = df.drop(columns=drop)
    return df


def load_transactions_safe():
    path = TRANSACTIONS_FILE if TRANSACTIONS_FILE.exists() else BASE_DIR / "transactions.csv"
    return load_transactions(path)


def load_graph_safe():
    path = _resolve_graph_path()
    return load_graph(path)


def case_id(cluster_id) -> str:
    return f"CASE-{int(cluster_id):03d}"


def get_case_by_id(results, cluster_id):
    for r in results:
        if int(r["cluster_id"]) == int(cluster_id):
            return r
    return None


def sort_cases(results):
    return sorted(
        results,
        key=lambda r: (
            RISK_ORDER.get(str(r.get("risk_level", "LOW")).upper(), 9),
            -float(r.get("risk_score", 0)),
            -int(r.get("internal_volume", 0)),
        ),
    )


def primary_roles(role_summary: dict) -> str:
    if not role_summary:
        return "—"
    parts = []
    mapping = [
        ("probable_mules", "mule"),
        ("probable_coordinators", "coordinator"),
        ("probable_consolidators", "consolidator"),
        ("probable_cash_out", "cash-out"),
        ("suspected_victims", "victim"),
    ]
    for key, label in mapping:
        n = role_summary.get(key, 0)
        if n:
            parts.append(f"{n} {label}")
    if not parts:
        unk = role_summary.get("unknown", 0)
        return f"{unk} unclassified" if unk else "—"
    return ", ".join(parts)


def top_risk_factor_labels(case: dict, limit=4) -> list:
    factors = case.get("risk_factors") or []
    labels = []
    for f in factors[:limit]:
        if isinstance(f, dict):
            labels.append(f.get("factor") or f.get("name") or str(f))
        else:
            labels.append(str(f))
    return labels


def compute_command_metrics(results, accounts_df, transactions_df):
    n_accounts = len(accounts_df)
    n_tx = len(transactions_df)
    n_cases = len(results)
    medium_plus = [
        r for r in results
        if str(r.get("risk_level", "")).upper() in SUSPICIOUS_RISK_LEVELS
    ]
    high_crit = [
        r for r in results
        if str(r.get("risk_level", "")).upper() in {"HIGH", "CRITICAL"}
    ]
    suspicious_volume = sum(int(r.get("internal_volume", 0)) for r in medium_plus)
    rapid_total = 0
    for r in results:
        mf = r.get("money_flow") or {}
        rapid_total += int(mf.get("rapid_forwarding_events", 0) or 0)

    level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in results:
        lvl = str(r.get("risk_level", "LOW")).upper()
        if lvl in level_counts:
            level_counts[lvl] += 1

    return {
        "total_accounts": n_accounts,
        "total_transactions": n_tx,
        "candidate_networks": n_cases,
        "medium_plus_networks": len(medium_plus),
        "high_critical_networks": len(high_crit),
        "suspicious_transaction_volume": suspicious_volume,
        "rapid_forwarding_events": rapid_total,
        "risk_distribution": level_counts,
    }


def system_status(accounts_df, transactions_df, results):
    """Build transparency / system status fields from filesystem timestamps."""
    paths = {
        "accounts": ACCOUNTS_FILE if ACCOUNTS_FILE.exists() else BASE_DIR / "accounts.csv",
        "transactions": TRANSACTIONS_FILE if TRANSACTIONS_FILE.exists() else BASE_DIR / "transactions.csv",
        "cluster_results": _resolve_cluster_results_path(),
        "graph": _resolve_graph_path(),
    }
    mtimes = {}
    for name, p in paths.items():
        p = Path(p)
        if p.exists():
            mtimes[name] = datetime.fromtimestamp(p.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            mtimes[name] = "missing"

    detection_ok = (
        Path(paths["cluster_results"]).exists() and Path(paths["graph"]).exists()
    )
    return {
        "accounts_mtime": mtimes.get("accounts"),
        "transactions_mtime": mtimes.get("transactions"),
        "results_mtime": mtimes.get("cluster_results"),
        "graph_mtime": mtimes.get("graph"),
        "total_accounts": len(accounts_df),
        "total_transactions": len(transactions_df),
        "detection_status": "Ready" if detection_ok else "Incomplete",
        "candidate_cases": len(results),
    }


def case_transactions(case, transactions_df):
    """All transactions touching any cluster member."""
    members = set(case["members"])
    mask = (
        transactions_df["from_account"].isin(members)
        | transactions_df["to_account"].isin(members)
    )
    tx = transactions_df.loc[mask].copy()
    if len(tx) == 0:
        return tx

    sinks = identify_sink_accounts(members, transactions_df)

    def classify(row):
        src, dst = row["from_account"], row["to_account"]
        src_in = src in members
        dst_in = dst in members
        if src_in and dst_in:
            if dst in sinks and src not in sinks:
                return "internal→exit_sink"
            return "internal"
        if not src_in and dst_in:
            return "external_inbound"
        if src_in and not dst_in:
            return "external_exit"
        return "other"

    tx["flow_class"] = tx.apply(classify, axis=1)
    tx["is_exit"] = tx["flow_class"].isin(["external_exit", "internal→exit_sink"])
    tx["is_high_value"] = tx["amount"] >= tx["amount"].quantile(0.75) if len(tx) else False
    return tx.sort_values("timestamp")


def mark_rapid_forwarding(tx_df, members, window_hours=None):
    """Flag outbound rows that form a rapid-forward pair with a prior inbound."""
    if window_hours is None:
        window_hours = RAPID_FORWARDING_WINDOW_HOURS
    if len(tx_df) == 0:
        tx_df = tx_df.copy()
        tx_df["is_rapid_forward"] = []
        return tx_df

    window_sec = float(window_hours) * 3600.0
    members = set(members)
    rapid_idx = set()

    for aid in members:
        recv = tx_df[tx_df["to_account"] == aid].sort_values("timestamp")
        sent = tx_df[tx_df["from_account"] == aid].sort_values("timestamp")
        if len(recv) == 0 or len(sent) == 0:
            continue
        used = set()
        in_times = list(zip(recv.index, recv["timestamp"]))
        for oi, ots in zip(sent.index, sent["timestamp"]):
            best = None
            for ii, its in in_times:
                if ii in used:
                    continue
                if its <= ots:
                    delay = (ots - its).total_seconds()
                    if 0 <= delay <= window_sec:
                        if best is None or its > best[1]:
                            best = (ii, its)
            if best is not None:
                used.add(best[0])
                rapid_idx.add(oi)

    out = tx_df.copy()
    out["is_rapid_forward"] = out.index.isin(rapid_idx)
    return out


def account_profile_map(case):
    return {p["account_id"]: p for p in case.get("account_profiles", [])}


def account_investigation(account_id, case, G, transactions_df):
    """Full investigation stats for one account within a case."""
    members = set(case["members"])
    profiles = account_profile_map(case)
    profile = profiles.get(account_id, {
        "account_id": account_id,
        "probable_role": "unknown",
        "role_confidence": 0,
        "role_evidence": ["No role profile available for this account."],
    })

    flow = compute_account_money_flow(account_id, members, transactions_df)
    temporal = compute_temporal_flow(account_id, members, transactions_df)

    sent = transactions_df[transactions_df["from_account"] == account_id]
    received = transactions_df[transactions_df["to_account"] == account_id]
    all_tx = pd.concat([sent, received], ignore_index=True)

    first_ts = last_ts = None
    active_hours = 0.0
    if len(all_tx):
        first_ts = all_tx["timestamp"].min()
        last_ts = all_tx["timestamp"].max()
        active_hours = (last_ts - first_ts).total_seconds() / 3600.0

    degree = G.degree(account_id) if account_id in G else 0
    weighted_degree = 0
    if account_id in G:
        weighted_degree = sum(
            d.get("weight", 1) for _, _, d in G.edges(account_id, data=True)
        )

    counterparties = set(sent["to_account"]) | set(received["from_account"])
    internal_cp = counterparties & members
    external_cp = counterparties - members

    neighbors = list(G.neighbors(account_id)) if account_id in G else []

    return {
        "profile": profile,
        "flow": flow,
        "temporal": temporal,
        "first_transaction": first_ts.isoformat() if pd.notna(first_ts) and first_ts is not None else None,
        "last_transaction": last_ts.isoformat() if pd.notna(last_ts) and last_ts is not None else None,
        "active_duration_hours": round(active_hours, 4),
        "degree": int(degree),
        "weighted_degree": int(weighted_degree),
        "unique_counterparties": len(counterparties),
        "internal_counterparties": len(internal_cp),
        "external_counterparties": len(external_cp),
        "neighbors": neighbors,
    }


def build_case_summary_text(case) -> str:
    n = case.get("size", len(case.get("members", [])))
    mf = case.get("money_flow") or {}
    rapid = int(mf.get("rapid_forwarding_events", 0) or 0)
    exit_vol = int(mf.get("exit_outbound_volume", 0) or mf.get("external_outbound_volume", 0) or 0)
    internal = int(case.get("internal_volume", 0) or 0)
    parts = [
        f"This network (algorithmically inferred) contains {n} accounts "
        f"with approximately ₹{internal:,} in internal transaction volume."
    ]
    if rapid > 0:
        parts.append(
            f"The analysis detected {rapid} rapid forwarding event(s) "
            f"(inbound followed by outbound within {RAPID_FORWARDING_WINDOW_HOURS}h)."
        )
    if exit_vol > 0:
        parts.append(
            f"Approximately ₹{exit_vol:,} was transferred toward exit destinations "
            f"(external accounts and/or sink endpoints)."
        )
    rs = case.get("role_summary") or {}
    role_bits = primary_roles(rs)
    if role_bits and role_bits != "—":
        parts.append(
            f"Inferred roles within the network include: {role_bits}."
        )
    return " ".join(parts)


def build_risk_indicators(case) -> list:
    indicators = []
    mf = case.get("money_flow") or {}
    rapid = int(mf.get("rapid_forwarding_events", 0) or 0)
    if rapid:
        indicators.append(f"{rapid} rapid forwarding event(s) detected")

    exit_vol = int(mf.get("exit_outbound_volume", 0) or 0)
    if exit_vol:
        indicators.append(f"₹{exit_vol:,} transferred to exit destinations")

    ext_out = int(mf.get("external_outbound_volume", 0) or 0)
    if ext_out and ext_out != exit_vol:
        indicators.append(f"₹{ext_out:,} true external outbound volume")

    fwd_ratio = mf.get("estimated_forwarding_ratio")
    if fwd_ratio:
        indicators.append(
            f"Estimated forwarding ratio of {float(fwd_ratio):.1%} relative to network inflow"
        )

    rs = case.get("role_summary") or {}
    if rs.get("probable_mules"):
        indicators.append(f"{rs['probable_mules']} probable mule account(s)")
    if rs.get("probable_coordinators"):
        indicators.append(f"{rs['probable_coordinators']} probable coordinator(s)")
    if rs.get("probable_consolidators"):
        indicators.append(f"{rs['probable_consolidators']} probable consolidator(s)")
    if rs.get("probable_cash_out"):
        indicators.append(f"{rs['probable_cash_out']} probable cash-out account(s)")
    if rs.get("suspected_victims"):
        indicators.append(f"{rs['suspected_victims']} suspected victim account(s)")

    for f in (case.get("risk_factors") or [])[:5]:
        if isinstance(f, dict):
            label = f.get("factor") or f.get("name") or f.get("description")
            if label:
                indicators.append(str(label))
        elif f:
            indicators.append(str(f))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for i in indicators:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def build_case_report(case) -> dict:
    """JSON-serializable investigation package (no ground truth)."""
    mf = case.get("money_flow") or {}
    report = {
        "case_id": case_id(case["cluster_id"]),
        "cluster_id": case["cluster_id"],
        "risk_level": case.get("risk_level"),
        "risk_score": case.get("risk_score"),
        "account_count": case.get("size"),
        "members": list(case.get("members", [])),
        "risk_factors": case.get("risk_factors"),
        "feature_scores": case.get("feature_scores"),
        "explanation": case.get("explanation"),
        "network_structure_summary": case.get("network_structure_summary"),
        "role_summary": case.get("role_summary"),
        "money_flow_summary": {
            "internal_volume": mf.get("internal_volume"),
            "external_inbound_volume": mf.get("external_inbound_volume"),
            "external_outbound_volume": mf.get("external_outbound_volume"),
            "exit_outbound_volume": mf.get("exit_outbound_volume"),
            "estimated_forwarded_volume": mf.get("estimated_forwarded_volume"),
            "estimated_forwarding_ratio": mf.get("estimated_forwarding_ratio"),
            "rapid_forwarding_events": mf.get("rapid_forwarding_events"),
        },
        "top_money_flow_paths": (case.get("money_flow_paths") or [])[:10],
        "account_profiles": case.get("account_profiles"),
        "case_summary": build_case_summary_text(case),
        "risk_indicators": build_risk_indicators(case),
        "disclaimer": (
            "Role labels are algorithmic inferences based on transaction and "
            "network behavior and are not confirmed identities."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return strip_forbidden(report)


def case_report_markdown(report: dict) -> str:
    lines = [
        f"# Investigation Report — {report['case_id']}",
        "",
        f"**Risk level:** {report.get('risk_level')}  ",
        f"**Risk score:** {report.get('risk_score')}  ",
        f"**Accounts:** {report.get('account_count')}  ",
        "",
        "## Case summary",
        report.get("case_summary", ""),
        "",
        "## Risk indicators",
    ]
    for ind in report.get("risk_indicators") or []:
        lines.append(f"- {ind}")
    lines += [
        "",
        "## Network structure",
        str(report.get("network_structure_summary") or "—"),
        "",
        "## Role summary",
    ]
    rs = report.get("role_summary") or {}
    for k, v in rs.items():
        if v:
            lines.append(f"- {k}: {v}")
    lines += ["", "## Money-flow summary"]
    mf = report.get("money_flow_summary") or {}
    for k, v in mf.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Top money-flow paths"]
    for p in report.get("top_money_flow_paths") or []:
        path = " → ".join(p.get("path") or [])
        lines.append(
            f"- {path} | ₹{p.get('total_volume', 0):,} | "
            f"{p.get('transaction_count', 0)} tx | {p.get('time_span_hours', 0)}h"
        )
    lines += ["", "## Account profiles"]
    for ap in report.get("account_profiles") or []:
        lines.append(
            f"### {ap.get('account_id')} — {ap.get('probable_role')} "
            f"(confidence {ap.get('role_confidence')})"
        )
        for ev in ap.get("role_evidence") or []:
            lines.append(f"- {ev}")
        lines.append("")
    lines += [
        "## Disclaimer",
        report.get("disclaimer", ""),
        "",
        f"_Generated at {report.get('generated_at')}_",
    ]
    return "\n".join(lines)


def neighborhood_subgraph(G, account_id, depth=1):
    if account_id not in G:
        return nx.Graph()
    nodes = {account_id}
    frontier = {account_id}
    for _ in range(depth):
        nxt = set()
        for n in frontier:
            nxt.update(G.neighbors(n))
        nodes |= nxt
        frontier = nxt
    return G.subgraph(nodes).copy()
