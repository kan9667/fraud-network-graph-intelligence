"""Context-aware role classification for suspicious network clusters.

Roles are inferred from transaction structure relative to a candidate cluster.
Does not load or use ground-truth labels.
"""

from __future__ import annotations

import networkx as nx

from src.config import (
    MIN_ROLE_SCORE,
    MIN_ROLE_CONFIDENCE,
    SUSPICIOUS_RISK_LEVELS,
    CASHOUT_MIN_INTERNAL_INBOUND_VOLUME,
    CASHOUT_MIN_EXTERNAL_OUTBOUND_VOLUME,
    CASHOUT_MIN_PCT_OUTBOUND_TO_EXTERNAL,
    CASHOUT_MIN_FORWARDING_RATIO,
    CASHOUT_MIN_SCORE,
    CASHOUT_MAX_EXTERNAL_DESTINATIONS_FOR_CONCENTRATION,
    VICTIM_MIN_OUTBOUND_TO_CLUSTER,
    VICTIM_MAX_INTERNAL_INBOUND_RATIO,
    VICTIM_MIN_SCORE,
    RAPID_FORWARDING_WINDOW_HOURS,
    RISK_LEVEL_THRESHOLDS,
)
from src.money_flow import (
    compute_account_money_flow,
    compute_temporal_flow,
    identify_sink_accounts,
)


def classify_cluster_roles(
    members,
    G,
    accounts_df,
    transactions_df,
    cluster_risk_score=None,
    cluster_risk_level=None,
):
    """Classify roles for accounts in a candidate cluster.

    Parameters
    ----------
    cluster_risk_score / cluster_risk_level :
        Optional risk context. Cash-out and aggressive roles require a
        suspicious cluster (MEDIUM+ by default) or are heavily gated.
    """
    members = set(members)
    cluster_tx = transactions_df[
        transactions_df["from_account"].isin(members)
        | transactions_df["to_account"].isin(members)
    ]
    subgraph = G.subgraph(members)

    betweenness = _safe_betweenness(subgraph)
    degree_cent = nx.degree_centrality(subgraph)

    is_suspicious = _is_suspicious_cluster(cluster_risk_score, cluster_risk_level)
    sink_accounts = identify_sink_accounts(members, transactions_df)

    profiles = []
    for account_id in members:
        profile = _classify_single_account(
            account_id,
            members,
            subgraph,
            cluster_tx,
            transactions_df,
            betweenness,
            degree_cent,
            is_suspicious=is_suspicious,
            sink_accounts=sink_accounts,
        )
        profiles.append(profile)

    profiles.sort(key=lambda p: p["role_confidence"], reverse=True)
    summary = _compute_role_summary(profiles)
    network_summary = _generate_network_summary(summary, members)

    return profiles, summary, network_summary


def _is_suspicious_cluster(risk_score, risk_level):
    if risk_level is not None:
        return str(risk_level).upper() in SUSPICIOUS_RISK_LEVELS
    if risk_score is not None:
        return risk_score >= RISK_LEVEL_THRESHOLDS.get("MEDIUM", 30)
    # Unknown context: allow scoring but cash-out gates still apply
    return True


def _safe_betweenness(subgraph):
    n = subgraph.number_of_nodes()
    if n <= 1:
        return {}
    components = list(nx.connected_components(subgraph))
    if len(components) == 1:
        return nx.betweenness_centrality(subgraph, weight="weight", normalized=True)
    result = {}
    for comp in components:
        if len(comp) >= 2:
            sg = subgraph.subgraph(comp)
            result.update(
                nx.betweenness_centrality(sg, weight="weight", normalized=True)
            )
        else:
            result[next(iter(comp))] = 0.0
    return result


def _classify_single_account(
    account_id,
    members,
    subgraph,
    cluster_tx,
    transactions_df,
    betweenness,
    degree_cent,
    is_suspicious=True,
    sink_accounts=None,
):
    flow = compute_account_money_flow(
        account_id, members, transactions_df, sink_accounts=sink_accounts
    )
    temporal = compute_temporal_flow(
        account_id, members, transactions_df, RAPID_FORWARDING_WINDOW_HOURS
    )
    graph_feats = _compute_graph_features(
        account_id, members, subgraph, cluster_tx, betweenness, degree_cent
    )
    feats = {**flow, **temporal, **graph_feats, "is_suspicious_cluster": is_suspicious}

    # Non-suspicious clusters: do not invent fraud roles from ordinary activity
    if not is_suspicious:
        return {
            "account_id": account_id,
            "probable_role": "unknown",
            "role_confidence": 0,
            "role_evidence": [
                "Cluster risk is LOW — role labels reserved for suspicious "
                "candidate networks; account left unclassified"
            ],
        }

    mule_score, mule_evidence = _score_mule(feats)
    coord_score, coord_evidence = _score_coordinator(feats, subgraph)
    consolidator_score, consolidator_evidence = _score_consolidator(feats)
    cash_out_score, cash_out_evidence = _score_cash_out(feats)
    victim_score, victim_evidence = _score_victim(feats)

    role_scores = [
        ("probable_mule", mule_score, mule_evidence),
        ("probable_coordinator", coord_score, coord_evidence),
        ("probable_consolidator", consolidator_score, consolidator_evidence),
        ("probable_cash_out", cash_out_score, cash_out_evidence),
        ("suspected_victim", victim_score, victim_evidence),
    ]

    best_role, best_score, best_evidence = _resolve_role(role_scores, feats)

    if best_score < MIN_ROLE_SCORE:
        best_role = "unknown"
        best_score = max(0, int(best_score * 0.5))
        best_evidence = [_evidence_unknown(feats)]

    # Confidence from selected score and gap vs best alternative.
    # Priority rules may pick a role that is not the raw max score; in that
    # case keep confidence proportional to the selected score rather than
    # collapsing margin to zero against a non-selected higher raw score.
    other_scores = [s for n, s, _ in role_scores if n != best_role]
    best_other = max(other_scores) if other_scores else 0
    margin = best_score - best_other
    if margin >= 0:
        confidence = min(
            100, int(best_score * (0.5 + 0.5 * min(margin / 30, 1.0)))
        )
    else:
        # Priority override (e.g. consolidator over cash-out)
        confidence = min(100, int(best_score * 0.85))
    if confidence < MIN_ROLE_CONFIDENCE and best_role != "unknown":
        best_role = "unknown"
        best_evidence = [_evidence_unknown(feats)]
        confidence = max(0, int(confidence * 0.5))

    return {
        "account_id": account_id,
        "probable_role": best_role,
        "role_confidence": confidence,
        "role_evidence": best_evidence[:5],
    }


def _resolve_role(role_scores, feats):
    """Priority resolution with multi-signal gates already applied in scorers.

    Order (after gates):
      1. Victim when strongest and gated
      2. Consolidator when clear multi-source concentration (before cash-out)
      3. Cash-out when multi-signal exit pattern beats mule/consol
      4. Coordinator when clearly structural (high margin)
      5. Mule vs remaining by score
    """
    scores = {name: score for name, score, _ in role_scores}
    by_name = {name: (score, ev) for name, score, ev in role_scores}

    consol_s = scores.get("probable_consolidator", 0)
    mule_s = scores.get("probable_mule", 0)
    cash_s = scores.get("probable_cash_out", 0)
    coord_s = scores.get("probable_coordinator", 0)
    victim_s = scores.get("suspected_victim", 0)

    if (
        victim_s >= VICTIM_MIN_SCORE
        and victim_s >= mule_s
        and victim_s >= consol_s
        and victim_s >= cash_s
    ):
        s, ev = by_name["suspected_victim"]
        return "suspected_victim", s, ev

    sources = feats.get("unique_internal_sources", 0) or 0
    exit_dests = feats.get("unique_exit_destinations", 0) or 0
    int_dests = feats.get("unique_internal_destinations", 0) or 0
    out_dests = exit_dests + int_dests

    # Strong concentration → consolidator even if some exit volume exists
    if (
        consol_s >= MIN_ROLE_SCORE
        and sources >= 3
        and out_dests <= 2
    ):
        s, ev = by_name["probable_consolidator"]
        return "probable_consolidator", s, ev

    # Cash-out only when multi-signal score clears a higher bar and
    # is at least as strong as mule (mule wins ties on internal passthrough)
    if (
        cash_s >= CASHOUT_MIN_SCORE
        and cash_s > mule_s
        and cash_s >= consol_s
    ):
        s, ev = by_name["probable_cash_out"]
        return "probable_cash_out", s, ev

    behavioral_best = max(mule_s, consol_s, cash_s, victim_s)
    if coord_s >= 60 and coord_s > behavioral_best + 35:
        s, ev = by_name["probable_coordinator"]
        return "probable_coordinator", s, ev

    if consol_s >= mule_s and consol_s >= MIN_ROLE_SCORE:
        s, ev = by_name["probable_consolidator"]
        return "probable_consolidator", s, ev

    if mule_s >= MIN_ROLE_SCORE and mule_s >= consol_s:
        s, ev = by_name["probable_mule"]
        return "probable_mule", s, ev

    if coord_s >= MIN_ROLE_SCORE:
        s, ev = by_name["probable_coordinator"]
        return "probable_coordinator", s, ev

    best = max(role_scores, key=lambda x: x[1])
    return best


def _compute_graph_features(
    account_id, members, subgraph, cluster_tx, betweenness, degree_cent
):
    sent = cluster_tx[cluster_tx["from_account"] == account_id]
    received = cluster_tx[cluster_tx["to_account"] == account_id]

    inbound_count = len(received)
    outbound_count = len(sent)
    inbound_volume = int(received["amount"].sum()) if inbound_count > 0 else 0
    outbound_volume = int(sent["amount"].sum()) if outbound_count > 0 else 0

    avg_inbound = round(inbound_volume / inbound_count, 2) if inbound_count > 0 else 0.0
    avg_outbound = round(outbound_volume / outbound_count, 2) if outbound_count > 0 else 0.0

    betw = betweenness.get(account_id, 0.0)
    deg_cent_val = degree_cent.get(account_id, 0.0)
    degree = subgraph.degree(account_id) if account_id in subgraph else 0
    weighted_deg = (
        sum(d.get("weight", 1) for _, _, d in subgraph.edges(account_id, data=True))
        if account_id in subgraph
        else 0
    )

    return {
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "avg_inbound_amount": avg_inbound,
        "avg_outbound_amount": avg_outbound,
        "degree": degree,
        "weighted_degree": weighted_deg,
        "degree_centrality": round(deg_cent_val, 4),
        "betweenness_centrality": round(betw, 4),
        "cluster_size": subgraph.number_of_nodes(),
    }


# ---------------------------------------------------------------------------
# Role scorers
# ---------------------------------------------------------------------------

def _score_mule(f):
    """Pass-through: multiple sources/destinations, high forwarding, often internal."""
    s = 0
    ev = []

    inbound_src = f["unique_internal_sources"] + f["unique_external_sources"]
    outbound_dst = f["unique_internal_destinations"] + f["unique_external_destinations"]
    if inbound_src < 2 or outbound_dst < 2:
        return 0, ["Limited send-and-forward activity detected"]

    # Prefer internal forwarding (stays in network) for mule vs cash-out
    int_in = f["internal_inbound_volume"]
    int_out = f["internal_outbound_volume"]
    ext_out = f["external_outbound_volume"]
    inbound = f["inbound_volume"]
    outbound = f["outbound_volume"]

    if inbound <= 0 or outbound <= 0:
        return 0, ["No bidirectional flow"]

    fwd_ratio = f["estimated_forwarding_ratio"]
    pct_fwd = round(outbound / inbound * 100, 1) if inbound > 0 else 0.0

    if pct_fwd >= 60:
        s += 35
        ev.append(
            f"Forwarded {pct_fwd}% of received funds "
            f"(₹{outbound:,} of ₹{inbound:,})"
        )
    elif pct_fwd >= 30:
        s += 18
        ev.append(f"Forwarded {pct_fwd}% of received funds")

    if inbound_src >= 3 and outbound_dst >= 3:
        s += 20
        ev.append(
            f"Multiple sources ({inbound_src}) and destinations ({outbound_dst})"
        )
    elif inbound_src >= 2 and outbound_dst >= 2:
        s += 10

    vol = outbound + inbound
    if vol >= 500_000:
        s += 15
        ev.append(f"Total flow of ₹{vol:,} suggests active pass-through role")
    elif vol >= 100_000:
        s += 8

    # Internal passthrough bonus (money stays in network)
    if int_in > 0 and int_out >= int_in * 0.4 and int_out >= ext_out:
        s += 20
        ev.append(
            f"Primarily forwards within the network "
            f"(₹{int_out:,} internal out vs ₹{ext_out:,} external out)"
        )

    if f.get("rapid_forwarding_events", 0) >= 2:
        s += 15
        med = f.get("median_inbound_to_outbound_delay_hours")
        if med is not None:
            ev.append(
                f"{f['rapid_forwarding_events']} rapid forwarding event(s); "
                f"median inbound→outbound delay {med:.2f} hours"
            )
        else:
            ev.append(
                f"{f['rapid_forwarding_events']} rapid forwarding event(s) "
                f"within {RAPID_FORWARDING_WINDOW_HOURS}h"
            )

    if 2 <= f["outbound_count"] <= 20 and 2 <= f["inbound_count"] <= 20:
        s += 10

    return min(s, 100), ev


def _score_coordinator(f, subgraph):
    s = 0
    ev = []

    betw = f["betweenness_centrality"]
    n = subgraph.number_of_nodes()
    if n < 3:
        return 0, ["Network too small for coordinator analysis"]

    deg = f["degree"]
    avg_deg = (2 * subgraph.number_of_edges()) / n if n > 0 else 1

    articulation_points = set(nx.articulation_points(subgraph)) if n >= 3 else set()
    is_articulation = f["account_id"] in articulation_points

    if is_articulation and betw > 0.1:
        s += 40
        ev.append(
            f"Articulation point with betweenness {betw} — "
            f"removing this account would fragment the network"
        )

    if n >= 5 and betw > 0.3:
        s += 30
        ev.append(
            f"Betweenness centrality {betw} in a {n}-node network — "
            f"routes transactions between otherwise separate subgroups"
        )
    elif n >= 5 and betw > 0.15:
        s += 15
        ev.append(f"Betweenness centrality {betw} suggests bridging role")

    if deg > avg_deg * 1.5 and n >= 5:
        s += 15
        ev.append(
            f"Degree {deg} is {((deg / avg_deg) - 1) * 100:.0f}% above "
            f"cluster average ({avg_deg:.1f})"
        )

    # Connectivity breadth within cluster
    internal_cp = f["unique_internal_sources"] + f["unique_internal_destinations"]
    if internal_cp >= n * 0.6 and n >= 5:
        s += 15
        ev.append(f"Linked to {internal_cp} counterparties inside the cluster")

    return min(s, 100), ev


def _score_consolidator(f):
    s = 0
    ev = []

    ib = f["inbound_volume"]
    ob = f["outbound_volume"]
    int_in = f["internal_inbound_volume"]
    int_sources = f["unique_internal_sources"]
    int_dests = f["unique_internal_destinations"]
    ext_dests = f["unique_external_destinations"]

    if ib > 0 and ob > 0:
        ratio = ib / ob
        if ratio >= 2.0:
            s += 30
            ev.append(
                f"Receives {ratio:.1f}x more than sends out "
                f"(₹{ib:,} in vs ₹{ob:,} out)"
            )
        elif ratio >= 1.3:
            s += 15
            ev.append(f"Receives more than sends (₹{ib:,} in vs ₹{ob:,} out)")

    # Many internal sources, few destinations (internal or external cash-out hop)
    out_dests = int_dests + ext_dests
    if int_sources >= 4 and out_dests <= 2:
        s += 35
        ev.append(
            f"Receives from {int_sources} cluster accounts but sends to only "
            f"{out_dests} destination(s) — concentration pattern"
        )
    elif int_sources >= 3 and int_sources > out_dests:
        s += 18
        ev.append(
            f"Receives from {int_sources} accounts, "
            f"sends to {out_dests}"
        )

    if int_in >= 500_000:
        s += 20
        ev.append(f"Consolidated ₹{int_in:,} from within the suspicious network")
    elif int_in >= 100_000:
        s += 10

    if f["internal_inbound_count"] >= 3 and f["avg_inbound_amount"] > 50_000:
        s += 12
        ev.append(
            f"Average inbound transaction of ₹{f['avg_inbound_amount']:,.0f} "
            f"across {f['inbound_count']} receipts"
        )

    return min(s, 100), ev


def _score_cash_out(f):
    """Multi-signal cash-out: core/cluster inbound → concentrated exit flow.

    Exit destinations include true external accounts and in-cluster sink
    accounts (Louvain often absorbs cash-out endpoints into the community).

    Ordinary outbound transfers alone must NOT produce a cash-out role.
    """
    s = 0
    ev = []

    # Prefer core (non-sink) inbound; fall back to all internal inbound
    core_in = f.get("core_inbound_volume", 0) or 0
    int_in = f["internal_inbound_volume"]
    suspicious_in = max(core_in, int_in)

    exit_out = f.get("exit_outbound_volume", 0) or f["external_outbound_volume"]
    ext_out = f["external_outbound_volume"]
    pct_exit = f.get("percentage_outbound_to_exit", 0.0) or f[
        "percentage_outbound_to_external"
    ]
    fwd_ratio = max(
        f.get("exit_forwarding_ratio", 0.0) or 0.0,
        f.get("estimated_forwarding_ratio", 0.0) or 0.0,
    )
    exit_dests = f.get("unique_exit_destinations", 0) or f[
        "unique_external_destinations"
    ]
    core_sources = f.get("unique_core_sources", 0) or f["unique_internal_sources"]
    is_suspicious = f.get("is_suspicious_cluster", True)

    # Sink endpoints themselves are destinations, not cash-out agents
    if f.get("is_sink"):
        return 0, ["Account is a sink endpoint, not a cash-out intermediary"]

    # ---- Hard gates (all must pass for any meaningful score) ----
    if not is_suspicious:
        return 0, [
            "Cluster is not classified as suspicious — cash-out not inferred"
        ]

    # Fan-out to many in-cluster destinations (even if sink-like) is mule-style
    # passthrough, not a concentrated cash-out exit — unless true external volume.
    int_dests = f.get("unique_internal_destinations", 0) or 0
    int_sources = f.get("unique_internal_sources", 0) or 0
    if ext_out == 0 and int_dests >= 3 and int_sources >= 2:
        return 0, [
            "Fan-out to multiple in-network destinations looks like mule "
            "passthrough rather than concentrated cash-out"
        ]

    if suspicious_in < CASHOUT_MIN_INTERNAL_INBOUND_VOLUME:
        return 0, [
            f"Insufficient suspicious inbound (₹{suspicious_in:,} from network; "
            f"need ≥ ₹{CASHOUT_MIN_INTERNAL_INBOUND_VOLUME:,})"
        ]

    if exit_out < CASHOUT_MIN_EXTERNAL_OUTBOUND_VOLUME:
        return 0, [
            f"Insufficient exit outflow (₹{exit_out:,} to external/sink "
            f"destinations; need ≥ ₹{CASHOUT_MIN_EXTERNAL_OUTBOUND_VOLUME:,})"
        ]

    if pct_exit < CASHOUT_MIN_PCT_OUTBOUND_TO_EXTERNAL:
        return 0, [
            f"Only {pct_exit:.0f}% of outbound volume is exit-bound "
            f"(need ≥ {CASHOUT_MIN_PCT_OUTBOUND_TO_EXTERNAL:.0f}%)"
        ]

    if fwd_ratio < CASHOUT_MIN_FORWARDING_RATIO:
        return 0, [
            f"Exit flow not sufficiently linked to suspicious inbound "
            f"(forwarding ratio {fwd_ratio:.2f})"
        ]

    # ---- Soft scoring (multi-signal) ----
    s += 25
    ev.append(
        f"Received ₹{suspicious_in:,} from {core_sources} "
        f"account(s) within the suspicious network"
    )

    s += 25
    if ext_out > 0 and ext_out == exit_out:
        ev.append(
            f"Transferred ₹{exit_out:,} to {exit_dests} external destination(s)"
        )
    else:
        ev.append(
            f"Transferred ₹{exit_out:,} to {exit_dests} exit destination(s) "
            f"(external accounts and/or sink endpoints)"
        )

    if pct_exit >= 80:
        s += 15
    elif pct_exit >= 60:
        s += 10
    else:
        s += 5
    ev.append(
        f"{pct_exit:.0f}% of outbound volume left the active network core"
    )

    if fwd_ratio >= 0.7:
        s += 15
    elif fwd_ratio >= 0.5:
        s += 10
    else:
        s += 5

    if 1 <= exit_dests <= CASHOUT_MAX_EXTERNAL_DESTINATIONS_FOR_CONCENTRATION:
        s += 10
        ev.append(
            f"Exit outflow concentrated across {exit_dests} destination(s)"
        )

    rapid = f.get("rapid_forwarding_events", 0)
    med = f.get("median_inbound_to_outbound_delay_hours")
    if rapid >= 1 and med is not None:
        s += 10
        if med < 1:
            mins = med * 60
            ev.append(
                f"Median delay between suspicious inbound and outbound "
                f"transfers was {mins:.0f} minutes"
            )
        else:
            ev.append(
                f"Median delay between suspicious inbound and outbound "
                f"transfers was {med:.2f} hours"
            )
    elif rapid >= 1:
        s += 5
        ev.append(
            f"{rapid} rapid forwarding event(s) within "
            f"{RAPID_FORWARDING_WINDOW_HOURS}h window"
        )

    if exit_out >= 500_000 and suspicious_in >= 500_000:
        s += 10
    elif exit_out >= 200_000:
        s += 5

    return min(s, 100), ev


def _score_victim(f):
    """Suspected victim: primarily external to the ring's internal activity,
    sends into the cluster, limited reciprocal traffic.
    """
    s = 0
    ev = []

    # Outbound into cluster
    int_out = f["internal_outbound_volume"]
    int_in = f["internal_inbound_volume"]
    ext_in = f["external_inbound_volume"]
    ext_out = f["external_outbound_volume"]
    int_dests = f["unique_internal_destinations"]
    int_sources = f["unique_internal_sources"]

    if int_out < VICTIM_MIN_OUTBOUND_TO_CLUSTER:
        return 0, ["Limited transfers into the suspicious cluster"]

    # One-way: sends into cluster, little/no receive from cluster
    if int_in == 0 and int_out > 0:
        s += 40
        ev.append(
            f"Sent ₹{int_out:,} into the network but received nothing back "
            f"from cluster accounts"
        )
    elif int_out > 0 and int_in / max(int_out, 1) <= VICTIM_MAX_INTERNAL_INBOUND_RATIO:
        s += 25
        ev.append(
            f"Primarily one-way flow into the cluster "
            f"(₹{int_out:,} in-cluster out vs ₹{int_in:,} in-cluster in)"
        )
    else:
        # Strong reciprocal internal activity → not a classic victim
        return 0, ["Reciprocal cluster activity inconsistent with victim pattern"]

    # Limited reciprocal / fan-out into cluster
    if int_dests >= 1 and int_sources == 0:
        s += 20
        ev.append(
            f"Sends to {int_dests} account(s) in the cluster but receives from none"
        )
    elif int_dests >= 2 and int_sources <= 1:
        s += 12
        ev.append(
            f"Sends to {int_dests} cluster accounts with minimal reciprocal inflow"
        )

    # Low participation as internal hub (not mule/coordinator)
    if f["degree"] <= 3 and f["betweenness_centrality"] < 0.15:
        s += 10
        ev.append("Low network centrality — peripheral to the cluster structure")

    # External accounts that mainly feed the cluster (if they have little external out)
    if ext_out == 0 and int_out > 0:
        s += 10
        ev.append("No external destinations — funds flow into the cluster only")

    # Do not treat high external cash-out style accounts as victims
    if ext_out >= CASHOUT_MIN_EXTERNAL_OUTBOUND_VOLUME and int_in >= CASHOUT_MIN_INTERNAL_INBOUND_VOLUME:
        s = max(0, s - 30)

    if s < VICTIM_MIN_SCORE:
        # Return score as-is for comparison but evidence of weak pattern
        if not ev:
            ev = ["Insufficient victim-pattern evidence"]

    return min(s, 100), ev


def _evidence_unknown(f):
    vol = f.get("outbound_volume", 0) + f.get("inbound_volume", 0)
    cp = (
        f.get("unique_internal_sources", 0)
        + f.get("unique_internal_destinations", 0)
        + f.get("unique_external_sources", 0)
        + f.get("unique_external_destinations", 0)
    )
    return (
        f"Account with ₹{vol:,} in flow across {cp} counterparties "
        f"— does not match a single role profile strongly"
    )


def _compute_role_summary(profiles):
    summary = {
        "probable_mules": 0,
        "probable_coordinators": 0,
        "probable_consolidators": 0,
        "probable_cash_out": 0,
        "suspected_victims": 0,
        "unknown": 0,
    }
    role_map = {
        "probable_mule": "probable_mules",
        "probable_coordinator": "probable_coordinators",
        "probable_consolidator": "probable_consolidators",
        "probable_cash_out": "probable_cash_out",
        "suspected_victim": "suspected_victims",
        "unknown": "unknown",
    }
    for p in profiles:
        key = role_map.get(p["probable_role"])
        if key:
            summary[key] += 1
    return summary


def _generate_network_summary(summary, members):
    n = len(members)
    parts = []
    if summary["probable_mules"] > 0:
        parts.append(f"{summary['probable_mules']} probable mule(s)")
    if summary["probable_coordinators"] > 0:
        parts.append(f"{summary['probable_coordinators']} probable coordinator(s)")
    if summary["probable_consolidators"] > 0:
        parts.append(f"{summary['probable_consolidators']} probable consolidator(s)")
    if summary["probable_cash_out"] > 0:
        parts.append(f"{summary['probable_cash_out']} probable cash-out account(s)")
    if summary["suspected_victims"] > 0:
        parts.append(f"{summary['suspected_victims']} suspected victim(s)")
    if summary["unknown"] > 0:
        parts.append(f"{summary['unknown']} unclassified")

    if not parts:
        return f"{n}-account network; role analysis incomplete"

    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = "; ".join(parts[:-1]) + "; and " + parts[-1]
    return f"{n}-account network with {joined}."
