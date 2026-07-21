"""Account-level and cluster-level money-flow intelligence.

All volumes and paths are derived from actual transactions relative to a
cluster membership set. External = counterparty not in the cluster.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import (
    RAPID_FORWARDING_WINDOW_HOURS,
    MONEY_FLOW_MAX_PATH_LENGTH,
    MONEY_FLOW_TOP_PATHS,
    MONEY_FLOW_TOP_FLOWS,
    MONEY_FLOW_MIN_EDGE_VOLUME,
)


def _empty_account_flow(account_id):
    return {
        "account_id": account_id,
        "internal_inbound_volume": 0,
        "internal_outbound_volume": 0,
        "external_inbound_volume": 0,
        "external_outbound_volume": 0,
        "internal_inbound_count": 0,
        "internal_outbound_count": 0,
        "external_inbound_count": 0,
        "external_outbound_count": 0,
        "inbound_volume": 0,
        "outbound_volume": 0,
        "percentage_inbound_from_cluster": 0.0,
        "percentage_outbound_to_external": 0.0,
        "estimated_forwarded_volume": 0,
        "estimated_forwarding_ratio": 0.0,
        "unique_internal_sources": 0,
        "unique_internal_destinations": 0,
        "unique_external_sources": 0,
        "unique_external_destinations": 0,
    }


def identify_sink_accounts(members, transactions_df, sink_out_in_ratio=0.2):
    """Accounts that primarily receive and rarely send (exit/sink endpoints).

    Includes non-members that only appear as destinations. Used when Louvain
    folds cash-out destinations into the same community as the ring.
    """
    members = set(members)
    relevant = transactions_df[
        transactions_df["from_account"].isin(members)
        | transactions_df["to_account"].isin(members)
    ]
    if len(relevant) == 0:
        return set()

    accounts = set(relevant["from_account"]) | set(relevant["to_account"])
    sinks = set()
    for aid in accounts:
        sent = relevant[relevant["from_account"] == aid]
        received = relevant[relevant["to_account"] == aid]
        in_vol = int(received["amount"].sum()) if len(received) else 0
        out_vol = int(sent["amount"].sum()) if len(sent) else 0
        if in_vol <= 0:
            continue
        if out_vol == 0 or out_vol <= sink_out_in_ratio * in_vol:
            sinks.add(aid)
    return sinks


def compute_account_money_flow(account_id, members, transactions_df, sink_accounts=None):
    """Compute money-flow features for one account relative to a cluster.

    Exit destinations = true external (not in members) OR sink accounts.
    This keeps cash-out detection meaningful when community detection absorbs
    exit endpoints into the cluster.
    """
    members = set(members)
    if sink_accounts is None:
        sink_accounts = identify_sink_accounts(members, transactions_df)

    sent = transactions_df[transactions_df["from_account"] == account_id]
    received = transactions_df[transactions_df["to_account"] == account_id]

    recv_int = received[received["from_account"].isin(members)]
    recv_ext = received[~received["from_account"].isin(members)]
    sent_int = sent[sent["to_account"].isin(members)]
    sent_ext = sent[~sent["to_account"].isin(members)]

    # Exit destinations: outside cluster OR in-cluster sinks (not self)
    exit_mask = sent["to_account"].map(
        lambda d: d != account_id and (d not in members or d in sink_accounts)
    )
    sent_exit = sent[exit_mask] if len(sent) else sent

    # Core (non-sink) internal sources — funds from active ring members
    core_sources = members - sink_accounts - {account_id}
    recv_core = received[received["from_account"].isin(core_sources)]

    int_in_vol = int(recv_int["amount"].sum()) if len(recv_int) else 0
    int_out_vol = int(sent_int["amount"].sum()) if len(sent_int) else 0
    ext_in_vol = int(recv_ext["amount"].sum()) if len(recv_ext) else 0
    ext_out_vol = int(sent_ext["amount"].sum()) if len(sent_ext) else 0
    exit_out_vol = int(sent_exit["amount"].sum()) if len(sent_exit) else 0
    core_in_vol = int(recv_core["amount"].sum()) if len(recv_core) else 0

    inbound_volume = int_in_vol + ext_in_vol
    outbound_volume = int_out_vol + ext_out_vol

    pct_in_from_cluster = (
        round(int_in_vol / inbound_volume * 100, 2) if inbound_volume > 0 else 0.0
    )
    pct_out_to_external = (
        round(ext_out_vol / outbound_volume * 100, 2) if outbound_volume > 0 else 0.0
    )
    pct_out_to_exit = (
        round(exit_out_vol / outbound_volume * 100, 2) if outbound_volume > 0 else 0.0
    )

    estimated_forwarded = int(min(int_in_vol, outbound_volume)) if int_in_vol > 0 else 0
    estimated_forwarded_external = (
        int(min(int_in_vol, ext_out_vol)) if int_in_vol > 0 and ext_out_vol > 0 else 0
    )
    estimated_forwarded_exit = (
        int(min(max(core_in_vol, int_in_vol), exit_out_vol))
        if (core_in_vol > 0 or int_in_vol > 0) and exit_out_vol > 0
        else 0
    )
    fwd_for_ratio = max(
        estimated_forwarded_exit,
        estimated_forwarded_external,
        estimated_forwarded,
    )
    base_in = max(core_in_vol, int_in_vol)
    estimated_forwarding_ratio = (
        round(fwd_for_ratio / base_in, 4) if base_in > 0 else 0.0
    )
    exit_forwarding_ratio = (
        round(estimated_forwarded_exit / base_in, 4) if base_in > 0 else 0.0
    )

    unique_exit_dests = (
        int(sent_exit["to_account"].nunique()) if len(sent_exit) else 0
    )
    unique_core_sources = (
        int(recv_core["from_account"].nunique()) if len(recv_core) else 0
    )

    return {
        "account_id": account_id,
        "internal_inbound_volume": int_in_vol,
        "internal_outbound_volume": int_out_vol,
        "external_inbound_volume": ext_in_vol,
        "external_outbound_volume": ext_out_vol,
        "exit_outbound_volume": exit_out_vol,
        "core_inbound_volume": core_in_vol,
        "internal_inbound_count": int(len(recv_int)),
        "internal_outbound_count": int(len(sent_int)),
        "external_inbound_count": int(len(recv_ext)),
        "external_outbound_count": int(len(sent_ext)),
        "exit_outbound_count": int(len(sent_exit)),
        "inbound_volume": inbound_volume,
        "outbound_volume": outbound_volume,
        "percentage_inbound_from_cluster": pct_in_from_cluster,
        "percentage_outbound_to_external": pct_out_to_external,
        "percentage_outbound_to_exit": pct_out_to_exit,
        "estimated_forwarded_volume": estimated_forwarded,
        "estimated_forwarded_external_volume": estimated_forwarded_external,
        "estimated_forwarded_exit_volume": estimated_forwarded_exit,
        "estimated_forwarding_ratio": estimated_forwarding_ratio,
        "exit_forwarding_ratio": exit_forwarding_ratio,
        "unique_internal_sources": int(recv_int["from_account"].nunique()) if len(recv_int) else 0,
        "unique_internal_destinations": int(sent_int["to_account"].nunique()) if len(sent_int) else 0,
        "unique_external_sources": int(recv_ext["from_account"].nunique()) if len(recv_ext) else 0,
        "unique_external_destinations": int(sent_ext["to_account"].nunique()) if len(sent_ext) else 0,
        "unique_exit_destinations": unique_exit_dests,
        "unique_core_sources": unique_core_sources,
        "is_sink": account_id in sink_accounts,
    }


def compute_temporal_flow(
    account_id,
    members,
    transactions_df,
    window_hours=None,
):
    """Inbound→outbound delay stats and rapid-forwarding events for one account.

    A rapid forwarding event: receive then later send within `window_hours`.
    Delays are computed pairing each outbound with the most recent prior inbound.
    """
    if window_hours is None:
        window_hours = RAPID_FORWARDING_WINDOW_HOURS
    window_sec = float(window_hours) * 3600.0

    members = set(members)
    received = transactions_df[transactions_df["to_account"] == account_id].copy()
    sent = transactions_df[transactions_df["from_account"] == account_id].copy()

    empty = {
        "median_inbound_to_outbound_delay_hours": None,
        "min_inbound_to_outbound_delay_hours": None,
        "max_inbound_to_outbound_delay_hours": None,
        "rapid_forwarding_events": 0,
        "rapid_forwarding_volume": 0,
    }

    if len(received) == 0 or len(sent) == 0:
        return empty

    # Ensure datetime dtype (defensive if caller passed raw CSV strings)
    if not pd.api.types.is_datetime64_any_dtype(received["timestamp"]):
        received["timestamp"] = pd.to_datetime(
            received["timestamp"], errors="coerce", format="ISO8601"
        )
    if not pd.api.types.is_datetime64_any_dtype(sent["timestamp"]):
        sent["timestamp"] = pd.to_datetime(
            sent["timestamp"], errors="coerce", format="ISO8601"
        )

    received = received.dropna(subset=["timestamp"]).sort_values("timestamp")
    sent = sent.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(received) == 0 or len(sent) == 0:
        return empty

    # Prefer suspicious (cluster-internal) inbound when available for delay pairing
    recv_int = received[received["from_account"].isin(members)]
    inbound_pool = recv_int if len(recv_int) > 0 else received

    delays_sec = []
    rapid_events = 0
    rapid_volume = 0

    in_times = inbound_pool["timestamp"].tolist()
    in_amounts = inbound_pool["amount"].tolist()
    out_times = sent["timestamp"].tolist()
    out_amounts = sent["amount"].tolist()
    out_dests = sent["to_account"].tolist()

    # Greedy: for each outbound, match most recent unused inbound before it
    used_in = set()
    for oi, (ots, oamt, odest) in enumerate(zip(out_times, out_amounts, out_dests)):
        best_ii = None
        best_ts = None
        for ii, (its, iamt) in enumerate(zip(in_times, in_amounts)):
            if ii in used_in:
                continue
            if its <= ots:
                if best_ts is None or its > best_ts:
                    best_ts = its
                    best_ii = ii
        if best_ii is None:
            continue
        used_in.add(best_ii)
        delay = (ots - in_times[best_ii]).total_seconds()
        if delay < 0:
            continue
        delays_sec.append(delay)
        if delay <= window_sec:
            rapid_events += 1
            rapid_volume += int(min(in_amounts[best_ii], oamt))

    if not delays_sec:
        return empty

    delays_h = np.array(delays_sec) / 3600.0
    return {
        "median_inbound_to_outbound_delay_hours": round(float(np.median(delays_h)), 4),
        "min_inbound_to_outbound_delay_hours": round(float(np.min(delays_h)), 4),
        "max_inbound_to_outbound_delay_hours": round(float(np.max(delays_h)), 4),
        "rapid_forwarding_events": int(rapid_events),
        "rapid_forwarding_volume": int(rapid_volume),
    }


def _aggregate_directed_edges(transactions_df, min_volume=MONEY_FLOW_MIN_EDGE_VOLUME):
    """Aggregate directed (from, to) edges with volume, count, time span."""
    if len(transactions_df) == 0:
        return {}
    edges = {}
    for (src, dst), grp in transactions_df.groupby(["from_account", "to_account"]):
        vol = int(grp["amount"].sum())
        if vol < min_volume:
            continue
        ts = grp["timestamp"]
        first, last = ts.min(), ts.max()
        span_h = (last - first).total_seconds() / 3600.0 if len(grp) > 1 else 0.0
        edges[(src, dst)] = {
            "source_account": src,
            "destination_account": dst,
            "total_volume": vol,
            "transaction_count": int(len(grp)),
            "time_span_hours": round(span_h, 4),
            "first_timestamp": first,
            "last_timestamp": last,
        }
    return edges


def find_money_flow_paths(
    members,
    transactions_df,
    role_by_account=None,
    max_path_length=None,
    top_n=None,
    min_edge_volume=None,
):
    """Find high-volume short money-flow paths involving the cluster.

    Prioritizes:
    - high-volume edges
    - short paths (length 1–max)
    - paths connecting role candidates
    - paths with external endpoints

    Does not enumerate every graph path.
    """
    if max_path_length is None:
        max_path_length = MONEY_FLOW_MAX_PATH_LENGTH
    if top_n is None:
        top_n = MONEY_FLOW_TOP_PATHS
    if min_edge_volume is None:
        min_edge_volume = MONEY_FLOW_MIN_EDGE_VOLUME
    if role_by_account is None:
        role_by_account = {}

    members = set(members)
    # Transactions touching the cluster
    cluster_tx = transactions_df[
        transactions_df["from_account"].isin(members)
        | transactions_df["to_account"].isin(members)
    ]
    edges = _aggregate_directed_edges(cluster_tx, min_volume=min_edge_volume)
    if not edges:
        return []

    # Adjacency for short path expansion (outgoing)
    out_adj = defaultdict(list)
    for (src, dst), meta in edges.items():
        out_adj[src].append((dst, meta))

    candidates = []

    # Length-1 paths: all edges (already high-signal)
    for (src, dst), meta in edges.items():
        path = [src, dst]
        score = _path_priority_score(path, meta["total_volume"], members, role_by_account)
        candidates.append({
            "source_account": src,
            "destination_account": dst,
            "path": path,
            "total_volume": meta["total_volume"],
            "transaction_count": meta["transaction_count"],
            "time_span_hours": meta["time_span_hours"],
            "_score": score,
        })

    # Length 2..max: expand only from high-volume edges and role-relevant nodes
    role_nodes = {
        a for a, r in role_by_account.items()
        if r and r not in ("unknown",)
    }
    seed_nodes = set(members) | role_nodes
    # Also seeds from top volume edges
    top_edges = sorted(edges.values(), key=lambda e: e["total_volume"], reverse=True)[:30]
    for e in top_edges:
        seed_nodes.add(e["source_account"])
        seed_nodes.add(e["destination_account"])

    # Limited expansion: only follow top-k outgoing edges per node (by volume)
    max_branch = 4
    max_candidates = 500
    for start in seed_nodes:
        if len(candidates) >= max_candidates:
            break
        stack = [([start], None, 0, 0)]  # path, first_ts, vol, count
        while stack and len(candidates) < max_candidates:
            path, first_ts, acc_vol, acc_cnt = stack.pop()
            if len(path) - 1 >= max_path_length:
                continue
            curr = path[-1]
            outgoing = sorted(
                out_adj.get(curr, []),
                key=lambda x: x[1]["total_volume"],
                reverse=True,
            )[:max_branch]
            for nxt, meta in outgoing:
                if nxt in path:
                    continue  # no cycles
                new_path = path + [nxt]
                new_vol = (
                    meta["total_volume"]
                    if acc_vol == 0
                    else min(acc_vol, meta["total_volume"])
                )
                new_cnt = acc_cnt + meta["transaction_count"]
                new_first = (
                    meta["first_timestamp"]
                    if first_ts is None
                    else min(first_ts, meta["first_timestamp"])
                )
                new_last = meta["last_timestamp"]
                span_h = (new_last - new_first).total_seconds() / 3600.0
                if len(new_path) >= 3:
                    # length-1 already recorded; add multi-hop only
                    score = _path_priority_score(
                        new_path, new_vol, members, role_by_account
                    )
                    candidates.append({
                        "source_account": new_path[0],
                        "destination_account": new_path[-1],
                        "path": new_path,
                        "total_volume": int(new_vol),
                        "transaction_count": int(new_cnt),
                        "time_span_hours": round(span_h, 4),
                        "_score": score,
                    })
                if len(new_path) - 1 < max_path_length:
                    stack.append((new_path, new_first, new_vol, new_cnt))

    # Deduplicate by path tuple, keep highest volume
    best = {}
    for c in candidates:
        key = tuple(c["path"])
        if key not in best or c["total_volume"] > best[key]["total_volume"]:
            best[key] = c

    ranked = sorted(
        best.values(),
        key=lambda c: (c["_score"], c["total_volume"]),
        reverse=True,
    )[:top_n]

    for c in ranked:
        c.pop("_score", None)
    return ranked


def _path_priority_score(path, volume, members, role_by_account):
    """Heuristic priority: volume, short length, roles, external endpoints."""
    length = len(path) - 1
    length_bonus = max(0, 5 - length) * 10_000  # prefer shorter
    vol_score = volume

    role_bonus = 0
    for node in path:
        role = role_by_account.get(node, "unknown")
        if role and role != "unknown":
            role_bonus += 50_000

    external_bonus = 0
    if path[0] not in members:
        external_bonus += 40_000
    if path[-1] not in members:
        external_bonus += 60_000

    return vol_score + length_bonus + role_bonus + external_bonus


def summarize_cluster_money_flow(
    members,
    transactions_df,
    role_by_account=None,
    window_hours=None,
    top_flows=None,
):
    """Cluster-level money-flow summary for cluster_results.json."""
    if window_hours is None:
        window_hours = RAPID_FORWARDING_WINDOW_HOURS
    if top_flows is None:
        top_flows = MONEY_FLOW_TOP_FLOWS
    if role_by_account is None:
        role_by_account = {}

    members = set(members)
    sinks = identify_sink_accounts(members, transactions_df)

    internal = transactions_df[
        transactions_df["from_account"].isin(members)
        & transactions_df["to_account"].isin(members)
    ]
    ext_in = transactions_df[
        ~transactions_df["from_account"].isin(members)
        & transactions_df["to_account"].isin(members)
    ]
    ext_out = transactions_df[
        transactions_df["from_account"].isin(members)
        & ~transactions_df["to_account"].isin(members)
    ]
    # Exit flows: to true external or to in-cluster sinks
    exit_dests = sinks | (
        set(transactions_df["to_account"]) - members
    )
    exit_out = transactions_df[
        transactions_df["from_account"].isin(members)
        & transactions_df["to_account"].isin(exit_dests)
        & ~transactions_df["to_account"].isin(
            # exclude self-loops if any
            []
        )
    ]
    # Refine: from member to (external OR sink), and source is not itself sink-only noise
    exit_rows = []
    for _, row in transactions_df.iterrows():
        src, dst = row["from_account"], row["to_account"]
        if src not in members:
            continue
        if dst == src:
            continue
        if dst not in members or dst in sinks:
            exit_rows.append(row)
    exit_out = pd.DataFrame(exit_rows) if exit_rows else ext_out.iloc[0:0]

    internal_volume = int(internal["amount"].sum()) if len(internal) else 0
    external_inbound_volume = int(ext_in["amount"].sum()) if len(ext_in) else 0
    external_outbound_volume = int(ext_out["amount"].sum()) if len(ext_out) else 0
    exit_outbound_volume = int(exit_out["amount"].sum()) if len(exit_out) else 0

    estimated_forwarded_exit = int(min(
        external_inbound_volume + internal_volume,
        max(external_outbound_volume, exit_outbound_volume),
    )) if max(external_outbound_volume, exit_outbound_volume) > 0 else 0
    total_in = external_inbound_volume + internal_volume
    estimated_forwarding_ratio = (
        round(estimated_forwarded_exit / total_in, 4) if total_in > 0 else 0.0
    )

    top_internal_flows = _top_directed_flows(internal, top_flows)
    combined_ext = (
        pd.concat([ext_in, ext_out, exit_out], ignore_index=True)
        if len(ext_in) or len(ext_out) or len(exit_out)
        else ext_in
    )
    top_external_flows = _top_directed_flows(combined_ext, top_flows)

    rapid_total = 0
    for aid in members:
        temporal = compute_temporal_flow(aid, members, transactions_df, window_hours)
        rapid_total += temporal["rapid_forwarding_events"]

    money_flow_paths = find_money_flow_paths(
        members, transactions_df, role_by_account=role_by_account
    )

    return {
        "money_flow": {
            "internal_volume": internal_volume,
            "external_inbound_volume": external_inbound_volume,
            "external_outbound_volume": external_outbound_volume,
            "exit_outbound_volume": exit_outbound_volume,
            "estimated_forwarded_volume": estimated_forwarded_exit,
            "estimated_forwarding_ratio": estimated_forwarding_ratio,
            "top_internal_flows": top_internal_flows,
            "top_external_flows": top_external_flows,
            "rapid_forwarding_events": rapid_total,
        },
        "money_flow_paths": money_flow_paths,
    }


def _top_directed_flows(tx_df, top_n):
    if tx_df is None or len(tx_df) == 0:
        return []
    edges = _aggregate_directed_edges(tx_df)
    ranked = sorted(edges.values(), key=lambda e: e["total_volume"], reverse=True)[:top_n]
    result = []
    for e in ranked:
        result.append({
            "source_account": e["source_account"],
            "destination_account": e["destination_account"],
            "total_volume": e["total_volume"],
            "transaction_count": e["transaction_count"],
            "time_span_hours": e["time_span_hours"],
        })
    return result


def is_external(account_id, members):
    """True if account_id is not a cluster member."""
    return account_id not in set(members)
