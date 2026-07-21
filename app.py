"""
Investigator Command Center — Fraud Network Intelligence Dashboard

Loads detection outputs only. Does not use ground-truth labels.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import BASE_DIR, OUTPUT_DIR, RAPID_FORWARDING_WINDOW_HOURS
from src.dashboard_data import (
    ROLE_COLORS,
    ROLE_DISPLAY,
    RISK_COLORS,
    account_investigation,
    account_profile_map,
    build_case_report,
    build_case_summary_text,
    build_risk_indicators,
    case_id,
    case_report_markdown,
    case_transactions,
    compute_command_metrics,
    get_case_by_id,
    load_accounts_safe,
    load_cluster_results_raw,
    load_graph_safe,
    load_transactions_safe,
    mark_rapid_forwarding,
    primary_roles,
    sort_cases,
    system_status,
    top_risk_factor_labels,
)
from src.cluster_viz import (
    build_cluster_network_html,
    build_neighborhood_html,
    role_legend_markdown,
)

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Investigator Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    div[data-testid="stMetric"] {
        background: #1a2332;
        border: 1px solid #2c3e50;
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stMetric"] label { color: #95a5a6 !important; }
    .case-card {
        border: 1px solid #2c3e50;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: #151c27;
    }
    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }
    .disclaimer {
        color: #95a5a6;
        font-size: 0.88rem;
        border-left: 3px solid #f39c12;
        padding-left: 12px;
        margin: 0.5rem 0 1rem 0;
    }
    .path-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        background: #0f1419;
        border: 1px solid #2c3e50;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .section-note { color: #7f8c8d; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

NAV_ITEMS = [
    "Command Center",
    "Fraud Cases",
    "Network Investigation",
    "Account Investigation",
    "Money Flow",
    "Timeline",
    "Evidence",
]

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_cluster_results():
    return load_cluster_results_raw()


@st.cache_data(show_spinner=False)
def cached_accounts():
    return load_accounts_safe()


@st.cache_data(show_spinner=False)
def cached_transactions():
    return load_transactions_safe()


@st.cache_resource(show_spinner=False)
def cached_graph():
    return load_graph_safe()


@st.cache_data(show_spinner=False)
def cached_cluster_html(cluster_id: int, role_json: str):
    """Cache PyVis HTML per case + role map fingerprint."""
    results = load_cluster_results_raw()
    case = get_case_by_id(results, cluster_id)
    if case is None:
        return "<html><body>Case not found</body></html>"
    G = load_graph_safe()
    role_by_account = json.loads(role_json)
    return build_cluster_network_html(G, case["members"], role_by_account)


@st.cache_data(show_spinner=False)
def cached_neighborhood_html(account_id: str, role_json: str):
    G = load_graph_safe()
    role_by_account = json.loads(role_json)
    return build_neighborhood_html(G, account_id, role_by_account)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def risk_badge_html(level: str) -> str:
    level = str(level or "LOW").upper()
    color = RISK_COLORS.get(level, "#7f8c8d")
    return (
        f'<span class="risk-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color};">{level}</span>'
    )


def ensure_session():
    if "nav" not in st.session_state:
        st.session_state.nav = "Command Center"
    if "selected_cluster_id" not in st.session_state:
        st.session_state.selected_cluster_id = None
    if "selected_account_id" not in st.session_state:
        st.session_state.selected_account_id = None


def select_case(cluster_id, go_to="Network Investigation"):
    st.session_state.selected_cluster_id = int(cluster_id)
    st.session_state.selected_account_id = None
    st.session_state.nav = go_to


def require_data():
    try:
        results = cached_cluster_results()
        accounts = cached_accounts()
        transactions = cached_transactions()
        G = cached_graph()
    except FileNotFoundError as e:
        st.error(
            "Detection outputs not found. Run `python3 detect_fraud.py` first, "
            f"or use **Run detection** in the sidebar.\n\nDetails: {e}"
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load detection data: {e}")
        st.stop()

    if not results:
        st.warning("No candidate clusters in results.")
        st.stop()
    return results, accounts, transactions, G


def selected_case(results):
    cid = st.session_state.selected_cluster_id
    if cid is None:
        # default to highest-priority case
        sorted_r = sort_cases(results)
        cid = sorted_r[0]["cluster_id"]
        st.session_state.selected_cluster_id = int(cid)
    case = get_case_by_id(results, cid)
    if case is None:
        sorted_r = sort_cases(results)
        case = sorted_r[0]
        st.session_state.selected_cluster_id = int(case["cluster_id"])
    return case


def role_json_for_case(case) -> str:
    return json.dumps(
        {p["account_id"]: p["probable_role"] for p in case.get("account_profiles", [])}
    )


def money_flow_fields(case):
    mf = case.get("money_flow") or {}
    return {
        "internal": int(mf.get("internal_volume", case.get("internal_volume", 0)) or 0),
        "ext_in": int(mf.get("external_inbound_volume", 0) or 0),
        "ext_out": int(mf.get("external_outbound_volume", 0) or 0),
        "exit": int(mf.get("exit_outbound_volume", 0) or 0),
        "forwarded": int(mf.get("estimated_forwarded_volume", 0) or 0),
        "fwd_ratio": float(mf.get("estimated_forwarding_ratio", 0) or 0),
        "rapid": int(mf.get("rapid_forwarding_events", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(results, accounts, transactions):
    with st.sidebar:
        st.markdown("### 🛡️ Command Center")
        st.caption("Fraud Network Intelligence")

        nav = st.radio(
            "Navigation",
            NAV_ITEMS,
            index=NAV_ITEMS.index(st.session_state.nav)
            if st.session_state.nav in NAV_ITEMS
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.nav = nav

        st.divider()
        st.markdown("**Active case**")
        case_options = {
            f"{case_id(r['cluster_id'])} · {r.get('risk_level')} · score {r.get('risk_score')}": int(
                r["cluster_id"]
            )
            for r in sort_cases(results)
        }
        labels = list(case_options.keys())
        current = st.session_state.selected_cluster_id
        default_idx = 0
        if current is not None:
            for i, cid in enumerate(case_options.values()):
                if cid == int(current):
                    default_idx = i
                    break
        picked = st.selectbox("Case", labels, index=default_idx, label_visibility="collapsed")
        st.session_state.selected_cluster_id = case_options[picked]

        st.divider()
        if st.button("Run detection pipeline", use_container_width=True):
            with st.spinner("Running detect_fraud.py…"):
                proc = subprocess.run(
                    ["python3", "detect_fraud.py"],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                )
            cached_cluster_results.clear()
            cached_accounts.clear()
            cached_transactions.clear()
            cached_graph.clear()
            cached_cluster_html.clear()
            cached_neighborhood_html.clear()
            if proc.returncode == 0:
                st.success("Detection complete. Reload to refresh.")
                st.rerun()
            else:
                st.error(proc.stderr[-800:] if proc.stderr else "Detection failed")

        st.divider()
        status = system_status(accounts, transactions, results)
        st.markdown("**System status**")
        st.caption(f"Detection: **{status['detection_status']}**")
        st.caption(f"Accounts: {status['total_accounts']:,}")
        st.caption(f"Transactions: {status['total_transactions']:,}")
        st.caption(f"Cases: {status['candidate_cases']}")
        st.caption(f"Results: {status['results_mtime']}")
        st.caption(f"Graph: {status['graph_mtime']}")
        st.markdown(
            '<p class="disclaimer">Role labels are algorithmic inferences '
            "based on transaction and network behavior and are not confirmed "
            "identities.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_command_center(results, accounts, transactions):
    st.title("Command Center")
    st.caption("Operational overview of algorithmically detected candidate networks")

    metrics = compute_command_metrics(results, accounts, transactions)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total accounts", f"{metrics['total_accounts']:,}")
    c2.metric("Total transactions", f"{metrics['total_transactions']:,}")
    c3.metric("Candidate networks", metrics["candidate_networks"])
    c4.metric("Medium+ risk networks", metrics["medium_plus_networks"])

    c5, c6, c7 = st.columns(3)
    c5.metric("High / Critical networks", metrics["high_critical_networks"])
    c6.metric(
        "Suspicious internal volume",
        f"₹{metrics['suspicious_transaction_volume']:,}",
    )
    c7.metric("Rapid forwarding events", f"{metrics['rapid_forwarding_events']:,}")

    st.markdown(
        '<p class="disclaimer">Role labels are algorithmic inferences based on '
        "transaction and network behavior and are not confirmed identities.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Risk distribution")
        dist = metrics["risk_distribution"]
        chart_df = pd.DataFrame(
            {
                "Risk level": list(dist.keys()),
                "Networks": list(dist.values()),
            }
        )
        st.bar_chart(chart_df.set_index("Risk level"), height=280)
        for lvl, n in dist.items():
            st.markdown(
                f"{risk_badge_html(lvl)} &nbsp; **{n}** network(s)",
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("Top priority cases")
        priority = [
            r
            for r in sort_cases(results)
            if str(r.get("risk_level", "")).upper() in {"MEDIUM", "HIGH", "CRITICAL"}
        ]
        if not priority:
            priority = sort_cases(results)[:5]
            st.caption("No MEDIUM+ cases in current run — showing top by score.")

        for r in priority[:6]:
            mf = money_flow_fields(r)
            factors = top_risk_factor_labels(r, 3)
            with st.container(border=True):
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(
                        f"**{case_id(r['cluster_id'])}** &nbsp; "
                        f"{risk_badge_html(r.get('risk_level'))} &nbsp; "
                        f"Score **{r.get('risk_score')}**",
                        unsafe_allow_html=True,
                    )
                with h2:
                    if st.button(
                        "Investigate",
                        key=f"prio_{r['cluster_id']}",
                        use_container_width=True,
                    ):
                        select_case(r["cluster_id"])
                        st.rerun()
                st.caption(
                    f"{r.get('size')} accounts · internal ₹{mf['internal']:,} · "
                    f"exit ₹{mf['exit']:,} · rapid fwd {mf['rapid']}"
                )
                if factors:
                    st.caption("Risk factors: " + " · ".join(factors[:3]))


def page_fraud_cases(results):
    st.title("Fraud Cases")
    st.caption("Case queue of candidate networks sorted by investigative priority")

    f1, f2, f3, f4 = st.columns([1, 1, 1.2, 1.2])
    with f1:
        levels = st.multiselect(
            "Risk level",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["MEDIUM", "HIGH", "CRITICAL", "LOW"],
        )
    with f2:
        min_score = st.slider("Minimum risk score", 0, 100, 0)
    with f3:
        role_filter = st.multiselect(
            "Must include role",
            [
                "probable_mule",
                "probable_coordinator",
                "probable_consolidator",
                "probable_cash_out",
                "suspected_victim",
            ],
            format_func=lambda x: ROLE_DISPLAY.get(x, x),
        )
    with f4:
        sort_mode = st.selectbox(
            "Sort by",
            ["Risk (default)", "Risk score", "Internal volume", "Exit volume", "Size"],
        )

    rows = []
    for r in results:
        lvl = str(r.get("risk_level", "LOW")).upper()
        if lvl not in levels:
            continue
        if float(r.get("risk_score", 0)) < min_score:
            continue
        rs = r.get("role_summary") or {}
        role_key_map = {
            "probable_mule": "probable_mules",
            "probable_coordinator": "probable_coordinators",
            "probable_consolidator": "probable_consolidators",
            "probable_cash_out": "probable_cash_out",
            "suspected_victim": "suspected_victims",
        }
        if role_filter:
            ok = True
            for rf in role_filter:
                key = role_key_map.get(rf, rf)
                if not rs.get(key, 0):
                    # also check profiles
                    profiles = r.get("account_profiles") or []
                    if not any(p.get("probable_role") == rf for p in profiles):
                        ok = False
                        break
            if not ok:
                continue
        mf = money_flow_fields(r)
        rows.append(
            {
                "Case ID": case_id(r["cluster_id"]),
                "cluster_id": r["cluster_id"],
                "Risk level": lvl,
                "Risk score": r.get("risk_score"),
                "Accounts": r.get("size"),
                "Internal volume": mf["internal"],
                "Exit volume": mf["exit"],
                "Rapid forwarding": mf["rapid"],
                "Primary roles": primary_roles(rs),
            }
        )

    if sort_mode == "Risk score":
        rows.sort(key=lambda x: -x["Risk score"])
    elif sort_mode == "Internal volume":
        rows.sort(key=lambda x: -x["Internal volume"])
    elif sort_mode == "Exit volume":
        rows.sort(key=lambda x: -x["Exit volume"])
    elif sort_mode == "Size":
        rows.sort(key=lambda x: -x["Accounts"])
    else:
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        rows.sort(key=lambda x: (order.get(x["Risk level"], 9), -x["Risk score"]))

    st.caption(f"{len(rows)} case(s) match filters")
    if not rows:
        st.info("No cases match the current filters.")
        return

    display = pd.DataFrame(rows).drop(columns=["cluster_id"])
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Internal volume": st.column_config.NumberColumn(format="₹%d"),
            "Exit volume": st.column_config.NumberColumn(format="₹%d"),
        },
    )

    st.subheader("Open a case")
    cols = st.columns(min(4, len(rows)))
    for i, row in enumerate(rows[:12]):
        with cols[i % len(cols)]:
            if st.button(
                f"{row['Case ID']} ({row['Risk level']})",
                key=f"open_{row['cluster_id']}",
                use_container_width=True,
            ):
                select_case(row["cluster_id"])
                st.rerun()


def page_network(case, G):
    cid = case_id(case["cluster_id"])
    st.title(f"Network Investigation — {cid}")
    st.markdown(
        f"{risk_badge_html(case.get('risk_level'))} &nbsp; "
        f"**Risk score:** {case.get('risk_score')} &nbsp;·&nbsp; "
        f"**{case.get('size')} accounts**",
        unsafe_allow_html=True,
    )

    mf = money_flow_fields(case)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Internal volume", f"₹{mf['internal']:,}")
    m2.metric("Exit volume", f"₹{mf['exit']:,}")
    m3.metric("Rapid forwarding", mf["rapid"])
    m4.metric("Density", f"{case.get('density', 0):.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk factors")
        factors = case.get("risk_factors") or []
        if factors:
            for f in factors:
                if isinstance(f, dict):
                    st.markdown(
                        f"- **{f.get('factor', f.get('name', 'factor'))}** — "
                        f"{f.get('description', f.get('detail', ''))}"
                    )
                else:
                    st.markdown(f"- {f}")
        else:
            st.caption("No discrete risk factors listed for this case.")
        if case.get("explanation"):
            with st.expander("Model explanation"):
                st.write(case["explanation"])

    with c2:
        st.subheader("Network structure")
        st.write(case.get("network_structure_summary") or "—")
        st.subheader("Role summary")
        rs = case.get("role_summary") or {}
        pretty = {
            "probable_mules": "Probable mule",
            "probable_coordinators": "Probable coordinator",
            "probable_consolidators": "Probable consolidator",
            "probable_cash_out": "Probable cash-out",
            "suspected_victims": "Suspected victim",
            "unknown": "Unknown",
        }
        role_df = pd.DataFrame(
            [{"Role": pretty.get(k, k), "Count": v} for k, v in rs.items() if v]
        )
        if len(role_df):
            st.dataframe(role_df, hide_index=True, use_container_width=True)
        st.subheader("Money-flow summary")
        st.markdown(
            f"- Internal: **₹{mf['internal']:,}**\n"
            f"- External inbound: **₹{mf['ext_in']:,}**\n"
            f"- External outbound: **₹{mf['ext_out']:,}**\n"
            f"- Exit volume: **₹{mf['exit']:,}**\n"
            f"- Estimated forwarded: **₹{mf['forwarded']:,}** "
            f"(ratio {mf['fwd_ratio']:.1%})"
        )

    st.subheader("Cluster network graph")
    st.markdown(role_legend_markdown(), unsafe_allow_html=True)
    st.caption("Nodes colored by algorithmically inferred role. Gray external nodes are counterparties outside the case membership.")
    try:
        html = cached_cluster_html(int(case["cluster_id"]), role_json_for_case(case))
        components.html(html, height=640, scrolling=True)
    except Exception as e:
        st.warning(f"Could not render network graph: {e}")

    st.subheader("Accounts in this case")
    profiles = case.get("account_profiles") or []
    if profiles:
        pdf = pd.DataFrame(
            [
                {
                    "Account": p["account_id"],
                    "Probable role": ROLE_DISPLAY.get(
                        p.get("probable_role"), p.get("probable_role")
                    ),
                    "Confidence": p.get("role_confidence"),
                }
                for p in profiles
            ]
        )
        st.dataframe(pdf, hide_index=True, use_container_width=True)
        picks = [p["account_id"] for p in profiles]
        acc = st.selectbox("Select account for deeper investigation", picks)
        if st.button("Open account investigation", type="primary"):
            st.session_state.selected_account_id = acc
            st.session_state.nav = "Account Investigation"
            st.rerun()


def page_account(case, G, transactions):
    st.title("Account Investigation")
    profiles = case.get("account_profiles") or []
    members = [p["account_id"] for p in profiles] or list(case.get("members", []))
    if not members:
        st.warning("No accounts in this case.")
        return

    default = st.session_state.selected_account_id
    idx = members.index(default) if default in members else 0
    account_id = st.selectbox(
        "Account",
        members,
        index=idx,
        format_func=lambda a: f"{a} — {ROLE_DISPLAY.get(account_profile_map(case).get(a, {}).get('probable_role', 'unknown'), 'unknown')}",
    )
    st.session_state.selected_account_id = account_id

    inv = account_investigation(account_id, case, G, transactions)
    profile = inv["profile"]
    flow = inv["flow"]
    temporal = inv["temporal"]

    st.markdown(
        f"### {account_id} &nbsp; "
        f"`{ROLE_DISPLAY.get(profile.get('probable_role'), profile.get('probable_role'))}` "
        f"· confidence **{profile.get('role_confidence')}**"
    )
    st.caption(f"Case {case_id(case['cluster_id'])} · algorithmic inference only")

    st.subheader("Role evidence")
    for ev in profile.get("role_evidence") or []:
        st.markdown(f"- {ev}")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Transaction statistics**")
        st.write(f"Inbound count: {flow.get('internal_inbound_count', 0) + flow.get('external_inbound_count', 0)}")
        st.write(f"Outbound count: {flow.get('internal_outbound_count', 0) + flow.get('external_outbound_count', 0)}")
        st.write(f"Inbound volume: ₹{flow.get('inbound_volume', 0):,}")
        st.write(f"Outbound volume: ₹{flow.get('outbound_volume', 0):,}")
        st.write(f"Internal inbound: ₹{flow.get('internal_inbound_volume', 0):,}")
        st.write(f"Internal outbound: ₹{flow.get('internal_outbound_volume', 0):,}")
        st.write(f"External inbound: ₹{flow.get('external_inbound_volume', 0):,}")
        st.write(f"External outbound: ₹{flow.get('external_outbound_volume', 0):,}")
        if flow.get("exit_outbound_volume"):
            st.write(f"Exit outbound: ₹{flow.get('exit_outbound_volume', 0):,}")
    with t2:
        st.markdown("**Network statistics**")
        st.write(f"Degree: {inv['degree']}")
        st.write(f"Weighted degree: {inv['weighted_degree']}")
        st.write(f"Unique counterparties: {inv['unique_counterparties']}")
        st.write(f"Internal counterparties: {inv['internal_counterparties']}")
        st.write(f"External counterparties: {inv['external_counterparties']}")
    with t3:
        st.markdown("**Temporal statistics**")
        st.write(f"First transaction: {inv['first_transaction'] or '—'}")
        st.write(f"Last transaction: {inv['last_transaction'] or '—'}")
        st.write(f"Active duration (h): {inv['active_duration_hours']}")
        st.write(f"Rapid forwarding events: {temporal.get('rapid_forwarding_events', 0)}")
        med = temporal.get("median_inbound_to_outbound_delay_hours")
        st.write(
            f"Median inbound→outbound delay (h): {med if med is not None else '—'}"
        )
        st.write(
            f"Min delay (h): {temporal.get('min_inbound_to_outbound_delay_hours') if temporal.get('min_inbound_to_outbound_delay_hours') is not None else '—'}"
        )
        st.write(
            f"Max delay (h): {temporal.get('max_inbound_to_outbound_delay_hours') if temporal.get('max_inbound_to_outbound_delay_hours') is not None else '—'}"
        )

    st.subheader("Immediate network neighborhood")
    try:
        html = cached_neighborhood_html(account_id, role_json_for_case(case))
        components.html(html, height=440, scrolling=True)
    except Exception as e:
        st.warning(f"Neighborhood graph unavailable: {e}")
        if inv["neighbors"]:
            st.write("Neighbors:", ", ".join(inv["neighbors"][:30]))


def page_money_flow(case):
    st.title(f"Money Flow — {case_id(case['cluster_id'])}")
    st.caption("Derived from actual transactions; paths are not invented")

    mf = money_flow_fields(case)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Internal volume", f"₹{mf['internal']:,}")
    c2.metric("External inbound", f"₹{mf['ext_in']:,}")
    c3.metric("External outbound", f"₹{mf['ext_out']:,}")
    c4.metric("Exit volume", f"₹{mf['exit']:,}")
    c5, c6 = st.columns(2)
    c5.metric("Estimated forwarded", f"₹{mf['forwarded']:,}")
    c6.metric("Forwarding ratio", f"{mf['fwd_ratio']:.1%}")

    st.subheader("Top money-flow paths")
    paths = case.get("money_flow_paths") or []
    if not paths:
        st.info("No multi-hop money-flow paths recorded for this case.")
    else:
        for i, p in enumerate(paths[:10], 1):
            nodes = p.get("path") or []
            vol = int(p.get("total_volume", 0) or 0)
            # Build a simple vertical flow diagram
            lines = []
            for j, node in enumerate(nodes):
                lines.append(f"**{node}**")
                if j < len(nodes) - 1:
                    lines.append(f"<div style='color:#5dade2;padding-left:8px'>↓ ₹{vol:,}</div>")
            html = (
                f"<div class='path-box'><div style='color:#7f8c8d;margin-bottom:6px'>"
                f"Path {i} · {p.get('transaction_count', 0)} tx · "
                f"{p.get('time_span_hours', 0)}h span · total ₹{vol:,}"
                f"</div>{''.join(lines)}</div>"
            )
            st.markdown(html, unsafe_allow_html=True)

    mf_raw = case.get("money_flow") or {}
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top internal flows")
        internal = mf_raw.get("top_internal_flows") or []
        if internal:
            st.dataframe(
                pd.DataFrame(internal)[
                    ["source_account", "destination_account", "total_volume", "transaction_count"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No internal aggregate flows.")
    with col_b:
        st.subheader("Top external / exit flows")
        external = mf_raw.get("top_external_flows") or []
        if external:
            st.dataframe(
                pd.DataFrame(external)[
                    ["source_account", "destination_account", "total_volume", "transaction_count"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No external aggregate flows.")


def page_timeline(case, transactions):
    st.title(f"Timeline — {case_id(case['cluster_id'])}")
    st.caption(
        f"Chronological transactions for case members. "
        f"Rapid forwarding window: {RAPID_FORWARDING_WINDOW_HOURS}h."
    )

    tx = case_transactions(case, transactions)
    if len(tx) == 0:
        st.info("No transactions involving this case.")
        return
    tx = mark_rapid_forwarding(tx, case["members"])

    f1, f2, f3 = st.columns(3)
    with f1:
        min_amt = st.number_input("Minimum amount (₹)", min_value=0, value=0, step=1000)
    with f2:
        classes = sorted(tx["flow_class"].dropna().unique().tolist())
        selected_classes = st.multiselect(
            "Flow class",
            classes,
            default=classes,
        )
    with f3:
        only_flags = st.checkbox("Only rapid / high-value / exit", value=False)

    tmin, tmax = tx["timestamp"].min(), tx["timestamp"].max()
    # date range filter via slider on timestamps as dates
    try:
        d1, d2 = st.date_input(
            "Date range",
            value=(tmin.date(), tmax.date()),
            min_value=tmin.date(),
            max_value=tmax.date(),
        )
        if isinstance(d1, tuple):
            pass
        mask_date = (tx["timestamp"].dt.date >= d1) & (tx["timestamp"].dt.date <= d2)
    except Exception:
        mask_date = pd.Series(True, index=tx.index)

    filtered = tx.loc[mask_date].copy()
    filtered = filtered[filtered["amount"] >= min_amt]
    if selected_classes:
        filtered = filtered[filtered["flow_class"].isin(selected_classes)]
    if only_flags:
        filtered = filtered[
            filtered["is_rapid_forward"]
            | filtered["is_high_value"]
            | filtered["is_exit"]
        ]

    filtered = filtered.sort_values("timestamp")
    st.caption(f"Showing {len(filtered):,} of {len(tx):,} transactions")

    show = filtered[
        [
            "timestamp",
            "from_account",
            "to_account",
            "amount",
            "flow_class",
            "is_exit",
            "is_rapid_forward",
            "is_high_value",
        ]
    ].copy()
    show["timestamp"] = show["timestamp"].astype(str)
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=480,
        column_config={
            "amount": st.column_config.NumberColumn("Amount", format="₹%d"),
            "is_exit": st.column_config.CheckboxColumn("Exit"),
            "is_rapid_forward": st.column_config.CheckboxColumn("Rapid fwd"),
            "is_high_value": st.column_config.CheckboxColumn("High value"),
        },
    )

    # Simple volume-over-time chart
    if len(filtered):
        daily = (
            filtered.set_index("timestamp")
            .resample("D")["amount"]
            .sum()
            .reset_index()
        )
        daily.columns = ["date", "volume"]
        st.subheader("Daily transaction volume")
        st.bar_chart(daily.set_index("date"))


def page_evidence(case):
    st.title(f"Evidence — {case_id(case['cluster_id'])}")
    st.markdown(
        f"{risk_badge_html(case.get('risk_level'))} &nbsp; "
        f"Risk score **{case.get('risk_score')}** · {case.get('size')} accounts",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="disclaimer">All statements below are algorithmically inferred from '
        "transaction and network structure. They are not determinations of guilt.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Case summary")
    st.write(build_case_summary_text(case))

    st.subheader("Risk indicators")
    for ind in build_risk_indicators(case):
        st.markdown(f"- {ind}")

    st.subheader("Network structure")
    st.write(case.get("network_structure_summary") or "—")

    st.subheader("Role evidence")
    profiles = sorted(
        case.get("account_profiles") or [],
        key=lambda p: (-(p.get("role_confidence") or 0), p.get("account_id")),
    )
    important = [
        p
        for p in profiles
        if p.get("probable_role") not in (None, "unknown")
        or (p.get("role_confidence") or 0) >= 40
    ]
    if not important:
        important = profiles[:8]

    for p in important:
        role = ROLE_DISPLAY.get(p.get("probable_role"), p.get("probable_role"))
        with st.expander(
            f"{p.get('account_id')} — {role} (confidence {p.get('role_confidence')})"
        ):
            for ev in p.get("role_evidence") or []:
                st.markdown(f"- {ev}")

    st.subheader("Generate case report")
    report = build_case_report(case)
    md = case_report_markdown(report)
    j = json.dumps(report, indent=2, default=str)

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Download JSON report",
            data=j,
            file_name=f"{case_id(case['cluster_id'])}_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with b2:
        st.download_button(
            "Download Markdown report",
            data=md,
            file_name=f"{case_id(case['cluster_id'])}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("Preview JSON report"):
        st.code(j[:4000] + ("…" if len(j) > 4000 else ""), language="json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ensure_session()
    results, accounts, transactions, G = require_data()
    render_sidebar(results, accounts, transactions)

    # Ground-truth guard for UI dataframes
    for col in ("is_fraud_ring", "true_label_majority", "ground_truth_role", "fraud_ring_id"):
        if col in accounts.columns:
            accounts = accounts.drop(columns=[col])

    nav = st.session_state.nav
    case = selected_case(results)

    if nav == "Command Center":
        page_command_center(results, accounts, transactions)
    elif nav == "Fraud Cases":
        page_fraud_cases(results)
    elif nav == "Network Investigation":
        page_network(case, G)
    elif nav == "Account Investigation":
        page_account(case, G, transactions)
    elif nav == "Money Flow":
        page_money_flow(case)
    elif nav == "Timeline":
        page_timeline(case, transactions)
    elif nav == "Evidence":
        page_evidence(case)
    else:
        page_command_center(results, accounts, transactions)


if __name__ == "__main__":
    main()
