import json
import shutil
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import louvain_communities

from src.config import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    ACCOUNTS_FILE,
    TRANSACTIONS_FILE,
    GRAPH_FILE,
    CLUSTER_RESULTS_FILE,
    LOUVAIN_SEED,
    MIN_CLUSTER_SIZE,
)
from src.loader import load_accounts, load_transactions, load_accounts_with_validation
from src.graph_builder import build_graph, save_graph
from src.features import extract_cluster_features
from src.risk_scorer import score_cluster
from src.role_classifier import classify_cluster_roles
from src.money_flow import summarize_cluster_money_flow

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

root_accounts = BASE_DIR / "accounts.csv"
root_transactions = BASE_DIR / "transactions.csv"
if not ACCOUNTS_FILE.exists() and root_accounts.exists():
    shutil.copy2(root_accounts, ACCOUNTS_FILE)
    print(f"Copied {root_accounts} \u2192 {ACCOUNTS_FILE}")
if not TRANSACTIONS_FILE.exists() and root_transactions.exists():
    shutil.copy2(root_transactions, TRANSACTIONS_FILE)
    print(f"Copied {root_transactions} \u2192 {TRANSACTIONS_FILE}")

accounts_df = load_accounts()
transactions_df = load_transactions()
load_accounts_with_validation(accounts_df, transactions_df)

G = build_graph(accounts_df, transactions_df)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

save_graph(G, GRAPH_FILE)
print(f"Graph saved to {GRAPH_FILE}")

communities = louvain_communities(G, weight="weight", seed=LOUVAIN_SEED)
print(f"\nFound {len(communities)} clusters total.\n")

results = []
for idx, cluster in enumerate(communities):
    if len(cluster) < MIN_CLUSTER_SIZE:
        continue

    subgraph = G.subgraph(cluster)
    density = nx.density(subgraph)

    members_set = set(cluster)
    internal_mask = (
        transactions_df["from_account"].isin(members_set)
        & transactions_df["to_account"].isin(members_set)
    )
    total_volume = int(transactions_df.loc[internal_mask, "amount"].sum())

    features = extract_cluster_features(members_set, G, accounts_df, transactions_df)
    risk = score_cluster(members_set, features, G)
    account_profiles, role_summary, network_structure_summary = classify_cluster_roles(
        members_set,
        G,
        accounts_df,
        transactions_df,
        cluster_risk_score=risk["risk_score"],
        cluster_risk_level=risk["risk_level"],
    )
    role_by_account = {
        p["account_id"]: p["probable_role"] for p in account_profiles
    }
    flow_summary = summarize_cluster_money_flow(
        members_set,
        transactions_df,
        role_by_account=role_by_account,
    )

    results.append({
        "cluster_id": idx,
        "size": len(cluster),
        "density": round(density, 2),
        "internal_volume": total_volume,
        "members": list(cluster),
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "risk_factors": risk["risk_factors"],
        "feature_scores": risk["feature_scores"],
        "features": features,
        "explanation": risk["explanation"],
        "account_profiles": account_profiles,
        "role_summary": role_summary,
        "network_structure_summary": network_structure_summary,
        "money_flow": flow_summary["money_flow"],
        "money_flow_paths": flow_summary["money_flow_paths"],
    })

results.sort(key=lambda r: (-r["risk_score"], -r["internal_volume"]))

print("Clusters ranked by risk score:\n")
for r in results:
    summary = r.get("role_summary", {})
    roles = ", ".join(
        f"{k}:{v}" for k, v in summary.items() if v > 0
    )
    print(f"  Cluster {r['cluster_id']}: risk={r['risk_score']} ({r['risk_level']}) | "
          f"{r['size']} accounts | density={r['density']} | volume=\u20b9{r['internal_volume']:,} | "
          f"{r.get('network_structure_summary', '')}")

with open(CLUSTER_RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved enriched results to {CLUSTER_RESULTS_FILE}")

root_results = BASE_DIR / "cluster_results.json"
with open(root_results, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved compatibility copy to {root_results}")
