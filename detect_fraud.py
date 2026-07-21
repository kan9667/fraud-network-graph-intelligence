import csv
import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict

# --- Step 1: load the data we generated ---
accounts = {}
with open("accounts.csv") as f:
    for row in csv.DictReader(f):
        accounts[row["account_id"]] = row

transactions = []
with open("transactions.csv") as f:
    for row in csv.DictReader(f):
        transactions.append(row)

# --- Step 2: build the graph ---
# Every account is a "node". We draw a "edge" (connection) between two
# accounts if they transacted with each other, OR if they share a phone/device.
G = nx.Graph()

for acc_id in accounts:
    G.add_node(acc_id)

# edges from transactions (weighted by how much money / how many times)
edge_weights = defaultdict(int)
for tx in transactions:
    a, b = tx["from_account"], tx["to_account"]
    key = tuple(sorted([a, b]))
    edge_weights[key] += 1

for (a, b), weight in edge_weights.items():
    G.add_edge(a, b, weight=weight)

# edges from shared phone/device — this is the strongest fraud signal
by_phone = defaultdict(list)
by_device = defaultdict(list)
for acc_id, info in accounts.items():
    by_phone[info["phone"]].append(acc_id)
    by_device[info["device_id"]].append(acc_id)

for group in list(by_phone.values()) + list(by_device.values()):
    if len(group) > 1:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                # a shared phone/device is a much stronger signal than one transaction,
                # so give it more weight
                G.add_edge(group[i], group[j], weight=10)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Step 3: run community detection ---
# This automatically groups nodes that are much more tightly connected to
# each other than to the rest of the graph — exactly what a fraud ring looks like.
communities = louvain_communities(G, weight="weight", seed=42)

print(f"\nFound {len(communities)} clusters total.\n")

# --- Step 4: score each cluster and check against the "answer key" ---
results = []
for idx, cluster in enumerate(communities):
    if len(cluster) < 3:
        continue  # skip tiny/singleton clusters, not interesting

    subgraph = G.subgraph(cluster)
    total_volume = sum(
        int(tx["amount"]) for tx in transactions
        if tx["from_account"] in cluster and tx["to_account"] in cluster
    )
    density = nx.density(subgraph)

    # this is our "answer key" check — in a real system you would NOT know this,
    # it's only here so you can measure your own accuracy
    true_labels = [accounts[a]["is_fraud_ring"] for a in cluster]
    majority_label = max(set(true_labels), key=true_labels.count)

    results.append({
        "cluster_id": idx,
        "size": len(cluster),
        "density": round(density, 2),
        "internal_volume": total_volume,
        "true_label_majority": majority_label,
        "members": list(cluster)
    })

# sort by internal transaction volume — highest first (most suspicious)
results.sort(key=lambda r: r["internal_volume"], reverse=True)

print("Top clusters by internal transaction volume (most suspicious first):\n")
for r in results[:8]:
    print(f"Cluster {r['cluster_id']}: {r['size']} accounts | "
          f"density={r['density']} | volume=₹{r['internal_volume']:,} | "
          f"true label majority: {r['true_label_majority']}")

# save full results for the dashboard to use later
import json
with open("cluster_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved detailed results to cluster_results.json")
