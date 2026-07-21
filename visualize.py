import csv
import json
import networkx as nx
from pyvis.network import Network

# --- load everything we already built ---
accounts = {}
with open("accounts.csv") as f:
    for row in csv.DictReader(f):
        accounts[row["account_id"]] = row

transactions = []
with open("transactions.csv") as f:
    for row in csv.DictReader(f):
        transactions.append(row)

with open("cluster_results.json") as f:
    cluster_results = json.load(f)

# build a lookup: account_id -> which suspicious cluster it belongs to (if any)
account_to_cluster = {}
for cluster in cluster_results:
    if cluster["size"] < 200:  # skip the giant "normal" cluster, only color the small dense ones
        for member in cluster["members"]:
            account_to_cluster[member] = cluster["cluster_id"]

# a distinct color per fraud cluster; gray for everyone else
colors = ["#e74c3c", "#e67e22", "#9b59b6", "#c0392b", "#d35400", "#8e44ad"]

# --- rebuild the graph exactly like detect_fraud.py did ---
G = nx.Graph()
for acc_id in accounts:
    G.add_node(acc_id)

from collections import defaultdict
edge_weights = defaultdict(int)
for tx in transactions:
    a, b = tx["from_account"], tx["to_account"]
    key = tuple(sorted([a, b]))
    edge_weights[key] += 1
for (a, b), w in edge_weights.items():
    G.add_edge(a, b, weight=w)

by_phone = defaultdict(list)
by_device = defaultdict(list)
for acc_id, info in accounts.items():
    by_phone[info["phone"]].append(acc_id)
    by_device[info["device_id"]].append(acc_id)
for group in list(by_phone.values()) + list(by_device.values()):
    if len(group) > 1:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                G.add_edge(group[i], group[j], weight=10)

# --- build the interactive visualization ---
net = Network(height="800px", width="100%", bgcolor="#1a1a1a", font_color="white")
net.barnes_hut()  # physics engine that spreads nodes out naturally

for node in G.nodes():
    if node in account_to_cluster:
        cluster_id = account_to_cluster[node]
        color = colors[cluster_id % len(colors)]
        size = 25
        title = f"{node} — FLAGGED (cluster {cluster_id})"
    else:
        color = "#7f8c8d"  # gray = normal account
        size = 10
        title = f"{node} — normal"
    net.add_node(node, label="", title=title, color=color, size=size)

for u, v, data in G.edges(data=True):
    net.add_edge(u, v, value=data.get("weight", 1))

net.show("fraud_graph.html", notebook=False)
print("Saved visualization to fraud_graph.html")
