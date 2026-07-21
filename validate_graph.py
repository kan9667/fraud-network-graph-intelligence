import json
from src.graph_builder import load_graph
from src.config import GRAPH_FILE, CLUSTER_RESULTS_FILE

G = load_graph(GRAPH_FILE)

shared_phone_count = 0
shared_device_count = 0
transaction_count = 0
multi_type_count = 0

for _, _, data in G.edges(data=True):
    types = data.get("relationship_types", set())
    if "transaction" in types:
        transaction_count += 1
    if "shared_phone" in types:
        shared_phone_count += 1
    if "shared_device" in types:
        shared_device_count += 1
    if len(types) > 1:
        multi_type_count += 1

has_ground_truth_nodes = False
for _, data in G.nodes(data=True):
    if "is_fraud_ring" in data:
        has_ground_truth_nodes = True
        break

print("=== Graph Validation ===")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Transaction relationships: {transaction_count}")
print(f"Shared phone relationships: {shared_phone_count}")
print(f"Shared device relationships: {shared_device_count}")
print(f"Edges with multiple relationship types: {multi_type_count}")
print(f"Ground truth in node attributes: {has_ground_truth_nodes}")
print()

with open(CLUSTER_RESULTS_FILE) as f:
    results = json.load(f)

has_true_label = any("true_label_majority" in r for r in results)

print("=== Cluster Results Validation ===")
print(f"Clusters: {len(results)}")
print(f"Contains true_label_majority: {has_true_label}")
for r in results:
    print(f"  Cluster {r['cluster_id']}: size={r['size']}, "
          f"density={r['density']}, volume={r['internal_volume']}")
