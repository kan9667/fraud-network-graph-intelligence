import csv
import json

# -- load the "answer key" (which accounts are ACTUALLY fraud, from generation) ---
accounts = {}
with open("accounts.csv") as f:
    for row in csv.DictReader(f):
        accounts[row["account_id"]] = row["is_fraud_ring"]

# --- load what our algorithm flagged ---
with open("cluster_results.json") as f:
    results = json.load(f)

# treat any cluster smaller than 50 as "flagged as suspicious" by our system
flagged_accounts = set()
for r in results:
    if r["size"] < 50:
        flagged_accounts.update(r["members"])

# the real fraud accounts, according to how we generated the data
actual_fraud_accounts = {acc_id for acc_id, label in accounts.items() if label != "no"}

# --- compute confusion matrix components ---
true_positives = flagged_accounts & actual_fraud_accounts        # correctly flagged fraud
false_positives = flagged_accounts - actual_fraud_accounts        # flagged but actually innocent
false_negatives = actual_fraud_accounts - flagged_accounts        # fraud we missed
true_negatives = set(accounts.keys()) - flagged_accounts - actual_fraud_accounts

tp, fp, fn, tn = len(true_positives), len(false_positives), len(false_negatives), len(true_negatives)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=== Fraud Detection Performance ===")
print(f"Total accounts:         {len(accounts)}")
print(f"Actual fraud accounts:  {len(actual_fraud_accounts)}")
print(f"Flagged accounts:       {len(flagged_accounts)}")
print()
print(f"True positives:  {tp}  (correctly caught fraud)")
print(f"False positives: {fp}  (innocent accounts wrongly flagged)")
print(f"False negatives: {fn}  (fraud we missed)")
print(f"True negatives:  {tn}  (innocent accounts correctly left alone)")
print()
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 score:  {f1:.2%}")

# save for the slide deck
with open("metrics.json", "w") as f:
    json.dump({
        "total_accounts": len(accounts),
        "actual_fraud": len(actual_fraud_accounts),
        "flagged": len(flagged_accounts),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }, f, indent=2)
print("\nSaved to metrics.json")