# accounts.csv — every account, with a phone, device_id, and a hidden is_fraud_ring label 
# (this label is your "answer key" — you'll use it later to check if your algorithm found the real rings, 
# but your detection algorithm won't be shown this column)

# transactions.csv — a log of money moving between accounts


import random
import csv
from faker import Faker

fake = Faker()
random.seed(42)  # makes results repeatable every time you run this

NUM_NORMAL_ACCOUNTS = 200
NUM_FRAUD_RINGS = 4
RING_SIZE = 7

accounts = []
transactions = []
account_id_counter = 1

# --- Step 1: create normal, innocent accounts ---
for i in range(NUM_NORMAL_ACCOUNTS):
    accounts.append({
        "account_id": f"ACC{account_id_counter:04d}",
        "phone": fake.phone_number(),
        "device_id": f"DEV{random.randint(1000,9999)}",
        "is_fraud_ring": "no"
    })
    account_id_counter += 1

# --- Step 2: normal accounts transact randomly with each other (background noise) ---
normal_ids = [a["account_id"] for a in accounts]
for i in range(600):
    a, b = random.sample(normal_ids, 2)
    transactions.append({
        "from_account": a,
        "to_account": b,
        "amount": random.randint(500, 20000),
        "timestamp": fake.date_time_this_year().isoformat()
    })

# --- Step 3: create fraud rings ---
# Each ring shares ONE phone number / device ID across several accounts,
# and transacts heavily within itself (money bouncing between the same small group).
for ring_num in range(NUM_FRAUD_RINGS):
    ring_ids = []
    shared_phone = fake.phone_number()
    shared_device = f"DEV{random.randint(1000,9999)}"
    for j in range(RING_SIZE):
        acc_id = f"ACC{account_id_counter:04d}"
        accounts.append({
            "account_id": acc_id,
            "phone": shared_phone,      # <-- the giveaway signal
            "device_id": shared_device, # <-- the giveaway signal
            "is_fraud_ring": f"ring_{ring_num+1}"
        })
        ring_ids.append(acc_id)
        account_id_counter += 1

    # heavy internal transactions = classic money-mule pattern
    for k in range(25):
        a, b = random.sample(ring_ids, 2)
        transactions.append({
            "from_account": a,
            "to_account": b,
            "amount": random.randint(50000, 400000),
            "timestamp": fake.date_time_this_year().isoformat()
        })

# --- Step 4: save to CSV files ---
with open("accounts.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["account_id", "phone", "device_id", "is_fraud_ring"])
    writer.writeheader()
    writer.writerows(accounts)

with open("transactions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["from_account", "to_account", "amount", "timestamp"])
    writer.writeheader()
    writer.writerows(transactions)

print(f"Done. {len(accounts)} accounts, {len(transactions)} transactions written.")
print(f"{NUM_FRAUD_RINGS} fraud rings of {RING_SIZE} accounts each are hidden inside.")
