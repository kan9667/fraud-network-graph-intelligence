"""
Synthetic fraud-network generator (Phase 5A).

Produces a realistic benchmark dataset with four distinct fraud scenarios
designed to exercise different detection signals:

  Ring 1 - Classic money mule network   (shared phone + shared device)
  Ring 2 - Device/identity reuse ring    (shared device only)
  Ring 3 - Low-identity-signal network   (unique phones/devices - behavior only)
  Ring 4 - High-noise fraud network      (partial device reuse + shared IP)

Outputs (written to project root AND data/ for compatibility):
  accounts.csv           - operational account records (NO ground-truth roles)
  transactions.csv       - enriched transaction log
  data/ground_truth_roles.csv - evaluation-only mapping (account_id, ring, role)

The detection pipeline (graph_builder, features, risk_scorer, role_classifier,
detect_fraud) NEVER loads ground_truth_roles.csv.
"""

import csv
import random
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker


fake = Faker()
random.seed(42)
Faker.seed(42)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

NUM_NORMAL_ACCOUNTS = 200
NUM_NORMAL_TRANSACTIONS = 650

TRANSACTION_TYPES = ["transfer", "merchant_payment", "wallet_transfer", "bank_transfer"]
CHANNELS = ["mobile", "web", "api", "branch"]
CHANNEL_WEIGHTS = [5, 3, 2, 1]

# A fixed reference "now" so the dataset is deterministic across runs/clocks.
REFERENCE_NOW = datetime(2026, 7, 1, 12, 0, 0)

ACCOUNT_FIELDS = [
    "account_id",
    "phone",
    "device_id",
    "ip_address",
    "email",
    "account_created_at",
    "account_status",
    "is_fraud_ring",
]
TRANSACTION_FIELDS = [
    "transaction_id",
    "from_account",
    "to_account",
    "amount",
    "timestamp",
    "transaction_type",
    "channel",
]
GROUND_TRUTH_FIELDS = ["account_id", "fraud_ring_id", "ground_truth_role"]

ACCOUNT_STATUSES = ["active", "active", "active", "active", "frozen", "closed"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# State container - tracks every account, transaction, and ground-truth entry
# --------------------------------------------------------------------------

class GenState:
    def __init__(self):
        self.accounts = []
        self.transactions = []
        self.ground_truth = []
        self._acct_counter = 1
        self._txn_counter = 1
        self.scenario_log = []  # human-readable summaries per scenario

    # ----- id helpers -----
    def _next_account_id(self):
        aid = f"ACC{self._acct_counter:04d}"
        self._acct_counter += 1
        return aid

    def _next_txn_id(self):
        tid = f"TXN{self._txn_counter:06d}"
        self._txn_counter += 1
        return tid

    # ----- entity factories -----
    def add_account(self, phone, device, ip=None, email=None,
                    status="active", fraud_ring="no", created_at=None,
                    ground_truth_role=None, ground_truth_ring=None):
        aid = self._next_account_id()
        if ip is None:
            ip = fake.ipv4_private()
        if email is None:
            email = fake.email()
        if created_at is None:
            created_at = fake.date_time_between(
                start_date=REFERENCE_NOW - timedelta(days=730),
                end_date=REFERENCE_NOW - timedelta(days=30),
            )
        self.accounts.append({
            "account_id": aid,
            "phone": phone,
            "device_id": device,
            "ip_address": ip,
            "email": email,
            "account_created_at": created_at.isoformat(),
            "account_status": status,
            "is_fraud_ring": fraud_ring,
        })
        gt_ring = ground_truth_ring if ground_truth_ring is not None else (
            fraud_ring if fraud_ring != "no" else "none"
        )
        gt_role = ground_truth_role if ground_truth_role is not None else (
            "normal" if fraud_ring == "no" else "mule"
        )
        self.ground_truth.append({
            "account_id": aid,
            "fraud_ring_id": gt_ring,
            "ground_truth_role": gt_role,
        })
        return aid

    def add_transaction(self, from_id, to_id, amount, timestamp,
                        tx_type=None, channel=None):
        if tx_type is None:
            tx_type = random.choice(TRANSACTION_TYPES)
        if channel is None:
            channel = random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]
        self.transactions.append({
            "transaction_id": self._next_txn_id(),
            "from_account": from_id,
            "to_account": to_id,
            "amount": int(amount),
            "timestamp": timestamp.isoformat(),
            "transaction_type": tx_type,
            "channel": channel,
        })

    # ----- convenience identity factories -----
    @staticmethod
    def new_phone():
        return fake.phone_number()

    @staticmethod
    def new_device(prefix="DEV"):
        return f"{prefix}{random.randint(10000, 99999)}"

    @staticmethod
    def new_ip():
        return fake.ipv4_private()

    @staticmethod
    def new_email():
        return fake.email()

    def log_scenario(self, ring_id, title, details):
        self.scenario_log.append({"ring_id": ring_id, "title": title, **details})


# --------------------------------------------------------------------------
# Normal accounts & background noise
# --------------------------------------------------------------------------

def generate_normal_world(state):
    """Create 200 normal accounts with realistic sharing patterns.

    Importantly: some legitimate accounts share devices (family) and IPs
    (coworkers) so that identity reuse alone is not fraud.
    """
    # Assign device groups first - ~15 family pairs share a device.
    indices = list(range(NUM_NORMAL_ACCOUNTS))
    random.shuffle(indices)

    device_of = {}
    family_devices = [f"DEVFAM{i:03d}" for i in range(15)]
    ptr = 0
    for fam_dev in family_devices:
        if ptr + 2 > NUM_NORMAL_ACCOUNTS:
            break
        for _ in range(2):
            device_of[indices[ptr]] = fam_dev
            ptr += 1
    # Remaining accounts get unique devices.
    for i in range(NUM_NORMAL_ACCOUNTS):
        if i not in device_of:
            device_of[i] = f"DEVN{i:04d}{random.randint(0, 9)}"

    # Assign IP groups - ~10 clusters of coworkers/shareholders.
    ip_of = {}
    remaining = [i for i in range(NUM_NORMAL_ACCOUNTS)]
    random.shuffle(remaining)
    ip_ptr = 0
    for _ in range(10):
        if ip_ptr + 2 > NUM_NORMAL_ACCOUNTS:
            break
        size = random.randint(2, 3)
        if ip_ptr + size > NUM_NORMAL_ACCOUNTS:
            size = NUM_NORMAL_ACCOUNTS - ip_ptr
        shared_ip = fake.ipv4_private()
        for _ in range(size):
            ip_of[remaining[ip_ptr]] = shared_ip
            ip_ptr += 1

    normal_ids = []
    for i in range(NUM_NORMAL_ACCOUNTS):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=device_of[i],
            ip=ip_of.get(i, fake.ipv4_private()),
            status=random.choice(ACCOUNT_STATUSES),
            fraud_ring="no",
        )
        normal_ids.append(aid)

    # Background noise: regular peer-to-peer + merchant payments.
    for _ in range(NUM_NORMAL_TRANSACTIONS):
        a, b = random.sample(normal_ids, 2)
        amount = random.randint(500, 20000)
        ts = fake.date_time_between(
            start_date=REFERENCE_NOW - timedelta(days=180),
            end_date=REFERENCE_NOW,
        )
        tx_type = random.choices(
            TRANSACTION_TYPES, weights=[3, 4, 2, 1]
        )[0]
        state.add_transaction(a, b, amount, ts, tx_type=tx_type)

    # A handful of long-running relationships (repeat counterparties).
    for _ in range(40):
        a, b = random.sample(normal_ids, 2)
        base = fake.date_time_between(
            start_date=REFERENCE_NOW - timedelta(days=90),
            end_date=REFERENCE_NOW - timedelta(days=10),
        )
        for k in range(random.randint(3, 6)):
            ts = base + timedelta(days=k * random.randint(5, 20))
            state.add_transaction(
                a, b, random.randint(1000, 8000), ts,
                tx_type="merchant_payment",
            )

    return normal_ids


# --------------------------------------------------------------------------
# Shared temporal helpers
# --------------------------------------------------------------------------

def _burst(base, count, max_delta_minutes):
    """Return `count` timestamps within `max_delta_minutes` of `base`."""
    out = []
    for _ in range(count):
        delta = timedelta(minutes=random.randint(1, max(1, max_delta_minutes)))
        out.append(base + delta)
    return sorted(out)


# --------------------------------------------------------------------------
# Scenario 1 - Classic money mule network (shared phone + shared device)
# --------------------------------------------------------------------------

def scenario_1_classic_mule_network(state, normal_ids):
    ring_id = "ring_1"
    shared_phone = fake.phone_number()
    shared_device = "DEVR1SHARED"

    # Core ring members all share phone + device.
    mules = []
    for _ in range(3):
        aid = state.add_account(
            phone=shared_phone, device=shared_device,
            fraud_ring=ring_id, ground_truth_role="mule",
            ground_truth_ring=ring_id,
        )
        mules.append(aid)

    coord = state.add_account(
        phone=shared_phone, device=shared_device,
        fraud_ring=ring_id, ground_truth_role="coordinator",
        ground_truth_ring=ring_id,
    )
    consol = state.add_account(
        phone=shared_phone, device=shared_device,
        fraud_ring=ring_id, ground_truth_role="consolidator",
        ground_truth_ring=ring_id,
    )
    cashout = state.add_account(
        phone=shared_phone, device=shared_device, status="frozen",
        fraud_ring=ring_id, ground_truth_role="cash_out",
        ground_truth_ring=ring_id,
    )
    core = mules + [coord, consol, cashout]

    # External victims - defrauded outside the core cluster.
    victims = []
    for _ in range(4):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVV"),
            fraud_ring="no",
            ground_truth_role="victim",
            ground_truth_ring=ring_id,
        )
        victims.append(aid)

    # External cash-out destinations (where funds exit the system).
    ext_dests = []
    for _ in range(2):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVC"),
            fraud_ring="no",
            ground_truth_role="cash_out",
            ground_truth_ring=ring_id,
        )
        ext_dests.append(aid)

    # Timeline: ~80 days ago, burst over a few days.
    base = REFERENCE_NOW - timedelta(days=80, hours=random.randint(0, 12))

    # Stage 1 - victims pay mules.
    for i, victim in enumerate(victims[:3]):
        mule = mules[i]
        for ts in _burst(base + timedelta(minutes=random.randint(5, 240)),
                         random.randint(1, 2), 180):
            state.add_transaction(
                victim, mule, random.randint(40000, 180000), ts,
                tx_type=random.choice(["wallet_transfer", "transfer"]),
            )
    state.add_transaction(
        victims[3], random.choice(mules),
        random.randint(30000, 90000),
        base + timedelta(minutes=random.randint(10, 300)),
        tx_type="wallet_transfer",
    )

    # Stage 2 - mules rapidly forward to coordinator (10-120 min).
    for mule in mules:
        forward_base = base + timedelta(hours=random.randint(1, 6))
        for ts in _burst(forward_base, random.randint(2, 3), 120):
            state.add_transaction(
                mule, coord, random.randint(25000, 140000), ts,
                tx_type="wallet_transfer",
            )

    # Stage 3 - coordinator consolidates into consolidator.
    for ts in _burst(base + timedelta(hours=random.randint(6, 18)),
                     random.randint(2, 4), 180):
        state.add_transaction(
            coord, consol, random.randint(60000, 220000), ts,
            tx_type="transfer",
        )

    # Stage 4 - consolidator moves bulk to internal cash-out account.
    for ts in _burst(base + timedelta(hours=random.randint(20, 36)),
                     random.randint(1, 2), 240):
        state.add_transaction(
            consol, cashout, random.randint(120000, 280000), ts,
            tx_type="bank_transfer",
        )

    # Stage 5 - cash-out exits funds externally.
    for dest in ext_dests:
        state.add_transaction(
            cashout, dest, random.randint(80000, 240000),
            base + timedelta(hours=random.randint(40, 72)),
            tx_type="bank_transfer",
        )

    # Light intra-core chatter (mules occasionally cross-send).
    for _ in range(3):
        a, b = random.sample(core, 2)
        state.add_transaction(
            a, b, random.randint(10000, 60000),
            base + timedelta(hours=random.randint(2, 60)),
        )

    state.log_scenario(ring_id, "Classic money mule network", {
        "identity_signal": "shared phone + shared device",
        "core_accounts": len(core),
        "external_victims": len(victims),
        "external_destinations": len(ext_dests),
        "flow": "victims -> mules -> coordinator -> consolidator -> cash-out -> external",
    })
    return core


# --------------------------------------------------------------------------
# Scenario 2 - Device/identity reuse ring (shared device only, unique phones)
# --------------------------------------------------------------------------

def scenario_2_device_reuse_ring(state, normal_ids):
    ring_id = "ring_2"
    shared_device = "DEVR2SHARED"

    # All core accounts share one device but have unique phones.
    roles = ["mule", "mule", "coordinator", "consolidator", "consolidator", "cash_out"]
    core = []
    by_role = {}
    for role in roles:
        aid = state.add_account(
            phone=fake.phone_number(),  # unique phones - tests device signal
            device=shared_device,
            fraud_ring=ring_id, ground_truth_role=role,
            ground_truth_ring=ring_id,
        )
        core.append(aid)
        by_role.setdefault(role, []).append(aid)

    mules = by_role["mule"]
    coord = by_role["coordinator"][0]
    consols = by_role["consolidator"]
    cashout = by_role["cash_out"][0]

    # External inflows + outflows (NOT isolated).
    inflow_sources = []
    for _ in range(3):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVR2S"),
            fraud_ring="no",
            ground_truth_role="victim",
            ground_truth_ring=ring_id,
        )
        inflow_sources.append(aid)

    ext_dests = []
    for _ in range(2):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVR2D"),
            fraud_ring="no",
            ground_truth_role="cash_out",
            ground_truth_ring=ring_id,
        )
        ext_dests.append(aid)

    base = REFERENCE_NOW - timedelta(days=55, hours=random.randint(0, 12))

    # External sources feed mules.
    for i, src in enumerate(inflow_sources):
        mule = mules[i % len(mules)]
        for ts in _burst(base + timedelta(minutes=random.randint(5, 200)),
                         random.randint(1, 2), 180):
            state.add_transaction(src, mule, random.randint(30000, 150000), ts)

    # Heavy internal circulation - characteristic mule churn.
    for _ in range(12):
        a, b = random.sample(core, 2)
        state.add_transaction(
            a, b, random.randint(20000, 150000),
            base + timedelta(hours=random.randint(1, 72)),
            tx_type=random.choice(["transfer", "wallet_transfer"]),
        )

    # Mules -> coordinator -> consolidators -> cashout.
    for mule in mules:
        for ts in _burst(base + timedelta(hours=random.randint(2, 12)),
                         random.randint(2, 3), 120):
            state.add_transaction(mule, coord, random.randint(20000, 90000), ts)
    for consol in consols:
        for ts in _burst(base + timedelta(hours=random.randint(10, 30)),
                         random.randint(1, 2), 180):
            state.add_transaction(coord, consol, random.randint(50000, 150000), ts)
    state.add_transaction(
        consol, cashout, random.randint(100000, 220000),
        base + timedelta(hours=random.randint(30, 48)),
        tx_type="bank_transfer",
    )
    # Cash-out exits externally.
    for dest in ext_dests:
        state.add_transaction(
            cashout, dest, random.randint(60000, 180000),
            base + timedelta(hours=random.randint(48, 72)),
            tx_type="bank_transfer",
        )

    state.log_scenario(ring_id, "Device reuse ring", {
        "identity_signal": "shared device only (unique phones)",
        "core_accounts": len(core),
        "external_victims": len(inflow_sources),
        "external_destinations": len(ext_dests),
        "flow": "external -> mules <-> coordinator -> consolidators -> cash-out -> external",
    })
    return core


# --------------------------------------------------------------------------
# Scenario 3 - Low-identity-signal network (unique phones + unique devices)
# --------------------------------------------------------------------------

def scenario_3_low_identity_signal(state, normal_ids):
    ring_id = "ring_3"
    # Every account has unique phone AND unique device - no identity reuse.
    # Detection must rely purely on transaction patterns.

    roles = ["mule", "mule", "mule", "coordinator", "consolidator", "cash_out"]
    core = []
    by_role = {}
    for role in roles:
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVR3U"),  # unique
            ip=state.new_ip(),                   # unique IPs too
            fraud_ring=ring_id, ground_truth_role=role,
            ground_truth_ring=ring_id,
        )
        core.append(aid)
        by_role.setdefault(role, []).append(aid)

    mules = by_role["mule"]
    coord = by_role["coordinator"][0]
    consol = by_role["consolidator"][0]
    cashout = by_role["cash_out"][0]

    # Many external victims - this ring has wide reach.
    victims = []
    for _ in range(6):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVR3V"),
            fraud_ring="no",
            ground_truth_role="victim",
            ground_truth_ring=ring_id,
        )
        victims.append(aid)

    ext_dests = []
    for _ in range(3):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVR3E"),
            fraud_ring="no",
            ground_truth_role="cash_out",
            ground_truth_ring=ring_id,
        )
        ext_dests.append(aid)

    base = REFERENCE_NOW - timedelta(days=30, hours=random.randint(0, 12))

    # Each victim pays a different mule - distributed inbound.
    for i, victim in enumerate(victims):
        mule = mules[i % len(mules)]
        for ts in _burst(base + timedelta(minutes=random.randint(5, 360)),
                         random.randint(1, 2), 240):
            state.add_transaction(
                victim, mule, random.randint(20000, 120000), ts,
                tx_type=random.choice(["transfer", "wallet_transfer"]),
            )

    # Strong forwarding pattern: mules -> coordinator (rapid, high volume).
    for mule in mules:
        forward_base = base + timedelta(hours=random.randint(1, 10))
        for ts in _burst(forward_base, random.randint(3, 4), 90):
            state.add_transaction(
                mule, coord, random.randint(15000, 80000), ts,
                tx_type="wallet_transfer",
            )

    # Coordinator concentrates into consolidator.
    for ts in _burst(base + timedelta(hours=random.randint(12, 24)),
                     random.randint(3, 5), 150):
        state.add_transaction(
            coord, consol, random.randint(40000, 160000), ts,
            tx_type="transfer",
        )

    # Consolidator -> cashout (large).
    state.add_transaction(
        consol, cashout, random.randint(180000, 350000),
        base + timedelta(hours=random.randint(24, 40)),
        tx_type="bank_transfer",
    )

    # Cashout distributes externally (multiple destinations = strong cashout sig).
    for dest in ext_dests:
        state.add_transaction(
            cashout, dest, random.randint(80000, 200000),
            base + timedelta(hours=random.randint(40, 72)),
            tx_type="bank_transfer",
        )

    state.log_scenario(ring_id, "Low-identity-signal network", {
        "identity_signal": "unique phones + unique devices (behavior-only)",
        "core_accounts": len(core),
        "external_victims": len(victims),
        "external_destinations": len(ext_dests),
        "flow": "victims -> mules -> coordinator -> consolidator -> cash-out -> external",
    })
    return core


# --------------------------------------------------------------------------
# Scenario 4 - High-noise fraud network (partial device reuse + shared IP)
# --------------------------------------------------------------------------

def scenario_4_high_noise(state, normal_ids):
    ring_id = "ring_4"
    shared_ip = fake.ipv4_private()

    # Two devices, shared across the ring (partial reuse) + all share one IP.
    devices_pool = ["DEVR4A", "DEVR4B"]
    roles = ["mule", "mule", "coordinator", "consolidator", "cash_out"]
    core = []
    by_role = {}
    for i, role in enumerate(roles):
        aid = state.add_account(
            phone=fake.phone_number(),       # unique phones
            device=devices_pool[i % 2],      # two devices alternating
            ip=shared_ip,                     # all share one IP
            fraud_ring=ring_id, ground_truth_role=role,
            ground_truth_ring=ring_id,
        )
        core.append(aid)
        by_role.setdefault(role, []).append(aid)

    mules = by_role["mule"]
    coord = by_role["coordinator"][0]
    consol = by_role["consolidator"][0]
    cashout = by_role["cash_out"][0]

    victims = []
    for _ in range(3):
        aid = state.add_account(
            phone=fake.phone_number(),
            device=state.new_device("DEVR4V"),
            fraud_ring="no",
            ground_truth_role="victim",
            ground_truth_ring=ring_id,
        )
        victims.append(aid)

    ext_dest = state.add_account(
        phone=fake.phone_number(),
        device=state.new_device("DEVR4E"),
        fraud_ring="no",
        ground_truth_role="cash_out",
        ground_truth_ring=ring_id,
    )

    base = REFERENCE_NOW - timedelta(days=12, hours=random.randint(0, 12))

    # Standard fraud flow.
    for i, victim in enumerate(victims):
        mule = mules[i % len(mules)]
        state.add_transaction(
            victim, mule, random.randint(25000, 120000),
            base + timedelta(minutes=random.randint(5, 240)),
            tx_type="wallet_transfer",
        )
    for mule in mules:
        for ts in _burst(base + timedelta(hours=random.randint(1, 8)),
                         2, 90):
            state.add_transaction(mule, coord, random.randint(20000, 80000), ts)
    for ts in _burst(base + timedelta(hours=random.randint(8, 18)), 2, 150):
        state.add_transaction(coord, consol, random.randint(50000, 130000), ts)
    state.add_transaction(
        consol, cashout, random.randint(100000, 200000),
        base + timedelta(hours=random.randint(18, 32)),
        tx_type="bank_transfer",
    )
    state.add_transaction(
        cashout, ext_dest, random.randint(80000, 180000),
        base + timedelta(hours=random.randint(32, 48)),
        tx_type="bank_transfer",
    )

    # HIGH-NOISE LAYER: fraud accounts interact heavily with normal accounts.
    # This blurs the boundary and tests robustness.
    noise_count_per_acct = random.randint(6, 10)
    for fraud_acct in core:
        for _ in range(noise_count_per_acct):
            normal = random.choice(normal_ids)
            direction = random.choice(["out", "in"])
            amt = random.randint(1000, 12000)
            ts = base + timedelta(hours=random.randint(1, 80))
            if direction == "out":
                state.add_transaction(
                    fraud_acct, normal, amt, ts,
                    tx_type=random.choice(["transfer", "merchant_payment"]),
                )
            else:
                state.add_transaction(
                    normal, fraud_acct, amt, ts,
                    tx_type=random.choice(["transfer", "merchant_payment"]),
                )

    state.log_scenario(ring_id, "High-noise fraud network", {
        "identity_signal": "partial device reuse (2 devices) + shared IP",
        "core_accounts": len(core),
        "external_victims": len(victims),
        "external_destinations": 1,
        "noise_transactions_per_core_account": noise_count_per_acct,
        "flow": "victims -> mules -> coordinator -> consolidator -> cash-out -> external "
                 "(with heavy noise to/from normal accounts)",
    })
    return core


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------

def write_outputs(state):
    root_accounts = BASE_DIR / "accounts.csv"
    root_transactions = BASE_DIR / "transactions.csv"
    data_accounts = DATA_DIR / "accounts.csv"
    data_transactions = DATA_DIR / "transactions.csv"
    ground_truth_path = DATA_DIR / "ground_truth_roles.csv"

    def _write(path, rows, fields):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    _write(root_accounts, state.accounts, ACCOUNT_FIELDS)
    _write(root_transactions, state.transactions, TRANSACTION_FIELDS)
    _write(data_accounts, state.accounts, ACCOUNT_FIELDS)
    _write(data_transactions, state.transactions, TRANSACTION_FIELDS)
    _write(ground_truth_path, state.ground_truth, GROUND_TRUTH_FIELDS)

    return root_accounts, root_transactions, ground_truth_path


# --------------------------------------------------------------------------
# Summary printer
# --------------------------------------------------------------------------

def print_summary(state):
    role_counts = Counter(g["ground_truth_role"] for g in state.ground_truth)
    ring_counts = Counter(g["fraud_ring_id"] for g in state.ground_truth
                          if g["fraud_ring_id"] != "none")

    print("=" * 70)
    print("PHASE 5A - SYNTHETIC FRAUD NETWORK DATASET GENERATED")
    print("=" * 70)
    print(f"Total accounts:           {len(state.accounts)}")
    print(f"Total transactions:       {len(state.transactions)}")
    print(f"Fraud networks:           {len(set(g['fraud_ring_id'] for g in state.ground_truth) - {'none'})}")
    normal_count = sum(1 for g in state.ground_truth
                       if g["fraud_ring_id"] == "none")
    print(f"Normal accounts:          {normal_count}")
    print()
    print("Ground-truth role distribution:")
    for role in ["victim", "mule", "coordinator", "consolidator",
                 "cash_out", "normal"]:
        print(f"  {role:15s} {role_counts.get(role, 0)}")
    print()
    print("Per-scenario summaries:")
    for s in state.scenario_log:
        print(f"\n  [{s['ring_id']}] {s['title']}")
        print(f"     identity signal:  {s.get('identity_signal', '-')}")
        print(f"     core accounts:    {s.get('core_accounts', '-')}")
        print(f"     external victims: {s.get('external_victims', '-')}")
        print(f"     ext destinations: {s.get('external_destinations', '-')}")
        if "noise_transactions_per_core_account" in s:
            print(f"     noise tx/account: {s['noise_transactions_per_core_account']}")
        print(f"     flow:             {s.get('flow', '-')}")
    print()
    print("Output files:")
    print(f"  accounts.csv   ({len(state.accounts)} rows)")
    print(f"  transactions.csv ({len(state.transactions)} rows)")
    print(f"  data/ground_truth_roles.csv ({len(state.ground_truth)} rows)")
    print()
    print("NOTE: ground_truth_roles.csv is for evaluation ONLY.")
    print("      The detection pipeline (detect_fraud.py) does not load it.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    state = GenState()

    normal_ids = generate_normal_world(state)

    scenario_1_classic_mule_network(state, normal_ids)
    scenario_2_device_reuse_ring(state, normal_ids)
    scenario_3_low_identity_signal(state, normal_ids)
    scenario_4_high_noise(state, normal_ids)

    # Shuffle transactions so fraud isn't visually clustered in the file.
    random.shuffle(state.transactions)

    write_outputs(state)
    print_summary(state)


if __name__ == "__main__":
    main()
