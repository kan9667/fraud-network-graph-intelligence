from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

ACCOUNTS_FILE = DATA_DIR / "accounts.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"

GRAPH_FILE = OUTPUT_DIR / "graph.gpickle"
CLUSTER_RESULTS_FILE = OUTPUT_DIR / "cluster_results.json"
FRAUD_GRAPH_HTML = OUTPUT_DIR / "fraud_graph.html"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

LOUVAIN_SEED = 42
TRANSACTION_EDGE_WEIGHT = 1
SHARED_IDENTITY_EDGE_WEIGHT = 10
MIN_CLUSTER_SIZE = 3

ACCOUNT_COLUMNS = [
    "account_id",
    "phone",
    "device_id",
    "ip_address",
    "email",
    "account_created_at",
    "account_status",
    "is_fraud_ring",
]
TRANSACTION_COLUMNS = [
    "transaction_id",
    "from_account",
    "to_account",
    "amount",
    "timestamp",
    "transaction_type",
    "channel",
]

GROUND_TRUTH_FILE = DATA_DIR / "ground_truth_roles.csv"
GROUND_TRUTH_COLUMNS = [
    "account_id",
    "fraud_ring_id",
    "ground_truth_role",
]

RISK_WEIGHTS = {
    "network_structure": 0.25,
    "identity_reuse": 0.30,
    "transaction_velocity": 0.10,
    "money_flow_concentration": 0.15,
    "rapid_forwarding": 0.10,
    "temporal_anomaly": 0.05,
    "external_counterparty": 0.05,
}

_weight_sum = sum(RISK_WEIGHTS.values())
if abs(_weight_sum - 1.0) > 1e-6:
    raise ValueError(f"RISK_WEIGHTS must sum to 1.0, got {_weight_sum}")

RISK_LEVEL_THRESHOLDS = {
    "LOW": 0,
    "MEDIUM": 30,
    "HIGH": 60,
    "CRITICAL": 80,
}

# ---------------------------------------------------------------------------
# Role classification & money-flow intelligence (Phase 5B)
# ---------------------------------------------------------------------------

# Rapid forwarding: outbound after inbound within this window (hours)
RAPID_FORWARDING_WINDOW_HOURS = 24

# Minimum role score / confidence before assigning a non-unknown role
MIN_ROLE_SCORE = 40
MIN_ROLE_CONFIDENCE = 40

# Cluster risk levels treated as "suspicious" for aggressive role inference
SUSPICIOUS_RISK_LEVELS = frozenset({"MEDIUM", "HIGH", "CRITICAL"})

# Cash-out multi-signal gates (absolute floors; scores also use relative signals)
CASHOUT_MIN_INTERNAL_INBOUND_VOLUME = 50_000
CASHOUT_MIN_EXTERNAL_OUTBOUND_VOLUME = 50_000
CASHOUT_MIN_PCT_OUTBOUND_TO_EXTERNAL = 50.0  # percent of outbound volume
CASHOUT_MIN_FORWARDING_RATIO = 0.40  # ext_out / max(internal_in, 1)
CASHOUT_MIN_SCORE = 55
CASHOUT_MAX_EXTERNAL_DESTINATIONS_FOR_CONCENTRATION = 5

# Victim calibration floors
VICTIM_MIN_OUTBOUND_TO_CLUSTER = 20_000
VICTIM_MAX_INTERNAL_INBOUND_RATIO = 0.25  # inbound from cluster / outbound to cluster
VICTIM_MIN_SCORE = 45

# Money-flow path discovery
MONEY_FLOW_MAX_PATH_LENGTH = 4
MONEY_FLOW_TOP_PATHS = 10
MONEY_FLOW_TOP_FLOWS = 10
MONEY_FLOW_MIN_EDGE_VOLUME = 1

# ---------------------------------------------------------------------------
# Offline evaluation only (Phase 7) — NOT used by detection or dashboard
# ---------------------------------------------------------------------------

# A planted fraud ring is "recovered" when its best-matching suspicious
# cluster meets both thresholds (objective measurement; not tuned for 100%).
RING_RECOVERY_MIN_RECALL = 0.40
RING_RECOVERY_MIN_F1 = 0.30

# Suspicious levels used when evaluating account-level detection
EVAL_SUSPICIOUS_LEVELS_MEDIUM_PLUS = frozenset({"MEDIUM", "HIGH", "CRITICAL"})
EVAL_SUSPICIOUS_LEVELS_HIGH_PLUS = frozenset({"HIGH", "CRITICAL"})

EVALUATION_REPORT_JSON = OUTPUT_DIR / "evaluation_report.json"
EVALUATION_REPORT_MD = OUTPUT_DIR / "evaluation_report.md"

# Scenario metadata for presentation tables (generation design labels only)
EVAL_SCENARIO_META = {
    "ring_1": {
        "title": "Classic money mule network",
        "identity_signal": "shared phone + shared device",
    },
    "ring_2": {
        "title": "Device reuse ring",
        "identity_signal": "shared device only (unique phones)",
    },
    "ring_3": {
        "title": "Low-identity-signal network",
        "identity_signal": "unique phones + unique devices (behavior-only)",
    },
    "ring_4": {
        "title": "High-noise fraud network",
        "identity_signal": "partial device reuse + shared IP",
    },
}

# Feature-score families for evaluation-only signal ablation
ABLATION_SIGNAL_FAMILIES = {
    "identity": ["identity_reuse"],
    "behavior": [
        "network_structure",
        "money_flow_concentration",
        "transaction_velocity",
        "external_counterparty",
    ],
    "temporal": ["rapid_forwarding", "temporal_anomaly"],
    "combined": [
        "network_structure",
        "identity_reuse",
        "transaction_velocity",
        "money_flow_concentration",
        "rapid_forwarding",
        "temporal_anomaly",
        "external_counterparty",
    ],
}
