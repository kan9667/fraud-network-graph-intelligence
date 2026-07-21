import json
import pandas as pd

from src.config import (
    ACCOUNTS_FILE,
    TRANSACTIONS_FILE,
    ACCOUNT_COLUMNS,
    TRANSACTION_COLUMNS,
)


def parse_iso_timestamps(series):
    """Parse ISO-8601 timestamps with variable fractional-second precision.

    Accepts timestamps with or without microseconds, and with timezone info
    when present. Mixed naive/aware values are normalized to UTC.

    Invalid values become NaT (caller decides whether to raise).
    """
    try:
        return pd.to_datetime(series, errors="coerce", format="ISO8601")
    except (ValueError, TypeError):
        # Mixed timezones (e.g. some naive, some offset-aware) require utc=True
        return pd.to_datetime(series, errors="coerce", format="ISO8601", utc=True)


def load_accounts(path=None):
    if path is None:
        path = ACCOUNTS_FILE
    path = str(path)
    df = pd.read_csv(path)

    missing = [c for c in ACCOUNT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"accounts.csv missing required columns: {missing}")

    if df["account_id"].duplicated().any():
        dupes = df.loc[df["account_id"].duplicated(), "account_id"].tolist()
        raise ValueError(f"Duplicate account_ids found: {dupes}")

    if "account_created_at" in df.columns:
        df["account_created_at"] = parse_iso_timestamps(df["account_created_at"])

    return df


def load_transactions(path=None):
    if path is None:
        path = TRANSACTIONS_FILE
    path = str(path)
    df = pd.read_csv(path)

    missing = [c for c in TRANSACTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"transactions.csv missing required columns: {missing}")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad_amounts = df["amount"].isna()
    if bad_amounts.any():
        bad_rows = df.index[bad_amounts].tolist()
        raise ValueError(f"Non-numeric amounts found in rows: {bad_rows}")

    df["timestamp"] = parse_iso_timestamps(df["timestamp"])
    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        bad_rows = df.index[bad_ts].tolist()
        raise ValueError(f"Unparseable timestamps found in rows: {bad_rows}")

    return df


def load_accounts_with_validation(accounts_df, transactions_df):
    known_ids = set(accounts_df["account_id"])
    tx_ids = set(transactions_df["from_account"]) | set(transactions_df["to_account"])
    missing = tx_ids - known_ids
    if missing:
        raise ValueError(
            f"Transactions reference {len(missing)} account(s) not in accounts.csv"
        )


def load_json(path):
    path = str(path)
    with open(path) as f:
        return json.load(f)
