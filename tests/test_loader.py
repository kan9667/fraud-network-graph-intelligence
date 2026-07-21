"""Tests for CSV loading and ISO-8601 timestamp parsing."""

import io
import textwrap

import pandas as pd
import pytest

from src.loader import load_transactions, parse_iso_timestamps


def _tx_csv(*timestamp_rows):
    """Build a minimal transactions CSV string with the given timestamps."""
    header = (
        "transaction_id,from_account,to_account,amount,"
        "timestamp,transaction_type,channel\n"
    )
    lines = []
    for i, ts in enumerate(timestamp_rows):
        lines.append(
            f"TXN{i:04d},ACC0001,ACC0002,1000,{ts},P2P,UPI\n"
        )
    return header + "".join(lines)


class TestParseIsoTimestamps:
    def test_timestamp_with_microseconds(self):
        s = pd.Series(["2026-06-16T09:08:58.215068"])
        result = parse_iso_timestamps(s)
        assert result.notna().all()
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2026-06-16 09:08:58.215068")

    def test_timestamp_without_microseconds(self):
        s = pd.Series(["2026-04-12T08:48:00"])
        result = parse_iso_timestamps(s)
        assert result.notna().all()
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2026-04-12 08:48:00")

    def test_invalid_timestamp_becomes_nat(self):
        s = pd.Series(["not-a-timestamp", "2026-99-99T99:99:99"])
        result = parse_iso_timestamps(s)
        assert result.isna().all()

    def test_mixed_timestamp_formats(self):
        s = pd.Series([
            "2026-06-16T09:08:58.215068",
            "2026-04-12T08:48:00",
            "2026-06-21T08:00:00",
            "2026-05-03T18:22:30.001791",
        ])
        result = parse_iso_timestamps(s)
        assert result.notna().all()
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2026-06-16 09:08:58.215068")
        assert result.iloc[1] == pd.Timestamp("2026-04-12 08:48:00")

    def test_timezone_aware_timestamps(self):
        s = pd.Series([
            "2026-06-16T09:08:58.215068Z",
            "2026-04-12T08:48:00+00:00",
        ])
        result = parse_iso_timestamps(s)
        assert result.notna().all()
        assert pd.api.types.is_datetime64_any_dtype(result)


class TestLoadTransactionsTimestamps:
    def test_loads_with_microseconds(self, tmp_path):
        path = tmp_path / "tx.csv"
        path.write_text(_tx_csv("2026-06-16T09:08:58.215068"))
        df = load_transactions(path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].notna().all()

    def test_loads_without_microseconds(self, tmp_path):
        path = tmp_path / "tx.csv"
        path.write_text(_tx_csv("2026-04-12T08:48:00", "2026-06-21T08:00:00"))
        df = load_transactions(path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].notna().all()
        assert len(df) == 2

    def test_loads_mixed_precision(self, tmp_path):
        path = tmp_path / "tx.csv"
        path.write_text(
            _tx_csv(
                "2026-06-16T09:08:58.215068",
                "2026-04-12T08:48:00",
                "2026-05-03T18:22:30.001791",
                "2026-06-21T08:00:00",
            )
        )
        df = load_transactions(path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].notna().all()
        assert len(df) == 4

    def test_invalid_timestamp_raises(self, tmp_path):
        path = tmp_path / "tx.csv"
        path.write_text(_tx_csv("2026-06-16T09:08:58.215068", "not-a-date"))
        with pytest.raises(ValueError, match="Unparseable timestamps"):
            load_transactions(path)

    def test_non_numeric_amount_raises(self, tmp_path):
        path = tmp_path / "tx.csv"
        path.write_text(
            "transaction_id,from_account,to_account,amount,"
            "timestamp,transaction_type,channel\n"
            "TXN0001,ACC0001,ACC0002,not-a-number,"
            "2026-04-12T08:48:00,P2P,UPI\n"
        )
        with pytest.raises(ValueError, match="Non-numeric amounts"):
            load_transactions(path)
