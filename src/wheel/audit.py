"""Audit ledger for advisory recommendations.

Append-only JSONL with strict schema. Every Claude call is logged with:
- Inputs (feature digest, feature vector)
- Raw response + parsed recommendation
- Guardrails applied
- Effective params used
- Latency, token count, cost

Used for postmortem attribution and scoring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional


@dataclass
class AdvisoryCall:
    """Single record: one call to Claude, logged to ledger."""

    # Identification
    advice_id: str  # "YYYY-MM-DDTHH:MM:SSZ-<8-hex>"
    ts: str  # ISO 8601, when call was made
    symbol: str  # ticker

    # Prompt versioning
    prompt_version: str  # e.g., "delta.v2", bumped on any prompt change
    model: str  # e.g., "claude-sonnet-4-5-20250929", pinned (not alias)

    # Inputs
    features_dict: dict  # serialized FeatureVector
    features_digest: str  # sha256(json.dumps(features, sort_keys=True))

    # Response
    raw_response: str  # full API response body, for audit
    parsed_recommendation: dict  # extracted JSON: {put_delta_target, call_delta_target, ...}

    # Guardrails
    guardrails_applied: list[str] = field(default_factory=list)  # e.g., ["clamp put_delta_target 0.35→0.32"]
    effective_params: dict = field(default_factory=dict)  # {put_delta_target: 0.32, ...} after clamps

    # Metrics
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    # Optional: realized outcome (filled in later)
    outcome: Optional[dict] = None  # {realized_assignment_prob, realized_iv_move, ...}
    outcome_ts: Optional[str] = None  # when outcome was recorded

    def to_dict(self) -> dict:
        """Serialize to JSON."""
        return asdict(self)

    def to_jsonl(self) -> str:
        """Serialize to one line of JSONL."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_dict(d: dict) -> AdvisoryCall:
        """Deserialize from dict."""
        return AdvisoryCall(**d)

    @staticmethod
    def from_jsonl(line: str) -> AdvisoryCall:
        """Deserialize from one line of JSONL."""
        return AdvisoryCall.from_dict(json.loads(line))


class AuditLedger:
    """Append-only JSONL ledger of advisory calls.

    Location: state/advisory/ledger.jsonl
    One record per Claude call, never modified (only appended).
    """

    def __init__(self, path: str | Path = "state/advisory/ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, call: AdvisoryCall) -> None:
        """Append one call record to the ledger."""
        with open(self.path, "a") as f:
            f.write(call.to_jsonl())
            f.write("\n")

    def read_all(self) -> list[AdvisoryCall]:
        """Read all records from the ledger."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(AdvisoryCall.from_jsonl(line))
        return records

    def read_by_symbol(self, symbol: str, limit: Optional[int] = None) -> list[AdvisoryCall]:
        """Read all records for a symbol, most recent first."""
        all_records = self.read_all()
        filtered = [r for r in all_records if r.symbol == symbol]
        filtered.reverse()  # most recent first
        if limit:
            filtered = filtered[:limit]
        return filtered

    def stats(self) -> dict:
        """Compute aggregate stats from the ledger."""
        records = self.read_all()
        if not records:
            return {"total_calls": 0, "total_cost_usd": 0.0, "total_tokens": 0}

        total_cost = sum(r.cost_usd for r in records)
        total_tokens = sum(r.tokens_in + r.tokens_out for r in records)

        return {
            "total_calls": len(records),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_call": total_cost / len(records),
            "latest_call_ts": records[-1].ts if records else None,
        }


class CurrentAdvisory:
    """Cache of latest advisory per symbol.

    Location: state/advisory/current.json
    {
        "AAPL": {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "ts": "2025-09-22T13:30:00Z",
            "recommendation": {...},
            "effective_params": {...},
            "expires_at": "2025-09-23T13:30:00Z"
        }
    }
    """

    def __init__(self, path: str | Path = "state/advisory/current.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        """Load current advisory state."""
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            return json.load(f)

    def save(self, data: dict) -> None:
        """Write current advisory state."""
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, symbol: str) -> Optional[dict]:
        """Get advisory for one symbol."""
        data = self.load()
        return data.get(symbol)

    def set(self, symbol: str, advice: dict) -> None:
        """Update advisory for one symbol."""
        data = self.load()
        data[symbol] = advice
        self.save(data)

    def is_expired(self, symbol: str) -> bool:
        """Check if advisory for symbol is expired."""
        advice = self.get(symbol)
        if not advice or "expires_at" not in advice:
            return True
        expires_at = datetime.fromisoformat(advice["expires_at"])
        return datetime.now() > expires_at

    def invalidate(self, symbol: str) -> None:
        """Mark advisory as expired (force refresh on next read)."""
        advice = self.get(symbol)
        if advice:
            advice["expires_at"] = datetime.now().isoformat()
            self.set(symbol, advice)


def compute_features_digest(features_dict: dict) -> str:
    """Compute SHA256 digest of feature vector for caching/dedup.

    Args:
        features_dict: serialized FeatureVector

    Returns:
        hex digest
    """
    s = json.dumps(features_dict, sort_keys=True, separators=(",", ":"))
    return sha256(s.encode()).hexdigest()


# Example usage / validation
if __name__ == "__main__":
    # Create a sample call record
    call = AdvisoryCall(
        advice_id="2025-09-22T13:30:00Z-abc1def2",
        ts="2025-09-22T13:30:00Z",
        symbol="AAPL",
        prompt_version="delta.v1",
        model="claude-sonnet-4-5-20250929",
        features_dict={"iv_rank": 45, "market_regime": "normal"},
        features_digest="abc123",
        raw_response='{"recommendation": {"put_delta_target": 0.30}}',
        parsed_recommendation={"put_delta_target": 0.30, "call_delta_target": 0.25},
        guardrails_applied=["clamp put_delta_target 0.30→0.30"],
        effective_params={"put_delta_target": 0.30, "call_delta_target": 0.25},
        latency_ms=2140,
        tokens_in=3800,
        tokens_out=620,
        cost_usd=0.0207,
    )

    print("AdvisoryCall:")
    print(json.dumps(call.to_dict(), indent=2))

    # Test ledger operations
    ledger = AuditLedger("state/advisory/test_ledger.jsonl")
    ledger.append(call)
    records = ledger.read_all()
    print(f"\nLedger has {len(records)} record(s)")
    print(f"Stats: {ledger.stats()}")

    # Test current advisory cache
    current = CurrentAdvisory("state/advisory/test_current.json")
    current.set("AAPL", {
        "advice_id": call.advice_id,
        "ts": call.ts,
        "recommendation": call.parsed_recommendation,
        "effective_params": call.effective_params,
        "expires_at": "2025-09-23T13:30:00Z",
    })
    loaded = current.get("AAPL")
    print(f"\nStored and loaded AAPL advice: {loaded}")
