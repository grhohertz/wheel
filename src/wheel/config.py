"""Runtime settings and the paper-trading guard.

The guard is deliberately blunt: any mode other than ``paper`` raises. Live
execution is not implemented anywhere in this package, so a misconfigured
environment fails loudly at start-up instead of silently doing something
surprising.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PAPER = "paper"
LIVE = "live"


class LiveTradingDisabled(RuntimeError):
    """Raised whenever something attempts to leave paper-trading mode."""


@dataclass(frozen=True)
class StrategyParams:
    """Tunable knobs for the wheel."""

    target_delta: float = 0.25  # aggressive: sell further OTM for better risk-adjusted returns
    delta_tolerance: float = 0.15
    min_dte: int = 30
    max_dte: int = 45
    profit_target: float = 0.50  # buy back after capturing 50% of the credit
    roll_dte: int = 7  # manage anything with <= 7 days left
    contract_multiplier: int = 100
    min_open_interest: int = 50
    max_spread_pct: float = 0.20  # (ask-bid)/mid
    max_collateral_pct: float = 0.50  # single position may not eat >50% of equity
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.02  # fill this far inside the spread, against us
    avoid_below_basis: bool = True  # never sell a call under the share cost basis
    risk_free_rate: float = 0.04

    def __post_init__(self) -> None:
        if not 0.0 < self.target_delta < 1.0:
            raise ValueError("target_delta must be in (0, 1)")
        if self.min_dte > self.max_dte:
            raise ValueError("min_dte must be <= max_dte")
        if not 0.0 < self.profit_target <= 1.0:
            raise ValueError("profit_target must be in (0, 1]")


@dataclass(frozen=True)
class Settings:
    """Top-level configuration. ``mode`` is always validated on construction."""

    mode: str = PAPER
    account_id: str = "PAPER-0001"
    starting_cash: float = 100_000.0
    state_path: str = "state/paper_account.json"
    watchlist: tuple[str, ...] = ("AAPL", "MSFT", "KO", "F", "T")
    params: StrategyParams = field(default_factory=StrategyParams)

    def __post_init__(self) -> None:
        assert_paper_mode(self.mode)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        watchlist = env.get("WHEEL_WATCHLIST", "")
        kwargs: dict[str, object] = {
            "mode": env.get("WHEEL_MODE", PAPER).strip().lower(),
            "account_id": env.get("WHEEL_ACCOUNT_ID", "PAPER-0001"),
            "state_path": env.get("WHEEL_STATE_PATH", "state/paper_account.json"),
        }
        if "WHEEL_STARTING_CASH" in env:
            kwargs["starting_cash"] = float(env["WHEEL_STARTING_CASH"])
        if watchlist.strip():
            kwargs["watchlist"] = tuple(
                s.strip().upper() for s in watchlist.split(",") if s.strip()
            )
        return cls(**kwargs)  # type: ignore[arg-type]


def assert_paper_mode(mode: str) -> None:
    """Raise unless ``mode`` is exactly ``paper``."""

    if (mode or "").strip().lower() != PAPER:
        raise LiveTradingDisabled(
            f"mode={mode!r} is refused: this build only supports paper trading. "
            "Set WHEEL_MODE=paper."
        )
