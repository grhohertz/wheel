"""Simulated brokerage.

:class:`PaperBroker` is the ONLY broker in this package. It keeps cash, equity
and option positions, a full fill ledger, and handles expiration/assignment —
enough to run the wheel end to end without touching a live venue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

from .config import LiveTradingDisabled, StrategyParams
from .greeks import CALL, PUT
from .models import EquityPosition, OptionContract, OptionPosition, TradeRecord, utcnow


class InsufficientFunds(RuntimeError):
    pass


class InsufficientShares(RuntimeError):
    pass


class PositionNotFound(KeyError):
    pass


@dataclass
class PaperBroker:
    """In-memory simulated account with a persistent JSON ledger."""

    cash: float = 100_000.0
    account_id: str = "PAPER-0001"
    params: StrategyParams = field(default_factory=StrategyParams)
    equities: dict[str, EquityPosition] = field(default_factory=dict)
    options: dict[str, OptionPosition] = field(default_factory=dict)
    ledger: list[TradeRecord] = field(default_factory=list)
    realized_pnl: float = 0.0

    mode: str = "paper"

    def __post_init__(self) -> None:
        if self.mode != "paper":
            raise LiveTradingDisabled("PaperBroker only supports mode='paper'")

    # ------------------------------------------------------------------
    # fills
    # ------------------------------------------------------------------
    def sell_fill_price(self, contract: OptionContract) -> float:
        """Conservative sell fill: inside the spread, against us."""

        mid = contract.mid
        return round(max(contract.bid, mid * (1.0 - self.params.slippage_pct)), 2)

    def buy_fill_price(self, contract: OptionContract) -> float:
        mid = contract.mid
        return round(min(contract.ask, mid * (1.0 + self.params.slippage_pct)), 2)

    def _record(self, **kw) -> TradeRecord:
        rec = TradeRecord(ts=utcnow().isoformat(timespec="seconds"), **kw)
        self.ledger.append(rec)
        return rec

    # ------------------------------------------------------------------
    # equities
    # ------------------------------------------------------------------
    def buy_shares(self, symbol: str, quantity: int, price: float, note: str = "") -> TradeRecord:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        cost = quantity * price
        if cost > self.cash + 1e-9:
            raise InsufficientFunds(f"need ${cost:,.2f}, have ${self.cash:,.2f}")
        self.cash = round(self.cash - cost, 2)
        pos = self.equities.setdefault(symbol.upper(), EquityPosition(symbol.upper()))
        total = pos.quantity + quantity
        pos.average_cost = round((pos.cost_basis + cost) / total, 4) if total else 0.0
        pos.quantity = total
        return self._record(
            action="BUY_SHARES", symbol=symbol.upper(), quantity=quantity, price=price,
            commission=0.0, cash_delta=round(-cost, 2), note=note,
        )

    def sell_shares(self, symbol: str, quantity: int, price: float, note: str = "") -> TradeRecord:
        sym = symbol.upper()
        pos = self.equities.get(sym)
        if pos is None or pos.quantity < quantity:
            held = 0 if pos is None else pos.quantity
            raise InsufficientShares(f"{sym}: need {quantity} shares, hold {held}")
        proceeds = quantity * price
        realized = round((price - pos.average_cost) * quantity, 2)
        self.cash = round(self.cash + proceeds, 2)
        self.realized_pnl = round(self.realized_pnl + realized, 2)
        pos.quantity -= quantity
        if pos.quantity == 0:
            pos.average_cost = 0.0
            del self.equities[sym]
        return self._record(
            action="SELL_SHARES", symbol=sym, quantity=-quantity, price=price,
            commission=0.0, cash_delta=round(proceeds, 2), realized_pnl=realized, note=note,
        )

    def shares_held(self, symbol: str) -> int:
        pos = self.equities.get(symbol.upper())
        return pos.quantity if pos else 0

    # ------------------------------------------------------------------
    # options
    # ------------------------------------------------------------------
    def sell_to_open(
        self, contract: OptionContract, contracts: int = 1, price: float | None = None, note: str = ""
    ) -> TradeRecord:
        """Short ``contracts`` of ``contract``. Covered calls are enforced."""

        if contracts <= 0:
            raise ValueError("contracts must be positive")
        mult = self.params.contract_multiplier
        if contract.is_call():
            needed = contracts * mult
            held = self.shares_held(contract.underlying)
            covered = held - self._shares_committed(contract.underlying)
            if covered < needed:
                raise InsufficientShares(
                    f"{contract.underlying}: covered call needs {needed} uncommitted shares, "
                    f"{covered} available (naked calls are disabled)"
                )
        else:
            collateral = contracts * contract.strike * mult
            available = self.available_cash()
            if collateral > available + 1e-9:
                raise InsufficientFunds(
                    f"cash-secured put needs ${collateral:,.2f} collateral, "
                    f"${available:,.2f} uncommitted (cash ${self.cash:,.2f} less "
                    f"${self.put_collateral_committed():,.2f} already securing open puts)"
                )

        fill = price if price is not None else self.sell_fill_price(contract)
        commission = round(self.params.commission_per_contract * contracts, 2)
        credit = round(fill * contracts * mult - commission, 2)
        self.cash = round(self.cash + credit, 2)

        key = contract.symbol
        pos = self.options.get(key)
        if pos is None:
            self.options[key] = OptionPosition(
                underlying=contract.underlying, expiry=contract.expiry, strike=contract.strike,
                right=contract.right, quantity=-contracts, average_price=fill,
                opened_at=contract.as_of, multiplier=mult,
            )
        else:
            prior = abs(pos.quantity)
            total = prior + contracts
            pos.average_price = round((pos.average_price * prior + fill * contracts) / total, 4)
            pos.quantity -= contracts
        return self._record(
            action="SELL_TO_OPEN", symbol=key, quantity=-contracts, price=fill,
            commission=commission, cash_delta=credit, note=note,
        )

    def buy_to_close(
        self, option_symbol: str, contracts: int | None = None, price: float = 0.0, note: str = ""
    ) -> TradeRecord:
        pos = self.options.get(option_symbol)
        if pos is None:
            raise PositionNotFound(option_symbol)
        contracts = contracts or abs(pos.quantity)
        if contracts > abs(pos.quantity):
            raise ValueError(f"cannot close {contracts}, only {abs(pos.quantity)} open")
        mult = pos.multiplier
        commission = round(self.params.commission_per_contract * contracts, 2)
        debit = round(price * contracts * mult + commission, 2)
        self.cash = round(self.cash - debit, 2)
        realized = round((pos.average_price - price) * contracts * mult - commission, 2)
        self.realized_pnl = round(self.realized_pnl + realized, 2)

        pos.quantity += contracts
        if pos.quantity == 0:
            del self.options[option_symbol]
        return self._record(
            action="BUY_TO_CLOSE", symbol=option_symbol, quantity=contracts, price=price,
            commission=commission, cash_delta=round(-debit, 2), realized_pnl=realized, note=note,
        )

    def _shares_committed(self, underlying: str) -> int:
        """Shares already pledged against open short calls."""

        return sum(
            abs(p.quantity) * p.multiplier
            for p in self.options.values()
            if p.underlying == underlying.upper() and p.right.upper() == CALL and p.is_short
        )

    def open_short_options(self, underlying: str | None = None) -> list[OptionPosition]:
        return [
            p
            for p in self.options.values()
            if p.is_short and (underlying is None or p.underlying == underlying.upper())
        ]

    # ------------------------------------------------------------------
    # expiration / assignment
    # ------------------------------------------------------------------
    def process_expirations(self, as_of: date, spot_of: Callable[[str], float]) -> list[TradeRecord]:
        """Settle every option at/after expiry: worthless, or assigned."""

        out: list[TradeRecord] = []
        for symbol, pos in list(self.options.items()):
            if pos.expiry > as_of:
                continue
            spot = spot_of(pos.underlying)
            contracts = abs(pos.quantity)
            itm = (spot > pos.strike) if pos.right.upper() == CALL else (spot < pos.strike)
            credit = pos.credit_received()
            del self.options[symbol]

            if not itm:
                self.realized_pnl = round(self.realized_pnl + credit, 2)
                out.append(
                    self._record(
                        action="EXPIRED_WORTHLESS", symbol=symbol, quantity=contracts, price=0.0,
                        commission=0.0, cash_delta=0.0, realized_pnl=credit,
                        note=f"spot {spot:.2f} vs strike {pos.strike:.2f}",
                    )
                )
                continue

            self.realized_pnl = round(self.realized_pnl + credit, 2)
            out.append(
                self._record(
                    action="ASSIGNED", symbol=symbol, quantity=contracts, price=pos.strike,
                    commission=0.0, cash_delta=0.0, realized_pnl=credit,
                    note=f"ITM at expiry (spot {spot:.2f})",
                )
            )
            shares = contracts * pos.multiplier
            if pos.right.upper() == CALL:
                out.append(
                    self.sell_shares(pos.underlying, shares, pos.strike, note="called away")
                )
            else:
                cost = shares * pos.strike
                if cost > self.cash + 1e-9:
                    raise InsufficientFunds(
                        f"put assignment needs ${cost:,.2f}, have ${self.cash:,.2f}"
                    )
                out.append(
                    self.buy_shares(pos.underlying, shares, pos.strike, note="put assigned")
                )
        return out

    # ------------------------------------------------------------------
    # valuation / persistence
    # ------------------------------------------------------------------
    def equity_value(self, prices: dict[str, float]) -> float:
        return round(
            sum(p.market_value(prices.get(sym, p.average_cost)) for sym, p in self.equities.items()), 2
        )

    def option_value(self, marks: dict[str, float]) -> float:
        """Signed market value of option positions (negative for open shorts)."""

        return round(sum(p.market_value(marks.get(sym, 0.0)) for sym, p in self.options.items()), 2)

    def net_liquidation(self, prices: dict[str, float], marks: dict[str, float] | None = None) -> float:
        return round(self.cash + self.equity_value(prices) + self.option_value(marks or {}), 2)

    def put_collateral_committed(self) -> float:
        """Cash locked up securing open short puts (strike x multiplier x contracts)."""

        return round(
            sum(
                abs(p.quantity) * p.strike * p.multiplier
                for p in self.open_short_options()
                if p.right.upper() == "P"
            ),
            2,
        )

    def available_cash(self) -> float:
        """Cash that is NOT already securing an open short put."""

        return round(self.cash - self.put_collateral_committed(), 2)

    def collateral_used(self, prices: dict[str, float]) -> float:
        return round(
            sum(p.collateral(prices.get(p.underlying, p.strike)) for p in self.open_short_options()), 2
        )

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "mode": self.mode,
            "cash": round(self.cash, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "equities": [p.to_dict() for p in self.equities.values()],
            "options": [p.to_dict() for p in self.options.values()],
            "ledger": [t.to_dict() for t in self.ledger],
        }

    @classmethod
    def from_dict(cls, d: dict, params: StrategyParams | None = None) -> "PaperBroker":
        broker = cls(
            cash=float(d.get("cash", 0.0)),
            account_id=d.get("account_id", "PAPER-0001"),
            params=params or StrategyParams(),
            mode=d.get("mode", "paper"),
        )
        broker.realized_pnl = float(d.get("realized_pnl", 0.0))
        for e in d.get("equities", []):
            pos = EquityPosition.from_dict(e)
            broker.equities[pos.symbol] = pos
        for o in d.get("options", []):
            pos = OptionPosition.from_dict(o)
            broker.options[pos.symbol] = pos
        broker.ledger = [TradeRecord.from_dict(t) for t in d.get("ledger", [])]
        return broker

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(
        cls, path: str | Path, params: StrategyParams | None = None, default_cash: float = 100_000.0
    ) -> "PaperBroker":
        p = Path(path)
        if not p.exists():
            return cls(cash=default_cash, params=params or StrategyParams())
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), params=params)


def total_credits(ledger: Iterable[TradeRecord]) -> float:
    return round(sum(t.cash_delta for t in ledger if t.action == "SELL_TO_OPEN"), 2)


__all__ = [
    "InsufficientFunds",
    "InsufficientShares",
    "PaperBroker",
    "PositionNotFound",
    "total_credits",
    "CALL",
    "PUT",
]
