"""Core value objects: quotes, contracts, positions, trades, recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum

from .greeks import CALL, PUT, Greeks, black_scholes, year_fraction


class Action(str, Enum):
    """What the strategy wants to do about a symbol."""

    SELL_COVERED_CALL = "SELL_COVERED_CALL"
    SELL_CASH_SECURED_PUT = "SELL_CASH_SECURED_PUT"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    ROLL = "ROLL"
    HOLD = "HOLD"
    SKIP = "SKIP"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def occ_symbol(underlying: str, expiry: date, right: str, strike: float) -> str:
    """OCC-21 option symbol, e.g. ``AAPL  260116C00185000``."""

    root = underlying.upper()[:6].ljust(6)
    return f"{root}{expiry:%y%m%d}{right.upper()}{int(round(strike * 1000)):08d}"


@dataclass(frozen=True)
class Quote:
    """Underlying snapshot."""

    symbol: str
    price: float
    iv: float
    div_yield: float = 0.0
    as_of: date = field(default_factory=lambda: utcnow().date())

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("quote price must be positive")


@dataclass(frozen=True)
class OptionContract:
    """A single quoted option contract."""

    underlying: str
    expiry: date
    strike: float
    right: str  # 'C' or 'P'
    bid: float
    ask: float
    iv: float
    open_interest: int = 0
    volume: int = 0
    as_of: date = field(default_factory=lambda: utcnow().date())

    @property
    def symbol(self) -> str:
        return occ_symbol(self.underlying, self.expiry, self.right, self.strike)

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2.0, 4)

    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 4)

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        return float("inf") if mid <= 0 else self.spread / mid

    @property
    def dte(self) -> int:
        return (self.expiry - self.as_of).days

    def is_call(self) -> bool:
        return self.right.upper() == CALL

    def is_put(self) -> bool:
        return self.right.upper() == PUT

    def greeks(self, spot: float, rate: float = 0.04, div_yield: float = 0.0) -> Greeks:
        return black_scholes(
            spot=spot,
            strike=self.strike,
            t=year_fraction(self.dte),
            rate=rate,
            sigma=self.iv,
            div_yield=div_yield,
            right=self.right,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expiry"] = self.expiry.isoformat()
        d["as_of"] = self.as_of.isoformat()
        d["symbol"] = self.symbol
        return d


@dataclass
class EquityPosition:
    symbol: str
    quantity: int = 0
    average_cost: float = 0.0

    @property
    def cost_basis(self) -> float:
        return round(self.quantity * self.average_cost, 2)

    def market_value(self, price: float) -> float:
        return round(self.quantity * price, 2)

    def unrealized(self, price: float) -> float:
        return round(self.market_value(price) - self.cost_basis, 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EquityPosition":
        return cls(symbol=d["symbol"], quantity=int(d["quantity"]), average_cost=float(d["average_cost"]))


@dataclass
class OptionPosition:
    """Open option position. ``quantity`` is negative when short."""

    underlying: str
    expiry: date
    strike: float
    right: str
    quantity: int
    average_price: float  # per share, always positive (the credit/debit paid)
    opened_at: date = field(default_factory=lambda: utcnow().date())
    multiplier: int = 100

    @property
    def symbol(self) -> str:
        return occ_symbol(self.underlying, self.expiry, self.right, self.strike)

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def dte(self, as_of: date) -> int:
        return (self.expiry - as_of).days

    def credit_received(self) -> float:
        """Total premium collected when short (0 for longs)."""

        return round(-self.quantity * self.average_price * self.multiplier, 2) if self.is_short else 0.0

    def market_value(self, option_price: float) -> float:
        return round(self.quantity * option_price * self.multiplier, 2)

    def unrealized(self, option_price: float) -> float:
        """Short: credit minus current cost to close."""

        opened = self.quantity * self.average_price * self.multiplier
        return round(self.quantity * option_price * self.multiplier - opened, 2)

    def profit_captured(self, option_price: float) -> float:
        """Fraction of the original credit already captured (shorts only)."""

        if not self.is_short or self.average_price <= 0:
            return 0.0
        return (self.average_price - option_price) / self.average_price

    def collateral(self, spot: float) -> float:
        """Cash/shares tied up. Covered calls consume shares, not cash."""

        if not self.is_short:
            return 0.0
        contracts = abs(self.quantity)
        if self.right.upper() == PUT:
            return round(contracts * self.strike * self.multiplier, 2)
        return round(contracts * spot * self.multiplier, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expiry"] = self.expiry.isoformat()
        d["opened_at"] = self.opened_at.isoformat()
        d["symbol"] = self.symbol
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OptionPosition":
        return cls(
            underlying=d["underlying"],
            expiry=date.fromisoformat(d["expiry"]),
            strike=float(d["strike"]),
            right=d["right"],
            quantity=int(d["quantity"]),
            average_price=float(d["average_price"]),
            opened_at=date.fromisoformat(d["opened_at"]),
            multiplier=int(d.get("multiplier", 100)),
        )


@dataclass
class TradeRecord:
    """One simulated fill in the ledger."""

    ts: str
    action: str
    symbol: str
    quantity: int
    price: float
    commission: float
    cash_delta: float
    realized_pnl: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        return cls(**d)


@dataclass
class Recommendation:
    """What the engine proposes for one symbol, with the math behind it."""

    symbol: str
    action: Action
    rationale: str
    spot: float = 0.0
    contract: OptionContract | None = None
    contracts: int = 0
    greeks: Greeks | None = None
    credit: float = 0.0  # total premium for the whole order
    buyback_target: float = 0.0  # per-share price to close at
    collateral: float = 0.0
    collateral_pct: float = 0.0
    annualized_yield: float = 0.0
    confidence: float = 0.0
    risks: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        return self.action not in (Action.SKIP, Action.HOLD)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "rationale": self.rationale,
            "spot": round(self.spot, 4),
            "contract": self.contract.to_dict() if self.contract else None,
            "contracts": self.contracts,
            "greeks": (
                {
                    "price": round(self.greeks.price, 4),
                    "delta": round(self.greeks.delta, 4),
                    "gamma": round(self.greeks.gamma, 5),
                    "theta": round(self.greeks.theta, 4),
                    "vega": round(self.greeks.vega, 4),
                    "rho": round(self.greeks.rho, 4),
                }
                if self.greeks
                else None
            ),
            "credit": round(self.credit, 2),
            "buyback_target": round(self.buyback_target, 4),
            "collateral": round(self.collateral, 2),
            "collateral_pct": round(self.collateral_pct, 4),
            "annualized_yield": round(self.annualized_yield, 4),
            "confidence": round(self.confidence, 3),
            "risks": list(self.risks),
        }
