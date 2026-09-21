"""Market-data providers.

``SyntheticMarketData`` is the default: a deterministic, offline option-chain
generator so the whole engine runs (and tests) with zero network access and zero
credentials. Swap in a real provider later by implementing :class:`MarketData`
— nothing else in the package needs to change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol, Sequence

from .greeks import CALL, PUT, black_scholes, year_fraction
from .models import OptionContract, Quote


class MarketData(Protocol):
    """Minimal surface the engine needs from any data source."""

    def get_quote(self, symbol: str, as_of: date) -> Quote: ...

    def get_chain(self, symbol: str, as_of: date) -> list[OptionContract]: ...


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.upper().encode()).hexdigest()[:12], 16)


def _spread(lo: float, hi: float, seed: int, salt: int = 0) -> float:
    """Deterministic value in [lo, hi)."""

    x = (seed >> (salt % 24)) ^ (seed * (salt + 1))
    return lo + (hi - lo) * ((x % 10_000) / 10_000.0)


def strike_increment(spot: float) -> float:
    if spot < 25:
        return 1.0
    if spot < 100:
        return 2.5
    if spot < 250:
        return 5.0
    return 10.0


def next_fridays(as_of: date, count: int = 12) -> list[date]:
    """The next ``count`` Fridays strictly after ``as_of``."""

    days_ahead = (4 - as_of.weekday()) % 7 or 7
    first = as_of + timedelta(days=days_ahead)
    return [first + timedelta(weeks=i) for i in range(count)]


@dataclass
class SyntheticMarketData:
    """Deterministic offline chain generator (same inputs -> same chain)."""

    rate: float = 0.04
    expiry_count: int = 12
    strike_span: float = 0.30  # +/- 30% around spot
    drift_per_day: float = 0.0  # move spot deterministically across sessions

    # -- underlying -----------------------------------------------------
    def get_quote(self, symbol: str, as_of: date | None = None) -> Quote:
        as_of = as_of or date.today()
        seed = _seed(symbol)
        base = round(_spread(18.0, 380.0, seed, 1), 2)
        iv = round(_spread(0.18, 0.62, seed, 5), 4)
        div = round(_spread(0.0, 0.035, seed, 9), 4)
        price = round(base * (1.0 + self.drift_per_day * as_of.toordinal() % 1), 2) if self.drift_per_day else base
        return Quote(symbol=symbol.upper(), price=price, iv=iv, div_yield=div, as_of=as_of)

    # -- chain ----------------------------------------------------------
    def get_chain(self, symbol: str, as_of: date | None = None) -> list[OptionContract]:
        as_of = as_of or date.today()
        quote = self.get_quote(symbol, as_of)
        seed = _seed(symbol)
        spot = quote.price
        inc = strike_increment(spot)
        lo = max(inc, round((spot * (1 - self.strike_span)) / inc) * inc)
        hi = round((spot * (1 + self.strike_span)) / inc) * inc

        strikes: list[float] = []
        k = lo
        while k <= hi + 1e-9:
            strikes.append(round(k, 2))
            k += inc

        contracts: list[OptionContract] = []
        for expiry in next_fridays(as_of, self.expiry_count):
            t = year_fraction((expiry - as_of).days)
            if t <= 0:
                continue
            for strike in strikes:
                for right in (CALL, PUT):
                    iv = self.smile_iv(quote.iv, spot, strike, t)
                    theo = black_scholes(
                        spot, strike, t, self.rate, iv, quote.div_yield, right
                    ).price
                    theo = max(theo, 0.01)
                    width = max(0.02, theo * _spread(0.02, 0.08, seed, int(strike) % 17))
                    bid = max(0.01, round(theo - width / 2.0, 2))
                    ask = round(bid + width, 2)
                    moneyness = abs(strike / spot - 1.0)
                    oi = int(max(5, 4000 * (1.0 - min(moneyness * 3.0, 0.95))))
                    contracts.append(
                        OptionContract(
                            underlying=symbol.upper(),
                            expiry=expiry,
                            strike=strike,
                            right=right,
                            bid=bid,
                            ask=ask,
                            iv=round(iv, 4),
                            open_interest=oi,
                            volume=max(1, oi // 8),
                            as_of=as_of,
                        )
                    )
        return contracts

    @staticmethod
    def smile_iv(base_iv: float, spot: float, strike: float, t: float) -> float:
        """Convex smile with a put-side skew, damped as expiry lengthens."""

        import math

        m = math.log(strike / spot)
        damp = 1.0 / (1.0 + 2.0 * t)
        iv = base_iv * (1.0 + damp * (0.9 * m * m - 0.22 * m))
        return min(max(iv, 0.05), 3.0)


@dataclass
class StaticMarketData:
    """Fixed quotes/chains — used by tests and for replaying a captured chain."""

    quotes: dict[str, Quote]
    chains: dict[str, Sequence[OptionContract]]

    def get_quote(self, symbol: str, as_of: date | None = None) -> Quote:
        try:
            return self.quotes[symbol.upper()]
        except KeyError as exc:
            raise KeyError(f"no quote configured for {symbol}") from exc

    def get_chain(self, symbol: str, as_of: date | None = None) -> list[OptionContract]:
        return list(self.chains.get(symbol.upper(), ()))


def price_contract(
    contract: OptionContract, spot: float, as_of: date, rate: float = 0.04, div_yield: float = 0.0
) -> float:
    """Theoretical mark for an existing contract at a (possibly later) date."""

    t = year_fraction((contract.expiry - as_of).days)
    return black_scholes(spot, contract.strike, t, rate, contract.iv, div_yield, contract.right).price


__all__ = [
    "MarketData",
    "StaticMarketData",
    "SyntheticMarketData",
    "next_fridays",
    "price_contract",
    "strike_increment",
    "CALL",
    "PUT",
]
