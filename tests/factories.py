"""Deterministic fixtures shared across the test suite."""

from __future__ import annotations

from datetime import date, timedelta

from wheel.greeks import black_scholes, year_fraction
from wheel.models import OptionContract, Quote

AS_OF = date(2026, 1, 5)
SPOT = 50.0
IV = 0.35
RATE = 0.04


def make_quote(symbol: str = "TEST", spot: float = SPOT, iv: float = IV, div: float = 0.0) -> Quote:
    return Quote(symbol=symbol, price=spot, iv=iv, div_yield=div, as_of=AS_OF)


def make_chain(
    symbol: str = "TEST",
    spot: float = SPOT,
    iv: float = IV,
    as_of: date = AS_OF,
    dtes: tuple[int, ...] = (7, 35, 60),
    div: float = 0.0,
    open_interest: int = 1000,
) -> list[OptionContract]:
    """A clean, liquid chain: $1 strikes from 80% to 120% of spot."""

    contracts: list[OptionContract] = []
    strikes = [round(spot * 0.8 + i, 2) for i in range(int(spot * 0.4) + 1)]
    for dte in dtes:
        expiry = as_of + timedelta(days=dte)
        for strike in strikes:
            for right in ("C", "P"):
                theo = max(
                    black_scholes(spot, strike, year_fraction(dte), RATE, iv, div, right).price, 0.05
                )
                width = max(0.02, theo * 0.04)
                bid = round(max(0.05, theo - width / 2), 2)
                contracts.append(
                    OptionContract(
                        underlying=symbol,
                        expiry=expiry,
                        strike=strike,
                        right=right,
                        bid=bid,
                        ask=round(bid + width, 2),
                        iv=iv,
                        open_interest=open_interest,
                        volume=open_interest // 4,
                        as_of=as_of,
                    )
                )
    return contracts
