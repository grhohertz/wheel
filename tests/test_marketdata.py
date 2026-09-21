from datetime import date

import pytest
from wheel.marketdata import SyntheticMarketData, next_fridays, strike_increment

AS_OF = date(2026, 1, 5)  # a Monday


def test_next_fridays_are_fridays_and_forward():
    fridays = next_fridays(AS_OF, 5)
    assert len(fridays) == 5
    assert all(f.weekday() == 4 for f in fridays)
    assert all(f > AS_OF for f in fridays)
    assert fridays == sorted(fridays)


def test_strike_increment_tiers():
    assert strike_increment(10) == 1.0
    assert strike_increment(50) == 2.5
    assert strike_increment(150) == 5.0
    assert strike_increment(500) == 10.0


def test_quote_is_deterministic():
    md = SyntheticMarketData()
    a = md.get_quote("AAPL", AS_OF)
    b = md.get_quote("AAPL", AS_OF)
    assert a == b
    assert a.price > 0
    assert 0.05 <= a.iv <= 1.0
    assert md.get_quote("MSFT", AS_OF).price != a.price


def test_chain_is_well_formed():
    md = SyntheticMarketData(expiry_count=4)
    quote = md.get_quote("KO", AS_OF)
    chain = md.get_chain("KO", AS_OF)
    assert chain
    for c in chain:
        assert c.ask >= c.bid > 0
        assert c.iv > 0
        assert c.dte > 0
        assert c.right in ("C", "P")
        assert c.underlying == "KO"
    calls = [c for c in chain if c.is_call()]
    puts = [c for c in chain if c.is_put()]
    assert len(calls) == len(puts)
    # Calls get cheaper as strikes rise, within a single expiry.
    exp = min(c.expiry for c in chain)
    same = sorted((c for c in calls if c.expiry == exp), key=lambda c: c.strike)
    mids = [c.mid for c in same]
    assert mids[0] > mids[-1]
    assert all(c.strike > 0 for c in same)
    assert any(c.strike > quote.price for c in same)


def test_chain_deltas_span_the_wheel_target():
    md = SyntheticMarketData(expiry_count=8)
    quote = md.get_quote("F", AS_OF)
    chain = md.get_chain("F", AS_OF)
    deltas = [
        abs(c.greeks(quote.price, 0.04, quote.div_yield).delta)
        for c in chain
        if c.is_call() and 30 <= c.dte <= 45
    ]
    assert deltas, "expected contracts in the 30-45 DTE window"
    assert min(deltas) < 0.30 < max(deltas)


def test_smile_is_convex_and_bounded():
    md = SyntheticMarketData()
    base = 0.35
    atm = md.smile_iv(base, 100, 100, 0.1)
    wing_up = md.smile_iv(base, 100, 130, 0.1)
    wing_dn = md.smile_iv(base, 100, 70, 0.1)
    assert atm == pytest.approx(base, abs=1e-9)
    assert wing_up > atm and wing_dn > atm
    assert 0.05 <= wing_dn <= 3.0
