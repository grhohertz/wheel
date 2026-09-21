from datetime import date, timedelta

import pytest
from factories import AS_OF, make_chain
from wheel.broker import InsufficientFunds, InsufficientShares, PaperBroker, PositionNotFound
from wheel.config import LiveTradingDisabled, StrategyParams


def _call(strike: float = 53.0, dte: int = 35):
    return next(
        c
        for c in make_chain()
        if c.is_call() and c.strike == strike and c.expiry == AS_OF + timedelta(days=dte)
    )


def _put(strike: float = 47.0, dte: int = 35):
    return next(
        c
        for c in make_chain()
        if c.is_put() and c.strike == strike and c.expiry == AS_OF + timedelta(days=dte)
    )


@pytest.fixture()
def broker() -> PaperBroker:
    return PaperBroker(cash=100_000.0, params=StrategyParams())


def test_live_mode_is_refused():
    with pytest.raises(LiveTradingDisabled):
        PaperBroker(mode="live")


def test_buy_and_sell_shares_tracks_basis_and_realized(broker):
    broker.buy_shares("TEST", 100, 50.0)
    assert broker.cash == pytest.approx(95_000.0)
    assert broker.shares_held("TEST") == 100
    broker.buy_shares("TEST", 100, 60.0)
    assert broker.equities["TEST"].average_cost == pytest.approx(55.0)
    broker.sell_shares("TEST", 200, 58.0)
    assert broker.realized_pnl == pytest.approx(600.0)
    assert broker.shares_held("TEST") == 0
    assert "TEST" not in broker.equities


def test_cannot_oversell_shares(broker):
    with pytest.raises(InsufficientShares):
        broker.sell_shares("TEST", 1, 50.0)


def test_cannot_buy_without_cash(broker):
    with pytest.raises(InsufficientFunds):
        broker.buy_shares("TEST", 10_000, 50.0)


def test_naked_calls_are_blocked(broker):
    with pytest.raises(InsufficientShares):
        broker.sell_to_open(_call(), 1)


def test_covered_call_credits_cash_net_of_commission(broker):
    broker.buy_shares("TEST", 100, 50.0)
    cash_before = broker.cash
    contract = _call()
    trade = broker.sell_to_open(contract, 1)
    expected_fill = broker.sell_fill_price(contract)
    assert trade.price == pytest.approx(expected_fill)
    assert broker.cash == pytest.approx(cash_before + expected_fill * 100 - 0.65)
    pos = broker.options[contract.symbol]
    assert pos.quantity == -1
    assert pos.credit_received() == pytest.approx(expected_fill * 100)


def test_second_covered_call_needs_more_shares(broker):
    broker.buy_shares("TEST", 100, 50.0)
    broker.sell_to_open(_call(), 1)
    with pytest.raises(InsufficientShares):
        broker.sell_to_open(_call(strike=54.0), 1)


def test_cash_secured_put_requires_collateral(broker):
    small = PaperBroker(cash=1_000.0)
    with pytest.raises(InsufficientFunds):
        small.sell_to_open(_put(), 1)
    broker.sell_to_open(_put(), 1)
    assert broker.options[_put().symbol].quantity == -1


def test_buy_to_close_realizes_profit(broker):
    broker.buy_shares("TEST", 100, 50.0)
    contract = _call()
    fill = broker.sell_fill_price(contract)
    broker.sell_to_open(contract, 1)
    broker.buy_to_close(contract.symbol, 1, price=fill / 2)
    assert contract.symbol not in broker.options
    expected = (fill - fill / 2) * 100 - 0.65
    assert broker.realized_pnl == pytest.approx(expected)


def test_buy_to_close_unknown_position(broker):
    with pytest.raises(PositionNotFound):
        broker.buy_to_close("NOPE  260206C00053000", 1, 1.0)


def test_expire_worthless_keeps_full_credit(broker):
    broker.buy_shares("TEST", 100, 50.0)
    contract = _call()
    fill = broker.sell_fill_price(contract)
    broker.sell_to_open(contract, 1)
    trades = broker.process_expirations(contract.expiry, lambda s: 48.0)
    assert [t.action for t in trades] == ["EXPIRED_WORTHLESS"]
    assert broker.realized_pnl == pytest.approx(fill * 100)
    assert broker.shares_held("TEST") == 100
    assert not broker.options


def test_call_assignment_sells_shares_at_strike(broker):
    broker.buy_shares("TEST", 100, 50.0)
    contract = _call(strike=53.0)
    fill = broker.sell_fill_price(contract)
    broker.sell_to_open(contract, 1)
    trades = broker.process_expirations(contract.expiry, lambda s: 60.0)
    assert [t.action for t in trades] == ["ASSIGNED", "SELL_SHARES"]
    assert broker.shares_held("TEST") == 0
    # premium + (53 - 50) * 100 of capital gain
    assert broker.realized_pnl == pytest.approx(fill * 100 + 300.0)


def test_put_assignment_buys_shares_at_strike(broker):
    contract = _put(strike=47.0)
    broker.sell_to_open(contract, 1)
    trades = broker.process_expirations(contract.expiry, lambda s: 40.0)
    assert [t.action for t in trades] == ["ASSIGNED", "BUY_SHARES"]
    assert broker.shares_held("TEST") == 100
    assert broker.equities["TEST"].average_cost == pytest.approx(47.0)


def test_valuation_and_collateral(broker):
    broker.buy_shares("TEST", 100, 50.0)
    contract = _call()
    broker.sell_to_open(contract, 1)
    prices = {"TEST": 52.0}
    marks = {contract.symbol: 1.50}
    assert broker.equity_value(prices) == pytest.approx(5_200.0)
    assert broker.option_value(marks) == pytest.approx(-150.0)
    assert broker.net_liquidation(prices, marks) == pytest.approx(
        broker.cash + 5_200.0 - 150.0
    )
    assert broker.collateral_used(prices) == pytest.approx(5_200.0)


def test_state_round_trip(tmp_path):
    b = PaperBroker(cash=50_000.0, account_id="PAPER-XYZ")
    b.buy_shares("TEST", 100, 50.0)
    b.sell_to_open(_call(), 1)
    path = b.save(tmp_path / "state.json")
    restored = PaperBroker.load(path)
    assert restored.account_id == "PAPER-XYZ"
    assert restored.cash == pytest.approx(b.cash)
    assert restored.shares_held("TEST") == 100
    assert list(restored.options) == list(b.options)
    assert len(restored.ledger) == len(b.ledger)


def test_load_missing_state_returns_fresh_account(tmp_path):
    b = PaperBroker.load(tmp_path / "nope.json", default_cash=25_000.0)
    assert b.cash == pytest.approx(25_000.0)
    assert not b.ledger
