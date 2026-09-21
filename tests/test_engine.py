from datetime import date, timedelta

import pytest
from factories import AS_OF, make_chain, make_quote
from wheel.broker import PaperBroker
from wheel.config import LiveTradingDisabled, Settings, StrategyParams, assert_paper_mode
from wheel.engine import WheelEngine
from wheel.marketdata import StaticMarketData, SyntheticMarketData
from wheel.models import Action


def build(cash: float = 100_000.0) -> tuple[WheelEngine, PaperBroker]:
    quote = make_quote()
    market = StaticMarketData({"TEST": quote}, {"TEST": make_chain()})
    broker = PaperBroker(cash=cash, params=StrategyParams())
    return WheelEngine(market, broker, StrategyParams()), broker


def test_no_shares_yields_cash_secured_put():
    engine, broker = build()
    recs = engine.scan(["TEST"], AS_OF)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.action is Action.SELL_CASH_SECURED_PUT
    assert rec.contract is not None and rec.contract.is_put()
    assert rec.contracts >= 1
    assert rec.credit > 0
    assert rec.collateral_pct <= 0.50 + 1e-9
    assert rec.buyback_target == pytest.approx(rec.contract.mid * 0.5, abs=0.01)
    assert 0.05 <= rec.confidence <= 0.95


def test_shares_yield_covered_call():
    engine, broker = build()
    broker.buy_shares("TEST", 100, 50.0)
    rec = next(r for r in engine.scan(["TEST"], AS_OF) if r.symbol == "TEST")
    assert rec.action is Action.SELL_COVERED_CALL
    assert rec.contracts == 1
    assert rec.contract is not None and rec.contract.strike >= 50.0


def test_execute_opens_the_position_and_credits_cash():
    engine, broker = build()
    broker.buy_shares("TEST", 100, 50.0)
    cash_before = broker.cash
    recs = engine.scan(["TEST"], AS_OF)
    trades = engine.execute(recs, AS_OF)
    assert len(trades) == 1
    assert trades[0].action == "SELL_TO_OPEN"
    assert broker.cash > cash_before
    assert len(broker.open_short_options("TEST")) == 1


def test_open_short_switches_to_management():
    engine, broker = build()
    broker.buy_shares("TEST", 100, 50.0)
    engine.execute(engine.scan(["TEST"], AS_OF), AS_OF)
    rec = next(r for r in engine.scan(["TEST"], AS_OF) if r.symbol == "TEST")
    assert rec.action in (Action.HOLD, Action.BUY_TO_CLOSE, Action.ROLL)
    assert rec.credit > 0
    # HOLD must not generate a fill.
    if rec.action is Action.HOLD:
        assert engine.execute([rec], AS_OF) == []


def test_roll_near_expiry_closes_the_short():
    engine, broker = build()
    broker.buy_shares("TEST", 100, 50.0)
    engine.execute(engine.scan(["TEST"], AS_OF), AS_OF)
    pos = broker.open_short_options("TEST")[0]
    near = pos.expiry - timedelta(days=3)
    recs = engine.scan(["TEST"], near)
    assert recs[0].action is Action.ROLL
    trades = engine.execute(recs, near)
    assert [t.action for t in trades] == ["BUY_TO_CLOSE"]
    assert not broker.open_short_options("TEST")


def test_settlement_expires_otm_short_worthless():
    engine, broker = build()
    broker.buy_shares("TEST", 100, 50.0)
    engine.execute(engine.scan(["TEST"], AS_OF), AS_OF)
    pos = broker.open_short_options("TEST")[0]
    credit = pos.credit_received()
    settled = engine.settle(pos.expiry)  # static spot 50 < strike -> OTM
    assert [t.action for t in settled] == ["EXPIRED_WORTHLESS"]
    assert broker.realized_pnl == pytest.approx(credit)


def test_run_cycle_reports_everything():
    engine, broker = build()
    result = engine.run_cycle(["TEST"], AS_OF, execute=True)
    assert result.as_of == AS_OF
    assert result.recommendations
    assert result.trades
    assert result.actionable()


def test_portfolio_snapshot_math():
    engine, broker = build()
    broker.buy_shares("TEST", 100, 50.0)
    engine.execute(engine.scan(["TEST"], AS_OF), AS_OF)
    p = engine.portfolio(AS_OF)
    assert p["mode"] == "paper"
    assert p["equity_value"] == pytest.approx(5_000.0)
    assert p["open_credit"] > 0
    assert p["net_liquidation"] == pytest.approx(
        p["cash"] + p["equity_value"] + p["option_value"]
    )
    assert len(p["positions"]["options"]) == 1
    assert p["positions"]["options"][0]["dte"] > 0


def test_collateral_cap_blocks_oversized_put():
    engine, broker = build(cash=4_000.0)  # < one contract of collateral
    rec = engine.scan(["TEST"], AS_OF)[0]
    assert rec.action is Action.SKIP
    assert "collateral" in rec.rationale or "cannot collateralise" in rec.rationale


def test_engine_refuses_live_mode():
    with pytest.raises(LiveTradingDisabled):
        Settings(mode="live")
    with pytest.raises(LiveTradingDisabled):
        assert_paper_mode("LIVE")
    assert_paper_mode("paper")  # no raise


def test_settings_from_env_parses_watchlist():
    s = Settings.from_env(
        {"WHEEL_MODE": "paper", "WHEEL_WATCHLIST": "aapl, ko ,f", "WHEEL_STARTING_CASH": "25000"}
    )
    assert s.watchlist == ("AAPL", "KO", "F")
    assert s.starting_cash == pytest.approx(25_000.0)


def test_end_to_end_against_synthetic_data():
    market = SyntheticMarketData()
    broker = PaperBroker(cash=250_000.0)
    engine = WheelEngine(market, broker, StrategyParams())
    result = engine.run_cycle(["KO", "F"], date(2026, 1, 5), execute=True)
    assert result.recommendations
    # Every actionable recommendation must be a premium-collecting short.
    for rec in result.actionable():
        assert rec.action in (Action.SELL_CASH_SECURED_PUT, Action.SELL_COVERED_CALL)
        assert rec.credit > 0
    assert broker.net_liquidation({"KO": market.get_quote("KO").price}) > 0


def test_scan_does_not_double_spend_collateral_across_symbols():
    """Two symbols must not both plan to secure puts with the same dollars."""

    quote = make_quote()
    market = StaticMarketData(
        {"AAA": quote, "BBB": quote}, {"AAA": make_chain(), "BBB": make_chain()}
    )
    params = StrategyParams()
    broker = PaperBroker(cash=100_000.0, params=params)
    engine = WheelEngine(market, broker, params)

    recs = [r for r in engine.scan(["AAA", "BBB"], AS_OF) if r.action is Action.SELL_CASH_SECURED_PUT]
    assert recs, "expected at least one cash-secured put"
    assert sum(r.collateral for r in recs) <= broker.available_cash() + 1e-6


def test_executing_a_full_scan_never_overdraws_the_account():
    quote = make_quote()
    market = StaticMarketData(
        {"AAA": quote, "BBB": quote}, {"AAA": make_chain(), "BBB": make_chain()}
    )
    params = StrategyParams()
    broker = PaperBroker(cash=100_000.0, params=params)
    engine = WheelEngine(market, broker, params)

    engine.run_cycle(["AAA", "BBB"], as_of=AS_OF, execute=True)
    assert broker.available_cash() >= 0.0
    assert broker.put_collateral_committed() <= broker.cash + 1e-6
