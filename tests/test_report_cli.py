import json

import pytest
from factories import AS_OF, make_chain, make_quote
from wheel.broker import PaperBroker
from wheel.cli import main
from wheel.config import StrategyParams
from wheel.engine import WheelEngine
from wheel.marketdata import StaticMarketData
from wheel.report import BANNER, render_portfolio, render_scan, render_trades


def _engine() -> tuple[WheelEngine, PaperBroker]:
    market = StaticMarketData({"TEST": make_quote()}, {"TEST": make_chain()})
    broker = PaperBroker(cash=100_000.0, params=StrategyParams())
    return WheelEngine(market, broker, StrategyParams()), broker


def test_render_scan_has_banner_and_block():
    engine, broker = _engine()
    broker.buy_shares("TEST", 100, 50.0)
    out = render_scan(engine.scan(["TEST"], AS_OF))
    assert BANNER in out
    assert "SYMBOL: TEST" in out
    assert "Buy-back target" in out
    assert "Confidence" in out


def test_render_portfolio_and_trades():
    engine, broker = _engine()
    broker.buy_shares("TEST", 100, 50.0)
    trades = engine.execute(engine.scan(["TEST"], AS_OF), AS_OF)
    text = render_portfolio(engine.portfolio(AS_OF))
    assert "Net liquidation" in text
    assert "TEST" in text
    assert "SELL_TO_OPEN" in render_trades(trades)
    assert render_trades([]) == "no fills\n"


def test_cli_price_json(capsys):
    assert main(["--json", "price", "AAPL", "--strike", "190", "--dte", "35"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"] == "AAPL"
    assert payload["strike"] == pytest.approx(190.0)
    assert 0.0 <= payload["delta"] <= 1.0
    assert payload["theta_per_day"] < 0


def test_cli_reset_scan_run_roundtrip(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WHEEL_MODE", "paper")
    monkeypatch.setenv("WHEEL_WATCHLIST", "F")
    state = str(tmp_path / "paper.json")

    assert main(["--state", state, "reset", "--cash", "500000"]) == 0
    assert "reset paper account" in capsys.readouterr().out

    assert main(["--state", state, "--json", "scan", "F", "--date", "2026-01-05"]) == 0
    recs = json.loads(capsys.readouterr().out)
    assert recs and recs[0]["symbol"] == "F"

    assert main(["--state", state, "run", "F", "--execute", "--date", "2026-01-05"]) == 0
    out = capsys.readouterr().out
    assert BANNER in out
    assert "Net liquidation" in out

    saved = json.loads(open(state, encoding="utf-8").read())
    assert saved["mode"] == "paper"
    assert saved["ledger"], "executing should have written fills to the ledger"

    assert main(["--state", state, "--json", "positions"]) == 0
    portfolio = json.loads(capsys.readouterr().out)
    assert portfolio["mode"] == "paper"

    assert main(["--state", state, "ledger", "--limit", "5"]) == 0
    assert "SELL_TO_OPEN" in capsys.readouterr().out
