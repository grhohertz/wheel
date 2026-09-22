import pytest
from wheel.config import LiveTradingDisabled
from wheel.execution import (
    ExecutionClient,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    PaperExecutionClient,
)
from wheel.feeds import FeedManager, PaperFeed, Quote
from wheel.live import (
    LiveEngine,
    LiveMetrics,
    band_signal,
    build_paper_engine,
    check_live_trading_disabled,
    synthetic_ticks,
)


class FakeClock:
    """Monotonic clock advanced explicitly by the loop's sleep()."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class NotAPaperClient(ExecutionClient):
    def submit_order(self, order):  # pragma: no cover - never reached
        raise AssertionError("live client must never be constructed into the loop")

    def cancel_order(self, order_id):  # pragma: no cover
        raise AssertionError

    def get_order_status(self, order_id):  # pragma: no cover
        raise AssertionError

    def get_fills(self, order_id=None):  # pragma: no cover
        raise AssertionError


def engine_for(prices, *, signal=None, risk_gate=None, shares=0, **kw) -> LiveEngine:
    feed = PaperFeed.from_prices("GLD", prices, spread_pct=0.001)
    client = PaperExecutionClient(clock=lambda: 0.0)
    eng = LiveEngine(FeedManager(feed), client, signal, risk_gate=risk_gate, **kw)
    eng.subscribe("GLD")
    if shares:
        eng.seed_position("GLD", shares)
    return eng


# -- the paper guard --------------------------------------------------------


def test_check_live_trading_disabled_rejects_live_mode():
    check_live_trading_disabled("paper")
    with pytest.raises(LiveTradingDisabled):
        check_live_trading_disabled("live")


def test_check_live_trading_disabled_reads_env(monkeypatch):
    monkeypatch.setenv("WHEEL_MODE", "live")
    with pytest.raises(LiveTradingDisabled):
        check_live_trading_disabled()
    monkeypatch.setenv("WHEEL_MODE", "paper")
    check_live_trading_disabled()


def test_check_live_trading_disabled_rejects_non_paper_client():
    with pytest.raises(LiveTradingDisabled):
        check_live_trading_disabled("paper", NotAPaperClient())


def test_live_engine_refuses_live_mode_at_construction():
    feed = PaperFeed.from_prices("GLD", [100.0])
    with pytest.raises(LiveTradingDisabled):
        LiveEngine(feed, PaperExecutionClient(), mode="live")
    with pytest.raises(LiveTradingDisabled):
        LiveEngine(feed, NotAPaperClient())


# -- end to end -------------------------------------------------------------


def test_signal_to_order_to_fill_end_to_end():
    # anchor 100, band 1% -> the 98.0 tick triggers a buy that fills at last
    eng = engine_for(
        [100.0, 99.9, 98.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100, max_position=800),
    )
    metrics = eng.run_loop()
    assert metrics.signals == 1
    assert metrics.orders_submitted == 1
    assert metrics.fills == 1
    assert metrics.filled_qty == 100
    assert eng.position("GLD") == 100
    assert eng.tracker.avg_price("GLD") == pytest.approx(98.0)
    assert eng.orders[0].status is OrderStatus.FILLED


def test_loop_consumes_every_tick_then_stops():
    eng = engine_for([100.0, 101.0, 102.0])
    metrics = eng.run_loop()
    assert metrics.ticks == 3
    assert eng.feeds.exhausted is True
    # re-running an exhausted feed is a no-op, not a hang
    assert eng.run_loop().ticks == 3


def test_max_ticks_caps_the_loop():
    eng = engine_for([100.0] * 10)
    metrics = eng.run_loop(max_ticks=4)
    assert metrics.ticks == 4


def test_duration_stops_the_loop():
    clock = FakeClock()
    eng = engine_for([100.0] * 100, clock=clock, sleep=clock.sleep)
    metrics = eng.run_loop(duration=5.0, interval=1.0)
    assert metrics.ticks == 5
    assert metrics.elapsed == pytest.approx(5.0)


def test_risk_gate_rejects_order_and_counts_breach():
    calls = []

    def gate(order, quote, engine):
        calls.append(order.symbol)
        return False

    eng = engine_for(
        [100.0, 98.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100),
        risk_gate=gate,
    )
    metrics = eng.run_loop()
    assert calls == ["GLD"]
    assert metrics.signals == 1
    assert metrics.risk_rejections == 1
    assert metrics.orders_submitted == 0
    assert metrics.fills == 0
    assert eng.position("GLD") == 0
    assert eng.orders[0].status is OrderStatus.REJECTED


def test_risk_gate_accepts_risk_report_like_objects():
    class Report:
        ok = True

    eng = engine_for(
        [100.0, 98.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100),
        risk_gate=lambda o, q, e: Report(),
    )
    metrics = eng.run_loop()
    assert metrics.orders_submitted == 1 and metrics.risk_rejections == 0


def test_resting_limit_order_fills_on_a_later_tick():
    eng = engine_for([100.0, 99.0, 95.0])
    eng.execution.on_quote(Quote.from_last("GLD", 100.0))
    order = eng.execution.submit_order(Order.limit("GLD", 10, OrderSide.BUY, 96.0))
    assert order.status is OrderStatus.NEW
    eng.run_loop()
    assert order.status is OrderStatus.FILLED
    assert eng.position("GLD") == 10
    assert eng.metrics.fills == 1


def test_band_signal_respects_position_limits():
    # already at max_position -> no buy; min_position floor -> no sell
    eng = engine_for(
        [98.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100, max_position=100),
        shares=100,
    )
    assert eng.run_loop().signals == 0

    eng2 = engine_for(
        [102.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100, min_position=0),
    )
    assert eng2.run_loop().signals == 0


def test_band_signal_ignores_other_symbols():
    sig = band_signal("GLD", 100.0, band=0.01)
    eng = engine_for([100.0])
    assert sig(Quote.from_last("SLV", 50.0), eng) is None


def test_sell_signal_reduces_position():
    eng = engine_for(
        [102.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100, min_position=0),
        shares=800,
    )
    eng.run_loop()
    assert eng.position("GLD") == 700
    assert eng.tracker.net_qty("GLD") == -100


def test_backpressure_counter_surfaces_in_metrics():
    feed = PaperFeed.from_prices("GLD", [100.0] * 10)
    mgr = FeedManager(feed, max_buffer=1)
    eng = LiveEngine(mgr, PaperExecutionClient())
    eng.subscribe("GLD")
    # pump ticks in without draining to force eviction (max_buffer=1)
    mgr.poll()
    mgr.poll()
    mgr.poll()
    assert mgr.dropped == 2
    eng.run_loop()
    # the loop's own first poll evicts once more before draining
    assert eng.metrics.dropped_ticks == mgr.dropped == 3


def test_report_and_metrics_shape():
    eng = engine_for(
        [100.0, 98.0],
        signal=band_signal("GLD", 100.0, band=0.01, qty=100),
    )
    eng.run_loop()
    report = eng.report()
    assert report["mode"] == "paper"
    assert report["positions"] == {"GLD": 100}
    assert set(report["metrics"]) >= {"ticks", "orders_submitted", "fills", "risk_rejections"}
    assert report["fills"]["net_qty"] == 100
    assert LiveMetrics().to_dict()["ticks"] == 0


# -- helpers ----------------------------------------------------------------


def test_synthetic_ticks_are_deterministic():
    a = synthetic_ticks("GLD", 250.0, 20, seed=7)
    b = synthetic_ticks("GLD", 250.0, 20, seed=7)
    c = synthetic_ticks("GLD", 250.0, 20, seed=8)
    assert [x.last for x in a] == [x.last for x in b]
    assert [x.last for x in a] != [x.last for x in c]
    assert len(a) == 20 and all(x.bid <= x.last <= x.ask for x in a)


def test_build_paper_engine_runs_a_demo():
    clock = FakeClock()
    eng = build_paper_engine(
        "GLD",
        start_price=250.0,
        ticks=60,
        seed=42,
        band=0.001,
        qty=100,
        max_position=800,
        shares=800,
        clock=clock,
        sleep=clock.sleep,
    )
    metrics = eng.run_loop()
    assert metrics.ticks == 60
    assert metrics.orders_submitted >= 1
    assert metrics.fills == metrics.orders_submitted
    assert 0 <= eng.position("GLD") <= 800
    assert isinstance(eng.tracker.fills[0], Fill)
