"""The live loop -- paper only.

:class:`LiveEngine` wires a feed to a signal to a risk gate to an execution
client:

    poll feed -> drain ticks -> signal -> risk gate -> submit -> collect fills

Every entry point is guarded by :func:`check_live_trading_disabled`, which
refuses any mode other than ``paper`` and any execution client that is not a
:class:`~wheel.execution.PaperExecutionClient`.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import PAPER, LiveTradingDisabled, assert_paper_mode
from .execution import (
    ExecutionClient,
    Fill,
    FillTracker,
    Order,
    OrderSide,
    OrderStatus,
    PaperExecutionClient,
)
from .feeds import FeedManager, MarketFeed, PaperFeed, Quote

__all__ = [
    "LiveEngine",
    "LiveMetrics",
    "band_signal",
    "check_live_trading_disabled",
    "synthetic_ticks",
]

SignalFn = Callable[[Quote, "LiveEngine"], "Order | None"]
RiskGate = Callable[[Order, Quote, "LiveEngine"], object]


def check_live_trading_disabled(
    mode: str | None = None,
    client: ExecutionClient | None = None,
) -> None:
    """Hard-fail anything that is not paper trading.

    ``mode`` defaults to ``$WHEEL_MODE``. When ``client`` is supplied it must be
    a :class:`~wheel.execution.PaperExecutionClient` -- a real venue adapter
    cannot be smuggled into the loop.
    """

    resolved = os.environ.get("WHEEL_MODE", PAPER) if mode is None else mode
    assert_paper_mode(resolved)
    if client is not None and not isinstance(client, PaperExecutionClient):
        raise LiveTradingDisabled(
            f"execution client {type(client).__name__} is refused: "
            "this build only trades through PaperExecutionClient."
        )


@dataclass
class LiveMetrics:
    """Counters for one loop run."""

    ticks: int = 0
    signals: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    fills: int = 0
    filled_qty: int = 0
    risk_rejections: int = 0
    dropped_ticks: int = 0
    loops: int = 0
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "signals": self.signals,
            "orders_submitted": self.orders_submitted,
            "orders_rejected": self.orders_rejected,
            "fills": self.fills,
            "filled_qty": self.filled_qty,
            "risk_rejections": self.risk_rejections,
            "dropped_ticks": self.dropped_ticks,
            "loops": self.loops,
            "elapsed": round(self.elapsed, 4),
        }


def band_signal(
    symbol: str,
    anchor: float,
    *,
    band: float = 0.01,
    qty: int = 100,
    max_position: int = 800,
    min_position: int = 0,
) -> SignalFn:
    """A deliberately simple mean-reversion demo signal.

    Buy ``qty`` when the last trade is ``band`` below ``anchor``, sell when it is
    ``band`` above, clamped to ``[min_position, max_position]``. It exists to
    exercise the loop end to end -- it is not a trading recommendation.
    """

    sym = symbol.upper()
    lo = anchor * (1.0 - band)
    hi = anchor * (1.0 + band)

    def _signal(quote: Quote, engine: "LiveEngine") -> Order | None:
        if quote.symbol.upper() != sym:
            return None
        held = engine.position(sym)
        if quote.last <= lo and held + qty <= max_position:
            return Order.market(sym, qty, OrderSide.BUY)
        if quote.last >= hi and held - qty >= min_position:
            return Order.market(sym, qty, OrderSide.SELL)
        return None

    return _signal


def synthetic_ticks(
    symbol: str,
    start_price: float,
    count: int,
    *,
    seed: int = 20240101,
    vol_per_tick: float = 0.0015,
    spread_pct: float = 0.0008,
    interval: float = 1.0,
    start_ts: float = 0.0,
    volume: int = 100,
) -> list[Quote]:
    """A deterministic random-walk tick sequence (same seed -> same path)."""

    rng = random.Random(seed)
    price = float(start_price)
    out: list[Quote] = []
    for i in range(max(0, count)):
        price = max(0.05, price * (1.0 + rng.gauss(0.0, vol_per_tick)))
        out.append(
            Quote.from_last(
                symbol,
                round(price, 4),
                spread_pct=spread_pct,
                volume=volume,
                timestamp=start_ts + i * interval,
            )
        )
    return out


class LiveEngine:
    """Feed -> signal -> risk -> execution, in one small event loop."""

    def __init__(
        self,
        feed: MarketFeed | FeedManager,
        execution: ExecutionClient,
        signal_fn: SignalFn | None = None,
        *,
        risk_gate: RiskGate | None = None,
        mode: str = PAPER,
        tracker: FillTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        check_live_trading_disabled(mode, execution)
        self.feeds = feed if isinstance(feed, FeedManager) else FeedManager(feed)
        self.execution = execution
        self.signal_fn = signal_fn
        self.risk_gate = risk_gate
        self.mode = mode
        self.tracker = tracker if tracker is not None else FillTracker()
        self.metrics = LiveMetrics()
        self.orders: list[Order] = []
        self._positions: dict[str, int] = {}
        self._clock = clock
        self._sleep = sleep

    # -- state -----------------------------------------------------------
    def position(self, symbol: str) -> int:
        return self._positions.get(symbol.upper(), 0)

    @property
    def positions(self) -> dict[str, int]:
        return {k: v for k, v in self._positions.items() if v}

    def seed_position(self, symbol: str, qty: int) -> None:
        self._positions[symbol.upper()] = int(qty)

    def subscribe(self, *symbols: str) -> None:
        for symbol in symbols:
            self.feeds.subscribe(symbol)

    # -- plumbing ---------------------------------------------------------
    def _record(self, fills: Sequence[Fill]) -> None:
        for fill in fills:
            self.tracker.add(fill)
            self._positions[fill.symbol] = self.position(fill.symbol) + fill.signed_qty
            self.metrics.fills += 1
            self.metrics.filled_qty += fill.qty

    def _allowed(self, order: Order, quote: Quote) -> bool:
        if self.risk_gate is None:
            return True
        verdict = self.risk_gate(order, quote, self)
        ok = bool(getattr(verdict, "ok", verdict))
        if not ok:
            self.metrics.risk_rejections += 1
        return ok

    def handle_quote(self, quote: Quote) -> list[Fill]:
        """Process one tick: match resting orders, then evaluate the signal."""

        self.metrics.ticks += 1
        fills: list[Fill] = []
        if isinstance(self.execution, PaperExecutionClient):
            resting = self.execution.on_quote(quote)
            fills.extend(resting)
            self._record(resting)

        if self.signal_fn is None:
            return fills

        order = self.signal_fn(quote, self)
        if order is None:
            return fills
        self.metrics.signals += 1
        if not self._allowed(order, quote):
            order.status = OrderStatus.REJECTED
            order.reason = "risk gate rejected"
            self.orders.append(order)
            return fills

        before = len(self.execution.get_fills())
        submitted = self.execution.submit_order(order)
        self.orders.append(submitted)
        self.metrics.orders_submitted += 1
        if submitted.status is OrderStatus.REJECTED:
            self.metrics.orders_rejected += 1
        new_fills = self.execution.get_fills()[before:]
        fills.extend(new_fills)
        self._record(new_fills)
        return fills

    def step(self) -> list[Quote]:
        """One pass: poll every feed, drain the buffer, handle each tick."""

        self.metrics.loops += 1
        self.feeds.poll()
        ticks = self.feeds.drain()
        for quote in ticks:
            self.handle_quote(quote)
        self.metrics.dropped_ticks = self.feeds.dropped
        return ticks

    def run_loop(
        self,
        *,
        max_ticks: int | None = None,
        duration: float | None = None,
        interval: float = 0.0,
        stop_when_exhausted: bool = True,
    ) -> LiveMetrics:
        """Run until ticks run out, ``max_ticks`` is hit, or ``duration`` elapses."""

        check_live_trading_disabled(self.mode, self.execution)
        started = self._clock()
        idle = 0
        while True:
            if max_ticks is not None and self.metrics.ticks >= max_ticks:
                break
            if duration is not None and (self._clock() - started) >= duration:
                break
            ticks = self.step()
            if ticks:
                idle = 0
            else:
                idle += 1
                if stop_when_exhausted and (self.feeds.exhausted or idle >= 3):
                    break
            if interval > 0:
                self._sleep(interval)
        self.metrics.elapsed = self._clock() - started
        return self.metrics

    # -- reporting ---------------------------------------------------------
    def report(self) -> dict:
        return {
            "mode": self.mode,
            "metrics": self.metrics.to_dict(),
            "positions": self.positions,
            "fills": self.tracker.summary(),
            "open_orders": [
                o.to_dict() for o in self.orders if o.is_open
            ],
        }


def build_paper_engine(
    symbol: str,
    *,
    start_price: float,
    ticks: int,
    seed: int = 20240101,
    band: float = 0.004,
    qty: int = 100,
    max_position: int = 800,
    shares: int = 0,
    slippage_pct: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LiveEngine:
    """Assemble a fully paper-wired engine (used by ``wheel.cli live``)."""

    feed = PaperFeed({symbol.upper(): synthetic_ticks(symbol, start_price, ticks, seed=seed)})
    client = PaperExecutionClient(slippage_pct=slippage_pct)
    engine = LiveEngine(
        FeedManager(feed),
        client,
        band_signal(symbol, start_price, band=band, qty=qty, max_position=max_position),
        clock=clock,
        sleep=sleep,
    )
    engine.subscribe(symbol)
    if shares:
        engine.seed_position(symbol, shares)
    return engine
