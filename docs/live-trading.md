# Live trading (paper only)

Phase 5b wires three new modules together so the strategy can be driven by a
tick stream instead of an end-of-day scan:

```
PaperFeed ──▶ FeedManager ──▶ LiveEngine ──▶ risk gate ──▶ PaperExecutionClient
  ticks        bounded buffer     signal        accept/reject      fills
                                     │                                │
                                     └──────────── FillTracker ◀──────┘
```

| Module | Class | Role |
|---|---|---|
| `wheel/feeds.py` | `Quote` | one tick: `symbol, bid, ask, last, volume, timestamp` (+ `mid`, `spread`, `spread_pct`) |
| | `MarketFeed` | abstract base: `subscribe` / `unsubscribe` / `get_quote` / `poll` |
| | `PaperFeed` | replays a cached tick sequence; `from_prices()` builds one from a price path |
| | `FeedManager` | multiplexes feeds into one **bounded** buffer with explicit backpressure |
| `wheel/execution.py` | `Order` | `symbol, qty, side, order_type, price, status, filled_qty, avg_price` |
| | `ExecutionClient` | abstract base: `submit_order` / `cancel_order` / `get_order_status` / `get_fills` |
| | `PaperExecutionClient` | deterministic simulated matching |
| | `FillTracker` | VWAP, net position, realised slippage |
| `wheel/live.py` | `LiveEngine` | the loop: poll → signal → risk → submit → collect fills |
| | `LiveMetrics` | ticks, signals, orders, fills, risk rejections, dropped ticks |
| | `check_live_trading_disabled` | the guard (see below) |

> **Naming:** `wheel.feeds.Quote` is a *tick*. `wheel.models.Quote` is the
> end-of-day underlying snapshot the option pricer uses. The package exports
> them as `TickQuote` and `Quote` respectively.

## Paper vs live toggle

There is no live toggle. `check_live_trading_disabled()` is called in
`LiveEngine.__init__` **and** again at the top of `run_loop()`, and it fails
closed on two axes:

1. **Mode** — delegates to `wheel.config.assert_paper_mode`, so anything other
   than `WHEEL_MODE=paper` raises `LiveTradingDisabled`.
2. **Client type** — the execution client must be a `PaperExecutionClient`.
   Handing `LiveEngine` any other `ExecutionClient` subclass raises
   `LiveTradingDisabled` before a single tick is polled.

```python
from wheel.live import LiveEngine, check_live_trading_disabled

check_live_trading_disabled("live")          # -> LiveTradingDisabled
LiveEngine(feed, SomeBrokerClient())         # -> LiveTradingDisabled
```

Wiring a real venue therefore means *deliberately* editing that guard — it
cannot happen by misconfiguration.

## Fill model

| Order type | Fills when | Price |
|---|---|---|
| `MARKET` | on the next tick (immediately if a quote is already cached) | `last × (1 ± slippage_pct)`, signed against us |
| `LIMIT` buy | `ask <= limit` | `min(limit, ask)` |
| `LIMIT` sell | `bid >= limit` | `max(limit, bid)` |

A market order submitted before any tick has been seen is **rejected**
(`status=REJECTED`, `reason="no quote available for market order"`) rather than
filled at a guessed price. `max_fill_qty` caps a single execution, which is how
partial fills (`PARTIALLY_FILLED`) are produced; `Order.apply_fill` keeps the
running VWAP in `avg_price`.

`FillTracker.realized_slippage(symbol, expected, side)` uses a
*positive-is-worse* convention: paying above expectation on a buy and selling
below expectation both come back positive.

## Backpressure

`FeedManager` never grows without bound. When the buffer is full:

* `drop_policy="oldest"` (default) evicts the head — you keep the freshest ticks;
* `drop_policy="newest"` discards the arriving tick — you keep the earliest.

Either way `FeedManager.dropped` increments and surfaces as
`LiveMetrics.dropped_ticks`, so a loop that cannot keep up is observable rather
than silently lossy.

## CLI demo

```bash
python -m wheel.cli live --symbol GLD --shares 800 --duration 60
python -m wheel.cli live --symbol GLD --shares 800 --duration 60 --json
python -m wheel.cli live --symbol GLD --duration 10 --realtime   # actually sleeps
```

`--duration N` means **N simulated seconds** at `--interval` seconds per tick
(default 1.0), so the 60-second demo replays 60 ticks and returns immediately.
Pass `--realtime` to sleep between ticks and watch it in wall-clock time. The
tick path is a seeded random walk (`--seed`), so a given invocation is exactly
reproducible.

The demo signal (`band_signal`) buys a clip when price drops `--band` below the
anchor and sells one when it rises the same distance above, clamped to
`--max-position`. **It is a loop exerciser, not a trading recommendation** — the
wheel's real entry logic lives in `wheel.strategy`.

## Known limitations

* **No real connectivity.** No WebSocket, no REST, no broker SDK. `PaperFeed`
  and `PaperExecutionClient` are the only implementations that ship.
* **No queue position or market impact.** A crossing limit order fills in full
  (up to `max_fill_qty`) with no consideration of displayed size.
* **Synchronous polling.** `run_loop` is a plain `while` loop, not asyncio — one
  thread, no reconnect/heartbeat logic, no partial-day gap handling.
* **Equity clips only.** `LiveEngine` tracks share positions; option orders are
  representable as `Order`s but the loop does not yet build option signals from
  `wheel.strategy`. That is the natural next phase.
* **Risk gate is a callable, not yet auto-wired to `RiskAggregator`.** Pass one
  in — e.g. a closure that builds a `RiskReport` and returns it; `LiveEngine`
  honours anything exposing `.ok` or a plain bool.
* **No persistence.** Orders, fills and positions live in memory for the run;
  nothing is written to the paper-account JSON yet.
