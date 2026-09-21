# Wheel Trader — paper-trading wheel-strategy engine (Python)

A single-account, **paper-trading-only** implementation of the options wheel:
sell cash-secured puts → take assignment → sell covered calls against the shares → repeat.
Pure Python 3.10+, **zero runtime dependencies**, 64 unit tests.

> Live trading is not implemented anywhere in this package. `WHEEL_MODE` must be
> `paper`; anything else raises `LiveTradingDisabled` at start-up.

---

## Quick start

```bash
cd ~/projects/wheel
export PYTHONPATH=src                      # or: pip install -e .

python3 -m wheel.cli reset --cash 250000
python3 -m wheel.cli scan                  # recommendations, no fills
python3 -m wheel.cli run --execute         # settle + scan + simulated fills
python3 -m wheel.cli positions
python3 -m wheel.cli simulate --days 12 --step 7
```

Installed as a package (`pip install -e .`), the same commands are available as
`wheel-trader scan`, `wheel-trader run --execute`, …

### Commands

| Command | What it does |
|---|---|
| `scan [SYMBOLS]` | Recommendations only — never mutates state |
| `run [SYMBOLS] [--execute]` | Settle expiries/assignments, scan, optionally simulate fills and save |
| `positions` | Cash, equity, options, NAV, realized P&L, collateral used |
| `ledger [--limit N]` | Recent simulated fills |
| `price SYMBOL --strike --dte --iv --right` | Ad-hoc Black-Scholes price + Greeks |
| `reset [--cash N]` | Wipe the paper account |
| `simulate --days N --step D` | Advance the clock and run repeated cycles |

Global flags: `--state PATH` (account JSON), `--date YYYY-MM-DD` (as-of), `--json`
(machine-readable output for every command).

### Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `WHEEL_MODE` | `paper` | Anything else is refused |
| `WHEEL_ACCOUNT_ID` | `PAPER-0001` | Single account scope |
| `WHEEL_STARTING_CASH` | `100000` | Seed cash on reset |
| `WHEEL_STATE_PATH` | `state/paper_account.json` | Where the account is persisted |
| `WHEEL_WATCHLIST` | `AAPL,MSFT,KO,F,T` | Symbols scanned in addition to holdings |

---

## Layout

```
src/wheel/
  config.py       Settings + StrategyParams + the paper-mode guard
  models.py       OCC symbols, OptionContract/Position, EquityPosition, TradeRecord, Recommendation
  greeks.py       Black-Scholes price + delta/gamma/theta/vega/rho, implied vol (bisection)
  marketdata.py   MarketData protocol; SyntheticMarketData (deterministic) + StaticMarketData (tests)
  strategy.py     Candidate selection: ~0.30 delta, 30-45 DTE, liquidity + basis filters; manage/roll rules
  broker.py       PaperBroker: fills w/ slippage+commission, assignment, expiry, buying power, JSON state
  engine.py       WheelEngine: settle → scan → execute cycle, recommendation construction
  report.py       Human-readable and JSON rendering
  cli.py          argparse front-end
tests/            64 tests across greeks, strategy, broker, engine, marketdata, report/CLI
```

## Strategy rules (defaults, all in `StrategyParams`)

- Target **|delta| ≈ 0.30** (±0.15), **30–45 DTE**.
- Buy back at **50% of credit captured**; manage/roll at **≤ 7 DTE**.
- Liquidity filters: open interest ≥ 50, bid/ask spread ≤ 20% of mid.
- **Covered calls only** — a call can never be sold without uncommitted shares
  behind it (`InsufficientShares`), and never **below cost basis**.
- Cash-secured puts only — collateral = strike × 100 × contracts, and no single
  position may consume more than **50% of NAV**.
- Fills modelled with **2% slippage against you** + **$0.65/contract commission**.

### Buying power is real

Collateral is committed for the life of a short put: `available_cash = cash −
Σ(strike × 100 × contracts)`. The scanner spends that budget down *across*
symbols within one scan, so two tickers can't plan to secure puts with the same
dollars, and the broker rejects the fill if they try.

## Safety rails

1. `assert_paper_mode()` runs on every `Settings` construction.
2. No network calls, no order-routing code path exists.
3. Every rendered report is wrapped in `*** PAPER TRADING — ALL FILLS ARE SIMULATED ***`.
4. Every simulated fill is appended to an audit ledger persisted in the state file.

---

## Example

```
SYMBOL: KO
Current Price: $378.62 | IV: 42% | DTE: 32
Recommendation: SELL_CASH_SECURED_PUT 3x KO 2026-02-06 $360 Put @ $10.30
  Greeks (short): d=+0.316, G=-0.0076, Th=+0.256/day, V=-0.398
  Premium: $3,090.00 (collected, net of the multiplier)
  Buy-back target: $5.15 (50% of credit)
  Collateral: $108,000.00 (43.2% of NAV)
  Annualized yield: 32.6%
  Confidence: 92% | Risk: IV crush, early assignment, assignment leaves you long 100 shares per contract
```

12-week simulated run from $250k (synthetic prices — illustrative only):

```
2026-01-05  fills=0   NAV=$249,882.10
...
2026-03-23  fills=3   NAV=$262,553.90     realized P&L $12,587.70
```

## Tests

```bash
python3 -m pytest -q        # 64 passed
```

`conftest.py` puts `src/` on the path, so tests run without installing.

---

## Wiring a real data feed (Schwab)

`marketdata.py` defines a two-method protocol:

```python
class MarketData(Protocol):
    def get_quote(self, symbol: str, as_of: date) -> Quote: ...
    def get_chain(self, symbol: str, as_of: date) -> list[OptionContract]: ...
```

`SyntheticMarketData` (deterministic pseudo-random walk, seeded per symbol) is
the default. To use Schwab quotes/chains, implement that protocol against
`GET /marketdata/quotes` and `GET /marketdata/chains` and pass the instance to
`WheelEngine(market=...)`. **Nothing else changes** — the broker stays simulated,
which is exactly the point: real prices, fake fills.

`schwab_paper_trader.py` at the repo root is the earlier single-file scaffold,
superseded by `src/wheel/` and kept only as a reference for the API TODOs.
