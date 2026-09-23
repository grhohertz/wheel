# Wheel Trading Agent: Market-Driven Delta Targeting

## Summary
Built a complete adaptive delta targeting system for covered call rolls, integrated with live market data from Schwab. The agent recommends rolls based on IV regime, trend, earnings risk, and portfolio state — then executes via human-in-the-loop on Schwab UI.

## Architecture

### 1. Adaptive Delta Targeting
**File**: `wheel_trading_agent.py`

- **DeltaPolicy**: Configurable weights for decision drivers
  - Base: 0.30 delta (30 delta = ~70% probability OTM)
  - Floor: 0.15 (protect profitability)
  - Ceiling: 0.40 (stay profitable on IV crush)
  - Weights for: IV rank, trend, earnings, exit intent, portfolio excess

- **RollContext**: Market state container
  - spot price, cost basis
  - iv_rank, ma50, ma200 (trend)
  - earnings_in_window, want_exit flags
  - portfolio_delta_excess

- **compute_target_delta()**: Pure decision function
  - High IV (rich vol) → lower delta (sell further OTM, collect premium)
  - Low IV (cheap vol) → higher delta (sell closer to money, stay profitable)
  - Uptrend → reduce delta (don't cap the runner)
  - Earnings → reduce delta (gap buffer)
  - Exit intent ITM → increase delta (accelerate close)

### 2. Market Data Analytics
**File**: `src/wheel/market_analytics.py`

- **fetch_market_context()**
  - Fetches current price, IV from Schwab quote API
  - Fetches 1-year daily OHLCV history from Schwab pricehistory API
  - Computes 50/200-day moving averages
  - Computes annualized volatility from daily returns
  - Estimates IV rank (percentile within historical range)
  - Graceful degradation: falls back to None if any fetch fails

- **Helper functions**
  - compute_moving_average(): SMA over window
  - compute_price_volatility(): annualized σ from daily returns
  - compute_iv_rank(): IV percentile calculation
  - extract_closes(): parse OHLCV candles

### 3. Check Expiring Calls
**File**: `wheel_trading_agent.py` → `check_expiring_calls()`

- Scans for short calls expiring today/tomorrow
- Fetches live market context (MAs, IV rank, etc.)
- Builds RollContext with real market data
- Computes adaptive target delta
- Finds call strike closest to target delta
- Returns Trade with roll_metadata (old/new strikes, expiries, qty, target delta)

### 4. Roll Executor
**File**: `src/wheel/roll_executor.py`

- **build_roll_order()**: Creates spread order structure
  - BUY-to-close old call
  - SELL-to-open new call
  
- **execute_roll()**: Executes order
  - Dry-run mode: logs order for audit
  - Live mode: raises NotImplementedError (Schwab API limitation)
  - Roadmap: integrate with Alpaca or Interactive Brokers
  
- **fetch_optimal_strike()**: Find strike nearest target delta from option chain

### 5. Portfolio Reconciliation
**File**: `src/wheel/portfolio_reconciliation.py`

- **PendingRoll**: Roll lifecycle tracking
  - recommended_at → old_filled_at → new_filled_at
  - Stores order IDs, fill prices, timestamps
  - Computes net credit on completion
  - States: pending, partially_filled, filled, expired, cancelled

- **RollHistory**: Audit log
  - Serialize/deserialize to JSON (roll_history.json)
  - Methods: net_credit_ytd(), completed_rolls(), pending_rolls(), expired_rolls()
  - Supports querying by symbol and old expiry

- **PortfolioReconciler**: State management
  - record_roll_recommendation(): capture new roll
  - record_fill(): update as legs fill
  - check_roll_expiry(): mark expired
  - summary(): generate overview

## Decision Logic: Example Scenarios

### Scenario 1: High IV, Strong Uptrend
- Current IV rank: 85%
- spot > ma50 > ma200
- **Target delta**: 0.19 (vs base 0.30)
- **Action**: Sell 19 delta call (further OTM)
- **Why**: Rich vol allows collecting premium further away; uptrend shouldn't be capped

### Scenario 2: Low IV, Downtrend
- Current IV rank: 20%
- spot < ma50
- **Target delta**: 0.40 (vs base 0.30)
- **Action**: Sell 40 delta call (closer to money)
- **Why**: Cheap vol requires premium closer to money; downtrend supports assignment

### Scenario 3: Earnings Before Expiry
- earnings_in_window = True
- **Target delta**: 0.20 (vs base 0.30)
- **Action**: Sell 20 delta call (large gap buffer)
- **Why**: Avoid overnight gap risk; lower delta = higher margin of safety

### Scenario 4: Want to Exit, Profitable
- spot > cost_basis * 1.15
- want_exit = True
- **Target delta**: 0.35 (vs base 0.30)
- **Action**: Sell 35 delta call (closer to money)
- **Why**: Accept assignment to close position at profit

## Output Format

### Console
```
🎯 Wheel Trading Agent — Portfolio-Aware Trade Recommendation
...
📋 EXECUTION INSTRUCTIONS (Roll)
  1. Open Schwab web/mobile: GLD options
  2. Create a spread order:
     BUY-TO-CLOSE:  8 GLD $404.0 calls @ 2026-09-23
     SELL-TO-OPEN:  8 GLD $414.10 calls @ 2026-11-01
  3. Target net credit: match current bid-ask spreads
  4. Post in Schwab until filled or 5-10min window expires
  5. Confirm fill against portfolio before market close
```

### JSON
```json
{
  "valid": true,
  "recommendation": "ROLL_CALL",
  "underlying": "GLD",
  "action": "BUY 8 @ $404.0 / SELL 8 @ $414.10 (exp 2026-11-01)",
  "rationale": "Short calls expire in 1 day(s) — wheel is turning! Rolling to Δ=0.30 strike ($414.10) out to 2026-11-01 (~40 days). Drivers: IV_rank=50% uptrend -0.05",
  "risk_score": 6,
  "expected_return": 0.12,
  "conflicts": [],
  "roll_metadata": {
    "old_strike": 404.0,
    "old_expiry": "2026-09-23",
    "new_strike": 414.10,
    "new_expiry": "2026-11-01",
    "quantity": 8,
    "target_delta": 0.30
  }
}
```

## Graceful Degradation

- **Market data unavailable**: Falls back to base 0.30 delta (conservative)
- **Earnings API not ready**: Assumes no earnings (safe)
- **Schwab order placement**: Logs for manual execution (controlled)
- **Portfolio conflict**: Reports conflicts, doesn't recommend (safe)

## Files Changed
1. `wheel_trading_agent.py`: Add DeltaPolicy, RollContext, integration
2. `src/wheel/schwab.py`: Add pricehistory_raw() endpoint
3. `src/wheel/market_analytics.py`: Market data calculations (NEW)
4. `src/wheel/roll_executor.py`: Order building & execution (NEW)
5. `src/wheel/portfolio_reconciliation.py`: Roll tracking & audit (NEW)

## Next Steps (Roadmap)

### Phase 1: Order Monitoring
- Poll Schwab API for order status on pending rolls
- Update PortfolioReconciler on fill events
- Auto-move rolls to "filled" when both legs complete
- Track execution slippage vs. recommendation

### Phase 2: Automated Execution
- Integrate with Alpaca API (supports rest order placement)
- Or: Interactive Brokers (supports REST via IBKR API)
- Remove manual step; recommend roll → execute → track

### Phase 3: Earnings Calendar
- Wire earnings date API (e.g. Finnhub, Alpha Vantage)
- Auto-detect earnings before contract expiry
- Adjust delta targeting dynamically

### Phase 4: Portfolio Rebalancing
- Track portfolio-level delta budget
- Constrain rolls to respect portfolio delta ceiling
- Coordinate across multiple positions (multi-leg logic)

### Phase 5: Performance Analytics
- Dashboard: YTD net credit, win rate, avg hold time
- Slippage analysis: recommendation delta vs. actual fills
- Risk metrics: max loss, sharpe ratio, sortino

## Testing
All scenarios tested locally with mock data:
```bash
# Test market data functions
python3 -c "from wheel.market_analytics import *; ..."

# Test delta targeting logic
python3 -c "from wheel_trading_agent import compute_target_delta; ..."

# Test live agent (with Schwab creds)
python3 wheel_trading_agent.py --live
```

## Safety Guardrails
1. **Base delta = 0.30**: Conservative default if market data unavailable
2. **Clamps [0.15, 0.40]**: Ensures profitability and risk management
3. **Graceful degradation**: None < NoneType errors; uses fallbacks
4. **Manual execution**: No REST order API call until Alpaca/IB integrated
5. **Portfolio conflicts**: Refuses to recommend conflicting trades
6. **Audit trail**: All rolls persisted in roll_history.json for reconciliation
