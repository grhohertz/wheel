#!/usr/bin/env python3
"""
Integration test: Market-driven delta targeting full flow.

Simulates:
1. Portfolio with short calls expiring in 1 day
2. Market context fetch (IV rank, MAs, etc.)
3. Delta targeting decision
4. Roll recommendation with execution instructions
5. Fill recording and reconciliation
"""

import json
import sys
from pathlib import Path
from datetime import datetime, date, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wheel_trading_agent import (
    Portfolio, Position, DeltaPolicy, RollContext, 
    compute_target_delta, Trade
)
from wheel.market_analytics import (
    compute_moving_average,
    compute_iv_rank,
    compute_price_volatility,
)
from wheel.portfolio_reconciliation import (
    PortfolioReconciler, PendingRoll
)


def test_full_flow():
    """Run the full market-driven delta targeting flow."""
    
    print("=" * 70)
    print("🧪 INTEGRATION TEST: Market-Driven Delta Targeting")
    print("=" * 70)
    
    # Step 1: Create a portfolio with expiring short calls
    print("\n1️⃣  Portfolio State")
    print("-" * 70)
    
    portfolio = Portfolio(
        cash=50000.0,
        positions=[
            Position(
                symbol="GLD",
                type="equity",
                quantity=100,
                avg_price=400.0,
            ),
            Position(
                symbol="GLD",
                type="call",
                quantity=-1,  # short 1 call (100 shares per contract)
                avg_price=5.0,  # sold for $5
                strike=404.0,
                expiry=(date.today()).isoformat(),  # expires TODAY
            ),
        ]
    )
    
    print(f"Cash: ${portfolio.cash:,.2f}")
    print(f"Long GLD: 100 shares @ $400/share = ${100 * 400:,.2f}")
    print(f"Short GLD calls: 1 contract @ $404 strike, expiry: TODAY")
    print(f"  (Premium collected: $500 = $5 × 100 shares)")
    
    # Step 2: Simulate market context
    print("\n2️⃣  Market Context (Simulated)")
    print("-" * 70)
    
    # Simulate 1 year of price history (252 trading days)
    # Create an uptrend: start at 380, end at 429
    closes = []
    for i in range(252):
        # Uptrend with some noise
        base = 380 + (i * (429 - 380) / 252)
        noise = (i % 3) - 1  # -1, 0, or 1
        closes.append(base + noise)
    
    ma50 = compute_moving_average(closes, 50)
    ma200 = compute_moving_average(closes, 200)
    
    # Simulate IV percentiles (need at least 10)
    historic_ivs = [0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.35]
    current_iv = 0.28
    iv_rank = compute_iv_rank(current_iv, historic_ivs)
    
    print(f"Current price: ${closes[-1]}")
    print(f"50-day MA: ${ma50:.2f}")
    print(f"200-day MA: ${ma200:.2f}")
    print(f"Trend: {'UPTREND' if closes[-1] > ma50 > ma200 else 'DOWNTREND'}")
    print(f"Current IV: {current_iv:.1%}")
    print(f"IV Rank: {iv_rank:.0%} (percentile in historical range)")
    
    # Step 3: Build RollContext with market data
    print("\n3️⃣  Roll Context (Market-Driven Decision Input)")
    print("-" * 70)
    
    ctx = RollContext(
        spot=closes[-1],
        cost_basis=400.0,
        iv_rank=iv_rank,
        ma50=ma50,
        ma200=ma200,
        earnings_in_window=False,
        want_exit=False,
        portfolio_delta_excess=None,
    )
    
    print(f"Spot: ${ctx.spot}")
    print(f"Cost basis: ${ctx.cost_basis}")
    print(f"IV rank: {ctx.iv_rank:.0%}")
    print(f"MA50: ${ctx.ma50:.2f}")
    print(f"MA200: ${ctx.ma200:.2f}")
    print(f"Earnings in window: {ctx.earnings_in_window}")
    print(f"Want exit: {ctx.want_exit}")
    
    # Step 4: Compute adaptive target delta
    print("\n4️⃣  Delta Targeting Decision")
    print("-" * 70)
    
    policy = DeltaPolicy()
    target_delta, explain = compute_target_delta(ctx, policy)
    
    print(f"Base delta: {policy.base:.2f}")
    print(f"Adjustments:")
    for reason in explain["reasons"]:
        print(f"  • {reason}")
    print(f"Final target delta: {target_delta:.3f}")
    print(f"Recommendation: Sell call with delta ≈ {target_delta:.1%}")
    print(f"  (i.e., ~{100-target_delta*100:.0f}% chance of staying profitable)")
    
    # Step 5: Build roll recommendation
    print("\n5️⃣  Roll Recommendation")
    print("-" * 70)
    
    old_strike = 404.0
    new_strike = 410.0  # Approximation (actual would come from chain)
    old_expiry = date.today().isoformat()
    new_expiry = (date.today() + timedelta(days=40)).isoformat()
    
    trade = Trade(
        recommendation="ROLL_CALL",
        underlying="GLD",
        action=f"BUY 1 @ ${old_strike} / SELL 1 @ ${new_strike:.2f} (exp {new_expiry})",
        rationale=(
            f"Short calls expire today! Rolling to Δ={target_delta:.2f} "
            f"strike (${new_strike:.2f}) out to {new_expiry} (~40 days). "
            f"Drivers: {'; '.join(explain['reasons'])}"
        ),
        risk_score=6,
        expected_return=0.12,
        valid=True,
    )
    
    trade.roll_metadata = {
        "old_strike": old_strike,
        "old_expiry": old_expiry,
        "new_strike": new_strike,
        "new_expiry": new_expiry,
        "quantity": 1,
        "target_delta": target_delta,
    }
    
    print(f"Recommendation: {trade.recommendation}")
    print(f"Underlying: {trade.underlying}")
    print(f"Action: {trade.action}")
    print(f"Rationale: {trade.rationale}")
    
    # Step 6: Portfolio reconciliation setup
    print("\n6️⃣  Recording Roll in Audit Trail")
    print("-" * 70)
    
    reconciler = PortfolioReconciler(Path("state/paper_account.json"))
    
    roll = reconciler.record_roll_recommendation(
        symbol="GLD",
        underlying="GLD",
        old_strike=old_strike,
        old_expiry=old_expiry,
        new_strike=new_strike,
        new_expiry=new_expiry,
        quantity=1,
        target_delta=target_delta,
    )
    
    print(f"Roll recorded: {roll.symbol} {roll.old_strike} → {roll.new_strike}")
    print(f"Status: {roll.status}")
    print(f"Recommended at: {roll.recommended_at}")
    
    # Step 7: Simulate fills
    print("\n7️⃣  Simulating Fill Events")
    print("-" * 70)
    
    # BUY-TO-CLOSE old call at $4.50
    reconciler.record_fill(
        symbol="GLD",
        old_expiry=old_expiry,
        leg="old",
        price=4.50,
        timestamp=datetime.now().isoformat(),
    )
    print(f"✓ BUY-TO-CLOSE: 1 GLD $404 call @ $4.50")
    
    # SELL-TO-OPEN new call at $2.75
    reconciler.record_fill(
        symbol="GLD",
        old_expiry=old_expiry,
        leg="new",
        price=2.75,
        timestamp=datetime.now().isoformat(),
    )
    print(f"✓ SELL-TO-OPEN: 1 GLD $410 call @ $2.75")
    
    # Step 8: Check reconciliation
    print("\n8️⃣  Reconciliation Summary")
    print("-" * 70)
    
    summary = reconciler.summary()
    
    print(f"Total rolls recommended: {summary['total_rolls_recommended']}")
    print(f"Pending or partial: {summary['pending_or_partial']}")
    print(f"Completed: {summary['completed']}")
    print(f"Expired: {summary['expired']}")
    print(f"Net credit YTD: ${summary['net_credit_ytd']:.2f}")
    
    # Reload the roll to check status
    completed_rolls = reconciler.roll_history.completed_rolls()
    if completed_rolls:
        r = completed_rolls[0]
        print(f"\nCompleted roll details:")
        print(f"  Old cost: ${r.old_fill_price:.2f}")
        print(f"  New credit: ${r.new_fill_price:.2f}")
        print(f"  Net credit: ${r.net_credit:.2f} per share")
        print(f"  Total credit (1 contract): ${r.net_credit * 100:.2f}")
    
    # Step 9: Calculate PnL
    print("\n9️⃣  P&L Analysis")
    print("-" * 70)
    
    if completed_rolls:
        r = completed_rolls[0]
        initial_credit = 5.0  # Sold original call for $5
        net_credit = r.net_credit  # Received from roll
        total_credit = initial_credit + net_credit
        
        print(f"Original short call credit: ${initial_credit:.2f}")
        print(f"Roll net credit: ${net_credit:.2f}")
        print(f"Total credit collected: ${total_credit:.2f} per share")
        print(f"Total for 100 shares: ${total_credit * 100:.2f}")
        print(f"ROI on ${old_strike * 100:.0f} collateral: {(total_credit * 100) / (old_strike * 100):.2%}")
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_full_flow()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
