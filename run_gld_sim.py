#!/usr/bin/env python3
"""
Monte Carlo simulation for 800 shares of GLD wheel strategy.
"""
import sys
import json
from pathlib import Path
from datetime import date, timedelta
from dataclasses import dataclass

sys.path.insert(0, "src")

from wheel.broker import PaperBroker
from wheel.config import Settings, StrategyParams
from wheel.engine import WheelEngine
from wheel.marketdata import SyntheticMarketData
from wheel.models import EquityPosition

# Setup
settings = Settings.from_env()
state_file = Path("state/gld_sim.json")
state_file.parent.mkdir(parents=True, exist_ok=True)

# Create a fresh broker instance
broker = PaperBroker(
    account_id="paper_gld",
    cash=90000.0,  # $250k starting - $160k for 800 GLD @ $200
)

# Manually add GLD position
gld_position = EquityPosition(
    symbol="GLD",
    quantity=800,
    average_cost=200.0,
)
broker.equities["GLD"] = gld_position

# Setup engine with relaxed delta filter for synthetic data
# Use wide delta tolerance to catch any viable call from the synthetic market
params = StrategyParams(target_delta=0.30, delta_tolerance=0.25)
market = SyntheticMarketData(rate=0.05)
engine = WheelEngine(market, broker, params, settings)

# Simulate 13 weeks (90 days), one cycle per week
print("=" * 80)
print("GLD Wheel Strategy Simulation — 800 shares, 13 weeks")
print("=" * 80)
print()

start_date = date.today()
start_nav = engine.portfolio(start_date)["net_liquidation"]

print(f"Starting: {start_date.isoformat()}")
print(f"GLD position: 800 shares @ $200/share (cost basis)")
print(f"Cash: ${broker.cash:,.2f}")
print(f"Starting NAV: ${start_nav:,.2f}")
print()
print(f"{'Date':<12} | {'Fills':<6} | {'NAV':<15} | {'Delta NAV':<15} | {'Pos':<8}")
print("-" * 75)

prev_nav = start_nav
for i in range(13):
    day = start_date + timedelta(days=i * 7)
    
    # Debug: check what scan recommends
    recs = engine.scan(["GLD"], day)
    if i == 0:  # First week only
        print(f"\n[DEBUG] {day.isoformat()}: {len(recs)} recommendations")
        for rec in recs:
            print(f"  {rec.symbol}: {rec.action} @ ${rec.spot:.2f} — {rec.rationale[:70]}")
        # Also print the chain to see what deltas are available
        chain = engine.market.get_chain("GLD", day)
        print(f"  Chain has {len(chain)} contracts")
        if chain:
            print(f"    Sample: {chain[0]}")
    
    result = engine.run_cycle(["GLD"], day, execute=True)
    acted = len(result.trades) + len(result.settlements)
    nav = engine.portfolio(day)["net_liquidation"]
    delta = nav - prev_nav
    
    # Get GLD quantity
    gld_qty = engine.broker.equities.get("GLD", EquityPosition("GLD")).quantity
    
    print(f"{day.isoformat()} | {acted:<6} | ${nav:>13,.2f} | ${delta:>13,.2f} | {gld_qty:<8}")
    prev_nav = nav

print()
final_portfolio = engine.portfolio(start_date + timedelta(days=12 * 7))
print("=" * 80)
print("Final Portfolio")
print("=" * 80)
print(json.dumps(final_portfolio, indent=2))

# Save state for reference
broker.save(state_file)
print()
print(f"✅ State saved: {state_file}")
print(f"✅ Simulation complete: {start_nav:,.2f} → {final_portfolio['net_liquidation']:,.2f}")
print(f"✅ P&L: ${final_portfolio['net_liquidation'] - start_nav:+,.2f}")
