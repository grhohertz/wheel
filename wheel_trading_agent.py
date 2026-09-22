#!/usr/bin/env python3
"""
Wheel Trading Agent — Analyzes paper trading positions and recommends the next trade.

Fetches portfolio state, validates scanner recommendations against conflicts,
and recommends only non-conflicting trades.

Usage:
    python wheel_trading_agent.py [--state STATE_FILE]
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import os
from collections import defaultdict
from datetime import datetime

# Load .env early (before any Settings construction)
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Position:
    """Represents a portfolio position (equity or option)."""
    symbol: str
    type: str  # "equity", "call", "put"
    quantity: int
    avg_price: float
    strike: Optional[float] = None
    expiry: Optional[str] = None
    
    def is_long(self) -> bool:
        return self.quantity > 0
    
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class Portfolio:
    """Represents the full portfolio state."""
    cash: float
    positions: list[Position] = field(default_factory=list)
    
    def get_underlying_positions(self, symbol: str) -> dict:
        """Get all positions for an underlying: shares, calls, puts."""
        result = {"equity": None, "calls": [], "puts": []}
        for pos in self.positions:
            if pos.symbol == symbol:
                if pos.type == "equity":
                    result["equity"] = pos
                elif pos.type == "call":
                    result["calls"].append(pos)
                elif pos.type == "put":
                    result["puts"].append(pos)
        return result
    
    def has_conflicts(self, recommendation: dict) -> list[str]:
        """Check if a recommendation conflicts with existing positions."""
        conflicts = []
        symbol = recommendation.get("symbol")
        action = recommendation.get("action")
        
        positions = self.get_underlying_positions(symbol)
        equity = positions["equity"]
        calls = positions["calls"]
        puts = positions["puts"]
        
        if action == "SELL_CASH_SECURED_PUT":
            # Check if we already have a short call on this underlying
            if any(c.is_short() for c in calls):
                conflicts.append(
                    f"Already have short call on {symbol}; selling put creates conflicting wheel-on-wheel"
                )
            # Check if we have long equity + existing puts
            if equity and equity.is_long() and any(p.is_short() for p in puts):
                conflicts.append(
                    f"Already have long {symbol} + short put; can't stack another put"
                )
        
        elif action == "SELL_CALL":
            # Check if we don't have long equity
            if not equity or not equity.is_long():
                conflicts.append(
                    f"Can't sell covered call on {symbol} without long equity position"
                )
            # Check if we already have a short call
            if any(c.is_short() for c in calls):
                conflicts.append(
                    f"Already have short call on {symbol}; use ROLL_CALL instead"
                )
        
        elif action == "BUY_SHARES":
            # Generally safe but warn if we have puts outstanding
            if any(p.is_short() for p in puts):
                conflicts.append(
                    f"Buying shares while short puts on {symbol} could trigger double assignment"
                )
        
        return conflicts


@dataclass
class Trade:
    """Represents a trade recommendation."""
    recommendation: str  # "sell_call", "close_call", "sell_put", "buy_shares", "hold"
    underlying: str
    action: str
    rationale: str
    risk_score: int  # 1–10
    expected_return: float  # percentage
    conflicts: list[str] = field(default_factory=list)
    valid: bool = True


def run_cli_command(cmd: list) -> dict:
    """Run a wheel CLI command and parse JSON output."""
    try:
        # Use the project's python executable from venv
        venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
        
        env = os.environ.copy()
        env["PATH"] = f"{venv_python.parent}:{env.get('PATH', '')}"
        
        result = subprocess.run(
            [str(venv_python), "-m", "wheel.cli"] + cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode != 0:
            print(f"Error running {' '.join(cmd)}:")
            print(result.stderr)
            return {}
        return json.loads(result.stdout) if result.stdout else {}
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd)}")
        return {}
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from {' '.join(cmd)}")
        print(f"Output: {result.stdout}")
        return {}


def parse_option_symbol(symbol: str) -> tuple[str, str, float, str]:
    """
    Parse OCC option symbol format: UNDERLYING YYMMDD[C/P]STRIKE
    
    E.g. 'GLD   261106C00210000' -> ('GLD', '2026-11-06', 210.0, 'C')
    
    Returns: (underlying, expiry_iso, strike, right)
    """
    symbol = symbol.strip()
    # Format is 6 chars underlying, 6 digit date, C/P, 8 digit strike
    if len(symbol) < 15:
        return None, None, None, None
    
    underlying = symbol[:6].strip()
    date_str = symbol[6:12]  # YYMMDD
    right = symbol[12]  # C or P
    strike_str = symbol[13:21]  # 8 digits
    
    try:
        yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
        # Assume 20yy if yy >= 50 (or 21yy if < 50; adjust as needed)
        century = 2000 if yy < 50 else 2000
        year = century + yy
        expiry = f"{year:04d}-{mm:02d}-{dd:02d}"
        
        strike = float(strike_str) / 1000.0
        return underlying, expiry, strike, right
    except (ValueError, IndexError):
        return None, None, None, None


def get_portfolio_state(state_file: Optional[str] = None) -> Portfolio:
    """
    Fetch and parse portfolio state from state JSON file.
    
    Walks the ledger to compute current positions.
    """
    if not state_file:
        state_file = Path(__file__).parent / "state" / "paper_account.json"
    
    state_file = Path(state_file)
    if not state_file.exists():
        print(f"⚠️  State file not found: {state_file}")
        return Portfolio(cash=0.0, positions=[])
    
    try:
        with open(state_file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse state file: {state_file}")
        return Portfolio(cash=0.0, positions=[])
    
    cash = data.get("cash", 0.0)
    
    # Walk the ledger to build position map
    position_map = defaultdict(lambda: {"quantity": 0, "avg_price": 0.0, "cost_basis": 0.0})
    
    for fill in data.get("ledger", []):
        symbol = fill.get("symbol", "").strip()
        quantity = fill.get("quantity", 0)
        price = fill.get("price", 0.0)
        
        if not symbol or quantity == 0:
            continue
        
        pos_key = symbol
        pos = position_map[pos_key]
        
        # Update position (FIFO cost basis)
        old_qty = pos["quantity"]
        new_qty = old_qty + quantity
        
        if new_qty == 0:
            # Position closed
            del position_map[pos_key]
        else:
            # Update cost basis
            if new_qty > 0:  # Increasing long
                pos["cost_basis"] = (pos["cost_basis"] + quantity * price) if old_qty >= 0 else quantity * price
            else:  # Increasing short
                pos["cost_basis"] = (pos["cost_basis"] + quantity * price) if old_qty <= 0 else quantity * price
            
            pos["quantity"] = new_qty
            pos["avg_price"] = pos["cost_basis"] / abs(new_qty) if new_qty != 0 else 0.0
    
    # Convert position map to Position objects
    positions = []
    for symbol, pos_data in position_map.items():
        qty = pos_data["quantity"]
        avg_price = pos_data["avg_price"]
        
        # Determine position type
        if symbol[0:3].isalpha() and len(symbol.strip()) <= 6:
            # Equity: typically short symbol like 'GLD', 'AAPL'
            pos_type = "equity"
            positions.append(Position(
                symbol=symbol.strip(),
                type=pos_type,
                quantity=qty,
                avg_price=avg_price,
            ))
        else:
            # Option: OCC format
            underlying, expiry, strike, right = parse_option_symbol(symbol)
            if underlying:
                pos_type = "call" if right == "C" else "put"
                positions.append(Position(
                    symbol=underlying,
                    type=pos_type,
                    quantity=qty,
                    avg_price=avg_price,
                    strike=strike,
                    expiry=expiry,
                ))
    
    return Portfolio(cash=cash, positions=positions)


def analyze_positions(state_file: Optional[str] = None) -> Optional[Trade]:
    """
    Fetch positions, scan recommendations, and validate trade against portfolio.
    
    Returns the first conflict-free recommendation, or reports conflicts.
    """
    # Load settings to get the configured watchlist
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from wheel.config import Settings
    settings = Settings.from_env()
    

    
    print("🔄 Fetching portfolio state...")
    portfolio = get_portfolio_state(state_file)
    
    print(f"✓ Portfolio loaded: ${portfolio.cash:,.2f} cash, {len(portfolio.positions)} positions")
    if portfolio.positions:
        print("\n📊 Current Positions:")
        for pos in portfolio.positions:
            qty_str = f"{abs(pos.quantity)} {pos.type}"
            side = "LONG" if pos.is_long() else "SHORT"
            expiry_str = f" exp {pos.expiry}" if pos.expiry else ""
            strike_str = f" ${pos.strike}" if pos.strike else ""
            print(f"  {side:5} {qty_str:20} {pos.symbol:8}{strike_str}{expiry_str}")
    
    watchlist_str = ", ".join(settings.watchlist)
    print(f"\n🔄 Running scan to find recommendations (watchlist: {watchlist_str})...")
    # Scan the configured watchlist for multiple candidate underlyings
    scan_result = run_cli_command(["scan", "--json"] + list(settings.watchlist))
    if not scan_result:
        print("❌ Failed to fetch scan results")
        return None

    if not isinstance(scan_result, list) or not scan_result:
        print("📊 No recommendations available.")
        return Trade(
            recommendation="hold",
            underlying="CASH",
            action="Monitor for entry opportunity",
            rationale="No scanner recommendations at this time",
            risk_score=0,
            expected_return=0.0,
            valid=True,
        )

    print(f"✓ Scanner found {len(scan_result)} candidates")
    
    # Evaluate each recommendation for conflicts
    print("\n📋 Validating recommendations against portfolio:")
    for i, rec in enumerate(scan_result, 1):
        symbol = rec.get("symbol")
        action = rec.get("action")
        conflicts = portfolio.has_conflicts(rec)
        
        if conflicts:
            print(f"\n  ❌ #{i} {symbol} {action}")
            for conflict in conflicts:
                print(f"     └─ {conflict}")
        else:
            print(f"\n  ✅ #{i} {symbol} {action} — VALID")
    
    # Find first conflict-free recommendation
    valid_rec = None
    for rec in scan_result:
        if not portfolio.has_conflicts(rec):
            valid_rec = rec
            break
    
    if not valid_rec:
        print("\n⚠️  All recommendations conflict with existing positions.")
        print("   Consider: rolling calls, closing positions, or holding cash.")
        return Trade(
            recommendation="hold",
            underlying="CASH",
            action="Review existing positions before new trades",
            rationale="All scanner recommendations conflict with current portfolio",
            risk_score=5,
            expected_return=0.0,
            valid=False,
        )
    
    # Skip SKIP actions when building the final recommendation
    if valid_rec.get("action") == "SKIP":
        # Find first non-SKIP, conflict-free recommendation
        for rec in scan_result:
            if rec.get("action") != "SKIP" and not portfolio.has_conflicts(rec):
                valid_rec = rec
                break
        else:
            # All trade recommendations have conflicts; only SKIPs are available
            print("\n⚠️  All actionable recommendations conflict with existing positions.")
            print("   Consider: rolling calls, closing positions, or holding cash.")
            return Trade(
                recommendation="hold",
                underlying="CASH",
                action="Review existing positions before new trades",
                rationale="All tradeable recommendations conflict with current portfolio",
                risk_score=5,
                expected_return=0.0,
                valid=False,
            )
    
    # Format the valid recommendation
    contract = valid_rec.get("contract") or {}
    strike = contract.get("strike", 0.0)
    annualized = valid_rec.get("annualized_yield", 0.03)
    
    trade = Trade(
        recommendation=valid_rec.get("action", "hold"),
        underlying=valid_rec.get("symbol", "UNKNOWN"),
        action=f"{valid_rec.get('action')} {valid_rec.get('symbol')} @ ${strike} strike",
        rationale=valid_rec.get("rationale", "Scanner recommendation"),
        risk_score=int(valid_rec.get("risk_score", 3)),
        expected_return=float(annualized),
        valid=True,
    )
    
    return trade


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wheel Trading Agent")
    parser.add_argument("--state", help="State file path (default: state/paper_account.json)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 Wheel Trading Agent — Portfolio-Aware Trade Recommendation")
    print("=" * 70)

    trade = analyze_positions(args.state)
    if not trade:
        print("\n❌ Analysis failed")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📋 FINAL RECOMMENDATION")
    print("=" * 70)
    print(f"Valid:         {'✅ YES' if trade.valid else '❌ NO - CONFLICTS'}")
    print(f"Action:        {trade.action}")
    print(f"Recommendation: {trade.recommendation}")
    print(f"Underlying:     {trade.underlying}")
    print(f"Rationale:      {trade.rationale}")
    print(f"Risk Score:     {trade.risk_score}/10")
    print(f"Expected Return: {trade.expected_return * 100:.1f}%")
    if trade.conflicts:
        print(f"\nConflicts detected:")
        for conflict in trade.conflicts:
            print(f"  ⚠️  {conflict}")
    print("=" * 70)

    # Output as JSON for automation
    result = {
        "valid": trade.valid,
        "recommendation": trade.recommendation,
        "underlying": trade.underlying,
        "action": trade.action,
        "rationale": trade.rationale,
        "risk_score": trade.risk_score,
        "expected_return": trade.expected_return,
        "conflicts": trade.conflicts,
    }
    print(json.dumps(result, indent=2))
    
    sys.exit(0 if trade.valid else 1)


if __name__ == "__main__":
    main()
