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
from datetime import datetime, timedelta, date

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
            # Get collateral needed from the recommendation (scan already calculated it)
            collateral_needed = recommendation.get("collateral", 0)
            portfolio_cash = self.cash
            
            if portfolio_cash < collateral_needed:
                conflicts.append(
                    f"Insufficient cash collateral: need ${collateral_needed:,.0f}, have ${portfolio_cash:,.0f}"
                )
            
            # Also conflict if we have multiple short puts already
            if len([p for p in puts if p.is_short()]) > 0:
                conflicts.append(
                    f"Already have short put on {symbol}; can't stack multiple puts"
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
    roll_metadata: Optional[dict] = None  # metadata for roll executor


@dataclass(frozen=True)
class DeltaPolicy:
    """Adaptive delta targeting rules for roll decisions."""
    base: float = 0.30                    # base target delta
    lo: float = 0.15                      # floor
    hi: float = 0.40                      # ceiling
    w_ivr: float = 0.10                   # IV rank weight
    w_trend: float = 0.05                 # trend weight
    w_earnings: float = 0.05              # earnings-in-window weight
    w_exit: float = 0.05                  # exit-intent weight
    w_portfolio: float = 0.03             # portfolio-delta-excess weight
    min_basis_margin: float = 0.01        # never sell below cost_basis * 1.01
    min_credit_pct: float = 0.003         # min credit as % of strike
    dte_min: int = 30                     # min days to expiry
    dte_max: int = 45                     # max days to expiry


@dataclass
class RollContext:
    """Market and portfolio context for delta targeting."""
    spot: float                           # current underlying price
    cost_basis: float                     # average cost per share
    iv_rank: Optional[float] = None       # 0..1, percentile of IV over 252d
    ma50: Optional[float] = None          # 50-day moving average
    ma200: Optional[float] = None         # 200-day moving average
    earnings_in_window: bool = False      # earnings before expiry?
    want_exit: bool = False               # user wants to close position
    portfolio_delta_excess: Optional[float] = None  # portfolio delta vs budget
    round_trip_cost: float = 5.0          # bid-ask round-trip in $
    dividend_before_expiry: float = 0.0   # dividend amount before expiry


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, x))


def compute_target_delta(ctx: RollContext, policy: DeltaPolicy = DeltaPolicy()) -> tuple[float, dict]:
    """
    Compute adaptive target delta based on market regime.
    Returns (target_delta, explanation_dict) for logging and debugging.
    """
    d = policy.base
    reasons = []
    
    # 1. IV rank: rich vol -> paid better per unit of risk -> take less risk
    if ctx.iv_rank is not None:
        ivr = clamp(ctx.iv_rank, 0.0, 1.0)
        adj = -policy.w_ivr * (2 * ivr - 1)  # ivr=1.0 -> -0.10, ivr=0.0 -> +0.10
        d += adj
        reasons.append(f"IV_rank={ivr:.2%} {adj:+.3f}")
    
    # 2. Trend: don't cap a runner
    if ctx.ma50 and ctx.ma200:
        if ctx.spot > ctx.ma50 > ctx.ma200:
            d -= policy.w_trend
            reasons.append(f"uptrend -{policy.w_trend:.2f}")
        elif ctx.spot < ctx.ma50:
            d += policy.w_trend
            reasons.append(f"downtrend +{policy.w_trend:.2f}")
    
    # 3. Earnings in contract life -> gap buffer
    if ctx.earnings_in_window:
        d -= policy.w_earnings
        reasons.append(f"earnings_in_window -{policy.w_earnings:.2f}")
    
    # 4. Exit intent
    if ctx.want_exit and ctx.spot > ctx.cost_basis * 1.15:
        d += policy.w_exit
        reasons.append(f"exit_intent +{policy.w_exit:.2f}")
    
    # 5. Portfolio delta excess (optional)
    if ctx.portfolio_delta_excess is not None:
        n = clamp(ctx.portfolio_delta_excess, -1.0, 1.0) * policy.w_portfolio
        d += n
        reasons.append(f"portfolio_delta {n:+.3f}")
    
    final = clamp(d, policy.lo, policy.hi)
    
    return final, {
        "target_delta": final,
        "raw_computed": d,
        "reasons": reasons,
        "clamped": final != d,
    }


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


def get_portfolio_state(state_file: Optional[str] = None, live: bool = False) -> Portfolio:
    """
    Fetch and parse portfolio state from state JSON file or live Schwab account.
    
    If live=True, fetches actual account positions from Schwab API.
    Otherwise, walks the ledger in the state file to compute current positions.
    """
    if live:
        # Fetch from live Schwab account
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from wheel.schwab import SchwabCredentials, SchwabAuth, SchwabClient
        
        try:
            creds = SchwabCredentials.from_env()
            auth = SchwabAuth(creds)
            client = SchwabClient(auth)
            positions_data = client.positions()
            
            # Fetch account info to get cash balance
            accounts = client.accounts(with_positions=False)
            cash = 0.0
            if accounts:
                # Get cash from the first account's securitiesAccount
                account = accounts[0]
                sec_acct = account.get("securitiesAccount", {})
                
                # Look for cash in initialBalances or currentBalances
                if "currentBalances" in sec_acct:
                    balances = sec_acct["currentBalances"]
                    if isinstance(balances, dict):
                        cash = float(balances.get("cashAvailable", balances.get("cash", 0.0)))
                elif "initialBalances" in sec_acct:
                    balances = sec_acct["initialBalances"]
                    if isinstance(balances, dict):
                        cash = float(balances.get("cashAvailable", balances.get("cash", 0.0)))
            
            # Convert Schwab position format to our Portfolio format
            positions: list[Position] = []
            
            for pos in positions_data:
                symbol = pos.get("symbol", "").strip()
                quantity = int(pos.get("quantity", 0))
                avg_price = float(pos.get("average_price", 0.0))
                asset_type = pos.get("asset_type", "")
                
                if not symbol or quantity == 0:
                    continue
                
                # Determine position type
                if asset_type == "OPTION":
                    # Parse OCC format
                    underlying, expiry, strike, right = parse_option_symbol(symbol)
                    if underlying:
                        pos_type = "call" if right == "C" else "put"
                        positions.append(Position(
                            symbol=underlying,
                            type=pos_type,
                            quantity=quantity,
                            avg_price=avg_price,
                            strike=strike,
                            expiry=expiry,
                        ))
                else:
                    # Equity
                    positions.append(Position(
                        symbol=symbol,
                        type="equity",
                        quantity=quantity,
                        avg_price=avg_price,
                    ))
            
            return Portfolio(cash=cash, positions=positions)
        except Exception as e:
            print(f"❌ Failed to fetch from Schwab: {e}")
            return Portfolio(cash=0.0, positions=[])
    
    # Fall back to state file
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


def get_call_chain(symbol: str, days_to_expiry: int = 40) -> Optional[list]:
    """Fetch option chain for a symbol from Schwab."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    try:
        from wheel.schwab import SchwabCredentials, SchwabAuth, SchwabClient
        creds = SchwabCredentials.from_env()
        auth = SchwabAuth(creds)
        client = SchwabClient(auth)
        chain = client.chain(symbol, dte=days_to_expiry)
        if chain and "callExpDateMap" in chain:
            calls = []
            for expiry_str, expiry_data in chain["callExpDateMap"].items():
                for strike_str, strike_data in expiry_data.items():
                    if strike_data and len(strike_data) > 0:
                        calls.append(strike_data[0])
            return calls
    except Exception as e:
        pass  # Silently fail if chain unavailable
    return None


def find_strike_by_delta(calls: list, target_delta: float = 0.30) -> Optional[dict]:
    """Find the call strike closest to a target delta."""
    if not calls:
        return None
    
    # Look for the call with delta closest to target
    best = None
    best_diff = float('inf')
    
    for call in calls:
        call_delta = abs(call.get("delta", 0.0))
        diff = abs(call_delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best = call
    
    return best


def get_current_price(symbol: str) -> Optional[float]:
    """Fetch current stock price from Schwab."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    try:
        from wheel.schwab import SchwabCredentials, SchwabAuth, SchwabClient
        creds = SchwabCredentials.from_env()
        auth = SchwabAuth(creds)
        client = SchwabClient(auth)
        quote = client.quote(symbol)
        if quote and "quote" in quote:
            return float(quote["quote"].get("mark", quote["quote"].get("last", 0.0)))
    except Exception as e:
        pass  # Silently fail if price unavailable
    return None



def check_expiring_calls(portfolio: Portfolio, policy: DeltaPolicy = DeltaPolicy()) -> Optional[Trade]:
    """
    Check if any short calls expire today or tomorrow.
    If so, recommend rolling them immediately with adaptive delta targeting.
    Returns a Trade recommendation or None if no urgent rolls needed.
    """
    today = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")
    
    expiring_calls = []
    for pos in portfolio.positions:
        if pos.type == "call" and pos.is_short() and pos.expiry:
            try:
                exp_date = datetime.strptime(pos.expiry, "%Y-%m-%d")
                days_left = (exp_date - today).days
                if days_left <= 1:  # Expiring today or tomorrow
                    expiring_calls.append({
                        "symbol": pos.symbol,
                        "quantity": abs(pos.quantity),
                        "strike": pos.strike,
                        "expiry": pos.expiry,
                        "days_left": days_left,
                        "cost_basis": pos.avg_price,
                    })
            except ValueError:
                pass
    
    if expiring_calls:
        call = expiring_calls[0]
        old_strike = call["strike"]
        symbol = call["symbol"]
        cost_basis = call.get("cost_basis", old_strike)
        
        # Fetch call chain ~40 days out
        calls = get_call_chain(symbol, days_to_expiry=40)
        target_strike = old_strike * 1.025  # Fallback: 2.5% bump
        target_delta = 0.30
        delta_explain = {}
        
        # Build RollContext with market data for adaptive delta targeting
        ctx = RollContext(
            spot=get_current_price(symbol) or old_strike,
            cost_basis=cost_basis,
            iv_rank=None,  # Fetch below via market data
            ma50=None,
            ma200=None,
            earnings_in_window=False,
            want_exit=False,
            portfolio_delta_excess=None,
        )
        
        # Fetch market data (IV rank, MAs, earnings)
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from wheel.schwab import SchwabCredentials, SchwabAuth, SchwabClient
            from wheel.market_analytics import fetch_market_context
            
            creds = SchwabCredentials.from_env()
            auth = SchwabAuth(creds)
            client = SchwabClient(auth)
            
            market_ctx = fetch_market_context(client, symbol)
            
            # Update RollContext with real market data
            ctx.spot = market_ctx.get("current_price", ctx.spot)
            ctx.iv_rank = market_ctx.get("iv_rank")
            ctx.ma50 = market_ctx.get("ma50")
            ctx.ma200 = market_ctx.get("ma200")
            ctx.earnings_in_window = market_ctx.get("earnings_in_window", False)
        except Exception as e:
            # Graceful degradation: use fallback values
            pass
        
        # Compute adaptive target delta with market context
        target_delta, delta_explain = compute_target_delta(ctx, policy)
        
        if calls:
            delta_call = find_strike_by_delta(calls, target_delta=target_delta)
            if delta_call and "strike" in delta_call:
                target_strike = float(delta_call["strike"])
        
        target_expiry_dt = today + timedelta(days=40)
        target_expiry = target_expiry_dt.strftime("%Y-%m-%d")
        delta_reason = "; ".join(delta_explain.get("reasons", [f"Δ={target_delta:.2f}"]))
        
        # Build the trade recommendation with all details
        trade = Trade(
            recommendation="ROLL_CALL",
            underlying=call["symbol"],
            action=f"BUY {call['quantity']} @ ${old_strike} / SELL {call['quantity']} @ ${target_strike:.2f} (exp {target_expiry})",
            rationale=f"Short calls expire in {call['days_left']} day(s) — wheel is turning! Rolling to Δ={target_delta:.2f} strike (${target_strike:.2f}) out to {target_expiry} (~40 days). Drivers: {delta_reason}",
            risk_score=6,
            expected_return=0.12,
            valid=True,
        )
        
        # Attach roll executor metadata for later execution
        trade.roll_metadata = {
            "old_strike": old_strike,
            "old_expiry": call["expiry"],
            "new_strike": target_strike,
            "new_expiry": target_expiry,
            "quantity": call["quantity"],
            "target_delta": target_delta,
        }
        
        return trade
    
    return None


def analyze_positions(state_file: Optional[str] = None, live: bool = False) -> Optional[Trade]:
    """
    Fetch positions, scan recommendations, and validate trade against portfolio.
    
    If live=True, fetches actual positions from Schwab API instead of the state file.
    Returns the first conflict-free recommendation, or reports conflicts.
    """
    # Load settings to get the configured watchlist
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from wheel.config import Settings
    settings = Settings.from_env()
    
    source = "Schwab account" if live else "paper trading state"
    print(f"🔄 Fetching portfolio state from {source}...")
    portfolio = get_portfolio_state(state_file, live=live)
    
    # Check for expiring short calls FIRST — this is urgent wheel action
    urgent_roll = check_expiring_calls(portfolio)
    if urgent_roll:
        return urgent_roll
    
    print(f"✓ Portfolio loaded: ${portfolio.cash:,.2f} cash, {len(portfolio.positions)} positions")
    if portfolio.positions:
        print("\n📊 Current Positions:")
        for pos in portfolio.positions:
            qty_str = f"{abs(pos.quantity):4d} {pos.type:8s}"
            side = "LONG" if pos.is_long() else "SHORT"
            strike_str = f"${pos.strike:8.2f}" if pos.strike else "        "
            expiry_str = f"exp {pos.expiry}" if pos.expiry else ""
            print(f"  {side:5s} {qty_str:15s} {pos.symbol:6s} {strike_str:10s} {expiry_str}")
    
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
    parser.add_argument("--live", action="store_true", help="Fetch actual positions from Schwab account instead of paper trading state")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 Wheel Trading Agent — Portfolio-Aware Trade Recommendation")
    print("=" * 70)

    trade = analyze_positions(args.state, live=args.live)
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
    
    # If this is a roll, show execution instructions
    if trade.roll_metadata:
        meta = trade.roll_metadata
        print(f"\n📋 EXECUTION INSTRUCTIONS (Roll)")
        print(f"  1. Open Schwab web/mobile: {trade.underlying} options")
        print(f"  2. Create a spread order (or two GTC orders):")
        print(f"     BUY-TO-CLOSE:  {meta['quantity']} {trade.underlying} ${meta['old_strike']} calls @ {meta['old_expiry']}")
        print(f"     SELL-TO-OPEN:  {meta['quantity']} {trade.underlying} ${meta['new_strike']:.2f} calls @ {meta['new_expiry']}")
        print(f"  3. Target net credit: match current bid-ask spreads")
        print(f"  4. Post in Schwab until filled or 5-10min window expires")
        print(f"  5. Confirm fill against portfolio before market close")
    
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
    if trade.roll_metadata:
        result["roll_metadata"] = trade.roll_metadata
    print(json.dumps(result, indent=2))
    
    sys.exit(0 if trade.valid else 1)


if __name__ == "__main__":
    main()
