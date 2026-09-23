"""Roll execution — place spreads and manage rolls against Schwab API.

Converts a roll recommendation (old strike/expiry → new strike/expiry) into
a two-leg spread order: BUY-to-close old, SELL-to-open new.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Callable
from .schwab import SchwabClient, SchwabError


@dataclass
class RollResult:
    """Result of attempting a roll execution."""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    credit_received: Optional[float] = None
    error: Optional[str] = None


def build_roll_order(
    symbol: str,
    quantity: int,
    old_strike: float,
    old_expiry: date,
    new_strike: float,
    new_expiry: date,
) -> dict:
    """Build a two-leg roll spread order (BUY old call, SELL new call).
    
    Args:
        symbol: underlying (e.g. 'GLD')
        quantity: number of contracts
        old_strike: current short call strike
        old_expiry: current short call expiry
        new_strike: target new call strike
        new_expiry: target new call expiry
    
    Returns:
        Order dict in Schwab format (details below implementation)
    """
    # Schwab order structure for a spread:
    # - orderType: "NET_CREDIT" (want to receive credit)
    # - orderStrategyType: "SINGLE" (each leg) or "MULTI_LEG" (spread)
    # - orderLegCollection: [{instrument, quantity, instruction}, ...]
    # - price: target net credit
    
    # NOTE: Schwab REST API doesn't expose trade execution directly from apps.
    # In production, we'd either:
    # 1. Use Schwab's web/desktop UI to execute manually
    # 2. Use a broker that exposes REST order placement (e.g. TD Ameritrade legacy)
    # 3. Use a third-party order routing service (e.g. Alpaca, Interactive Brokers)
    #
    # For this MVP: we return the order dict so it can be logged/audited,
    # and raise NotImplementedError with clear instructions.
    
    return {
        "strategy": "ROLL_CALL",
        "underlying": symbol,
        "legs": [
            {
                "action": "BUY_TO_CLOSE",
                "type": "CALL",
                "strike": old_strike,
                "expiry": old_expiry.isoformat(),
                "quantity": quantity,
            },
            {
                "action": "SELL_TO_OPEN",
                "type": "CALL",
                "strike": new_strike,
                "expiry": new_expiry.isoformat(),
                "quantity": quantity,
            },
        ],
    }


def execute_roll(
    client: SchwabClient,
    symbol: str,
    quantity: int,
    old_strike: float,
    old_expiry: date,
    new_strike: float,
    new_expiry: date,
    dry_run: bool = True,
) -> RollResult:
    """Execute a roll order via Schwab API.
    
    NOTE: Schwab API does not currently expose order placement for retail.
    This function builds the order and returns it for manual execution or
    third-party routing.
    
    Args:
        client: authenticated SchwabClient
        symbol: underlying (e.g. 'GLD')
        quantity: number of contracts
        old_strike: current short call strike
        old_expiry: current short call expiry
        new_strike: target new call strike
        new_expiry: target new call expiry
        dry_run: when True, logs order without placing it
    
    Returns:
        RollResult with order details
    """
    order = build_roll_order(symbol, quantity, old_strike, old_expiry, new_strike, new_expiry)
    
    if dry_run:
        return RollResult(
            success=True,
            message=f"[DRY RUN] Roll ready to execute: {order}",
            order_id=None,
        )
    
    # Schwab retail API does not support order placement.
    # For production, this would route to:
    # 1. A manual execution workflow (Slack/email notification to trader)
    # 2. A broker with REST order API (Alpaca, IB, etc.)
    # 3. A third-party order management system
    raise NotImplementedError(
        "Schwab retail API does not expose order placement. "
        "To execute this roll, use one of:\n"
        "  1. Schwab mobile/web UI (manual execution)\n"
        "  2. Alpaca API (if you have an Alpaca account)\n"
        "  3. Interactive Brokers (if you have an IB account)\n"
        f"\nOrder to execute:\n{order}"
    )


def fetch_optimal_strike(
    client: SchwabClient,
    symbol: str,
    target_delta: float,
    target_expiry: date,
) -> Optional[float]:
    """Fetch the call strike closest to a target delta for a given expiry.
    
    Args:
        client: authenticated SchwabClient
        symbol: underlying (e.g. 'GLD')
        target_delta: desired delta (e.g. 0.30)
        target_expiry: desired expiry date
    
    Returns:
        Strike price closest to target delta, or None if not found
    """
    try:
        # Fetch call chain around target expiry
        chain = client.chain_raw(symbol)
        
        # Walk the chain to find contracts at target_expiry
        for exp_key, strikes in (chain.get("callExpDateMap", {}) or {}).items():
            # Parse expiry from key (format: "2026-01-16:30")
            exp_parts = exp_key.split(":")[0].split("-")
            if len(exp_parts) == 3:
                exp_year, exp_month, exp_day = map(int, exp_parts)
                exp = date(exp_year, exp_month, exp_day)
                if exp != target_expiry:
                    continue
            
            # Found the target expiry; walk strikes for closest delta
            best_strike = None
            best_diff = float("inf")
            
            for strike_str, legs in (strikes or {}).items():
                if not legs:
                    continue
                leg = legs[0]
                
                # Get delta from the leg
                delta = abs(leg.get("delta", 0.0))
                diff = abs(delta - target_delta)
                
                if diff < best_diff:
                    best_diff = diff
                    best_strike = float(strike_str)
            
            return best_strike
    except Exception as e:
        pass  # Silently fail; caller will use fallback
    
    return None
