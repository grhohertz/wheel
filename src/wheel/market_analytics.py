"""Market data analytics — IV rank, moving averages, earnings calendar.

Pure functions for computing adaptive delta targeting inputs from raw market data.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional
from .schwab import SchwabClient


def compute_moving_average(prices: list[float], period: int) -> Optional[float]:
    """Compute simple moving average over a list of closing prices.
    
    Args:
        prices: list of close prices (oldest first)
        period: window size (e.g. 50, 200)
    
    Returns:
        MA or None if fewer than `period` prices available
    """
    if not prices or len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def compute_iv_rank(current_iv: float, iv_hist: list[float], lookback_days: int = 252) -> Optional[float]:
    """Compute IV rank as a percentile within historical range.
    
    IV rank = (IV - min_IV) / (max_IV - min_IV)
    
    Args:
        current_iv: current implied volatility (as decimal, e.g. 0.25)
        iv_hist: list of historical IVs over lookback period
        lookback_days: typically 252 (1 year of trading days)
    
    Returns:
        IV rank in [0.0, 1.0] or None if insufficient data
    """
    if not iv_hist or len(iv_hist) < 10:
        return None
    
    min_iv = min(iv_hist)
    max_iv = max(iv_hist)
    
    if max_iv <= min_iv:
        return 0.5  # flat period; neutral
    
    rank = (current_iv - min_iv) / (max_iv - min_iv)
    return max(0.0, min(1.0, rank))  # clamp to [0, 1]


def extract_closes(pricehistory: dict) -> list[float]:
    """Extract close prices from Schwab pricehistory response.
    
    Args:
        pricehistory: response from SchwabClient.pricehistory_raw()
    
    Returns:
        list of close prices in chronological order (oldest first)
    """
    closes = []
    for candle in pricehistory.get("candles", []) or []:
        close = candle.get("close")
        if close:
            closes.append(float(close))
    return closes


def fetch_market_context(
    client: SchwabClient,
    symbol: str,
    contract: Optional[dict] = None,
) -> dict:
    """Fetch all market data needed for adaptive delta targeting.
    
    Args:
        client: authenticated SchwabClient
        symbol: stock ticker (e.g. 'GLD')
        contract: optional option contract dict with 'expiry' key for earnings lookup
    
    Returns:
        dict with keys:
        - current_price: float
        - ma50: float or None
        - ma200: float or None
        - iv_rank: float [0,1] or None
        - current_iv: float
        - earnings_in_window: bool (default False if lookup fails)
    """
    result = {
        "current_price": 0.0,
        "ma50": None,
        "ma200": None,
        "iv_rank": None,
        "current_iv": 0.30,
        "earnings_in_window": False,
    }
    
    try:
        # Fetch quote for current price and IV
        quote = client.get_quote(symbol)
        result["current_price"] = quote.price
        result["current_iv"] = quote.iv
        
        # Fetch 1 year of daily price history
        pricehistory = client.pricehistory_raw(
            symbol,
            period_type="year",
            period=1,
            frequency_type="daily",
            frequency=1,
        )
        closes = extract_closes(pricehistory)
        
        # Compute 50/200 day moving averages
        if closes:
            result["ma50"] = compute_moving_average(closes, 50)
            result["ma200"] = compute_moving_average(closes, 200)
            
            # Compute IV rank from historical closes
            # (In production: fetch historical IV from option chain or third-party source)
            # For now: approximate IV rank from price volatility
            if len(closes) >= 252:
                recent_iv = compute_price_volatility(closes[-252:])
                hist_ivs = [compute_price_volatility(closes[i:i+252])
                           for i in range(0, max(0, len(closes)-252), 63)]
                result["iv_rank"] = compute_iv_rank(result["current_iv"], hist_ivs)
        
        # TODO: Fetch earnings date from fundamental data or earnings calendar API
        # For now: assume no earnings in window
        result["earnings_in_window"] = False
        
    except Exception as e:
        # Graceful degradation: return defaults if any fetch fails
        pass
    
    return result


def compute_price_volatility(closes: list[float]) -> float:
    """Estimate annualized volatility (σ) from daily close prices.
    
    Args:
        closes: list of daily close prices (at least 2)
    
    Returns:
        annualized volatility as decimal (e.g. 0.25 = 25%)
    """
    if len(closes) < 2:
        return 0.30  # default
    
    # Compute daily returns
    returns = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0:
            returns.append((closes[i] / closes[i-1]) - 1.0)
    
    if not returns:
        return 0.30
    
    # Compute daily standard deviation
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    daily_std = variance ** 0.5
    
    # Annualize (252 trading days)
    annual_vol = daily_std * (252 ** 0.5)
    return annual_vol


def check_earnings_in_window(
    client: SchwabClient,
    symbol: str,
    expiry: date,
) -> bool:
    """Check if earnings are expected before expiry.
    
    Args:
        client: authenticated SchwabClient
        symbol: stock ticker
        expiry: option expiry date
    
    Returns:
        True if earnings expected between now and expiry
    """
    # TODO: Fetch earnings calendar from fundamental data or external API
    # For now: return False (no earnings data)
    return False
