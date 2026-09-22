"""Feature engineering for advisory layer.

Pure Python, no dependencies. Computes the derived features (IV rank, term structure,
position state, earnings proximity) that Claude will use to recommend delta targets.

All features are deterministic and cached per symbol + TTL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class IVMetrics:
    """Implied volatility statistics."""

    iv_30d: float  # IV at ~30 DTE
    iv_60d: float  # IV at ~60 DTE
    iv_rank: float  # (IV - 52w_low) / (52w_high - 52w_low), 0..100
    iv_percentile: float  # 0..100, smoothed against historical distribution
    term_structure_slope: float  # (IV_60d - IV_30d) / IV_30d, -1..1


@dataclass(frozen=True)
class VolMetrics:
    """Realized and implied volatility comparison."""

    rv_20d: float  # 20-day realized volatility
    rv_60d: float  # 60-day realized volatility
    rv_iv_ratio_30d: float  # rv_20d / iv_30d, <1 means IV is expensive
    skew_put_call: float  # (put_iv - call_iv) / call_iv at same strike, put-weighted


@dataclass(frozen=True)
class PositionState:
    """Current holdings affecting the recommendation."""

    shares_held: int
    csp_open_count: int  # cash-secured puts
    ccall_open_count: int  # covered calls
    avg_cost_per_share: float
    cash_available: float
    collateral_used_pct: float  # (put_req + call_margin) / total_bp
    days_in_position_avg: float  # avg days held across open positions


@dataclass(frozen=True)
class EarningsState:
    """Earnings event proximity."""

    dte_to_next_earnings: Optional[int]  # days to next earnings, None if > 90 or unknown
    earnings_in_30d: bool  # any earnings in next 30 days?
    earnings_in_45d: bool  # any earnings in next 45 days?
    dte_to_exp_vs_earnings: Optional[float]  # (dte_to_expiry / dte_to_earnings) if earnings < 90d


@dataclass(frozen=True)
class UnderlyingMetrics:
    """Price and trend."""

    price: float
    change_1d_pct: float  # today's % return
    change_5d_pct: float  # this week's % return
    change_30d_pct: float  # this month's % return
    atr_20d: float  # average true range (annualized vol proxy)


@dataclass(frozen=True)
class FeatureVector:
    """Complete feature set for a single symbol, ready for Claude.

    All fields are serializable to JSON. No Option/Greeks objects — only
    scalars and bounded strings.
    """

    symbol: str
    ts: datetime  # when computed
    price_data: UnderlyingMetrics
    iv_data: IVMetrics
    vol_data: VolMetrics
    position_state: PositionState
    earnings_data: EarningsState

    # Derived context flags (string for robustness)
    market_regime: str  # "high_iv", "low_iv", "vol_crush", "vol_spike", "normal"
    position_type: str  # "neutral", "long_shares", "short_puts", "mixed"
    trade_season: str  # "earnings_risk", "before_earnings", "post_earnings", "quiet"

    # Metadata
    data_source: str = "wheel"  # source identifier
    confidence: float = 1.0  # 0..1, how confident we are in these numbers

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d

    @staticmethod
    def from_dict(d: dict) -> FeatureVector:
        """Deserialize from dict."""
        d = d.copy()
        if isinstance(d["ts"], str):
            d["ts"] = datetime.fromisoformat(d["ts"])
        return FeatureVector(**d)


def classify_market_regime(iv_rank: float, iv_percentile: float, term_slope: float) -> str:
    """Classify market regime from IV shape.

    Args:
        iv_rank: (IV - low) / (high - low), 0..100
        iv_percentile: smoothed percentile, 0..100
        term_slope: (IV_60 - IV_30) / IV_30

    Returns:
        regime label.
    """
    # High IV absolute level
    if iv_percentile > 75:
        if term_slope < -0.02:
            return "vol_crush"  # IV dropping, was high
        return "high_iv"

    # Low IV absolute level
    if iv_percentile < 25:
        if term_slope > 0.05:
            return "vol_spike"  # IV rising, was low
        return "low_iv"

    # Steep forward curve (rising IV into the future)
    if term_slope > 0.08:
        return "vol_spike"

    # Steep backwardation (falling IV into the future)
    if term_slope < -0.08:
        return "vol_crush"

    return "normal"


def classify_position_type(
    shares_held: int, csp_open: int, ccall_open: int
) -> str:
    """Classify net position.

    Args:
        shares_held: shares currently owned
        csp_open: cash-secured puts outstanding
        ccall_open: covered calls outstanding

    Returns:
        position label.
    """
    if shares_held > 0 and ccall_open > 0 and csp_open == 0:
        return "long_shares"  # wheel leg 3: covered call
    if csp_open > 0 and shares_held == 0 and ccall_open == 0:
        return "short_puts"  # wheel leg 1: naked puts
    if shares_held > 0 and csp_open == 0 and ccall_open == 0:
        return "long_shares"  # accumulating, no premium sale
    if csp_open > 0 or ccall_open > 0:
        return "mixed"  # multiple wheel legs
    return "neutral"


def classify_trade_season(
    dte_to_earnings: Optional[int], earnings_in_30d: bool, earnings_in_45d: bool
) -> str:
    """Classify earnings proximity.

    Args:
        dte_to_earnings: days to next earnings, None if > 90 or unknown
        earnings_in_30d: earnings in next 30 days
        earnings_in_45d: earnings in next 45 days

    Returns:
        trade season label.
    """
    if dte_to_earnings is None or dte_to_earnings > 90:
        return "quiet"

    if dte_to_earnings > 14:
        if earnings_in_30d:
            return "before_earnings"
        return "quiet"

    # dte_to_earnings <= 14: earnings is within 2 weeks
    return "earnings_risk"


def compute_iv_rank(
    current_iv: float,
    iv_52w_low: float,
    iv_52w_high: float,
) -> float:
    """Compute IV rank: where current IV sits in 52-week range.

    Args:
        current_iv: current implied vol (decimal, e.g., 0.25)
        iv_52w_low: 52-week low IV
        iv_52w_high: 52-week high IV

    Returns:
        IV rank, 0..100. Returns 50 if range is zero (flat vol environment).
    """
    if iv_52w_high <= iv_52w_low:
        return 50.0  # undefined; assume neutral

    rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)
    return max(0.0, min(100.0, rank * 100.0))


def compute_realized_vol(returns: list[float], annualize: bool = True) -> float:
    """Compute realized volatility from daily returns.

    Args:
        returns: list of daily returns (decimals, e.g., [0.005, -0.001, ...])
        annualize: if True, multiply by sqrt(252)

    Returns:
        Realized vol (decimal).
    """
    if not returns or len(returns) < 2:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(max(0.0, variance))

    if annualize:
        return std_dev * math.sqrt(252.0)
    return std_dev


def compute_atr(high_prices: list[float], low_prices: list[float], close_prices: list[float]) -> float:
    """Compute average true range (14-period default), annualized as vol estimate.

    Args:
        high_prices: high prices (most recent = last)
        low_prices: low prices (most recent = last)
        close_prices: close prices (most recent = last)

    Returns:
        ATR expressed as annualized vol (decimal).
    """
    if not (high_prices and low_prices and close_prices):
        return 0.0
    if len(high_prices) < 2:
        return 0.0

    period = min(14, len(high_prices))
    true_ranges = []

    for i in range(len(high_prices)):
        if i == 0:
            tr = high_prices[i] - low_prices[i]
        else:
            tr = max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i - 1]),
                abs(low_prices[i] - close_prices[i - 1]),
            )
        true_ranges.append(tr)

    atr = sum(true_ranges[-period:]) / period
    current_price = close_prices[-1]
    if current_price <= 0:
        return 0.0

    atr_pct = atr / current_price
    return atr_pct * math.sqrt(252.0)  # annualize


def build_feature_vector(
    symbol: str,
    price: float,
    iv_30d: float,
    iv_60d: float,
    iv_52w_low: float,
    iv_52w_high: float,
    rv_20d: float,
    rv_60d: float,
    skew_put_call: float,
    shares_held: int,
    csp_open_count: int,
    ccall_open_count: int,
    avg_cost_per_share: float,
    cash_available: float,
    collateral_used_pct: float,
    days_in_position_avg: float,
    dte_to_next_earnings: Optional[int],
    change_1d_pct: float,
    change_5d_pct: float,
    change_30d_pct: float,
    atr_20d: float,
) -> FeatureVector:
    """Build a complete feature vector from market and position data.

    All inputs are floats/ints/None. No Option objects. Returns a JSON-serializable
    FeatureVector ready for Claude.

    Args:
        symbol: ticker (e.g., "AAPL")
        price: current price
        iv_30d, iv_60d: implied vol at ~30 and ~60 DTE (decimals)
        iv_52w_low, iv_52w_high: 52-week IV range
        rv_20d, rv_60d: 20-day and 60-day realized vol (decimals)
        skew_put_call: (put_iv - call_iv) / call_iv, put-weighted
        shares_held, csp_open_count, ccall_open_count: position counts
        avg_cost_per_share, cash_available, collateral_used_pct, days_in_position_avg: position state
        dte_to_next_earnings: days to next earnings, None if unknown or > 90
        change_1d_pct, change_5d_pct, change_30d_pct: price changes (% decimals)
        atr_20d: 20-day ATR as annualized vol estimate

    Returns:
        FeatureVector with all fields populated.
    """
    # IV metrics
    iv_rank = compute_iv_rank(iv_30d, iv_52w_low, iv_52w_high)
    # Percentile: smoothed rank; in production this would come from historical
    # IV percentile data. For now, use rank as a proxy.
    iv_percentile = iv_rank

    term_slope = (iv_60d - iv_30d) / iv_30d if iv_30d > 0 else 0.0

    iv_data = IVMetrics(
        iv_30d=iv_30d,
        iv_60d=iv_60d,
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        term_structure_slope=term_slope,
    )

    # Vol metrics
    rv_iv_ratio = rv_20d / iv_30d if iv_30d > 0 else 1.0

    vol_data = VolMetrics(
        rv_20d=rv_20d,
        rv_60d=rv_60d,
        rv_iv_ratio_30d=rv_iv_ratio,
        skew_put_call=skew_put_call,
    )

    # Position state
    position_state = PositionState(
        shares_held=shares_held,
        csp_open_count=csp_open_count,
        ccall_open_count=ccall_open_count,
        avg_cost_per_share=avg_cost_per_share,
        cash_available=cash_available,
        collateral_used_pct=collateral_used_pct,
        days_in_position_avg=days_in_position_avg,
    )

    # Earnings state
    earnings_in_30d = dte_to_next_earnings is not None and dte_to_next_earnings <= 30
    earnings_in_45d = dte_to_next_earnings is not None and dte_to_next_earnings <= 45
    dte_to_exp_vs_earnings = None
    if dte_to_next_earnings is not None and dte_to_next_earnings > 0:
        # Assume 35 DTE target
        dte_to_exp_vs_earnings = 35.0 / dte_to_next_earnings if dte_to_next_earnings > 0 else None

    earnings_data = EarningsState(
        dte_to_next_earnings=dte_to_next_earnings,
        earnings_in_30d=earnings_in_30d,
        earnings_in_45d=earnings_in_45d,
        dte_to_exp_vs_earnings=dte_to_exp_vs_earnings,
    )

    # Price metrics
    price_data = UnderlyingMetrics(
        price=price,
        change_1d_pct=change_1d_pct,
        change_5d_pct=change_5d_pct,
        change_30d_pct=change_30d_pct,
        atr_20d=atr_20d,
    )

    # Derived classifications
    market_regime = classify_market_regime(iv_rank, iv_percentile, term_slope)
    position_type = classify_position_type(shares_held, csp_open_count, ccall_open_count)
    trade_season = classify_trade_season(dte_to_next_earnings, earnings_in_30d, earnings_in_45d)

    return FeatureVector(
        symbol=symbol,
        ts=datetime.now(),
        price_data=price_data,
        iv_data=iv_data,
        vol_data=vol_data,
        position_state=position_state,
        earnings_data=earnings_data,
        market_regime=market_regime,
        position_type=position_type,
        trade_season=trade_season,
    )


# Example usage / validation
if __name__ == "__main__":
    fv = build_feature_vector(
        symbol="AAPL",
        price=185.50,
        iv_30d=0.28,
        iv_60d=0.30,
        iv_52w_low=0.15,
        iv_52w_high=0.45,
        rv_20d=0.22,
        rv_60d=0.24,
        skew_put_call=-0.08,
        shares_held=100,
        csp_open_count=1,
        ccall_open_count=0,
        avg_cost_per_share=180.0,
        cash_available=50000.0,
        collateral_used_pct=0.25,
        days_in_position_avg=5.0,
        dte_to_next_earnings=23,
        change_1d_pct=0.015,
        change_5d_pct=0.032,
        change_30d_pct=0.08,
        atr_20d=0.18,
    )

    import json

    print(json.dumps(fv.to_dict(), indent=2, default=str))
