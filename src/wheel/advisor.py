"""Market advisor: analyzes chain data and recommends delta targets.

Responsibilities:
1. Compute chain delta distribution statistics (mean, std, quantiles)
2. Analyze market regime (IV rank, vol structure, skew, earnings proximity)
3. Build a FeatureVector from quote + chain + optional position state
4. Call ClaudeAdvisor to get delta recommendations
5. Apply policy guardrails and cache the result

The advisor is the "policy layer" that sits between raw market data and
the wheel strategy's selection logic. It can be run:
- Off (default): strategy uses hard-coded target_delta from config
- Shadow: advisor runs but doesn't affect strategy; logs recommendations
- Overlay (future): advisor recommendations override config targets
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .client import ClaudeAdvisor
from .features import build_feature_vector
from .greeks import CALL, PUT, black_scholes, year_fraction
from .models import OptionContract, Quote

logger = logging.getLogger(__name__)


@dataclass
class ChainStats:
    """Delta distribution statistics for a chain."""

    call_deltas: list[float]
    put_deltas: list[float]

    # Call (short calls) statistics
    call_mean_delta: float
    call_median_delta: float
    call_std_delta: float
    call_min_delta: float
    call_max_delta: float
    call_liquid_count: int  # strikes with bid >= 0.05 and OI >= threshold

    # Put (short puts) statistics
    put_mean_delta: float
    put_median_delta: float
    put_std_delta: float
    put_min_delta: float
    put_max_delta: float
    put_liquid_count: int

    # Moneyness distribution
    spot_price: float
    call_otm_ratio: float  # fraction of calls > spot (truly OTM)
    put_otm_ratio: float  # fraction of puts < spot (truly OTM)

    # Skew
    iv_skew: float  # avg IV of OTM puts vs ATM calls


class AdviceConfig:
    """Configuration for advisory behavior."""

    def __init__(
        self,
        mode: str | None = None,
        min_oi: int = 100,
        max_spread_pct: float = 0.10,
        advisor: Optional[ClaudeAdvisor] = None,
    ):
        """Initialize advisory config.

        Args:
            mode: "off" (default), "shadow", or "overlay"
            min_oi: minimum open interest for liquidity stats
            max_spread_pct: maximum spread % for liquidity
            advisor: ClaudeAdvisor instance (lazy-init if None)
        """
        self.mode = mode or os.getenv("ADVISOR_MODE", "off")
        self.min_oi = min_oi
        self.max_spread_pct = max_spread_pct
        self.advisor = advisor

    def is_enabled(self) -> bool:
        """Return True if advisor is not in 'off' mode."""
        return self.mode in ("shadow", "overlay")

    def lazy_advisor(self) -> Optional[ClaudeAdvisor]:
        """Get or initialize the advisor."""
        if self.advisor is None and self.is_enabled():
            try:
                self.advisor = ClaudeAdvisor()
            except ValueError as e:
                logger.warning(f"Could not initialize advisor: {e}")
        return self.advisor


def _compute_deltas(
    contracts: list[OptionContract],
    spot: float,
    rate: float = 0.04,
    div_yield: float = 0.0,
) -> dict[OptionContract, float]:
    """Compute deltas for all contracts in the chain.

    Args:
        contracts: list of OptionContract
        spot: spot price
        rate: risk-free rate
        div_yield: dividend yield

    Returns:
        dict mapping contract -> delta (absolute value, positive for calls/puts)
    """
    out = {}
    for c in contracts:
        g = c.greeks(spot, rate, div_yield)
        out[c] = abs(g.delta)
    return out


def analyze_chain(
    chain: list[OptionContract],
    quote: Quote,
    as_of: date,
    min_oi: int = 100,
    max_spread_pct: float = 0.10,
) -> ChainStats:
    """Analyze the delta distribution of a chain.

    Args:
        chain: list of OptionContract
        quote: current quote
        as_of: reference date
        min_oi: minimum OI to be considered liquid
        max_spread_pct: maximum spread % to be considered liquid

    Returns:
        ChainStats with delta distribution and liquidity metrics
    """
    spot = quote.price
    deltas = _compute_deltas(chain, spot, div_yield=quote.div_yield)

    call_deltas: list[float] = []
    call_liquid_count = 0
    put_deltas: list[float] = []
    put_liquid_count = 0

    call_otm_count = 0
    call_count = 0
    put_otm_count = 0
    put_count = 0

    iv_skew_put_sum = 0.0
    iv_skew_put_count = 0

    for c in chain:
        if c.dte <= 0:
            continue

        delta = deltas.get(c, 0.0)
        liquid = c.bid >= 0.05 and c.open_interest >= min_oi and c.spread_pct <= max_spread_pct

        if c.is_call():
            call_deltas.append(delta)
            if liquid:
                call_liquid_count += 1
            if c.strike > spot:
                call_otm_count += 1
            call_count += 1
        else:  # put
            put_deltas.append(delta)
            if liquid:
                put_liquid_count += 1
            if c.strike < spot:
                put_otm_count += 1
            put_count += 1
            # Accumulate IV skew (OTM puts vs ATM)
            if c.strike < spot:
                iv_skew_put_sum += c.iv
                iv_skew_put_count += 1

    # Compute statistics
    def quantiles(data: list[float]) -> tuple[float, float, float, float, float]:
        if not data:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        data_sorted = sorted(data)
        mean = sum(data_sorted) / len(data_sorted)
        median = data_sorted[len(data_sorted) // 2]
        variance = sum((x - mean) ** 2 for x in data_sorted) / len(data_sorted)
        std = variance ** 0.5
        return mean, median, std, min(data_sorted), max(data_sorted)

    call_mean, call_median, call_std, call_min, call_max = quantiles(call_deltas)
    put_mean, put_median, put_std, put_min, put_max = quantiles(put_deltas)

    call_otm_ratio = call_otm_count / call_count if call_count > 0 else 0.0
    put_otm_ratio = put_otm_count / put_count if put_count > 0 else 0.0

    # IV skew: average of OTM puts
    iv_skew = iv_skew_put_sum / iv_skew_put_count if iv_skew_put_count > 0 else quote.iv

    return ChainStats(
        call_deltas=call_deltas,
        put_deltas=put_deltas,
        call_mean_delta=call_mean,
        call_median_delta=call_median,
        call_std_delta=call_std,
        call_min_delta=call_min,
        call_max_delta=call_max,
        call_liquid_count=call_liquid_count,
        put_mean_delta=put_mean,
        put_median_delta=put_median,
        put_std_delta=put_std,
        put_min_delta=put_min,
        put_max_delta=put_max,
        put_liquid_count=put_liquid_count,
        spot_price=spot,
        call_otm_ratio=call_otm_ratio,
        put_otm_ratio=put_otm_ratio,
        iv_skew=iv_skew,
    )


class Advisor:
    """Market advisor: analyzes chain, builds feature vector, calls Claude."""

    def __init__(self, config: Optional[AdviceConfig] = None):
        """Initialize advisor.

        Args:
            config: AdviceConfig (or None for defaults)
        """
        self.config = config or AdviceConfig()

    def get_advice(
        self,
        symbol: str,
        quote: Quote,
        chain: list[OptionContract],
        as_of: date,
        shares_held: int = 0,
        cash_available: float = 0.0,
        cost_basis: float | None = None,
        force_refresh: bool = False,
    ) -> dict:
        """Get advisory recommendation for a symbol.

        Analyzes chain, builds feature vector, calls advisor if enabled.
        Returns dict with put_delta_target, call_delta_target, dte_target, rationale, etc.

        Args:
            symbol: ticker symbol
            quote: current quote
            chain: list of option contracts
            as_of: reference date
            shares_held: number of shares currently held
            cash_available: cash available for new positions
            cost_basis: average cost per share (for covered call floor)
            force_refresh: call Claude even if cache hit

        Returns:
            {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": "...",
                "advice_id": "...",
                "from_cache": bool,
                "chain_stats": ChainStats dict,
                "cost_usd": float
            }
        """
        # Step 1: Analyze chain
        try:
            chain_stats = analyze_chain(chain, quote, as_of, self.config.min_oi, self.config.max_spread_pct)
        except Exception as e:
            logger.exception(f"chain analysis failed for {symbol}: {e}")
            chain_stats = None

        # Step 2: If advisor is disabled, return defaults
        if not self.config.is_enabled():
            return {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": "advisor mode is off",
                "advice_id": None,
                "from_cache": False,
                "chain_stats": chain_stats.to_dict() if chain_stats else None,
                "cost_usd": 0.0,
            }

        # Step 3: Build feature vector
        try:
            fv = build_feature_vector(
                symbol=symbol,
                price=quote.price,
                iv_30d=quote.iv,
                iv_60d=quote.iv,  # single quote doesn't have term structure; use same
                iv_52w_low=quote.iv * 0.7,  # dummy; could be enriched
                iv_52w_high=quote.iv * 1.3,  # dummy
                rv_20d=quote.iv * 0.95,  # dummy
                rv_60d=quote.iv,  # dummy
                skew_put_call=chain_stats.iv_skew - quote.iv if chain_stats else 0.0,
                shares_held=shares_held,
                csp_open_count=0,  # not known; caller can override if needed
                ccall_open_count=0,
                avg_cost_per_share=cost_basis or quote.price,
                cash_available=cash_available,
                collateral_used_pct=0.0,  # not known
                days_in_position_avg=0.0,
                dte_to_next_earnings=None,  # not known
                change_1d_pct=0.0,  # not known
                change_5d_pct=0.0,
                change_30d_pct=0.0,
                atr_20d=quote.iv * 0.25,  # dummy proxy
            )
        except Exception as e:
            logger.exception(f"feature vector build failed for {symbol}: {e}")
            return {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": f"feature vector build failed: {e}",
                "advice_id": None,
                "from_cache": False,
                "chain_stats": chain_stats.to_dict() if chain_stats else None,
                "cost_usd": 0.0,
            }

        # Step 4: Call advisor
        try:
            advisor_instance = self.config.lazy_advisor()
            if not advisor_instance:
                return {
                    "put_delta_target": 0.25,
                    "call_delta_target": 0.30,
                    "dte_target": 35,
                    "rationale": "advisor not initialized",
                    "advice_id": None,
                    "from_cache": False,
                    "chain_stats": chain_stats.to_dict() if chain_stats else None,
                    "cost_usd": 0.0,
                }

            result = advisor_instance.advise(fv, force_refresh=force_refresh)
            result["chain_stats"] = chain_stats.to_dict() if chain_stats else None
            return result

        except Exception as e:
            logger.exception(f"advisor call failed for {symbol}: {e}")
            return {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": f"advisor call failed: {e}",
                "advice_id": None,
                "from_cache": False,
                "chain_stats": chain_stats.to_dict() if chain_stats else None,
                "cost_usd": 0.0,
            }


# Extensions for ChainStats
def _chain_stats_to_dict(self: ChainStats) -> dict:
    """Convert ChainStats to dict (for JSON serialization)."""
    return {
        "call_mean_delta": round(self.call_mean_delta, 4),
        "call_median_delta": round(self.call_median_delta, 4),
        "call_std_delta": round(self.call_std_delta, 4),
        "call_min_delta": round(self.call_min_delta, 4),
        "call_max_delta": round(self.call_max_delta, 4),
        "call_liquid_count": self.call_liquid_count,
        "put_mean_delta": round(self.put_mean_delta, 4),
        "put_median_delta": round(self.put_median_delta, 4),
        "put_std_delta": round(self.put_std_delta, 4),
        "put_min_delta": round(self.put_min_delta, 4),
        "put_max_delta": round(self.put_max_delta, 4),
        "put_liquid_count": self.put_liquid_count,
        "spot_price": round(self.spot_price, 2),
        "call_otm_ratio": round(self.call_otm_ratio, 3),
        "put_otm_ratio": round(self.put_otm_ratio, 3),
        "iv_skew": round(self.iv_skew, 4),
    }


ChainStats.to_dict = _chain_stats_to_dict


__all__ = ["ChainStats", "AdviceConfig", "Advisor", "analyze_chain"]
