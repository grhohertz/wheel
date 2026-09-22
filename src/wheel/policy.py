"""Policy layer: seam functions for advisory integration.

Two functions that the core engine calls instead of hardcoded constants:
- get_delta_target(symbol, leg, default=0.30): returns optimal delta
- get_mc_overlay(symbol): returns vol/drift adjustments for MC engine

Both read from advisory cache with fallback to defaults.
All failures are non-fatal: exceptions log and return defaults.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

from wheel.audit import CurrentAdvisory
from wheel.client import ClaudeAdvisor

logger = logging.getLogger(__name__)

# Guardrail constants (immutable policy)
DELTA_FLOOR, DELTA_CEIL = 0.15, 0.35
DTE_FLOOR, DTE_CEIL = 30, 40
VOL_MULT_FLOOR, VOL_MULT_CEIL = 0.80, 1.30
DRIFT_FLOOR, DRIFT_CEIL = -0.10, 0.20


class MCOverlay(NamedTuple):
    """Monte Carlo parameter adjustments from Claude."""

    vol_multiplier: float  # apply to sigma: sigma_eff = sigma_iv * vol_multiplier
    drift_annual: float  # annual drift adjustment (decimal, e.g., 0.04)
    enabled: bool  # False = use identity, don't adjust MC

    @staticmethod
    def identity() -> MCOverlay:
        """No adjustment."""
        return MCOverlay(vol_multiplier=1.0, drift_annual=0.0, enabled=False)

    def apply_to_vol(self, sigma_iv: float) -> float:
        """Apply vol overlay to IV-implied sigma."""
        if not self.enabled:
            return sigma_iv
        adjusted = sigma_iv * self.vol_multiplier
        # Clamp to reasonable bounds
        return max(0.05, min(1.0, adjusted))

    def apply_to_drift(self, base_drift: float) -> float:
        """Apply drift overlay to base annual drift."""
        if not self.enabled:
            return base_drift
        adjusted = base_drift + self.drift_annual
        return max(DRIFT_FLOOR, min(DRIFT_CEIL, adjusted))


def get_delta_target(
    symbol: str,
    leg: str = "csp",
    default: float = 0.25,
    advisor: Optional[ClaudeAdvisor] = None,
    current_cache: Optional[CurrentAdvisory] = None,
    advisor_mode: Optional[str] = None,
) -> float:
    """Get optimal delta target for the next wheel leg.

    Policy: read from advisory cache. Clamp and fallback on any error.

    Args:
        symbol: ticker (e.g., "AAPL")
        leg: "csp" (put) or "ccall" (covered call)
        default: fallback delta if cache miss or error
        advisor: ClaudeAdvisor instance (lazy-init if None)
        current_cache: CurrentAdvisory cache (lazy-init if None)
        advisor_mode: override ADVISOR_MODE env (for testing)

    Returns:
        Clamped delta target (0.15–0.35).
    """
    mode = advisor_mode or os.getenv("ADVISOR_MODE", "off")
    if mode == "off":
        return default

    try:
        if current_cache is None:
            current_cache = CurrentAdvisory()

        advice = current_cache.get(symbol)
        if advice is None:
            logger.debug(f"No cached advice for {symbol}, using default {default:.2f}")
            return default

        # Extract recommendation
        recommendation = advice.get("recommendation", {})
        if leg == "csp":
            target = recommendation.get("put_delta_target", default)
        else:
            target = recommendation.get("call_delta_target", default)

        # Clamp
        clamped = max(DELTA_FLOOR, min(DELTA_CEIL, target))
        if clamped != target:
            logger.info(f"{symbol}/{leg}: clamped delta {target:.2f} → {clamped:.2f}")

        return clamped

    except Exception as e:
        logger.exception(f"get_delta_target failed for {symbol}: {e}")
        return default


def get_mc_overlay(
    symbol: str,
    advisor: Optional[ClaudeAdvisor] = None,
    current_cache: Optional[CurrentAdvisory] = None,
    advisor_mode: Optional[str] = None,
) -> MCOverlay:
    """Get Monte Carlo parameter adjustments from advisory.

    Policy: read overlay params from cache. Return identity on any error.

    Args:
        symbol: ticker
        advisor: ClaudeAdvisor (unused in MVP, for future expansion)
        current_cache: CurrentAdvisory cache
        advisor_mode: override ADVISOR_MODE env

    Returns:
        MCOverlay with vol_multiplier and drift adjustments.
    """
    mode = advisor_mode or os.getenv("ADVISOR_MODE", "off")
    if mode not in ("shadow", "overlay"):
        return MCOverlay.identity()

    try:
        if current_cache is None:
            current_cache = CurrentAdvisory()

        advice = current_cache.get(symbol)
        if advice is None:
            return MCOverlay.identity()

        # In MVP, we don't have overlay params in the recommendation yet.
        # For now, always return identity. Phase 4 will add regime_advisor
        # that emits vol_multiplier, drift_annual, etc.
        return MCOverlay.identity()

    except Exception as e:
        logger.exception(f"get_mc_overlay failed for {symbol}: {e}")
        return MCOverlay.identity()


# Example usage / validation
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test delta target lookup
    delta = get_delta_target("AAPL", leg="csp", default=0.25, advisor_mode="off")
    print(f"Delta (advisor_mode=off): {delta:.2f}")

    delta_shadow = get_delta_target("AAPL", leg="csp", default=0.25, advisor_mode="shadow")
    print(f"Delta (advisor_mode=shadow): {delta_shadow:.2f}")

    # Test MC overlay
    overlay = get_mc_overlay("AAPL", advisor_mode="off")
    print(f"MC Overlay (off): {overlay}")

    overlay_shadow = get_mc_overlay("AAPL", advisor_mode="shadow")
    print(f"MC Overlay (shadow): {overlay_shadow}")

    # Test clamping
    print("\nClamping tests:")
    print(f"  0.10 → {get_delta_target('TEST', leg='csp', default=0.10, advisor_mode='off'):.2f} (should be 0.10)")
    print(f"  0.40 → {get_delta_target('TEST', leg='csp', default=0.40, advisor_mode='off'):.2f} (should be 0.40)")
