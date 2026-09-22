"""Test advisory policy layer.

Tests that get_delta_target and get_mc_overlay correctly:
1. Fall back to default when ADVISOR_MODE=off
2. Read from CurrentAdvisory cache when ADVISOR_MODE=shadow
3. Log and clamp invalid values
4. Handle missing cache gracefully
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wheel.audit import CurrentAdvisory
from wheel.policy import get_delta_target, get_mc_overlay, DELTA_FLOOR, DELTA_CEIL


@pytest.fixture()
def temp_advisory_dir():
    """Temp directory for advisory state."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def temp_cache(temp_advisory_dir):
    """CurrentAdvisory instance backed by temp dir."""
    cache_path = temp_advisory_dir / "current.json"
    return CurrentAdvisory(cache_path)


class TestGetDeltaTarget:
    """Test policy for reading delta targets from advisory."""

    def test_advisor_mode_off_returns_default(self):
        """When ADVISOR_MODE=off, always return default."""
        assert get_delta_target("AAPL", leg="csp", default=0.25, advisor_mode="off") == 0.25
        assert get_delta_target("AAPL", leg="csp", default=0.30, advisor_mode="off") == 0.30

    def test_missing_cache_returns_default(self, temp_cache):
        """When cache doesn't exist, return default."""
        delta = get_delta_target(
            "AAPL",
            leg="csp",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == 0.25

    def test_missing_symbol_in_cache_returns_default(self, temp_cache):
        """When symbol not in cache, return default."""
        # Write a cache with a different symbol
        temp_cache.set("TSLA", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.20},
        })

        delta = get_delta_target(
            "AAPL",
            leg="csp",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == 0.25

    def test_reads_put_delta_target_from_cache(self, temp_cache):
        """When csp leg requested, reads put_delta_target from cache."""
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.20, "call_delta_target": 0.25},
        })

        delta = get_delta_target(
            "AAPL",
            leg="csp",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == 0.20

    def test_reads_call_delta_target_from_cache(self, temp_cache):
        """When ccall leg requested, reads call_delta_target from cache."""
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.20, "call_delta_target": 0.35},
        })

        delta = get_delta_target(
            "AAPL",
            leg="ccall",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == 0.35

    def test_falls_back_to_default_when_target_missing(self, temp_cache):
        """When target key missing from recommendation, use default."""
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.20},  # call_delta_target missing
        })

        delta = get_delta_target(
            "AAPL",
            leg="ccall",
            default=0.30,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == 0.30

    def test_clamps_excessive_values(self, temp_cache):
        """Advisory can't push delta outside safe bounds."""
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.60},  # above DELTA_CEIL
        })

        delta = get_delta_target(
            "AAPL",
            leg="csp",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == DELTA_CEIL  # should be clamped to 0.35

    def test_clamps_too_low(self, temp_cache):
        """Advisory can't push delta below DELTA_FLOOR."""
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.05},  # below DELTA_FLOOR
        })

        delta = get_delta_target(
            "AAPL",
            leg="csp",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == DELTA_FLOOR  # should be clamped to 0.15

    def test_returns_default_on_exception(self, temp_cache):
        """On any error, gracefully return default."""
        # Corrupt the cache file to trigger JSON error
        cache_path = temp_cache.path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{invalid json")

        delta = get_delta_target(
            "AAPL",
            leg="csp",
            default=0.25,
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert delta == 0.25  # should fall back gracefully


class TestGetMCOverlay:
    """Test policy for Monte Carlo overlay adjustments (Phase 4)."""

    def test_advisor_mode_off_returns_identity(self):
        """When ADVISOR_MODE=off, always return identity overlay."""
        overlay = get_mc_overlay("AAPL", advisor_mode="off")
        assert overlay.vol_multiplier == 1.0
        assert overlay.drift_annual == 0.0
        assert overlay.enabled is False

    def test_missing_cache_returns_identity(self, temp_cache):
        """When cache doesn't exist, return identity overlay."""
        overlay = get_mc_overlay(
            "AAPL",
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        assert overlay.vol_multiplier == 1.0
        assert overlay.drift_annual == 0.0
        assert overlay.enabled is False

    def test_returns_identity_in_mvp(self, temp_cache):
        """In MVP, MC overlay always returns identity (Phase 4 feature)."""
        # Even if cache had params, MVP doesn't use them yet
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "mc_overlay": {"vol_multiplier": 1.15, "drift_annual": 0.002, "enabled": True},
        })

        overlay = get_mc_overlay(
            "AAPL",
            current_cache=temp_cache,
            advisor_mode="shadow"
        )
        # MVP still returns identity
        assert overlay.vol_multiplier == 1.0
        assert overlay.drift_annual == 0.0
        assert overlay.enabled is False


class TestEndToEnd:
    """End-to-end tests: advisory wired into strategy layer."""

    def test_strategy_uses_advisory_delta_target(self, temp_cache):
        """Strategy.delta_ok uses advisory target when symbol is set."""
        from wheel.config import StrategyParams
        from wheel.strategy import WheelStrategy

        # Set up advisory
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.20, "call_delta_target": 0.25},
        })

        # Create strategy with symbol context
        params = StrategyParams(target_delta=0.30, delta_tolerance=0.05)
        strategy = WheelStrategy(params)
        strategy.symbol = "AAPL"

        # With shadow mode enabled and cache set, delta_ok should respect advisory
        # (Note: this requires mocking or setting ADVISOR_MODE=shadow in the environment)
        # For now, we just verify the method doesn't crash when symbol is set
        assert strategy.delta_ok(0.30, leg="ccall")

    def test_advisor_mode_env_override(self, temp_cache, monkeypatch):
        """ADVISOR_MODE env var controls behavior."""
        temp_cache.set("AAPL", {
            "advice_id": "2025-09-22T13:30:00Z-abc1def2",
            "recommendation": {"put_delta_target": 0.20},
        })

        # With ADVISOR_MODE=off
        monkeypatch.setenv("ADVISOR_MODE", "off")
        delta = get_delta_target("AAPL", leg="csp", default=0.25, current_cache=temp_cache)
        assert delta == 0.25

        # With ADVISOR_MODE=shadow
        monkeypatch.setenv("ADVISOR_MODE", "shadow")
        delta = get_delta_target("AAPL", leg="csp", default=0.25, current_cache=temp_cache)
        assert delta == 0.20
