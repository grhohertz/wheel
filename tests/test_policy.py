"""Tests for the advisory seam layer (:mod:`wheel.policy`).

The contract under test is "the advisor can never break or surprise the
engine": off by default, clamped when on, and total in the face of missing or
malformed state.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from wheel import policy
from wheel.policy import (
    OFF,
    OVERLAY,
    SHADOW,
    AdviceConfig,
    advisor_mode,
    clamp,
    get_delta_target,
    get_dte_target,
    get_mc_overlay,
    load_advice,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    policy.clear_cache()
    yield
    policy.clear_cache()


def write_advisory(tmp_path, symbol="AAPL", ttl_hours=24, **rec):
    """Write a current.json in the shape CurrentAdvisory persists."""

    recommendation = {
        "put_delta_target": 0.20,
        "call_delta_target": 0.25,
        "dte_target": 35,
        "rationale": "test fixture",
    }
    recommendation.update(rec)
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                symbol: {
                    "advice_id": "test-1",
                    "ts": datetime.now().isoformat(),
                    "recommendation": recommendation,
                    "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
                }
            }
        )
    )
    return str(path)


def cfg_for(path, mode=OVERLAY):
    return AdviceConfig(mode=mode, advisory_path=path)


# ----------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------
def test_mode_defaults_to_off_with_empty_env():
    assert advisor_mode({}) == OFF


def test_unknown_mode_falls_back_to_off():
    assert advisor_mode({"ADVISOR_MODE": "yolo"}) == OFF


def test_mode_is_case_insensitive():
    assert advisor_mode({"ADVISOR_MODE": "OverLay"}) == OVERLAY


def test_off_mode_is_not_enabled():
    cfg = AdviceConfig(mode=OFF)
    assert not cfg.enabled and not cfg.authoritative


def test_shadow_reads_but_is_not_authoritative():
    cfg = AdviceConfig(mode=SHADOW)
    assert cfg.enabled and not cfg.authoritative


# ----------------------------------------------------------------------
# delta targets
# ----------------------------------------------------------------------
def test_off_mode_returns_default_without_touching_disk(tmp_path):
    path = write_advisory(tmp_path, put_delta_target=0.31)
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path, OFF)) == 0.25


def test_shadow_mode_returns_default(tmp_path):
    path = write_advisory(tmp_path, put_delta_target=0.31)
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path, SHADOW)) == 0.25


def test_overlay_mode_returns_advisory_value(tmp_path):
    path = write_advisory(tmp_path, put_delta_target=0.31)
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path)) == pytest.approx(0.31)


def test_call_leg_reads_the_call_key(tmp_path):
    path = write_advisory(tmp_path, call_delta_target=0.33)
    assert get_delta_target("AAPL", "ccall", 0.25, cfg_for(path)) == pytest.approx(0.33)


def test_symbol_lookup_is_case_insensitive(tmp_path):
    path = write_advisory(tmp_path, symbol="AAPL", put_delta_target=0.30)
    assert get_delta_target("aapl", "csp", 0.25, cfg_for(path)) == pytest.approx(0.30)


def test_unknown_symbol_falls_back(tmp_path):
    path = write_advisory(tmp_path, symbol="AAPL")
    assert get_delta_target("TSLA", "csp", 0.25, cfg_for(path)) == 0.25


def test_missing_file_falls_back(tmp_path):
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(str(tmp_path / "nope.json"))) == 0.25


def test_corrupt_json_falls_back(tmp_path):
    path = tmp_path / "current.json"
    path.write_text("{not json")
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(str(path))) == 0.25


def test_expired_advisory_falls_back(tmp_path):
    path = write_advisory(tmp_path, ttl_hours=-1, put_delta_target=0.31)
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path)) == 0.25


def test_non_numeric_value_falls_back(tmp_path):
    path = write_advisory(tmp_path, put_delta_target="aggressive")
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path)) == 0.25


# ----------------------------------------------------------------------
# guardrails
# ----------------------------------------------------------------------
def test_put_delta_clamped_high(tmp_path):
    path = write_advisory(tmp_path, put_delta_target=0.95)
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path)) == pytest.approx(0.35)


def test_put_delta_clamped_low(tmp_path):
    path = write_advisory(tmp_path, put_delta_target=0.01)
    assert get_delta_target("AAPL", "csp", 0.25, cfg_for(path)) == pytest.approx(0.15)


def test_call_delta_has_its_own_band(tmp_path):
    path = write_advisory(tmp_path, call_delta_target=0.02)
    assert get_delta_target("AAPL", "ccall", 0.25, cfg_for(path)) == pytest.approx(0.20)


def test_clamp_helper_is_inclusive():
    assert clamp(0.5, (0.0, 1.0)) == 0.5
    assert clamp(-1.0, (0.0, 1.0)) == 0.0
    assert clamp(9.0, (0.0, 1.0)) == 1.0


# ----------------------------------------------------------------------
# dte
# ----------------------------------------------------------------------
def test_dte_target_returned_in_overlay(tmp_path):
    path = write_advisory(tmp_path, dte_target=40)
    assert get_dte_target("AAPL", 35, cfg_for(path)) == 40


def test_dte_target_clamped(tmp_path):
    path = write_advisory(tmp_path, dte_target=400)
    assert get_dte_target("AAPL", 35, cfg_for(path)) == 45


def test_dte_default_in_shadow(tmp_path):
    path = write_advisory(tmp_path, dte_target=40)
    assert get_dte_target("AAPL", 35, cfg_for(path, SHADOW)) == 35


# ----------------------------------------------------------------------
# load_advice
# ----------------------------------------------------------------------
def test_load_advice_returns_recommendation(tmp_path):
    path = write_advisory(tmp_path)
    rec = load_advice("AAPL", cfg_for(path))
    assert rec is not None and rec["rationale"] == "test fixture"


def test_load_advice_is_none_when_off(tmp_path):
    path = write_advisory(tmp_path)
    assert load_advice("AAPL", cfg_for(path, OFF)) is None


# ----------------------------------------------------------------------
# phase 4 seam
# ----------------------------------------------------------------------
def test_mc_overlay_identity_without_advisory(tmp_path):
    overlay = get_mc_overlay("AAPL", cfg_for(str(tmp_path / "nope.json")))
    assert overlay.is_identity


def test_mc_overlay_from_regime_label(tmp_path):
    path = write_advisory(tmp_path, regime="vol_spike")
    overlay = get_mc_overlay("AAPL", cfg_for(path))
    assert overlay.regime == "vol_spike"
    assert overlay.vol_mult > 1.0
    assert overlay.source == "advisory"


def test_mc_overlay_from_explicit_block(tmp_path):
    path = write_advisory(
        tmp_path, mc_overlay={"regime": "custom", "vol_mult": 1.2, "mu_shift": -0.01}
    )
    overlay = get_mc_overlay("AAPL", cfg_for(path))
    assert overlay.regime == "custom"
    assert overlay.vol_mult == pytest.approx(1.2)
    assert overlay.mu_shift == pytest.approx(-0.01)


def test_mc_overlay_is_identity_in_shadow(tmp_path):
    path = write_advisory(tmp_path, regime="vol_spike")
    assert get_mc_overlay("AAPL", cfg_for(path, SHADOW)).is_identity


def test_mc_overlay_clamps_a_hostile_advisory(tmp_path):
    path = write_advisory(tmp_path, mc_overlay={"vol_mult": 99.0, "mu_shift": 5.0})
    overlay = get_mc_overlay("AAPL", cfg_for(path))
    assert overlay.vol_mult == pytest.approx(2.50)
    assert overlay.mu_shift == pytest.approx(0.15)
