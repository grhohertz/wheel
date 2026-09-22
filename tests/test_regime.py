"""Tests for Phase 4 — the regime advisor (:mod:`wheel.regime`).

Three things must hold:

1. Overlays are **bounded** — no advisory, however hostile, produces a
   degenerate simulation.
2. Overlays are **pure** — applying one returns new params, leaves the input
   alone, and keeps the run reproducible.
3. Overlays are **directionally correct** — a vol spike must widen the
   distribution and hurt; a vol crush must narrow it and help.
"""

from __future__ import annotations

import pytest

from wheel.config import StrategyParams
from wheel.monte_carlo import GLD_DRIFT, GLD_VOL, MonteCarloParams, run_monte_carlo
from wheel.regime import (
    DELTA_SHIFT_CLAMP,
    IV_MULT_CLAMP,
    MU_SHIFT_CLAMP,
    NORMAL,
    REGIME_OVERLAYS,
    REGIMES,
    RegimeOverlay,
    VOL_MULT_CLAMP,
    compare_regimes,
    infer_regime,
    overlay_for,
    overlay_from_iv,
    regime_scenarios,
    run_regime,
    run_with_overlay,
)


def base_params(**kw) -> MonteCarloParams:
    """A small, fast simulation — enough paths to be stable, few enough to be quick."""

    defaults = dict(
        symbol="GLD", shares=800, paths=40, days=252, spot=310.0,
        mu=GLD_DRIFT, sigma=GLD_VOL, seed=7,
    )
    defaults.update(kw)
    return MonteCarloParams(**defaults)


# ----------------------------------------------------------------------
# construction + guardrails
# ----------------------------------------------------------------------
def test_identity_overlay_is_identity():
    o = RegimeOverlay.identity()
    assert o.is_identity
    assert o.regime == NORMAL
    assert o.source == "identity"


def test_every_regime_label_has_an_overlay():
    for label in REGIMES:
        assert label in REGIME_OVERLAYS
        assert REGIME_OVERLAYS[label].regime == label


def test_every_table_overlay_carries_a_rationale():
    for label, overlay in REGIME_OVERLAYS.items():
        assert overlay.rationale, f"{label} has no rationale"


def test_vol_mult_clamped_high():
    assert RegimeOverlay(vol_mult=100.0).vol_mult == pytest.approx(VOL_MULT_CLAMP[1])


def test_vol_mult_clamped_low():
    assert RegimeOverlay(vol_mult=-5.0).vol_mult == pytest.approx(VOL_MULT_CLAMP[0])


def test_iv_mult_clamped():
    assert RegimeOverlay(iv_mult=50.0).iv_mult == pytest.approx(IV_MULT_CLAMP[1])


def test_mu_shift_clamped_both_ways():
    assert RegimeOverlay(mu_shift=9.0).mu_shift == pytest.approx(MU_SHIFT_CLAMP[1])
    assert RegimeOverlay(mu_shift=-9.0).mu_shift == pytest.approx(MU_SHIFT_CLAMP[0])


def test_delta_shift_clamped():
    assert RegimeOverlay(delta_shift=1.0).delta_shift == pytest.approx(DELTA_SHIFT_CLAMP[1])


def test_dte_shift_is_clamped_and_integral():
    o = RegimeOverlay(dte_shift=99)
    assert o.dte_shift == 14 and isinstance(o.dte_shift, int)


def test_overlay_is_frozen():
    with pytest.raises(Exception):
        RegimeOverlay().vol_mult = 2.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# serialization
# ----------------------------------------------------------------------
def test_to_dict_round_trips_through_from_dict():
    original = REGIME_OVERLAYS["vol_spike"]
    restored = RegimeOverlay.from_dict(original.to_dict())
    assert restored.regime == original.regime
    assert restored.vol_mult == pytest.approx(original.vol_mult)
    assert restored.iv_mult == pytest.approx(original.iv_mult)
    assert restored.mu_shift == pytest.approx(original.mu_shift)


def test_from_dict_tolerates_garbage():
    o = RegimeOverlay.from_dict({"vol_mult": "lots", "regime": 42, "mu_shift": None})
    assert o.vol_mult == 1.0
    assert o.regime == NORMAL
    assert o.mu_shift == 0.0


def test_from_dict_on_empty_dict_is_identity():
    assert RegimeOverlay.from_dict({}).is_identity


def test_to_dict_is_json_safe():
    import json

    json.dumps(REGIME_OVERLAYS["high_iv"].to_dict())


# ----------------------------------------------------------------------
# lookup + inference
# ----------------------------------------------------------------------
def test_overlay_for_unknown_regime_is_normal():
    assert overlay_for("nonsense").regime == NORMAL


def test_overlay_for_is_case_and_space_insensitive():
    assert overlay_for("  VOL_SPIKE ").regime == "vol_spike"


def test_overlay_for_stamps_source():
    assert overlay_for("high_iv", source="advisory").source == "advisory"


def test_infer_regime_high_iv_at_top_of_range():
    assert infer_regime(0.44, 0.15, 0.45) in ("high_iv", "vol_crush")


def test_infer_regime_low_iv_at_bottom_of_range():
    assert infer_regime(0.16, 0.15, 0.45) in ("low_iv", "vol_spike")


def test_infer_regime_midrange_is_normal():
    assert infer_regime(0.30, 0.15, 0.45) == NORMAL


def test_overlay_from_iv_marks_source_inferred():
    assert overlay_from_iv(0.44, 0.15, 0.45).source == "inferred"


# ----------------------------------------------------------------------
# apply()
# ----------------------------------------------------------------------
def test_apply_does_not_mutate_the_input():
    p = base_params()
    REGIME_OVERLAYS["vol_spike"].apply(p)
    assert p.sigma == pytest.approx(GLD_VOL)
    assert p.overlay is None


def test_apply_scales_sigma_by_vol_mult():
    p = base_params()
    out = REGIME_OVERLAYS["vol_spike"].apply(p)
    assert out.sigma == pytest.approx(GLD_VOL * REGIME_OVERLAYS["vol_spike"].vol_mult)


def test_apply_scales_option_iv_by_iv_mult():
    p = base_params()
    o = REGIME_OVERLAYS["high_iv"]
    out = o.apply(p)
    assert out.option_iv == pytest.approx(p.option_iv * o.iv_mult)


def test_apply_shifts_drift():
    p = base_params()
    out = REGIME_OVERLAYS["vol_spike"].apply(p)
    assert out.mu == pytest.approx(GLD_DRIFT + REGIME_OVERLAYS["vol_spike"].mu_shift)


def test_apply_shifts_the_delta_target():
    p = base_params(strategy=StrategyParams(target_delta=0.25))
    out = REGIME_OVERLAYS["vol_spike"].apply(p)
    assert out.strategy.target_delta < 0.25


def test_apply_shifts_entry_dte():
    p = base_params(entry_dte=35, close_dte=21)
    out = REGIME_OVERLAYS["low_iv"].apply(p)
    assert out.entry_dte == 35 + REGIME_OVERLAYS["low_iv"].dte_shift


def test_apply_stamps_the_overlay_on_the_result():
    out = REGIME_OVERLAYS["vol_crush"].apply(base_params())
    assert out.overlay is not None and out.overlay.regime == "vol_crush"


def test_apply_keeps_entry_dte_above_close_dte():
    p = base_params(entry_dte=25, close_dte=21)
    out = RegimeOverlay(dte_shift=-14).apply(p)
    assert out.entry_dte > out.close_dte


def test_apply_identity_leaves_the_economics_alone():
    p = base_params()
    out = RegimeOverlay.identity().apply(p)
    assert out.sigma == pytest.approx(p.sigma)
    assert out.mu == pytest.approx(p.mu)
    assert out.option_iv == pytest.approx(p.option_iv)
    assert out.strategy.target_delta == pytest.approx(p.strategy.target_delta)


def test_apply_result_is_still_a_valid_params_object():
    for label in REGIMES:
        out = REGIME_OVERLAYS[label].apply(base_params())
        assert out.sigma > 0 and out.option_iv > 0
        assert out.close_dte < out.entry_dte


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------
def test_run_with_overlay_is_deterministic():
    p = base_params()
    a = run_with_overlay(p, REGIME_OVERLAYS["high_iv"]).to_dict()
    b = run_with_overlay(p, REGIME_OVERLAYS["high_iv"]).to_dict()
    assert a == b


def test_run_with_no_overlay_matches_a_plain_run():
    p = base_params()
    assert (
        run_with_overlay(p).metrics["total_pnl"]["mean"]
        == run_monte_carlo(p).metrics["total_pnl"]["mean"]
    )


def test_summary_config_reports_the_regime():
    d = run_regime(base_params(), "vol_crush").to_dict()
    assert d["config"]["regime"] == "vol_crush"
    assert d["overlay"]["source"] == "table"


def test_plain_run_reports_no_regime():
    d = run_monte_carlo(base_params()).to_dict()
    assert d["config"]["regime"] is None
    assert d["overlay"] is None


# ----------------------------------------------------------------------
# scenarios — the Phase 4 payoff
# ----------------------------------------------------------------------
def test_regime_scenarios_covers_every_regime():
    out = regime_scenarios(base_params())
    assert set(out) == set(REGIMES)


def test_regime_scenarios_honours_an_explicit_subset():
    out = regime_scenarios(base_params(), ["normal", "vol_spike"])
    assert set(out) == {"normal", "vol_spike"}


def test_vol_spike_widens_the_distribution_versus_low_iv():
    out = regime_scenarios(base_params(), ["low_iv", "vol_spike"])
    quiet = out["low_iv"].metrics["total_pnl"]["std"]
    loud = out["vol_spike"].metrics["total_pnl"]["std"]
    assert loud > quiet


def test_vol_spike_drawdown_exceeds_low_iv_drawdown():
    out = regime_scenarios(base_params(), ["low_iv", "vol_spike"])
    assert (
        out["vol_spike"].metrics["max_drawdown_pct"]["mean"]
        > out["low_iv"].metrics["max_drawdown_pct"]["mean"]
    )


def test_higher_iv_regimes_collect_more_premium():
    out = regime_scenarios(base_params(), ["low_iv", "high_iv"])
    assert (
        out["high_iv"].metrics["premium_collected"]["mean"]
        > out["low_iv"].metrics["premium_collected"]["mean"]
    )


def test_normal_regime_matches_the_unconditioned_baseline():
    p = base_params()
    assert run_regime(p, NORMAL).metrics["total_pnl"]["mean"] == pytest.approx(
        run_monte_carlo(p).metrics["total_pnl"]["mean"]
    )


# ----------------------------------------------------------------------
# comparison table
# ----------------------------------------------------------------------
def test_compare_regimes_shape():
    out = compare_regimes(base_params(), ["normal", "vol_spike"])
    assert out["baseline"]["symbol"] == "GLD"
    assert [r["regime"] for r in out["rows"]] == ["normal", "vol_spike"]
    row = out["rows"][1]
    assert set(row) == {"regime", "overlay", "inputs", "outcome"}
    assert "prob_profit" in row["outcome"]


def test_compare_regimes_is_json_serializable():
    import json

    json.dumps(compare_regimes(base_params(), ["normal"]))


def test_compare_regimes_inputs_reflect_the_overlay():
    out = compare_regimes(base_params(), ["vol_spike"])
    row = out["rows"][0]
    assert row["inputs"]["sigma"] > out["baseline"]["sigma"]
