"""Monte Carlo simulator: determinism, GBM sanity, and wheel accounting."""

from __future__ import annotations

import math
import random

import pytest

from wheel.greeks import CALL, PUT, black_scholes
from wheel.monte_carlo import (
    MonteCarloParams,
    delta_strike_ratio,
    gbm_path,
    percentile,
    run_monte_carlo,
    simulate_path,
    snap_strike,
)


def _params(**kw) -> MonteCarloParams:
    base = dict(symbol="GLD", shares=800, paths=20, days=60, spot=310.0, seed=7)
    base.update(kw)
    return MonteCarloParams(**base)


# -- parameter validation ------------------------------------------------
@pytest.mark.parametrize(
    "kw",
    [
        {"paths": 0},
        {"days": 0},
        {"spot": 0.0},
        {"sigma": 0.0},
        {"shares": -100},
        {"close_dte": 40, "entry_dte": 35},
    ],
)
def test_invalid_params_raise(kw):
    with pytest.raises(ValueError):
        _params(**kw)


def test_option_iv_defaults_above_realised_vol():
    p = _params(sigma=0.12)
    assert p.option_iv > p.sigma
    assert _params(iv=0.25).option_iv == 0.25


# -- GBM -----------------------------------------------------------------
def test_gbm_path_length_and_positivity():
    path = gbm_path(random.Random(1), 310.0, 0.08, 0.12, 0.0, 252)
    assert len(path) == 253  # inclusive of the starting spot
    assert path[0] == 310.0
    assert all(p > 0 for p in path)


def test_gbm_drift_and_vol_match_inputs():
    """Mean log-return and its stdev should recover (mu - s^2/2)*dt and s*sqrt(dt)."""

    rng = random.Random(42)
    dt = 1.0 / 252
    mu, sigma = 0.08, 0.12
    logs = []
    for _ in range(400):
        path = gbm_path(rng, 100.0, mu, sigma, 0.0, 252)
        logs += [math.log(path[i] / path[i - 1]) for i in range(1, len(path))]
    mean = sum(logs) / len(logs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
    assert mean == pytest.approx((mu - 0.5 * sigma**2) * dt, abs=2e-5)
    assert sd == pytest.approx(sigma * math.sqrt(dt), rel=0.05)


# -- strike selection ----------------------------------------------------
@pytest.mark.parametrize("right", [CALL, PUT])
def test_delta_strike_ratio_hits_the_target(right):
    t, iv = 35 / 365.0, 0.15
    ratio = delta_strike_ratio(0.30, t, iv, 0.04, 0.0, right)
    delta = abs(black_scholes(1.0, ratio, t, 0.04, iv, 0.0, right).delta)
    assert delta == pytest.approx(0.30, abs=1e-3)
    assert (ratio > 1.0) if right == CALL else (ratio < 1.0)


def test_snap_strike_rounds_away_from_the_money():
    # strike_increment(310) == 10.0
    assert snap_strike(311.3, 310.0, CALL) == 320.0  # rounded up, further OTM
    assert snap_strike(298.7, 310.0, PUT) == 290.0  # rounded down, further OTM
    # strike_increment(60) == 2.5
    assert snap_strike(61.2, 60.0, CALL) == 62.5
    assert snap_strike(58.8, 60.0, PUT) == 57.5


# -- percentiles ---------------------------------------------------------
def test_percentile_interpolates():
    v = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert percentile(v, 0.50) == 20.0
    assert percentile(v, 0.10) == pytest.approx(4.0)
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0], 0.9) == 3.0


# -- single path ---------------------------------------------------------
def test_simulate_path_flat_market_collects_premium():
    """Zero-vol, zero-drift path: the calls expire worthless and we keep the credit."""

    p = _params(days=252, sigma=0.12, iv=0.15)
    flat = [310.0] * (p.days + 1)
    r = simulate_path(flat, p)
    assert r.cycles > 0
    assert r.premium_collected > 0
    assert r.total_pnl > 0  # premium with no adverse move
    assert r.max_drawdown_pct >= 0.0


def test_simulate_path_rally_triggers_call_away():
    """A steady rally through the short call must end in a call-away."""

    p = _params(days=252)
    rally = [310.0 * (1.0 + 0.004) ** i for i in range(p.days + 1)]
    r = simulate_path(rally, p)
    assert r.called_away >= 1
    assert r.final_nav > 0


def test_simulate_path_records_a_nav_for_every_day():
    p = _params(days=45)
    r = simulate_path([310.0] * (p.days + 1), p)
    assert r.final_spot == 310.0
    assert r.return_pct == pytest.approx(r.total_pnl / p.initial_nav)


# -- aggregation ---------------------------------------------------------
def test_run_monte_carlo_is_deterministic_for_a_seed():
    a = run_monte_carlo(_params()).to_dict()
    b = run_monte_carlo(_params()).to_dict()
    assert a == b
    assert run_monte_carlo(_params(seed=8)).to_dict() != a


def test_summary_shape_and_percentile_ordering():
    s = run_monte_carlo(_params(paths=50)).to_dict()
    assert s["config"]["symbol"] == "GLD"
    assert s["config"]["shares"] == 800
    assert s["headline"]["paths"] == 50
    assert 0.0 <= s["headline"]["prob_profit"] <= 1.0
    for key, d in s["metrics"].items():
        assert d["p10"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p90"], key
        assert d["min"] <= d["p10"] and d["p90"] <= d["max"], key


def test_zero_shares_starts_on_the_put_side_without_cash():
    """No shares and no cash: nothing to sell, but the run must not explode."""

    s = run_monte_carlo(_params(shares=0, paths=3)).to_dict()
    assert s["metrics"]["total_pnl"]["mean"] == 0.0
