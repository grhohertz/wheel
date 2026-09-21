import math

import pytest
from wheel.greeks import black_scholes, implied_vol, norm_cdf, year_fraction


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_put_call_parity():
    s, k, t, r, sig, q = 100.0, 95.0, 0.5, 0.04, 0.30, 0.01
    c = black_scholes(s, k, t, r, sig, q, "C").price
    p = black_scholes(s, k, t, r, sig, q, "P").price
    lhs = c - p
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_delta_bounds_and_atm():
    atm_call = black_scholes(100, 100, 0.25, 0.04, 0.30, 0.0, "C")
    atm_put = black_scholes(100, 100, 0.25, 0.04, 0.30, 0.0, "P")
    assert 0.0 < atm_call.delta < 1.0
    assert -1.0 < atm_put.delta < 0.0
    assert atm_call.delta == pytest.approx(0.55, abs=0.06)
    # Same strike/expiry: call delta - put delta == exp(-qT) == 1 with q=0.
    assert atm_call.delta - atm_put.delta == pytest.approx(1.0, abs=1e-9)


def test_gamma_vega_positive_theta_negative_for_long():
    g = black_scholes(50, 52, 0.1, 0.04, 0.4, 0.0, "C")
    assert g.gamma > 0
    assert g.vega > 0
    assert g.theta < 0  # long options decay


def test_deep_otm_is_cheap_and_low_delta():
    g = black_scholes(50, 90, 0.08, 0.04, 0.25, 0.0, "C")
    assert g.price < 0.05
    assert g.delta < 0.05


def test_expired_returns_intrinsic():
    itm = black_scholes(60, 50, 0.0, 0.04, 0.3, 0.0, "C")
    assert itm.price == pytest.approx(10.0)
    assert itm.delta == pytest.approx(1.0)
    otm = black_scholes(40, 50, 0.0, 0.04, 0.3, 0.0, "C")
    assert otm.price == 0.0
    assert otm.delta == 0.0
    put = black_scholes(40, 50, 0.0, 0.04, 0.3, 0.0, "P")
    assert put.price == pytest.approx(10.0)
    assert put.delta == pytest.approx(-1.0)


def test_implied_vol_round_trip():
    for sigma in (0.12, 0.35, 0.80):
        price = black_scholes(100, 105, 0.35, 0.04, sigma, 0.01, "C").price
        assert implied_vol(price, 100, 105, 0.35, 0.04, 0.01, "C") == pytest.approx(sigma, abs=1e-3)


def test_implied_vol_rejects_sub_intrinsic():
    with pytest.raises(ValueError):
        implied_vol(0.5, 60, 50, 0.25, 0.04, 0.0, "C")


def test_bad_inputs():
    with pytest.raises(ValueError):
        black_scholes(100, 100, 0.25, 0.04, 0.3, 0.0, "X")
    with pytest.raises(ValueError):
        black_scholes(-1, 100, 0.25, 0.04, 0.3, 0.0, "C")


def test_scaled_greeks_flip_for_shorts():
    g = black_scholes(50, 53, 0.1, 0.04, 0.35, 0.0, "C")
    short = g.scaled(-1, 100)
    assert short.delta == pytest.approx(-g.delta * 100)
    assert short.theta > 0  # short options earn decay


def test_year_fraction():
    assert year_fraction(365) == pytest.approx(1.0)
    assert year_fraction(-5) == 0.0
