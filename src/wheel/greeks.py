"""Black-Scholes-Merton pricing and greeks in pure stdlib Python.

No numpy/scipy dependency: the normal CDF comes from :func:`math.erf`.

Conventions
-----------
* ``T`` is time to expiry in YEARS.
* ``sigma`` and ``r``/``q`` are decimals (0.35 == 35%).
* ``theta`` is returned PER CALENDAR DAY (annual theta / 365).
* ``vega`` is returned per ONE VOLATILITY POINT (annual vega / 100).
* ``rho`` is returned per ONE PERCENT of rate (annual rho / 100).
* All values describe a LONG one-unit position; negate for shorts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DAYS_PER_YEAR = 365.0
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)

CALL = "C"
PUT = "P"


@dataclass(frozen=True)
class Greeks:
    """Theoretical value plus the first-order risk sensitivities."""

    price: float
    delta: float
    gamma: float
    theta: float  # per day
    vega: float  # per 1 vol point
    rho: float  # per 1% rate

    def scaled(self, quantity: int, multiplier: int = 100) -> "Greeks":
        """Position-level greeks. ``quantity`` is negative for short options."""

        f = quantity * multiplier
        return Greeks(
            price=self.price * f,
            delta=self.delta * f,
            gamma=self.gamma * f,
            theta=self.theta * f,
            vega=self.vega * f,
            rho=self.rho * f,
        )


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _normalise_right(right: str) -> str:
    r = (right or "").strip().upper()[:1]
    if r not in (CALL, PUT):
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    return r


def _intrinsic(spot: float, strike: float, right: str) -> float:
    return max(0.0, spot - strike) if right == CALL else max(0.0, strike - spot)


def d1_d2(
    spot: float, strike: float, t: float, rate: float, sigma: float, div_yield: float = 0.0
) -> tuple[float, float]:
    vt = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * sigma * sigma) * t) / vt
    return d1, d1 - vt


def black_scholes(
    spot: float,
    strike: float,
    t: float,
    rate: float = 0.04,
    sigma: float = 0.30,
    div_yield: float = 0.0,
    right: str = CALL,
) -> Greeks:
    """Price + greeks for one long contract-unit (per share)."""

    right = _normalise_right(right)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")

    # Degenerate cases: expired or zero-vol => intrinsic value, binary delta.
    if t <= 0.0 or sigma <= 0.0:
        intrinsic = _intrinsic(spot, strike, right)
        itm = intrinsic > 0.0
        delta = (1.0 if right == CALL else -1.0) if itm else 0.0
        return Greeks(price=intrinsic, delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    d1, d2 = d1_d2(spot, strike, t, rate, sigma, div_yield)
    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-div_yield * t)
    nd1, nd2 = norm_cdf(d1), norm_cdf(d2)
    pdf_d1 = norm_pdf(d1)
    sqrt_t = math.sqrt(t)

    gamma = disc_q * pdf_d1 / (spot * sigma * sqrt_t)
    vega_annual = spot * disc_q * pdf_d1 * sqrt_t
    common_theta = -(spot * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)

    if right == CALL:
        price = spot * disc_q * nd1 - strike * disc_r * nd2
        delta = disc_q * nd1
        theta_annual = common_theta - rate * strike * disc_r * nd2 + div_yield * spot * disc_q * nd1
        rho_annual = strike * t * disc_r * nd2
    else:
        price = strike * disc_r * norm_cdf(-d2) - spot * disc_q * norm_cdf(-d1)
        delta = disc_q * (nd1 - 1.0)
        theta_annual = (
            common_theta
            + rate * strike * disc_r * norm_cdf(-d2)
            - div_yield * spot * disc_q * norm_cdf(-d1)
        )
        rho_annual = -strike * t * disc_r * norm_cdf(-d2)

    return Greeks(
        price=max(price, 0.0),
        delta=delta,
        gamma=gamma,
        theta=theta_annual / DAYS_PER_YEAR,
        vega=vega_annual / 100.0,
        rho=rho_annual / 100.0,
    )


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t: float,
    rate: float = 0.04,
    div_yield: float = 0.0,
    right: str = CALL,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Back out sigma from a market price via bisection (robust, no derivatives)."""

    right = _normalise_right(right)
    if t <= 0.0:
        raise ValueError("cannot invert volatility on an expired option")
    intrinsic = _intrinsic(spot, strike, right)
    if price < intrinsic - 1e-9:
        raise ValueError(f"price {price} is below intrinsic {intrinsic}")

    def f(sigma: float) -> float:
        return black_scholes(spot, strike, t, rate, sigma, div_yield, right).price - price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0.0:
        return lo
    if f_hi < 0.0:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if abs(val) < tol or (hi - lo) < tol:
            return mid
        if val < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def year_fraction(days: float) -> float:
    """Calendar days -> year fraction, floored at zero."""

    return max(days, 0.0) / DAYS_PER_YEAR
