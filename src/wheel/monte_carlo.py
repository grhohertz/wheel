"""Monte Carlo simulation of the wheel strategy.

Roll thousands of synthetic price paths through the same rules the live engine
uses (sell ~0.30-delta premium, buy back at the profit target, close into the
short-DTE window, take assignment when it comes) and report the distribution of
outcomes instead of a single lucky/unlucky backtest.

Model
-----
* **Paths** — geometric Brownian motion under the *real-world* measure::

      S_{t+1} = S_t * exp((mu - q - sigma^2/2) * dt + sigma * sqrt(dt) * Z)

  with ``dt = 1/252``. ``mu`` is the total-return drift, ``q`` the dividend
  yield, so the *price* drift is ``mu - q`` (GLD pays nothing, ``q = 0``).

* **Option marks** — Black-Scholes from :mod:`wheel.greeks`, priced off an
  ``option_iv`` that sits ABOVE realised ``sigma``. That spread is the volatility
  risk premium and is the entire reason the wheel earns anything: sell vol at
  15%, realise 12%.

* **Clock** — paths advance in TRADING days; DTE thresholds (35 DTE entry,
  21 DTE exit) are quoted in CALENDAR days, as they are on a real chain. The
  two are bridged by ``CALENDAR_PER_TRADING`` and year fractions stay on the
  ``/365`` convention used everywhere else in this package.

Everything is SIMULATED. Nothing here touches a broker, a network, or a real
order. Pure stdlib — no numpy — so it runs anywhere the rest of the package does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

from .config import StrategyParams
from .greeks import CALL, PUT, black_scholes
from .marketdata import strike_increment

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.0
CALENDAR_PER_TRADING = CALENDAR_DAYS_PER_YEAR / TRADING_DAYS_PER_YEAR

#: Historical GLD annualised total-return drift.
GLD_DRIFT = 0.08
#: Historical GLD annualised realised volatility.
GLD_VOL = 0.12
#: Typical spread of implied over realised vol — the wheel's edge.
VOL_RISK_PREMIUM = 0.03

#: Don't open a fresh cycle with less than this many trading days on the clock.
MIN_TRADING_DAYS_TO_OPEN = 10

PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


# ----------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MonteCarloParams:
    """Everything that defines one Monte Carlo experiment."""

    symbol: str = "GLD"
    shares: int = 800
    paths: int = 1000
    days: int = TRADING_DAYS_PER_YEAR
    spot: float = 310.0
    mu: float = GLD_DRIFT
    sigma: float = GLD_VOL
    iv: float | None = None  # option IV; defaults to sigma + VOL_RISK_PREMIUM
    rate: float = 0.04
    div_yield: float = 0.0
    entry_dte: int = 35  # calendar DTE targeted when opening a new short
    close_dte: int = 21  # calendar DTE at which we close, win or lose
    seed: int = 20240101
    strategy: StrategyParams = field(default_factory=StrategyParams)

    def __post_init__(self) -> None:
        if self.shares < 0:
            raise ValueError("shares must be >= 0")
        if self.paths < 1:
            raise ValueError("paths must be >= 1")
        if self.days < 1:
            raise ValueError("days must be >= 1")
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.option_iv <= 0:
            raise ValueError("option iv must be positive")
        if self.close_dte >= self.entry_dte:
            raise ValueError("close_dte must be < entry_dte")

    @property
    def option_iv(self) -> float:
        """IV used to price the options we sell (>= realised vol in practice)."""

        return self.sigma + VOL_RISK_PREMIUM if self.iv is None else self.iv

    @property
    def initial_nav(self) -> float:
        return self.shares * self.spot


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------
@dataclass
class Cycle:
    """One opened-and-resolved short option, with the P&L it produced.

    ``pnl`` is the net premium captured PLUS any share P&L realised when the
    cycle ended in assignment or a call-away — i.e. the full economic result of
    that turn of the wheel.
    """

    kind: str  # 'CC' (covered call) or 'CSP' (cash-secured put)
    opened_day: int
    closed_day: int
    strike: float
    contracts: int
    credit: float  # total premium received, net of costs
    close_cost: float  # total paid to close (0 if it expired)
    equity_pnl: float  # realised share P&L booked by this cycle
    outcome: str

    @property
    def pnl(self) -> float:
        return round(self.credit - self.close_cost + self.equity_pnl, 2)


@dataclass
class PathResult:
    """Outcome of a single simulated year."""

    total_pnl: float
    return_pct: float
    max_drawdown_pct: float
    sharpe: float
    win_rate: float
    largest_win: float
    largest_loss: float
    cycles: int
    assignments: int
    called_away: int
    premium_collected: float
    final_nav: float
    final_spot: float


# ----------------------------------------------------------------------
# price paths
# ----------------------------------------------------------------------
def gbm_path(rng: random.Random, spot: float, mu: float, sigma: float, q: float, days: int) -> list[float]:
    """One GBM price path of ``days`` steps, inclusive of the starting spot."""

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    drift = (mu - q - 0.5 * sigma * sigma) * dt
    shock = sigma * math.sqrt(dt)
    prices = [spot]
    s = spot
    for _ in range(days):
        s *= math.exp(drift + shock * rng.gauss(0.0, 1.0))
        prices.append(s)
    return prices


# ----------------------------------------------------------------------
# strike selection
# ----------------------------------------------------------------------
def _abs_delta(ratio: float, t: float, sigma: float, rate: float, q: float, right: str) -> float:
    """|delta| of a ``strike/spot == ratio`` option. Spot-invariant by homogeneity."""

    return abs(black_scholes(1.0, ratio, t, rate, sigma, q, right).delta)


@lru_cache(maxsize=512)
def delta_strike_ratio(
    target_delta: float, t: float, sigma: float, rate: float, q: float, right: str
) -> float:
    """Strike/spot multiple whose |delta| equals ``target_delta``.

    Black-Scholes delta depends on strike and spot only through their ratio, so
    this is solved ONCE per (tenor, vol, rate, yield) tuple and reused for every
    path and every day — the difference between a 2-second run and a 2-minute one.
    Bisection: robust, derivative-free, and monotone in the search direction.
    """

    # Calls: |delta| falls as the strike rises. Puts: |delta| falls as it drops.
    lo, hi = (1.0, 5.0) if right == CALL else (0.05, 1.0)
    f_near = _abs_delta(lo if right == CALL else hi, t, sigma, rate, q, right)
    if target_delta >= f_near:
        # Target is richer than at-the-money — just sell the ATM strike.
        return 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        d = _abs_delta(mid, t, sigma, rate, q, right)
        if abs(d - target_delta) < 1e-6:
            return mid
        if right == CALL:
            lo, hi = (mid, hi) if d > target_delta else (lo, mid)
        else:
            lo, hi = (lo, mid) if d > target_delta else (mid, hi)
    return 0.5 * (lo + hi)


def snap_strike(raw: float, spot: float, right: str) -> float:
    """Round a theoretical strike onto the real chain's increment grid.

    Rounded AWAY from the money (calls up, puts down) so the snapped contract is
    never more aggressive than the delta target we solved for.
    """

    inc = strike_increment(spot)
    snapped = math.ceil(raw / inc) * inc if right == CALL else math.floor(raw / inc) * inc
    return round(max(snapped, inc), 2)


# ----------------------------------------------------------------------
# single-path simulation
# ----------------------------------------------------------------------
@dataclass
class _ShortOption:
    """Bookkeeping for the one open short position a path may carry."""

    right: str
    strike: float
    contracts: int
    credit_per_share: float  # the mid we sold at, per share
    expiry_cal: float  # calendar-day offset from t=0
    opened_day: int
    credit: float  # total cash received, net of slippage + commission


def simulate_path(prices: list[float], p: MonteCarloParams) -> PathResult:
    """Run the wheel over one price path and return its performance record."""

    sp = p.strategy
    mult = sp.contract_multiplier
    iv = p.option_iv
    entry_t = p.entry_dte / CALENDAR_DAYS_PER_YEAR

    # Precomputed once: the ~0.30-delta strike multiple at our entry tenor.
    call_ratio = delta_strike_ratio(sp.target_delta, entry_t, iv, p.rate, p.div_yield, CALL)
    put_ratio = delta_strike_ratio(sp.target_delta, entry_t, iv, p.rate, p.div_yield, PUT)

    shares = p.shares
    basis = p.spot  # share cost basis; drives call-away P&L and the no-sell-below rule
    cash = 0.0
    pos: _ShortOption | None = None
    cycles: list[Cycle] = []
    premium_collected = 0.0
    assignments = 0
    called_away = 0

    initial_nav = p.initial_nav
    navs: list[float] = []

    def mark(option: _ShortOption, spot: float, cal: float) -> float:
        """Per-share theoretical value of the short, floored at zero."""

        t = max(option.expiry_cal - cal, 0.0) / CALENDAR_DAYS_PER_YEAR
        return black_scholes(spot, option.strike, t, p.rate, iv, p.div_yield, option.right).price

    def close_cost(price_per_share: float, contracts: int) -> float:
        """Cash out the door to buy back, including slippage and commission."""

        fill = price_per_share * (1.0 + sp.slippage_pct)
        return fill * contracts * mult + sp.commission_per_contract * contracts

    def open_credit(price_per_share: float, contracts: int) -> float:
        """Cash in the door on a sale, net of slippage and commission."""

        fill = price_per_share * (1.0 - sp.slippage_pct)
        return fill * contracts * mult - sp.commission_per_contract * contracts

    for day in range(len(prices)):
        spot = prices[day]
        cal = day * CALENDAR_PER_TRADING

        # -- 1. manage the open short -----------------------------------
        if pos is not None:
            dte = pos.expiry_cal - cal
            if dte <= 0.0:
                # Expiration: settle intrinsically.
                if pos.right == CALL and spot >= pos.strike:
                    # Called away — book the share P&L, roll to the put side.
                    equity_pnl = (pos.strike - basis) * shares
                    cash += pos.strike * shares
                    shares = 0
                    called_away += 1
                    outcome = "called_away"
                elif pos.right == PUT and spot <= pos.strike:
                    # Assigned — buy the shares, roll back to the call side.
                    bought = pos.contracts * mult
                    cash -= pos.strike * bought
                    shares += bought
                    basis = pos.strike
                    equity_pnl = 0.0
                    assignments += 1
                    outcome = "assigned"
                else:
                    equity_pnl = 0.0
                    outcome = "expired"
                cycles.append(
                    Cycle(
                        kind="CC" if pos.right == CALL else "CSP",
                        opened_day=pos.opened_day, closed_day=day, strike=pos.strike,
                        contracts=pos.contracts, credit=pos.credit, close_cost=0.0,
                        equity_pnl=equity_pnl, outcome=outcome,
                    )
                )
                pos = None
            else:
                m = mark(pos, spot, cal)
                captured = (pos.credit_per_share - m) / pos.credit_per_share if pos.credit_per_share > 0 else 1.0
                itm = (
                    spot >= pos.strike if pos.right == CALL else spot <= pos.strike
                )
                # Take the profit target, or step aside inside the gamma window —
                # but only while the short is OTM. An ITM short is carried to
                # expiry instead of being bought back at intrinsic: taking the
                # assignment IS the wheel (shares called away, then sell puts to
                # get back in). Buying it back would just lock in the loss.
                if captured >= sp.profit_target or (dte <= p.close_dte and not itm):
                    cost = close_cost(m, pos.contracts)
                    cash -= cost
                    cycles.append(
                        Cycle(
                            kind="CC" if pos.right == CALL else "CSP",
                            opened_day=pos.opened_day, closed_day=day, strike=pos.strike,
                            contracts=pos.contracts, credit=pos.credit, close_cost=cost,
                            equity_pnl=0.0,
                            outcome="profit_target" if captured >= sp.profit_target else "dte_exit",
                        )
                    )
                    pos = None

        # -- 2. open a new cycle if flat --------------------------------
        remaining = len(prices) - 1 - day
        if pos is None and remaining >= MIN_TRADING_DAYS_TO_OPEN:
            if shares >= mult:
                # Covered call: never below the share basis (same rule as the engine).
                raw = spot * call_ratio
                if sp.avoid_below_basis:
                    raw = max(raw, basis)
                strike = snap_strike(max(raw, spot), spot, CALL)
                contracts = shares // mult
                right = CALL
            else:
                # Cash-secured put: only as many as the free cash collateralises.
                strike = snap_strike(min(spot * put_ratio, spot), spot, PUT)
                contracts = int(cash // (strike * mult)) if strike > 0 else 0
                right = PUT
            if contracts > 0:
                theo = black_scholes(spot, strike, entry_t, p.rate, iv, p.div_yield, right).price
                if theo >= 0.05:  # same liquidity floor the strategy uses
                    credit = open_credit(theo, contracts)
                    cash += credit
                    premium_collected += credit
                    pos = _ShortOption(
                        right=right, strike=strike, contracts=contracts,
                        credit_per_share=theo, expiry_cal=cal + p.entry_dte,
                        opened_day=day, credit=credit,
                    )

        # -- 3. mark the book -------------------------------------------
        liability = pos.contracts * mult * mark(pos, spot, cal) if pos is not None else 0.0
        navs.append(cash + shares * spot - liability)

    # -- horizon: mark any survivor to market so cycle P&L reconciles ----
    if pos is not None:
        m = mark(pos, prices[-1], (len(prices) - 1) * CALENDAR_PER_TRADING)
        cycles.append(
            Cycle(
                kind="CC" if pos.right == CALL else "CSP",
                opened_day=pos.opened_day, closed_day=len(prices) - 1, strike=pos.strike,
                contracts=pos.contracts, credit=pos.credit,
                close_cost=m * pos.contracts * mult, equity_pnl=0.0,
                outcome="marked_to_market",
            )
        )

    final_nav = navs[-1]
    total_pnl = final_nav - initial_nav

    # -- risk metrics ----------------------------------------------------
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        peak = max(peak, nav)
        if peak > 0:
            max_dd = max(max_dd, (peak - nav) / peak)

    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, len(navs)) if navs[i - 1] > 0]
    sharpe = 0.0
    if len(rets) > 1:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            rf_daily = p.rate / TRADING_DAYS_PER_YEAR
            sharpe = (mean_r - rf_daily) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)

    pnls = [c.pnl for c in cycles]
    wins = [x for x in pnls if x > 0]
    return PathResult(
        total_pnl=total_pnl,
        return_pct=total_pnl / initial_nav if initial_nav else 0.0,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        win_rate=len(wins) / len(pnls) if pnls else 0.0,
        largest_win=max(pnls) if pnls else 0.0,
        largest_loss=min(pnls) if pnls else 0.0,
        cycles=len(cycles),
        assignments=assignments,
        called_away=called_away,
        premium_collected=premium_collected,
        final_nav=final_nav,
        final_spot=prices[-1],
    )


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an ALREADY-SORTED list."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _distribution(values: list[float]) -> dict:
    """mean / std / P10-P90 for one metric across every path."""

    n = len(values)
    mean = sum(values) / n if n else 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    ordered = sorted(values)
    out = {"mean": mean, "std": math.sqrt(var), "min": ordered[0] if n else 0.0,
           "max": ordered[-1] if n else 0.0}
    for q in PERCENTILES:
        out[f"p{int(q * 100)}"] = percentile(ordered, q)
    return out


@dataclass
class MonteCarloSummary:
    """Aggregated distribution across every simulated path."""

    params: MonteCarloParams
    results: list[PathResult]

    @property
    def metrics(self) -> dict[str, dict]:
        r = self.results
        return {
            "total_pnl": _distribution([x.total_pnl for x in r]),
            "return_pct": _distribution([x.return_pct for x in r]),
            "max_drawdown_pct": _distribution([x.max_drawdown_pct for x in r]),
            "sharpe": _distribution([x.sharpe for x in r]),
            "win_rate": _distribution([x.win_rate for x in r]),
            "largest_win": _distribution([x.largest_win for x in r]),
            "largest_loss": _distribution([x.largest_loss for x in r]),
            "premium_collected": _distribution([x.premium_collected for x in r]),
            "final_nav": _distribution([x.final_nav for x in r]),
            "final_spot": _distribution([x.final_spot for x in r]),
            "cycles": _distribution([float(x.cycles) for x in r]),
        }

    @property
    def headline(self) -> dict:
        r = self.results
        n = len(r)
        total_cycles = sum(x.cycles for x in r)
        total_wins = sum(round(x.win_rate * x.cycles) for x in r)
        buy_hold = [(x.final_spot - self.params.spot) * self.params.shares for x in r]
        return {
            "paths": n,
            "prob_profit": sum(1 for x in r if x.total_pnl > 0) / n if n else 0.0,
            "prob_beat_buy_hold": (
                sum(1 for x, bh in zip(r, buy_hold) if x.total_pnl > bh) / n if n else 0.0
            ),
            "buy_hold_mean_pnl": sum(buy_hold) / n if n else 0.0,
            "total_cycles": total_cycles,
            "overall_cycle_win_rate": total_wins / total_cycles if total_cycles else 0.0,
            "mean_assignments": sum(x.assignments for x in r) / n if n else 0.0,
            "mean_called_away": sum(x.called_away for x in r) / n if n else 0.0,
        }

    def to_dict(self) -> dict:
        p = self.params
        return {
            "config": {
                "symbol": p.symbol, "shares": p.shares, "paths": p.paths, "days": p.days,
                "spot": round(p.spot, 4), "mu": p.mu, "sigma": p.sigma,
                "option_iv": round(p.option_iv, 4), "rate": p.rate, "div_yield": p.div_yield,
                "entry_dte": p.entry_dte, "close_dte": p.close_dte,
                "target_delta": p.strategy.target_delta,
                "profit_target": p.strategy.profit_target,
                "initial_nav": round(p.initial_nav, 2), "seed": p.seed,
            },
            "headline": self.headline,
            "metrics": self.metrics,
        }


def run_monte_carlo(params: MonteCarloParams) -> MonteCarloSummary:
    """Simulate every path and aggregate. Deterministic for a given ``seed``."""

    rng = random.Random(params.seed)
    results = [
        simulate_path(
            gbm_path(rng, params.spot, params.mu, params.sigma, params.div_yield, params.days),
            params,
        )
        for _ in range(params.paths)
    ]
    return MonteCarloSummary(params=params, results=results)


__all__ = [
    "CALENDAR_PER_TRADING",
    "Cycle",
    "GLD_DRIFT",
    "GLD_VOL",
    "MonteCarloParams",
    "MonteCarloSummary",
    "PathResult",
    "TRADING_DAYS_PER_YEAR",
    "VOL_RISK_PREMIUM",
    "delta_strike_ratio",
    "gbm_path",
    "percentile",
    "run_monte_carlo",
    "simulate_path",
    "snap_strike",
]
