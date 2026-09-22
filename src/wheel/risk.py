"""Portfolio risk management for the paper-trading wheel engine.

Four independently-testable layers:

* :class:`PositionLimits`       -- hard caps on notional and greek exposure.
* :class:`DrawdownMonitor`      -- peak-to-trough equity tracking with a stop trigger.
* :class:`PortfolioConstraints` -- statistical floors/ceilings (Sharpe, VaR, correlation).
* :class:`RiskAggregator`       -- rolls positions into one exposure and runs every check.

Pure stdlib: no numpy/scipy. Nothing in this module places, blocks, or cancels an
order -- callers consult :meth:`RiskAggregator.evaluate` and decide for themselves.

Conventions
-----------
* Greek exposures are POSITION-level (already multiplied by quantity x multiplier),
  signed, and follow :mod:`wheel.greeks` units (theta/day, vega per vol point).
* ``max_var`` and drawdown thresholds are POSITIVE fractions: ``0.05`` == a 5% loss.
* Limits default to infinity so an unconfigured limit never fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .greeks import black_scholes, year_fraction
from .models import EquityPosition, OptionPosition, Quote

TRADING_DAYS = 252
DEFAULT_RATE = 0.04

__all__ = [
    "DrawdownMonitor",
    "PortfolioConstraints",
    "PortfolioExposure",
    "PositionLimits",
    "RiskAggregator",
    "RiskBreach",
    "RiskReport",
    "Severity",
    "TRADING_DAYS",
]


class Severity(str, Enum):
    """How loudly a limit violation should be treated."""

    WARN = "warn"
    BREACH = "breach"


@dataclass(frozen=True)
class RiskBreach:
    """One violated (or nearly violated) risk rule."""

    code: str
    message: str
    limit: float
    observed: float
    severity: Severity = Severity.BREACH

    @property
    def utilisation(self) -> float:
        """``|observed| / |limit|`` -- 1.0 means exactly at the cap."""

        if self.limit == 0:
            return math.inf if self.observed else 0.0
        if math.isinf(self.limit):
            return 0.0
        return abs(self.observed) / abs(self.limit)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "limit": self.limit,
            "observed": self.observed,
            "severity": self.severity.value,
            "utilisation": round(self.utilisation, 4),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(frozen=True)
class PortfolioExposure:
    """Aggregated, signed, position-level risk of the whole book."""

    notional: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    contracts: int = 0
    shares: int = 0

    def __add__(self, other: "PortfolioExposure") -> "PortfolioExposure":
        if not isinstance(other, PortfolioExposure):  # pragma: no cover - defensive
            return NotImplemented
        return PortfolioExposure(
            notional=self.notional + other.notional,
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            contracts=self.contracts + other.contracts,
            shares=self.shares + other.shares,
        )

    def rounded(self, places: int = 4) -> "PortfolioExposure":
        return PortfolioExposure(
            notional=round(self.notional, places),
            delta=round(self.delta, places),
            gamma=round(self.gamma, places),
            theta=round(self.theta, places),
            vega=round(self.vega, places),
            contracts=self.contracts,
            shares=self.shares,
        )

    def to_dict(self) -> dict:
        r = self.rounded(2)
        return {
            "notional": r.notional,
            "delta": r.delta,
            "gamma": round(self.gamma, 4),
            "theta": r.theta,
            "vega": r.vega,
            "contracts": self.contracts,
            "shares": self.shares,
        }


@dataclass(frozen=True)
class PositionLimits:
    """Hard caps on book size. Every limit is an ABSOLUTE magnitude."""

    max_notional: float = math.inf
    max_delta_exposure: float = math.inf
    max_gamma_exposure: float = math.inf
    max_vega_exposure: float = math.inf
    max_contracts: int = 2**31
    warn_ratio: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "max_notional",
            "max_delta_exposure",
            "max_gamma_exposure",
            "max_vega_exposure",
            "max_contracts",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 < self.warn_ratio <= 1.0:
            raise ValueError("warn_ratio must be in (0, 1]")

    def _compare(self, code: str, observed: float, limit: float, unit: str) -> RiskBreach | None:
        magnitude = abs(observed)
        if math.isinf(limit):
            return None
        if magnitude > limit:
            return RiskBreach(
                code=code,
                message=f"{unit} {magnitude:,.2f} exceeds cap {limit:,.2f}",
                limit=limit,
                observed=observed,
                severity=Severity.BREACH,
            )
        if limit > 0 and magnitude >= limit * self.warn_ratio:
            return RiskBreach(
                code=code,
                message=f"{unit} {magnitude:,.2f} is within {int((1 - self.warn_ratio) * 100)}% of cap {limit:,.2f}",
                limit=limit,
                observed=observed,
                severity=Severity.WARN,
            )
        return None

    def check(self, exposure: PortfolioExposure) -> list[RiskBreach]:
        """Return every breach/warning raised by ``exposure``."""

        candidates = (
            self._compare("max_notional", exposure.notional, self.max_notional, "notional"),
            self._compare("max_delta_exposure", exposure.delta, self.max_delta_exposure, "delta"),
            self._compare("max_gamma_exposure", exposure.gamma, self.max_gamma_exposure, "gamma"),
            self._compare("max_vega_exposure", exposure.vega, self.max_vega_exposure, "vega"),
            self._compare("max_contracts", exposure.contracts, float(self.max_contracts), "contracts"),
        )
        return [c for c in candidates if c is not None]

    def allows(self, exposure: PortfolioExposure) -> bool:
        """True when no hard BREACH fires (warnings are tolerated)."""

        return not any(b.severity is Severity.BREACH for b in self.check(exposure))


@dataclass
class DrawdownMonitor:
    """Tracks running peak equity and the peak-to-trough decline from it.

    ``threshold`` is the stop level as a positive fraction (0.20 == 20%).
    """

    threshold: float = 0.20
    peak: float = 0.0
    trough: float = 0.0
    last: float = 0.0
    max_drawdown: float = 0.0
    started: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")

    def update(self, equity: float) -> float:
        """Feed one equity observation; returns the CURRENT drawdown fraction."""

        if equity <= 0:
            raise ValueError("equity must be positive")
        if not self.started:
            self.peak = self.trough = self.last = equity
            self.started = True
            self.max_drawdown = 0.0
            return 0.0

        self.last = equity
        if equity > self.peak:
            # New high-water mark: the trough resets with it.
            self.peak = equity
            self.trough = equity
        elif equity < self.trough:
            self.trough = equity
        self.max_drawdown = max(self.max_drawdown, (self.peak - self.trough) / self.peak)
        return self.current_drawdown

    def extend(self, series: Iterable[float]) -> float:
        for value in series:
            self.update(value)
        return self.current_drawdown

    @classmethod
    def from_series(cls, series: Sequence[float], threshold: float = 0.20) -> "DrawdownMonitor":
        monitor = cls(threshold=threshold)
        monitor.extend(series)
        return monitor

    @property
    def current_drawdown(self) -> float:
        """Decline from the running peak to the LATEST observation."""

        if not self.started or self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - self.last) / self.peak)

    @property
    def triggered(self) -> bool:
        return self.current_drawdown >= self.threshold

    def check(self) -> list[RiskBreach]:
        dd = self.current_drawdown
        if dd >= self.threshold:
            return [
                RiskBreach(
                    code="max_drawdown",
                    message=f"drawdown {dd:.2%} at/over stop {self.threshold:.2%}",
                    limit=self.threshold,
                    observed=dd,
                    severity=Severity.BREACH,
                )
            ]
        if dd >= self.threshold * 0.8:
            return [
                RiskBreach(
                    code="max_drawdown",
                    message=f"drawdown {dd:.2%} approaching stop {self.threshold:.2%}",
                    limit=self.threshold,
                    observed=dd,
                    severity=Severity.WARN,
                )
            ]
        return []

    def reset(self, equity: float | None = None) -> None:
        """Forget history. With ``equity`` supplied, restart from that level."""

        self.peak = self.trough = self.last = 0.0
        self.max_drawdown = 0.0
        self.started = False
        if equity is not None:
            self.update(equity)


@dataclass(frozen=True)
class PortfolioConstraints:
    """Statistical guardrails computed from a return series.

    ``max_var`` is a positive loss fraction at ``confidence`` (0.05 == 5%).
    """

    min_sharpe: float = -math.inf
    max_var: float = math.inf
    correlation_bound: float = 1.0
    confidence: float = 0.95
    risk_free_rate: float = 0.0
    periods_per_year: int = TRADING_DAYS

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if not 0.0 <= self.correlation_bound <= 1.0:
            raise ValueError("correlation_bound must be in [0, 1]")
        if self.max_var < 0:
            raise ValueError("max_var must be >= 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    # ---- statistics (stateless, reusable) --------------------------------
    @staticmethod
    def sharpe_ratio(
        returns: Sequence[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = TRADING_DAYS,
    ) -> float:
        """Annualised Sharpe of a PERIODIC return series (sample stdev)."""

        if len(returns) < 2:
            raise ValueError("sharpe_ratio needs at least 2 observations")
        per_period_rf = risk_free_rate / periods_per_year
        excess = [r - per_period_rf for r in returns]
        mean = sum(excess) / len(excess)
        var = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
        if var <= 0:
            return 0.0
        return (mean / math.sqrt(var)) * math.sqrt(periods_per_year)

    @staticmethod
    def value_at_risk(returns: Sequence[float], confidence: float = 0.95) -> float:
        """Historical VaR as a POSITIVE loss fraction (0.0 when no loss tail)."""

        if not returns:
            raise ValueError("value_at_risk needs at least 1 observation")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        ordered = sorted(returns)
        rank = (1.0 - confidence) * (len(ordered) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            quantile = ordered[int(rank)]
        else:
            weight = rank - low
            quantile = ordered[low] * (1 - weight) + ordered[high] * weight
        return max(0.0, -quantile)

    @staticmethod
    def correlation(a: Sequence[float], b: Sequence[float]) -> float:
        """Pearson correlation. Returns 0.0 when either series is constant."""

        if len(a) != len(b):
            raise ValueError("series must be the same length")
        if len(a) < 2:
            raise ValueError("correlation needs at least 2 observations")
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        var_a = sum((x - mean_a) ** 2 for x in a)
        var_b = sum((y - mean_b) ** 2 for y in b)
        if var_a <= 0 or var_b <= 0:
            return 0.0
        return cov / math.sqrt(var_a * var_b)

    # ---- enforcement -----------------------------------------------------
    def check(
        self,
        returns: Sequence[float] = (),
        correlations: Mapping[str, float] | None = None,
    ) -> list[RiskBreach]:
        """Evaluate Sharpe/VaR against ``returns`` and any named correlations."""

        breaches: list[RiskBreach] = []
        if len(returns) >= 2 and not math.isinf(self.min_sharpe):
            sharpe = self.sharpe_ratio(returns, self.risk_free_rate, self.periods_per_year)
            if sharpe < self.min_sharpe:
                breaches.append(
                    RiskBreach(
                        code="min_sharpe",
                        message=f"sharpe {sharpe:.2f} below floor {self.min_sharpe:.2f}",
                        limit=self.min_sharpe,
                        observed=sharpe,
                    )
                )
        if returns and not math.isinf(self.max_var):
            var = self.value_at_risk(returns, self.confidence)
            if var > self.max_var:
                breaches.append(
                    RiskBreach(
                        code="max_var",
                        message=f"VaR({self.confidence:.0%}) {var:.2%} exceeds {self.max_var:.2%}",
                        limit=self.max_var,
                        observed=var,
                    )
                )
        for label, value in (correlations or {}).items():
            if abs(value) > self.correlation_bound:
                breaches.append(
                    RiskBreach(
                        code="correlation_bound",
                        message=f"correlation {label} {value:+.2f} exceeds bound {self.correlation_bound:.2f}",
                        limit=self.correlation_bound,
                        observed=value,
                    )
                )
        return breaches


@dataclass(frozen=True)
class RiskReport:
    """Result of one full risk evaluation."""

    exposure: PortfolioExposure
    breaches: tuple[RiskBreach, ...] = ()
    drawdown: float = 0.0

    @property
    def ok(self) -> bool:
        """True when nothing at BREACH severity fired."""

        return not self.hard_breaches

    @property
    def hard_breaches(self) -> tuple[RiskBreach, ...]:
        return tuple(b for b in self.breaches if b.severity is Severity.BREACH)

    @property
    def warnings(self) -> tuple[RiskBreach, ...]:
        return tuple(b for b in self.breaches if b.severity is Severity.WARN)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(b.code for b in self.breaches)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "drawdown": round(self.drawdown, 6),
            "exposure": self.exposure.to_dict(),
            "breaches": [b.to_dict() for b in self.breaches],
        }


@dataclass
class RiskAggregator:
    """Rolls a book of positions into one :class:`PortfolioExposure` and checks it."""

    limits: PositionLimits = field(default_factory=PositionLimits)
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    drawdown: DrawdownMonitor = field(default_factory=DrawdownMonitor)
    rate: float = DEFAULT_RATE

    @staticmethod
    def _quote(quotes: Mapping[str, Quote], symbol: str) -> Quote:
        try:
            return quotes[symbol]
        except KeyError as exc:  # pragma: no cover - re-raised with context below
            raise KeyError(f"no quote for underlying {symbol!r}") from exc

    def exposure(
        self,
        quotes: Mapping[str, Quote],
        as_of: date,
        option_positions: Iterable[OptionPosition] = (),
        equity_positions: Iterable[EquityPosition] = (),
    ) -> PortfolioExposure:
        """Aggregate signed, position-level greeks plus gross notional.

        Shares contribute 1.0 delta each; options are priced with Black-Scholes
        using the quote's IV and dividend yield.
        """

        total = PortfolioExposure()
        for pos in equity_positions:
            if pos.quantity == 0:
                continue
            quote = self._quote(quotes, pos.symbol)
            total = total + PortfolioExposure(
                notional=abs(pos.quantity * quote.price),
                delta=float(pos.quantity),
                shares=pos.quantity,
            )

        for pos in option_positions:
            if pos.quantity == 0:
                continue
            quote = self._quote(quotes, pos.underlying)
            t = year_fraction(pos.dte(as_of))
            g = black_scholes(
                quote.price, pos.strike, t, self.rate, quote.iv, quote.div_yield, pos.right
            ).scaled(pos.quantity, pos.multiplier)
            total = total + PortfolioExposure(
                notional=abs(pos.quantity) * pos.multiplier * quote.price,
                delta=g.delta,
                gamma=g.gamma,
                theta=g.theta,
                vega=g.vega,
                contracts=abs(pos.quantity),
            )
        return total

    def evaluate(
        self,
        quotes: Mapping[str, Quote],
        as_of: date,
        option_positions: Iterable[OptionPosition] = (),
        equity_positions: Iterable[EquityPosition] = (),
        equity: float | None = None,
        returns: Sequence[float] = (),
        correlations: Mapping[str, float] | None = None,
    ) -> RiskReport:
        """Run every layer and return a consolidated :class:`RiskReport`."""

        exposure = self.exposure(
            quotes=quotes,
            as_of=as_of,
            option_positions=option_positions,
            equity_positions=equity_positions,
        )
        breaches: list[RiskBreach] = list(self.limits.check(exposure))
        if equity is not None:
            self.drawdown.update(equity)
        breaches.extend(self.drawdown.check())
        breaches.extend(self.constraints.check(returns, correlations))
        return RiskReport(
            exposure=exposure,
            breaches=tuple(breaches),
            drawdown=self.drawdown.current_drawdown,
        )
