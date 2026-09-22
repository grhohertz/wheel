"""Phase 4 — regime advisor: vol/drift overlays for the Monte Carlo.

Phases 0-2 made the *entry rule* adaptive: the advisor recommends a delta and
the strategy honours it. That still leaves the forward-looking model static.
:mod:`wheel.monte_carlo` prices a whole simulated year off one drift and one
volatility (GLD's long-run 8% / 12%), so it answers "what does the wheel do in
an average year" and nothing else. In a vol spike that estimate is not merely
imprecise, it is biased in the dangerous direction — it understates both the
assignment rate and the drawdown.

Phase 4 closes that gap with a **regime overlay**: a small, bounded, auditable
transform applied to the simulation parameters before any path is drawn.

    raw params ──▶ RegimeOverlay.apply() ──▶ conditioned params ──▶ run_monte_carlo

Three properties make this safe to leave switched on:

* **Bounded.** Every multiplier is clamped (:data:`VOL_MULT_CLAMP` and friends)
  so a bad advisory can shift the distribution but can never produce a
  degenerate or absurd one.
* **Pure.** ``apply`` returns a new frozen ``MonteCarloParams``; nothing is
  mutated and the same seed still reproduces the same paths.
* **Offline.** No live market feed is required. A regime can be named
  explicitly, inferred from whatever IV you already have, or read from the
  advisory cache — and the most useful mode, :func:`regime_scenarios`, simply
  runs *every* regime on one seed to produce a stress matrix.

Regime labels match :func:`wheel.features.classify_market_regime` exactly:
``low_iv``, ``normal``, ``high_iv``, ``vol_spike``, ``vol_crush``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .features import classify_market_regime, compute_iv_rank

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .monte_carlo import MonteCarloParams, MonteCarloSummary

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# guardrails
# ----------------------------------------------------------------------
#: Realised-vol multiplier may at most halve or 2.5x the baseline.
VOL_MULT_CLAMP = (0.50, 2.50)
#: Implied-vol multiplier band. Wider on the top side: IV overshoots RV in a panic.
IV_MULT_CLAMP = (0.50, 3.00)
#: Additive annualised drift shift, in absolute return terms.
MU_SHIFT_CLAMP = (-0.15, 0.15)
#: Additive delta-target shift applied to the strategy leg.
DELTA_SHIFT_CLAMP = (-0.10, 0.10)
#: Additive shift to the entry DTE, in calendar days.
DTE_SHIFT_CLAMP = (-14, 14)
#: Hard floor/ceiling on the conditioned volatility itself.
SIGMA_CLAMP = (0.01, 2.00)
#: Hard floor/ceiling on the conditioned option IV.
IV_CLAMP = (0.01, 3.00)

NORMAL = "normal"
REGIMES = ("low_iv", "normal", "high_iv", "vol_spike", "vol_crush")


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


# ----------------------------------------------------------------------
# the overlay
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeOverlay:
    """A bounded transform from baseline simulation parameters to conditioned ones.

    Attributes:
        regime: the label this overlay represents.
        vol_mult: multiplier on realised vol (``sigma``) — drives the paths.
        iv_mult: multiplier on option IV — drives the premium we collect.
        mu_shift: additive annualised drift adjustment.
        delta_shift: additive shift to the strategy's delta target.
        dte_shift: additive shift to the entry DTE.
        source: provenance — ``table``, ``advisory``, ``inferred``, ``identity``.
        rationale: one line of why, carried into the report and the audit trail.
    """

    regime: str = NORMAL
    vol_mult: float = 1.0
    iv_mult: float = 1.0
    mu_shift: float = 0.0
    delta_shift: float = 0.0
    dte_shift: int = 0
    source: str = "table"
    rationale: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass: clamp through object.__setattr__ so a caller can
        # never construct an out-of-band overlay, however it was built.
        object.__setattr__(self, "vol_mult", _clamp(float(self.vol_mult), VOL_MULT_CLAMP))
        object.__setattr__(self, "iv_mult", _clamp(float(self.iv_mult), IV_MULT_CLAMP))
        object.__setattr__(self, "mu_shift", _clamp(float(self.mu_shift), MU_SHIFT_CLAMP))
        object.__setattr__(
            self, "delta_shift", _clamp(float(self.delta_shift), DELTA_SHIFT_CLAMP)
        )
        object.__setattr__(
            self, "dte_shift", int(round(_clamp(float(self.dte_shift), DTE_SHIFT_CLAMP)))
        )

    # ------------------------------------------------------------------
    @classmethod
    def identity(cls) -> "RegimeOverlay":
        """The no-op overlay: simulate exactly the parameters you passed in."""

        return cls(
            regime=NORMAL,
            source="identity",
            rationale="no regime opinion; baseline parameters unchanged",
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str = "advisory") -> "RegimeOverlay":
        """Build from a JSON block (an advisory's ``mc_overlay``). Never raises."""

        def num(key: str, fallback: float) -> float:
            raw = d.get(key, fallback)
            return float(raw) if isinstance(raw, (int, float)) else fallback

        regime = d.get("regime")
        return cls(
            regime=regime if isinstance(regime, str) else NORMAL,
            vol_mult=num("vol_mult", 1.0),
            iv_mult=num("iv_mult", 1.0),
            mu_shift=num("mu_shift", 0.0),
            delta_shift=num("delta_shift", 0.0),
            dte_shift=int(num("dte_shift", 0)),
            source=source,
            rationale=str(d.get("rationale", "")),
        )

    @property
    def is_identity(self) -> bool:
        return (
            self.vol_mult == 1.0
            and self.iv_mult == 1.0
            and self.mu_shift == 0.0
            and self.delta_shift == 0.0
            and self.dte_shift == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "vol_mult": round(self.vol_mult, 4),
            "iv_mult": round(self.iv_mult, 4),
            "mu_shift": round(self.mu_shift, 4),
            "delta_shift": round(self.delta_shift, 4),
            "dte_shift": self.dte_shift,
            "source": self.source,
            "rationale": self.rationale,
        }

    # ------------------------------------------------------------------
    def apply(self, params: "MonteCarloParams") -> "MonteCarloParams":
        """Return a new ``MonteCarloParams`` conditioned on this regime.

        ``sigma`` moves the *paths*; ``iv`` moves the *premium*. They are
        deliberately separate multipliers — the whole reason a vol spike is
        survivable for a seller is that IV overshoots RV, and the whole reason
        a vol crush is profitable is that it does so on the way back down.

        The strategy's delta target and the entry DTE shift too, because a
        regime call that changes the world but not your behaviour is just a
        worse estimate.
        """

        from .config import StrategyParams  # local: config has no regime dependency

        sigma = _clamp(params.sigma * self.vol_mult, SIGMA_CLAMP)
        iv = _clamp(params.option_iv * self.iv_mult, IV_CLAMP)
        mu = params.mu + self.mu_shift

        strategy = params.strategy
        if self.delta_shift:
            target = _clamp(strategy.target_delta + self.delta_shift, (0.05, 0.60))
            strategy = replace(strategy, target_delta=target)
            if not isinstance(strategy, StrategyParams):  # pragma: no cover - defensive
                strategy = params.strategy

        entry_dte = params.entry_dte + self.dte_shift
        close_dte = params.close_dte
        # entry must stay strictly above close; pull close along if we squeezed.
        if entry_dte <= close_dte:
            entry_dte = close_dte + 7
        if entry_dte < 7:
            entry_dte, close_dte = 14, 7

        return replace(
            params,
            mu=mu,
            sigma=sigma,
            iv=iv,
            entry_dte=entry_dte,
            close_dte=close_dte,
            strategy=strategy,
            overlay=self,
        )


# ----------------------------------------------------------------------
# the calibrated table
# ----------------------------------------------------------------------
#: Hand-calibrated overlays, one per regime label.
#:
#: The numbers encode the standard vol-seller's asymmetry: implied vol moves
#: further and faster than realised vol, in both directions, so ``iv_mult`` is
#: always more extreme than ``vol_mult``. Drift shifts are deliberately small —
#: regime tells you about *variance*, and claiming it tells you much about
#: *direction* is how backtests get flattered.
REGIME_OVERLAYS: dict[str, RegimeOverlay] = {
    "low_iv": RegimeOverlay(
        regime="low_iv",
        vol_mult=0.75,
        iv_mult=0.80,
        mu_shift=0.01,
        delta_shift=0.03,
        dte_shift=7,
        rationale=(
            "Options are cheap and realised vol is quiet. Premium per cycle "
            "collapses, so move closer to the money and further out in time to "
            "get paid at all; benign tape earns a small positive drift tilt."
        ),
    ),
    NORMAL: RegimeOverlay(
        regime=NORMAL,
        vol_mult=1.0,
        iv_mult=1.0,
        mu_shift=0.0,
        delta_shift=0.0,
        dte_shift=0,
        rationale="Baseline regime: long-run drift and vol, unadjusted.",
    ),
    "high_iv": RegimeOverlay(
        regime="high_iv",
        vol_mult=1.35,
        iv_mult=1.55,
        mu_shift=-0.02,
        delta_shift=-0.03,
        dte_shift=0,
        rationale=(
            "IV sits in the top quartile. Premium is rich and the vol risk "
            "premium is wide, but paths are wider too — step further OTM and "
            "accept a mild negative drift tilt, since elevated IV clusters with "
            "drawdown rather than rallies."
        ),
    ),
    "vol_spike": RegimeOverlay(
        regime="vol_spike",
        vol_mult=1.60,
        iv_mult=1.70,
        mu_shift=-0.04,
        delta_shift=-0.05,
        dte_shift=-7,
        rationale=(
            "Vol is rising off a low base — the regime that assigns wheel "
            "sellers. Widen the paths hard, shorten duration so less capital is "
            "trapped through the move, and sell materially further OTM."
        ),
    ),
    "vol_crush": RegimeOverlay(
        regime="vol_crush",
        vol_mult=0.85,
        iv_mult=1.15,
        mu_shift=0.02,
        delta_shift=0.02,
        dte_shift=0,
        rationale=(
            "The best regime for a seller: realised vol is decaying while "
            "options are still marked off the old high. Narrow paths, still-rich "
            "premium — lean in slightly and harvest the decay."
        ),
    ),
}


def overlay_for(regime: str, source: str = "table") -> RegimeOverlay:
    """Look up the calibrated overlay for ``regime``; unknown labels → normal."""

    key = (regime or "").strip().lower()
    base = REGIME_OVERLAYS.get(key)
    if base is None:
        logger.debug("unknown regime %r; using %s", regime, NORMAL)
        base = REGIME_OVERLAYS[NORMAL]
    return base if source == "table" else replace(base, source=source)


def infer_regime(
    current_iv: float,
    iv_52w_low: float,
    iv_52w_high: float,
    term_slope: float = 0.0,
) -> str:
    """Classify a regime from IV levels alone — no live feed required.

    Reuses the Phase 0 classifier so the label the simulator conditions on is
    the same label the advisor prompt sees.
    """

    iv_rank = float(compute_iv_rank(current_iv, iv_52w_low, iv_52w_high))
    # Without a history series there is nothing to smooth, so rank doubles as
    # the percentile. Callers holding a real IV series should classify directly.
    return classify_market_regime(iv_rank, iv_rank, term_slope)


def overlay_from_iv(
    current_iv: float,
    iv_52w_low: float,
    iv_52w_high: float,
    term_slope: float = 0.0,
) -> RegimeOverlay:
    """Infer the regime from IV and return its calibrated overlay."""

    return overlay_for(
        infer_regime(current_iv, iv_52w_low, iv_52w_high, term_slope), source="inferred"
    )


# ----------------------------------------------------------------------
# running scenarios
# ----------------------------------------------------------------------
def run_with_overlay(
    params: "MonteCarloParams", overlay: Optional[RegimeOverlay] = None
) -> "MonteCarloSummary":
    """Condition ``params`` on ``overlay`` and run the Monte Carlo."""

    from .monte_carlo import run_monte_carlo

    overlay = overlay or RegimeOverlay.identity()
    return run_monte_carlo(overlay.apply(params))


def run_regime(params: "MonteCarloParams", regime: str) -> "MonteCarloSummary":
    """Run the Monte Carlo conditioned on a named regime."""

    return run_with_overlay(params, overlay_for(regime))


def regime_scenarios(
    params: "MonteCarloParams", regimes: Optional[Iterable[str]] = None
) -> dict[str, "MonteCarloSummary"]:
    """Run the same experiment under every regime — the Phase 4 stress matrix.

    The seed is held constant across regimes, so every difference in the output
    is attributable to the overlay and not to path noise. This is the headline
    deliverable: without any live market data you can read off what the wheel
    does when the world changes shape.
    """

    labels = list(regimes) if regimes is not None else list(REGIMES)
    return {label: run_regime(params, label) for label in labels}


def compare_regimes(
    params: "MonteCarloParams", regimes: Optional[Iterable[str]] = None
) -> dict[str, Any]:
    """JSON-ready comparison table across regimes.

    Returns a dict with a ``baseline`` block (the unconditioned parameters) and
    one ``rows`` entry per regime carrying the overlay, the conditioned inputs,
    and the outcome distribution that matters for sizing decisions.
    """

    summaries = regime_scenarios(params, regimes)
    rows = []
    for label, summary in summaries.items():
        p = summary.params
        overlay = p.overlay or RegimeOverlay.identity()
        metrics = summary.metrics
        head = summary.headline
        rows.append(
            {
                "regime": label,
                "overlay": overlay.to_dict(),
                "inputs": {
                    "mu": round(p.mu, 4),
                    "sigma": round(p.sigma, 4),
                    "option_iv": round(p.option_iv, 4),
                    "target_delta": round(p.strategy.target_delta, 4),
                    "entry_dte": p.entry_dte,
                    "close_dte": p.close_dte,
                },
                "outcome": {
                    "mean_pnl": metrics["total_pnl"]["mean"],
                    "p10_pnl": metrics["total_pnl"]["p10"],
                    "p50_pnl": metrics["total_pnl"]["p50"],
                    "p90_pnl": metrics["total_pnl"]["p90"],
                    "mean_return_pct": metrics["return_pct"]["mean"],
                    "mean_max_drawdown_pct": metrics["max_drawdown_pct"]["mean"],
                    "mean_premium": metrics["premium_collected"]["mean"],
                    "prob_profit": head["prob_profit"],
                    "prob_beat_buy_hold": head["prob_beat_buy_hold"],
                    "mean_assignments": head["mean_assignments"],
                    "mean_called_away": head["mean_called_away"],
                    "cycle_win_rate": head["overall_cycle_win_rate"],
                },
            }
        )

    return {
        "baseline": {
            "symbol": params.symbol,
            "shares": params.shares,
            "paths": params.paths,
            "days": params.days,
            "spot": round(params.spot, 4),
            "mu": round(params.mu, 4),
            "sigma": round(params.sigma, 4),
            "option_iv": round(params.option_iv, 4),
            "seed": params.seed,
            "initial_nav": round(params.initial_nav, 2),
        },
        "rows": rows,
    }


__all__ = [
    "DELTA_SHIFT_CLAMP",
    "DTE_SHIFT_CLAMP",
    "IV_MULT_CLAMP",
    "MU_SHIFT_CLAMP",
    "NORMAL",
    "REGIMES",
    "REGIME_OVERLAYS",
    "RegimeOverlay",
    "VOL_MULT_CLAMP",
    "compare_regimes",
    "infer_regime",
    "overlay_for",
    "overlay_from_iv",
    "regime_scenarios",
    "run_regime",
    "run_with_overlay",
]
