"""Wheel-strategy selection and position management.

Entry rules (all shorts, premium-collecting):

* **Cash-secured put** — no shares yet: sell an OTM put at ~|delta| target,
  30-45 DTE, fully cash-collateralised.
* **Covered call** — 100+ shares held: sell an OTM call at ~delta target,
  30-45 DTE, never below the share cost basis, never naked.

Management rules:

* Buy back once ``profit_target`` (default 50%) of the credit is captured.
* Roll (close, then re-open next scan) at ``roll_dte`` (default 7) or less.
* Otherwise hold and let theta work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import StrategyParams
from .greeks import CALL, PUT, Greeks
from .models import Action, OptionContract, OptionPosition, Quote


@dataclass(frozen=True)
class Candidate:
    """A contract that passed every filter, with its scoring math."""

    contract: OptionContract
    greeks: Greeks
    delta_distance: float
    annualized_yield: float
    credit_per_contract: float

    @property
    def sort_key(self) -> tuple[float, float]:
        # Closest to target delta wins; ties broken by richer annualized yield.
        return (round(self.delta_distance, 3), -self.annualized_yield)


@dataclass(frozen=True)
class ManagementDecision:
    action: Action
    reason: str
    captured: float = 0.0
    dte: int = 0


class WheelStrategy:
    """Stateless selector/manager — all state lives in the broker."""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()

    # ------------------------------------------------------------------
    # filters
    # ------------------------------------------------------------------
    def liquidity_ok(self, contract: OptionContract) -> bool:
        p = self.params
        return (
            contract.bid >= 0.05
            and contract.open_interest >= p.min_open_interest
            and contract.spread_pct <= p.max_spread_pct
        )

    def dte_ok(self, contract: OptionContract, as_of: date) -> bool:
        dte = (contract.expiry - as_of).days
        return self.params.min_dte <= dte <= self.params.max_dte

    def delta_ok(self, delta: float) -> bool:
        p = self.params
        return abs(abs(delta) - p.target_delta) <= p.delta_tolerance

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def _candidates(
        self,
        chain: list[OptionContract],
        quote: Quote,
        as_of: date,
        right: str,
        min_strike: float | None = None,
        max_strike: float | None = None,
    ) -> list[Candidate]:
        p = self.params
        out: list[Candidate] = []
        for c in chain:
            if c.right.upper() != right:
                continue
            if not self.dte_ok(c, as_of) or not self.liquidity_ok(c):
                continue
            if min_strike is not None and c.strike < min_strike - 1e-9:
                continue
            if max_strike is not None and c.strike > max_strike + 1e-9:
                continue
            g = c.greeks(quote.price, p.risk_free_rate, quote.div_yield)
            if not self.delta_ok(g.delta):
                continue
            dte = max((c.expiry - as_of).days, 1)
            credit = c.mid * p.contract_multiplier
            basis = c.strike if right == PUT else quote.price
            ann = (c.mid / basis) * (365.0 / dte)
            out.append(
                Candidate(
                    contract=c,
                    greeks=g,
                    delta_distance=abs(abs(g.delta) - p.target_delta),
                    annualized_yield=ann,
                    credit_per_contract=round(credit, 2),
                )
            )
        return sorted(out, key=lambda x: x.sort_key)

    def select_covered_call(
        self,
        chain: list[OptionContract],
        quote: Quote,
        as_of: date,
        shares_held: int,
        cost_basis: float | None = None,
    ) -> Candidate | None:
        """Best OTM call to sell against held shares (None if nothing qualifies)."""

        if shares_held < self.params.contract_multiplier:
            return None
        floor = quote.price
        if self.params.avoid_below_basis and cost_basis:
            floor = max(floor, cost_basis)
        cands = self._candidates(chain, quote, as_of, CALL, min_strike=floor)
        return cands[0] if cands else None

    def select_cash_secured_put(
        self, chain: list[OptionContract], quote: Quote, as_of: date, buying_power: float
    ) -> Candidate | None:
        """Best OTM put to sell, capped by available cash collateral."""

        mult = self.params.contract_multiplier
        max_strike = min(quote.price, buying_power / mult) if buying_power > 0 else 0.0
        if max_strike <= 0:
            return None
        cands = self._candidates(chain, quote, as_of, PUT, max_strike=max_strike)
        return cands[0] if cands else None

    # ------------------------------------------------------------------
    # management
    # ------------------------------------------------------------------
    def manage(self, position: OptionPosition, mark: float, as_of: date) -> ManagementDecision:
        """Decide what to do with an open short option given its current mark."""

        p = self.params
        dte = position.dte(as_of)
        captured = position.profit_captured(mark)
        if not position.is_short:
            return ManagementDecision(Action.HOLD, "long option — not managed by the wheel", captured, dte)
        if dte <= 0:
            return ManagementDecision(Action.BUY_TO_CLOSE, "expired — settle now", captured, dte)
        if captured >= p.profit_target:
            return ManagementDecision(
                Action.BUY_TO_CLOSE,
                f"captured {captured:.0%} of the credit (target {p.profit_target:.0%}) at ${mark:.2f}",
                captured,
                dte,
            )
        if dte <= p.roll_dte:
            return ManagementDecision(
                Action.ROLL,
                f"{dte} DTE <= roll threshold {p.roll_dte} — close and re-sell further out",
                captured,
                dte,
            )
        return ManagementDecision(
            Action.HOLD,
            f"{dte} DTE, {captured:.0%} captured — let theta work (target {p.profit_target:.0%})",
            captured,
            dte,
        )

    # ------------------------------------------------------------------
    # scoring helpers
    # ------------------------------------------------------------------
    def confidence(self, candidate: Candidate, quote: Quote) -> float:
        """Heuristic 0-1 score: delta fit, liquidity, and premium richness."""

        p = self.params
        delta_fit = 1.0 - min(candidate.delta_distance / max(p.delta_tolerance, 1e-9), 1.0)
        spread_fit = 1.0 - min(candidate.contract.spread_pct / max(p.max_spread_pct, 1e-9), 1.0)
        oi_fit = min(candidate.contract.open_interest / 1000.0, 1.0)
        yield_fit = min(candidate.annualized_yield / 0.30, 1.0)
        raw = 0.40 * delta_fit + 0.25 * spread_fit + 0.15 * oi_fit + 0.20 * yield_fit
        return round(min(max(raw, 0.05), 0.95), 3)

    def risks(self, candidate: Candidate, quote: Quote) -> tuple[str, ...]:
        out: list[str] = ["IV crush", "early assignment"]
        if quote.div_yield > 0.005 and candidate.contract.is_call():
            out.append(f"dividend ({quote.div_yield:.2%}) raises early-assignment odds")
        if candidate.contract.spread_pct > 0.10:
            out.append(f"wide spread ({candidate.contract.spread_pct:.0%} of mid)")
        if quote.iv > 0.55:
            out.append(f"elevated IV ({quote.iv:.0%}) — underlying can gap")
        if candidate.contract.is_put():
            out.append("assignment leaves you long 100 shares per contract")
        return tuple(out)


__all__ = ["Candidate", "ManagementDecision", "WheelStrategy"]
