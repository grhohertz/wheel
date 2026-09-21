"""Human-readable rendering of recommendations and portfolio state.

The recommendation block intentionally matches the format the Wheel Trader
agent prompt specifies, so agent output and engine output are interchangeable.
"""

from __future__ import annotations

from .models import Action, Recommendation

BANNER = "*** PAPER TRADING — ALL FILLS ARE SIMULATED ***"


def render_recommendation(rec: Recommendation) -> str:
    lines = [f"SYMBOL: {rec.symbol}"]
    if rec.action in (Action.SKIP, Action.HOLD):
        lines.append(f"Current Price: ${rec.spot:,.2f}")
        lines.append(f"Recommendation: {rec.action.value} — {rec.rationale}")
        return "\n".join(lines)

    c = rec.contract
    g = rec.greeks
    if c is not None:
        lines.append(
            f"Current Price: ${rec.spot:,.2f} | IV: {c.iv:.0%} | DTE: {c.dte}"
        )
        kind = "Call" if c.is_call() else "Put"
        lines.append(
            f"Recommendation: {rec.action.value} {rec.contracts}x "
            f"{c.underlying} {c.expiry:%Y-%m-%d} ${c.strike:g} {kind} @ ${c.mid:.2f}"
        )
    else:
        lines.append(f"Current Price: ${rec.spot:,.2f}")
        lines.append(f"Recommendation: {rec.action.value} {rec.contracts}x contract(s)")

    if g is not None:
        # Short position: flip the sign so theta reads as income.
        lines.append(
            f"  Greeks (short): d={-g.delta:+.3f}, G={-g.gamma:+.4f}, "
            f"Th={-g.theta:+.3f}/day, V={-g.vega:+.3f}"
        )
    lines.append(f"  Premium: ${rec.credit:,.2f} (collected, net of the multiplier)")
    if rec.buyback_target:
        lines.append(f"  Buy-back target: ${rec.buyback_target:.2f} (50% of credit)")
    if rec.collateral:
        lines.append(
            f"  Collateral: ${rec.collateral:,.2f} ({rec.collateral_pct:.1%} of NAV)"
        )
    if rec.annualized_yield:
        lines.append(f"  Annualized yield: {rec.annualized_yield:.1%}")
    lines.append(f"  Rationale: {rec.rationale}")
    lines.append(
        f"  Confidence: {rec.confidence:.0%} | Risk: {', '.join(rec.risks) if rec.risks else 'n/a'}"
    )
    return "\n".join(lines)


def render_scan(recs: list[Recommendation]) -> str:
    blocks = [BANNER, ""]
    actionable = [r for r in recs if r.actionable]
    passive = [r for r in recs if not r.actionable]
    for r in actionable:
        blocks.append(render_recommendation(r))
        blocks.append("")
    if passive:
        blocks.append("-- no action --")
        for r in passive:
            blocks.append(f"  {r.symbol:<6} {r.action.value:<22} {r.rationale}")
    return "\n".join(blocks).rstrip() + "\n"


def render_portfolio(p: dict) -> str:
    lines = [
        BANNER,
        "",
        f"Account {p['account_id']} ({p['mode']})  as of {p['as_of']}",
        f"  Cash:            ${p['cash']:>14,.2f}",
        f"  Equity value:    ${p['equity_value']:>14,.2f}",
        f"  Option value:    ${p['option_value']:>14,.2f}",
        f"  Net liquidation: ${p['net_liquidation']:>14,.2f}",
        f"  Realized P&L:    ${p['realized_pnl']:>14,.2f}",
        f"  Open credit:     ${p['open_credit']:>14,.2f}",
        f"  Collateral used: ${p['collateral_used']:>14,.2f}",
    ]
    eq = p["positions"]["equities"]
    if eq:
        lines += ["", "Shares:", f"  {'SYM':<6} {'QTY':>6} {'AVG':>10} {'LAST':>10} {'UNREAL':>12}"]
        for e in eq:
            lines.append(
                f"  {e['symbol']:<6} {e['quantity']:>6} {e['average_cost']:>10.2f} "
                f"{e['price']:>10.2f} {e['unrealized']:>12,.2f}"
            )
    op = p["positions"]["options"]
    if op:
        lines += [
            "",
            "Options (short = negative qty):",
            f"  {'SYMBOL':<22} {'QTY':>5} {'AVG':>8} {'MARK':>8} {'DTE':>5} {'CAPT':>7} {'UNREAL':>10}",
        ]
        for o in op:
            lines.append(
                f"  {o['symbol']:<22} {o['quantity']:>5} {o['average_price']:>8.2f} "
                f"{o['mark']:>8.2f} {o['dte']:>5} {o['captured']:>6.0%} {o['unrealized']:>10,.2f}"
            )
    return "\n".join(lines) + "\n"


def render_trades(trades) -> str:
    if not trades:
        return "no fills\n"
    lines = [f"  {'ACTION':<18} {'SYMBOL':<22} {'QTY':>5} {'PRICE':>9} {'CASH':>12} {'REAL':>10}"]
    for t in trades:
        lines.append(
            f"  {t.action:<18} {t.symbol:<22} {t.quantity:>5} {t.price:>9.2f} "
            f"{t.cash_delta:>12,.2f} {t.realized_pnl:>10,.2f}"
        )
    return "\n".join(lines) + "\n"


__all__ = ["BANNER", "render_portfolio", "render_recommendation", "render_scan", "render_trades"]
