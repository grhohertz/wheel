"""Command-line interface.

    python -m wheel.cli scan
    python -m wheel.cli run --execute
    python -m wheel.cli positions
    python -m wheel.cli ledger
    python -m wheel.cli price AAPL --strike 190 --dte 35
    python -m wheel.cli reset --cash 100000
    python -m wheel.cli monte-carlo --shares 800 --symbol GLD --paths 1000 --days 252
    python -m wheel.cli live --symbol GLD --shares 800 --duration 60

Every command runs against the local paper account JSON (``--state``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from .broker import PaperBroker
from .config import PAPER, Settings, assert_paper_mode
from .engine import WheelEngine
from .greeks import black_scholes, year_fraction
from .live import build_paper_engine
from .marketdata import SyntheticMarketData
from .monte_carlo import GLD_DRIFT, GLD_VOL, MonteCarloParams, run_monte_carlo
from .regime import REGIMES, compare_regimes, overlay_for
from .report import (
    BANNER,
    render_monte_carlo,
    render_portfolio,
    render_regime_matrix,
    render_scan,
    render_trades,
)
from .audit import AuditLedger
from .client import ClaudeAdvisor
from .features import build_feature_vector


def _as_of(value: str | None) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else date.today()


def build_engine(args: argparse.Namespace) -> tuple[WheelEngine, Settings, Path]:
    settings = Settings.from_env()
    assert_paper_mode(settings.mode)
    state = Path(args.state or settings.state_path)
    broker = PaperBroker.load(state, params=settings.params, default_cash=settings.starting_cash)
    broker.account_id = settings.account_id
    market = SyntheticMarketData(rate=settings.params.risk_free_rate)
    return WheelEngine(market, broker, settings.params, settings), settings, state


def _symbols(args: argparse.Namespace, settings: Settings) -> list[str]:
    if args.symbols:
        return [s.upper() for s in args.symbols]
    return list(settings.watchlist)


def cmd_scan(args: argparse.Namespace) -> int:
    engine, settings, _ = build_engine(args)
    recs = engine.scan(_symbols(args, settings), _as_of(args.date))
    if args.json:
        print(json.dumps([r.to_dict() for r in recs], indent=2))
    else:
        print(render_scan(recs))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    engine, settings, state = build_engine(args)
    result = engine.run_cycle(_symbols(args, settings), _as_of(args.date), execute=args.execute)
    if args.json:
        print(
            json.dumps(
                {
                    "as_of": result.as_of.isoformat(),
                    "settlements": [t.to_dict() for t in result.settlements],
                    "recommendations": [r.to_dict() for r in result.recommendations],
                    "trades": [t.to_dict() for t in result.trades],
                    "portfolio": engine.portfolio(result.as_of),
                },
                indent=2,
            )
        )
    else:
        print(render_scan(result.recommendations))
        if result.settlements:
            print("Settlements:")
            print(render_trades(result.settlements))
        if args.execute:
            print("Simulated fills:")
            print(render_trades(result.trades))
        print(render_portfolio(engine.portfolio(result.as_of)))
    if args.execute:
        engine.broker.save(state)
        print(f"state saved -> {state}", file=sys.stderr)
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    engine, _, _ = build_engine(args)
    p = engine.portfolio(_as_of(args.date))
    print(json.dumps(p, indent=2) if args.json else render_portfolio(p))
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    engine, _, _ = build_engine(args)
    trades = engine.broker.ledger[-args.limit :]
    if args.json:
        print(json.dumps([t.to_dict() for t in trades], indent=2))
    else:
        print(BANNER)
        print(render_trades(trades))
    return 0


def cmd_price(args: argparse.Namespace) -> int:
    engine, _, _ = build_engine(args)
    as_of = _as_of(args.date)
    quote = engine.market.get_quote(args.symbol, as_of)
    strike = args.strike or round(quote.price, 2)
    g = black_scholes(
        quote.price, strike, year_fraction(args.dte), args.rate, args.iv or quote.iv,
        quote.div_yield, args.right,
    )
    out = {
        "symbol": quote.symbol,
        "spot": quote.price,
        "strike": strike,
        "dte": args.dte,
        "iv": args.iv or quote.iv,
        "right": args.right.upper(),
        "price": round(g.price, 4),
        "delta": round(g.delta, 4),
        "gamma": round(g.gamma, 5),
        "theta_per_day": round(g.theta, 4),
        "vega_per_vol_pt": round(g.vega, 4),
        "rho_per_pct": round(g.rho, 4),
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:>16}: {v}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    state = Path(args.state or settings.state_path)
    broker = PaperBroker(
        cash=args.cash, account_id=settings.account_id, params=settings.params, mode=PAPER
    )
    broker.save(state)
    print(f"reset paper account {settings.account_id} to ${args.cash:,.2f} -> {state}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Roll the engine forward day by day to show the wheel turning."""

    engine, settings, state = build_engine(args)
    start = _as_of(args.date)
    symbols = _symbols(args, settings)
    for i in range(args.days):
        day = start + timedelta(days=i * args.step)
        result = engine.run_cycle(symbols, day, execute=True)
        acted = len(result.trades) + len(result.settlements)
        nav = engine.portfolio(day)["net_liquidation"]
        print(f"{day.isoformat()}  fills={acted:<3} NAV=${nav:,.2f}")
    engine.broker.save(state)
    print()
    print(render_portfolio(engine.portfolio(start + timedelta(days=(args.days - 1) * args.step))))
    return 0


def cmd_monte_carlo(args: argparse.Namespace) -> int:
    """Distribution of wheel outcomes over thousands of simulated years.

    Spot and IV default to the engine's market-data quote for the symbol (same
    source ``price`` and ``scan`` use) so the simulation starts from the book the
    rest of the CLI sees; ``--spot`` / ``--iv`` override either one.
    """

    params = _mc_params(args)
    regime = getattr(args, "regime", None)
    if regime:
        params = overlay_for(regime).apply(params)
    summary = run_monte_carlo(params).to_dict()
    print(json.dumps(summary, indent=2) if args.json else render_monte_carlo(summary))
    return 0


def _mc_params(args: argparse.Namespace) -> MonteCarloParams:
    """Build simulation parameters from the shared Monte-Carlo flags."""

    engine, _, _ = build_engine(args)
    quote = engine.market.get_quote(args.symbol, _as_of(args.date))
    return MonteCarloParams(
        symbol=args.symbol.upper(),
        shares=args.shares,
        paths=args.paths,
        days=args.days,
        spot=args.spot if args.spot else quote.price,
        mu=args.mu,
        sigma=args.sigma,
        iv=args.iv,
        rate=args.rate,
        div_yield=quote.div_yield if args.div_yield is None else args.div_yield,
        entry_dte=args.entry_dte,
        close_dte=args.close_dte,
        seed=args.seed,
        strategy=engine.params,
    )


def cmd_regime(args: argparse.Namespace) -> int:
    """Phase 4 stress matrix: the same wheel, run under every volatility regime.

    One seed is held across all regimes, so each row differs only by the regime
    overlay applied to drift, realised vol, option IV, delta and DTE. No live
    market data is required — this answers "what does this position do when the
    world changes shape" entirely offline.
    """

    labels = [r.strip().lower() for r in args.regimes.split(",") if r.strip()]
    if labels == ["all"] or not labels:
        labels = list(REGIMES)

    unknown = [r for r in labels if r not in REGIMES]
    if unknown:
        print(
            f"error: unknown regime(s) {', '.join(unknown)}; "
            f"choose from {', '.join(REGIMES)} or 'all'",
            file=sys.stderr,
        )
        return 1

    comparison = compare_regimes(_mc_params(args), labels)
    print(
        json.dumps(comparison, indent=2)
        if args.json
        else render_regime_matrix(comparison)
    )
    return 0



def _add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Global flags, accepted before OR after the subcommand.

    Subparser copies default to SUPPRESS so an unset flag there does not clobber a
    value given at the top level. (Fresh action objects per parser — sharing them via
    ``parents=`` lets ``set_defaults`` leak across parsers.)
    """

    none_default = argparse.SUPPRESS if suppress else None
    flag_default = argparse.SUPPRESS if suppress else False
    parser.add_argument("--state", default=none_default, help="path to the paper-account JSON")
    parser.add_argument(
        "--json", action="store_true", default=flag_default, help="machine-readable output"
    )
    parser.add_argument("--date", default=none_default, help="as-of date (YYYY-MM-DD)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wheel-trader", description="Paper-trading wheel strategy engine"
    )
    _add_common(p, suppress=False)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help)
        _add_common(child, suppress=True)
        return child

    s = add("scan", "recommendations only, no fills")
    s.add_argument("symbols", nargs="*")
    s.set_defaults(func=cmd_scan)

    r = add("run", "settle, scan, and optionally execute")
    r.add_argument("symbols", nargs="*")
    r.add_argument("--execute", action="store_true", help="simulate the fills and persist state")
    r.set_defaults(func=cmd_run)

    pos = add("positions", "portfolio snapshot")
    pos.set_defaults(func=cmd_positions)

    led = add("ledger", "recent simulated fills")
    led.add_argument("--limit", type=int, default=25)
    led.set_defaults(func=cmd_ledger)

    pr = add("price", "ad-hoc Black-Scholes pricing")
    pr.add_argument("symbol")
    pr.add_argument("--strike", type=float)
    pr.add_argument("--dte", type=int, default=35)
    pr.add_argument("--iv", type=float)
    pr.add_argument("--rate", type=float, default=0.04)
    pr.add_argument("--right", default="C", choices=["C", "P", "c", "p"])
    pr.set_defaults(func=cmd_price)

    rs = add("reset", "wipe the paper account")
    rs.add_argument("--cash", type=float, default=100_000.0)
    rs.set_defaults(func=cmd_reset)

    sim = add("simulate", "advance the clock and run repeated cycles")
    sim.add_argument("symbols", nargs="*")
    sim.add_argument("--days", type=int, default=8, help="number of cycles")
    sim.add_argument("--step", type=int, default=7, help="calendar days per cycle")
    sim.set_defaults(func=cmd_simulate)

    mc = add("monte-carlo", "Monte Carlo distribution of wheel outcomes")
    mc.add_argument("--symbol", default="GLD", help="underlying to simulate")
    mc.add_argument("--shares", type=int, default=800, help="shares held at t=0")
    mc.add_argument("--paths", type=int, default=1000, help="number of simulated years")
    mc.add_argument("--days", type=int, default=252, help="trading days per path")
    mc.add_argument("--spot", type=float, help="starting price (default: market quote)")
    mc.add_argument("--mu", type=float, default=GLD_DRIFT, help="annual drift (total return)")
    mc.add_argument("--sigma", type=float, default=GLD_VOL, help="annual realised volatility")
    mc.add_argument("--iv", type=float, help="option IV (default: sigma + vol risk premium)")
    mc.add_argument("--rate", type=float, default=0.04, help="risk-free rate")
    mc.add_argument("--div-yield", type=float, dest="div_yield", help="dividend yield override")
    mc.add_argument("--entry-dte", type=int, default=35, dest="entry_dte", help="DTE at entry")
    mc.add_argument("--close-dte", type=int, default=21, dest="close_dte", help="DTE exit window")
    mc.add_argument("--seed", type=int, default=20240101, help="RNG seed (runs are reproducible)")
    mc.add_argument(
        "--regime",
        choices=list(REGIMES),
        help="condition the simulation on a volatility regime (Phase 4 overlay)",
    )
    mc.set_defaults(func=cmd_monte_carlo)

    reg = add("regime", "stress matrix: the same wheel under every volatility regime")
    reg.add_argument("--symbol", default="GLD", help="underlying to simulate")
    reg.add_argument("--shares", type=int, default=800, help="shares held at t=0")
    reg.add_argument("--paths", type=int, default=1000, help="number of simulated years")
    reg.add_argument("--days", type=int, default=252, help="trading days per path")
    reg.add_argument("--spot", type=float, help="starting price (default: market quote)")
    reg.add_argument("--mu", type=float, default=GLD_DRIFT, help="baseline annual drift")
    reg.add_argument("--sigma", type=float, default=GLD_VOL, help="baseline annual vol")
    reg.add_argument("--iv", type=float, help="baseline option IV (default: sigma + VRP)")
    reg.add_argument("--rate", type=float, default=0.04, help="risk-free rate")
    reg.add_argument("--div-yield", type=float, dest="div_yield", help="dividend yield override")
    reg.add_argument("--entry-dte", type=int, default=35, dest="entry_dte", help="DTE at entry")
    reg.add_argument("--close-dte", type=int, default=21, dest="close_dte", help="DTE exit window")
    reg.add_argument("--seed", type=int, default=20240101, help="RNG seed (held across regimes)")
    reg.add_argument(
        "--regimes",
        default="all",
        help="comma-separated regimes to compare, or 'all' (default)",
    )
    reg.set_defaults(func=cmd_regime)

    adv = add("advisor", "Refresh advisory cache (Claude recommendations)")
    adv.add_argument("symbols", nargs="*", help="symbols to advise (default: watchlist)")
    adv.add_argument("--force", action="store_true", help="ignore cache, call Claude")
    adv.set_defaults(func=cmd_advisor)

    lv = add("live", "paper live-loop demo: feed -> signal -> risk -> simulated fills")
    lv.add_argument("--symbol", default="GLD", help="underlying to stream")
    lv.add_argument("--shares", type=int, default=800, help="starting share position")
    lv.add_argument("--duration", type=float, default=60.0, help="seconds of tape to replay")
    lv.add_argument("--interval", type=float, default=1.0, help="seconds per tick")
    lv.add_argument("--spot", type=float, help="starting price (default: market quote)")
    lv.add_argument("--band", type=float, default=0.004, help="signal trigger band around anchor")
    lv.add_argument("--qty", type=int, default=100, help="shares per clip")
    lv.add_argument("--max-position", type=int, default=800, dest="max_position")
    lv.add_argument("--slippage", type=float, default=0.0, help="market-order slippage fraction")
    lv.add_argument("--seed", type=int, default=20240101, help="tick-path seed (reproducible)")
    lv.add_argument("--realtime", action="store_true", help="actually sleep between ticks")
    lv.set_defaults(func=cmd_live)

    return p


def cmd_advisor(args: argparse.Namespace) -> int:
    """Refresh advisory cache. Calls Claude if cache miss or --force."""
    try:
        advisor = ClaudeAdvisor()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    engine, settings, state = build_engine(args)
    symbols = _symbols(args, settings) or list(settings.watchlist)

    ledger = AuditLedger()
    results = []

    for symbol in symbols:
        print(f"{symbol}...", end=" ", flush=True)

        # Build feature vector from market data
        # TODO: fetch real market data (IV, prices, etc.)
        # For now, stub with synthetic data
        market = engine.market
        bid, ask = market.option_quote(symbol, 185, 35, "P")
        spot = market.price(symbol)

        fv = build_feature_vector(
            symbol=symbol,
            price=spot,
            iv_30d=0.28,  # placeholder
            iv_60d=0.30,
            iv_52w_low=0.15,
            iv_52w_high=0.45,
            rv_20d=0.22,
            rv_60d=0.24,
            skew_put_call=-0.05,
            shares_held=0,
            csp_open_count=0,
            ccall_open_count=0,
            avg_cost_per_share=0,
            cash_available=settings.starting_cash,
            collateral_used_pct=0.0,
            days_in_position_avg=0,
            dte_to_next_earnings=None,
            change_1d_pct=0.01,
            change_5d_pct=0.02,
            change_30d_pct=0.05,
            atr_20d=0.18,
        )

        result = advisor.advise(fv, force_refresh=args.force)
        results.append({symbol: result})
        status = "cache" if result.get("from_cache") else "fresh"
        print(f"✓ ({status})")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        stats = ledger.stats()
        print(f"\nAdvisor stats:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Total cost: ${stats['total_cost_usd']:.2f}")
        for r in results:
            for symbol, advice in r.items():
                print(f"\n{symbol}:")
                print(f"  Put delta:  {advice['put_delta_target']:.2f}")
                print(f"  Call delta: {advice['call_delta_target']:.2f}")
                print(f"  DTE target: {advice['dte_target']}")
                print(f"  Rationale: {advice['rationale']}")

    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Replay a synthetic tape through the paper live loop.

    ``--duration`` is *simulated* seconds at ``--interval`` per tick, so the
    60-second demo returns immediately unless ``--realtime`` is passed.
    """

    settings = Settings.from_env()
    assert_paper_mode(settings.mode)
    symbol = args.symbol.upper()
    interval = max(0.01, args.interval)
    ticks = max(1, int(round(args.duration / interval)))

    if args.spot:
        spot = float(args.spot)
    else:
        market = SyntheticMarketData(rate=settings.params.risk_free_rate)
        spot = market.get_quote(symbol, _as_of(args.date)).price

    engine = build_paper_engine(
        symbol,
        start_price=spot,
        ticks=ticks,
        seed=args.seed,
        band=args.band,
        qty=args.qty,
        max_position=args.max_position,
        shares=args.shares,
        slippage_pct=args.slippage,
    )
    engine.run_loop(interval=interval if args.realtime else 0.0)

    report = engine.report()
    report["symbol"] = symbol
    report["anchor"] = round(spot, 4)
    report["ticks_planned"] = ticks
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    m = engine.metrics
    fills = report["fills"]
    print(BANNER)
    print(f"live (PAPER) {symbol}  anchor {spot:,.2f}  {ticks} ticks @ {interval:g}s")
    print("-" * 60)
    for label, value in (
        ("ticks", m.ticks),
        ("signals", m.signals),
        ("orders submitted", m.orders_submitted),
        ("orders rejected", m.orders_rejected),
        ("risk rejections", m.risk_rejections),
        ("fills", m.fills),
        ("shares filled", m.filled_qty),
        ("dropped ticks", m.dropped_ticks),
    ):
        print(f"  {label:<18} {value:>10,}")
    print("-" * 60)
    print(f"  {'position':<18} {engine.position(symbol):>10,}")
    print(f"  {'fill VWAP':<18} {fills['avg_price']:>10,.4f}")
    print(f"  {'traded notional':<18} {fills['notional']:>10,.2f}")
    print(f"  {'realised slippage':<18} "
          f"{engine.tracker.realized_slippage(symbol, spot):>10,.4f}")
    if report["open_orders"]:
        print(f"  {'open orders':<18} {len(report['open_orders']):>10,}")
    return 0


def _load_env_file(path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ.
    
    Only sets values that are not already in os.environ (shell env takes
    precedence). Lines starting with # are ignored; empty lines are skipped.
    No quote handling — values are used verbatim after stripping whitespace.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass  # .env is optional


def main(argv: list[str] | None = None) -> int:
    _load_env_file()  # Load .env before parsing args
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
