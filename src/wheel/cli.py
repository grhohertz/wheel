"""Command-line interface.

    python -m wheel.cli scan
    python -m wheel.cli run --execute
    python -m wheel.cli positions
    python -m wheel.cli ledger
    python -m wheel.cli price AAPL --strike 190 --dte 35
    python -m wheel.cli reset --cash 100000

Every command runs against the local paper account JSON (``--state``).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from .broker import PaperBroker
from .config import PAPER, Settings, assert_paper_mode
from .engine import WheelEngine
from .greeks import black_scholes, year_fraction
from .marketdata import SyntheticMarketData
from .report import BANNER, render_portfolio, render_scan, render_trades


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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
