"""The engine: turn market data + account state into recommendations, then
(optionally) execute them against the paper broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .broker import PaperBroker
from .config import Settings, StrategyParams, assert_paper_mode
from .greeks import CALL, PUT
from .marketdata import MarketData, price_contract
from .models import Action, OptionContract, OptionPosition, Quote, Recommendation, TradeRecord
from .strategy import Candidate, WheelStrategy


@dataclass
class CycleResult:
    as_of: date
    recommendations: list[Recommendation] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    settlements: list[TradeRecord] = field(default_factory=list)

    def actionable(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.actionable]


class WheelEngine:
    """Single-account, paper-only wheel runner."""

    def __init__(
        self,
        market: MarketData,
        broker: PaperBroker,
        params: StrategyParams | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            assert_paper_mode(settings.mode)
        assert_paper_mode(broker.mode)
        self.market = market
        self.broker = broker
        self.params = params or (settings.params if settings else StrategyParams())
        self.settings = settings
        self.strategy = WheelStrategy(self.params)

    # ------------------------------------------------------------------
    # marks
    # ------------------------------------------------------------------
    def quotes(self, symbols: list[str], as_of: date) -> dict[str, Quote]:
        return {s.upper(): self.market.get_quote(s, as_of) for s in symbols}

    def mark_option(self, pos: OptionPosition, quote: Quote, as_of: date, chain: list[OptionContract]) -> float:
        for c in chain:
            if c.symbol == pos.symbol:
                return c.mid
        synthetic = OptionContract(
            underlying=pos.underlying, expiry=pos.expiry, strike=pos.strike, right=pos.right,
            bid=0.0, ask=0.0, iv=quote.iv, as_of=as_of,
        )
        return round(
            price_contract(synthetic, quote.price, as_of, self.params.risk_free_rate, quote.div_yield), 2
        )

    def _universe(self, symbols: list[str] | None) -> list[str]:
        held = {p.symbol for p in self.broker.equities.values()}
        held |= {p.underlying for p in self.broker.options.values()}
        watch = set(s.upper() for s in (symbols or (self.settings.watchlist if self.settings else ())))
        return sorted(held | watch)

    # ------------------------------------------------------------------
    # scan
    # ------------------------------------------------------------------
    def scan(self, symbols: list[str] | None = None, as_of: date | None = None) -> list[Recommendation]:
        as_of = as_of or date.today()
        recs: list[Recommendation] = []
        prices = {
            s: self.market.get_quote(s, as_of).price for s in self._universe(symbols)
        }
        equity = self.broker.net_liquidation(prices)
        # Buying power is consumed as we go: two symbols must not both plan to
        # collateralise puts with the same dollars.
        avail = self.broker.available_cash()

        for symbol in self._universe(symbols):
            quote = self.market.get_quote(symbol, as_of)
            chain = self.market.get_chain(symbol, as_of)
            open_shorts = self.broker.open_short_options(symbol)

            if open_shorts:
                for pos in open_shorts:
                    mark = self.mark_option(pos, quote, as_of, chain)
                    decision = self.strategy.manage(pos, mark, as_of)
                    recs.append(
                        Recommendation(
                            symbol=symbol,
                            action=decision.action,
                            rationale=decision.reason,
                            spot=quote.price,
                            contracts=abs(pos.quantity),
                            credit=pos.credit_received(),
                            buyback_target=round(pos.average_price * (1 - self.params.profit_target), 2),
                            collateral=pos.collateral(quote.price),
                            collateral_pct=round(pos.collateral(quote.price) / equity, 4) if equity else 0.0,
                            confidence=0.9 if decision.action != Action.HOLD else 0.6,
                            risks=("assignment risk" if decision.dte <= self.params.roll_dte else "time risk",),
                            greeks=None,
                        )
                    )
                continue

            shares = self.broker.shares_held(symbol)
            if shares >= self.params.contract_multiplier:
                eq = self.broker.equities[symbol]
                cand = self.strategy.select_covered_call(chain, quote, as_of, shares, eq.average_cost)
                if cand is None:
                    recs.append(
                        Recommendation(
                            symbol=symbol, action=Action.SKIP, spot=quote.price,
                            rationale=(
                                "no call passed the filters "
                                f"(|delta|~{self.params.target_delta:.2f}+/-{self.params.delta_tolerance:.2f}, "
                                f"{self.params.min_dte}-{self.params.max_dte} DTE, strike >= "
                                f"basis ${eq.average_cost:.2f}, liquid)"
                            ),
                        )
                    )
                else:
                    recs.append(
                        self._build_rec(
                            Action.SELL_COVERED_CALL, symbol, quote, cand,
                            contracts=shares // self.params.contract_multiplier, equity=equity,
                        )
                    )
                continue

            cand = self.strategy.select_cash_secured_put(chain, quote, as_of, avail)
            if cand is None:
                recs.append(
                    Recommendation(
                        symbol=symbol, action=Action.SKIP, spot=quote.price,
                        rationale=(
                            f"no put passed the filters or uncommitted cash ${avail:,.0f} "
                            "cannot collateralise one contract"
                        ),
                    )
                )
                continue

            max_by_cash = int(avail // (cand.contract.strike * self.params.contract_multiplier))
            max_by_risk = (
                int((equity * self.params.max_collateral_pct)
                    // (cand.contract.strike * self.params.contract_multiplier))
                if equity > 0 else 0
            )
            contracts = max(min(max_by_cash, max_by_risk), 0)
            if contracts == 0:
                recs.append(
                    Recommendation(
                        symbol=symbol, action=Action.SKIP, spot=quote.price,
                        rationale=(
                            f"collateral ${cand.contract.strike * 100:,.0f}/contract exceeds the "
                            f"{self.params.max_collateral_pct:.0%} equity cap (${equity:,.0f} NAV)"
                        ),
                    )
                )
                continue
            avail = round(
                avail - contracts * cand.contract.strike * self.params.contract_multiplier, 2
            )
            recs.append(
                self._build_rec(Action.SELL_CASH_SECURED_PUT, symbol, quote, cand, contracts, equity)
            )
        return recs

    def _build_rec(
        self, action: Action, symbol: str, quote: Quote, cand: Candidate, contracts: int, equity: float
    ) -> Recommendation:
        mult = self.params.contract_multiplier
        c = cand.contract
        collateral = (
            contracts * c.strike * mult if c.is_put() else contracts * quote.price * mult
        )
        return Recommendation(
            symbol=symbol,
            action=action,
            rationale=(
                f"sell {contracts}x {c.expiry:%Y-%m-%d} ${c.strike:g} "
                f"{'call' if c.is_call() else 'put'} @ ${c.mid:.2f} "
                f"({c.dte} DTE, |delta|={abs(cand.greeks.delta):.2f}, "
                f"{cand.annualized_yield:.1%} annualized)"
            ),
            spot=quote.price,
            contract=c,
            contracts=contracts,
            greeks=cand.greeks,
            credit=round(cand.credit_per_contract * contracts, 2),
            buyback_target=round(c.mid * (1 - self.params.profit_target), 2),
            collateral=round(collateral, 2),
            collateral_pct=round(collateral / equity, 4) if equity else 0.0,
            annualized_yield=cand.annualized_yield,
            confidence=self.strategy.confidence(cand, quote),
            risks=self.strategy.risks(cand, quote),
        )

    # ------------------------------------------------------------------
    # execution (simulated)
    # ------------------------------------------------------------------
    def execute(self, recs: list[Recommendation], as_of: date | None = None) -> list[TradeRecord]:
        as_of = as_of or date.today()
        trades: list[TradeRecord] = []
        closed: set[str] = set()
        for rec in recs:
            if not rec.actionable:
                continue
            if rec.action in (Action.SELL_COVERED_CALL, Action.SELL_CASH_SECURED_PUT):
                if rec.contract is None or rec.contracts <= 0:
                    continue
                trades.append(
                    self.broker.sell_to_open(rec.contract, rec.contracts, note=rec.action.value)
                )
            elif rec.action in (Action.BUY_TO_CLOSE, Action.ROLL):
                quote = self.market.get_quote(rec.symbol, as_of)
                chain = self.market.get_chain(rec.symbol, as_of)
                for pos in self.broker.open_short_options(rec.symbol):
                    if pos.symbol in closed:
                        continue
                    mark = self.mark_option(pos, quote, as_of, chain)
                    # Re-confirm per position: one symbol can hold several shorts.
                    decision = self.strategy.manage(pos, mark, as_of)
                    if decision.action not in (Action.BUY_TO_CLOSE, Action.ROLL):
                        continue
                    closed.add(pos.symbol)
                    trades.append(
                        self.broker.buy_to_close(pos.symbol, abs(pos.quantity), mark, note=rec.action.value)
                    )
        return trades

    def settle(self, as_of: date | None = None) -> list[TradeRecord]:
        as_of = as_of or date.today()
        return self.broker.process_expirations(
            as_of, lambda sym: self.market.get_quote(sym, as_of).price
        )

    def run_cycle(
        self, symbols: list[str] | None = None, as_of: date | None = None, execute: bool = True
    ) -> CycleResult:
        as_of = as_of or date.today()
        settlements = self.settle(as_of)
        recs = self.scan(symbols, as_of)
        trades = self.execute(recs, as_of) if execute else []
        return CycleResult(as_of=as_of, recommendations=recs, trades=trades, settlements=settlements)

    # ------------------------------------------------------------------
    # portfolio
    # ------------------------------------------------------------------
    def portfolio(self, as_of: date | None = None) -> dict:
        as_of = as_of or date.today()
        symbols = self._universe(None)
        prices: dict[str, float] = {}
        marks: dict[str, float] = {}
        for sym in symbols:
            quote = self.market.get_quote(sym, as_of)
            prices[sym] = quote.price
            chain = self.market.get_chain(sym, as_of)
            for pos in self.broker.open_short_options(sym):
                marks[pos.symbol] = self.mark_option(pos, quote, as_of, chain)

        nav = self.broker.net_liquidation(prices, marks)
        open_credit = round(sum(p.credit_received() for p in self.broker.open_short_options()), 2)
        return {
            "as_of": as_of.isoformat(),
            "account_id": self.broker.account_id,
            "mode": self.broker.mode,
            "cash": round(self.broker.cash, 2),
            "equity_value": self.broker.equity_value(prices),
            "option_value": self.broker.option_value(marks),
            "net_liquidation": nav,
            "realized_pnl": round(self.broker.realized_pnl, 2),
            "open_credit": open_credit,
            "collateral_used": self.broker.collateral_used(prices),
            "positions": {
                "equities": [
                    {
                        **p.to_dict(),
                        "price": prices.get(sym, p.average_cost),
                        "unrealized": p.unrealized(prices.get(sym, p.average_cost)),
                    }
                    for sym, p in self.broker.equities.items()
                ],
                "options": [
                    {
                        **p.to_dict(),
                        "mark": marks.get(p.symbol, 0.0),
                        "dte": p.dte(as_of),
                        "captured": round(p.profit_captured(marks.get(p.symbol, 0.0)), 4),
                        "unrealized": p.unrealized(marks.get(p.symbol, 0.0)),
                    }
                    for p in self.broker.options.values()
                ],
            },
        }


__all__ = ["CycleResult", "WheelEngine", "CALL", "PUT"]
