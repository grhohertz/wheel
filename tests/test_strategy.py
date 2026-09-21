from datetime import date, timedelta

import pytest
from factories import AS_OF, SPOT, make_chain, make_quote
from wheel.config import StrategyParams
from wheel.models import Action, OptionPosition
from wheel.strategy import WheelStrategy


@pytest.fixture()
def strategy() -> WheelStrategy:
    return WheelStrategy(StrategyParams())


def test_covered_call_targets_delta_and_dte(strategy):
    chain, quote = make_chain(), make_quote()
    cand = strategy.select_covered_call(chain, quote, AS_OF, shares_held=100, cost_basis=45.0)
    assert cand is not None
    assert cand.contract.is_call()
    assert 30 <= cand.contract.dte <= 45
    assert abs(cand.greeks.delta) == pytest.approx(0.30, abs=0.15)
    assert cand.contract.strike > SPOT  # OTM only
    assert cand.credit_per_contract > 0
    assert cand.annualized_yield > 0


def test_covered_call_requires_100_shares(strategy):
    cand = strategy.select_covered_call(make_chain(), make_quote(), AS_OF, shares_held=99)
    assert cand is None


def test_covered_call_never_sells_below_cost_basis(strategy):
    chain, quote = make_chain(), make_quote()
    baseline = strategy.select_covered_call(chain, quote, AS_OF, shares_held=100, cost_basis=45.0)
    assert baseline is not None
    floor = baseline.contract.strike + 1.0
    cand = strategy.select_covered_call(chain, quote, AS_OF, shares_held=100, cost_basis=floor)
    assert cand is not None
    assert cand.contract.strike >= floor


def test_covered_call_skipped_when_basis_is_far_above_market(strategy):
    # Underwater shares: every strike at/above basis is too far OTM to pay.
    cand = strategy.select_covered_call(make_chain(), make_quote(), AS_OF, 100, cost_basis=75.0)
    assert cand is None


def test_cash_secured_put_is_otm_and_collateral_capped(strategy):
    chain, quote = make_chain(), make_quote()
    cand = strategy.select_cash_secured_put(chain, quote, AS_OF, buying_power=100_000)
    assert cand is not None
    assert cand.contract.is_put()
    assert cand.contract.strike <= SPOT
    assert 30 <= cand.contract.dte <= 45
    assert abs(cand.greeks.delta) == pytest.approx(0.30, abs=0.15)


def test_cash_secured_put_blocked_without_collateral(strategy):
    assert strategy.select_cash_secured_put(make_chain(), make_quote(), AS_OF, buying_power=100.0) is None


def test_illiquid_chain_is_rejected(strategy):
    chain = make_chain(open_interest=1)  # below min_open_interest
    assert strategy.select_covered_call(chain, make_quote(), AS_OF, 100, 40.0) is None
    assert strategy.select_cash_secured_put(chain, make_quote(), AS_OF, 100_000) is None


def _short_call(avg: float = 2.00, dte: int = 35) -> OptionPosition:
    return OptionPosition(
        underlying="TEST", expiry=AS_OF + timedelta(days=dte), strike=53.0, right="C",
        quantity=-1, average_price=avg, opened_at=AS_OF,
    )


def test_manage_holds_when_theta_still_working(strategy):
    d = strategy.manage(_short_call(), mark=1.80, as_of=AS_OF)
    assert d.action is Action.HOLD
    assert d.captured == pytest.approx(0.10)


def test_manage_buys_back_at_profit_target(strategy):
    d = strategy.manage(_short_call(), mark=0.95, as_of=AS_OF)
    assert d.action is Action.BUY_TO_CLOSE
    assert d.captured >= 0.50


def test_manage_rolls_near_expiry(strategy):
    d = strategy.manage(_short_call(dte=5), mark=1.90, as_of=AS_OF)
    assert d.action is Action.ROLL
    assert d.dte == 5


def test_manage_settles_expired(strategy):
    d = strategy.manage(_short_call(dte=0), mark=0.02, as_of=AS_OF)
    assert d.action is Action.BUY_TO_CLOSE


def test_confidence_and_risks(strategy):
    chain, quote = make_chain(), make_quote(div=0.02)
    cand = strategy.select_covered_call(chain, quote, AS_OF, 100, 40.0)
    assert cand is not None
    conf = strategy.confidence(cand, quote)
    assert 0.05 <= conf <= 0.95
    risks = strategy.risks(cand, quote)
    assert "IV crush" in risks
    assert any("dividend" in r for r in risks)


def test_params_validation():
    with pytest.raises(ValueError):
        StrategyParams(target_delta=1.5)
    with pytest.raises(ValueError):
        StrategyParams(min_dte=60, max_dte=30)
    with pytest.raises(ValueError):
        StrategyParams(profit_target=0.0)
