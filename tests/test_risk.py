"""Unit + end-to-end tests for the risk layer."""

from datetime import date, timedelta

import pytest
from factories import AS_OF, IV, SPOT, make_quote
from wheel.models import EquityPosition, OptionPosition
from wheel.risk import (
    DrawdownMonitor,
    PortfolioConstraints,
    PortfolioExposure,
    PositionLimits,
    RiskAggregator,
    RiskBreach,
    RiskReport,
    Severity,
)

EXPIRY = AS_OF + timedelta(days=35)


def short_put(strike: float = 47.0, qty: int = -2, underlying: str = "TEST") -> OptionPosition:
    return OptionPosition(
        underlying=underlying,
        expiry=EXPIRY,
        strike=strike,
        right="P",
        quantity=qty,
        average_price=1.10,
        opened_at=AS_OF,
    )


def short_call(strike: float = 53.0, qty: int = -2, underlying: str = "TEST") -> OptionPosition:
    return OptionPosition(
        underlying=underlying,
        expiry=EXPIRY,
        strike=strike,
        right="C",
        quantity=qty,
        average_price=0.95,
        opened_at=AS_OF,
    )


# --------------------------------------------------------------------------
# PositionLimits
# --------------------------------------------------------------------------
class TestPositionLimits:
    def test_defaults_never_fire(self):
        limits = PositionLimits()
        huge = PortfolioExposure(notional=1e12, delta=1e9, gamma=1e6, vega=1e6, contracts=10**6)
        assert limits.check(huge) == []
        assert limits.allows(huge)

    def test_notional_breach(self):
        limits = PositionLimits(max_notional=10_000)
        breaches = limits.check(PortfolioExposure(notional=12_500))
        assert [b.code for b in breaches] == ["max_notional"]
        assert breaches[0].severity is Severity.BREACH
        assert breaches[0].observed == 12_500
        assert breaches[0].limit == 10_000
        assert not limits.allows(PortfolioExposure(notional=12_500))

    def test_notional_within_limit_is_clean(self):
        limits = PositionLimits(max_notional=10_000)
        assert limits.check(PortfolioExposure(notional=5_000)) == []

    def test_warning_band_just_under_cap(self):
        limits = PositionLimits(max_notional=10_000, warn_ratio=0.8)
        breaches = limits.check(PortfolioExposure(notional=8_500))
        assert len(breaches) == 1
        assert breaches[0].severity is Severity.WARN
        # A warning is not a hard stop.
        assert limits.allows(PortfolioExposure(notional=8_500))

    def test_delta_limit_uses_absolute_value(self):
        limits = PositionLimits(max_delta_exposure=500)
        short = limits.check(PortfolioExposure(delta=-620))
        long = limits.check(PortfolioExposure(delta=620))
        assert [b.code for b in short] == ["max_delta_exposure"]
        assert [b.code for b in long] == ["max_delta_exposure"]
        assert short[0].observed == -620

    def test_gamma_and_vega_and_contracts(self):
        limits = PositionLimits(
            max_gamma_exposure=5.0, max_vega_exposure=100.0, max_contracts=10
        )
        exposure = PortfolioExposure(gamma=9.0, vega=-250.0, contracts=25)
        codes = sorted(b.code for b in limits.check(exposure))
        assert codes == ["max_contracts", "max_gamma_exposure", "max_vega_exposure"]

    def test_multiple_limits_report_independently(self):
        limits = PositionLimits(max_notional=1_000, max_delta_exposure=10)
        breaches = limits.check(PortfolioExposure(notional=9_999, delta=99))
        assert len(breaches) == 2

    def test_utilisation(self):
        b = RiskBreach(code="x", message="", limit=100.0, observed=-150.0)
        assert b.utilisation == pytest.approx(1.5)
        assert RiskBreach("x", "", 0.0, 0.0).utilisation == 0.0
        assert b.to_dict()["severity"] == "breach"

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError):
            PositionLimits(max_notional=-1)
        with pytest.raises(ValueError):
            PositionLimits(warn_ratio=0.0)
        with pytest.raises(ValueError):
            PositionLimits(warn_ratio=1.5)


# --------------------------------------------------------------------------
# DrawdownMonitor
# --------------------------------------------------------------------------
class TestDrawdownMonitor:
    def test_monotonic_rise_has_no_drawdown(self):
        m = DrawdownMonitor.from_series([100, 110, 125, 140], threshold=0.10)
        assert m.current_drawdown == 0.0
        assert m.max_drawdown == 0.0
        assert m.peak == 140
        assert not m.triggered

    def test_simple_decline(self):
        m = DrawdownMonitor.from_series([100, 120, 90], threshold=0.50)
        assert m.peak == 120
        assert m.current_drawdown == pytest.approx(0.25)
        assert m.max_drawdown == pytest.approx(0.25)

    def test_partial_recovery_keeps_historical_max(self):
        m = DrawdownMonitor.from_series([100, 200, 100, 150], threshold=0.90)
        assert m.current_drawdown == pytest.approx(0.25)  # 150 vs peak 200
        assert m.max_drawdown == pytest.approx(0.50)  # trough 100 vs peak 200

    def test_new_peak_resets_trough(self):
        m = DrawdownMonitor(threshold=0.50)
        m.extend([100, 80, 250])
        assert m.peak == 250
        assert m.trough == 250
        assert m.current_drawdown == 0.0
        assert m.max_drawdown == pytest.approx(0.20)  # the 100 -> 80 leg survives

    def test_trigger_at_threshold(self):
        m = DrawdownMonitor(threshold=0.20)
        m.extend([1000, 900])
        assert not m.triggered
        m.update(800)
        assert m.triggered
        assert m.current_drawdown == pytest.approx(0.20)

    def test_check_severities(self):
        m = DrawdownMonitor(threshold=0.20)
        m.extend([1000, 990])
        assert m.check() == []  # 1% drawdown

        m.update(835)  # 16.5% -- inside the 80% warn band
        warn = m.check()
        assert len(warn) == 1 and warn[0].severity is Severity.WARN

        m.update(700)  # 30%
        hard = m.check()
        assert len(hard) == 1 and hard[0].severity is Severity.BREACH
        assert hard[0].code == "max_drawdown"

    def test_update_returns_current_drawdown(self):
        m = DrawdownMonitor(threshold=0.30)
        assert m.update(100) == 0.0
        assert m.update(75) == pytest.approx(0.25)

    def test_reset(self):
        m = DrawdownMonitor.from_series([100, 50], threshold=0.10)
        assert m.triggered
        m.reset(200)
        assert not m.triggered
        assert m.peak == 200
        assert m.max_drawdown == 0.0

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            DrawdownMonitor(threshold=0.0)
        with pytest.raises(ValueError):
            DrawdownMonitor(threshold=1.5)
        with pytest.raises(ValueError):
            DrawdownMonitor().update(-5)

    def test_empty_monitor_is_flat(self):
        m = DrawdownMonitor()
        assert m.current_drawdown == 0.0
        assert not m.triggered
        assert m.check() == []


# --------------------------------------------------------------------------
# PortfolioConstraints
# --------------------------------------------------------------------------
class TestPortfolioConstraints:
    def test_sharpe_of_constant_positive_returns(self):
        # Zero variance -> undefined ratio; we return 0.0 rather than inf.
        assert PortfolioConstraints.sharpe_ratio([0.01] * 10) == 0.0

    def test_sharpe_positive_for_upward_drift(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.008, 0.012]
        assert PortfolioConstraints.sharpe_ratio(returns) > 0

    def test_sharpe_negative_for_losing_series(self):
        returns = [-0.01, -0.02, 0.005, -0.015, -0.008]
        assert PortfolioConstraints.sharpe_ratio(returns) < 0

    def test_sharpe_risk_free_drag(self):
        returns = [0.001] * 5 + [0.002] * 5
        rich = PortfolioConstraints.sharpe_ratio(returns, risk_free_rate=0.0)
        poor = PortfolioConstraints.sharpe_ratio(returns, risk_free_rate=0.50)
        assert poor < rich

    def test_sharpe_needs_two_points(self):
        with pytest.raises(ValueError):
            PortfolioConstraints.sharpe_ratio([0.01])

    def test_var_is_positive_loss_magnitude(self):
        returns = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        var = PortfolioConstraints.value_at_risk(returns, confidence=0.90)
        assert var > 0
        assert var == pytest.approx(0.055, abs=1e-9)

    def test_var_zero_when_no_losses(self):
        assert PortfolioConstraints.value_at_risk([0.01, 0.02, 0.03]) == 0.0

    def test_var_bad_inputs(self):
        with pytest.raises(ValueError):
            PortfolioConstraints.value_at_risk([])
        with pytest.raises(ValueError):
            PortfolioConstraints.value_at_risk([0.01, 0.02], confidence=1.0)

    def test_correlation_perfect_and_inverse(self):
        a = [1.0, 2.0, 3.0, 4.0]
        assert PortfolioConstraints.correlation(a, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
        assert PortfolioConstraints.correlation(a, [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_correlation_constant_series_is_zero(self):
        assert PortfolioConstraints.correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0

    def test_correlation_bad_inputs(self):
        with pytest.raises(ValueError):
            PortfolioConstraints.correlation([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            PortfolioConstraints.correlation([1.0], [1.0])

    def test_check_flags_sharpe_floor(self):
        c = PortfolioConstraints(min_sharpe=5.0)
        breaches = c.check([-0.01, -0.02, 0.005, -0.015])
        assert [b.code for b in breaches] == ["min_sharpe"]

    def test_check_passes_when_sharpe_clears_floor(self):
        c = PortfolioConstraints(min_sharpe=-10.0)
        assert c.check([0.01, 0.02, -0.005, 0.015]) == []

    def test_check_flags_var(self):
        c = PortfolioConstraints(max_var=0.02, confidence=0.90)
        returns = [-0.25, -0.20, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        codes = [b.code for b in c.check(returns)]
        assert "max_var" in codes

    def test_check_flags_correlation(self):
        c = PortfolioConstraints(correlation_bound=0.6)
        breaches = c.check(correlations={"GLD/SLV": 0.91, "GLD/TLT": -0.12})
        assert [b.code for b in breaches] == ["correlation_bound"]
        assert breaches[0].observed == pytest.approx(0.91)

    def test_check_multi_leg_portfolio_all_clear(self):
        """Two legs whose blended returns are healthy on every axis."""

        leg_a = [0.012, 0.008, -0.003, 0.015, 0.006, 0.011, 0.004, 0.009]
        leg_b = [0.004, -0.002, 0.009, 0.003, 0.012, 0.001, 0.007, 0.005]
        blended = [(x + y) / 2 for x, y in zip(leg_a, leg_b)]
        corr = PortfolioConstraints.correlation(leg_a, leg_b)
        c = PortfolioConstraints(min_sharpe=0.0, max_var=0.05, correlation_bound=0.95)
        assert c.check(blended, {"a/b": corr}) == []

    def test_check_multi_leg_portfolio_breaches_stack(self):
        leg = [-0.08, -0.06, -0.09, -0.07, -0.05]
        c = PortfolioConstraints(min_sharpe=1.0, max_var=0.01, correlation_bound=0.5)
        codes = sorted(b.code for b in c.check(leg, {"a/b": 0.99}))
        assert codes == ["correlation_bound", "max_var", "min_sharpe"]

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError):
            PortfolioConstraints(confidence=1.0)
        with pytest.raises(ValueError):
            PortfolioConstraints(correlation_bound=1.5)
        with pytest.raises(ValueError):
            PortfolioConstraints(max_var=-0.1)
        with pytest.raises(ValueError):
            PortfolioConstraints(periods_per_year=0)


# --------------------------------------------------------------------------
# RiskAggregator
# --------------------------------------------------------------------------
class TestRiskAggregator:
    quotes = {"TEST": make_quote("TEST", SPOT, IV)}

    def test_empty_book_is_flat(self):
        agg = RiskAggregator()
        exposure = agg.exposure(quotes=self.quotes, as_of=AS_OF)
        assert exposure == PortfolioExposure()

    def test_shares_contribute_one_delta_each(self):
        agg = RiskAggregator()
        exposure = agg.exposure(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=[EquityPosition("TEST", 800, 48.0)],
        )
        assert exposure.delta == pytest.approx(800.0)
        assert exposure.shares == 800
        assert exposure.notional == pytest.approx(800 * SPOT)
        assert exposure.gamma == 0.0

    def test_short_put_is_long_delta_short_gamma(self):
        agg = RiskAggregator()
        exposure = agg.exposure(
            quotes=self.quotes, as_of=AS_OF, option_positions=[short_put()]
        )
        assert exposure.delta > 0  # short put = bullish
        assert exposure.gamma < 0  # short options are short gamma
        assert exposure.theta > 0  # and collect decay
        assert exposure.vega < 0
        assert exposure.contracts == 2

    def test_short_call_is_short_delta(self):
        agg = RiskAggregator()
        exposure = agg.exposure(
            quotes=self.quotes, as_of=AS_OF, option_positions=[short_call()]
        )
        assert exposure.delta < 0
        assert exposure.gamma < 0

    def test_covered_call_nets_toward_flat(self):
        """800 shares + 8 short calls should be materially less long than shares alone."""

        agg = RiskAggregator()
        shares = [EquityPosition("TEST", 800, 48.0)]
        exposure = agg.exposure(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=shares,
            option_positions=[short_call(qty=-8)],
        )
        assert 0 < exposure.delta < 800
        assert exposure.shares == 800
        assert exposure.contracts == 8

    def test_multi_leg_greeks_are_additive(self):
        agg = RiskAggregator()
        legs = [short_put(), short_call()]
        combined = agg.exposure(quotes=self.quotes, as_of=AS_OF, option_positions=legs)
        parts = [
            agg.exposure(quotes=self.quotes, as_of=AS_OF, option_positions=[leg]) for leg in legs
        ]
        assert combined.delta == pytest.approx(parts[0].delta + parts[1].delta)
        assert combined.gamma == pytest.approx(parts[0].gamma + parts[1].gamma)
        assert combined.contracts == 4

    def test_zero_quantity_positions_ignored(self):
        agg = RiskAggregator()
        exposure = agg.exposure(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=[EquityPosition("TEST", 0, 0.0)],
            option_positions=[short_put(qty=0)],
        )
        assert exposure == PortfolioExposure()

    def test_missing_quote_raises(self):
        agg = RiskAggregator()
        with pytest.raises(KeyError):
            agg.exposure(
                quotes={}, as_of=AS_OF, option_positions=[short_put(underlying="NOPE")]
            )

    def test_evaluate_clean_book(self):
        agg = RiskAggregator(
            limits=PositionLimits(max_notional=1e9, max_delta_exposure=1e6),
            constraints=PortfolioConstraints(min_sharpe=-10.0, max_var=0.90),
            drawdown=DrawdownMonitor(threshold=0.25),
        )
        report = agg.evaluate(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=[EquityPosition("TEST", 800, 48.0)],
            option_positions=[short_call(qty=-8)],
            equity=100_000,
            returns=[0.01, 0.005, -0.002, 0.008],
        )
        assert isinstance(report, RiskReport)
        assert report.ok
        assert report.breaches == ()
        assert report.drawdown == 0.0
        assert report.exposure.shares == 800

    def test_evaluate_end_to_end_breaches(self):
        agg = RiskAggregator(
            limits=PositionLimits(max_notional=1_000, max_delta_exposure=50),
            constraints=PortfolioConstraints(min_sharpe=3.0, max_var=0.01, correlation_bound=0.5),
            drawdown=DrawdownMonitor(threshold=0.10),
        )
        agg.drawdown.update(100_000)
        report = agg.evaluate(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=[EquityPosition("TEST", 800, 48.0)],
            equity=80_000,  # -20% from peak
            returns=[-0.05, -0.04, -0.06, -0.03],
            correlations={"TEST/SPY": 0.97},
        )
        codes = set(report.codes)
        assert {"max_notional", "max_delta_exposure", "max_drawdown", "min_sharpe", "max_var", "correlation_bound"} <= codes
        assert not report.ok
        assert report.drawdown == pytest.approx(0.20)
        assert len(report.hard_breaches) >= 6

    def test_evaluate_tracks_drawdown_across_calls(self):
        agg = RiskAggregator(drawdown=DrawdownMonitor(threshold=0.50))
        for equity in (100.0, 120.0, 90.0):
            report = agg.evaluate(quotes=self.quotes, as_of=AS_OF, equity=equity)
        assert report.drawdown == pytest.approx(0.25)
        assert agg.drawdown.peak == 120.0

    def test_warnings_do_not_fail_the_report(self):
        agg = RiskAggregator(limits=PositionLimits(max_notional=50_000, warn_ratio=0.5))
        report = agg.evaluate(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=[EquityPosition("TEST", 800, 48.0)],  # 40k notional
        )
        assert report.warnings
        assert report.ok

    def test_report_serialises(self):
        agg = RiskAggregator(limits=PositionLimits(max_delta_exposure=10))
        report = agg.evaluate(
            quotes=self.quotes,
            as_of=AS_OF,
            equity_positions=[EquityPosition("TEST", 100, 48.0)],
        )
        d = report.to_dict()
        assert d["ok"] is False
        assert d["exposure"]["shares"] == 100
        assert d["breaches"][0]["code"] == "max_delta_exposure"

    def test_expired_option_has_intrinsic_delta_only(self):
        agg = RiskAggregator()
        expired = OptionPosition(
            underlying="TEST",
            expiry=AS_OF,
            strike=45.0,
            right="C",
            quantity=-1,
            average_price=5.0,
            opened_at=AS_OF - timedelta(days=30),
        )
        exposure = agg.exposure(
            quotes=self.quotes, as_of=AS_OF, option_positions=[expired]
        )
        assert exposure.delta == pytest.approx(-100.0)  # deep ITM short call
        assert exposure.gamma == 0.0


def test_as_of_in_the_future_shrinks_theta():
    """Closer to expiry -> a short option's positive theta grows."""

    agg = RiskAggregator()
    quotes = {"TEST": make_quote("TEST", SPOT, IV)}
    far = agg.exposure(quotes=quotes, as_of=AS_OF, option_positions=[short_put()])
    near = agg.exposure(
        quotes=quotes, as_of=AS_OF + timedelta(days=28), option_positions=[short_put()]
    )
    assert near.theta > far.theta > 0
