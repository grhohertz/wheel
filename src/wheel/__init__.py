"""Paper-trading wheel-strategy engine.

The wheel: sell cash-secured puts -> get assigned shares -> sell covered calls
against those shares -> get called away -> repeat.

Everything in this package is SIMULATED. No module here places a live order;
:class:`wheel.broker.PaperBroker` is the only broker implementation and
:func:`wheel.config.assert_paper_mode` hard-fails any non-paper configuration.
"""

from .config import Settings, assert_paper_mode
from .policy import AdviceConfig, advisor_mode, get_delta_target, get_dte_target, get_mc_overlay
from .regime import REGIME_OVERLAYS, RegimeOverlay, compare_regimes, overlay_for, regime_scenarios
from .greeks import Greeks, black_scholes, implied_vol
from .feeds import FeedManager, MarketFeed, PaperFeed
from .feeds import Quote as TickQuote
from .execution import (
    ExecutionClient,
    Fill,
    FillTracker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperExecutionClient,
)
from .live import LiveEngine, LiveMetrics, band_signal, check_live_trading_disabled
from .risk import (
    DrawdownMonitor,
    PortfolioConstraints,
    PortfolioExposure,
    PositionLimits,
    RiskAggregator,
    RiskBreach,
    RiskReport,
    Severity,
)
from .models import (
    Action,
    EquityPosition,
    OptionContract,
    OptionPosition,
    Quote,
    Recommendation,
    TradeRecord,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "AdviceConfig",
    "REGIME_OVERLAYS",
    "RegimeOverlay",
    "advisor_mode",
    "compare_regimes",
    "get_delta_target",
    "get_dte_target",
    "get_mc_overlay",
    "overlay_for",
    "regime_scenarios",
    "DrawdownMonitor",
    "EquityPosition",
    "ExecutionClient",
    "FeedManager",
    "Fill",
    "FillTracker",
    "LiveEngine",
    "LiveMetrics",
    "MarketFeed",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperExecutionClient",
    "PaperFeed",
    "TickQuote",
    "band_signal",
    "check_live_trading_disabled",
    "Greeks",
    "PortfolioConstraints",
    "PortfolioExposure",
    "PositionLimits",
    "RiskAggregator",
    "RiskBreach",
    "RiskReport",
    "Severity",
    "OptionContract",
    "OptionPosition",
    "Quote",
    "Recommendation",
    "Settings",
    "TradeRecord",
    "TradeRecord",
    "assert_paper_mode",
    "black_scholes",
    "implied_vol",
    "__version__",
]
