"""Paper-trading wheel-strategy engine.

The wheel: sell cash-secured puts -> get assigned shares -> sell covered calls
against those shares -> get called away -> repeat.

Everything in this package is SIMULATED. No module here places a live order;
:class:`wheel.broker.PaperBroker` is the only broker implementation and
:func:`wheel.config.assert_paper_mode` hard-fails any non-paper configuration.
"""

from .advisor import Advisor, AdviceConfig
from .config import Settings, assert_paper_mode
from .greeks import Greeks, black_scholes, implied_vol
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
    "Advisor",
    "AdviceConfig",
    "EquityPosition",
    "Greeks",
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
