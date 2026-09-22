"""Order execution.

:class:`PaperExecutionClient` is the ONLY client in this package -- it matches
orders against replayed ticks. No module here opens a socket or talks to a
broker. A real adapter would subclass :class:`ExecutionClient`; the live loop
would not change.
"""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from .feeds import Quote

__all__ = [
    "ExecutionClient",
    "ExecutionError",
    "Fill",
    "FillTracker",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperExecutionClient",
    "UnknownOrder",
]


class ExecutionError(RuntimeError):
    """Base class for execution problems."""


class UnknownOrder(ExecutionError, KeyError):
    """Raised when an order id is not known to the client."""


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class Fill:
    """One execution against an order."""

    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    price: float
    timestamp: float = field(default_factory=_now)

    @property
    def notional(self) -> float:
        return round(self.qty * self.price, 6)

    @property
    def signed_qty(self) -> int:
        return self.qty * self.side.sign

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "price": self.price,
            "notional": self.notional,
            "timestamp": self.timestamp,
        }


@dataclass
class Order:
    """A working (or terminal) order. ``qty`` is always a positive magnitude."""

    symbol: str
    qty: int
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_qty: int = 0
    avg_price: float = 0.0
    order_id: str = ""
    created_at: float = field(default_factory=_now)
    reason: str = ""

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        self.side = OrderSide(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        if self.qty <= 0:
            raise ValueError("order qty must be positive")
        if self.order_type is OrderType.LIMIT and (self.price is None or self.price <= 0):
            raise ValueError("limit orders require a positive price")
        if self.order_type is OrderType.MARKET:
            self.price = None

    # -- constructors ---------------------------------------------------
    @classmethod
    def market(cls, symbol: str, qty: int, side: OrderSide | str) -> "Order":
        return cls(symbol=symbol, qty=qty, side=OrderSide(side), order_type=OrderType.MARKET)

    @classmethod
    def limit(cls, symbol: str, qty: int, side: OrderSide | str, price: float) -> "Order":
        return cls(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side),
            order_type=OrderType.LIMIT,
            price=price,
        )

    # -- state ----------------------------------------------------------
    @property
    def remaining(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_done(self) -> bool:
        return not self.is_open

    @property
    def signed_filled_qty(self) -> int:
        return self.filled_qty * self.side.sign

    def apply_fill(self, fill: Fill) -> None:
        """Fold a fill into the order, updating VWAP and status."""

        if fill.qty <= 0:
            raise ValueError("fill qty must be positive")
        if fill.qty > self.remaining:
            raise ExecutionError(
                f"fill of {fill.qty} exceeds {self.remaining} remaining on {self.order_id}"
            )
        notional = self.avg_price * self.filled_qty + fill.price * fill.qty
        self.filled_qty += fill.qty
        self.avg_price = round(notional / self.filled_qty, 6)
        self.status = OrderStatus.FILLED if self.remaining == 0 else OrderStatus.PARTIALLY_FILLED

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "price": self.price,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "created_at": self.created_at,
            "reason": self.reason,
        }


class ExecutionClient(ABC):
    """Everything the live loop needs from a venue."""

    name: str = "client"

    @abstractmethod
    def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus: ...

    @abstractmethod
    def get_fills(self, order_id: str | None = None) -> list[Fill]: ...


class PaperExecutionClient(ExecutionClient):
    """Deterministic simulated matching against the last seen tick.

    * MARKET  -- fills immediately at ``last`` (adjusted by ``slippage_pct``).
    * LIMIT   -- rests until the quote crosses: a buy fills when ``ask <= limit``
      (at the better of the two), a sell fills when ``bid >= limit``.

    ``max_fill_qty`` caps the size of any single fill, which is how partial
    fills are produced.
    """

    name = "paper"

    def __init__(
        self,
        *,
        slippage_pct: float = 0.0,
        max_fill_qty: int | None = None,
        clock=_now,
    ) -> None:
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be >= 0")
        if max_fill_qty is not None and max_fill_qty < 1:
            raise ValueError("max_fill_qty must be >= 1")
        self.slippage_pct = slippage_pct
        self.max_fill_qty = max_fill_qty
        self._clock = clock
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._quotes: dict[str, Quote] = {}
        self._ids = itertools.count(1)

    # -- book -----------------------------------------------------------
    @property
    def orders(self) -> tuple[Order, ...]:
        return tuple(self._orders.values())

    @property
    def open_orders(self) -> tuple[Order, ...]:
        return tuple(o for o in self._orders.values() if o.is_open)

    def _next_id(self) -> str:
        return f"PO-{next(self._ids):06d}"

    def _quote(self, symbol: str) -> Quote | None:
        return self._quotes.get(symbol.upper())

    # -- ExecutionClient -------------------------------------------------
    def submit_order(self, order: Order) -> Order:
        if not order.order_id:
            order.order_id = self._next_id()
        if order.order_id in self._orders:
            raise ExecutionError(f"duplicate order id {order.order_id}")
        self._orders[order.order_id] = order
        quote = self._quote(order.symbol)
        if quote is None and order.order_type is OrderType.MARKET:
            order.status = OrderStatus.REJECTED
            order.reason = "no quote available for market order"
            return order
        if quote is not None:
            self._try_fill(order, quote)
        return order

    def cancel_order(self, order_id: str) -> Order:
        order = self._get(order_id)
        if order.is_open:
            order.status = OrderStatus.CANCELLED
            order.reason = "cancelled by client"
        return order

    def get_order_status(self, order_id: str) -> OrderStatus:
        return self._get(order_id).status

    def get_fills(self, order_id: str | None = None) -> list[Fill]:
        if order_id is None:
            return list(self._fills)
        self._get(order_id)
        return [f for f in self._fills if f.order_id == order_id]

    def get_order(self, order_id: str) -> Order:
        return self._get(order_id)

    def _get(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise UnknownOrder(f"unknown order {order_id!r}") from exc

    # -- matching --------------------------------------------------------
    def on_quote(self, quote: Quote) -> list[Fill]:
        """Record a tick and match every resting order on that symbol."""

        self._quotes[quote.symbol.upper()] = quote
        produced: list[Fill] = []
        for order in list(self._orders.values()):
            if order.is_open and order.symbol == quote.symbol.upper():
                produced.extend(self._try_fill(order, quote))
        return produced

    def _fill_price(self, order: Order, quote: Quote) -> float | None:
        if order.order_type is OrderType.MARKET:
            slip = quote.last * self.slippage_pct * order.side.sign
            return round(quote.last + slip, 6)
        limit = float(order.price or 0.0)
        if order.side is OrderSide.BUY:
            return round(min(limit, quote.ask), 6) if quote.ask <= limit else None
        return round(max(limit, quote.bid), 6) if quote.bid >= limit else None

    def _try_fill(self, order: Order, quote: Quote) -> list[Fill]:
        price = self._fill_price(order, quote)
        if price is None or price <= 0:
            return []
        qty = order.remaining
        if self.max_fill_qty is not None:
            qty = min(qty, self.max_fill_qty)
        if qty <= 0:
            return []
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            timestamp=self._clock(),
        )
        order.apply_fill(fill)
        self._fills.append(fill)
        return [fill]


@dataclass
class FillTracker:
    """Accumulates fills; computes VWAP and realised slippage."""

    fills: list[Fill] = field(default_factory=list)

    def add(self, fill: Fill) -> Fill:
        self.fills.append(fill)
        return fill

    def extend(self, fills) -> None:
        for fill in fills:
            self.add(fill)

    def __len__(self) -> int:
        return len(self.fills)

    def _select(self, symbol: str | None = None, side: OrderSide | None = None) -> list[Fill]:
        out = self.fills
        if symbol is not None:
            sym = symbol.upper()
            out = [f for f in out if f.symbol == sym]
        if side is not None:
            side = OrderSide(side)
            out = [f for f in out if f.side is side]
        return list(out)

    def filled_qty(self, symbol: str | None = None, side: OrderSide | None = None) -> int:
        return sum(f.qty for f in self._select(symbol, side))

    def net_qty(self, symbol: str | None = None) -> int:
        """Signed position delta implied by the fills (buys positive)."""

        return sum(f.signed_qty for f in self._select(symbol))

    def notional(self, symbol: str | None = None, side: OrderSide | None = None) -> float:
        return round(sum(f.notional for f in self._select(symbol, side)), 6)

    def avg_price(self, symbol: str | None = None, side: OrderSide | None = None) -> float:
        selected = self._select(symbol, side)
        qty = sum(f.qty for f in selected)
        if qty == 0:
            return 0.0
        return round(sum(f.qty * f.price for f in selected) / qty, 6)

    def realized_slippage(
        self,
        symbol: str,
        expected_price: float,
        side: OrderSide | None = None,
    ) -> float:
        """Cost of execution vs ``expected_price``. Positive == worse than expected."""

        selected = self._select(symbol, side)
        if not selected:
            return 0.0
        qty = sum(f.qty for f in selected)
        cost = sum(f.qty * (f.price - expected_price) * f.side.sign for f in selected)
        return round(cost / qty, 6)

    def summary(self, symbol: str | None = None) -> dict:
        return {
            "symbol": (symbol or "*").upper(),
            "fills": len(self._select(symbol)),
            "bought": self.filled_qty(symbol, OrderSide.BUY),
            "sold": self.filled_qty(symbol, OrderSide.SELL),
            "net_qty": self.net_qty(symbol),
            "avg_price": self.avg_price(symbol),
            "notional": self.notional(symbol),
        }
