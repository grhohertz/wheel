import pytest
from wheel.execution import (
    ExecutionError,
    Fill,
    FillTracker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperExecutionClient,
    UnknownOrder,
)
from wheel.feeds import Quote


def q(last: float, *, bid: float | None = None, ask: float | None = None, ts: float = 0.0) -> Quote:
    return Quote(
        symbol="GLD",
        bid=last - 0.05 if bid is None else bid,
        ask=last + 0.05 if ask is None else ask,
        last=last,
        volume=100,
        timestamp=ts,
    )


# -- Order ----------------------------------------------------------------


def test_order_market_constructor_normalises():
    o = Order.market("gld", 100, "BUY")
    assert o.symbol == "GLD"
    assert o.side is OrderSide.BUY
    assert o.order_type is OrderType.MARKET
    assert o.price is None
    assert o.status is OrderStatus.NEW
    assert o.remaining == 100 and o.is_open


def test_order_limit_requires_price():
    with pytest.raises(ValueError):
        Order(symbol="GLD", qty=1, side=OrderSide.BUY, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        Order.market("GLD", 0, OrderSide.BUY)


def test_order_apply_fill_computes_vwap_and_status():
    o = Order.market("GLD", 100, OrderSide.BUY)
    o.order_id = "X1"
    o.apply_fill(Fill("X1", "GLD", OrderSide.BUY, 40, 100.0))
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert o.filled_qty == 40
    o.apply_fill(Fill("X1", "GLD", OrderSide.BUY, 60, 110.0))
    assert o.status is OrderStatus.FILLED
    assert o.avg_price == pytest.approx(106.0)
    assert o.remaining == 0 and o.is_done


def test_order_overfill_rejected():
    o = Order.market("GLD", 10, OrderSide.SELL)
    o.order_id = "X2"
    with pytest.raises(ExecutionError):
        o.apply_fill(Fill("X2", "GLD", OrderSide.SELL, 11, 100.0))


# -- PaperExecutionClient --------------------------------------------------


def test_market_order_fills_immediately_at_last():
    client = PaperExecutionClient(clock=lambda: 1.0)
    client.on_quote(q(100.0))
    order = client.submit_order(Order.market("GLD", 50, OrderSide.BUY))
    assert order.status is OrderStatus.FILLED
    assert order.avg_price == 100.0
    assert order.order_id.startswith("PO-")
    fills = client.get_fills(order.order_id)
    assert len(fills) == 1 and fills[0].qty == 50


def test_market_order_without_quote_is_rejected():
    client = PaperExecutionClient()
    order = client.submit_order(Order.market("GLD", 10, OrderSide.BUY))
    assert order.status is OrderStatus.REJECTED
    assert "no quote" in order.reason
    assert client.get_fills() == []


def test_market_order_applies_slippage_against_us():
    client = PaperExecutionClient(slippage_pct=0.01)
    client.on_quote(q(100.0))
    buy = client.submit_order(Order.market("GLD", 10, OrderSide.BUY))
    sell = client.submit_order(Order.market("GLD", 10, OrderSide.SELL))
    assert buy.avg_price == pytest.approx(101.0)
    assert sell.avg_price == pytest.approx(99.0)


def test_limit_buy_rests_until_ask_crosses():
    client = PaperExecutionClient()
    client.on_quote(q(100.0))
    order = client.submit_order(Order.limit("GLD", 10, OrderSide.BUY, 99.0))
    assert order.status is OrderStatus.NEW
    assert client.open_orders == (order,)

    client.on_quote(q(99.5))  # ask 99.55 -> still above the limit
    assert client.get_order_status(order.order_id) is OrderStatus.NEW

    fills = client.on_quote(q(98.0))  # ask 98.05 <= 99.0 -> crosses
    assert order.status is OrderStatus.FILLED
    assert order.avg_price == pytest.approx(98.05)
    assert len(fills) == 1


def test_limit_sell_rests_until_bid_crosses():
    client = PaperExecutionClient()
    client.on_quote(q(100.0))
    order = client.submit_order(Order.limit("GLD", 10, OrderSide.SELL, 105.0))
    assert order.status is OrderStatus.NEW
    client.on_quote(q(106.0))  # bid 105.95 >= 105
    assert order.status is OrderStatus.FILLED
    assert order.avg_price == pytest.approx(105.95)


def test_partial_fills_accumulate_across_ticks():
    client = PaperExecutionClient(max_fill_qty=30)
    client.on_quote(q(100.0))
    order = client.submit_order(Order.market("GLD", 100, OrderSide.BUY))
    assert order.status is OrderStatus.PARTIALLY_FILLED and order.filled_qty == 30
    client.on_quote(q(102.0))
    client.on_quote(q(104.0))
    client.on_quote(q(106.0))
    assert order.status is OrderStatus.FILLED
    assert order.filled_qty == 100
    assert order.avg_price == pytest.approx((30 * 100 + 30 * 102 + 30 * 104 + 10 * 106) / 100)
    assert len(client.get_fills(order.order_id)) == 4


def test_cancel_stops_further_fills():
    client = PaperExecutionClient()
    client.on_quote(q(100.0))
    order = client.submit_order(Order.limit("GLD", 10, OrderSide.BUY, 90.0))
    client.cancel_order(order.order_id)
    assert order.status is OrderStatus.CANCELLED
    client.on_quote(q(80.0))
    assert order.filled_qty == 0
    assert client.get_fills(order.order_id) == []


def test_unknown_order_lookups_raise():
    client = PaperExecutionClient()
    with pytest.raises(UnknownOrder):
        client.get_order_status("nope")
    with pytest.raises(UnknownOrder):
        client.cancel_order("nope")


def test_duplicate_order_id_rejected():
    client = PaperExecutionClient()
    client.on_quote(q(100.0))
    order = client.submit_order(Order.market("GLD", 1, OrderSide.BUY))
    dup = Order.market("GLD", 1, OrderSide.BUY)
    dup.order_id = order.order_id
    with pytest.raises(ExecutionError):
        client.submit_order(dup)


def test_other_symbols_are_not_matched():
    client = PaperExecutionClient()
    client.on_quote(q(100.0))
    order = client.submit_order(Order.limit("SLV", 5, OrderSide.BUY, 200.0))
    assert order.status is OrderStatus.NEW
    client.on_quote(q(50.0))  # GLD tick must not fill an SLV order
    assert order.filled_qty == 0


# -- FillTracker -----------------------------------------------------------


def test_fill_tracker_avg_price_and_net_qty():
    t = FillTracker()
    t.add(Fill("1", "GLD", OrderSide.BUY, 100, 100.0))
    t.add(Fill("2", "GLD", OrderSide.BUY, 100, 102.0))
    t.add(Fill("3", "GLD", OrderSide.SELL, 50, 110.0))
    assert len(t) == 3
    assert t.avg_price("GLD", OrderSide.BUY) == pytest.approx(101.0)
    assert t.filled_qty("GLD") == 250
    assert t.net_qty("GLD") == 150
    assert t.notional("GLD", OrderSide.SELL) == pytest.approx(5500.0)


def test_fill_tracker_realized_slippage_sign_convention():
    t = FillTracker()
    t.add(Fill("1", "GLD", OrderSide.BUY, 10, 101.0))
    # paid 1.00 above expectation -> positive (worse)
    assert t.realized_slippage("GLD", 100.0, OrderSide.BUY) == pytest.approx(1.0)

    s = FillTracker()
    s.add(Fill("2", "GLD", OrderSide.SELL, 10, 99.0))
    # sold 1.00 below expectation -> also positive (worse)
    assert s.realized_slippage("GLD", 100.0, OrderSide.SELL) == pytest.approx(1.0)


def test_fill_tracker_empty_is_safe():
    t = FillTracker()
    assert t.avg_price("GLD") == 0.0
    assert t.realized_slippage("GLD", 100.0) == 0.0
    assert t.summary("GLD")["fills"] == 0


def test_fill_tracker_extend_and_summary():
    client = PaperExecutionClient()
    client.on_quote(q(100.0))
    client.submit_order(Order.market("GLD", 10, OrderSide.BUY))
    t = FillTracker()
    t.extend(client.get_fills())
    summary = t.summary("GLD")
    assert summary["bought"] == 10 and summary["net_qty"] == 10
    assert summary["avg_price"] == pytest.approx(100.0)
