import pytest
from wheel.feeds import FeedManager, MarketFeed, PaperFeed, Quote, UnknownSymbol


def ticks(symbol: str = "GLD", prices=(100.0, 101.0, 102.0)) -> list[Quote]:
    return [
        Quote.from_last(symbol, p, spread_pct=0.01, volume=10, timestamp=float(i))
        for i, p in enumerate(prices)
    ]


# -- Quote ---------------------------------------------------------------


def test_quote_derived_fields():
    q = Quote(symbol="GLD", bid=99.5, ask=100.5, last=100.0, volume=25, timestamp=1.0)
    assert q.mid == 100.0
    assert q.spread == 1.0
    assert q.spread_pct == pytest.approx(0.01)
    assert q.to_dict()["symbol"] == "GLD"


def test_quote_rejects_crossed_and_bad_values():
    with pytest.raises(ValueError):
        Quote(symbol="GLD", bid=101.0, ask=100.0, last=100.0)
    with pytest.raises(ValueError):
        Quote(symbol="GLD", bid=99.0, ask=101.0, last=0.0)
    with pytest.raises(ValueError):
        Quote(symbol="GLD", bid=99.0, ask=101.0, last=100.0, volume=-1)


def test_quote_from_last_builds_symmetric_spread():
    q = Quote.from_last("gld", 100.0, spread_pct=0.02)
    assert q.symbol == "GLD"
    assert q.bid < q.last < q.ask
    assert q.mid == pytest.approx(100.0, abs=1e-6)


# -- PaperFeed ------------------------------------------------------------


def test_paper_feed_replays_sequence_in_order():
    feed = PaperFeed({"GLD": ticks()})
    feed.subscribe("gld")
    seen = []
    while not feed.exhausted:
        seen.extend(feed.poll())
    assert [q.last for q in seen] == [100.0, 101.0, 102.0]
    assert feed.poll() == []


def test_paper_feed_get_quote_tracks_last_tick():
    feed = PaperFeed({"GLD": ticks()})
    feed.subscribe("GLD")
    assert feed.get_quote("GLD") is None
    feed.poll()
    assert feed.get_quote("GLD").last == 100.0
    feed.poll()
    assert feed.get_quote("GLD").last == 101.0


def test_paper_feed_unknown_symbol_rejected():
    feed = PaperFeed({"GLD": ticks()})
    with pytest.raises(UnknownSymbol):
        feed.subscribe("SPY")


def test_paper_feed_unsubscribe_stops_delivery():
    feed = PaperFeed({"GLD": ticks()})
    feed.subscribe("GLD")
    assert feed.poll()
    feed.unsubscribe("GLD")
    assert feed.poll() == []
    assert feed.subscriptions == frozenset()


def test_paper_feed_loop_wraps_and_never_exhausts():
    feed = PaperFeed({"GLD": ticks()}, loop=True)
    feed.subscribe("GLD")
    lasts = [feed.poll()[0].last for _ in range(5)]
    assert lasts == [100.0, 101.0, 102.0, 100.0, 101.0]
    assert feed.exhausted is False


def test_paper_feed_from_prices_and_reset():
    feed = PaperFeed.from_prices("GLD", [10.0, 11.0], interval=0.5)
    feed.subscribe("GLD")
    assert feed.remaining("GLD") == 2
    first = feed.poll()[0]
    assert first.timestamp == 0.0
    assert feed.poll()[0].timestamp == 0.5
    feed.reset()
    assert feed.remaining("GLD") == 2


def test_paper_feed_multiplexes_two_symbols():
    feed = PaperFeed({"GLD": ticks("GLD"), "SLV": ticks("SLV", (20.0, 21.0, 22.0))})
    feed.subscribe("GLD")
    feed.subscribe("SLV")
    batch = feed.poll()
    assert sorted(q.symbol for q in batch) == ["GLD", "SLV"]


def test_market_feed_is_abstract():
    with pytest.raises(TypeError):
        MarketFeed()  # type: ignore[abstract]


# -- FeedManager -----------------------------------------------------------


def test_feed_manager_buffers_and_drains():
    mgr = FeedManager(PaperFeed({"GLD": ticks()}))
    mgr.subscribe("GLD")
    assert mgr.poll() == 1
    assert mgr.buffered == 1
    drained = mgr.drain()
    assert [q.last for q in drained] == [100.0]
    assert mgr.buffered == 0
    assert mgr.latest("GLD").last == 100.0


def test_feed_manager_merges_multiple_feeds():
    mgr = FeedManager(
        PaperFeed({"GLD": ticks("GLD")}),
        PaperFeed({"SLV": ticks("SLV", (20.0, 21.0, 22.0))}),
    )
    mgr.subscribe("GLD")
    mgr.subscribe("SLV")
    mgr.poll()
    assert sorted(q.symbol for q in mgr.drain()) == ["GLD", "SLV"]
    assert mgr.subscriptions == frozenset({"GLD", "SLV"})


def test_feed_manager_unknown_symbol_raises():
    mgr = FeedManager(PaperFeed({"GLD": ticks()}))
    with pytest.raises(UnknownSymbol):
        mgr.subscribe("TSLA")


def test_feed_manager_backpressure_drops_oldest():
    mgr = FeedManager(PaperFeed({"GLD": ticks()}), max_buffer=2)
    mgr.subscribe("GLD")
    for _ in range(3):
        mgr.poll()
    assert mgr.buffered == 2
    assert mgr.dropped == 1
    # oldest evicted -> the two most recent survive
    assert [q.last for q in mgr.drain()] == [101.0, 102.0]


def test_feed_manager_backpressure_drops_newest():
    mgr = FeedManager(PaperFeed({"GLD": ticks()}), max_buffer=2, drop_policy="newest")
    mgr.subscribe("GLD")
    for _ in range(3):
        mgr.poll()
    assert mgr.dropped == 1
    assert [q.last for q in mgr.drain()] == [100.0, 101.0]


def test_feed_manager_rejects_bad_config():
    with pytest.raises(ValueError):
        FeedManager(max_buffer=0)
    with pytest.raises(ValueError):
        FeedManager(drop_policy="explode")


def test_feed_manager_exhausted_and_close():
    feed = PaperFeed({"GLD": ticks()})
    mgr = FeedManager(feed)
    mgr.subscribe("GLD")
    while mgr.poll():
        mgr.drain()
    assert mgr.exhausted is True
    mgr.close()
    assert feed.subscriptions == frozenset()
