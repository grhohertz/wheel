"""Market-data feeds.

Everything here is *paper*: :class:`PaperFeed` replays a cached tick sequence,
which is all the package needs for backtests, the live-loop demo, and tests.
A real venue adapter would subclass :class:`MarketFeed` and implement three
methods -- nothing downstream of :class:`FeedManager` would change.

Note the deliberate name collision: :class:`wheel.feeds.Quote` is a *tick*
(bid/ask/last/volume at an instant), whereas :class:`wheel.models.Quote` is an
end-of-day underlying snapshot used by the option pricer. They are re-exported
from the package as ``TickQuote`` and ``Quote`` respectively.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "FeedError",
    "FeedManager",
    "MarketFeed",
    "PaperFeed",
    "Quote",
    "UnknownSymbol",
]


class FeedError(RuntimeError):
    """Base class for feed problems."""


class UnknownSymbol(FeedError, KeyError):
    """Raised when subscribing to a symbol the feed cannot serve."""


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class Quote:
    """One market tick."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: int = 0
    timestamp: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.bid < 0.0 or self.ask < 0.0:
            raise ValueError("bid/ask must be >= 0")
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: ask {self.ask} < bid {self.bid}")
        if self.last <= 0.0:
            raise ValueError("last must be positive")
        if self.volume < 0:
            raise ValueError("volume must be >= 0")

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2.0, 6)

    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 6)

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        return float("inf") if mid <= 0 else self.spread / mid

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "volume": self.volume,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_last(
        cls,
        symbol: str,
        last: float,
        *,
        spread_pct: float = 0.001,
        volume: int = 0,
        timestamp: float | None = None,
    ) -> "Quote":
        """Build a tick around ``last`` with a symmetric synthetic spread."""

        half = max(0.005, abs(last) * spread_pct / 2.0)
        return cls(
            symbol=symbol.upper(),
            bid=round(max(0.01, last - half), 4),
            ask=round(last + half, 4),
            last=round(last, 4),
            volume=volume,
            timestamp=_now() if timestamp is None else timestamp,
        )


class MarketFeed(ABC):
    """Minimal subscribe/poll surface every feed implements."""

    name: str = "feed"

    def __init__(self) -> None:
        self._subscribed: set[str] = set()

    @property
    def subscriptions(self) -> frozenset[str]:
        return frozenset(self._subscribed)

    def subscribe(self, symbol: str) -> None:
        sym = symbol.upper()
        self._on_subscribe(sym)
        self._subscribed.add(sym)

    def unsubscribe(self, symbol: str) -> None:
        self._subscribed.discard(symbol.upper())

    def _on_subscribe(self, symbol: str) -> None:
        """Hook for subclasses to validate a symbol. Default: accept anything."""

    def close(self) -> None:
        self._subscribed.clear()

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote | None:
        """Most recent tick delivered for ``symbol`` (``None`` before the first)."""

    @abstractmethod
    def poll(self) -> list[Quote]:
        """Return the next batch of ticks (possibly empty)."""


class PaperFeed(MarketFeed):
    """Replays a cached tick sequence, one tick per symbol per :meth:`poll`."""

    name = "paper"

    def __init__(self, ticks: Mapping[str, Sequence[Quote]], *, loop: bool = False) -> None:
        super().__init__()
        self._ticks: dict[str, list[Quote]] = {
            sym.upper(): list(seq) for sym, seq in ticks.items()
        }
        self._cursor: dict[str, int] = {sym: 0 for sym in self._ticks}
        self._last: dict[str, Quote] = {}
        self.loop = loop

    @classmethod
    def from_prices(
        cls,
        symbol: str,
        prices: Iterable[float],
        *,
        spread_pct: float = 0.001,
        volume: int = 100,
        start_ts: float = 0.0,
        interval: float = 1.0,
        loop: bool = False,
    ) -> "PaperFeed":
        """Convenience builder: a price path becomes a tick sequence."""

        quotes = [
            Quote.from_last(
                symbol,
                price,
                spread_pct=spread_pct,
                volume=volume,
                timestamp=start_ts + i * interval,
            )
            for i, price in enumerate(prices)
        ]
        return cls({symbol.upper(): quotes}, loop=loop)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._ticks))

    def _on_subscribe(self, symbol: str) -> None:
        if symbol not in self._ticks:
            raise UnknownSymbol(f"{self.name} feed has no ticks for {symbol!r}")

    @property
    def exhausted(self) -> bool:
        """True once every subscribed symbol has replayed its whole sequence."""

        if self.loop or not self._subscribed:
            return not self._subscribed
        return all(self._cursor[s] >= len(self._ticks[s]) for s in self._subscribed)

    def remaining(self, symbol: str) -> int:
        sym = symbol.upper()
        return max(0, len(self._ticks.get(sym, ())) - self._cursor.get(sym, 0))

    def get_quote(self, symbol: str) -> Quote | None:
        return self._last.get(symbol.upper())

    def poll(self) -> list[Quote]:
        out: list[Quote] = []
        for sym in sorted(self._subscribed):
            seq = self._ticks[sym]
            if not seq:
                continue
            idx = self._cursor[sym]
            if idx >= len(seq):
                if not self.loop:
                    continue
                idx = 0
            quote = seq[idx]
            self._cursor[sym] = idx + 1
            self._last[sym] = quote
            out.append(quote)
        return out

    def reset(self) -> None:
        self._cursor = {sym: 0 for sym in self._ticks}
        self._last.clear()


class FeedManager:
    """Multiplexes feeds into one bounded buffer.

    Backpressure is explicit: the buffer never exceeds ``max_buffer``. With
    ``drop_policy='oldest'`` the head is evicted to make room; with ``'newest'``
    the incoming tick is discarded. Either way ``dropped`` counts the loss so a
    caller can alarm on it instead of silently falling behind.
    """

    def __init__(
        self,
        *feeds: MarketFeed,
        max_buffer: int = 1000,
        drop_policy: str = "oldest",
    ) -> None:
        if max_buffer < 1:
            raise ValueError("max_buffer must be >= 1")
        if drop_policy not in ("oldest", "newest"):
            raise ValueError("drop_policy must be 'oldest' or 'newest'")
        self._feeds: list[MarketFeed] = list(feeds)
        self.max_buffer = max_buffer
        self.drop_policy = drop_policy
        self._buffer: deque[Quote] = deque()
        self._latest: dict[str, Quote] = {}
        self.dropped = 0

    # -- wiring ---------------------------------------------------------
    def add_feed(self, feed: MarketFeed) -> MarketFeed:
        self._feeds.append(feed)
        return feed

    @property
    def feeds(self) -> tuple[MarketFeed, ...]:
        return tuple(self._feeds)

    @property
    def subscriptions(self) -> frozenset[str]:
        out: set[str] = set()
        for feed in self._feeds:
            out |= set(feed.subscriptions)
        return frozenset(out)

    def subscribe(self, symbol: str) -> list[MarketFeed]:
        """Subscribe on every feed that can serve ``symbol``."""

        accepted: list[MarketFeed] = []
        for feed in self._feeds:
            try:
                feed.subscribe(symbol)
            except UnknownSymbol:
                continue
            accepted.append(feed)
        if not accepted:
            raise UnknownSymbol(f"no attached feed serves {symbol.upper()!r}")
        return accepted

    def unsubscribe(self, symbol: str) -> None:
        for feed in self._feeds:
            feed.unsubscribe(symbol)

    # -- pumping --------------------------------------------------------
    def _offer(self, quote: Quote) -> None:
        if len(self._buffer) >= self.max_buffer:
            self.dropped += 1
            if self.drop_policy == "newest":
                return
            self._buffer.popleft()
        self._buffer.append(quote)
        self._latest[quote.symbol.upper()] = quote

    def poll(self) -> int:
        """Pump every feed once. Returns how many ticks were buffered."""

        before = len(self._buffer) + self.dropped
        for feed in self._feeds:
            for quote in feed.poll():
                self._offer(quote)
        return len(self._buffer) + self.dropped - before

    def drain(self) -> list[Quote]:
        """Remove and return everything buffered, oldest first."""

        out = list(self._buffer)
        self._buffer.clear()
        return out

    def latest(self, symbol: str) -> Quote | None:
        return self._latest.get(symbol.upper())

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def exhausted(self) -> bool:
        """True when no buffered ticks remain and no feed can produce more."""

        if self._buffer:
            return False
        return all(getattr(feed, "exhausted", False) for feed in self._feeds)

    def close(self) -> None:
        for feed in self._feeds:
            feed.close()
        self._buffer.clear()
