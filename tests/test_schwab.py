"""Schwab integration tests — fully offline (transport is injected)."""

from __future__ import annotations

import json
import time
from datetime import date

import pytest

from wheel.config import LiveTradingDisabled
from wheel.schwab import (
    ACCESS_TTL_SECONDS,
    SchwabAuth,
    SchwabAuthError,
    SchwabClient,
    SchwabCredentials,
    SchwabError,
    SchwabMarketData,
    TokenSet,
    TokenStore,
    build_authorize_url,
    extract_code,
    parse_chain,
    parse_positions,
    parse_quote,
)

CREDS = SchwabCredentials(
    app_key="APPKEY123",
    app_secret="SECRET456",
    callback_url="https://127.0.0.1:8182",
    token_path="unused.json",
)


class FakeTransport:
    """Records calls, returns queued (status, payload) responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if not self.responses:
            raise AssertionError(f"unexpected extra request: {method} {url}")
        status, payload = self.responses.pop(0)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return status, raw


def _store(tmp_path):
    return TokenStore(str(tmp_path / "tokens.json"))


# --------------------------------------------------------------------- creds


def test_credentials_from_env_requires_both_keys():
    with pytest.raises(SchwabAuthError):
        SchwabCredentials.from_env({"SCHWAB_APP_KEY": "only-key"})


def test_credentials_from_env_defaults():
    creds = SchwabCredentials.from_env(
        {"SCHWAB_APP_KEY": "k", "SCHWAB_APP_SECRET": "s"}
    )
    assert creds.callback_url == "https://127.0.0.1:8182"
    assert creds.token_path.endswith("schwab_tokens.json")


def test_basic_auth_is_base64_of_key_colon_secret():
    import base64

    assert base64.b64decode(CREDS.basic_auth).decode() == "APPKEY123:SECRET456"


# ---------------------------------------------------------------------- oauth


def test_build_authorize_url_has_required_params():
    url = build_authorize_url(CREDS)
    assert url.startswith("https://api.schwabapi.com/v1/oauth/authorize?")
    assert "client_id=APPKEY123" in url
    assert "response_type=code" in url
    assert "redirect_uri=https%3A%2F%2F127.0.0.1%3A8182" in url


def test_extract_code_decodes_value():
    url = "https://127.0.0.1:8182/?code=abc123%40&session=xyz"
    assert extract_code(url) == "abc123@"


def test_extract_code_rejects_url_without_code():
    with pytest.raises(SchwabAuthError):
        extract_code("https://127.0.0.1:8182/?error=access_denied")


def test_exchange_code_posts_form_and_persists(tmp_path):
    transport = FakeTransport(
        [(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": 1800})]
    )
    store = _store(tmp_path)
    auth = SchwabAuth(CREDS, store=store, transport=transport)

    tokens = auth.exchange_code("thecode@")

    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/oauth/token")
    assert call["headers"]["Authorization"].startswith("Basic ")
    assert b"grant_type=authorization_code" in call["body"]
    assert b"code=thecode%40" in call["body"]
    assert tokens.access_token == "AT"
    assert store.load().refresh_token == "RT"


def test_token_error_response_raises():
    transport = FakeTransport([(400, {"error": "invalid_grant", "error_description": "bad code"})])
    auth = SchwabAuth(CREDS, store=TokenStore("/dev/null/x"), transport=transport)
    with pytest.raises(SchwabAuthError, match="bad code"):
        auth.exchange_code("nope")


def test_access_token_refreshes_when_expired(tmp_path):
    store = _store(tmp_path)
    store.save(
        TokenSet(
            access_token="OLD",
            refresh_token="RT",
            expires_at=time.time() - 10,  # already dead
            refresh_expires_at=time.time() + 3600,
        )
    )
    transport = FakeTransport([(200, {"access_token": "NEW", "refresh_token": "RT2", "expires_in": 1800})])
    auth = SchwabAuth(CREDS, store=store, transport=transport)

    assert auth.access_token() == "NEW"
    assert b"grant_type=refresh_token" in transport.calls[0]["body"]
    assert store.load().access_token == "NEW"


def test_access_token_reuses_valid_token(tmp_path):
    store = _store(tmp_path)
    store.save(
        TokenSet(
            access_token="GOOD",
            refresh_token="RT",
            expires_at=time.time() + ACCESS_TTL_SECONDS,
            refresh_expires_at=time.time() + 3600,
        )
    )
    auth = SchwabAuth(CREDS, store=store, transport=FakeTransport([]))
    assert auth.access_token() == "GOOD"  # no transport call queued -> no network


def test_expired_refresh_token_demands_relogin(tmp_path):
    store = _store(tmp_path)
    store.save(
        TokenSet(
            access_token="OLD",
            refresh_token="RT",
            expires_at=time.time() - 10,
            refresh_expires_at=time.time() - 1,  # 7-day window blown
        )
    )
    auth = SchwabAuth(CREDS, store=store, transport=FakeTransport([]))
    with pytest.raises(SchwabAuthError, match="login"):
        auth.access_token()


def test_missing_tokens_demands_login(tmp_path):
    auth = SchwabAuth(CREDS, store=_store(tmp_path), transport=FakeTransport([]))
    with pytest.raises(SchwabAuthError, match="login"):
        auth.access_token()


def test_token_store_roundtrip_and_permissions(tmp_path):
    import os
    import stat

    store = _store(tmp_path)
    store.save(TokenSet("A", "R", time.time() + 60, time.time() + 600, scope="readonly"))
    loaded = store.load()
    assert (loaded.access_token, loaded.refresh_token, loaded.scope) == ("A", "R", "readonly")
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600
    store.clear()
    assert store.load() is None


# --------------------------------------------------------------------- parsers

QUOTE_PAYLOAD = {
    "GLD": {
        "assetMainType": "EQUITY",
        "quote": {
            "lastPrice": 245.31,
            "bidPrice": 245.28,
            "askPrice": 245.34,
            "volatility": 12.5,
        },
        "fundamental": {"divYield": 1.8},
    }
}


def test_parse_quote_scales_percentages():
    q = parse_quote("gld", QUOTE_PAYLOAD, as_of=date(2026, 3, 2))
    assert q.symbol == "GLD"
    assert q.price == 245.31
    assert q.iv == 0.125
    assert q.div_yield == 0.018
    assert q.as_of == date(2026, 3, 2)


def test_parse_quote_falls_back_to_mid_and_default_iv():
    payload = {"XYZ": {"quote": {"bidPrice": 10.0, "askPrice": 11.0}}}
    q = parse_quote("XYZ", payload, default_iv=0.42)
    assert q.price == 10.5
    assert q.iv == 0.42


def test_parse_quote_without_price_raises():
    with pytest.raises(SchwabError):
        parse_quote("XYZ", {"XYZ": {"quote": {}}})


CHAIN_PAYLOAD = {
    "symbol": "GLD",
    "status": "SUCCESS",
    "underlyingPrice": 245.31,
    "callExpDateMap": {
        "2026-04-17:45": {
            "250.0": [
                {
                    "putCall": "CALL",
                    "strikePrice": 250.0,
                    "bid": 3.10,
                    "ask": 3.30,
                    "volatility": 14.2,
                    "openInterest": 1200,
                    "totalVolume": 350,
                }
            ],
            "255.0": [
                {
                    "putCall": "CALL",
                    "strikePrice": 255.0,
                    "bid": 0.0,
                    "ask": 0.0,  # no market -> dropped
                    "volatility": 15.0,
                    "openInterest": 3,
                    "totalVolume": 0,
                }
            ],
        }
    },
    "putExpDateMap": {
        "2026-04-17:45": {
            "240.0": [
                {
                    "putCall": "PUT",
                    "strikePrice": 240.0,
                    "bid": 2.80,
                    "ask": 3.00,
                    "volatility": 15.6,
                    "openInterest": 800,
                    "totalVolume": 120,
                }
            ]
        }
    },
}


def test_parse_chain_builds_contracts():
    contracts = parse_chain(CHAIN_PAYLOAD, as_of=date(2026, 3, 3))
    assert len(contracts) == 2  # the no-market call is dropped
    call = [c for c in contracts if c.right == "C"][0]
    put = [c for c in contracts if c.right == "P"][0]
    assert call.underlying == "GLD"
    assert call.expiry == date(2026, 4, 17)
    assert call.strike == 250.0
    assert call.iv == 0.142
    assert call.open_interest == 1200
    assert call.dte == 45
    assert put.strike == 240.0
    assert put.mid == 2.90


def test_parse_chain_contracts_are_engine_compatible():
    """Greeks must compute off a parsed contract (the engine's real use)."""

    call = [c for c in parse_chain(CHAIN_PAYLOAD, as_of=date(2026, 3, 3)) if c.is_call()][0]
    greeks = call.greeks(spot=245.31)
    assert 0.0 < greeks.delta < 1.0
    assert greeks.theta < 0.0


def test_parse_chain_rejects_failed_payload():
    with pytest.raises(SchwabError):
        parse_chain({"status": "FAILED", "symbol": "GLD"})


def test_parse_chain_requires_symbol():
    with pytest.raises(SchwabError):
        parse_chain({"callExpDateMap": {}})


POSITIONS_PAYLOAD = [
    {
        "securitiesAccount": {
            "accountNumber": "12345678",
            "positions": [
                {
                    "instrument": {"symbol": "GLD", "assetType": "EQUITY"},
                    "longQuantity": 800.0,
                    "shortQuantity": 0.0,
                    "averagePrice": 201.44,
                    "marketValue": 196248.0,
                    "currentDayProfitLoss": 312.0,
                },
                {
                    "instrument": {
                        "symbol": "GLD   260417C00250000",
                        "assetType": "OPTION",
                        "underlyingSymbol": "GLD",
                    },
                    "longQuantity": 0.0,
                    "shortQuantity": 8.0,
                    "averagePrice": 3.2,
                    "marketValue": -2560.0,
                    "currentDayProfitLoss": 88.0,
                },
            ],
        }
    }
]


def test_parse_positions_signs_short_quantities():
    rows = parse_positions(POSITIONS_PAYLOAD)
    assert len(rows) == 2
    assert rows[0]["quantity"] == 800
    assert rows[1]["quantity"] == -8
    assert rows[1]["underlying"] == "GLD"
    assert rows[0]["account"] == "12345678"


def test_parse_positions_handles_empty_account():
    assert parse_positions([{"securitiesAccount": {"accountNumber": "1"}}]) == []


# ---------------------------------------------------------------------- client


def _authed(tmp_path, responses):
    store = _store(tmp_path)
    store.save(
        TokenSet("AT", "RT", time.time() + ACCESS_TTL_SECONDS, time.time() + 3600)
    )
    transport = FakeTransport(responses)
    auth = SchwabAuth(CREDS, store=store, transport=transport)
    return SchwabClient(auth, transport=transport), transport


def test_client_sends_bearer_token(tmp_path):
    client, transport = _authed(tmp_path, [(200, QUOTE_PAYLOAD)])
    client.quote_raw("GLD")
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer AT"
    assert "symbols=GLD" in transport.calls[0]["url"]


def test_client_chain_url_params(tmp_path):
    client, transport = _authed(tmp_path, [(200, CHAIN_PAYLOAD)])
    client.chain_raw("gld", strike_count=12, from_date=date(2026, 4, 1))
    url = transport.calls[0]["url"]
    assert "symbol=GLD" in url and "strikeCount=12" in url and "fromDate=2026-04-01" in url


def test_client_401_is_auth_error(tmp_path):
    client, _ = _authed(tmp_path, [(401, {"error": "unauthorized"})])
    with pytest.raises(SchwabAuthError):
        client.quote_raw("GLD")


def test_client_500_is_schwab_error(tmp_path):
    client, _ = _authed(tmp_path, [(500, b"boom")])
    with pytest.raises(SchwabError):
        client.quote_raw("GLD")


def test_client_positions_end_to_end(tmp_path):
    client, _ = _authed(tmp_path, [(200, POSITIONS_PAYLOAD)])
    rows = client.positions()
    assert [r["symbol"] for r in rows] == ["GLD", "GLD   260417C00250000"]


def test_place_order_is_hard_disabled(tmp_path):
    client, _ = _authed(tmp_path, [])
    with pytest.raises(LiveTradingDisabled):
        client.place_order(symbol="GLD", quantity=1)


# ------------------------------------------------------------------ marketdata


def test_marketdata_adapter_satisfies_protocol(tmp_path):
    client, transport = _authed(tmp_path, [(200, QUOTE_PAYLOAD), (200, CHAIN_PAYLOAD)])
    market = SchwabMarketData(client)

    quote = market.get_quote("GLD", as_of=date(2026, 3, 3))
    chain = market.get_chain("GLD", as_of=date(2026, 3, 3))

    assert quote.price == 245.31
    assert len(chain) == 2
    assert all(c.underlying == "GLD" for c in chain)


def test_marketdata_caches_within_window(tmp_path):
    client, transport = _authed(tmp_path, [(200, QUOTE_PAYLOAD)])
    market = SchwabMarketData(client, cache_seconds=60)
    market.get_quote("GLD")
    market.get_quote("GLD")  # served from cache; a second call would blow up FakeTransport
    assert len(transport.calls) == 1


def test_marketdata_is_usable_by_engine_protocol():
    from wheel.marketdata import MarketData

    assert hasattr(SchwabMarketData, "get_quote")
    assert hasattr(SchwabMarketData, "get_chain")
    assert isinstance(MarketData, type(MarketData))  # protocol import smoke check
