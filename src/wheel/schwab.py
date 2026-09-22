"""Schwab Trader API integration — OAuth + read-only market data.

Zero dependencies (urllib/json/ssl only). Two layers:

* :class:`SchwabAuth`  — three-legged OAuth 2.0 against ``api.schwabapi.com``,
  with an on-disk token store and automatic access-token refresh.
* :class:`SchwabMarketData` — implements the :class:`wheel.marketdata.MarketData`
  protocol (``get_quote`` / ``get_chain``) so it drops straight into
  ``WheelEngine(market=...)``.

SAFETY: this module never places, cancels, or modifies an order. The only
trader-API calls are account *reads*. :meth:`SchwabClient.place_order` exists
solely to raise :class:`wheel.config.LiveTradingDisabled` if anything tries.

CLI::

    python -m wheel.schwab login      # print auth URL, paste the redirect back
    python -m wheel.schwab status     # token validity / expiry
    python -m wheel.schwab refresh    # force an access-token refresh
    python -m wheel.schwab accounts   # account numbers + balances (read-only)
    python -m wheel.schwab quote GLD
    python -m wheel.schwab chain GLD --dte 45
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional

from .config import LiveTradingDisabled
from .greeks import CALL, PUT
from .models import OptionContract, Quote

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

AUTH_BASE = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
MARKETDATA_BASE = "https://api.schwabapi.com/marketdata/v1"
TRADER_BASE = "https://api.schwabapi.com/trader/v1"

DEFAULT_CALLBACK = "https://127.0.0.1:8182"
DEFAULT_TOKEN_PATH = "state/schwab_tokens.json"

ACCESS_TTL_SECONDS = 1800  # Schwab: access token lives 30 minutes
REFRESH_TTL_SECONDS = 7 * 24 * 3600  # Schwab: refresh token lives 7 days
REFRESH_SKEW_SECONDS = 120  # refresh this long before actual expiry

# Transport signature: (method, url, headers, body) -> (status, bytes)
Transport = Callable[[str, str, dict, Optional[bytes]], "tuple[int, bytes]"]


class SchwabError(RuntimeError):
    """Any non-2xx response or malformed payload from Schwab."""


class SchwabAuthError(SchwabError):
    """Credentials missing, expired, or rejected."""


# --------------------------------------------------------------------------- #
# Credentials / token storage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SchwabCredentials:
    """App-level credentials from developer.schwab.com."""

    app_key: str
    app_secret: str
    callback_url: str = DEFAULT_CALLBACK
    token_path: str = DEFAULT_TOKEN_PATH

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SchwabCredentials":
        env = dict(os.environ if env is None else env)
        key = env.get("SCHWAB_APP_KEY", "").strip()
        secret = env.get("SCHWAB_APP_SECRET", "").strip()
        if not key or not secret:
            raise SchwabAuthError(
                "SCHWAB_APP_KEY / SCHWAB_APP_SECRET are not set. "
                "See docs/schwab-setup.md (step 2)."
            )
        return cls(
            app_key=key,
            app_secret=secret,
            callback_url=env.get("SCHWAB_CALLBACK_URL", DEFAULT_CALLBACK).strip(),
            token_path=env.get("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH).strip(),
        )

    @property
    def basic_auth(self) -> str:
        raw = f"{self.app_key}:{self.app_secret}".encode()
        return base64.b64encode(raw).decode()


@dataclass
class TokenSet:
    """Access + refresh token pair with absolute expiry stamps (epoch seconds)."""

    access_token: str
    refresh_token: str
    expires_at: float
    refresh_expires_at: float
    scope: str = ""
    obtained_at: float = field(default_factory=lambda: time.time())

    @classmethod
    def from_response(cls, payload: dict, now: float | None = None) -> "TokenSet":
        now = time.time() if now is None else now
        if "access_token" not in payload:
            raise SchwabAuthError(f"token response missing access_token: {payload!r}")
        expires_in = float(payload.get("expires_in", ACCESS_TTL_SECONDS))
        return cls(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token", ""),
            expires_at=now + expires_in,
            refresh_expires_at=now + REFRESH_TTL_SECONDS,
            scope=payload.get("scope", ""),
            obtained_at=now,
        )

    def access_valid(self, now: float | None = None, skew: float = REFRESH_SKEW_SECONDS) -> bool:
        now = time.time() if now is None else now
        return bool(self.access_token) and now < (self.expires_at - skew)

    def refresh_valid(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return bool(self.refresh_token) and now < self.refresh_expires_at

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "refresh_expires_at": self.refresh_expires_at,
            "scope": self.scope,
            "obtained_at": self.obtained_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenSet":
        return cls(
            access_token=d["access_token"],
            refresh_token=d.get("refresh_token", ""),
            expires_at=float(d.get("expires_at", 0.0)),
            refresh_expires_at=float(d.get("refresh_expires_at", 0.0)),
            scope=d.get("scope", ""),
            obtained_at=float(d.get("obtained_at", 0.0)),
        )


class TokenStore:
    """Tokens on disk, mode 0600. ``state/`` is gitignored."""

    def __init__(self, path: str = DEFAULT_TOKEN_PATH) -> None:
        self.path = path

    def load(self) -> Optional[TokenSet]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return TokenSet.from_dict(json.load(fh))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError) as exc:
            raise SchwabAuthError(f"token file {self.path} is corrupt: {exc}") from exc

    def save(self, tokens: TokenSet) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(tokens.to_dict(), fh, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def urllib_transport(timeout: float = 15.0) -> Transport:
    """Default transport built on :mod:`urllib.request`."""

    context = ssl.create_default_context()

    def _send(method: str, url: str, headers: dict, body: Optional[bytes]) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except urllib.error.URLError as exc:  # network down, DNS, TLS
            raise SchwabError(f"network error calling {url}: {exc.reason}") from exc

    return _send


# --------------------------------------------------------------------------- #
# OAuth
# --------------------------------------------------------------------------- #


def build_authorize_url(creds: SchwabCredentials, state: str | None = None) -> str:
    """The URL the user opens in a browser to grant access."""

    params = {
        "client_id": creds.app_key,
        "redirect_uri": creds.callback_url,
        "response_type": "code",
    }
    if state:
        params["state"] = state
    return f"{AUTH_BASE}?{urllib.parse.urlencode(params)}"


def extract_code(redirect_url: str) -> str:
    """Pull ``code`` out of the URL Schwab redirects the browser to.

    Schwab appends ``@`` to the code; the value must be sent back decoded and
    intact, so we take the raw query value after percent-decoding.
    """

    parsed = urllib.parse.urlparse(redirect_url.strip())
    query = urllib.parse.parse_qs(parsed.query)
    codes = query.get("code")
    if not codes or not codes[0]:
        raise SchwabAuthError(
            "no ?code= found in that URL. Paste the FULL address bar contents "
            "after approving access (it starts with your callback URL)."
        )
    return codes[0]


class SchwabAuth:
    """Owns the token lifecycle: exchange, refresh, persist."""

    def __init__(
        self,
        creds: SchwabCredentials,
        store: TokenStore | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.creds = creds
        self.store = store or TokenStore(creds.token_path)
        self.transport = transport or urllib_transport()
        self._tokens: Optional[TokenSet] = None

    # -- token endpoint ------------------------------------------------- #
    def _token_request(self, form: dict) -> TokenSet:
        body = urllib.parse.urlencode(form).encode()
        headers = {
            "Authorization": f"Basic {self.creds.basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        status, raw = self.transport("POST", TOKEN_URL, headers, body)
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError as exc:
            raise SchwabAuthError(f"token endpoint returned non-JSON ({status})") from exc
        if status >= 400:
            raise SchwabAuthError(
                f"token request failed ({status}): "
                f"{payload.get('error_description') or payload.get('error') or payload}"
            )
        tokens = TokenSet.from_response(payload)
        self.store.save(tokens)
        self._tokens = tokens
        return tokens

    def exchange_code(self, code: str) -> TokenSet:
        """Authorization-code grant — run once per 7-day refresh window."""

        return self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.creds.callback_url,
            }
        )

    def refresh(self, refresh_token: str | None = None) -> TokenSet:
        """Refresh-token grant — good for 30 more minutes of access."""

        token = refresh_token
        if token is None:
            current = self.tokens()
            if current is None or not current.refresh_valid():
                raise SchwabAuthError(
                    "refresh token missing or older than 7 days — run "
                    "`python -m wheel.schwab login` again."
                )
            token = current.refresh_token
        return self._token_request({"grant_type": "refresh_token", "refresh_token": token})

    # -- accessors ------------------------------------------------------ #
    def tokens(self) -> Optional[TokenSet]:
        if self._tokens is None:
            self._tokens = self.store.load()
        return self._tokens

    def access_token(self) -> str:
        """A currently-valid access token, refreshing transparently."""

        tokens = self.tokens()
        if tokens is None:
            raise SchwabAuthError(
                "no saved tokens — run `python -m wheel.schwab login` first."
            )
        if tokens.access_valid():
            return tokens.access_token
        return self.refresh().access_token


# --------------------------------------------------------------------------- #
# Response parsers (pure — unit-testable without network)
# --------------------------------------------------------------------------- #


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # drop NaN


def parse_quote(symbol: str, payload: dict, as_of: date | None = None,
                default_iv: float = 0.30) -> Quote:
    """``GET /marketdata/v1/quotes`` -> :class:`wheel.models.Quote`."""

    as_of = as_of or date.today()
    node = payload.get(symbol.upper()) or payload.get(symbol) or {}
    quote = node.get("quote", node)
    fundamental = node.get("fundamental", {})

    price = _f(quote.get("lastPrice")) or _f(quote.get("mark"))
    if price <= 0:
        bid, ask = _f(quote.get("bidPrice")), _f(quote.get("askPrice"))
        price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    if price <= 0:
        raise SchwabError(f"no usable price for {symbol} in quote response")

    iv = _f(quote.get("volatility")) / 100.0
    if iv <= 0:
        iv = _f(fundamental.get("vol10DayAvg")) / 100.0
    if iv <= 0:
        iv = default_iv

    div = _f(fundamental.get("divYield")) / 100.0
    return Quote(symbol=symbol.upper(), price=round(price, 4), iv=round(iv, 4),
                 div_yield=round(div, 4), as_of=as_of)


def _expiry_from_key(key: str) -> date:
    """``"2026-01-16:30"`` -> ``date(2026, 1, 16)``."""

    head = key.split(":", 1)[0]
    return datetime.strptime(head, "%Y-%m-%d").date()


def parse_chain(payload: dict, as_of: date | None = None) -> list[OptionContract]:
    """``GET /marketdata/v1/chains`` -> list of :class:`OptionContract`."""

    if payload.get("status") == "FAILED":
        raise SchwabError(f"chain request failed: {payload}")
    underlying = (payload.get("symbol") or "").upper()
    if not underlying:
        raise SchwabError("chain response has no underlying symbol")
    as_of = as_of or date.today()

    contracts: list[OptionContract] = []
    for map_key, right in (("callExpDateMap", CALL), ("putExpDateMap", PUT)):
        for exp_key, strikes in (payload.get(map_key) or {}).items():
            expiry = _expiry_from_key(exp_key)
            for strike_key, legs in (strikes or {}).items():
                for leg in legs or []:
                    bid, ask = _f(leg.get("bid")), _f(leg.get("ask"))
                    if bid <= 0 and ask <= 0:
                        continue  # no market
                    iv = _f(leg.get("volatility")) / 100.0
                    if iv <= 0:
                        continue  # unusable for Greeks
                    contracts.append(
                        OptionContract(
                            underlying=underlying,
                            expiry=expiry,
                            strike=_f(leg.get("strikePrice"), _f(strike_key)),
                            right=right,
                            bid=round(max(bid, 0.0), 4),
                            ask=round(max(ask, bid), 4),
                            iv=round(iv, 4),
                            open_interest=int(_f(leg.get("openInterest"))),
                            volume=int(_f(leg.get("totalVolume"))),
                            as_of=as_of,
                        )
                    )
    contracts.sort(key=lambda c: (c.expiry, c.right, c.strike))
    return contracts


def parse_positions(payload: Any) -> list[dict]:
    """Flatten ``GET /trader/v1/accounts?fields=positions`` into plain dicts."""

    accounts = payload if isinstance(payload, list) else [payload]
    rows: list[dict] = []
    for entry in accounts:
        acct = (entry or {}).get("securitiesAccount", entry) or {}
        number = acct.get("accountNumber", "")
        for pos in acct.get("positions", []) or []:
            instrument = pos.get("instrument", {}) or {}
            long_qty = _f(pos.get("longQuantity"))
            short_qty = _f(pos.get("shortQuantity"))
            rows.append(
                {
                    "account": number,
                    "symbol": instrument.get("symbol", ""),
                    "asset_type": instrument.get("assetType", ""),
                    "underlying": instrument.get("underlyingSymbol", ""),
                    "quantity": long_qty - short_qty,
                    "average_price": _f(pos.get("averagePrice")),
                    "market_value": _f(pos.get("marketValue")),
                    "day_pl": _f(pos.get("currentDayProfitLoss")),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class SchwabClient:
    """Authenticated, READ-ONLY Schwab API client."""

    def __init__(self, auth: SchwabAuth, transport: Transport | None = None) -> None:
        self.auth = auth
        self.transport = transport or auth.transport

    # -- plumbing ------------------------------------------------------- #
    def _get(self, url: str) -> Any:
        headers = {
            "Authorization": f"Bearer {self.auth.access_token()}",
            "Accept": "application/json",
        }
        status, raw = self.transport("GET", url, headers, None)
        if status == 401:
            raise SchwabAuthError("401 from Schwab — token rejected; re-run login.")
        if status >= 400:
            raise SchwabError(f"GET {url} failed ({status}): {raw[:400].decode(errors='replace')}")
        try:
            return json.loads(raw.decode() or "{}")
        except json.JSONDecodeError as exc:
            raise SchwabError(f"non-JSON response from {url}") from exc

    # -- market data ---------------------------------------------------- #
    def quote_raw(self, symbol: str) -> dict:
        q = urllib.parse.urlencode({"symbols": symbol.upper(), "indicative": "false"})
        return self._get(f"{MARKETDATA_BASE}/quotes?{q}")

    def chain_raw(
        self,
        symbol: str,
        contract_type: str = "ALL",
        strike_count: int = 40,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> dict:
        params: dict[str, str] = {
            "symbol": symbol.upper(),
            "contractType": contract_type,
            "strikeCount": str(strike_count),
            "includeUnderlyingQuote": "true",
        }
        if from_date:
            params["fromDate"] = from_date.isoformat()
        if to_date:
            params["toDate"] = to_date.isoformat()
        return self._get(f"{MARKETDATA_BASE}/chains?{urllib.parse.urlencode(params)}")

    # -- trader (read-only) --------------------------------------------- #
    def account_numbers(self) -> list[dict]:
        return self._get(f"{TRADER_BASE}/accounts/accountNumbers")

    def accounts(self, with_positions: bool = True) -> list[dict]:
        suffix = "?fields=positions" if with_positions else ""
        return self._get(f"{TRADER_BASE}/accounts{suffix}")

    def positions(self) -> list[dict]:
        return parse_positions(self.accounts(with_positions=True))

    # -- hard stop ------------------------------------------------------ #
    def place_order(self, *args: Any, **kwargs: Any) -> None:
        raise LiveTradingDisabled(
            "SchwabClient is read-only by design: quotes, chains, and account "
            "reads only. Fills stay in wheel.broker.PaperBroker."
        )


@dataclass
class SchwabMarketData:
    """Adapter implementing :class:`wheel.marketdata.MarketData` over Schwab.

    Drop-in replacement for ``SyntheticMarketData``::

        market = SchwabMarketData(SchwabClient(SchwabAuth(SchwabCredentials.from_env())))
        engine = WheelEngine(market=market, ...)
    """

    client: SchwabClient
    strike_count: int = 40
    default_iv: float = 0.30
    cache_seconds: float = 30.0
    _cache: dict = field(default_factory=dict, repr=False)

    def _cached(self, key: str, producer: Callable[[], Any]) -> Any:
        now = time.time()
        hit = self._cache.get(key)
        if hit and (now - hit[0]) < self.cache_seconds:
            return hit[1]
        value = producer()
        self._cache[key] = (now, value)
        return value

    def get_quote(self, symbol: str, as_of: date | None = None) -> Quote:
        payload = self._cached(f"q:{symbol.upper()}", lambda: self.client.quote_raw(symbol))
        return parse_quote(symbol, payload, as_of=as_of, default_iv=self.default_iv)

    def get_chain(self, symbol: str, as_of: date | None = None) -> list[OptionContract]:
        payload = self._cached(
            f"c:{symbol.upper()}",
            lambda: self.client.chain_raw(symbol, strike_count=self.strike_count),
        )
        return parse_chain(payload, as_of=as_of)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _client_from_env() -> SchwabClient:
    creds = SchwabCredentials.from_env()
    return SchwabClient(SchwabAuth(creds))


def _cmd_login(args: argparse.Namespace) -> int:
    creds = SchwabCredentials.from_env()
    auth = SchwabAuth(creds)
    url = build_authorize_url(creds)
    print("1. Open this URL in a browser and approve access:\n")
    print(f"   {url}\n")
    print("2. After approving you land on a page that will NOT load (that is")
    print(f"   expected — nothing listens on {creds.callback_url}).")
    print("3. Copy the ENTIRE address-bar URL and paste it below.\n")
    redirect = args.redirect_url or input("Redirect URL: ").strip()
    code = extract_code(redirect)
    tokens = auth.exchange_code(code)
    expires = datetime.fromtimestamp(tokens.expires_at, tz=timezone.utc)
    print(f"\nSaved tokens to {creds.token_path}")
    print(f"  access token expires  {expires:%Y-%m-%d %H:%M:%SZ}")
    print(f"  refresh token expires {datetime.fromtimestamp(tokens.refresh_expires_at, tz=timezone.utc):%Y-%m-%d %H:%M:%SZ}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    creds = SchwabCredentials.from_env()
    tokens = TokenStore(creds.token_path).load()
    if tokens is None:
        print(f"no tokens at {creds.token_path} — run `python -m wheel.schwab login`")
        return 1
    now = time.time()
    print(f"token file      {creds.token_path}")
    print(f"access valid    {tokens.access_valid()}  ({(tokens.expires_at - now)/60:.1f} min left)")
    print(f"refresh valid   {tokens.refresh_valid()}  ({(tokens.refresh_expires_at - now)/3600:.1f} h left)")
    print(f"scope           {tokens.scope or '-'}")
    return 0 if tokens.refresh_valid() else 1


def _cmd_refresh(args: argparse.Namespace) -> int:
    auth = SchwabAuth(SchwabCredentials.from_env())
    tokens = auth.refresh()
    print(f"refreshed — access token good for {(tokens.expires_at - time.time())/60:.1f} min")
    return 0


def _cmd_accounts(args: argparse.Namespace) -> int:
    client = _client_from_env()
    rows = parse_positions(client.accounts(with_positions=True))
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no positions returned")
        return 0
    print(f"{'ACCOUNT':<12}{'SYMBOL':<22}{'TYPE':<10}{'QTY':>10}{'AVG':>12}{'MKT VAL':>14}")
    for r in rows:
        print(
            f"{r['account'][:10]:<12}{r['symbol'][:20]:<22}{r['asset_type'][:8]:<10}"
            f"{r['quantity']:>10.0f}{r['average_price']:>12.2f}{r['market_value']:>14.2f}"
        )
    return 0


def _cmd_quote(args: argparse.Namespace) -> int:
    client = _client_from_env()
    payload = client.quote_raw(args.symbol)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    q = parse_quote(args.symbol, payload)
    print(f"{q.symbol}  price={q.price}  iv={q.iv:.2%}  div_yield={q.div_yield:.2%}  as_of={q.as_of}")
    return 0


def _cmd_chain(args: argparse.Namespace) -> int:
    client = _client_from_env()
    payload = client.chain_raw(args.symbol, strike_count=args.strikes)
    contracts = parse_chain(payload)
    if args.dte is not None:
        contracts = [c for c in contracts if c.dte <= args.dte]
    if args.json:
        print(json.dumps([c.to_dict() for c in contracts], indent=2))
        return 0
    print(f"{len(contracts)} contracts for {args.symbol.upper()}")
    print(f"{'EXPIRY':<12}{'R':<3}{'STRIKE':>10}{'BID':>9}{'ASK':>9}{'IV':>8}{'OI':>9}{'DTE':>6}")
    for c in contracts[: args.limit]:
        print(
            f"{c.expiry.isoformat():<12}{c.right:<3}{c.strike:>10.2f}{c.bid:>9.2f}"
            f"{c.ask:>9.2f}{c.iv:>8.2%}{c.open_interest:>9d}{c.dte:>6d}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wheel.schwab", description="Schwab API setup + read-only data")
    sub = p.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login", help="run the OAuth authorization-code flow")
    login.add_argument("--redirect-url", dest="redirect_url", default=None,
                       help="paste the full redirect URL instead of being prompted")
    login.set_defaults(func=_cmd_login)

    sub.add_parser("status", help="show token validity").set_defaults(func=_cmd_status)
    sub.add_parser("refresh", help="force an access-token refresh").set_defaults(func=_cmd_refresh)

    accounts = sub.add_parser("accounts", help="list positions (read-only)")
    accounts.add_argument("--json", action="store_true")
    accounts.set_defaults(func=_cmd_accounts)

    quote = sub.add_parser("quote", help="underlying quote")
    quote.add_argument("symbol")
    quote.add_argument("--json", action="store_true")
    quote.set_defaults(func=_cmd_quote)

    chain = sub.add_parser("chain", help="option chain")
    chain.add_argument("symbol")
    chain.add_argument("--dte", type=int, default=None, help="only contracts within N days")
    chain.add_argument("--strikes", type=int, default=40, help="strikeCount sent to Schwab")
    chain.add_argument("--limit", type=int, default=40, help="rows to print")
    chain.add_argument("--json", action="store_true")
    chain.set_defaults(func=_cmd_chain)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (SchwabError, LiveTradingDisabled) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SchwabAuth",
    "SchwabAuthError",
    "SchwabClient",
    "SchwabCredentials",
    "SchwabError",
    "SchwabMarketData",
    "TokenSet",
    "TokenStore",
    "build_authorize_url",
    "extract_code",
    "main",
    "parse_chain",
    "parse_positions",
    "parse_quote",
]
