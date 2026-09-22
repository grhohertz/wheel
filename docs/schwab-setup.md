# Schwab API — setup and configuration

End-to-end guide to wiring real Schwab market data into the wheel engine.
**Read-only by design:** quotes, option chains, and account reads. Orders stay
simulated in `wheel.broker.PaperBroker` — `SchwabClient.place_order()` raises
`LiveTradingDisabled`.

---

## 1. Create the Schwab developer app

1. Go to <https://developer.schwab.com> and sign in with your Schwab.com
   credentials (the same login as the brokerage account).
2. **Dashboard → Apps → Create App**.
3. Select API products:
   - **Accounts and Trading Production** — needed for `/trader/v1/accounts`
     (position reads). Optional if you only want market data.
   - **Market Data Production** — needed for `/marketdata/v1/quotes` and
     `/marketdata/v1/chains`.
4. **Callback URL**: must be HTTPS. Use exactly:

   ```
   https://127.0.0.1:8182
   ```

   Nothing has to listen there — the login flow below copies the redirect URL
   out of the browser address bar by hand.
5. Submit. The app sits in **Approved - Pending** for a short while, then flips
   to **Ready For Use**. You cannot authenticate until it says Ready For Use.
6. Copy the **App Key** (client id) and **App Secret**.

> Schwab has no separate paper/sandbox environment. The API points at your real
> account — which is why this integration reads only, and all fills remain
> simulated locally.

---

## 2. Configure the environment

Copy `.env.example` to `.env` and fill it in (the file is gitignored), or export
directly:

```bash
export SCHWAB_APP_KEY='your-app-key'
export SCHWAB_APP_SECRET='your-app-secret'
export SCHWAB_CALLBACK_URL='https://127.0.0.1:8182'   # must match the app exactly
export SCHWAB_TOKEN_PATH='state/schwab_tokens.json'   # default; state/ is gitignored
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SCHWAB_APP_KEY` | yes | — | OAuth client id |
| `SCHWAB_APP_SECRET` | yes | — | OAuth client secret (Basic auth) |
| `SCHWAB_CALLBACK_URL` | no | `https://127.0.0.1:8182` | Must byte-match the app's callback |
| `SCHWAB_TOKEN_PATH` | no | `state/schwab_tokens.json` | Token file, written `0600` |

Never commit `.env` or the token file. `state/` and `.env` are both in
`.gitignore`.

---

## 3. Log in (authorization-code grant)

```bash
python -m wheel.schwab login
```

The command prints an authorize URL. Open it in a browser, approve the account,
and the browser lands on `https://127.0.0.1:8182/?code=...&session=...` which
**will fail to load — that is expected**. Copy the entire address bar and paste
it back at the prompt.

Non-interactive variant:

```bash
python -m wheel.schwab login --redirect-url 'https://127.0.0.1:8182/?code=C0.b...%40&session=...'
```

Tokens are written to `SCHWAB_TOKEN_PATH` with mode `0600`.

### Token lifetimes

| Token | Lifetime | Renewal |
|---|---|---|
| Access token | 30 minutes | Automatic — every client call refreshes when < 2 min remain |
| Refresh token | 7 days | Manual — re-run `login` once a week |

Check state any time:

```bash
python -m wheel.schwab status    # exit 1 if a re-login is needed
python -m wheel.schwab refresh   # force a new access token
```

---

## 4. Verify

```bash
python -m wheel.schwab quote GLD
python -m wheel.schwab chain GLD --dte 45 --limit 20
python -m wheel.schwab accounts            # requires the Trading API product
python -m wheel.schwab accounts --json
```

Expected quote output:

```
GLD  price=245.31  iv=12.50%  div_yield=1.80%  as_of=2026-03-03
```

---

## 5. Use it in the engine

`SchwabMarketData` implements the same two-method `MarketData` protocol as
`SyntheticMarketData`, so it is a drop-in swap — the broker stays simulated:

```python
from wheel.schwab import SchwabAuth, SchwabClient, SchwabCredentials, SchwabMarketData
from wheel.engine import WheelEngine
from wheel.config import Settings

creds  = SchwabCredentials.from_env()
client = SchwabClient(SchwabAuth(creds))
market = SchwabMarketData(client, strike_count=40, cache_seconds=30)

engine = WheelEngine(settings=Settings.from_env(), market=market)
```

Real prices, fake fills. Quotes and chains are cached for `cache_seconds` to
stay inside Schwab's rate limit (120 requests/minute per app).

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `token request failed (400): unsupported_token_type` | Callback URL mismatch | `SCHWAB_CALLBACK_URL` must byte-match the app's registered callback, including scheme and port |
| `token request failed (400): invalid_grant` | Code already used or > 30 s old | Re-run `login`; authorization codes are single-use and short-lived |
| `401 from Schwab — token rejected` | Refresh token older than 7 days | Re-run `login` |
| `no ?code= found in that URL` | Pasted the authorize URL instead of the redirect | Paste the address bar *after* approving |
| `403` on `/trader/v1/accounts` | App lacks the Accounts and Trading product, or is still pending | Add the product on the dashboard, wait for **Ready For Use** |
| `network error ... certificate verify failed` | Corporate TLS interception | Point `SSL_CERT_FILE` at your CA bundle |
| Empty chain | Symbol has no listed options, or market closed with no quotes | Try a liquid ETF (`GLD`, `SPY`) during market hours |

---

## 7. Safety model

- `SchwabClient` exposes **no** order endpoints; `place_order()` raises
  `LiveTradingDisabled`.
- `wheel.config.assert_paper_mode()` still gates the engine — `WHEEL_MODE` must
  be `paper`.
- The only writes this integration performs are to the local token file.
