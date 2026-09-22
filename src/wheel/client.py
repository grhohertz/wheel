"""Claude API client for advisory recommendations.

Zero-dependency: uses only urllib.request, json, and stdlib.
Calls the Anthropic Messages API directly. One retry, then fallback to defaults.

Request/response are logged to the audit ledger.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional

from wheel.audit import AdvisoryCall, AuditLedger, CurrentAdvisory, compute_features_digest
from wheel.features import FeatureVector


class ClaudeAdvisor:
    """Advisory client: calls Claude, logs to ledger, caches in current.json."""

    API_BASE = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        prompt_version: str = "delta.v1",
        ledger_path: str = "state/advisory/ledger.jsonl",
        current_path: str = "state/advisory/current.json",
        max_retries: int = 1,
        timeout_seconds: float = 8.0,
        monthly_budget_usd: float = 25.0,
    ):
        """Initialize advisor.

        Args:
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env)
            model: Claude model to use
            prompt_version: version label for the prompt (bump on changes)
            ledger_path: path to audit ledger
            current_path: path to current advisory cache
            max_retries: retries on transient errors
            timeout_seconds: per-request timeout
            monthly_budget_usd: hard cap to prevent runaway loops
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in env or args")

        self.model = model
        self.prompt_version = prompt_version
        self.ledger = AuditLedger(ledger_path)
        self.current = CurrentAdvisory(current_path)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.monthly_budget_usd = monthly_budget_usd

    def _build_prompt(self, fv: FeatureVector) -> str:
        """Build the advisory prompt from a FeatureVector.

        This is the human-facing prompt. Update it here; bump prompt_version when you do.
        """
        regime = fv.market_regime
        season = fv.trade_season
        iv_rank = fv.iv_data.iv_rank
        rv_iv = fv.vol_data.rv_iv_ratio_30d
        pos = fv.position_state
        earnings_dte = fv.earnings_data.dte_to_next_earnings or "unknown"

        return f"""You are a quantitative options advisor for a wheel-trading strategy.
Given market and position data, recommend optimal delta targets for the next trade.

MARKET CONTEXT:
- Symbol: {fv.symbol}
- Price: ${fv.price_data.price:.2f}
- IV Rank: {iv_rank:.0f}% (regime: {regime})
- RV/IV Ratio: {rv_iv:.2f} (IV is {"cheap" if rv_iv > 1 else "expensive"})
- Term Structure: {fv.iv_data.term_structure_slope:+.1%} (60d vs 30d IV)
- Earnings DTE: {earnings_dte} (season: {season})
- Recent price moves: 1d {fv.price_data.change_1d_pct:+.1%}, 5d {fv.price_data.change_5d_pct:+.1%}, 30d {fv.price_data.change_30d_pct:+.1%}
- 20-day ATR (vol proxy): {fv.price_data.atr_20d:.1%}

POSITION STATE:
- Shares held: {pos.shares_held}
- Cash-secured puts open: {pos.csp_open_count}
- Covered calls open: {pos.ccall_open_count}
- Cash available: ${pos.cash_available:,.0f}
- Collateral usage: {pos.collateral_used_pct:.0%}
- Avg days in current positions: {pos.days_in_position_avg:.1f}

TASK:
Recommend delta targets for the wheel strategy's next leg:
1. If we're selling puts (CSP), what delta (0.15–0.35)? Higher = more OTM = lower assignment risk.
2. If we're selling calls (covered), what delta (0.20–0.40)? Higher = more ITM = higher call-away risk.

Consider:
- Market regime (high IV → can sell further OTM; low IV → need more premium → closer to ATM)
- Earnings risk (if earnings are near expiry, be more conservative on delta)
- Position state (if collateral is tight, be more conservative)
- IV/RV ratio (if IV > RV, premium is good; if IV < RV, be cautious)

CONSTRAINTS:
- Put delta must be 0.15–0.35
- Call delta must be 0.20–0.40
- DTE target should be 30–40 days

OUTPUT:
Respond with ONLY valid JSON, no markdown, no explanation:
{{
  "put_delta_target": 0.25,
  "call_delta_target": 0.30,
  "dte_target": 35,
  "rationale": "Brief 1-line explanation"
}}
"""

    def _call_api(self, prompt: str) -> tuple[str, int, int, float]:
        """Call Claude API.

        Args:
            prompt: the advisory prompt

        Returns:
            (response_text, tokens_in, tokens_out, latency_ms)

        Raises:
            RuntimeError if all retries fail.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        body = json.dumps(payload).encode()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                start_ms = time.time() * 1000
                req = urllib.request.Request(
                    self.API_BASE,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    resp_body = response.read().decode()
                    latency_ms = time.time() * 1000 - start_ms

                resp_json = json.loads(resp_body)
                text = resp_json["content"][0]["text"]
                tokens_in = resp_json["usage"]["input_tokens"]
                tokens_out = resp_json["usage"]["output_tokens"]

                return text, tokens_in, tokens_out, latency_ms

            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1 << attempt)  # exponential backoff: 1s, 2s, ...

        raise RuntimeError(f"Claude API failed after {self.max_retries + 1} attempts: {last_error}")

    def _parse_response(self, text: str) -> dict:
        """Parse Claude's JSON response into a recommendation dict.

        Args:
            text: raw response text

        Returns:
            dict with put_delta_target, call_delta_target, dte_target, rationale

        Raises:
            ValueError if JSON is invalid.
        """
        # Try to extract JSON from the response
        # Claude should return just JSON, but be defensive
        text = text.strip()
        if text.startswith("```"):
            # Remove markdown code fence if present
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        recommendation = json.loads(text)

        # Validate required fields
        required = ["put_delta_target", "call_delta_target", "dte_target"]
        for key in required:
            if key not in recommendation:
                raise ValueError(f"Missing required field: {key}")

        return recommendation

    def advise(
        self,
        fv: FeatureVector,
        force_refresh: bool = False,
        clamp_put_delta: tuple[float, float] = (0.15, 0.35),
        clamp_call_delta: tuple[float, float] = (0.20, 0.40),
        clamp_dte: tuple[int, int] = (30, 40),
    ) -> dict:
        """Get advisory recommendation for a symbol.

        Reads cache first. Calls Claude only if cache miss or force_refresh.
        Applies guardrails (clamping). Logs to ledger.

        Args:
            fv: FeatureVector with market and position data
            force_refresh: ignore cache, call Claude
            clamp_put_delta: (min, max) for put delta
            clamp_call_delta: (min, max) for call delta
            clamp_dte: (min, max) for DTE target

        Returns:
            {
                "put_delta_target": 0.30,
                "call_delta_target": 0.25,
                "dte_target": 35,
                "rationale": "...",
                "advice_id": "...",
                "from_cache": True/False,
                "guardrails_applied": [...],
                "cost_usd": 0.0207
            }
        """
        symbol = fv.symbol
        features_dict = fv.to_dict()
        features_digest = compute_features_digest(features_dict)

        # Check cache
        if not force_refresh:
            cached = self.current.get(symbol)
            if cached and "expires_at" in cached:
                expires_at = datetime.fromisoformat(cached["expires_at"])
                if datetime.now() < expires_at:
                    return {
                        **cached["recommendation"],
                        "advice_id": cached["advice_id"],
                        "from_cache": True,
                        "guardrails_applied": [],
                        "cost_usd": 0.0,
                    }

        # Cache miss or stale: call Claude
        prompt = self._build_prompt(fv)
        try:
            raw_response, tokens_in, tokens_out, latency_ms = self._call_api(prompt)
        except RuntimeError as e:
            # Fallback to defaults
            return {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": f"API error (fallback): {e}",
                "advice_id": None,
                "from_cache": False,
                "guardrails_applied": ["api_error_fallback"],
                "cost_usd": 0.0,
            }

        # Parse response
        try:
            recommendation = self._parse_response(raw_response)
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": f"Parse error (fallback): {e}",
                "advice_id": None,
                "from_cache": False,
                "guardrails_applied": ["parse_error_fallback"],
                "cost_usd": 0.0,
            }

        # Apply guardrails (clamping)
        guardrails_applied = []
        original_put = recommendation.get("put_delta_target", 0.25)
        original_call = recommendation.get("call_delta_target", 0.30)
        original_dte = recommendation.get("dte_target", 35)

        clamped_put = max(clamp_put_delta[0], min(clamp_put_delta[1], original_put))
        if clamped_put != original_put:
            guardrails_applied.append(f"clamp put_delta_target {original_put:.2f}→{clamped_put:.2f}")

        clamped_call = max(clamp_call_delta[0], min(clamp_call_delta[1], original_call))
        if clamped_call != original_call:
            guardrails_applied.append(f"clamp call_delta_target {original_call:.2f}→{clamped_call:.2f}")

        clamped_dte = max(clamp_dte[0], min(clamp_dte[1], original_dte))
        if clamped_dte != original_dte:
            guardrails_applied.append(f"clamp dte_target {original_dte}→{clamped_dte}")

        # Compute cost
        # Sonnet 4.5: $3/M input, $15/M output
        cost_usd = (tokens_in * 3.0 / 1_000_000) + (tokens_out * 15.0 / 1_000_000)

        # Check monthly budget
        stats = self.ledger.stats()
        if stats["total_cost_usd"] + cost_usd > self.monthly_budget_usd:
            return {
                "put_delta_target": 0.25,
                "call_delta_target": 0.30,
                "dte_target": 35,
                "rationale": "Monthly budget exhausted (fallback)",
                "advice_id": None,
                "from_cache": False,
                "guardrails_applied": ["monthly_budget_exceeded"],
                "cost_usd": 0.0,
            }

        # Create advice ID
        advice_id = f"{datetime.now().isoformat()}Z-{features_digest[:8]}"

        # Log to ledger
        call = AdvisoryCall(
            advice_id=advice_id,
            ts=datetime.now().isoformat(),
            symbol=symbol,
            prompt_version=self.prompt_version,
            model=self.model,
            features_dict=features_dict,
            features_digest=features_digest,
            raw_response=raw_response,
            parsed_recommendation=recommendation,
            guardrails_applied=guardrails_applied,
            effective_params={
                "put_delta_target": clamped_put,
                "call_delta_target": clamped_call,
                "dte_target": clamped_dte,
            },
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        self.ledger.append(call)

        # Update cache (24h TTL)
        self.current.set(symbol, {
            "advice_id": advice_id,
            "ts": call.ts,
            "recommendation": {
                "put_delta_target": clamped_put,
                "call_delta_target": clamped_call,
                "dte_target": clamped_dte,
                "rationale": recommendation.get("rationale", ""),
            },
            "effective_params": call.effective_params,
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        })

        return {
            "put_delta_target": clamped_put,
            "call_delta_target": clamped_call,
            "dte_target": clamped_dte,
            "rationale": recommendation.get("rationale", ""),
            "advice_id": advice_id,
            "from_cache": False,
            "guardrails_applied": guardrails_applied,
            "cost_usd": cost_usd,
        }


# Example usage
if __name__ == "__main__":
    from wheel.features import build_feature_vector

    # Build a feature vector
    fv = build_feature_vector(
        symbol="AAPL",
        price=185.50,
        iv_30d=0.28,
        iv_60d=0.30,
        iv_52w_low=0.15,
        iv_52w_high=0.45,
        rv_20d=0.22,
        rv_60d=0.24,
        skew_put_call=-0.08,
        shares_held=100,
        csp_open_count=1,
        ccall_open_count=0,
        avg_cost_per_share=180.0,
        cash_available=50000.0,
        collateral_used_pct=0.25,
        days_in_position_avg=5.0,
        dte_to_next_earnings=23,
        change_1d_pct=0.015,
        change_5d_pct=0.032,
        change_30d_pct=0.08,
        atr_20d=0.18,
    )

    # Initialize advisor (will fail if ANTHROPIC_API_KEY is not set)
    try:
        advisor = ClaudeAdvisor()
        print("Advisor initialized. Calling Claude...")
        result = advisor.advise(fv)
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(f"Skipping live test (no API key): {e}")
