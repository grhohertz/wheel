"""Advisory seam layer — the only door between the engine and the advisor.

The strategy and the engine never talk to Claude, never read the ledger, and
never import :mod:`wheel.client`. They ask *this* module for a number and this
module decides where that number comes from:

``ADVISOR_MODE=off`` (default)
    Always return the caller's own config default. Zero file I/O, zero network,
    zero behaviour change. This is what the test-suite and every existing
    workflow run under.

``ADVISOR_MODE=shadow``
    The cached advisory *is* read and logged, but the caller still gets its
    config default back. Lets you run the advisor against a live book and diff
    what it *would* have done without letting it trade.

``ADVISOR_MODE=overlay``
    The cached advisory wins — clamped into the guardrail band first.

Every accessor is total: a missing cache file, corrupt JSON, an expired entry,
or a nonsense value all degrade to the caller's default rather than raising.
An advisory layer that can halt the engine is worse than no advisory layer.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .regime import RegimeOverlay

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------
OFF = "off"
SHADOW = "shadow"
OVERLAY = "overlay"
MODES = (OFF, SHADOW, OVERLAY)

DEFAULT_ADVISORY_PATH = "state/advisory/current.json"

# ----------------------------------------------------------------------
# guardrails — an advisory can never push a knob outside these bands
# ----------------------------------------------------------------------
PUT_DELTA_CLAMP = (0.15, 0.35)
CALL_DELTA_CLAMP = (0.20, 0.40)
DTE_CLAMP = (21, 45)

_LEG_KEYS = {
    "csp": ("put_delta_target", PUT_DELTA_CLAMP),
    "put": ("put_delta_target", PUT_DELTA_CLAMP),
    "ccall": ("call_delta_target", CALL_DELTA_CLAMP),
    "call": ("call_delta_target", CALL_DELTA_CLAMP),
}


def clamp(value: float, bounds: tuple[float, float]) -> float:
    """Squeeze ``value`` into ``bounds``. Used on every advisory number."""

    lo, hi = bounds
    return max(lo, min(hi, value))


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AdviceConfig:
    """How the policy layer should behave this process."""

    mode: str = OFF
    advisory_path: str = DEFAULT_ADVISORY_PATH
    honour_ttl: bool = True

    @property
    def enabled(self) -> bool:
        """True when the advisory file should be read at all."""

        return self.mode in (SHADOW, OVERLAY)

    @property
    def authoritative(self) -> bool:
        """True when advisory values actually replace config defaults."""

        return self.mode == OVERLAY

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "AdviceConfig":
        env = dict(os.environ if env is None else env)
        mode = env.get("ADVISOR_MODE", OFF).strip().lower()
        if mode not in MODES:
            logger.warning("ADVISOR_MODE=%r not recognised; falling back to %r", mode, OFF)
            mode = OFF
        return cls(
            mode=mode,
            advisory_path=env.get("WHEEL_ADVISORY_PATH", DEFAULT_ADVISORY_PATH),
            honour_ttl=env.get("WHEEL_ADVISORY_IGNORE_TTL", "").strip().lower()
            not in ("1", "true", "yes"),
        )


def advisor_mode(env: Optional[dict[str, str]] = None) -> str:
    """Current advisor mode — ``off`` unless ``ADVISOR_MODE`` says otherwise."""

    return AdviceConfig.from_env(env).mode


# ----------------------------------------------------------------------
# cache access
# ----------------------------------------------------------------------
_cache: dict[str, Any] | None = None
_cache_path: str | None = None


def clear_cache() -> None:
    """Drop the in-process advisory cache. Tests call this between cases."""

    global _cache, _cache_path
    _cache = None
    _cache_path = None


def _read_advisory_file(path: str) -> dict[str, Any]:
    """Load the advisory JSON, memoised per path. Never raises."""

    global _cache, _cache_path
    if _cache is not None and _cache_path == path:
        return _cache

    data: dict[str, Any] = {}
    try:
        p = Path(path)
        if p.exists():
            with open(p) as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("advisory cache unreadable at %s: %s", path, exc)

    _cache, _cache_path = data, path
    return data


def _expired(entry: dict[str, Any]) -> bool:
    raw = entry.get("expires_at")
    if not raw:
        return True
    try:
        return datetime.now() > datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return True


def load_advice(
    symbol: str, cfg: Optional[AdviceConfig] = None
) -> Optional[dict[str, Any]]:
    """Return the live recommendation dict for ``symbol``, or ``None``.

    ``None`` means "no opinion" for every reason that can occur: advisor is
    off, the file is missing, the symbol was never advised, or the entry aged
    out of its TTL.
    """

    cfg = cfg or AdviceConfig.from_env()
    if not cfg.enabled:
        return None

    entry = _read_advisory_file(cfg.advisory_path).get((symbol or "").upper())
    if not isinstance(entry, dict):
        return None

    if cfg.honour_ttl and _expired(entry):
        logger.debug("advisory for %s expired; using defaults", symbol)
        return None

    rec = entry.get("recommendation")
    return rec if isinstance(rec, dict) else None


# ----------------------------------------------------------------------
# accessors — what the engine actually calls
# ----------------------------------------------------------------------
def get_delta_target(
    symbol: str,
    leg: str = "csp",
    default: float = 0.25,
    cfg: Optional[AdviceConfig] = None,
) -> float:
    """Delta target for ``symbol``'s ``leg``, clamped into the guardrail band.

    Args:
        symbol: underlying ticker.
        leg: ``csp``/``put`` for the short-put leg, ``ccall``/``call`` for the
            covered-call leg.
        default: what to return when there is no usable advisory.
        cfg: override the env-derived config (tests).

    Returns:
        The advisory target in ``overlay`` mode, otherwise ``default``.
    """

    cfg = cfg or AdviceConfig.from_env()
    key, bounds = _LEG_KEYS.get((leg or "").lower(), _LEG_KEYS["csp"])

    rec = load_advice(symbol, cfg)
    if not rec:
        return default

    raw = rec.get(key)
    if not isinstance(raw, (int, float)):
        return default

    target = clamp(float(raw), bounds)
    if not cfg.authoritative:
        logger.info(
            "[shadow] %s %s advisory delta=%.3f (clamped from %.3f); using default %.3f",
            symbol, leg, target, float(raw), default,
        )
        return default

    if abs(target - float(raw)) > 1e-9:
        logger.warning(
            "advisory delta for %s %s clamped %.3f -> %.3f", symbol, leg, float(raw), target
        )
    return target


def get_dte_target(
    symbol: str, default: int = 35, cfg: Optional[AdviceConfig] = None
) -> int:
    """DTE target for ``symbol``, clamped to :data:`DTE_CLAMP`."""

    cfg = cfg or AdviceConfig.from_env()
    rec = load_advice(symbol, cfg)
    if not rec:
        return default

    raw = rec.get("dte_target")
    if not isinstance(raw, (int, float)):
        return default

    target = int(round(clamp(float(raw), DTE_CLAMP)))
    if not cfg.authoritative:
        logger.info("[shadow] %s advisory dte=%d; using default %d", symbol, target, default)
        return default
    return target


def get_mc_overlay(
    symbol: str, cfg: Optional[AdviceConfig] = None
) -> "RegimeOverlay":
    """Monte-Carlo vol/drift overlay for ``symbol`` — the Phase 4 seam.

    Resolution order:

    1. An explicit ``mc_overlay`` block on the advisory (hand-tuned or
       model-supplied multipliers).
    2. The advisory's ``regime`` label, mapped through the calibrated
       :data:`wheel.regime.REGIME_OVERLAYS` table.
    3. The identity overlay — simulate the raw parameters unchanged.

    In ``shadow`` mode the overlay is resolved and logged but the identity is
    returned, so a simulation never silently changes shape.
    """

    from .regime import RegimeOverlay, overlay_for  # local: avoids an import cycle

    cfg = cfg or AdviceConfig.from_env()
    rec = load_advice(symbol, cfg)
    if not rec:
        return RegimeOverlay.identity()

    block = rec.get("mc_overlay")
    if isinstance(block, dict):
        resolved = RegimeOverlay.from_dict(block, source="advisory")
    else:
        regime = rec.get("regime")
        if not isinstance(regime, str):
            return RegimeOverlay.identity()
        resolved = overlay_for(regime, source="advisory")

    if not cfg.authoritative:
        logger.info(
            "[shadow] %s advisory regime=%s vol_mult=%.2f mu_shift=%+.3f; simulating raw params",
            symbol, resolved.regime, resolved.vol_mult, resolved.mu_shift,
        )
        return RegimeOverlay.identity()
    return resolved


__all__ = [
    "AdviceConfig",
    "CALL_DELTA_CLAMP",
    "DTE_CLAMP",
    "OFF",
    "OVERLAY",
    "PUT_DELTA_CLAMP",
    "SHADOW",
    "advisor_mode",
    "clamp",
    "clear_cache",
    "get_delta_target",
    "get_dte_target",
    "get_mc_overlay",
    "load_advice",
]
