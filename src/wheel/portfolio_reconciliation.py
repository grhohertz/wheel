"""Portfolio reconciliation — track fills, update state, maintain consistency.

Monitors for fills on pending rolls, updates portfolio positions, and
keeps the paper trading state in sync with live fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
from pathlib import Path
import json


@dataclass
class PendingRoll:
    """A roll that was recommended but not yet filled."""
    symbol: str
    underlying: str
    old_strike: float
    old_expiry: str
    new_strike: float
    new_expiry: str
    quantity: int
    target_delta: float
    recommended_at: str  # ISO 8601 timestamp
    old_order_id: Optional[str] = None  # Schwab order ID for BUY-TO-CLOSE
    new_order_id: Optional[str] = None  # Schwab order ID for SELL-TO-OPEN
    old_filled_at: Optional[str] = None  # When BUY-TO-CLOSE filled
    new_filled_at: Optional[str] = None  # When SELL-TO-OPEN filled
    old_fill_price: Optional[float] = None
    new_fill_price: Optional[float] = None
    status: str = "pending"  # pending | partially_filled | filled | expired | cancelled
    cancel_reason: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Returns True if both legs are filled."""
        return self.old_filled_at is not None and self.new_filled_at is not None
    
    @property
    def net_credit(self) -> Optional[float]:
        """Returns net credit received (new - old)."""
        if not self.is_complete:
            return None
        return (self.new_fill_price or 0.0) - (self.old_fill_price or 0.0)


@dataclass
class RollHistory:
    """Complete history of rolls for audit and PnL tracking."""
    rolls: list[PendingRoll] = field(default_factory=list)
    
    def add_roll(self, roll: PendingRoll) -> None:
        """Add a new pending roll to history."""
        self.rolls.append(roll)
    
    def find_by_symbol_expiry(self, symbol: str, old_expiry: str) -> Optional[PendingRoll]:
        """Find a pending roll by symbol and old expiry."""
        for roll in self.rolls:
            if roll.symbol == symbol and roll.old_expiry == old_expiry and roll.status == "pending":
                return roll
        return None
    
    def update_roll_status(
        self,
        symbol: str,
        old_expiry: str,
        old_filled_at: Optional[str] = None,
        old_fill_price: Optional[float] = None,
        new_filled_at: Optional[str] = None,
        new_fill_price: Optional[float] = None,
    ) -> None:
        """Update roll status as fills come in."""
        roll = self.find_by_symbol_expiry(symbol, old_expiry)
        if not roll:
            return
        
        if old_filled_at:
            roll.old_filled_at = old_filled_at
            roll.old_fill_price = old_fill_price
        
        if new_filled_at:
            roll.new_filled_at = new_filled_at
            roll.new_fill_price = new_fill_price
        
        # Update status
        if roll.is_complete:
            roll.status = "filled"
        elif roll.old_filled_at or roll.new_filled_at:
            roll.status = "partially_filled"
    
    def mark_expired(self, symbol: str, old_expiry: str, reason: str = "expiry reached") -> None:
        """Mark a roll as expired without filling."""
        roll = self.find_by_symbol_expiry(symbol, old_expiry)
        if roll:
            roll.status = "expired"
            roll.cancel_reason = reason
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "rolls": [
                {
                    "symbol": r.symbol,
                    "underlying": r.underlying,
                    "old_strike": r.old_strike,
                    "old_expiry": r.old_expiry,
                    "new_strike": r.new_strike,
                    "new_expiry": r.new_expiry,
                    "quantity": r.quantity,
                    "target_delta": r.target_delta,
                    "recommended_at": r.recommended_at,
                    "old_order_id": r.old_order_id,
                    "new_order_id": r.new_order_id,
                    "old_filled_at": r.old_filled_at,
                    "new_filled_at": r.new_filled_at,
                    "old_fill_price": r.old_fill_price,
                    "new_fill_price": r.new_fill_price,
                    "status": r.status,
                    "cancel_reason": r.cancel_reason,
                    "net_credit": r.net_credit,
                }
                for r in self.rolls
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> RollHistory:
        """Deserialize from JSON-compatible dict."""
        history = cls()
        for roll_data in data.get("rolls", []):
            roll = PendingRoll(
                symbol=roll_data["symbol"],
                underlying=roll_data["underlying"],
                old_strike=roll_data["old_strike"],
                old_expiry=roll_data["old_expiry"],
                new_strike=roll_data["new_strike"],
                new_expiry=roll_data["new_expiry"],
                quantity=roll_data["quantity"],
                target_delta=roll_data["target_delta"],
                recommended_at=roll_data["recommended_at"],
                old_order_id=roll_data.get("old_order_id"),
                new_order_id=roll_data.get("new_order_id"),
                old_filled_at=roll_data.get("old_filled_at"),
                new_filled_at=roll_data.get("new_filled_at"),
                old_fill_price=roll_data.get("old_fill_price"),
                new_fill_price=roll_data.get("new_fill_price"),
                status=roll_data.get("status", "pending"),
                cancel_reason=roll_data.get("cancel_reason"),
            )
            history.rolls.append(roll)
        return history
    
    def save(self, path: Path) -> None:
        """Persist roll history to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> RollHistory:
        """Load roll history from disk."""
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
    
    def net_credit_ytd(self) -> float:
        """Total net credit from all completed rolls this year."""
        return sum(r.net_credit or 0.0 for r in self.rolls if r.status == "filled")
    
    def completed_rolls(self) -> list[PendingRoll]:
        """Return all completed rolls."""
        return [r for r in self.rolls if r.status == "filled"]
    
    def pending_rolls(self) -> list[PendingRoll]:
        """Return all pending or partially-filled rolls."""
        return [r for r in self.rolls if r.status in ("pending", "partially_filled")]
    
    def expired_rolls(self) -> list[PendingRoll]:
        """Return all expired rolls."""
        return [r for r in self.rolls if r.status == "expired"]


class PortfolioReconciler:
    """Reconcile portfolio state with fills and maintain consistency."""
    
    def __init__(self, state_path: Path = Path("state/paper_account.json")):
        self.state_path = state_path
        self.roll_history_path = state_path.parent / "roll_history.json"
        self.roll_history = RollHistory.load(self.roll_history_path)
    
    def record_roll_recommendation(
        self,
        symbol: str,
        underlying: str,
        old_strike: float,
        old_expiry: str,
        new_strike: float,
        new_expiry: str,
        quantity: int,
        target_delta: float,
    ) -> PendingRoll:
        """Record a new roll recommendation."""
        roll = PendingRoll(
            symbol=symbol,
            underlying=underlying,
            old_strike=old_strike,
            old_expiry=old_expiry,
            new_strike=new_strike,
            new_expiry=new_expiry,
            quantity=quantity,
            target_delta=target_delta,
            recommended_at=datetime.now().isoformat(),
        )
        self.roll_history.add_roll(roll)
        self.roll_history.save(self.roll_history_path)
        return roll
    
    def record_fill(
        self,
        symbol: str,
        old_expiry: str,
        leg: str,  # "old" or "new"
        price: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a fill on a roll leg."""
        timestamp = timestamp or datetime.now().isoformat()
        
        if leg == "old":
            self.roll_history.update_roll_status(
                symbol,
                old_expiry,
                old_filled_at=timestamp,
                old_fill_price=price,
            )
        elif leg == "new":
            self.roll_history.update_roll_status(
                symbol,
                old_expiry,
                new_filled_at=timestamp,
                new_fill_price=price,
            )
        
        self.roll_history.save(self.roll_history_path)
    
    def check_roll_expiry(self, symbol: str, old_expiry: str) -> None:
        """Mark a roll as expired if it wasn't filled by the target expiry."""
        exp_date = datetime.strptime(old_expiry, "%Y-%m-%d").date()
        now = date.today()
        
        if now >= exp_date:
            self.roll_history.mark_expired(symbol, old_expiry)
            self.roll_history.save(self.roll_history_path)
    
    def summary(self) -> dict:
        """Generate a reconciliation summary."""
        pending = self.roll_history.pending_rolls()
        completed = self.roll_history.completed_rolls()
        expired = self.roll_history.expired_rolls()
        
        return {
            "total_rolls_recommended": len(self.roll_history.rolls),
            "pending_or_partial": len(pending),
            "completed": len(completed),
            "expired": len(expired),
            "net_credit_ytd": self.roll_history.net_credit_ytd(),
            "pending_rolls": [
                {
                    "symbol": r.symbol,
                    "status": r.status,
                    "recommended_at": r.recommended_at,
                    "net_credit": r.net_credit,
                }
                for r in pending
            ],
        }
