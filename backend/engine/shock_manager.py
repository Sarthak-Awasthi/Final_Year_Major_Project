"""
shock_manager.py — Dynamic Shock Engine for the RL Playground.

Shocks are explicit world modifiers with lifecycle:
  activation → propagation → decay → expiry

They modify NPC rewards, adaptation pressure, and resource channels
to create non-stationary environments requiring agent adaptation.
"""

from __future__ import annotations

from typing import Any

from backend.config import SHOCK_CATALOG, SHOCK_MAX_ACTIVE, logger


class ShockManager:
    """Manages shock lifecycle: activation, decay, expiry, and effect aggregation."""

    def __init__(self) -> None:
        self._active_shocks: list[dict] = []
        self._shock_history: list[dict] = []  # expired shocks (bounded)
        self._next_id: int = 1
        self._max_history: int = 50

    # ── Activation ────────────────────────────────────────────────────────

    def activate_shock(
        self,
        shock_type: str,
        source: str = "event",
        scope: str = "village",
        target_location: str | None = None,
        duration: int | None = None,
        turn: int = 0,
    ) -> dict | None:
        """Create and register a new shock.

        Args:
            shock_type: Key into SHOCK_CATALOG (e.g. "famine", "bandit_raid").
            source: What triggered it ("player", "event", "quest").
            scope: "village" (global) or "location" (localized).
            target_location: Required if scope == "location".
            duration: Override catalog default duration.
            turn: Current game turn.

        Returns:
            The created shock dict, or None if type unknown or max active reached.
        """
        if len(self._active_shocks) >= SHOCK_MAX_ACTIVE:
            logger.warning("Max active shocks (%d) reached — ignoring %s", SHOCK_MAX_ACTIVE, shock_type)
            return None

        catalog_entry = SHOCK_CATALOG.get(shock_type)
        if catalog_entry is None:
            logger.warning("Unknown shock type: %s", shock_type)
            return None

        # Don't stack same type
        if any(s["shock_type"] == shock_type for s in self._active_shocks):
            logger.debug("Shock type %s already active — skipping", shock_type)
            return None

        shock_duration = duration or catalog_entry["duration"]
        shock: dict[str, Any] = {
            "shock_id": f"{shock_type}_{self._next_id:03d}",
            "shock_type": shock_type,
            "scope": scope,
            "target_location": target_location,
            "turn_started": turn,
            "duration": shock_duration,
            "decay_profile": catalog_entry.get("decay_profile", "linear"),
            "intensity": 1.0,
            "effects": dict(catalog_entry.get("effects", {})),
            "source": source,
            "name": catalog_entry.get("name", shock_type),
            "description": catalog_entry.get("description", ""),
        }
        self._next_id += 1
        self._active_shocks.append(shock)

        logger.info(
            "Shock activated: %s (type=%s, scope=%s, duration=%d, source=%s)",
            shock["shock_id"], shock_type, scope, shock_duration, source,
        )
        return shock

    # ── Tick (decay + expiry) ─────────────────────────────────────────────

    def tick(self, turn: int) -> list[dict]:
        """Advance all active shocks by one turn: decay intensity, expire finished.

        Args:
            turn: Current game turn.

        Returns:
            List of shocks that expired this tick.
        """
        expired: list[dict] = []
        still_active: list[dict] = []

        for shock in self._active_shocks:
            elapsed = turn - shock["turn_started"]

            if elapsed >= shock["duration"]:
                shock["intensity"] = 0.0
                expired.append(shock)
                self._shock_history.append(shock)
                logger.info("Shock expired: %s after %d turns", shock["shock_id"], elapsed)
                continue

            # Decay intensity
            if shock["decay_profile"] == "linear":
                shock["intensity"] = max(0.0, 1.0 - (elapsed / shock["duration"]))
            else:
                # "sudden" — full intensity until expiry
                shock["intensity"] = 1.0

            still_active.append(shock)

        self._active_shocks = still_active

        # Bound history
        if len(self._shock_history) > self._max_history:
            self._shock_history = self._shock_history[-self._max_history:]

        return expired

    # ── Effect Aggregation ────────────────────────────────────────────────

    def get_reward_modifier(self) -> float:
        """Aggregate reward_scale from all active shocks.

        Returns:
            Multiplicative modifier for community reward (default 1.0).
        """
        modifier = 1.0
        for shock in self._active_shocks:
            scale = shock["effects"].get("reward_scale", 1.0)
            # Interpolate toward the shock's scale based on intensity
            modifier *= 1.0 + (scale - 1.0) * shock["intensity"]
        return modifier

    def get_adaptation_pressure(self) -> float:
        """Compute aggregate shock pressure for NPC adaptation updates.

        Returns:
            Pressure value in [0.0, 1.0]. 0.0 means no shocks active.
        """
        if not self._active_shocks:
            return 0.0
        # Average intensity across active shocks
        total_intensity = sum(s["intensity"] for s in self._active_shocks)
        return min(1.0, total_intensity / len(self._active_shocks))

    def get_stat_drain(self) -> float:
        """Aggregate per-turn resource drain from active shocks.

        Returns:
            Total resource drain value (applied to NPC stats).
        """
        drain = 0.0
        for shock in self._active_shocks:
            drain += shock["effects"].get("resource_drain", 0.0) * shock["intensity"]
        return drain

    def get_trust_modifier(self) -> float:
        """Aggregate per-turn trust/reputation modifier from active shocks.

        Returns:
            Trust modifier (negative = distrust, positive = boost).
        """
        modifier = 0.0
        for shock in self._active_shocks:
            modifier += shock["effects"].get("trust_modifier", 0.0) * shock["intensity"]
        return modifier

    # ── Queries ───────────────────────────────────────────────────────────

    def get_active_shocks(self) -> list[dict]:
        """Return list of currently active shocks with current state."""
        return [
            {
                "shock_id": s["shock_id"],
                "shock_type": s["shock_type"],
                "name": s.get("name", s["shock_type"]),
                "scope": s["scope"],
                "target_location": s.get("target_location"),
                "turn_started": s["turn_started"],
                "duration": s["duration"],
                "intensity": round(s["intensity"], 4),
                "source": s["source"],
            }
            for s in self._active_shocks
        ]

    def get_shock_timeline(self) -> list[dict]:
        """Return full timeline: active + recently expired shocks."""
        timeline = []
        for s in self._shock_history:
            timeline.append({
                "shock_id": s["shock_id"],
                "shock_type": s["shock_type"],
                "turn_started": s["turn_started"],
                "duration": s["duration"],
                "source": s["source"],
                "status": "expired",
            })
        for s in self._active_shocks:
            timeline.append({
                "shock_id": s["shock_id"],
                "shock_type": s["shock_type"],
                "turn_started": s["turn_started"],
                "duration": s["duration"],
                "intensity": round(s["intensity"], 4),
                "source": s["source"],
                "status": "active",
            })
        return sorted(timeline, key=lambda x: x["turn_started"])

    @property
    def has_active_shocks(self) -> bool:
        """Return True if any shocks are currently active."""
        return len(self._active_shocks) > 0

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize shock state for save files."""
        return {
            "active_shocks": [dict(s) for s in self._active_shocks],
            "shock_history": [dict(s) for s in self._shock_history],
            "next_id": self._next_id,
        }

    def from_dict(self, data: dict) -> None:
        """Restore shock state from save data."""
        self._active_shocks = [dict(s) for s in data.get("active_shocks", [])]
        self._shock_history = [dict(s) for s in data.get("shock_history", [])]
        self._next_id = data.get("next_id", 1)

    def __len__(self) -> int:
        return len(self._active_shocks)
