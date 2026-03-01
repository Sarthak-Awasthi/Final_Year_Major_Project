"""
event_log.py — World memory / event log, witness detection, pruning.
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.config import MAX_EVENT_LOG_SIZE, logger


def create_event_id() -> str:
    """Generate a unique event ID."""
    return f"evt_{uuid.uuid4().hex[:8]}"


class EventLog:
    """Central event log — the world's memory."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._summaries: list[str] = []

    def add_entry(
        self,
        turn: int,
        time_of_day: str,
        event_type: str,
        actor: str,
        action: str,
        target: str | None,
        location: str,
        outcome: str,
        effects: dict[str, Any],
        witnesses: list[str],
        narration: str,
        importance: int,
        **extra: Any,
    ) -> dict[str, Any]:
        """Add a new event log entry."""
        entry: dict[str, Any] = {
            "event_id": create_event_id(),
            "turn": turn,
            "time_of_day": time_of_day,
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "target": target,
            "location": location,
            "outcome": outcome,
            "effects": effects,
            "witnesses": witnesses,
            "narration": narration,
            "importance": importance,
        }
        entry.update(extra)
        self.entries.append(entry)
        self._prune_if_needed()
        return entry

    def _prune_if_needed(self) -> None:
        """Prune event log if it exceeds max size."""
        if len(self.entries) > MAX_EVENT_LOG_SIZE:
            # Keep importance >= 2 events, prune trivial ones
            old = self.entries[: -MAX_EVENT_LOG_SIZE]
            important_old = [e for e in old if e.get("importance", 1) >= 2]
            trivial_count = len(old) - len(important_old)

            if trivial_count > 0:
                summary = (
                    f"Turns {old[0].get('turn', '?')}-{old[-1].get('turn', '?')}: "
                    f"{trivial_count} routine events pruned."
                )
                self._summaries.append(summary)

            # Keep important old events + recent events
            self.entries = important_old + self.entries[-MAX_EVENT_LOG_SIZE:]

    def get_recent(self, count: int = 10) -> list[dict]:
        """Get the most recent N entries."""
        return self.entries[-count:]

    def get_by_importance(self, min_importance: int = 3) -> list[dict]:
        """Get all entries at or above a given importance level."""
        return [e for e in self.entries if e.get("importance", 1) >= min_importance]

    def get_by_turn(self, turn: int) -> list[dict]:
        """Get all entries for a specific turn."""
        return [e for e in self.entries if e.get("turn") == turn]

    def get_by_actor(self, actor: str) -> list[dict]:
        """Get all entries by a specific actor."""
        return [e for e in self.entries if e.get("actor") == actor]

    def get_by_location(self, location: str) -> list[dict]:
        """Get all entries at a specific location."""
        return [e for e in self.entries if e.get("location") == location]

    def get_player_actions(self) -> list[dict]:
        """Get all player actions."""
        return [e for e in self.entries if e.get("actor") == "player"]

    def count_action_at_stage(self, action_id: str, stage_id: int) -> int:
        """Count how many times an action generated a dynamic CP at a specific stage."""
        return sum(
            1
            for e in self.entries
            if e.get("action") == action_id
            and e.get("event_type") == "quest_progress"
            and e.get("effects", {}).get("stage_id") == stage_id
            and e.get("effects", {}).get("is_dynamic", False)
        )

    def to_list(self) -> list[dict]:
        """Serialize the event log."""
        return self.entries.copy()

    def from_list(self, data: list[dict]) -> None:
        """Restore event log from list."""
        self.entries = data

    def __len__(self) -> int:
        return len(self.entries)


def detect_witnesses(
    event_location: str,
    actor_uid: str,
    npc_locations: dict[str, str],
) -> list[str]:
    """
    Find all NPCs who witness an event at a given location.

    Args:
        event_location: Where the event occurred
        actor_uid: UID of the acting entity (excluded from witnesses)
        npc_locations: Dict mapping npc_uid -> location_id
    Returns:
        List of witness NPC UIDs
    """
    witnesses = []
    for npc_uid, loc in npc_locations.items():
        if loc == event_location and npc_uid != actor_uid:
            witnesses.append(npc_uid)
    return witnesses


def compute_importance(
    event_type: str,
    action: str,
    outcome: str,
    effects: dict,
) -> int:
    """
    Compute the importance score (1-5) for an event.

    5 = Quest-critical
    4 = Major
    3 = Significant
    2 = Minor
    1 = Trivial
    """
    # Quest-critical events
    if event_type == "quest_progress":
        rep_change = abs(effects.get("reputation_change", 0))
        if effects.get("stage_transition") or effects.get("quest_complete") or effects.get("quest_fail"):
            return 5
        if rep_change >= 10:
            return 4
        return 3

    # Combat events are always major
    if event_type == "combat":
        if effects.get("incapacitated"):
            return 4
        return 4

    # Random events
    if event_type == "random_event":
        return 3

    # Player actions
    if event_type == "player_action":
        rep_change = 0
        rep_effects = effects.get("reputation", {})
        if isinstance(rep_effects, dict):
            rep_change = max(abs(v) for v in rep_effects.values()) if rep_effects else 0
        elif isinstance(rep_effects, (int, float)):
            rep_change = abs(rep_effects)

        if rep_change >= 10:
            return 4
        if rep_change >= 5 or action in ("attack", "steal", "intimidate"):
            return 3
        if action in ("move_to", "trade", "greet", "talk"):
            return 2
        if action in ("look", "wait", "rest", "status", "drop_item"):
            return 1
        return 2

    # NPC actions
    if event_type == "npc_action":
        if action in ("attack", "steal"):
            return 3
        if action in ("talk", "trade", "give_item"):
            return 2
        return 1

    # Dialogue
    if event_type == "dialogue":
        return 2

    return 1
