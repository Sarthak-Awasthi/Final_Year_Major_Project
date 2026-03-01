"""
events.py — Random events system, event catalog, trigger conditions.
"""

from __future__ import annotations

import json
import random
from typing import Any

from backend.config import DATA_DIR, RANDOM_EVENT_FREQUENCY_MULTIPLIER, logger


class RandomEventSystem:
    """Manages random world events that inject unpredictability."""

    def __init__(self) -> None:
        self.event_catalog: list[dict] = []
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Load event definitions from JSON."""
        path = DATA_DIR / "config" / "event_catalog.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.event_catalog = json.load(f)
        except FileNotFoundError:
            logger.warning("Event catalog not found, using empty catalog")
            self.event_catalog = []

    def check_events(
        self,
        turn: int,
        time_of_day: str,
        active_event_ids: list[str],
        player_reputation: dict[str, int],
        global_reputation: int,
        npc_locations: dict[str, str],
        frequency_multiplier: float = 1.0,
    ) -> list[dict]:
        """
        Check if any random events trigger this turn.

        Returns a list of newly triggered events.
        """
        triggered = []

        for event_def in self.event_catalog:
            event_id = event_def["id"]

            # Don't stack same event
            if event_id in active_event_ids:
                continue

            # Check trigger conditions
            if not self._check_trigger(event_def, turn, time_of_day, player_reputation, global_reputation, npc_locations):
                continue

            # Roll against probability
            base_prob = event_def.get("probability", 0.0)
            adjusted_prob = base_prob * frequency_multiplier * RANDOM_EVENT_FREQUENCY_MULTIPLIER

            if random.random() < adjusted_prob:
                duration_min = event_def.get("duration_min", 1)
                duration_max = event_def.get("duration_max", 1)
                duration = random.randint(duration_min, duration_max)

                event = {
                    "id": event_id,
                    "name": event_def.get("name", event_id),
                    "description": event_def.get("description", ""),
                    "turn_started": turn,
                    "duration": duration,
                    "effects": event_def.get("effects", {}),
                }
                triggered.append(event)

        return triggered

    def _check_trigger(
        self,
        event_def: dict,
        turn: int,
        time_of_day: str,
        player_reputation: dict[str, int],
        global_reputation: int,
        npc_locations: dict[str, str],
    ) -> bool:
        """Check if an event's trigger conditions are met."""
        trigger = event_def.get("trigger", {})

        # Minimum turn
        if turn < trigger.get("min_turn", 0):
            return False

        # Time of day requirement
        tod_req = trigger.get("time_of_day")
        if tod_req and time_of_day not in tod_req:
            return False

        # Player reputation threshold
        rep_below = trigger.get("player_reputation_below")
        if rep_below is not None:
            if not any(v < rep_below for v in player_reputation.values()):
                return False

        # Global reputation threshold
        rep_above = trigger.get("global_rep_above")
        if rep_above is not None:
            if global_reputation <= rep_above:
                return False

        # NPC at outdoor location
        if trigger.get("npc_at_outdoor"):
            from backend.config import OUTDOOR_LOCATIONS
            if not any(loc in OUTDOOR_LOCATIONS for loc in npc_locations.values()):
                return False

        return True

    def get_active_effects(self, active_events: list[dict]) -> dict[str, Any]:
        """Aggregate effects from all active events."""
        combined: dict[str, Any] = {
            "outdoor_search_ap_increase": 0,
            "sneak_bonus": 0.0,
            "steal_detection_bonus": 0.0,
            "trade_price_increase": 0.0,
            "social_reputation_bonus": 1.0,
            "npc_indoor_preference": False,
            "tavern_food_unavailable": False,
            "look_effectiveness_reduced": False,
            "guard_patrol_increase": False,
        }

        for event in active_events:
            effects = event.get("effects", {})
            combined["outdoor_search_ap_increase"] += effects.get("outdoor_search_ap_increase", 0)
            combined["sneak_bonus"] += effects.get("sneak_bonus", 0.0)
            combined["steal_detection_bonus"] += effects.get("steal_detection_bonus", 0.0)
            combined["trade_price_increase"] += effects.get("trade_price_increase", 0.0)
            if effects.get("social_reputation_bonus"):
                combined["social_reputation_bonus"] *= effects["social_reputation_bonus"]
            if effects.get("npc_indoor_preference"):
                combined["npc_indoor_preference"] = True
            if effects.get("tavern_food_unavailable"):
                combined["tavern_food_unavailable"] = True
            if effects.get("look_effectiveness_reduced"):
                combined["look_effectiveness_reduced"] = True
            if effects.get("guard_patrol_increase"):
                combined["guard_patrol_increase"] = True

        return combined
