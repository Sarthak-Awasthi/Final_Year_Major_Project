"""
world.py — World state, locations, time system.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from backend.config import (
    DATA_DIR,
    INDOOR_LOCATIONS,
    LOCATION_ADJACENCY,
    LOCATION_IDS,
    OUTDOOR_LOCATIONS,
    SOCIAL_LOCATIONS,
    TIME_PERIODS,
    TURNS_PER_PERIOD,
    logger,
)


@dataclass
class Location:
    """A single world location."""

    id: str
    name: str
    type: str  # "indoor" | "outdoor"
    description: str
    adjacent: list[str]
    objects: list[str]
    environment: str
    default_npcs: list[str]
    items_on_ground: list[dict] = field(default_factory=list)
    search_count: int = 0


class World:
    """Manages the game world: locations, time, and spatial relationships."""

    def __init__(self) -> None:
        self.locations: dict[str, Location] = {}
        self.turn: int = 0
        self.time_of_day: str = "morning"
        self.active_events: list[dict] = []
        self._load_locations()

    def _load_locations(self) -> None:
        """Load locations from JSON data file."""
        loc_path = DATA_DIR / "world" / "locations.json"
        with open(loc_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for loc_id, loc_data in data["locations"].items():
            self.locations[loc_id] = Location(
                id=loc_data["id"],
                name=loc_data["name"],
                type=loc_data["type"],
                description=loc_data["description"],
                adjacent=loc_data["adjacent"],
                objects=loc_data["objects"],
                environment=loc_data["environment"],
                default_npcs=loc_data.get("default_npcs", []),
            )

    def advance_turn(self) -> str:
        """Advance the game clock by one turn. Returns the new time period."""
        self.turn += 1
        period_index = (self.turn // TURNS_PER_PERIOD) % len(TIME_PERIODS)
        self.time_of_day = TIME_PERIODS[period_index]
        return self.time_of_day

    def get_time_period_index(self) -> int:
        """Return the index of the current time period."""
        return TIME_PERIODS.index(self.time_of_day)

    def is_adjacent(self, loc_a: str, loc_b: str) -> bool:
        """Check if two locations are adjacent."""
        return loc_b in LOCATION_ADJACENCY.get(loc_a, [])

    def get_adjacent(self, loc_id: str) -> list[str]:
        """Get adjacent locations for a given location."""
        return LOCATION_ADJACENCY.get(loc_id, [])

    def is_indoor(self, loc_id: str) -> bool:
        """Check if a location is indoor."""
        return loc_id in INDOOR_LOCATIONS

    def is_outdoor(self, loc_id: str) -> bool:
        """Check if a location is outdoor."""
        return loc_id in OUTDOOR_LOCATIONS

    def is_social(self, loc_id: str) -> bool:
        """Check if a location is a social location."""
        return loc_id in SOCIAL_LOCATIONS

    def get_location(self, loc_id: str) -> Location | None:
        """Get a location by ID."""
        return self.locations.get(loc_id)

    def get_time_bonus(self) -> int:
        """Get stealth time bonus based on current time."""
        if self.time_of_day == "night":
            return 3
        elif self.time_of_day == "evening":
            return 1
        return 0

    def add_event(self, event: dict) -> None:
        """Add an active random event."""
        self.active_events.append(event)

    def expire_events(self) -> list[dict]:
        """Remove expired events. Returns list of expired events."""
        expired = []
        remaining = []
        for e in self.active_events:
            if self.turn >= e.get("turn_started", 0) + e.get("duration", 0):
                expired.append(e)
            else:
                remaining.append(e)
        self.active_events = remaining
        return expired

    def has_active_event(self, event_id: str) -> bool:
        """Check if a specific event is currently active."""
        return any(e.get("id") == event_id for e in self.active_events)

    def to_dict(self) -> dict:
        """Serialize world state."""
        return {
            "turn": self.turn,
            "time_of_day": self.time_of_day,
            "active_events": self.active_events,
            "location_items": {
                loc_id: loc.items_on_ground
                for loc_id, loc in self.locations.items()
                if loc.items_on_ground
            },
            "location_search_counts": {
                loc_id: loc.search_count
                for loc_id, loc in self.locations.items()
                if loc.search_count > 0
            },
        }

    def from_dict(self, data: dict) -> None:
        """Restore world state from dict."""
        self.turn = data.get("turn", 0)
        self.time_of_day = data.get("time_of_day", "morning")
        self.active_events = data.get("active_events", [])
        for loc_id, items in data.get("location_items", {}).items():
            if loc_id in self.locations:
                self.locations[loc_id].items_on_ground = items
        for loc_id, count in data.get("location_search_counts", {}).items():
            if loc_id in self.locations:
                self.locations[loc_id].search_count = count
