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
class PointOfInterest:
    """A discoverable point of interest within a location."""

    poi_id: str
    name: str
    description: str
    discovery_hint: str
    discovered: bool = False
    discovered_by_default: bool = False
    discover_on_quest_stage: int | None = None
    discover_on_dialogue: dict | None = None  # {"npc_uid": str, "keywords": list[str]}
    searchable: bool = True
    search_bonus: float = 0.0
    items_hidden: list[str] = field(default_factory=list)
    examine_text: str | None = None
    reveals_pois: list[dict] | None = None  # [{"location": str, "poi_id": str}]

    def to_dict(self) -> dict:
        """Serialize POI state (only mutable fields needed for save)."""
        return {
            "poi_id": self.poi_id,
            "discovered": self.discovered,
            "items_hidden": self.items_hidden,
        }


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
    points_of_interest: list[PointOfInterest] = field(default_factory=list)


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
            # Build POI list
            pois: list[PointOfInterest] = []
            for poi_data in loc_data.get("points_of_interest", []):
                poi = PointOfInterest(
                    poi_id=poi_data["poi_id"],
                    name=poi_data["name"],
                    description=poi_data["description"],
                    discovery_hint=poi_data.get("discovery_hint", ""),
                    discovered=poi_data.get("discovered_by_default", False),
                    discovered_by_default=poi_data.get("discovered_by_default", False),
                    discover_on_quest_stage=poi_data.get("discover_on_quest_stage"),
                    discover_on_dialogue=poi_data.get("discover_on_dialogue"),
                    searchable=poi_data.get("searchable", True),
                    search_bonus=poi_data.get("search_bonus", 0.0),
                    items_hidden=list(poi_data.get("items_hidden", [])),
                    examine_text=poi_data.get("examine_text"),
                    reveals_pois=poi_data.get("reveals_pois"),
                )
                pois.append(poi)

            self.locations[loc_id] = Location(
                id=loc_data["id"],
                name=loc_data["name"],
                type=loc_data["type"],
                description=loc_data["description"],
                adjacent=loc_data["adjacent"],
                objects=loc_data["objects"],
                environment=loc_data["environment"],
                default_npcs=loc_data.get("default_npcs", []),
                points_of_interest=pois,
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

    def get_poi(self, loc_id: str, poi_id: str) -> PointOfInterest | None:
        """Get a specific POI within a location."""
        loc = self.locations.get(loc_id)
        if not loc:
            return None
        for poi in loc.points_of_interest:
            if poi.poi_id == poi_id:
                return poi
        return None

    def discover_poi(self, loc_id: str, poi_id: str) -> PointOfInterest | None:
        """Mark a POI as discovered. Returns the POI if newly discovered, else None."""
        poi = self.get_poi(loc_id, poi_id)
        if poi and not poi.discovered:
            poi.discovered = True
            logger.info("POI discovered: %s at %s", poi_id, loc_id)
            # Check if this POI reveals other POIs
            if poi.reveals_pois:
                for reveal in poi.reveals_pois:
                    self.discover_poi(reveal["location"], reveal["poi_id"])
            return poi
        return None

    def get_discovered_pois(self, loc_id: str) -> list[PointOfInterest]:
        """Return all discovered POIs at a location."""
        loc = self.locations.get(loc_id)
        if not loc:
            return []
        return [p for p in loc.points_of_interest if p.discovered]

    def check_quest_stage_discoveries(self, quest_stage: int) -> list[PointOfInterest]:
        """Discover all POIs triggered by reaching a quest stage.

        Returns list of newly discovered POIs.
        """
        newly_discovered: list[PointOfInterest] = []
        for loc in self.locations.values():
            for poi in loc.points_of_interest:
                if (
                    not poi.discovered
                    and poi.discover_on_quest_stage is not None
                    and quest_stage >= poi.discover_on_quest_stage
                ):
                    poi.discovered = True
                    logger.info(
                        "POI auto-discovered by quest stage %d: %s at %s",
                        quest_stage, poi.poi_id, loc.id,
                    )
                    newly_discovered.append(poi)
                    if poi.reveals_pois:
                        for reveal in poi.reveals_pois:
                            child = self.discover_poi(reveal["location"], reveal["poi_id"])
                            if child:
                                newly_discovered.append(child)
        return newly_discovered

    def check_dialogue_discoveries(
        self, npc_uid: str, dialogue_text: str
    ) -> list[PointOfInterest]:
        """Discover POIs triggered by NPC dialogue containing keywords.

        Returns list of newly discovered POIs.
        """
        newly_discovered: list[PointOfInterest] = []
        text_lower = dialogue_text.lower()
        for loc in self.locations.values():
            for poi in loc.points_of_interest:
                if poi.discovered:
                    continue
                trigger = poi.discover_on_dialogue
                if not trigger:
                    continue
                if trigger.get("npc_uid") != npc_uid:
                    continue
                keywords = trigger.get("keywords", [])
                if any(kw.lower() in text_lower for kw in keywords):
                    poi.discovered = True
                    logger.info(
                        "POI discovered via dialogue with %s: %s at %s",
                        npc_uid, poi.poi_id, loc.id,
                    )
                    newly_discovered.append(poi)
                    if poi.reveals_pois:
                        for reveal in poi.reveals_pois:
                            child = self.discover_poi(reveal["location"], reveal["poi_id"])
                            if child:
                                newly_discovered.append(child)
        return newly_discovered

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
            "poi_state": {
                loc_id: [poi.to_dict() for poi in loc.points_of_interest]
                for loc_id, loc in self.locations.items()
                if loc.points_of_interest
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
        # Restore POI discovery state
        for loc_id, poi_states in data.get("poi_state", {}).items():
            loc = self.locations.get(loc_id)
            if not loc:
                continue
            poi_map = {ps["poi_id"]: ps for ps in poi_states}
            for poi in loc.points_of_interest:
                saved = poi_map.get(poi.poi_id)
                if saved:
                    poi.discovered = saved.get("discovered", poi.discovered)
                    poi.items_hidden = saved.get("items_hidden", poi.items_hidden)
