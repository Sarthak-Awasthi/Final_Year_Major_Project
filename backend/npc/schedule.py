"""
schedule.py — Fallback schedule system for NPCs.

Provides deterministic behaviour when the Q-learner is in cold-start
mode or when a schedule-aligned action is desired.
"""

from __future__ import annotations

import random as _random

from backend.config import (
    LOCATION_ADJACENCY,
    LOCATION_IDS,
    MASTER_SEED,
    logger,
)
from backend.npc.npc import NPC

# Module-level seeded RNG
_rng = _random.Random(MASTER_SEED)


def get_scheduled_action(npc: NPC, time_period: str) -> dict:
    """Return the scheduled action and location for *time_period*.

    Looks up *npc.fallback_schedule* for a matching ``"time"`` entry.
    If no match is found, defaults to ``{"action": "wait", "location": <current>}``.

    Args:
        npc: The NPC whose schedule to consult.
        time_period: One of ``"morning"``, ``"midday"``, ``"afternoon"``,
                     ``"evening"``, ``"night"``.

    Returns:
        Dict with keys ``"action"`` (action id) and ``"location"`` (location id).
    """
    for entry in npc.fallback_schedule:
        if entry.get("time") == time_period:
            return {
                "action": entry["action"],
                "location": entry["location"],
            }

    # No schedule entry for this period — default to waiting in place
    logger.debug(
        "No schedule entry for %s at %s — defaulting to wait",
        npc.npc_uid,
        time_period,
    )
    return {"action": "wait", "location": npc.location}


def get_movement_destination(
    npc: NPC,
    time_period: str,
    game_context: dict,
) -> str:
    """Determine where the NPC should move, respecting adjacency.

    Resolution order:
      1. If the fallback schedule specifies a location for *time_period*
         **and** that location is adjacent → use it.
      2. If *game_context* contains a ``"goal_location"`` for this NPC
         **and** it's adjacent → use it.
      3. Weighted random from ``npc.movement_weights`` among **adjacent**
         locations only.

    Args:
        npc: The NPC deciding where to move.
        time_period: Current time period string.
        game_context: Shared game context dict (may contain
                      ``"goal_location"``).

    Returns:
        A valid adjacent location id, or the NPC's current location
        if no adjacent location is available.
    """
    adjacent = LOCATION_ADJACENCY.get(npc.location, [])
    if not adjacent:
        logger.warning(
            "NPC %s at unknown location '%s' — staying", npc.npc_uid, npc.location
        )
        return npc.location

    # 1. Schedule-based destination
    scheduled = get_scheduled_action(npc, time_period)
    sched_loc = scheduled["location"]
    if sched_loc != npc.location and sched_loc in adjacent:
        return sched_loc
    # If scheduled location is not adjacent but is reachable via
    # multi-hop, move toward it (pick the adjacent node closest to it).
    if sched_loc != npc.location and sched_loc not in adjacent:
        step = _step_toward(npc.location, sched_loc, adjacent)
        if step is not None:
            return step

    # 2. Goal location from context
    goal = game_context.get("goal_location")
    if goal and goal != npc.location and goal in adjacent:
        return goal
    if goal and goal != npc.location and goal not in adjacent:
        step = _step_toward(npc.location, goal, adjacent)
        if step is not None:
            return step

    # 3. Weighted random among adjacent
    weights = [npc.movement_weights.get(loc, 0.1) for loc in adjacent]
    total = sum(weights)
    if total <= 0:
        return _rng.choice(adjacent)
    probs = [w / total for w in weights]
    return _rng.choices(adjacent, weights=probs, k=1)[0]


def is_schedule_time(npc: NPC, time_period: str, action_id: str) -> bool:
    """Check whether *action_id* matches the schedule for *time_period*.

    Args:
        npc: The NPC to check.
        time_period: Current time period.
        action_id: Universal action id to compare.

    Returns:
        ``True`` if the schedule entry for this time has the same action.
    """
    for entry in npc.fallback_schedule:
        if entry.get("time") == time_period:
            return entry.get("action") == action_id
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step_toward(
    current: str,
    destination: str,
    adjacent: list[str],
) -> str | None:
    """Return the adjacent location that is closest to *destination*.

    Uses a simple BFS to check if *destination* is reachable via each
    adjacent node and picks the one with the shortest path.

    Returns:
        An adjacent location id, or ``None`` if unreachable.
    """
    # For such a small graph (5 nodes) a brute BFS from each neighbour
    # is perfectly fine.
    best: str | None = None
    best_dist = float("inf")

    for neighbor in adjacent:
        dist = _bfs_distance(neighbor, destination)
        if dist is not None and dist < best_dist:
            best_dist = dist
            best = neighbor

    return best


def _bfs_distance(start: str, end: str) -> int | None:
    """Return hop count from *start* to *end*, or ``None`` if unreachable."""
    if start == end:
        return 0
    visited: set[str] = {start}
    queue: list[tuple[str, int]] = [(start, 0)]
    while queue:
        node, dist = queue.pop(0)
        for neighbor in LOCATION_ADJACENCY.get(node, []):
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None
