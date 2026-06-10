"""
knowledge.py — NPC knowledge base.

Tracks what each NPC has witnessed or heard via gossip, and provides
helpers to query relevant knowledge for dialogue and decision-making.
"""

from __future__ import annotations

from typing import TypedDict

from backend.config import (
    REPUTATION_MAX,
    REPUTATION_MIN,
    REPUTATION_THRESHOLDS,
    logger,
)
from backend.npc.npc import NPC


# ── Data types ────────────────────────────────────────────────────────────────

class NPCKnowledgeEntry(TypedDict):
    """Schema for a single knowledge entry stored in ``npc.known_events``."""

    event_id: str
    source: str          # "witnessed" or "gossip"
    from_npc: str | None # UID of gossip source, or None if witnessed
    turn_learned: int
    importance: int      # 1–5
    decayed_effects: dict


# ── Adding knowledge ──────────────────────────────────────────────────────────

def add_witnessed_event(npc: NPC, event: dict, current_turn: int) -> None:
    """Record that *npc* directly witnessed *event*.

    Args:
        npc: The witnessing NPC.
        event: An EventLogEntry-style dict (must have ``event_id``,
               ``importance``, ``effects``).
        current_turn: The turn on which the event occurred.
    """
    entry: NPCKnowledgeEntry = {
        "event_id": event.get("event_id", "unknown"),
        "source": "witnessed",
        "from_npc": None,
        "turn_learned": current_turn,
        "importance": event.get("importance", 1),
        "decayed_effects": dict(event.get("effects", {})),
    }
    npc.known_events.append(dict(entry))
    logger.debug(
        "NPC %s witnessed event %s (turn %d)",
        npc.npc_uid,
        entry["event_id"],
        current_turn,
    )


def add_gossip_event(
    npc: NPC,
    event: dict,
    from_npc_uid: str,
    current_turn: int,
    decay_factor: float,
) -> None:
    """Record that *npc* heard about *event* through gossip.

    The ``effects`` values are multiplied by *decay_factor* to reflect
    the distortion introduced by second-hand information.

    Args:
        npc: The NPC receiving the gossip.
        event: The original event dict.
        from_npc_uid: The UID of the NPC who relayed the gossip.
        current_turn: Current game turn.
        decay_factor: Multiplier applied to numeric effect values.
    """
    # Decay numeric effects
    raw_effects = event.get("effects", {})
    decayed: dict = {}
    for k, v in raw_effects.items():
        if isinstance(v, (int, float)):
            decayed[k] = int(v * decay_factor) if isinstance(v, int) else v * decay_factor
        else:
            decayed[k] = v

    entry: NPCKnowledgeEntry = {
        "event_id": event.get("event_id", "unknown"),
        "source": "gossip",
        "from_npc": from_npc_uid,
        "turn_learned": current_turn,
        "importance": max(1, event.get("importance", 1) - 1),  # gossip loses 1 importance
        "decayed_effects": decayed,
    }
    npc.known_events.append(dict(entry))
    logger.debug(
        "NPC %s learned gossip about event %s from %s (turn %d)",
        npc.npc_uid,
        entry["event_id"],
        from_npc_uid,
        current_turn,
    )


# ── Querying knowledge ───────────────────────────────────────────────────────

def get_relevant_knowledge(
    npc: NPC,
    topic: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Retrieve the most relevant knowledge entries for *npc*.

    Entries are sorted by importance (descending), then recency (descending).
    If *topic* is provided, entries whose ``event_id`` contains the topic
    substring are prioritised.

    Args:
        npc: The NPC to query.
        topic: Optional topic string to filter / prioritise.
        limit: Maximum number of entries to return.

    Returns:
        List of knowledge entry dicts (up to *limit*).
    """
    events = list(npc.known_events)
    if not events:
        return []

    def _sort_key(e: dict) -> tuple:
        """(topic_match, importance, recency)."""
        topic_match = 1 if topic and topic.lower() in e.get("event_id", "").lower() else 0
        return (topic_match, e.get("importance", 0), e.get("turn_learned", 0))

    events.sort(key=_sort_key, reverse=True)
    return events[:limit]


def get_player_opinion_summary(npc: NPC) -> str:
    """Build a short textual summary of why the NPC feels this way about the player.

    Draws on the NPC's reputation toward the player and recent
    player-related knowledge entries.

    Args:
        npc: The NPC whose opinion we're summarising.

    Returns:
        A plain-English summary string.
    """
    player_rep = npc.npc_relationships.get("player", 0)

    # Determine label from thresholds
    label = "neutral"
    for tier, (low, high) in REPUTATION_THRESHOLDS.items():
        if low <= player_rep <= high:
            label = tier
            break

    # Gather player-related events
    player_events = [
        e for e in npc.known_events
        if "player" in e.get("event_id", "").lower()
        or e.get("decayed_effects", {}).get("actor") == "player"
    ]
    recent = sorted(player_events, key=lambda e: e.get("turn_learned", 0), reverse=True)[:3]

    parts: list[str] = [
        f"{npc.name} considers the player {label} (reputation {player_rep})."
    ]

    if recent:
        reasons = []
        for ev in recent:
            source_label = "witnessed" if ev.get("source") == "witnessed" else "heard about"
            reasons.append(
                f"{source_label} event '{ev.get('event_id', '?')}' "
                f"(importance {ev.get('importance', '?')})"
            )
        parts.append("Reasons: " + "; ".join(reasons) + ".")
    else:
        parts.append("No specific interactions with the player recorded.")

    return " ".join(parts)
