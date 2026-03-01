"""
interactions.py — NPC-NPC interactions and gossip propagation.

Handles autonomous NPC social actions, gossip cascades, and
target resolution for NPC decision-making.
"""

from __future__ import annotations

import random as _random

from backend.config import (
    GOSSIP_DECAY_FACTOR,
    GOSSIP_MAX_HOPS,
    GOSSIP_MIN_DELTA,
    GOSSIP_PROBABILITY,
    MASTER_SEED,
    REPUTATION_MAX,
    REPUTATION_MIN,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.npc.npc import NPC

# Module-level seeded RNG
_rng = _random.Random(MASTER_SEED)


# ── NPC–NPC interaction ──────────────────────────────────────────────────────

def resolve_npc_npc_interaction(
    source: NPC,
    target: NPC,
    action_id: str,
    game_context: dict,
) -> dict:
    """Resolve one NPC performing an action directed at another NPC.

    Args:
        source: The acting NPC.
        target: The target NPC.
        action_id: Universal action id.
        game_context: Shared game context dict.

    Returns:
        Dict describing the outcome:
        ``actor``, ``target``, ``action``, ``success``,
        ``effects``, ``narration``.
    """
    action_info = UNIVERSAL_ACTIONS.get(action_id, {})
    category = action_info.get("category", "utility")

    result: dict = {
        "actor": source.npc_uid,
        "target": target.npc_uid,
        "action": action_id,
        "success": True,
        "effects": {},
        "narration": "",
    }

    match category:
        case "social":
            result = _resolve_social(source, target, action_id, game_context, result)
        case "combat":
            result = _resolve_combat(source, target, action_id, game_context, result)
        case _:
            result["narration"] = (
                f"{source.name} performs {action_id} near {target.name}."
            )

    return result


def _resolve_social(
    source: NPC,
    target: NPC,
    action_id: str,
    game_context: dict,
    result: dict,
) -> dict:
    """Handle social actions between two NPCs."""
    relationship = source.npc_relationships.get(target.npc_uid, 0)

    match action_id:
        case "talk" | "greet":
            # Mild positive interaction
            delta = 1 if relationship >= 0 else 0
            _adjust_relationship(source, target, delta)
            result["effects"] = {"relationship_delta": delta}
            result["narration"] = f"{source.name} exchanges words with {target.name}."

        case "ask_info":
            if relationship >= 20:
                result["effects"] = {"info_shared": True, "relationship_delta": 1}
                _adjust_relationship(source, target, 1)
                result["narration"] = f"{target.name} shares information with {source.name}."
            else:
                result["effects"] = {"info_shared": False}
                result["narration"] = f"{target.name} is reluctant to share with {source.name}."

        case "trade":
            if relationship >= 0:
                income_delta = _rng.randint(0, 2)
                source.stats["income"] = min(10, source.stats["income"] + income_delta)
                target.stats["income"] = min(10, target.stats["income"] + income_delta)
                result["effects"] = {"income_delta": income_delta}
                result["narration"] = f"{source.name} and {target.name} conduct a trade."
            else:
                result["success"] = False
                result["narration"] = f"{target.name} refuses to trade with {source.name}."

        case "intimidate":
            if source.combat_stats["base_attack"] > target.combat_stats["base_defense"]:
                _adjust_relationship(source, target, -3)
                _adjust_relationship(target, source, -5)
                result["effects"] = {"relationship_delta": -3}
                result["narration"] = f"{source.name} intimidates {target.name}."
            else:
                result["success"] = False
                _adjust_relationship(target, source, -2)
                result["narration"] = f"{target.name} is unimpressed by {source.name}'s threats."

        case _:
            result["narration"] = f"{source.name} interacts with {target.name} ({action_id})."

    return result


def _resolve_combat(
    source: NPC,
    target: NPC,
    action_id: str,
    game_context: dict,
    result: dict,
) -> dict:
    """Handle combat-category actions between NPCs (simplified)."""
    if action_id == "attack":
        atk = source.combat_stats["base_attack"]
        dfn = target.combat_stats["base_defense"]
        damage = max(1, atk - dfn + _rng.randint(-3, 3))
        if target.is_defending:
            damage = max(1, damage // 2)
            target.is_defending = False
        target.modify_hp(-damage)
        _adjust_relationship(source, target, -10)
        _adjust_relationship(target, source, -10)
        result["effects"] = {"damage": damage, "target_hp": target.current_hp}
        result["narration"] = (
            f"{source.name} attacks {target.name} for {damage} damage."
        )
    elif action_id == "defend":
        source.is_defending = True
        result["narration"] = f"{source.name} takes a defensive stance."
    elif action_id == "flee":
        success = _rng.random() < 0.70
        result["success"] = success
        result["narration"] = (
            f"{source.name} flees from {target.name}."
            if success
            else f"{source.name} fails to flee from {target.name}."
        )
    return result


def _adjust_relationship(source: NPC, target: NPC, delta: int) -> None:
    """Adjust source's relationship score toward target, clamped."""
    current = source.npc_relationships.get(target.npc_uid, 0)
    source.npc_relationships[target.npc_uid] = max(
        REPUTATION_MIN, min(REPUTATION_MAX, current + delta)
    )


# ── Gossip propagation ───────────────────────────────────────────────────────

def propagate_gossip(
    source: NPC,
    target: NPC,
    event: dict,
    current_turn: int,
    game_context: dict | None = None,
) -> dict | None:
    """Attempt to propagate gossip from *source* to *target*.

    Rules:
      * 40 % base chance of gossip occurring
      * 50 % decay per hop
      * Max 3 hops
      * Min reputation delta of 2
      * 1 gossip event per NPC pair per turn (tracked via
        ``game_context["gossip_pairs"]``)

    Args:
        source: NPC who knows the event.
        target: NPC who may hear the gossip.
        event: The event dict being gossiped about.
        current_turn: Current game turn number.
        game_context: Shared context dict (must contain ``"gossip_pairs"`` set).

    Returns:
        Gossip result dict, or ``None`` if gossip did not occur.
    """
    # Cascade limit: 1 gossip per pair per turn
    if game_context is None:
        game_context = {}
    gossip_pairs: set = game_context.setdefault("gossip_pairs", set())
    pair_key = _pair_key(source.npc_uid, target.npc_uid, current_turn)
    if pair_key in gossip_pairs:
        return None
    gossip_pairs.add(pair_key)

    # Probability check
    if _rng.random() >= GOSSIP_PROBABILITY:
        return None

    # Determine gossip hop count and apply decay
    hop = event.get("gossip_hop", 0) + 1
    if hop > GOSSIP_MAX_HOPS:
        return None

    # Calculate reputation delta with decay
    raw_delta = event.get("reputation_delta", 0)
    decayed_delta = int(raw_delta * (GOSSIP_DECAY_FACTOR ** hop))
    if abs(decayed_delta) < GOSSIP_MIN_DELTA:
        return None

    # Apply the gossip
    about_uid = event.get("about", event.get("actor", ""))
    if about_uid and about_uid in target.npc_relationships:
        current_rep = target.npc_relationships.get(about_uid, 0)
        new_rep = max(REPUTATION_MIN, min(REPUTATION_MAX, current_rep + decayed_delta))
        target.npc_relationships[about_uid] = new_rep

    gossip_result = {
        "type": "gossip",
        "source": source.npc_uid,
        "target": target.npc_uid,
        "about": about_uid,
        "original_delta": raw_delta,
        "decayed_delta": decayed_delta,
        "hop": hop,
        "turn": current_turn,
    }

    logger.debug(
        "Gossip: %s → %s about %s (delta=%d, hop=%d)",
        source.npc_uid,
        target.npc_uid,
        about_uid,
        decayed_delta,
        hop,
    )
    return gossip_result


def _pair_key(uid_a: str, uid_b: str, turn: int) -> str:
    """Deterministic key for a gossip pair in a given turn."""
    a, b = sorted([uid_a, uid_b])
    return f"{a}:{b}:{turn}"


# ── Target resolution ─────────────────────────────────────────────────────────

def resolve_npc_target(
    npc: NPC,
    action_id: str,
    colocated_npcs: list[NPC],
    player_present: bool,
) -> str | None:
    """Determine the target for an NPC's chosen action.

    Target logic:
      * **Social actions**: prefer player if present, else pick a
        co-located NPC at random.
      * **Combat actions**: only target someone with hostile reputation;
        prefer hostile NPCs, then player if hostile.
      * **Other actions**: ``None`` (self / environment).

    Args:
        npc: The acting NPC.
        action_id: The chosen universal action id.
        colocated_npcs: Other NPCs at the same location (excludes *npc*).
        player_present: Whether the player is at the same location.

    Returns:
        Target UID string, ``"player"`` for the player, or ``None``.
    """
    action_info = UNIVERSAL_ACTIONS.get(action_id, {})
    category = action_info.get("category", "utility")

    if category == "social":
        if player_present:
            return "player"
        others = [n for n in colocated_npcs if n.npc_uid != npc.npc_uid]
        if others:
            return _rng.choice(others).npc_uid
        return None

    if category == "combat":
        # Only attack hostile targets
        hostile_npcs = [
            n
            for n in colocated_npcs
            if n.npc_uid != npc.npc_uid
            and npc.npc_relationships.get(n.npc_uid, 0) <= -50
        ]
        if hostile_npcs:
            return _rng.choice(hostile_npcs).npc_uid
        # Check if player is hostile
        if player_present and npc.npc_relationships.get("player", 0) <= -50:
            return "player"
        return None

    return None
