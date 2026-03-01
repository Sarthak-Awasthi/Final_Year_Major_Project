"""
actions.py — Player action resolution module.

Handles the **Action Resolution Pipeline** for all 27 universal player actions.
Every action goes through: Precondition Check → Context Evaluation → Outcome Resolution.

Design rules:
- All 27 actions are always available — never restrict.
- Preconditions determine outcomes, not availability.
- Hard-fail preconditions (no target present, etc.) narrate why and cost 0 AP.
- AP cost comes from config.py UNIVERSAL_ACTIONS dict.
- At 0 AP: only 0-AP actions + talk/greet at 0 AP (anti-soft-lock).
- In combat at 0 AP: defend/flee cost 0 AP but flee success drops to 40%.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from backend.config import (
    INDOOR_LOCATIONS,
    LOCATION_IDS,
    SOCIAL_LOCATIONS,
    SOCIAL_MODIFIERS,
    STAMINA_REGEN_PER_TURN,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.engine.combat import (
    compute_skill_probability,
    resolve_attack,
    resolve_flee,
)
from backend.engine.narration import (
    add_context_modifiers,
    get_template_narration,
    passive_perception_check,
)


# ─── ActionResult Dataclass ──────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Result of resolving a player action through the pipeline.

    Attributes:
        success: Whether the action achieved its intended effect.
        outcome: One of ``"success"``, ``"partial"``, ``"fail"``, ``"blocked"``.
        narration: Human-readable narrative text for the player.
        effects: Dict of changes applied (e.g. health, stamina, reputation, items).
        ap_cost: Actual AP spent (0 for hard-fails / anti-soft-lock overrides).
        importance: Event-log importance score (1–5).
        event_type: Categorization string for the event log entry.
    """

    success: bool
    outcome: str  # "success" | "partial" | "fail" | "blocked"
    narration: str
    effects: dict[str, Any] = field(default_factory=dict)
    ap_cost: int = 0
    importance: int = 1
    event_type: str = "player_action"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_base_ap(action_id: str) -> int:
    """Return the base AP cost for an action from the universal catalog."""
    return UNIVERSAL_ACTIONS.get(action_id, {}).get("base_ap", 0)


def _blocked(action_id: str, reason: str, ctx: dict[str, Any] | None = None) -> ActionResult:
    """Produce a hard-fail / blocked result with 0 AP cost."""
    narration = get_template_narration(action_id, "blocked", ctx)
    if not narration or narration == "Something happens.":
        narration = reason
    return ActionResult(
        success=False,
        outcome="blocked",
        narration=narration,
        effects={},
        ap_cost=0,
        importance=1,
        event_type="player_action",
    )


def _find_target_npc(
    target_npc_uid: str | None,
    player_location: str,
    npc_registry: dict[str, Any],
) -> dict | None:
    """Resolve a target NPC UID to its data dict, only if at the same location.

    Returns the NPC dict (must have at minimum ``uid``, ``name``, ``location``)
    or ``None`` when not present or not found.
    """
    if not target_npc_uid:
        return None
    npc = npc_registry.get(target_npc_uid)
    if npc is None:
        return None
    npc_loc = npc.get("location") or npc.get("current_location", "")
    if npc_loc != player_location:
        return None
    return npc


def _npcs_at_location(
    location: str,
    npc_registry: dict[str, Any],
) -> list[dict]:
    """Return list of NPC dicts at a given location."""
    result: list[dict] = []
    for npc in npc_registry.values():
        npc_loc = npc.get("location") or npc.get("current_location", "")
        if npc_loc == location:
            result.append(npc)
    return result


def _npc_locations(npc_registry: dict[str, Any]) -> dict[str, str]:
    """Build ``{npc_uid: location_id}`` mapping from the registry."""
    return {
        uid: npc.get("location") or npc.get("current_location", "")
        for uid, npc in npc_registry.items()
    }


def _check_stamina(
    action_id: str,
    player: Any,
    in_combat: bool,
) -> int | None:
    """Return the effective AP cost or ``None`` if the player cannot afford it.

    Implements:
    - 0-AP actions always allowed.
    - ``talk`` / ``greet`` cost 0 when stamina is 0 (anti-soft-lock).
    - In combat, ``defend`` / ``flee`` cost 0 when stamina is 0.
    """
    base_ap = _get_base_ap(action_id)

    # 0-cost actions are always permitted
    if base_ap == 0:
        return 0

    # Anti-soft-lock: talk/greet free at 0 stamina
    if action_id in ("talk", "greet") and player.stamina == 0:
        return 0

    # Combat anti-soft-lock: defend/flee free at 0 stamina
    if action_id in ("defend", "flee") and in_combat and player.stamina == 0:
        return 0

    # Normal cost check
    if player.can_afford_ap(base_ap):
        return base_ap

    return None  # cannot afford


# ─── Main Entry Point ────────────────────────────────────────────────────────

def resolve_action(
    action_id: str,
    player: Any,
    world: Any,
    npc_registry: dict[str, Any],
    event_log: Any,
    parsed_input: dict[str, Any],
) -> ActionResult:
    """Resolve a player action through the full pipeline.

    Pipeline stages:
        1. **Precondition Check** — validate AP budget, targets, location.
        2. **Context Evaluation** — gather modifiers (rep, time, weather, etc.).
        3. **Outcome Resolution** — probabilistic resolution & effects.

    Args:
        action_id: One of the 27 universal action IDs.
        player: The ``Player`` instance.
        world: The ``World`` instance.
        npc_registry: ``{npc_uid: npc_dict}`` — all NPC data.
        event_log: The ``EventLog`` instance.
        parsed_input: The ``ParsedInput`` dict from input parsing.

    Returns:
        An ``ActionResult`` with narration, effects, and metadata.
    """
    if action_id not in UNIVERSAL_ACTIONS:
        logger.warning("Unknown action_id '%s' — defaulting to 'wait'.", action_id)
        action_id = "wait"

    # ── Stamina gate ─────────────────────────────────────────────────────
    effective_ap = _check_stamina(action_id, player, player.in_combat)
    if effective_ap is None:
        return ActionResult(
            success=False,
            outcome="blocked",
            narration="You're too exhausted to do that right now.",
            effects={},
            ap_cost=0,
            importance=1,
            event_type="player_action",
        )

    # ── Clear defending flag at start of new action (except defend) ──────
    if action_id != "defend":
        player.is_defending = False

    # ── Dispatch to handler ──────────────────────────────────────────────
    handler = _ACTION_HANDLERS.get(action_id, _handle_wait)
    result: ActionResult = handler(
        player=player,
        world=world,
        npc_registry=npc_registry,
        event_log=event_log,
        parsed_input=parsed_input,
        effective_ap=effective_ap,
    )

    # ── Deduct AP ────────────────────────────────────────────────────────
    if result.ap_cost > 0:
        player.modify_stamina(-result.ap_cost)

    # ── Passive perception (Layer 4) ─────────────────────────────────────
    loc = world.get_location(player.location)
    if loc is not None:
        npcs_here = _npcs_at_location(player.location, npc_registry)
        perception = passive_perception_check(
            location_id=player.location,
            is_social_location=world.is_social(player.location),
            location_items=loc.items_on_ground,
            npcs_at_location=[
                {"npc_uid": n.get("uid", ""), "name": n.get("name", ""), "action_importance": 1}
                for n in npcs_here
            ],
        )
        if perception:
            result.narration += f" {perception['narration']}"
            result.effects["perception"] = perception

    # ── Context modifiers (Layer 3) ──────────────────────────────────────
    npc_name_map = {uid: n.get("name", uid) for uid, n in npc_registry.items()}
    result.narration = add_context_modifiers(
        base_narration=result.narration,
        time_of_day=world.time_of_day,
        weather=_get_active_weather(world),
        witnesses=result.effects.get("witnesses", []),
        npc_names=npc_name_map,
    )

    return result


def _get_active_weather(world: Any) -> str | None:
    """Extract an active weather description from world events."""
    for ev in world.active_events:
        eid = ev.get("id", "")
        if eid.startswith("weather_"):
            return ev.get("description", "The weather has changed.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Individual action handlers
# ══════════════════════════════════════════════════════════════════════════════
# Every handler shares the signature:
#   (player, world, npc_registry, event_log, parsed_input, effective_ap) -> ActionResult


# ─── Navigation ──────────────────────────────────────────────────────────────

def _handle_move_to(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``move_to`` — move player to an adjacent location."""
    target_loc = parsed_input.get("target_location") or ""

    if target_loc not in LOCATION_IDS:
        return _blocked("move_to", f"Unknown location: '{target_loc}'.",
                        {"target": target_loc or "nowhere"})

    if target_loc == player.location:
        return _blocked("move_to", "You're already here.",
                        {"target": target_loc})

    if not world.is_adjacent(player.location, target_loc):
        return _blocked("move_to", "You can't reach there from here.",
                        {"target": target_loc})

    old_location = player.location
    player.location = target_loc
    loc_obj = world.get_location(target_loc)
    loc_name = loc_obj.name if loc_obj else target_loc

    narration = get_template_narration("move_to", "success", {"target": loc_name})

    return ActionResult(
        success=True,
        outcome="success",
        narration=narration,
        effects={"moved_from": old_location, "moved_to": target_loc},
        ap_cost=effective_ap,
        importance=2,
        event_type="player_action",
    )


# ─── Exploration ─────────────────────────────────────────────────────────────

def _handle_look(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``look`` — observe current location, NPCs, items."""
    loc = world.get_location(player.location)
    if loc is None:
        return _blocked("look", "You can't see anything here.")

    npcs_here = _npcs_at_location(player.location, npc_registry)
    npc_names = [n.get("name", "someone") for n in npcs_here]
    item_names = [i.get("name", "an item") for i in loc.items_on_ground]

    parts: list[str] = [loc.description]
    if npc_names:
        parts.append(f"You see: {', '.join(npc_names)}.")
    if item_names:
        parts.append(f"On the ground: {', '.join(item_names)}.")
    if not npc_names and not item_names:
        parts.append("The area is quiet.")

    discovery = " ".join(parts)
    narration = get_template_narration("look", "success", {"discovery": discovery})

    return ActionResult(
        success=True,
        outcome="success",
        narration=narration,
        effects={
            "npcs_seen": [n.get("uid", "") for n in npcs_here],
            "items_seen": [i.get("id", "") for i in loc.items_on_ground],
        },
        ap_cost=effective_ap,
        importance=1,
        event_type="player_action",
    )


def _handle_search(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``search`` — thorough search using skill probability."""
    loc = world.get_location(player.location)
    if loc is None:
        return _blocked("search", "Nothing to search here.")

    loc.search_count += 1
    prob = compute_skill_probability("search", search_count=loc.search_count)

    if random.random() < prob and loc.items_on_ground:
        item = random.choice(loc.items_on_ground)
        narration = get_template_narration(
            "search", "success", {"discovery": item.get("name", "something")}
        )
        return ActionResult(
            success=True,
            outcome="success",
            narration=narration,
            effects={"discovered_item": item.get("id", ""), "search_count": loc.search_count},
            ap_cost=effective_ap,
            importance=2,
            event_type="player_action",
        )

    narration = get_template_narration("search", "fail")
    return ActionResult(
        success=False,
        outcome="fail",
        narration=narration,
        effects={"search_count": loc.search_count},
        ap_cost=effective_ap,
        importance=1,
        event_type="player_action",
    )


def _handle_examine(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``examine`` — inspect a specific target (NPC, item, or object)."""
    target_npc_uid = parsed_input.get("target_npc")
    target_item_id = parsed_input.get("target_item")
    loc = world.get_location(player.location)

    # Examine NPC
    if target_npc_uid:
        npc = _find_target_npc(target_npc_uid, player.location, npc_registry)
        if npc:
            name = npc.get("name", "them")
            desc = npc.get("description", "Nothing unusual stands out.")
            narration = get_template_narration(
                "examine", "success", {"target": name, "discovery": desc}
            )
            return ActionResult(
                success=True, outcome="success", narration=narration,
                effects={"examined_npc": target_npc_uid},
                ap_cost=effective_ap, importance=1, event_type="player_action",
            )
        return _blocked("examine", "They don't seem to be here.",
                        {"target": target_npc_uid})

    # Examine inventory item
    if target_item_id:
        item = player.get_item(target_item_id)
        if item:
            narration = get_template_narration(
                "examine", "success",
                {"target": item["name"], "discovery": item.get("description", "")},
            )
            return ActionResult(
                success=True, outcome="success", narration=narration,
                effects={"examined_item": target_item_id},
                ap_cost=effective_ap, importance=1, event_type="player_action",
            )
        # Check ground items
        if loc:
            for gi in loc.items_on_ground:
                if gi.get("id") == target_item_id:
                    narration = get_template_narration(
                        "examine", "success",
                        {"target": gi["name"], "discovery": gi.get("description", "")},
                    )
                    return ActionResult(
                        success=True, outcome="success", narration=narration,
                        effects={"examined_item": target_item_id},
                        ap_cost=effective_ap, importance=1, event_type="player_action",
                    )

    # Generic examine — look at surroundings
    if loc:
        discovery = loc.description
        narration = get_template_narration(
            "examine", "success", {"target": "your surroundings", "discovery": discovery}
        )
        return ActionResult(
            success=True, outcome="success", narration=narration,
            effects={}, ap_cost=effective_ap, importance=1, event_type="player_action",
        )

    return _blocked("examine", "There's nothing here to examine closely.")


# ─── Social ──────────────────────────────────────────────────────────────────

def _handle_talk(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``talk`` — initiate conversation with a target NPC."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("talk", "There's nobody around to talk to.")

    name = npc.get("name", "them")
    narration = get_template_narration("talk", "success", {"target": name})

    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"talked_to": npc.get("uid", "")},
        ap_cost=effective_ap, importance=2, event_type="dialogue",
    )


def _handle_greet(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``greet`` — greet an NPC. Tracks first-greeting bonus."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("greet", "There's nobody nearby to greet.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")
    effects: dict[str, Any] = {"greeted": npc_uid}

    first_time = not player.has_greeted(npc_uid)
    if first_time:
        player.mark_greeted(npc_uid)
        rep_change = player.modify_reputation(npc_uid, 2)
        effects["reputation"] = {npc_uid: rep_change}
        effects["first_greet"] = True

    narration = get_template_narration("greet", "success", {"target": name})
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects=effects, ap_cost=effective_ap, importance=2, event_type="dialogue",
    )


def _handle_ask_info(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``ask_info`` — ask NPC for information."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("ask_info", "There's nobody here to ask.")

    name = npc.get("name", "them")
    rep = player.get_reputation(npc.get("uid", ""))

    # Hostile NPCs refuse info
    if rep <= -50:
        narration = get_template_narration("ask_info", "fail", {"target": name})
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects={"asked": npc.get("uid", "")},
            ap_cost=effective_ap, importance=2, event_type="dialogue",
        )

    narration = get_template_narration(
        "ask_info", "success",
        {"target": name, "dialogue": "They share what they know."},
    )
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"asked": npc.get("uid", "")},
        ap_cost=effective_ap, importance=2, event_type="dialogue",
    )


def _handle_persuade(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``persuade`` — skill-check to convince an NPC."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("persuade", "There's nobody here to persuade.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")
    rep = player.get_reputation(npc_uid)
    social = parsed_input.get("social", "neutral")
    social_mod = SOCIAL_MODIFIERS.get(social, 0)

    prob = compute_skill_probability("persuade", reputation=rep, social_modifier=social_mod)
    success = random.random() < prob

    effects: dict[str, Any] = {"target_npc": npc_uid, "probability": round(prob, 3)}
    if success:
        rep_change = player.modify_reputation(npc_uid, 5)
        effects["reputation"] = {npc_uid: rep_change}
        narration = get_template_narration("persuade", "success", {"target": name})
        return ActionResult(
            success=True, outcome="success", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )
    else:
        rep_change = player.modify_reputation(npc_uid, -2)
        effects["reputation"] = {npc_uid: rep_change}
        narration = get_template_narration("persuade", "fail", {"target": name})
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=2, event_type="player_action",
        )


def _handle_trade(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``trade`` — attempt to trade with an NPC."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("trade", "There's nobody around to trade with.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")
    rep = player.get_reputation(npc_uid)

    # Hostile NPCs refuse trade
    if rep <= -50:
        narration = get_template_narration("trade", "fail", {"target": name})
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects={"target_npc": npc_uid},
            ap_cost=effective_ap, importance=2, event_type="player_action",
        )

    narration = get_template_narration("trade", "success", {"target": name})
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"traded_with": npc_uid},
        ap_cost=effective_ap, importance=2, event_type="player_action",
    )


def _handle_give_item(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``give_item`` — give an inventory item to an NPC."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("give_item", "There's nobody here to give items to.")

    target_item_id = parsed_input.get("target_item")
    if not target_item_id:
        return _blocked("give_item", "You have nothing suitable to give.")

    item = player.get_item(target_item_id)
    if not item:
        return _blocked("give_item", "You don't have that item.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")
    item_name = item.get("name", "an item")

    # Remove from player inventory
    removed = player.remove_item(target_item_id)
    if removed is None:
        return _blocked("give_item", "You can't give that away.")

    # Reputation boost from giving
    rep_change = player.modify_reputation(npc_uid, 5)

    narration = get_template_narration(
        "give_item", "success", {"target": name, "item": item_name}
    )
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={
            "gave_item": target_item_id,
            "to_npc": npc_uid,
            "reputation": {npc_uid: rep_change},
        },
        ap_cost=effective_ap, importance=3, event_type="player_action",
    )


def _handle_deceive(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``deceive`` — skill-check to lie / bluff an NPC."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("deceive", "There's nobody here to deceive.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")
    rep = player.get_reputation(npc_uid)
    social = parsed_input.get("social", "neutral")
    social_mod = SOCIAL_MODIFIERS.get(social, 0)

    prob = compute_skill_probability("deceive", reputation=rep, social_modifier=social_mod)
    success = random.random() < prob

    effects: dict[str, Any] = {"target_npc": npc_uid, "probability": round(prob, 3)}
    if success:
        narration = get_template_narration("deceive", "success", {"target": name})
        return ActionResult(
            success=True, outcome="success", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )
    else:
        # Caught lying — reputation penalty
        rep_change = player.modify_reputation(npc_uid, -10)
        effects["reputation"] = {npc_uid: rep_change}
        narration = get_template_narration("deceive", "fail", {"target": name})
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )


def _handle_intimidate(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``intimidate`` — skill-check to threaten an NPC.

    Uses the persuade formula with a negative reputation sign for harder
    checks against friendly NPCs (they won't be easily cowed).
    """
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("intimidate", "There's nobody here to intimidate.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")
    rep = player.get_reputation(npc_uid)
    social = parsed_input.get("social", "neutral")
    social_mod = SOCIAL_MODIFIERS.get(social, 0)

    # Intimidation is harder against allies, easier against enemies
    prob = compute_skill_probability("persuade", reputation=-rep, social_modifier=social_mod)
    success = random.random() < prob

    effects: dict[str, Any] = {"target_npc": npc_uid, "probability": round(prob, 3)}
    if success:
        rep_change = player.modify_reputation(npc_uid, -8)
        effects["reputation"] = {npc_uid: rep_change}
        narration = get_template_narration("intimidate", "success", {"target": name})
        return ActionResult(
            success=True, outcome="success", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )
    else:
        rep_change = player.modify_reputation(npc_uid, -3)
        effects["reputation"] = {npc_uid: rep_change}
        narration = get_template_narration("intimidate", "fail", {"target": name})
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )


# ─── Combat ──────────────────────────────────────────────────────────────────

def _handle_attack(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``attack`` — initiate or continue combat with a target NPC."""
    target_uid = parsed_input.get("target_npc")
    npc = _find_target_npc(target_uid, player.location, npc_registry)
    if npc is None:
        return _blocked("attack", "There's no one here to attack.", {"target": "thin air"})

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")

    # Enter combat state
    player.in_combat = True
    player.combat_target = npc_uid

    attacker = player.get_combat_dict()
    defender = {
        "name": name,
        "base_attack": npc.get("base_attack", 5),
        "base_defense": npc.get("base_defense", 3),
        "weapon_modifier": npc.get("weapon_modifier", 0),
        "armor_modifier": npc.get("armor_modifier", 0),
        "current_stamina": npc.get("stamina", 25),
        "max_stamina": npc.get("max_stamina", 50),
        "is_player": False,
        "is_defending": npc.get("is_defending", False),
    }

    result = resolve_attack(attacker, defender)

    effects: dict[str, Any] = {
        "target_npc": npc_uid,
        "hit": result.hit,
        "damage": result.damage,
    }

    # Apply damage to NPC
    if result.hit:
        effects["npc_health_change"] = -result.damage

    # Reputation penalty for attacking
    witnesses = list(
        uid for uid, loc in _npc_locations(npc_registry).items()
        if loc == player.location and uid != npc_uid
    )
    rep_change = player.modify_reputation(npc_uid, -15)
    effects["reputation"] = {npc_uid: rep_change}
    # Witness penalty
    for w_uid in witnesses:
        w_change = player.modify_reputation(w_uid, -5)
        effects["reputation"][w_uid] = w_change
    effects["witnesses"] = witnesses

    outcome = "success" if result.hit else "fail"
    narration = get_template_narration(
        "attack", outcome, {"target": name, "damage": result.damage}
    )

    return ActionResult(
        success=result.hit,
        outcome=outcome,
        narration=narration,
        effects=effects,
        ap_cost=effective_ap,
        importance=4,
        event_type="combat",
    )


def _handle_defend(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``defend`` — enter defensive stance (next incoming attack −50% damage)."""
    player.is_defending = True

    narration = get_template_narration("defend", "success")
    return ActionResult(
        success=True,
        outcome="success",
        narration=narration,
        effects={"defending": True},
        ap_cost=effective_ap,
        importance=2,
        event_type="combat",
    )


def _handle_flee(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``flee`` — attempt to escape combat.

    70% base success; drops to 40% at 0 stamina.
    Failure gives opponent a free attack at +20% hit.
    """
    target_uid = player.combat_target
    npc = npc_registry.get(target_uid, {}) if target_uid else {}
    name = npc.get("name", "your opponent")

    fleeing = player.get_combat_dict()
    opponent = {
        "name": name,
        "base_attack": npc.get("base_attack", 5),
        "base_defense": npc.get("base_defense", 3),
        "weapon_modifier": npc.get("weapon_modifier", 0),
        "armor_modifier": npc.get("armor_modifier", 0),
        "current_stamina": npc.get("stamina", 25),
        "max_stamina": npc.get("max_stamina", 50),
        "is_player": False,
        "is_defending": False,
    }

    flee_result = resolve_flee(fleeing, opponent)
    effects: dict[str, Any] = {"fled": flee_result["success"]}

    if flee_result["success"]:
        # Escape — pick a random adjacent location
        adjacent = world.get_adjacent(player.location)
        if adjacent:
            escape_dest = random.choice(adjacent)
            player.location = escape_dest
            effects["escaped_to"] = escape_dest
        player.in_combat = False
        player.combat_target = None
        loc_obj = world.get_location(player.location)
        dest_name = loc_obj.name if loc_obj else player.location
        narration = get_template_narration("flee", "success", {"target": dest_name})
        return ActionResult(
            success=True, outcome="success", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="combat",
        )
    else:
        # Failed — free attack
        free_atk = flee_result.get("free_attack")
        if free_atk and free_atk.hit:
            player.modify_health(-free_atk.damage)
            effects["damage_taken"] = free_atk.damage
        narration = get_template_narration("flee", "fail")
        if free_atk:
            narration += f" {free_atk.narrative}"
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="combat",
        )


# ─── Stealth ─────────────────────────────────────────────────────────────────

def _handle_sneak(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``sneak`` — attempt to move stealthily."""
    npcs_here = _npcs_at_location(player.location, npc_registry)
    time_bonus = world.get_time_bonus()
    prob = compute_skill_probability(
        "sneak", time_bonus=time_bonus, npcs_at_location=len(npcs_here),
    )
    success = random.random() < prob

    outcome = "success" if success else "fail"
    narration = get_template_narration("sneak", outcome)
    effects: dict[str, Any] = {"probability": round(prob, 3), "detected": not success}

    if not success:
        # Detected — nearby NPCs become suspicious
        for npc in npcs_here:
            uid = npc.get("uid", "")
            player.modify_reputation(uid, -2)
        effects["reputation_penalty"] = -2

    return ActionResult(
        success=success, outcome=outcome, narration=narration,
        effects=effects, ap_cost=effective_ap,
        importance=2 if success else 3, event_type="player_action",
    )


def _handle_hide(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``hide`` — attempt to conceal yourself."""
    npcs_here = _npcs_at_location(player.location, npc_registry)
    time_bonus = world.get_time_bonus()
    prob = compute_skill_probability(
        "hide", time_bonus=time_bonus, npcs_at_location=len(npcs_here),
    )
    success = random.random() < prob

    outcome = "success" if success else "fail"
    narration = get_template_narration("hide", outcome)
    effects: dict[str, Any] = {"probability": round(prob, 3), "hidden": success}

    return ActionResult(
        success=success, outcome=outcome, narration=narration,
        effects=effects, ap_cost=effective_ap,
        importance=2 if success else 3, event_type="player_action",
    )


def _handle_steal(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``steal`` — attempt to steal an item from a target NPC."""
    npc = _find_target_npc(parsed_input.get("target_npc"), player.location, npc_registry)
    if npc is None:
        return _blocked("steal", "There's nothing worth stealing here.")

    npc_uid = npc.get("uid", "")
    name = npc.get("name", "them")

    npcs_here = _npcs_at_location(player.location, npc_registry)
    time_bonus = world.get_time_bonus()
    prob = compute_skill_probability(
        "steal", time_bonus=time_bonus, npcs_at_location=len(npcs_here),
    )
    success = random.random() < prob

    effects: dict[str, Any] = {"target_npc": npc_uid, "probability": round(prob, 3)}

    if success:
        # Successful steal — get a random item from NPC (placeholder)
        stolen_item = {
            "id": f"stolen_{npc_uid}",
            "name": f"Stolen goods from {name}",
            "type": "misc",
            "quest_relevant": False,
            "description": f"Something you pilfered from {name}.",
            "effects": {},
            "slot": None,
            "stat_modifiers": None,
        }
        if not player.inventory_full():
            player.add_item(stolen_item)
            effects["stolen_item"] = stolen_item["id"]
        else:
            effects["inventory_full"] = True

        narration = get_template_narration(
            "steal", "success", {"item": stolen_item["name"], "target": name}
        )
        return ActionResult(
            success=True, outcome="success", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )
    else:
        # Caught! Major reputation hit
        rep_change = player.modify_reputation(npc_uid, -15)
        effects["reputation"] = {npc_uid: rep_change}
        witnesses = [
            n.get("uid", "") for n in npcs_here if n.get("uid") != npc_uid
        ]
        for w_uid in witnesses:
            w_rep = player.modify_reputation(w_uid, -5)
            effects["reputation"][w_uid] = w_rep
        effects["witnesses"] = witnesses

        narration = get_template_narration(
            "steal", "fail", {"item": "something", "target": name}
        )
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects=effects, ap_cost=effective_ap, importance=3, event_type="player_action",
        )


# ─── Utility ─────────────────────────────────────────────────────────────────

def _handle_pick_up(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``pick_up`` — pick up an item from the ground."""
    loc = world.get_location(player.location)
    if loc is None or not loc.items_on_ground:
        return _blocked("pick_up", "There's nothing here to pick up.")

    if player.inventory_full():
        return _blocked("pick_up", "Your inventory is full!")

    target_item_id = parsed_input.get("target_item")

    # Specific item
    if target_item_id:
        for i, ground_item in enumerate(loc.items_on_ground):
            if ground_item.get("id") == target_item_id:
                item = loc.items_on_ground.pop(i)
                player.add_item(item)
                narration = get_template_narration(
                    "pick_up", "success", {"item": item.get("name", "an item")}
                )
                return ActionResult(
                    success=True, outcome="success", narration=narration,
                    effects={"picked_up": item["id"]},
                    ap_cost=effective_ap, importance=2, event_type="player_action",
                )
        return _blocked("pick_up", "That item isn't here.")

    # No specific target — pick up first item
    item = loc.items_on_ground.pop(0)
    player.add_item(item)
    narration = get_template_narration(
        "pick_up", "success", {"item": item.get("name", "an item")}
    )
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"picked_up": item["id"]},
        ap_cost=effective_ap, importance=2, event_type="player_action",
    )


def _handle_use_item(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``use_item`` — use an item from inventory, applying its effects."""
    target_item_id = parsed_input.get("target_item")
    if not target_item_id:
        return _blocked("use_item", "You don't have any usable items.")

    item = player.get_item(target_item_id)
    if not item:
        return _blocked("use_item", "You don't have that item.")

    item_name = item.get("name", "an item")
    effects_applied: dict[str, Any] = {"used_item": target_item_id}
    effect_parts: list[str] = []

    item_effects = item.get("effects", {})

    # Apply heal
    if "heal" in item_effects:
        heal_amount = item_effects["heal"]
        actual = player.modify_health(heal_amount)
        effects_applied["health_change"] = actual
        effect_parts.append(f"+{actual} HP")

    # Apply stamina restoration
    if "stamina" in item_effects:
        stam_amount = item_effects["stamina"]
        actual = player.modify_stamina(stam_amount)
        effects_applied["stamina_change"] = actual
        effect_parts.append(f"+{actual} AP")

    # Remove consumables after use
    if item.get("type") == "consumable":
        player.remove_item(target_item_id)
        effects_applied["consumed"] = True

    effect_text = ", ".join(effect_parts) if effect_parts else "Nothing happens."
    narration = get_template_narration(
        "use_item", "success", {"item": item_name, "effect": effect_text}
    )

    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects=effects_applied, ap_cost=effective_ap, importance=2,
        event_type="player_action",
    )


def _handle_eat(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``eat`` — eat a food item to restore HP."""
    target_item_id = parsed_input.get("target_item")

    # Find a food item
    food = None
    if target_item_id:
        item = player.get_item(target_item_id)
        if item and item.get("type") == "consumable" and "heal" in item.get("effects", {}):
            food = item
    else:
        food_items = player.get_food_items()
        if food_items:
            food = food_items[0]

    if food is None:
        return _blocked("eat", "You don't have any food to eat.")

    heal_amount = food.get("effects", {}).get("heal", 5)
    actual_heal = player.modify_health(heal_amount)
    player.remove_item(food["id"])

    narration = get_template_narration(
        "eat", "success", {"item": food["name"], "heal": actual_heal}
    )
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"ate": food["id"], "health_change": actual_heal},
        ap_cost=effective_ap, importance=1, event_type="player_action",
    )


def _handle_rest(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``rest`` — restore +10 AP if at an indoor / safe location.

    Outdoor or combat locations resolve as ``wait`` instead.
    """
    is_indoor = world.is_indoor(player.location)
    in_combat = player.in_combat

    if not is_indoor or in_combat:
        # Falls back to wait behavior
        narration = get_template_narration("rest", "fail")
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects={"rested": False, "resolved_as": "wait"},
            ap_cost=0, importance=1, event_type="player_action",
        )

    regen = player.modify_stamina(10)
    narration = get_template_narration("rest", "success")
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"rested": True, "stamina_change": regen},
        ap_cost=0, importance=1, event_type="player_action",
    )


def _handle_wait(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``wait`` — advance time, run passive perception check."""
    narration = get_template_narration("wait", "success")
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"waited": True},
        ap_cost=0, importance=1, event_type="player_action",
    )


def _handle_drop_item(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``drop_item`` — drop a non-quest item from inventory."""
    target_item_id = parsed_input.get("target_item")
    if not target_item_id:
        if not player.inventory:
            return _blocked("drop_item", "You have nothing to drop.")
        # Default to first droppable item
        droppable = player.get_droppable_items()
        if not droppable:
            return _blocked("drop_item", "You can't drop quest-critical items.")
        target_item_id = droppable[0]["id"]

    item = player.get_item(target_item_id)
    if not item:
        return _blocked("drop_item", "You don't have that item.")

    if item.get("quest_relevant", False):
        return _blocked("drop_item", "You can't drop quest-critical items.")

    removed = player.remove_item(target_item_id)
    if removed is None:
        return _blocked("drop_item", "Unable to drop that item.")

    # Place on ground
    loc = world.get_location(player.location)
    if loc is not None:
        loc.items_on_ground.append(removed)

    narration = get_template_narration(
        "drop_item", "success", {"item": removed.get("name", "an item")}
    )
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"dropped": target_item_id},
        ap_cost=0, importance=1, event_type="player_action",
    )


def _handle_status(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``status`` — display current quest state and player stats."""
    qs = player.quest_state
    quest_info = (
        f"Stage {qs.get('current_stage', '?')}, "
        f"Checkpoint {qs.get('current_checkpoint', '?')}. "
        f"Completed: {', '.join(qs.get('completed_checkpoints', [])) or 'none'}."
    )
    narration = get_template_narration("status", "success", {"quest_info": quest_info})

    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={
            "health": player.health,
            "stamina": player.stamina,
            "location": player.location,
            "quest_state": qs,
        },
        ap_cost=0, importance=1, event_type="player_action",
    )


def _handle_equip(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``equip`` — equip a weapon or armor from inventory."""
    target_item_id = parsed_input.get("target_item")

    if not target_item_id:
        equipment = player.get_equipment_items()
        if not equipment:
            return _blocked("equip", "You have nothing to equip.")
        target_item_id = equipment[0]["id"]

    item = player.get_item(target_item_id)
    if not item:
        return _blocked("equip", "You don't have that item.")

    if item.get("type") != "equipment":
        return _blocked("equip", f"{item.get('name', 'That')} isn't equipment.")

    previous = player.equip_item(target_item_id)
    item_name = item.get("name", "an item")
    narration = get_template_narration("equip", "success", {"item": item_name})

    effects: dict[str, Any] = {"equipped": target_item_id, "slot": item.get("slot")}
    if previous:
        effects["unequipped"] = previous.get("id", "")

    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects=effects, ap_cost=effective_ap, importance=2, event_type="player_action",
    )


def _handle_work(
    *, player: Any, world: Any, npc_registry: dict, event_log: Any,
    parsed_input: dict, effective_ap: int,
) -> ActionResult:
    """Resolve ``work`` — perform manual labor.

    Most productive at Fields; grants reputation with nearby NPCs.
    """
    at_fields = player.location == "fields"

    if not at_fields:
        narration = get_template_narration("work", "fail")
        return ActionResult(
            success=False, outcome="fail", narration=narration,
            effects={"worked": False},
            ap_cost=effective_ap, importance=1, event_type="player_action",
        )

    # Grant reputation with all NPCs at this location
    npcs_here = _npcs_at_location(player.location, npc_registry)
    rep_effects: dict[str, int] = {}
    for npc in npcs_here:
        uid = npc.get("uid", "")
        change = player.modify_reputation(uid, 3)
        rep_effects[uid] = change

    narration = get_template_narration("work", "success")
    return ActionResult(
        success=True, outcome="success", narration=narration,
        effects={"worked": True, "reputation": rep_effects},
        ap_cost=effective_ap, importance=2, event_type="player_action",
    )


# ─── Handler Registry ────────────────────────────────────────────────────────

_ACTION_HANDLERS: dict[str, Any] = {
    # Navigation
    "move_to":    _handle_move_to,
    # Exploration
    "look":       _handle_look,
    "search":     _handle_search,
    "examine":    _handle_examine,
    # Social
    "talk":       _handle_talk,
    "greet":      _handle_greet,
    "ask_info":   _handle_ask_info,
    "persuade":   _handle_persuade,
    "trade":      _handle_trade,
    "give_item":  _handle_give_item,
    "deceive":    _handle_deceive,
    "intimidate": _handle_intimidate,
    # Combat
    "attack":     _handle_attack,
    "defend":     _handle_defend,
    "flee":       _handle_flee,
    # Stealth
    "sneak":      _handle_sneak,
    "hide":       _handle_hide,
    "steal":      _handle_steal,
    # Utility
    "pick_up":    _handle_pick_up,
    "use_item":   _handle_use_item,
    "eat":        _handle_eat,
    "rest":       _handle_rest,
    "wait":       _handle_wait,
    "drop_item":  _handle_drop_item,
    "status":     _handle_status,
    "equip":      _handle_equip,
    "work":       _handle_work,
}
