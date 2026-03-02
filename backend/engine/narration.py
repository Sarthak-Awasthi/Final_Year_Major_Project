"""
narration.py — Action outcome narration templates + LLM enhancement.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from backend.config import LLM_MAX_RETRIES, PASSIVE_PERCEPTION_BASE, PASSIVE_PERCEPTION_SOCIAL_BONUS, logger
from backend.llm.prompts import build_narration_prompt
from backend.llm.guardrails import sanitize_text

if TYPE_CHECKING:
    from backend.llm.llm_service import LLMService


# ─── Layer 1: Template Narration ──────────────────────────────────────────────
# Pre-written templates for all 27 actions × outcome types

ACTION_NARRATION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "move_to": {
        "success": [
            "You make your way to {target}.",
            "You travel to {target}. The path is familiar.",
        ],
        "blocked": [
            "You can't reach {target} from here. You'd need to go through the village center.",
            "There's no direct path to {target} from your current location.",
        ],
    },
    "look": {
        "success": [
            "You look around carefully. {discovery}",
            "You survey your surroundings. {discovery}",
        ],
        "fail": [
            "You look around but nothing stands out.",
            "A quick glance reveals nothing unusual.",
        ],
    },
    "search": {
        "success": [
            "You search the area thoroughly. {discovery}",
            "After careful examination, you find {discovery}.",
        ],
        "fail": [
            "You search around but find nothing of interest.",
            "Despite a thorough search, nothing stands out.",
        ],
    },
    "examine": {
        "success": [
            "You examine {target} closely. {discovery}",
            "Upon inspection, {target} reveals {discovery}.",
        ],
        "blocked": [
            "There's nothing here to examine closely.",
        ],
    },
    "talk": {
        "success": [
            "You approach {target} and strike up a conversation.",
            "{target} turns to listen as you speak.",
        ],
        "fail": [
            "{target} ignores you completely.",
            "{target} turns away, clearly uninterested.",
        ],
        "blocked": [
            "There's nobody around to talk to.",
        ],
    },
    "greet": {
        "success": [
            "You greet {target}. They acknowledge you with a nod.",
            "'Hello,' you say to {target}. They look up in response.",
        ],
        "fail": [
            "{target} barely glances at you and looks away.",
        ],
        "blocked": [
            "There's nobody nearby to greet.",
        ],
    },
    "ask_info": {
        "success": [
            "You ask {target} for information. {dialogue}",
            "{target} considers your question. {dialogue}",
        ],
        "fail": [
            "{target} shakes their head. 'Can't help you with that.'",
            "{target} crosses their arms. 'I don't know anything about that.'",
        ],
        "blocked": [
            "There's nobody here to ask.",
        ],
    },
    "persuade": {
        "success": [
            "Your words find their mark. {target} seems convinced.",
            "After careful reasoning, {target} nods in agreement.",
        ],
        "fail": [
            "{target} shakes their head firmly. 'I don't think so.'",
            "Your argument falls flat. {target} remains unconvinced.",
        ],
        "blocked": [
            "There's nobody here to persuade.",
        ],
    },
    "trade": {
        "success": [
            "You complete a trade with {target}.",
            "{target} agrees to the trade. Items change hands.",
        ],
        "fail": [
            "{target} isn't interested in trading right now.",
            "'I don't have anything you'd want,' {target} says.",
        ],
        "blocked": [
            "There's nobody around to trade with.",
        ],
    },
    "give_item": {
        "success": [
            "You hand {item} to {target}. They accept it gratefully.",
            "{target} takes {item} from you with a nod of thanks.",
        ],
        "fail": [
            "{target} refuses your offering.",
        ],
        "blocked": [
            "There's nobody here to give items to.",
            "You have nothing suitable to give.",
        ],
    },
    "present_item": {
        "success": [
            "You hold up {item} for {target} to inspect. They lean in for a closer look.",
            "You present {item} to {target}. They examine it carefully and nod.",
            "{target} studies {item} as you hold it out. Their expression shifts with recognition.",
        ],
        "fail": [
            "{target} glances at {item} but seems uninterested.",
            "{target} barely acknowledges what you're showing them.",
        ],
        "blocked": [
            "There's nobody here to show items to.",
            "You have nothing to present.",
        ],
    },
    "deceive": {
        "success": [
            "You spin a convincing tale. {target} seems to believe you.",
            "Your deception works. {target} is none the wiser.",
        ],
        "fail": [
            "{target} narrows their eyes. 'I don't believe a word of that.'",
            "Your deception falls apart. {target} sees through your lie.",
        ],
        "blocked": [
            "There's nobody here to deceive.",
        ],
    },
    "intimidate": {
        "success": [
            "You step forward menacingly. {target} backs away, cowed.",
            "Your threat lands. {target} swallows hard and complies.",
        ],
        "fail": [
            "{target} stands their ground. 'You don't scare me.'",
            "Your intimidation attempt fails. {target} looks unimpressed.",
        ],
        "blocked": [
            "There's nobody here to intimidate.",
        ],
    },
    "attack": {
        "success": [
            "You land a solid blow on {target}. They stagger back, taking {damage} damage.",
            "Your strike connects with {target}! {damage} damage dealt.",
        ],
        "fail": [
            "You swing at {target}, but they dodge aside.",
            "{target} deflects your attack effortlessly.",
        ],
        "blocked": [
            "There's no one here to attack.",
            "You clench your fists, but think better of it.",
        ],
    },
    "defend": {
        "success": [
            "You raise your guard, ready to deflect incoming blows.",
            "You brace yourself, adopting a defensive stance.",
        ],
        "blocked": [
            "You're not in combat. There's nothing to defend against.",
        ],
    },
    "flee": {
        "success": [
            "You turn and run! You escape to {target}.",
            "You break away from combat and flee to {target}.",
        ],
        "fail": [
            "You try to flee but are blocked! Your opponent presses the advantage.",
            "Your escape attempt fails! You're still locked in combat.",
        ],
    },
    "sneak": {
        "success": [
            "You move silently, sticking to the shadows.",
            "You creep forward unseen, careful not to make a sound.",
        ],
        "fail": [
            "You try to sneak but stumble. Someone notices you!",
            "A twig snaps underfoot. So much for stealth.",
        ],
    },
    "hide": {
        "success": [
            "You find a concealed spot and press yourself into the shadows.",
            "You duck behind cover and go still. Nobody seems to notice.",
        ],
        "fail": [
            "You look for a hiding spot but can't find adequate cover.",
            "You try to hide but are spotted!",
        ],
    },
    "steal": {
        "success": [
            "Your fingers are quick. You pocket {item} without being noticed.",
            "You deftly take {item} when no one is looking.",
        ],
        "fail": [
            "You reach for {item} but are caught red-handed!",
            "Your attempt to steal is noticed! {target} confronts you.",
        ],
        "blocked": [
            "There's nothing worth stealing here.",
        ],
    },
    "pick_up": {
        "success": [
            "You pick up {item}.",
            "You add {item} to your pack.",
        ],
        "blocked": [
            "There's nothing here to pick up.",
            "Your inventory is full!",
        ],
    },
    "use_item": {
        "success": [
            "You use {item}. {effect}",
        ],
        "blocked": [
            "You don't have any usable items.",
        ],
    },
    "eat": {
        "success": [
            "You eat {item}. It restores some of your strength. (+{heal} HP)",
            "You consume {item}. You feel a bit better. (+{heal} HP)",
        ],
        "blocked": [
            "You don't have any food to eat.",
        ],
    },
    "rest": {
        "success": [
            "You find a comfortable spot and rest for a while. (+10 AP)",
            "You sit down and catch your breath. Energy returns to your limbs. (+10 AP)",
        ],
        "fail": [
            "This isn't a safe place to rest. You wait instead.",
        ],
    },
    "wait": {
        "success": [
            "You wait and observe your surroundings.",
            "Time passes. You keep your eyes and ears open.",
        ],
    },
    "drop_item": {
        "success": [
            "You drop {item} on the ground.",
        ],
        "blocked": [
            "You can't drop quest-critical items.",
            "You have nothing to drop.",
        ],
    },
    "status": {
        "success": [
            "You check your quest journal. {quest_info}",
        ],
    },
    "equip": {
        "success": [
            "You equip {item}.",
            "You ready {item} for use.",
        ],
        "blocked": [
            "You have nothing to equip.",
        ],
    },
    "work": {
        "success": [
            "You put in some honest labor. It's tiring but satisfying.",
            "You work hard for a while. Your efforts don't go unnoticed.",
        ],
        "fail": [
            "There's not much work to be done here right now.",
        ],
    },
}


def get_template_narration(
    action_id: str,
    outcome: str,
    context: dict[str, Any] | None = None,
) -> str:
    """
    Get a template-based narration for an action outcome.

    Args:
        action_id: The universal action ID
        outcome: 'success', 'fail', 'partial', or 'blocked'
        context: Dict with template variables (target, item, damage, discovery, etc.)
    """
    ctx = context or {}
    templates = ACTION_NARRATION_TEMPLATES.get(action_id, {})
    options = templates.get(outcome, templates.get("success", ["Something happens."]))

    if not options:
        options = ["Something happens."]

    template = random.choice(options)

    # Safe format with defaults
    try:
        narration = template.format(
            target=ctx.get("target", "them"),
            item=ctx.get("item", "an item"),
            damage=ctx.get("damage", 0),
            heal=ctx.get("heal", 0),
            discovery=ctx.get("discovery", "nothing unusual"),
            dialogue=ctx.get("dialogue", ""),
            effect=ctx.get("effect", ""),
            quest_info=ctx.get("quest_info", "No active quests."),
        )
    except (KeyError, IndexError):
        narration = template

    return narration


def enhance_narration_with_llm(
    template_narration: str,
    action_id: str,
    actor_name: str,
    target_name: str | None,
    outcome_type: str,
    location: str,
    time_of_day: str,
    weather: str | None,
    emotion: str,
    social: str,
    witnesses: list[str],
    llm_service: LLMService | None = None,
) -> str:
    """Layer 2: Enhance template narration with LLM for atmospheric detail.

    If the LLM is unavailable or fails, returns the original template
    narration unchanged.

    Args:
        template_narration: Base template narration from Layer 1.
        action_id: The universal action ID.
        actor_name: Who performed the action.
        target_name: Target of the action (may be ``None``).
        outcome_type: Outcome category (success/fail/blocked/partial).
        location: Current location name.
        time_of_day: Current time period.
        weather: Current weather condition or ``None``.
        emotion: Actor's emotional tone.
        social: Actor's social register.
        witnesses: List of witness NPC names.
        llm_service: Optional LLM service handle.

    Returns:
        Enhanced narration string, or the original if LLM is unavailable.
    """
    if llm_service is None or not llm_service.available:
        return template_narration

    prompt = build_narration_prompt(
        action_id=action_id,
        actor_name=actor_name,
        target_name=target_name,
        outcome_type=outcome_type,
        template_text=template_narration,
        location=location,
        time_of_day=time_of_day,
        weather=weather,
        emotion=emotion,
        social=social,
        witnesses=witnesses,
    )

    for attempt in range(LLM_MAX_RETRIES):
        raw = llm_service.generate(prompt, temperature=0.7, max_tokens=300)
        if raw and raw.strip():
            enhanced = sanitize_text(raw.strip())
            # Strip leaked prompt/instruction content
            enhanced = _strip_leaked_prompt(enhanced)
            if len(enhanced) >= 10:
                logger.info("LLM narration enhancement succeeded on attempt %d", attempt + 1)
                return enhanced
        logger.debug("LLM narration attempt %d failed or empty", attempt + 1)

    return template_narration


import re as _re

# Patterns that indicate the LLM echoed back prompt structure
_LEAKED_PROMPT_PATTERNS = [
    _re.compile(r"##\s*(Action Details|Base Narration|Instructions).*", _re.DOTALL),
    _re.compile(r"-\s*(Action|Actor|Target|Outcome|Location|Time|Weather|Tone|Manner|Witnesses):\s*"),
    _re.compile(r"Respond with ONLY.*", _re.DOTALL),
    _re.compile(r"Do NOT (write|add|echo).*$", _re.MULTILINE),
    _re.compile(r"Enhance the narrative with atmospheric.*$", _re.MULTILINE),
    _re.compile(r"Keep (it to|the outcome).*$", _re.MULTILINE),
    _re.compile(r"Write in second person.*$", _re.MULTILINE),
]


def _strip_leaked_prompt(text: str) -> str:
    """Remove leaked prompt/instruction artifacts from LLM narration output.

    The LLM sometimes echoes back parts of its instruction prompt as part of
    the generated text. This strips those fragments.

    Args:
        text: Raw LLM output after basic sanitization.

    Returns:
        Cleaned narration text.
    """
    # If text contains "## Action Details" or similar, take only the text before it
    for marker in ["## Action Details", "## Base Narration", "## Instructions"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
        elif idx == 0:
            # The entire output is prompt echo — discard
            return ""

    # Strip individual leaked lines
    for pattern in _LEAKED_PROMPT_PATTERNS:
        text = pattern.sub("", text)

    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def add_context_modifiers(
    base_narration: str,
    time_of_day: str,
    weather: str | None = None,
    witnesses: list[str] | None = None,
    npc_names: dict[str, str] | None = None,
) -> str:
    """
    Layer 3: Append environmental and witness context to narration.
    """
    modifiers: list[str] = []

    # Time-of-day flavor
    time_flavor = {
        "morning": "The morning sun casts long shadows.",
        "midday": "The midday sun beats down overhead.",
        "afternoon": "The afternoon light grows warm and golden.",
        "evening": "Dusk settles over the village, painting the sky in amber.",
        "night": "Darkness blankets everything. Torchlight flickers nearby.",
    }
    if time_of_day in time_flavor and random.random() < 0.3:
        modifiers.append(time_flavor[time_of_day])

    # Weather
    if weather:
        modifiers.append(weather)

    # Witnesses
    if witnesses and npc_names:
        witness_names = [npc_names.get(w, w) for w in witnesses[:2]]
        if len(witness_names) == 1:
            modifiers.append(f"{witness_names[0]} watches nearby.")
        elif len(witness_names) >= 2:
            modifiers.append(f"{witness_names[0]} and {witness_names[1]} observe your actions.")

    if modifiers:
        return base_narration + " " + " ".join(modifiers)
    return base_narration


def passive_perception_check(
    location_id: str,
    is_social_location: bool,
    location_items: list[dict],
    npcs_at_location: list[dict],
) -> dict | None:
    """
    Layer 4: Automatic perception check.
    Fires on any action at a location or on entering a new location.

    Returns a perception result dict or None.
    """
    base_chance = PASSIVE_PERCEPTION_BASE
    if is_social_location:
        base_chance += PASSIVE_PERCEPTION_SOCIAL_BONUS

    # Check for quest-critical items
    for item in location_items:
        if item.get("quest_relevant") and random.random() < base_chance:
            return {
                "type": "item_noticed",
                "item": item,
                "narration": f"Something catches your eye: {item.get('description', 'an item')}",
            }

    # Check for notable NPC behavior
    for npc in npcs_at_location:
        if npc.get("action_importance", 0) >= 3 and random.random() < base_chance:
            return {
                "type": "npc_noticed",
                "npc_uid": npc.get("npc_uid"),
                "narration": f"You notice {npc.get('name', 'someone')} acting unusually.",
            }

    return None


# ─── NPC Action Narration Filtering ──────────────────────────────────────────

def filter_npc_narration(
    npc_uid: str,
    npc_name: str,
    action_id: str,
    location: str,
    player_location: str,
    importance: int,
    gossip_delta: int = 0,
) -> dict[str, Any]:
    """
    Determine how to display an NPC's action to the player.

    Returns:
        dict with 'display_type' ('full', 'brief', 'meanwhile', 'hidden')
        and 'narration' text.
    """
    if location == player_location:
        narration = get_template_narration(
            action_id, "success", {"target": npc_name}
        )
        return {"display_type": "full", "narration": narration}

    if importance >= 3:
        return {
            "display_type": "brief",
            "narration": f"Nearby: {npc_name} is doing something noteworthy.",
        }

    if gossip_delta >= 5:
        return {
            "display_type": "brief",
            "narration": f"Rumor: Someone is talking about you to {npc_name}.",
        }

    return {
        "display_type": "meanwhile",
        "narration": f"{npc_name} goes about their business.",
    }
