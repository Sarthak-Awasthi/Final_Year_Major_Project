"""
fallback.py — Template-based fallbacks for when LLM is unavailable.

Every LLM code path must have a working fallback.  This module provides
deterministic, template-driven alternatives that produce outputs
matching the same schemas as the LLM validators expect.
"""

from __future__ import annotations

import random as _random
from typing import Any

from backend.config import (
    ACTION_SYNONYMS,
    EMOTION_KEYWORDS,
    MASTER_SEED,
    SOCIAL_KEYWORDS,
    UNIVERSAL_ACTIONS,
    logger,
)

# Module-level seeded RNG
_rng = _random.Random(MASTER_SEED)

# ─── Action-category sets (built once) ────────────────────────────────────────

_COMBAT_IDS: set[str] = {
    aid for aid, m in UNIVERSAL_ACTIONS.items() if m["category"] == "combat"
}
_EXPLORATION_IDS: set[str] = {
    aid for aid, m in UNIVERSAL_ACTIONS.items() if m["category"] == "exploration"
}
_SOCIAL_IDS: set[str] = {
    aid for aid, m in UNIVERSAL_ACTIONS.items() if m["category"] == "social"
}
_STEALTH_IDS: set[str] = {
    aid for aid, m in UNIVERSAL_ACTIONS.items() if m["category"] == "stealth"
}

# ─── Checkpoint templates ─────────────────────────────────────────────────────

_CP_TEMPLATES: dict[str, dict[str, Any]] = {
    "unexpected_combat": {
        "description": "Your aggressive action draws attention. A confrontation unfolds around you.",
        "highlighted_actions": ["attack", "defend", "flee", "persuade"],
        "effects": {"health": 0, "stamina": -5, "reputation": -3},
        "hint": "Perhaps a less hostile approach would serve you better.",
    },
    "unexpected_explore": {
        "description": "You wander off the beaten path and discover something unexpected among the surroundings.",
        "highlighted_actions": ["examine", "search", "pick_up", "move_to"],
        "effects": {"health": 0, "stamina": -3, "reputation": 0},
        "hint": "Interesting find — but the village still needs your attention.",
    },
    "unexpected_social": {
        "description": "Your social overture leads to an unexpected exchange with a nearby villager.",
        "highlighted_actions": ["talk", "ask_info", "trade", "move_to"],
        "effects": {"health": 0, "stamina": -1, "reputation": 0},
        "hint": "This conversation might reveal something useful if you ask the right questions.",
    },
    "unexpected_stealth": {
        "description": "You slip into the shadows. The world carries on around you, unaware of your presence.",
        "highlighted_actions": ["sneak", "hide", "steal", "wait"],
        "effects": {"health": 0, "stamina": -4, "reputation": 0},
        "hint": "Stealth has its uses, but the direct path may be quicker.",
    },
}


def _categorise_action(action_id: str) -> str:
    """Map an action ID to its checkpoint template category."""
    if action_id in _COMBAT_IDS:
        return "unexpected_combat"
    if action_id in _EXPLORATION_IDS:
        return "unexpected_explore"
    if action_id in _SOCIAL_IDS:
        return "unexpected_social"
    if action_id in _STEALTH_IDS:
        return "unexpected_stealth"
    return _rng.choice(list(_CP_TEMPLATES.keys()))


def fallback_checkpoint(action_id: str, context: dict) -> dict:
    """Generate a dynamic checkpoint using templates.

    Args:
        action_id: The action that triggered deviation.
        context: Game context dict.  May contain ``location``,
            ``npc_name``, ``nudge_target`` for template customisation.

    Returns:
        Dict matching the checkpoint schema:
        ``{description, highlighted_actions, effects, hint}``.
    """
    key = _categorise_action(action_id)
    template = _CP_TEMPLATES[key]

    description = template["description"]
    hint = template["hint"]

    # Customise description with context if available
    npc_name = context.get("npc_name")
    location = context.get("location")
    if npc_name and "{npc_name}" not in description:
        description = description.rstrip(".") + f", near {npc_name}."
    if location:
        hint = hint.rstrip(".") + f" Consider heading toward {location}."

    nudge_target = context.get("nudge_target")
    if nudge_target:
        hint = hint.rstrip(".") + f" Your quest leads toward {nudge_target}."

    return {
        "description": description,
        "highlighted_actions": list(template["highlighted_actions"]),
        "effects": dict(template["effects"]),
        "hint": hint,
    }


# ─── Dialogue fallback ───────────────────────────────────────────────────────

# Action-to-response-key mapping
_DIALOGUE_KEY_MAP: dict[str, str] = {
    "talk": "greeting",
    "greet": "greeting",
    "ask_info": "ask_info",
}

# Reputation-based flavour prefixes
_REP_PREFIXES: dict[str, str] = {
    "trusted": "",
    "friendly": "",
    "neutral": "",
    "suspicious": "Hmph. ",
    "hostile": "*scowls* ",
}


def _rep_tier(reputation: int) -> str:
    """Map numeric reputation to tier label."""
    if reputation >= 50:
        return "trusted"
    if reputation >= 20:
        return "friendly"
    if reputation >= -19:
        return "neutral"
    if reputation >= -49:
        return "suspicious"
    return "hostile"


def fallback_dialogue(
    npc_name: str,
    archetype: str,
    action_id: str,
    reputation: int,
    generic_responses: dict[str, list[str]],
) -> dict:
    """Generate NPC dialogue using archetype generic responses.

    Args:
        npc_name: Display name of the NPC.
        archetype: NPC archetype key.
        action_id: The interaction action (talk, greet, ask_info, …).
        reputation: Player's reputation with this NPC (-100 to +100).
        generic_responses: Dict of response lists keyed by category
            (from archetype data).

    Returns:
        Dict matching the dialogue schema:
        ``{dialogue, mood_change, reveals_info, info_type}``.
    """
    # Map action to response key
    response_key = _DIALOGUE_KEY_MAP.get(action_id, "unknown")
    responses = generic_responses.get(response_key)

    if not responses:
        responses = generic_responses.get("unknown")
    if not responses:
        responses = ["..."]

    text = _rng.choice(responses)

    # Apply reputation-based prefix
    tier = _rep_tier(reputation)
    prefix = _REP_PREFIXES.get(tier, "")
    if prefix and not text.startswith(prefix):
        text = prefix + text

    return {
        "dialogue": text,
        "mood_change": 0,
        "reveals_info": False,
        "info_type": None,
    }


# ─── Input analysis fallback ─────────────────────────────────────────────────

def fallback_input_analysis(text: str) -> dict:
    """Analyse free-text input using keyword matching only.

    Scans for emotion keywords, social keywords, and action synonyms
    from the config module.

    Args:
        text: Raw player input string.

    Returns:
        Dict matching the input analysis schema:
        ``{emotion, intent, social, matched_action, confidence,
        interpreted_intent}``.
    """
    if not text or not text.strip():
        return {
            "emotion": "neutral",
            "intent": "",
            "social": "neutral",
            "matched_action": "UNKNOWN",
            "confidence": 0.0,
            "interpreted_intent": "",
        }

    lower = text.lower().strip()

    # ── Detect emotion ──
    detected_emotion = "neutral"
    best_emotion_score = 0
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_emotion_score:
            best_emotion_score = score
            detected_emotion = emotion

    # ── Detect social register ──
    detected_social = "neutral"
    best_social_score = 0
    for social, keywords in SOCIAL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_social_score:
            best_social_score = score
            detected_social = social

    # ── Match action ──
    matched_action = "UNKNOWN"
    best_action_score = 0
    confidence = 0.0

    for action_id, synonyms in ACTION_SYNONYMS.items():
        score = 0
        for synonym in synonyms:
            if synonym in lower:
                # Longer synonym matches are worth more
                score += len(synonym.split())
        if score > best_action_score:
            best_action_score = score
            matched_action = action_id
            # Confidence based on match quality
            confidence = min(1.0, score * 0.3)

    # Boost confidence for exact matches
    if matched_action != "UNKNOWN" and best_action_score >= 2:
        confidence = min(1.0, confidence + 0.2)

    return {
        "emotion": detected_emotion,
        "intent": lower,
        "social": detected_social,
        "matched_action": matched_action,
        "confidence": round(confidence, 2),
        "interpreted_intent": f"Player wants to {matched_action.replace('_', ' ')}"
        if matched_action != "UNKNOWN"
        else f"Unclear intent: {lower[:80]}",
    }


# ─── Narration fallback ──────────────────────────────────────────────────────

# Minimal template narrations when the full narration module is not wanted
_NARRATION_TEMPLATES: dict[str, dict[str, str]] = {
    "success": {
        "default": "You {action}. It goes well.",
        "move_to": "You make your way to {target}.",
        "look": "You look around carefully.",
        "search": "You search the area thoroughly.",
        "talk": "You speak with {target}.",
        "greet": "You greet {target}.",
        "ask_info": "You ask {target} for information.",
        "attack": "You strike at {target}!",
        "defend": "You raise your guard.",
        "flee": "You turn and run!",
        "rest": "You take a moment to rest.",
        "wait": "You wait and watch.",
    },
    "fail": {
        "default": "You try to {action}, but it doesn't work out.",
        "attack": "You swing at {target}, but miss.",
        "persuade": "{target} is unconvinced.",
        "deceive": "{target} sees through your deception.",
        "flee": "You try to flee, but can't escape!",
        "steal": "You fumble the attempt. Someone might have noticed.",
    },
    "blocked": {
        "default": "You can't do that right now.",
        "move_to": "You can't reach {target} from here.",
        "talk": "There's nobody around to talk to.",
        "attack": "There's no one here to attack.",
    },
}


def fallback_narration(action_id: str, context: dict) -> str:
    """Generate simple template narration text.

    Falls back to generic templates when the full narration module is
    not needed (e.g., during pre-training or when LLM enhancement is
    skipped).

    Args:
        action_id: The action being narrated.
        context: Dict with optional keys ``outcome_type``, ``target``,
            ``location``, ``time_of_day``.

    Returns:
        Narration text string.
    """
    outcome = context.get("outcome_type", "success")
    target = context.get("target", "someone")
    location = context.get("location", "here")
    time_of_day = context.get("time_of_day", "")

    templates = _NARRATION_TEMPLATES.get(outcome, _NARRATION_TEMPLATES["success"])
    template = templates.get(action_id, templates.get("default", "Something happens."))

    narration = template.format(
        action=action_id.replace("_", " "),
        target=target,
        location=location,
    )

    # Append time-of-day flavour
    time_flavour: dict[str, str] = {
        "morning": " The morning light casts long shadows.",
        "midday": " The midday sun beats down overhead.",
        "afternoon": " The afternoon wears on.",
        "evening": " Evening settles over the village.",
        "night": " Darkness surrounds you.",
    }
    if time_of_day in time_flavour:
        narration += time_flavour[time_of_day]

    return narration
