"""Dynamic checkpoint generation: LLM when available, category templates otherwise."""

from __future__ import annotations

import random
import re

from backend.config import LLM_MAX_RETRIES, UNIVERSAL_ACTIONS, logger
from backend.llm.guardrails import validate_checkpoint_output
from backend.llm.prompts import build_checkpoint_prompt
from backend.quest.mdp import Checkpoint


_COMBAT_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "combat"
}
_EXPLORATION_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "exploration"
}
# `talk` is its own UNIVERSAL_ACTIONS category — fold it into social so
# talk/greet/ask_info/persuade route to the social template instead of random.
_SOCIAL_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items()
    if meta["category"] in ("social", "talk")
}
_STEALTH_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "stealth"
}


CHECKPOINT_TEMPLATES: dict[str, dict] = {
    "unexpected_combat": {
        "description": "Your aggressive action draws attention. {npc_name} confronts you.",
        "highlighted_actions": ["attack", "persuade", "flee"],
        "effects": {"stamina": -5, "reputation": -3},
    },
    "unexpected_explore": {
        "description": "You wander off the path and discover {discovery}.",
        "highlighted_actions": ["examine", "search", "pick_up"],
        "effects": {"stamina": -3},
    },
    "unexpected_social": {
        "description": "You strike up a conversation with {npc_name}.",
        "highlighted_actions": ["ask_info", "trade", "talk", "move_to"],
        "effects": {"stamina": -1},
    },
    "unexpected_stealth": {
        "description": "You try to sneak around. {outcome}.",
        "highlighted_actions": ["sneak", "hide", "wait"],
        "effects": {"stamina": -4},
    },
}

_TEMPLATE_DEFAULTS: dict[str, str] = {
    "npc_name": "a nearby villager",
    "discovery": "a hidden alcove beneath the old stones",
    "outcome": "The shadows swallow your movement, but you sense you are not alone",
    "quest_hint": "Perhaps you should return to the main path",
    "next_landmark": "the village center",
    "nudge_hint": "Something tugs at your instinct — you feel drawn back toward the quest",
}


def generate_dynamic_checkpoint(
    stage_id: int,
    action_id: str,
    context: dict,
    llm_service: object | None = None,
) -> Checkpoint:
    """Build a dynamic checkpoint for an off-path action.

    Tries the LLM first; falls back to a category template on failure
    or when no LLM is available.
    """
    cp_id: str = context.get("checkpoint_id", f"{stage_id}_D0")

    if llm_service is not None:
        try:
            cp = _llm_checkpoint(stage_id, action_id, context, cp_id, llm_service)
            if cp is not None:
                logger.info("Dynamic CP %s generated via LLM", cp_id)
                return cp
        except Exception:
            logger.warning(
                "LLM checkpoint generation failed for %s; falling back to template",
                cp_id,
            )

    cp = _template_checkpoint(stage_id, action_id, context, cp_id)
    logger.info("Dynamic CP %s generated via template (%s)", cp_id, _select_template_key(action_id))
    return cp


def _llm_checkpoint(
    stage_id: int,
    action_id: str,
    context: dict,
    cp_id: str,
    llm_service: object,
) -> Checkpoint | None:
    """Ask the LLM for a checkpoint; return None if every retry fails validation."""
    if not hasattr(llm_service, "available") or not llm_service.available:
        return None

    location = context.get("location", "unknown")
    npc_name = context.get("npc_name", "")
    nudge_target = context.get("nudge_target", "")
    quest_stage_desc = context.get("stage_description", f"Quest stage {stage_id}")
    expected_next = context.get("expected_next", nudge_target or "continue the quest")
    health = context.get("health", 100)
    stamina = context.get("stamina", 50)
    reputation = context.get("reputation", 0)
    emotion = context.get("emotion", "neutral")
    social = context.get("social", "neutral")
    inventory_summary = context.get("inventory_summary", "various items")

    prompt = build_checkpoint_prompt(
        stage_desc=quest_stage_desc,
        player_action=action_id,
        emotion=emotion,
        social=social,
        location=location,
        health=health,
        stamina=stamina,
        reputation=reputation,
        expected_next=expected_next,
        inventory_summary=inventory_summary,
    )

    for attempt in range(LLM_MAX_RETRIES):
        raw = llm_service.generate(prompt, temperature=0.85)
        if raw is None:
            logger.debug("LLM checkpoint attempt %d returned None", attempt + 1)
            continue

        validated = validate_checkpoint_output(raw)
        if validated is None:
            logger.debug("LLM checkpoint validation failed on attempt %d", attempt + 1)
            continue

        logger.info("LLM checkpoint %s validated on attempt %d", cp_id, attempt + 1)

        return Checkpoint(
            checkpoint_id=cp_id,
            stage_id=stage_id,
            description=validated["description"],
            location=location,
            trigger={"action": action_id},
            completion_conditions=None,
            rewards={},
            highlighted_actions=validated["highlighted_actions"],
            next_checkpoint=nudge_target or None,
            hint=validated.get("hint", ""),
            is_dynamic=True,
            is_terminal=False,
            nudge_target=nudge_target or None,
        )

    logger.info("LLM checkpoint exhausted %d attempts for %s", LLM_MAX_RETRIES, cp_id)
    return None


def _template_checkpoint(
    stage_id: int,
    action_id: str,
    context: dict,
    cp_id: str,
) -> Checkpoint:
    """Build a checkpoint from the category-template matching `action_id`."""
    template_key = _select_template_key(action_id)
    template = CHECKPOINT_TEMPLATES[template_key]

    description = _fill_template(template["description"], context)
    highlighted = list(template["highlighted_actions"])

    nudge_target: str | None = context.get("nudge_target")
    hint_text = _fill_template("{nudge_hint}", context)

    return Checkpoint(
        checkpoint_id=cp_id,
        stage_id=stage_id,
        description=description,
        location=context.get("location", ""),
        trigger={"action": action_id},
        completion_conditions=None,
        rewards={},
        highlighted_actions=highlighted,
        next_checkpoint=nudge_target,
        hint=hint_text,
        is_dynamic=True,
        is_terminal=False,
        nudge_target=nudge_target,
    )


def _select_template_key(action_id: str) -> str:
    if action_id in _COMBAT_ACTIONS:
        return "unexpected_combat"
    if action_id in _EXPLORATION_ACTIONS:
        return "unexpected_explore"
    if action_id in _SOCIAL_ACTIONS:
        return "unexpected_social"
    if action_id in _STEALTH_ACTIONS:
        return "unexpected_stealth"
    return random.choice(list(CHECKPOINT_TEMPLATES.keys()))


def _fill_template(template: str, context: dict) -> str:
    """Substitute `{placeholder}` tokens; fall back to defaults; strip any left over."""
    merged: dict[str, str] = {
        **_TEMPLATE_DEFAULTS,
        **{k: str(v) for k, v in context.items() if isinstance(v, str)},
    }
    try:
        return template.format_map(merged)
    except KeyError as exc:
        logger.warning("Template fill — missing key: %s", exc)
        return re.sub(r"\{[^}]+\}", "", template)
