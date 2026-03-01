"""
checkpoint.py — Dynamic checkpoint generation with LLM and template fallback.

When a player deviates from the expected quest path an ad-hoc checkpoint
is generated.  The pipeline tries the LLM first (Phase 6), then falls
back to one of four category-based templates.
"""

from __future__ import annotations

import random
import re

from backend.config import UNIVERSAL_ACTIONS, logger
from backend.quest.mdp import Checkpoint


# ─── Action category sets (built once at import time) ────────────────────────

_COMBAT_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "combat"
}
_EXPLORATION_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "exploration"
}
_SOCIAL_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "social"
}
_STEALTH_ACTIONS: set[str] = {
    aid for aid, meta in UNIVERSAL_ACTIONS.items() if meta["category"] == "stealth"
}

# ─── Checkpoint templates ────────────────────────────────────────────────────

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

# ─── Default placeholder values ──────────────────────────────────────────────

_TEMPLATE_DEFAULTS: dict[str, str] = {
    "npc_name": "a nearby villager",
    "discovery": "a hidden alcove beneath the old stones",
    "outcome": "The shadows swallow your movement, but you sense you are not alone",
    "quest_hint": "Perhaps you should return to the main path",
    "next_landmark": "the village center",
    "nudge_hint": "Something tugs at your instinct — you feel drawn back toward the quest",
}


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_dynamic_checkpoint(
    stage_id: int,
    action_id: str,
    context: dict,
    llm_service: object | None = None,
) -> Checkpoint:
    """Generate a dynamic checkpoint for an unexpected player action.

    Attempts LLM generation first (when *llm_service* is provided).
    Falls back to a category-based template on failure or absence of LLM.

    Args:
        stage_id: Current quest stage number.
        action_id: The action that triggered deviation.
        context: Game context dict.  Expected keys:
            ``checkpoint_id``, ``location``, ``npc_name``, ``nudge_target``.
        llm_service: Optional LLM service handle.

    Returns:
        A new :class:`Checkpoint` ready to be inserted into the MDP.
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


# ─── LLM generation (Phase 6 stub) ──────────────────────────────────────────

def _llm_checkpoint(
    stage_id: int,
    action_id: str,
    context: dict,
    cp_id: str,
    llm_service: object,
) -> Checkpoint | None:
    """Attempt LLM-based checkpoint generation.

    This is a placeholder that will be wired up during Phase 6 (LLM
    integration).  Currently always returns ``None`` so the template
    fallback is used.
    """
    return None


# ─── Template generation ─────────────────────────────────────────────────────

def _template_checkpoint(
    stage_id: int,
    action_id: str,
    context: dict,
    cp_id: str,
) -> Checkpoint:
    """Build a Checkpoint from a pre-defined category template.

    Template selection:
        combat actions   → ``unexpected_combat``
        exploration      → ``unexpected_explore``
        social           → ``unexpected_social``
        stealth          → ``unexpected_stealth``
        other            → random choice
    """
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
        completion_conditions=None,  # dynamic CPs resolve on any highlighted action
        rewards={},
        highlighted_actions=highlighted,
        next_checkpoint=nudge_target,
        hint=hint_text,
        is_dynamic=True,
        is_terminal=False,
        nudge_target=nudge_target,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _select_template_key(action_id: str) -> str:
    """Map an action ID to the corresponding template key."""
    if action_id in _COMBAT_ACTIONS:
        return "unexpected_combat"
    if action_id in _EXPLORATION_ACTIONS:
        return "unexpected_explore"
    if action_id in _SOCIAL_ACTIONS:
        return "unexpected_social"
    if action_id in _STEALTH_ACTIONS:
        return "unexpected_stealth"
    # Navigation / utility / unknown → pick at random (seeded)
    return random.choice(list(CHECKPOINT_TEMPLATES.keys()))


def _fill_template(template: str, context: dict) -> str:
    """Fill ``{placeholder}`` values in *template* from *context*.

    Missing keys are resolved from ``_TEMPLATE_DEFAULTS``.  If a key is
    still unresolved after merging, it is silently stripped.

    Supported placeholders: ``{npc_name}``, ``{discovery}``,
    ``{outcome}``, ``{quest_hint}``, ``{next_landmark}``, ``{nudge_hint}``.
    """
    merged: dict[str, str] = {
        **_TEMPLATE_DEFAULTS,
        **{k: str(v) for k, v in context.items() if isinstance(v, str)},
    }
    try:
        return template.format_map(merged)
    except KeyError as exc:
        logger.warning("Template fill — missing key: %s", exc)
        return re.sub(r"\{[^}]+\}", "", template)
