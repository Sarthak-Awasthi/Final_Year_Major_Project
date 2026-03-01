"""
dialogue.py — Dialogue pipeline for NPC speech.

Resolution order:
  1. Scripted dialogue (exact match for context key)
  2. LLM generation (via LLMService if available)
  3. Archetype generic-response fallback
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

from backend.config import (
    LLM_MAX_RETRIES,
    MASTER_SEED,
    logger,
)
from backend.llm.guardrails import validate_dialogue_output
from backend.llm.prompts import build_dialogue_prompt
from backend.npc.npc import NPC

if TYPE_CHECKING:
    from backend.llm.llm_service import LLMService

# Module-level seeded RNG for all dialogue randomness
_rng = _random.Random(MASTER_SEED)


# ── Public API ────────────────────────────────────────────────────────────────

def resolve_dialogue(
    npc: NPC,
    action_id: str,
    player_input: str | None,
    emotion: str,
    social: str,
    context: dict,
    llm_service: LLMService | None = None,
) -> dict:
    """Run the full dialogue pipeline for player→NPC interaction.

    Args:
        npc: The NPC being spoken to.
        action_id: The universal action id (``talk``, ``greet``, ``ask_info``, …).
        player_input: Raw text from the player (may be ``None`` for button input).
        emotion: Detected emotion category.
        social: Detected social register.
        context: Game context dict (quest state, location, turn, …).
        llm_service: Optional LLM service for dialogue generation.

    Returns:
        Dict with keys:
        ``dialogue``, ``mood_change``, ``reveals_info``,
        ``info_type``, ``reputation_change``.
    """
    # 1. Scripted check
    scripted = _check_scripted(npc, action_id, context)
    if scripted is not None:
        logger.debug("Scripted dialogue hit for %s / %s", npc.npc_uid, action_id)
        return _build_result(scripted, mood_change=0, reveals_info=False)

    # 2. LLM generation (if available)
    llm_result = _try_llm(npc, action_id, player_input, emotion, social, context, llm_service)
    if llm_result is not None:
        return _build_result(
            llm_result["dialogue"],
            mood_change=llm_result.get("mood_change", 0),
            reveals_info=llm_result.get("reveals_info", False),
            info_type=llm_result.get("info_type"),
        )

    # 3. Generic fallback
    fallback = _generic_fallback(npc, action_id)
    logger.debug("Generic fallback for %s / %s", npc.npc_uid, action_id)
    return _build_result(fallback, mood_change=0, reveals_info=False)


def format_dialogue(npc_name: str, text: str) -> str:
    """Wrap dialogue text with the NPC's name for display.

    Args:
        npc_name: Display name of the NPC.
        text: The dialogue line.

    Returns:
        Formatted string, e.g. ``'Elder Maren: "Ah, welcome."'``
    """
    return f'{npc_name}: "{text}"'


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_scripted(npc: NPC, action_id: str, context: dict) -> str | None:
    """Look up scripted dialogue for the current interaction.

    Matching logic:
      * ``greet`` / ``talk`` → key ``"greeting"``
      * ``ask_info`` → key ``"quest_hint"`` if quest-relevant context,
        else ``"ask_info"`` if present
      * If NPC reputation toward player is hostile → ``"hostile"`` override

    Returns:
        Scripted text or ``None`` if no match.
    """
    # Hostile override
    player_rep = context.get("player_reputation", 0)
    if player_rep <= -50 and "hostile" in npc.scripted_dialogue:
        return npc.scripted_dialogue["hostile"]

    # Map action to scripted key
    key_map: dict[str, str] = {
        "greet": "greeting",
        "talk": "greeting",
        "ask_info": "quest_hint",
        "present_item": "present_item",
    }
    key = key_map.get(action_id)
    if key and key in npc.scripted_dialogue:
        return npc.scripted_dialogue[key]

    # Direct key match (e.g., custom scripted keys)
    if action_id in npc.scripted_dialogue:
        return npc.scripted_dialogue[action_id]

    return None


def _try_llm(
    npc: NPC,
    action_id: str,
    player_input: str | None,
    emotion: str,
    social: str,
    context: dict,
    llm_service: LLMService | None = None,
) -> dict | None:
    """Attempt LLM-generated dialogue via LLMService.

    Builds a dialogue prompt, sends it to the LLM, validates the output,
    and returns a validated dict or ``None`` on failure. Retries up to
    LLM_RETRY_ATTEMPTS times before giving up.

    Returns:
        Validated dict with dialogue/mood_change/reveals_info/info_type,
        or ``None`` if LLM is unavailable or all attempts fail.
    """
    if llm_service is None or not llm_service.available:
        return None

    # Build context summary for the prompt
    quest_state = context.get("quest_state", {})
    location = context.get("location", "unknown")
    turn = context.get("turn", 0)
    time_of_day = context.get("time_of_day", "morning")

    context_summary = (
        f"Location: {location}. Turn: {turn}. Time: {time_of_day}. "
        f"Quest stage: {quest_state.get('current_stage', 'unknown')}."
    )

    # Map happiness int to mood string
    happiness = npc.stats.get("happiness", 5)
    if happiness < 4:
        mood = "unhappy"
    elif happiness <= 7:
        mood = "content"
    else:
        mood = "cheerful"

    player_rep = context.get("player_reputation", 0)

    prompt = build_dialogue_prompt(
        npc_name=npc.name,
        npc_uid=npc.npc_uid,
        npc_role=npc.archetype,
        archetype=npc.archetype,
        personality=npc.personality,
        mood=mood,
        happiness=happiness,
        reputation=player_rep,
        context=context_summary,
        conversation_history=npc.conversation_history,
        player_input=player_input or action_id,
        emotion=emotion,
        social=social,
    )

    for attempt in range(LLM_MAX_RETRIES):
        raw = llm_service.generate(prompt, temperature=0.7)
        if raw is None:
            logger.debug("LLM dialogue attempt %d returned None", attempt + 1)
            continue

        validated = validate_dialogue_output(raw)
        if validated is not None:
            logger.info(
                "LLM dialogue for %s validated on attempt %d",
                npc.npc_uid,
                attempt + 1,
            )
            return validated

        logger.debug("LLM dialogue validation failed on attempt %d", attempt + 1)

    logger.info("LLM dialogue exhausted %d attempts for %s", LLM_MAX_RETRIES, npc.npc_uid)
    return None


def _generic_fallback(npc: NPC, action_id: str) -> str:
    """Pick a random generic response from the archetype data.

    Falls back through ``action_id`` → ``"unknown"`` → hard-coded default.

    Args:
        npc: The NPC providing the response.
        action_id: The interaction action id.

    Returns:
        A single dialogue string.
    """
    # Try action_id as key first (e.g. "greeting" for "greet")
    key_map: dict[str, str] = {
        "greet": "greeting",
        "talk": "greeting",
        "ask_info": "ask_info",
        "present_item": "present_item",
    }
    key = key_map.get(action_id, action_id)

    responses = npc.generic_responses.get(key)
    if not responses:
        responses = npc.generic_responses.get("unknown")
    if not responses:
        return "..."

    return _rng.choice(responses)


def _build_result(
    dialogue: str,
    mood_change: int = 0,
    reveals_info: bool = False,
    info_type: str | None = None,
    reputation_change: int = 0,
) -> dict:
    """Construct the standard dialogue result dict."""
    return {
        "dialogue": dialogue,
        "mood_change": mood_change,
        "reveals_info": reveals_info,
        "info_type": info_type,
        "reputation_change": reputation_change,
    }
