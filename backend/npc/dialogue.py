"""
dialogue.py — Dialogue pipeline for NPC speech.

Resolution order:
  1. Scripted dialogue (exact match for context key)
  2. LLM generation (placeholder — returns ``None`` for now)
  3. Archetype generic-response fallback
"""

from __future__ import annotations

import random as _random

from backend.config import (
    MASTER_SEED,
    logger,
)
from backend.npc.npc import NPC

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
) -> dict:
    """Run the full dialogue pipeline for player→NPC interaction.

    Args:
        npc: The NPC being spoken to.
        action_id: The universal action id (``talk``, ``greet``, ``ask_info``, …).
        player_input: Raw text from the player (may be ``None`` for button input).
        emotion: Detected emotion category.
        social: Detected social register.
        context: Game context dict (quest state, location, turn, …).

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

    # 2. LLM placeholder — always returns None for now
    llm_text = _try_llm(npc, action_id, player_input, emotion, social, context)
    if llm_text is not None:
        return _build_result(llm_text, mood_change=0, reveals_info=False)

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
) -> str | None:
    """Placeholder for LLM-generated dialogue.

    Always returns ``None`` until Phase 6 integration.
    """
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
