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
    fallback = _generic_fallback(npc, action_id, emotion, social)
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
      * Only use ``"greeting"`` on the FIRST interaction (no conversation history).
      * ``ask_info`` → quest-stage-aware hint or ``"quest_hint"``
      * If NPC reputation toward player is hostile → ``"hostile"`` override.
      * If NPC has been talked to before, skip greeting to allow fallback variety.

    Returns:
        Scripted text or ``None`` if no match.
    """
    # Hostile override — always applies
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

    # For greet/talk → "greeting": only use on FIRST conversation.
    # After the first exchange, return None to let LLM or generic fallback
    # produce varied responses.
    if key == "greeting":
        conversation_count = len(npc.conversation_history)
        if conversation_count > 0:
            # Already spoken before — skip scripted greeting
            return None
        if key in npc.scripted_dialogue:
            return npc.scripted_dialogue[key]
        return None

    # For ask_info: try quest-stage-aware dialogue first
    if key == "quest_hint":
        stage_hint = _get_quest_stage_dialogue(npc, context)
        if stage_hint is not None:
            return stage_hint
        # Fall back to generic quest_hint
        if key in npc.scripted_dialogue:
            return npc.scripted_dialogue[key]
        return None

    # For other keys (present_item, etc.): always return if available
    if key and key in npc.scripted_dialogue:
        return npc.scripted_dialogue[key]

    # Direct key match (e.g., custom scripted keys)
    if action_id in npc.scripted_dialogue:
        return npc.scripted_dialogue[action_id]

    return None


# ── Quest-stage-aware dialogue ────────────────────────────────────────────────

# Per-NPC, per-quest-stage scripted dialogue lines.
# Key format: (npc_uid, stage_number) → dialogue string
_QUEST_STAGE_DIALOGUE: dict[tuple[str, int], str] = {
    # Elder Maren — Stage 1 (Arrival): player hasn't reached her yet
    ("elder_m8b2", 1): (
        "You've come at a troubled time. Once you've settled in, come speak "
        "with me at my home. There is something I must ask of you."
    ),
    # Stage 2 (Seeking the Elder): player is looking for the Elder
    ("elder_m8b2", 2): (
        "I've been expecting you. Come closer — there is something weighing "
        "heavily on my heart, and I believe you may be the one to help."
    ),
    # Stage 3 (The Missing Artifact): Elder reveals the quest
    ("elder_m8b2", 3): (
        "My family's ancestral jade amulet — passed down for seven generations "
        "— has gone missing from its display case. It is more than a trinket; "
        "it is the heart of Thornhaven's heritage. Farmer Jak mentioned seeing "
        "a figure near the old oak in the fields at night. Will you investigate?"
    ),
    # Stage 4 (Investigation): player is investigating
    ("elder_m8b2", 4): (
        "Have you found anything yet? Jak's fields — near the old oak tree — "
        "that's where the clues may lie. Please, time is of the essence."
    ),
    # Stage 5+: later stages
    ("elder_m8b2", 5): (
        "What news do you bring? I can see it in your eyes — you've discovered "
        "something. Tell me everything."
    ),
    ("elder_m8b2", 6): (
        "We are so close to resolving this. I have faith in you, traveler."
    ),

    # Guard Aldric — aware of quest
    ("guard_a3f1", 3): (
        "The elder's been asking about her jade amulet. If someone stole it, "
        "they had to pass through this gate. I've been watching, but I might "
        "have missed something at night."
    ),
    ("guard_a3f1", 4): (
        "Any leads on the missing artifact? I've doubled my patrols near the "
        "gate, but nothing suspicious so far during daylight hours."
    ),

    # Guard Bryn
    ("guard_b7e2", 3): (
        "I heard about the elder's missing heirloom. Keep your eyes open — "
        "whoever took it is still in the village, or they slipped past at night."
    ),

    # Farmer Jak — key witness
    ("farmer_j4a1", 3): (
        "I told the elder already — I saw a shadow moving near the old oak "
        "tree in the dead of night. Couldn't make out who it was, but they "
        "were carrying something. The soil was disturbed there too."
    ),
    ("farmer_j4a1", 4): (
        "Did you check near the old oak? I swear I saw something buried "
        "there recently. The ground was loose, like someone had been digging."
    ),

    # Tessa — hears tavern gossip
    ("tavkeeper_t9c3", 3): (
        "Everyone's been whispering about the elder's jade amulet. A stranger "
        "passed through the tavern two nights ago — kept to himself, paid in "
        "odd coin. Left before dawn. Could be nothing, could be everything."
    ),
    ("tavkeeper_t9c3", 4): (
        "Still looking into the elder's problem? I asked around — nobody "
        "remembers that stranger's face clearly. But he asked about the "
        "old oak tree in the fields. Seemed very interested."
    ),

    # Old Petra — village rumors
    ("villager_c1d4", 2): (
        "Looking for the elder, dear? Her house is just up the path from "
        "the village center. She's been looking troubled lately — you'd be "
        "doing a kindness to visit her."
    ),
    ("villager_c1d4", 3): (
        "Oh, that poor Elder Maren. She's lost her jade amulet, you know — "
        "the one her grandmother wore. I've been having bad dreams about it. "
        "Fire and shadow, every night. It means something, I'm sure of it."
    ),
}


def _get_quest_stage_dialogue(npc: NPC, context: dict) -> str | None:
    """Look up quest-stage-aware scripted dialogue for an NPC.

    Args:
        npc: The NPC being spoken to.
        context: Game context dict with quest_state.

    Returns:
        Quest-stage-specific dialogue string, or ``None``.
    """
    quest_state = context.get("quest_state", {})
    current_stage = quest_state.get("current_stage")

    if current_stage is None:
        return None

    # Try exact stage match
    key = (npc.npc_uid, current_stage)
    if key in _QUEST_STAGE_DIALOGUE:
        return _QUEST_STAGE_DIALOGUE[key]

    # For higher stages, try the highest available stage for this NPC
    if current_stage > 1:
        for stage in range(current_stage - 1, 0, -1):
            fallback_key = (npc.npc_uid, stage)
            if fallback_key in _QUEST_STAGE_DIALOGUE:
                return _QUEST_STAGE_DIALOGUE[fallback_key]

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


def _generic_fallback(npc: NPC, action_id: str, emotion: str = "neutral", social: str = "neutral") -> str:
    """Pick a context-aware generic response from archetype data.

    Enhanced to consider emotion and social register. Falls back through
    ``{action_id}_{emotion}`` → ``{action_id}`` → ``"unknown"`` → hard-coded default.

    Args:
        npc: The NPC providing the response.
        action_id: The interaction action id.
        emotion: Detected emotion from player input.
        social: Social register of player input.

    Returns:
        A single dialogue string.
    """
    # Map action_id to the correct generic_responses key.
    # For "talk" after first conversation, use "talk" key (not "greeting").
    # For "greet" after first conversation, also use "talk" since the greeting is stale.
    key_map: dict[str, str] = {
        "ask_info": "ask_info",
        "present_item": "present_item",
    }
    key = key_map.get(action_id, action_id)

    # For greet/talk: use "talk" key for repeat conversations, "greeting" for first
    if action_id in ("greet", "talk"):
        has_history = len(npc.conversation_history) > 0
        key = "talk" if has_history else "greeting"

    # Try emotion-specific variant first (e.g., "greeting_friendly")
    emotion_key = f"{key}_{emotion}"
    responses = npc.generic_responses.get(emotion_key)

    if not responses:
        responses = npc.generic_responses.get(key)
    if not responses:
        responses = npc.generic_responses.get("unknown")
    if not responses:
        # Hard-coded last-resort responses based on action category
        return _hardcoded_fallback(npc.name, action_id, emotion, social)

    return _rng.choice(responses)


def _hardcoded_fallback(npc_name: str, action_id: str, emotion: str, social: str) -> str:
    """Produce a last-resort dialogue response using hardcoded templates.

    Used when neither scripted nor archetype generic responses are available.
    Considers the player's emotion and social register.

    Returns:
        A contextual fallback dialogue string.
    """
    # Emotion-based responses for talk/greet
    if emotion == "friendly" or social == "polite":
        responses = [
            f"*nods warmly* Thank you for your kind words.",
            "That's... appreciated. Truly.",
            "*a faint smile crosses their face* You're too kind.",
            "Well, that brightens the day a bit. Thank you.",
        ]
    elif emotion == "angry" or social == "rude":
        responses = [
            "*steps back* I don't appreciate that tone.",
            "Mind your manners, stranger. This is a peaceful village.",
            "*frowns deeply* Is that how they taught you to speak where you come from?",
            "I'd watch my words if I were you.",
        ]
    elif emotion == "curious":
        responses = [
            "Curious, are you? Well, I suppose there's no harm in asking.",
            "You've got questions. Everyone does when they first arrive.",
            "Hmm, I might know something about that. What exactly are you after?",
        ]
    elif emotion == "fearful":
        responses = [
            "Easy now. There's nothing to be afraid of here.",
            "*speaks in a calming tone* You're safe in the village, friend.",
            "You look troubled. What's got you spooked?",
        ]
    elif emotion == "threatening" or social == "intimidating":
        responses = [
            "*straightens up* Threats won't get you anywhere here.",
            "I'd think carefully before making enemies in this village.",
            "*narrows eyes* Is that a threat? Because it sounded like one.",
        ]
    elif action_id == "ask_info":
        responses = [
            "I'm not sure I can help with that. Try asking someone else.",
            "Hmm, I don't have much to say on that subject.",
            "You might want to speak with the elder about that.",
        ]
    else:
        responses = [
            "Hmm? What is it?",
            "I'm listening.",
            "Go on, then.",
            "Is there something you need?",
            "*acknowledges you with a nod*",
        ]

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
