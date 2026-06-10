"""NPC dialogue pipeline.

Resolution order: scripted → LLM → archetype generic → hardcoded fallback.
The earlier steps return None when they have nothing to say so the next
step gets a turn.
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

# Seeded so identical (seed, action) sequences produce identical fallback prose.
_rng = _random.Random(MASTER_SEED)


def resolve_dialogue(
    npc: NPC,
    action_id: str,
    player_input: str | None,
    emotion: str,
    social: str,
    context: dict,
    llm_service: LLMService | None = None,
) -> dict:
    """Run scripted → LLM → generic in sequence and return the first hit."""
    scripted = _check_scripted(npc, action_id, context)
    if scripted is not None:
        logger.debug("Scripted dialogue hit for %s / %s", npc.npc_uid, action_id)
        return _build_result(scripted, mood_change=0, reveals_info=False)

    llm_result = _try_llm(npc, action_id, player_input, emotion, social, context, llm_service)
    if llm_result is not None:
        return _build_result(
            llm_result["dialogue"],
            mood_change=llm_result.get("mood_change", 0),
            reveals_info=llm_result.get("reveals_info", False),
            info_type=llm_result.get("info_type"),
        )

    fallback = _generic_fallback(npc, action_id, emotion, social)
    logger.debug("Generic fallback for %s / %s", npc.npc_uid, action_id)
    return _build_result(fallback, mood_change=0, reveals_info=False)


def format_dialogue(npc_name: str, text: str) -> str:
    return f'{npc_name}: "{text}"'


def _check_scripted(npc: NPC, action_id: str, context: dict) -> str | None:
    """Return scripted text for this interaction, or None to let later layers respond."""
    player_rep = context.get("player_reputation", 0)
    if player_rep <= -50 and "hostile" in npc.scripted_dialogue:
        return npc.scripted_dialogue["hostile"]

    key_map: dict[str, str] = {
        "greet": "greeting",
        "talk": "greeting",
        "ask_info": "quest_hint",
        "present_item": "present_item",
        "give_item": "give_item",
    }
    key = key_map.get(action_id)

    # The scripted "greeting" line is the NPC's opener — using it on every
    # subsequent talk would make them sound stuck on repeat, so we cede to
    # the LLM / generic layers once any conversation has happened.
    if key == "greeting":
        if len(npc.conversation_history) > 0:
            return None
        if key in npc.scripted_dialogue:
            return npc.scripted_dialogue[key]
        return None

    if key == "quest_hint":
        stage_hint = _get_quest_stage_dialogue(npc, context)
        if stage_hint is not None:
            return stage_hint
        if key in npc.scripted_dialogue:
            return npc.scripted_dialogue[key]
        return None

    if key and key in npc.scripted_dialogue:
        return npc.scripted_dialogue[key]

    if action_id in npc.scripted_dialogue:
        return npc.scripted_dialogue[action_id]

    return None


# (npc_uid, stage) → line spoken when player uses ask_info at that stage.
# Missing entries fall back to the closest earlier stage entry for the NPC.
_QUEST_STAGE_DIALOGUE: dict[tuple[str, int], str] = {
    ("elder_m8b2", 1): (
        "You've come at a troubled time. Once you've settled in, come speak "
        "with me at my home. There is something I must ask of you."
    ),
    ("elder_m8b2", 2): (
        "I've been expecting you. Come closer — there is something weighing "
        "heavily on my heart, and I believe you may be the one to help."
    ),
    ("elder_m8b2", 3): (
        "My family's ancestral jade amulet — passed down for seven generations "
        "— has gone missing from its display case. It is more than a trinket; "
        "it is the heart of Thornhaven's heritage. Farmer Jak mentioned seeing "
        "a figure near the old oak in the fields at night. Will you investigate?"
    ),
    ("elder_m8b2", 4): (
        "Have you found anything yet? Jak's fields — near the old oak tree — "
        "that's where the clues may lie. Please, time is of the essence."
    ),
    ("elder_m8b2", 5): (
        "What news do you bring? I can see it in your eyes — you've discovered "
        "something. Tell me everything."
    ),
    ("elder_m8b2", 6): (
        "You've returned... and is that...? Could it be? Please, let me see it!"
    ),

    ("guard_a3f1", 3): (
        "The elder's been asking about her jade amulet. If someone stole it, "
        "they had to pass through this gate. I've been watching, but I might "
        "have missed something at night."
    ),
    ("guard_a3f1", 4): (
        "Any leads on the missing artifact? I've doubled my patrols near the "
        "gate, but nothing suspicious so far during daylight hours."
    ),

    ("guard_b7e2", 3): (
        "I heard about the elder's missing heirloom. Keep your eyes open — "
        "whoever took it is still in the village, or they slipped past at night."
    ),

    ("farmer_j4a1", 3): (
        "I told the elder already — I saw a shadow moving near the old oak "
        "tree in the dead of night. Couldn't make out who it was, but they "
        "were carrying something. The soil was disturbed there too."
    ),
    ("farmer_j4a1", 4): (
        "Did you check near the old oak? I swear I saw something buried "
        "there recently. The ground was loose, like someone had been digging."
    ),

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

    # Stage 7: Elder Maren after the amulet is returned
    ("elder_m8b2", 7): (
        "You have done something remarkable today, traveler. "
        "Thornhaven owes you a debt that words cannot repay. "
        "Please — accept this shield as a token of our gratitude. "
        "It was forged by my late husband; may it protect you as you have protected us."
    ),
}


def _get_quest_stage_dialogue(npc: NPC, context: dict) -> str | None:
    quest_state = context.get("quest_state", {})
    current_stage = quest_state.get("current_stage")

    if current_stage is None:
        return None

    key = (npc.npc_uid, current_stage)
    if key in _QUEST_STAGE_DIALOGUE:
        return _QUEST_STAGE_DIALOGUE[key]

    # Fall back to the most recent earlier stage so NPCs without late-game
    # lines keep saying their last known line instead of going silent.
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
    """Generate dialogue via LLM. Returns None if unavailable or all retries fail validation."""
    if llm_service is None or not llm_service.available:
        return None

    quest_state = context.get("quest_state", {})
    location = context.get("location", "unknown")
    turn = context.get("turn", 0)
    time_of_day = context.get("time_of_day", "morning")

    context_summary = (
        f"Location: {location}. Turn: {turn}. Time: {time_of_day}. "
        f"Quest stage: {quest_state.get('current_stage', 'unknown')}."
    )
    # Quest-situation summary — pending demands, item availability, gate target.
    # Without this the NPC has no idea what the quest expects of the player
    # right now and reverts to bland pleasantries.
    situation = context.get("quest_situation")
    if situation:
        context_summary += f"\n\n## Quest Situation\n{situation}"

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
        happiness=int(happiness),
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
    """Pick a line from the NPC's archetype response bank, narrowing by emotion when possible."""
    key_map: dict[str, str] = {
        "ask_info": "ask_info",
        "present_item": "present_item",
    }
    key = key_map.get(action_id, action_id)

    # Greeting → "talk" after first contact; the "greeting" bank reads as a
    # cold open and would feel jarring mid-conversation.
    if action_id in ("greet", "talk"):
        has_history = len(npc.conversation_history) > 0
        key = "talk" if has_history else "greeting"

    emotion_key = f"{key}_{emotion}"
    responses = npc.generic_responses.get(emotion_key)
    if not responses:
        responses = npc.generic_responses.get(key)
    if not responses:
        responses = npc.generic_responses.get("unknown")
    if not responses:
        return _hardcoded_fallback(npc.name, action_id, emotion, social)

    return _rng.choice(responses)


def _hardcoded_fallback(npc_name: str, action_id: str, emotion: str, social: str) -> str:
    """Final-tier dialogue: emotion/social-aware canned lines when even the archetype bank is empty."""
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
    return {
        "dialogue": dialogue,
        "mood_change": mood_change,
        "reveals_info": reveals_info,
        "info_type": info_type,
        "reputation_change": reputation_change,
    }
