"""Prompt templates for every LLM call.

Every builder runs its output through `truncate_to_budget` so the prompt
fits inside `LLM_MAX_PROMPT_TOKENS`.
"""

from __future__ import annotations

from backend.config import (
    LLM_CONVERSATION_CONTEXT,
    LLM_MAX_PROMPT_TOKENS,
    UNIVERSAL_ACTION_IDS,
    logger,
)


def estimate_tokens(text: str) -> int:
    """Approximate token count at 4 chars/token — close enough for English BPE."""
    return max(1, len(text) // 4)


def truncate_to_budget(prompt: str, max_tokens: int = LLM_MAX_PROMPT_TOKENS) -> str:
    """Shrink an over-budget prompt by progressively dropping bulky sections.

    Priority: trim conversation history → trim event log → drop inventory.
    Falls back to a hard char-count truncation as a last resort.
    """
    if estimate_tokens(prompt) <= max_tokens:
        return prompt

    lines = prompt.split("\n")
    result_lines: list[str] = []
    in_history = False
    history_lines: list[str] = []
    in_events = False
    event_lines: list[str] = []
    skip_inventory = False

    for line in lines:
        lower = line.lower().strip()

        if "conversation history" in lower or "recent conversation" in lower:
            in_history = True
            in_events = False
            result_lines.append(line)
            continue
        if "event log" in lower or "recent events" in lower:
            in_events = True
            in_history = False
            result_lines.append(line)
            continue
        if "inventory" in lower and ("items" in lower or ":" in lower):
            # Only drop inventory once we're already near budget — earlier
            # rejection would shed it unnecessarily on borderline prompts.
            if estimate_tokens("\n".join(result_lines)) > max_tokens * 0.8:
                skip_inventory = True
                result_lines.append("(inventory omitted for brevity)")
                continue

        if lower == "" or (lower.startswith("#") and not in_history and not in_events):
            if in_history:
                # 6 lines ≈ last 3 exchanges (player + npc turns alternate).
                trimmed = history_lines[-6:] if len(history_lines) > 6 else history_lines
                result_lines.extend(trimmed)
                history_lines = []
                in_history = False
            elif in_events:
                trimmed = event_lines[-3:] if len(event_lines) > 3 else event_lines
                result_lines.extend(trimmed)
                event_lines = []
                in_events = False
            result_lines.append(line)
            continue

        if in_history:
            history_lines.append(line)
        elif in_events:
            event_lines.append(line)
        elif skip_inventory:
            continue
        else:
            result_lines.append(line)

    if history_lines:
        trimmed = history_lines[-6:] if len(history_lines) > 6 else history_lines
        result_lines.extend(trimmed)
    if event_lines:
        trimmed = event_lines[-3:] if len(event_lines) > 3 else event_lines
        result_lines.extend(trimmed)

    prompt = "\n".join(result_lines)

    max_chars = max_tokens * 4
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
        logger.warning("Prompt hard-truncated to %d chars", max_chars)

    return prompt


def build_checkpoint_prompt(
    stage_desc: str,
    player_action: str,
    emotion: str,
    social: str,
    location: str,
    health: int,
    stamina: int,
    reputation: int,
    expected_next: str,
    inventory_summary: str,
) -> str:
    prompt = f"""You are a game master AI for a medieval village RPG.

## Task
The player deviated from the quest path. Generate a dynamic checkpoint that acknowledges their action and guides them back toward the main quest.

## Current Quest Stage
{stage_desc}

## Player State
- Location: {location}
- Health: {health}/100
- Stamina: {stamina}/50
- Reputation: {reputation}
- Emotion: {emotion}
- Social register: {social}
- Inventory: {inventory_summary}

## Deviation
The player performed: "{player_action}"
The expected next step was: "{expected_next}"

## Instructions
Create a checkpoint that:
1. Acknowledges what the player did
2. Provides an interesting consequence or discovery
3. Hints at returning to the main quest path
4. Lists 2-4 highlighted actions from the universal catalog

## Output Format (strict JSON)
{{
  "description": "A 1-3 sentence description of the new situation (10-500 chars)",
  "highlighted_actions": ["action_id_1", "action_id_2"],
  "effects": {{"health": 0, "stamina": -5, "reputation": 0}},
  "hint": "A subtle hint guiding the player back to the quest"
}}

Respond with ONLY the JSON object, no other text."""

    return truncate_to_budget(prompt)


def build_dialogue_prompt(
    npc_name: str,
    npc_uid: str,
    npc_role: str,
    archetype: str,
    personality: str,
    mood: str,
    happiness: int,
    reputation: int,
    context: str,
    conversation_history: list[dict],
    player_input: str,
    emotion: str,
    social: str,
) -> str:
    recent = conversation_history[-LLM_CONVERSATION_CONTEXT:]
    history_text = ""
    if recent:
        history_lines = []
        for entry in recent:
            # Engine stores history as {turn, action, player_text, npc_response};
            # other call-sites pass {role, text}. Handle both — the legacy
            # branch caused total amnesia when the engine format reached us.
            if "player_text" in entry or "npc_response" in entry:
                ptext = (entry.get("player_text") or "").strip()
                nresp = (entry.get("npc_response") or "").strip()
                action = entry.get("action", "")
                turn = entry.get("turn")
                turn_tag = f"t{turn}" if turn is not None else ""
                if ptext:
                    history_lines.append(f"  [{turn_tag} {action}] Player: {ptext}")
                elif action:
                    history_lines.append(f"  [{turn_tag}] Player: ({action})")
                if nresp:
                    history_lines.append(f"  [{turn_tag}] {npc_name}: {nresp}")
            else:
                role = entry.get("role", "unknown")
                text = entry.get("text", "")
                history_lines.append(f"  {role}: {text}")
        history_text = "\n".join(history_lines) if history_lines else "  (no prior conversation)"
    else:
        history_text = "  (no prior conversation)"

    if reputation >= 50:
        disposition = "trusting and warm"
    elif reputation >= 20:
        disposition = "friendly and open"
    elif reputation >= -19:
        disposition = "neutral and cautious"
    elif reputation >= -49:
        disposition = "suspicious and guarded"
    else:
        disposition = "hostile and dismissive"

    prompt = f"""You are roleplaying as {npc_name}, a {npc_role} in a village RPG.

## Character
- Name: {npc_name} (UID: {npc_uid})
- Archetype: {archetype}
- Personality: {personality}
- Current mood: {mood} (happiness: {happiness}/10)
- Disposition toward player: {disposition} (reputation: {reputation})

## Context
{context}

## Recent Conversation
{history_text}

## Current Interaction
The player ({emotion} tone, {social} manner) says/does: "{player_input}"

## Instructions
Respond in character as {npc_name}. Keep the response to 1-3 sentences.
Stay consistent with the Recent Conversation above — do NOT forget what you
have already asked of the player, what they have shown you, or what
unresolved demands sit between you. If you previously asked for something
(papers, payment, an answer) and the player has not provided it, hold your
ground rather than waving them past with empty pleasantries.
Consider your mood, personality, and relationship with the player.
If the player is rude or threatening, react accordingly.

## Output Format (strict JSON)
{{
  "dialogue": "Your in-character response (max 500 chars)",
  "mood_change": 0,
  "reveals_info": false,
  "info_type": null
}}

- mood_change: integer -3 to +3 (how this interaction changes YOUR mood)
- reveals_info: true if you share quest-relevant information
- info_type: null, or one of "quest_hint", "location_info", "npc_info", "item_info"

Respond with ONLY the JSON object, no other text."""

    return truncate_to_budget(prompt)


def build_input_analysis_prompt(
    player_text: str,
    location: str,
    npcs_present: list[str],
    highlighted_actions: list[str],
) -> str:
    """Prompt for parsing free-text input into (emotion, social, intent, action)."""
    npcs_str = ", ".join(npcs_present) if npcs_present else "nobody"
    highlighted_str = ", ".join(highlighted_actions) if highlighted_actions else "none"
    action_list = ", ".join(UNIVERSAL_ACTION_IDS)

    prompt = f"""You are a natural language parser for a medieval village RPG.

## Task
Analyze the player's free-text input and extract structured data.

## Context
- Location: {location}
- NPCs present: {npcs_str}
- Quest-highlighted actions: {highlighted_str}

## Available Actions
{action_list}

## Player Input
"{player_text}"

## Instructions
1. Determine the player's emotional tone
2. Determine their social register
3. Identify the intended game action from the available actions list
4. Extract any target (NPC, item, location) if mentioned

## Output Format (strict JSON)
{{
  "emotion": "neutral",
  "intent": "brief description of what the player wants to do",
  "social": "neutral",
  "matched_action": "action_id from the available actions list, or UNKNOWN",
  "confidence": 0.8,
  "interpreted_intent": "one-sentence natural language interpretation"
}}

Valid emotions: neutral, angry, friendly, fearful, curious, threatening
Valid social: neutral, polite, rude, deceptive, honest, intimidating

Respond with ONLY the JSON object, no other text."""

    return truncate_to_budget(prompt)


def build_narration_prompt(
    action_id: str,
    actor_name: str,
    target_name: str | None,
    outcome_type: str,
    template_text: str,
    location: str,
    time_of_day: str,
    weather: str | None,
    emotion: str,
    social: str,
    witnesses: list[str],
) -> str:
    """Prompt for LLM enrichment of a template narration line. Expects plain text back, not JSON."""
    target_str = target_name if target_name else "no specific target"
    weather_str = weather if weather else "clear"
    witnesses_str = ", ".join(witnesses) if witnesses else "nobody"
    location_readable = location.replace("_", " ").title()

    prompt = f"""You are a narrator for a medieval village RPG set in the village of Thornhaven.

## What Just Happened
The player performed "{action_id}" at {location_readable} during the {time_of_day}. Target: {target_str}. Outcome: {outcome_type}. Weather: {weather_str}.
{f'Nearby: {witnesses_str}.' if witnesses_str != 'nobody' else ''}

## Base Narration
"{template_text}"

## Your Task
Rewrite the base narration into vivid prose. You MUST include these specific details:
1. Name the location "{location_readable}" explicitly in the text
2. Reference the action "{action_id.replace('_', ' ')}" and its outcome
3. Mention the {time_of_day} time of day (light, shadows, atmosphere)
{f'4. Name at least one witness: {witnesses_str}' if witnesses_str != 'nobody' else ''}

RULES:
- Write EXACTLY 2-3 sentences
- Second person ("You ...") — never use the player's name "{actor_name}"
- No NPC dialogue, no stat numbers, no markdown
- No instructions or meta-commentary

Respond with ONLY the narration text."""

    return truncate_to_budget(prompt)


def build_action_decomposition_prompt(
    player_text: str,
    location: str,
    npcs_present: list[str],
    inventory_items: list[str],
    highlighted_actions: list[str],
) -> str:
    """Prompt that splits a compound command (e.g. 'show papers to guard') into atomic action steps."""
    npcs_str = ", ".join(npcs_present) if npcs_present else "nobody"
    items_str = ", ".join(inventory_items) if inventory_items else "nothing"
    highlighted_str = ", ".join(highlighted_actions) if highlighted_actions else "none"
    action_list = ", ".join(UNIVERSAL_ACTION_IDS)

    prompt = f"""You are an action planner for a medieval village RPG.

## Task
Break down the player's complex command into a sequence of 1-4 atomic game actions. Each step must use exactly one action from the available actions list.

## Context
- Location: {location}
- NPCs present: {npcs_str}
- Player inventory: {items_str}
- Quest-highlighted actions: {highlighted_str}

## Available Actions
{action_list}

## Player Command
"{player_text}"

## Rules
1. Each step is ONE atomic action from the list above
2. Include the target (NPC name, item name, or location) for each step
3. Order matters — steps execute sequentially
4. Maximum 4 steps
5. If the command is already a single action, return just 1 step
6. "show X to NPC" or "present X to NPC" = give_item (the NPC inspects and returns it)
7. "use X on NPC" = use_item targeting that NPC

## Output Format (strict JSON)
{{
  "steps": [
    {{"action_id": "give_item", "target_npc": "Aldric", "target_item": "travel_papers", "target_location": null, "description": "Present travel papers to Aldric"}},
    {{"action_id": "talk", "target_npc": "Aldric", "target_item": null, "target_location": null, "description": "Hear Aldric's response about the papers"}}
  ],
  "interpretation": "one-sentence summary of what the player wants to do"
}}

Each step must have: action_id (from available list), target_npc (name or null), target_item (name or null), target_location (name or null), description (short text).

Respond with ONLY the JSON object, no other text."""

    return truncate_to_budget(prompt)
