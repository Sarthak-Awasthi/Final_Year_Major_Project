"""
prompts.py — Prompt templates for all LLM use cases.

Every builder returns a fully-formed prompt string ready for the model.
Token estimation and budget truncation utilities are included so that
callers never exceed the 2 500-token hard prompt limit.
"""

from __future__ import annotations

from backend.config import (
    LLM_CONVERSATION_CONTEXT,
    LLM_MAX_PROMPT_TOKENS,
    UNIVERSAL_ACTION_IDS,
    logger,
)


# ─── Token estimation ────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token.

    This is a conservative heuristic that aligns well with typical BPE
    tokenizers for English text.

    Args:
        text: Input string.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def truncate_to_budget(prompt: str, max_tokens: int = LLM_MAX_PROMPT_TOKENS) -> str:
    """Trim a prompt to fit within the token budget.

    Truncation priority (applied in order until within budget):
      1. Conversation history → last 3 exchanges
      2. Event log → last 3 events
      3. Omit inventory details

    If the prompt is still over budget after all reductions, it is hard-
    truncated to ``max_tokens * 4`` characters (the inverse of our 4-
    chars-per-token estimate).

    Args:
        prompt: The original prompt string.
        max_tokens: Maximum allowed tokens.

    Returns:
        A prompt that fits within the budget.
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

        # Detect conversation history section
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
            # Check if we need to drop inventory
            if estimate_tokens("\n".join(result_lines)) > max_tokens * 0.8:
                skip_inventory = True
                result_lines.append("(inventory omitted for brevity)")
                continue

        # Reset section tracking on blank line or new header
        if lower == "" or (lower.startswith("#") and not in_history and not in_events):
            if in_history:
                # Keep only last 3 exchanges (6 lines: player + npc alternating)
                trimmed = history_lines[-6:] if len(history_lines) > 6 else history_lines
                result_lines.extend(trimmed)
                history_lines = []
                in_history = False
            elif in_events:
                # Keep only last 3 events
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

    # Flush any remaining section lines
    if history_lines:
        trimmed = history_lines[-6:] if len(history_lines) > 6 else history_lines
        result_lines.extend(trimmed)
    if event_lines:
        trimmed = event_lines[-3:] if len(event_lines) > 3 else event_lines
        result_lines.extend(trimmed)

    prompt = "\n".join(result_lines)

    # Hard-truncate as last resort
    max_chars = max_tokens * 4
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
        logger.warning("Prompt hard-truncated to %d chars", max_chars)

    return prompt


# ─── Prompt builders ──────────────────────────────────────────────────────────

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
    """Build a prompt for dynamic checkpoint generation.

    Args:
        stage_desc: Description of the current quest stage.
        player_action: The action that triggered deviation.
        emotion: Player's emotional tone.
        social: Player's social register.
        location: Current location name.
        health: Player health.
        stamina: Player stamina/AP.
        reputation: Average reputation value.
        expected_next: Description of the expected next checkpoint.
        inventory_summary: Brief inventory listing.

    Returns:
        Full prompt string for checkpoint generation.
    """
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
    """Build a prompt for NPC dialogue generation.

    Args:
        npc_name: Display name of the NPC.
        npc_uid: Unique identifier of the NPC.
        npc_role: Role description (e.g. 'village elder').
        archetype: Archetype key (e.g. 'elder').
        personality: Personality description string.
        mood: Current mood (e.g. 'content', 'anxious').
        happiness: Happiness stat (0-10).
        reputation: Player's reputation with this NPC (-100 to +100).
        context: Current game context summary.
        conversation_history: List of prior exchanges (dicts with 'role' and 'text').
        player_input: What the player said or did.
        emotion: Player's emotional tone.
        social: Player's social register.

    Returns:
        Full prompt string for dialogue generation.
    """
    # Truncate conversation history to last N entries
    recent = conversation_history[-LLM_CONVERSATION_CONTEXT:]
    history_text = ""
    if recent:
        history_lines = []
        for entry in recent:
            role = entry.get("role", "unknown")
            text = entry.get("text", "")
            history_lines.append(f"  {role}: {text}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "  (no prior conversation)"

    # Map reputation to disposition
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

    prompt = f"""You are roleplaying as {npc_name}, a {npc_role} in a medieval village.

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
    """Build a prompt for free-text input 3D analysis.

    Extracts emotion, intent, social register and maps to the universal
    action catalog.

    Args:
        player_text: Raw player input text.
        location: Current location name.
        npcs_present: List of NPC names present at the location.
        highlighted_actions: Actions currently highlighted by the quest.

    Returns:
        Full prompt string for input analysis.
    """
    npcs_str = ", ".join(npcs_present) if npcs_present else "nobody"
    highlighted_str = ", ".join(highlighted_actions) if highlighted_actions else "none"

    # Provide the full action catalog
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
    """Build a prompt for narrative text enhancement.

    The LLM enriches a template narration with context-aware details.

    Args:
        action_id: The action being narrated.
        actor_name: Who performed the action.
        target_name: Target of the action (may be None).
        outcome_type: Outcome category (success/fail/blocked/partial).
        template_text: The base template narration to enhance.
        location: Current location name.
        time_of_day: Current time period.
        weather: Current weather condition or None.
        emotion: Actor's emotional tone.
        social: Actor's social register.
        witnesses: List of witness NPC names.

    Returns:
        Full prompt string for narration enhancement.
        The expected output is plain text (not JSON).
    """
    target_str = target_name if target_name else "no specific target"
    weather_str = weather if weather else "clear"
    witnesses_str = ", ".join(witnesses) if witnesses else "nobody"

    prompt = f"""You are a narrator for a medieval village RPG. Enhance the following action narration with atmospheric details.

## Action Details
- Action: {action_id}
- Actor: {actor_name}
- Target: {target_str}
- Outcome: {outcome_type}
- Location: {location}
- Time: {time_of_day}
- Weather: {weather_str}
- Tone: {emotion}
- Manner: {social}
- Witnesses: {witnesses_str}

## Base Narration
"{template_text}"

## Instructions
Enhance the narrative with atmospheric details, focusing on the setting's ambiance and the actor's emotional state. Keep the outcome, time of day, and weather consistent.
Keep it to 2-3 sentences maximum.
Do NOT write NPC dialogue or speech — dialogue is handled separately.
Do NOT add game mechanics, stat changes, or action headers.
Do NOT echo back these instructions or action details.
Write in second person ("You...").

Respond with ONLY the enhanced narration text, nothing else."""

    return truncate_to_budget(prompt)


def build_action_decomposition_prompt(
    player_text: str,
    location: str,
    npcs_present: list[str],
    inventory_items: list[str],
    highlighted_actions: list[str],
) -> str:
    """Build a prompt to decompose complex free-text into atomic action steps.

    The LLM breaks down compound player intent (e.g. "show travel papers
    to the guard") into a sequence of game-engine-level atomic actions.

    Args:
        player_text: Raw player input text.
        location: Current location name.
        npcs_present: List of NPC display names present at the location.
        inventory_items: List of item names the player carries.
        highlighted_actions: Actions highlighted by the current quest checkpoint.

    Returns:
        Full prompt string for action decomposition.
    """
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
