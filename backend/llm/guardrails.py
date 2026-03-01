"""
guardrails.py — LLM output validation and sanitization.

Every piece of LLM output MUST pass through a validator before being
applied to the game state.  Functions return validated dicts or ``None``
on failure, never raising exceptions.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.config import UNIVERSAL_ACTION_IDS, logger

# ─── Valid enum sets ──────────────────────────────────────────────────────────

VALID_EMOTIONS: list[str] = [
    "neutral", "angry", "friendly", "fearful", "curious", "threatening",
]
VALID_SOCIAL: list[str] = [
    "neutral", "polite", "rude", "deceptive", "honest", "intimidating",
]

_VALID_EMOTION_SET: set[str] = set(VALID_EMOTIONS)
_VALID_SOCIAL_SET: set[str] = set(VALID_SOCIAL)
_VALID_ACTION_SET: set[str] = set(UNIVERSAL_ACTION_IDS)
_VALID_INFO_TYPES: set[str | None] = {
    None, "quest_hint", "location_info", "npc_info", "item_info",
}

# Patterns that suggest system-prompt leakage
_LEAKAGE_PATTERNS: list[str] = [
    r"(?i)system\s*prompt",
    r"(?i)you\s+are\s+a\s+(language|AI)\s+model",
    r"(?i)as\s+an?\s+AI",
    r"(?i)instructions?:\s",
    r"(?i)ignore\s+(all\s+)?previous",
]


# ─── Utility helpers ──────────────────────────────────────────────────────────

def clamp(value: int | float, min_val: int | float, max_val: int | float) -> int | float:
    """Clamp *value* between *min_val* and *max_val*, preserving type.

    Args:
        value: The value to clamp.
        min_val: Lower bound.
        max_val: Upper bound.

    Returns:
        Clamped value with the same numeric type as *value*.
    """
    clamped = max(min_val, min(max_val, value))
    return int(clamped) if isinstance(value, int) else float(clamped)


def strip_think_blocks(text: str) -> str:
    """Remove chain-of-thought reasoning from LLM output.

    Handles multiple patterns:
      1. Explicit ``<think>...</think>`` blocks (Qwen3 tagged mode)
      2. Unclosed ``<think>...`` blocks
      3. Untagged reasoning that Qwen3 sometimes emits before the actual
         response (detected by heuristic patterns)

    Args:
        text: Raw text from the LLM.

    Returns:
        Text with all reasoning content removed.
    """
    # 1. Closed <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. Unclosed <think> to end
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)

    # 3. Untagged reasoning heuristic for Qwen3.
    #    When the model ignores /no_think, it may emit free-form reasoning
    #    like "So, just the text. So, the user is playing..." before the
    #    actual response.  We detect this by looking for the pattern where
    #    reasoning precedes the real content.
    #
    #    Strategy: if the text contains a clear reasoning→response boundary,
    #    extract only the response.  Common boundaries:
    #      - "..." followed by a sentence starting with a capital letter
    #        that looks like actual narration (starts with You/The/A/An/")
    #      - Multiple "So," / "Okay," / "Let me" / "I need" preambles
    #    We look for the LAST occurrence of a reasoning prefix pattern
    #    followed by actual game content.
    if _looks_like_reasoning(cleaned):
        extracted = _extract_actual_response(cleaned)
        if extracted:
            cleaned = extracted

    return cleaned.strip()


def _looks_like_reasoning(text: str) -> bool:
    """Return True if text appears to start with LLM reasoning preamble."""
    # Common Qwen3 untagged reasoning starters
    reasoning_starts = (
        "so,", "okay,", "ok,", "let me", "i need to", "i will",
        "i should", "the user", "the player", "first,", "now,",
        "alright", "hmm", "well,", "let's", "i'll",
        "so just", "just the", "the action", "the narration",
        "the time", "the weather", "the outcome",
    )
    lower = text.lstrip().lower()
    return any(lower.startswith(s) for s in reasoning_starts)


def _extract_actual_response(text: str) -> str | None:
    """Try to extract the actual response from text that starts with reasoning.

    Looks for the transition from meta-reasoning to actual game content.
    Returns the extracted content, or None if no clear boundary is found.
    """
    # Pattern: find sentences starting with typical narration openers
    # after reasoning junk.  Look for the last "..." or "." boundary
    # followed by a narration-like sentence.
    narration_pattern = re.compile(
        r'(?:^|[.!?…]\s+)'
        r'((?:You |The |A |An |"|\'|Dawn |Dusk |Morning |Night |Sunlight |Moonlight |'
        r'Shadows |Torchlight |Rain |Wind |Fog |Snow |Storm |'
        r'Your |In the |At the |Around |Nearby |Here )'
        r'[A-Z][^.!?]*[.!?])'
        r'(.*)',
        re.DOTALL
    )
    match = narration_pattern.search(text)
    if match:
        result = (match.group(1) + (match.group(2) or "")).strip()
        # Only accept if it's substantial enough (not just a fragment)
        if len(result) >= 20:
            return result

    # Fallback: split on "..." (ellipsis often ends reasoning)
    if "..." in text:
        parts = text.split("...")
        # Take everything after the last ellipsis
        after = parts[-1].strip()
        if len(after) >= 20 and not _looks_like_reasoning(after):
            return after

    # Fallback: split on the last paragraph break
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        # Take the last paragraph if it looks like narration
        last = paragraphs[-1]
        if len(last) >= 20 and not _looks_like_reasoning(last):
            return last

    return None


def sanitize_text(text: str, max_length: int = 1200) -> str:
    """Strip HTML tags, system prompt artifacts, and truncate.

    Args:
        text: Raw text from LLM output.
        max_length: Maximum character length (default 1200 for narration).

    Returns:
        Cleaned text string.
    """
    # Strip <think>...</think> reasoning blocks (must come before HTML strip)
    cleaned = strip_think_blocks(text)

    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # Strip system prompt leakage
    for pattern in _LEAKAGE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    # Collapse excess whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # Truncate to max_length, ending at a sentence boundary if possible
    if len(cleaned) > max_length:
        # Try to cut at last sentence end within limit
        truncated = cleaned[:max_length]
        last_period = max(truncated.rfind('. '), truncated.rfind('! '), truncated.rfind('? '))
        if last_period > max_length * 0.6:
            cleaned = truncated[:last_period + 1]
        else:
            cleaned = truncated.rstrip() + "..."

    return cleaned


def parse_json_response(raw: str) -> dict | None:
    """Parse a JSON dict from raw LLM output.

    Handles common LLM quirks:
      * Markdown code-block wrapping (``\\`\\`\\`json ... \\`\\`\\```)
      * Trailing commas before closing braces/brackets
      * Leading/trailing whitespace

    Args:
        raw: Raw string response from the LLM.

    Returns:
        Parsed dict, or ``None`` on failure.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip <think>...</think> reasoning blocks before JSON parsing
    text = strip_think_blocks(text)

    # Strip markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    # If there are multiple JSON blocks, take the first one
    # Look for the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        logger.warning("LLM JSON parsed but is not a dict: %s", type(result).__name__)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("LLM JSON parse failed: %s", exc)
        return None


# ─── Domain validators ───────────────────────────────────────────────────────

def validate_checkpoint_output(raw: str) -> dict | None:
    """Validate LLM output for dynamic checkpoint generation.

    Schema requirements:
      * ``description``: str, 10–500 chars
      * ``highlighted_actions``: list of valid action IDs
      * ``effects``: dict (optional numeric values clamped)
      * ``hint``: str

    Args:
        raw: Raw LLM response string.

    Returns:
        Validated dict ready for use, or ``None`` on failure.
    """
    data = parse_json_response(raw)
    if data is None:
        logger.warning("Checkpoint validation: JSON parse failed")
        return None

    # --- description ---
    description = data.get("description")
    if not isinstance(description, str):
        logger.warning("Checkpoint validation: 'description' is not a string")
        return None
    description = sanitize_text(description)
    if len(description) < 10:
        logger.warning("Checkpoint validation: description too short (%d chars)", len(description))
        return None
    if len(description) > 500:
        description = description[:500]

    # --- highlighted_actions ---
    actions = data.get("highlighted_actions")
    if not isinstance(actions, list):
        logger.warning("Checkpoint validation: 'highlighted_actions' is not a list")
        return None
    valid_actions = [a for a in actions if isinstance(a, str) and a in _VALID_ACTION_SET]
    if not valid_actions:
        logger.warning("Checkpoint validation: no valid actions in highlighted_actions")
        return None

    # --- effects ---
    effects = data.get("effects", {})
    if not isinstance(effects, dict):
        effects = {}
    clamped_effects: dict[str, int] = {}
    effect_ranges: dict[str, tuple[int, int]] = {
        "health": (-50, 50),
        "stamina": (-20, 20),
        "reputation": (-30, 30),
    }
    for key, (lo, hi) in effect_ranges.items():
        if key in effects:
            val = effects[key]
            if isinstance(val, (int, float)):
                clamped_effects[key] = int(clamp(val, lo, hi))

    # --- hint ---
    hint = data.get("hint", "")
    if not isinstance(hint, str):
        hint = ""
    hint = sanitize_text(hint)

    return {
        "description": description,
        "highlighted_actions": valid_actions,
        "effects": clamped_effects,
        "hint": hint,
    }


def validate_dialogue_output(raw: str) -> dict | None:
    """Validate LLM output for NPC dialogue generation.

    Schema requirements:
      * ``dialogue``: str, max 500 chars
      * ``mood_change``: int, -3 to +3
      * ``reveals_info``: bool
      * ``info_type``: str | None (from valid info types)

    Args:
        raw: Raw LLM response string.

    Returns:
        Validated dict, or ``None`` on failure.
    """
    data = parse_json_response(raw)
    if data is None:
        logger.warning("Dialogue validation: JSON parse failed")
        return None

    # --- dialogue ---
    dialogue = data.get("dialogue")
    if not isinstance(dialogue, str) or not dialogue.strip():
        logger.warning("Dialogue validation: 'dialogue' is missing or empty")
        return None
    dialogue = sanitize_text(dialogue)

    # --- mood_change ---
    mood_change = data.get("mood_change", 0)
    if not isinstance(mood_change, (int, float)):
        mood_change = 0
    mood_change = int(clamp(mood_change, -3, 3))

    # --- reveals_info ---
    reveals_info = data.get("reveals_info", False)
    if not isinstance(reveals_info, bool):
        reveals_info = bool(reveals_info)

    # --- info_type ---
    info_type = data.get("info_type")
    if info_type is not None and info_type not in _VALID_INFO_TYPES:
        info_type = None

    return {
        "dialogue": dialogue,
        "mood_change": mood_change,
        "reveals_info": reveals_info,
        "info_type": info_type,
    }


def validate_input_analysis(raw: str) -> dict | None:
    """Validate LLM output for free-text input analysis.

    Schema requirements:
      * ``emotion``: one of VALID_EMOTIONS
      * ``intent``: str
      * ``social``: one of VALID_SOCIAL
      * ``matched_action``: valid action ID or ``"UNKNOWN"``
      * ``confidence``: float 0.0–1.0

    Args:
        raw: Raw LLM response string.

    Returns:
        Validated dict, or ``None`` on failure.
    """
    data = parse_json_response(raw)
    if data is None:
        logger.warning("Input analysis validation: JSON parse failed")
        return None

    # --- emotion ---
    emotion = data.get("emotion", "neutral")
    if not isinstance(emotion, str) or emotion.lower() not in _VALID_EMOTION_SET:
        emotion = "neutral"
    else:
        emotion = emotion.lower()

    # --- intent ---
    intent = data.get("intent", "")
    if not isinstance(intent, str):
        intent = ""
    intent = sanitize_text(intent)

    # --- social ---
    social = data.get("social", "neutral")
    if not isinstance(social, str) or social.lower() not in _VALID_SOCIAL_SET:
        social = "neutral"
    else:
        social = social.lower()

    # --- matched_action ---
    matched_action = data.get("matched_action", "UNKNOWN")
    if not isinstance(matched_action, str):
        matched_action = "UNKNOWN"
    if matched_action != "UNKNOWN" and matched_action not in _VALID_ACTION_SET:
        logger.debug(
            "Input analysis: invalid action '%s', setting to UNKNOWN",
            matched_action,
        )
        matched_action = "UNKNOWN"

    # --- confidence ---
    confidence = data.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = float(clamp(confidence, 0.0, 1.0))

    # --- interpreted_intent ---
    interpreted = data.get("interpreted_intent", intent)
    if not isinstance(interpreted, str):
        interpreted = intent
    interpreted = sanitize_text(interpreted)

    return {
        "emotion": emotion,
        "intent": intent,
        "social": social,
        "matched_action": matched_action,
        "confidence": confidence,
        "interpreted_intent": interpreted,
    }


def validate_action_decomposition(raw: str) -> dict | None:
    """Validate LLM output for action decomposition.

    Schema requirements:
      * ``steps``: list of 1–4 dicts, each with ``action_id`` in the
        universal catalog, plus optional ``target_npc``, ``target_item``,
        ``target_location``, ``description``.
      * ``interpretation``: str

    Args:
        raw: Raw LLM response string.

    Returns:
        Validated dict with ``steps`` and ``interpretation``, or ``None``.
    """
    data = parse_json_response(raw)
    if data is None:
        logger.warning("Action decomposition validation: JSON parse failed")
        return None

    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        logger.warning("Action decomposition: 'steps' missing or empty")
        return None

    validated_steps: list[dict] = []
    for step in steps[:4]:  # cap at 4
        if not isinstance(step, dict):
            continue
        action_id = step.get("action_id", "")
        if not isinstance(action_id, str) or action_id not in _VALID_ACTION_SET:
            logger.debug("Decomposition step has invalid action_id=%r", action_id)
            continue

        validated_steps.append({
            "action_id": action_id,
            "target_npc": step.get("target_npc") if isinstance(step.get("target_npc"), str) else None,
            "target_item": step.get("target_item") if isinstance(step.get("target_item"), str) else None,
            "target_location": step.get("target_location") if isinstance(step.get("target_location"), str) else None,
            "description": sanitize_text(str(step.get("description", "")), max_length=200),
        })

    if not validated_steps:
        logger.warning("Action decomposition: no valid steps after validation")
        return None

    interpretation = data.get("interpretation", "")
    if not isinstance(interpretation, str):
        interpretation = ""
    interpretation = sanitize_text(interpretation, max_length=300)

    return {
        "steps": validated_steps,
        "interpretation": interpretation,
    }
