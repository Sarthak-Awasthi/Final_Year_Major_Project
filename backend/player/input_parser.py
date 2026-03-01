"""
input_parser.py — NLP input parser for the MVP game.

Handles two input modes that both produce a unified ParsedInput dict:

1. **Action Palette** (button input) — direct 1:1 action mapping,
   defaults to neutral emotion/social.
2. **Free-Text Input** — NLP pipeline extracts 3 dimensions
   (emotion, intent/action, social), maps to universal action catalog.

Layered NLP Strategy:
    1. Primary:   Keyword / synonym dictionaries (fast, deterministic)
    2. Secondary:  spaCy ``doc.similarity()`` as tiebreaker
    3. Tertiary:   (stub) LLM for highest-accuracy 3D analysis
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np  # type: ignore[import-unresolved]

from backend.config import (
    ACTION_SYNONYMS,
    EMOTION_KEYWORDS,
    LOCATION_IDS,
    SOCIAL_KEYWORDS,
    UNIVERSAL_ACTION_IDS,
    UNIVERSAL_ACTIONS,
    logger,
)

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

# ─── Optional spaCy import ───────────────────────────────────────────────────
try:
    import spacy  # type: ignore[import-unresolved]

    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

# ─── Module-level state ──────────────────────────────────────────────────────
_nlp: Language | None = None
ACTION_VECTORS: dict[str, np.ndarray] = {}

# ─── ParsedInput type ────────────────────────────────────────────────────────

class ParsedInput(TypedDict, total=False):
    """Unified representation produced by both button and text input."""

    source: str            # "button" | "text"
    raw_text: str | None
    action_id: str | None  # from universal catalog (27 actions)
    target_npc: str | None  # resolved NPC UID
    target_item: str | None
    target_location: str | None
    confidence: float       # 1.0 for buttons, 0.0–1.0 for text
    emotion: str            # neutral/angry/friendly/fearful/curious/threatening
    intent: str             # free-form or action_id
    social: str             # neutral/polite/rude/deceptive/honest/intimidating
    queued_action: dict | None  # compound input: second action queued for next turn


# ─── Valid enum values ────────────────────────────────────────────────────────
_VALID_EMOTIONS: set[str] = {"neutral", "angry", "friendly", "fearful", "curious", "threatening"}
_VALID_SOCIALS: set[str] = {"neutral", "polite", "rude", "deceptive", "honest", "intimidating"}

# ─── NPC name → UID lookup tables ────────────────────────────────────────────
# Populated from NPC registry at runtime; maps lowercased name fragments to UIDs.
_NPC_NAME_MAP: dict[str, str] = {
    # Hard-coded short names for the 6 MVP NPCs as safety net.
    "maren":  "elder_m8b2",
    "elder":  "elder_m8b2",
    "elder maren": "elder_m8b2",
    "jak":    "farmer_j4a1",
    "farmer": "farmer_j4a1",
    "farmer jak": "farmer_j4a1",
    "tessa":  "tavkeeper_t9c3",
    "tavkeeper": "tavkeeper_t9c3",
    "aldric": "guard_a3f1",
    "guard aldric": "guard_a3f1",
    "bryn":   "guard_b7e2",
    "guard bryn": "guard_b7e2",
    "petra":  "villager_c1d4",
    "old petra": "villager_c1d4",
    "villager": "villager_c1d4",
}

# ─── Location name → ID mapping ──────────────────────────────────────────────
_LOCATION_NAME_MAP: dict[str, str] = {
    "gate":            "gate",
    "the gate":        "gate",
    "village center":  "village_center",
    "village":         "village_center",
    "center":          "village_center",
    "square":          "village_center",
    "town center":     "village_center",
    "elder's house":   "elders_house",
    "elders house":    "elders_house",
    "elder house":     "elders_house",
    "maren's house":   "elders_house",
    "fields":          "fields",
    "the fields":      "fields",
    "farm":            "fields",
    "farmland":        "fields",
    "tavern":          "tavern",
    "the tavern":      "tavern",
    "inn":             "tavern",
    "bar":             "tavern",
    "pub":             "tavern",
}

# ─── Compound-input splitter patterns ────────────────────────────────────────
_COMPOUND_SPLIT_RE = re.compile(
    r"\b(?:then|and then|after that|afterwards|next|also)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def init_nlp() -> None:
    """Load spaCy model and pre-compute action vectors.

    Call once at startup.  If spaCy or the model is unavailable the module
    falls back to keyword-only matching.
    """
    global _nlp  # noqa: PLW0603

    if not _SPACY_AVAILABLE:
        logger.warning("spaCy not installed — input parser will use keyword-only matching.")
        return

    try:
        _nlp = spacy.load("en_core_web_md")
        logger.info("spaCy model 'en_core_web_md' loaded for input_parser.")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_md' not found — "
            "input parser will use keyword-only matching."
        )
        _nlp = None
        return

    _precompute_action_vectors()


def _precompute_action_vectors() -> None:
    """Build ``ACTION_VECTORS`` from the universal action labels via spaCy."""
    if _nlp is None:
        return

    ACTION_VECTORS.clear()

    for action_id, meta in UNIVERSAL_ACTIONS.items():
        # Combine the label with all synonyms to create a richer vector.
        synonym_text = " ".join(ACTION_SYNONYMS.get(action_id, []))
        combined = f"{meta['label']} {synonym_text}".strip()
        doc = _nlp(combined)
        if doc.has_vector and np.any(doc.vector):
            ACTION_VECTORS[action_id] = doc.vector / (np.linalg.norm(doc.vector) + 1e-10)
        else:
            ACTION_VECTORS[action_id] = doc.vector

    logger.info("Pre-computed action vectors for %d actions.", len(ACTION_VECTORS))


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — Button input
# ═══════════════════════════════════════════════════════════════════════════════

def parse_button_input(
    action_id: str,
    target_npc: str | None = None,
    target_item: str | None = None,
    target_location: str | None = None,
) -> ParsedInput:
    """Create a ``ParsedInput`` from an action-palette button click.

    Parameters
    ----------
    action_id:
        Must be one of the 27 universal action IDs.
    target_npc:
        Resolved NPC UID (or *None*).
    target_item:
        Item ID the action targets (or *None*).
    target_location:
        Location ID for movement (or *None*).

    Returns
    -------
    ParsedInput
        Fully populated, confidence = 1.0, emotion/social = neutral.
    """
    if action_id not in UNIVERSAL_ACTIONS:
        logger.error("parse_button_input received unknown action_id=%r", action_id)

    return ParsedInput(
        source="button",
        raw_text=None,
        action_id=action_id,
        target_npc=target_npc,
        target_item=target_item,
        target_location=target_location,
        confidence=1.0,
        emotion="neutral",
        intent=action_id,
        social="neutral",
        queued_action=None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — Free-text input
# ═══════════════════════════════════════════════════════════════════════════════

def parse_text_input(
    text: str,
    npc_registry: dict[str, Any],
    player_location: str,
    player_inventory: list[dict] | None = None,
) -> ParsedInput:
    """Parse free-form player text into a ``ParsedInput``.

    Three dimensions are extracted independently:

    * **emotion** — how the player is feeling
    * **action / intent** — what the player wants to do (→ action_id)
    * **social** — the social register of the request

    Parameters
    ----------
    text:
        Raw player input string.
    npc_registry:
        Mapping of NPC UID → NPC data dicts (needs ``name``, ``location``).
    player_location:
        Current location ID of the player.
    player_inventory:
        Player's inventory list (optional, used for item-target extraction).

    Returns
    -------
    ParsedInput
        Populated dict.  ``confidence`` reflects how certain the action
        mapping is (1.0 = keyword hit, 0.4–0.99 = vector similarity).
    """
    if not text or not text.strip():
        logger.debug("parse_text_input received empty text.")
        return _empty_parsed_input(text)

    text_clean = text.strip()
    text_lower = text_clean.lower()

    # ── Handle compound input ────────────────────────────────────────────
    queued: dict | None = None
    parts = _COMPOUND_SPLIT_RE.split(text_clean, maxsplit=1)
    if len(parts) > 1:
        first_part = parts[0].strip()
        second_part = parts[1].strip()
        if first_part and second_part:
            text_clean = first_part
            text_lower = text_clean.lower()
            # Recursively parse the second part as a queued action.
            queued = dict(
                parse_text_input(second_part, npc_registry, player_location, player_inventory)
            )
            logger.debug(
                "Compound input detected — first=%r  queued=%r",
                first_part,
                second_part,
            )

    # ── spaCy processing (if available) ──────────────────────────────────
    doc: Doc | None = None
    if _nlp is not None:
        doc = _nlp(text_clean)

    # ── Extract three dimensions ─────────────────────────────────────────
    emotion = _extract_emotion(doc, text_lower)
    social = _extract_social(doc, text_lower)
    action_id, confidence = _extract_action(doc, text_lower)

    # ── Extract targets ──────────────────────────────────────────────────
    target_npc = _extract_target_npc(doc, text_lower, npc_registry, player_location)
    target_item = _extract_target_item(doc, text_lower, player_inventory or [])
    target_location = _extract_target_location(doc, text_lower)

    parsed = ParsedInput(
        source="text",
        raw_text=text_clean,
        action_id=action_id,
        target_npc=target_npc,
        target_item=target_item,
        target_location=target_location,
        confidence=confidence,
        emotion=emotion,
        intent=action_id or text_lower,
        social=social,
        queued_action=queued,
    )

    logger.debug(
        "Parsed text input: action=%s conf=%.2f emo=%s soc=%s npc=%s item=%s loc=%s",
        action_id,
        confidence,
        emotion,
        social,
        target_npc,
        target_item,
        target_location,
    )

    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
#  Dimension extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_emotion(doc: Doc | None, text_lower: str) -> str:
    """Determine the player's emotional tone from the input text.

    Uses keyword dictionaries for deterministic matching.

    Returns
    -------
    str
        One of ``_VALID_EMOTIONS``.
    """
    best_emotion = "neutral"
    best_count = 0

    for emotion, keywords in EMOTION_KEYWORDS.items():
        if emotion == "neutral":
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_emotion = emotion

    return best_emotion


def _extract_social(doc: Doc | None, text_lower: str) -> str:
    """Determine the social register of the input text.

    Uses keyword dictionaries for deterministic matching.

    Returns
    -------
    str
        One of ``_VALID_SOCIALS``.
    """
    best_social = "neutral"
    best_count = 0

    for social, keywords in SOCIAL_KEYWORDS.items():
        if social == "neutral":
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_social = social

    return best_social


def _extract_action(doc: Doc | None, text_lower: str) -> tuple[str | None, float]:
    """Map the input text to a universal action ID.

    Strategy
    --------
    1. **Keyword / synonym** — scan ``ACTION_SYNONYMS``.  Longer phrases
       are checked first so *"go to"* wins over *"go"*.
    2. **spaCy similarity** — fall back to vector cosine similarity between
       the input and pre-computed ``ACTION_VECTORS``.
    3. Return ``(None, 0.0)`` if nothing reaches the 0.4 threshold.

    Returns
    -------
    tuple[str | None, float]
        ``(action_id, confidence)``.
    """
    # ── 1. Keyword / synonym lookup (deterministic) ──────────────────────
    match = _keyword_action_match(text_lower)
    if match is not None:
        return match, 1.0

    # ── 2. spaCy similarity fallback ─────────────────────────────────────
    if doc is not None and ACTION_VECTORS:
        best_action, best_sim = _similarity_action_match(doc)
        if best_action is not None and best_sim >= 0.4:
            return best_action, round(float(best_sim), 4)

    # ── 3. No match ──────────────────────────────────────────────────────
    return None, 0.0


def _keyword_action_match(text_lower: str) -> str | None:
    """Try to match *text_lower* against ``ACTION_SYNONYMS``.

    Longer synonym phrases are checked first to prevent partial triggers
    (e.g. *"go to the gate"* should match ``move_to``, not ``look``).
    """
    # Build a flat list of (phrase, action_id) sorted longest-first.
    candidates: list[tuple[str, str]] = []
    for action_id, synonyms in ACTION_SYNONYMS.items():
        for phrase in synonyms:
            candidates.append((phrase.lower(), action_id))
    # Sort by descending phrase length for greedy matching.
    candidates.sort(key=lambda t: len(t[0]), reverse=True)

    for phrase, action_id in candidates:
        if phrase in text_lower:
            return action_id

    return None


def _similarity_action_match(doc: Doc) -> tuple[str | None, float]:
    """Find the closest action by cosine similarity of spaCy vectors.

    Returns
    -------
    tuple[str | None, float]
        ``(action_id, similarity)`` of the best match, or ``(None, 0.0)``.
    """
    if not doc.has_vector or not np.any(doc.vector):
        return None, 0.0

    input_vec = doc.vector / (np.linalg.norm(doc.vector) + 1e-10)

    best_id: str | None = None
    best_sim: float = -1.0

    for action_id, action_vec in ACTION_VECTORS.items():
        sim = float(np.dot(input_vec, action_vec))
        if sim > best_sim:
            best_sim = sim
            best_id = action_id

    return best_id, max(best_sim, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Target extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_target_npc(
    doc: Doc | None,
    text_lower: str,
    npc_registry: dict[str, Any],
    player_location: str,
) -> str | None:
    """Resolve mentioned NPC to a UID, preferring NPCs at the player's location.

    Strategy:
    1. Check hard-coded ``_NPC_NAME_MAP`` and dynamic names from registry.
    2. Use spaCy NER (``PERSON`` entities) as a secondary signal.
    3. Filter candidates to NPCs at the player's current location.
    """
    # Build a dynamic name map from the live registry.
    dynamic_map: dict[str, str] = dict(_NPC_NAME_MAP)
    for uid, npc in npc_registry.items():
        name: str = npc.get("name", "")
        if name:
            dynamic_map[name.lower()] = uid
            # Also add first-name / short-name fragments.
            parts = name.lower().split()
            for part in parts:
                if part not in {"old"}:  # Skip generic stopwords
                    dynamic_map.setdefault(part, uid)

    # ── 1. Direct substring scan (longest-first) ────────────────────────
    name_candidates = sorted(dynamic_map.keys(), key=len, reverse=True)
    matched_uid: str | None = None
    for name_fragment in name_candidates:
        if name_fragment in text_lower:
            matched_uid = dynamic_map[name_fragment]
            break

    # ── 2. spaCy NER fallback ────────────────────────────────────────────
    if matched_uid is None and doc is not None:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_lower = ent.text.lower()
                if ent_lower in dynamic_map:
                    matched_uid = dynamic_map[ent_lower]
                    break

    if matched_uid is None:
        return None

    # ── 3. Prefer NPCs at the player's location ─────────────────────────
    npc_data = npc_registry.get(matched_uid)
    if npc_data and npc_data.get("location") == player_location:
        return matched_uid

    # NPC exists but is elsewhere — still return the UID so the game engine
    # can decide how to handle it (e.g. "Jak isn't here").
    if matched_uid in npc_registry:
        logger.debug(
            "NPC %s matched but not at player location %s.",
            matched_uid,
            player_location,
        )
        return matched_uid

    return matched_uid


def _extract_target_item(
    doc: Doc | None,
    text_lower: str,
    player_inventory: list[dict],
) -> str | None:
    """Resolve mentioned item to an item ID from the player's inventory.

    Matches by item ``name`` (case-insensitive substring) then by ``id``.
    """
    # Sort by name length descending so longer names match first.
    sorted_items = sorted(
        player_inventory,
        key=lambda it: len(it.get("name", "")),
        reverse=True,
    )

    for item in sorted_items:
        item_name = item.get("name", "").lower()
        item_id = item.get("id", "").lower()
        if item_name and item_name in text_lower:
            return item["id"]
        if item_id and item_id in text_lower:
            return item["id"]

    # spaCy NER is unlikely to help with game-specific item names.
    return None


def _extract_target_location(doc: Doc | None, text_lower: str) -> str | None:
    """Resolve a mentioned location to a location ID.

    Checks the human-readable ``_LOCATION_NAME_MAP`` then falls back to
    raw ``LOCATION_IDS``.
    """
    # Longest-match-first to avoid "gate" matching inside "village gate".
    ordered = sorted(_LOCATION_NAME_MAP.keys(), key=len, reverse=True)
    for name in ordered:
        if name in text_lower:
            return _LOCATION_NAME_MAP[name]

    # Fall back to raw location IDs (e.g., "tavern", "fields").
    for loc_id in LOCATION_IDS:
        if loc_id.replace("_", " ") in text_lower or loc_id in text_lower:
            return loc_id

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_parsed_input(raw_text: str | None = None) -> ParsedInput:
    """Return a blank ``ParsedInput`` for empty / unparseable input."""
    return ParsedInput(
        source="text",
        raw_text=raw_text,
        action_id=None,
        target_npc=None,
        target_item=None,
        target_location=None,
        confidence=0.0,
        emotion="neutral",
        intent="",
        social="neutral",
        queued_action=None,
    )


def resolve_npc_uid(name_or_uid: str, npc_registry: dict[str, Any]) -> str | None:
    """Convenience: resolve a display name or partial name to a UID.

    Useful outside the parsing pipeline (e.g. button targeting).
    """
    # Already a UID?
    if name_or_uid in npc_registry:
        return name_or_uid

    lower = name_or_uid.lower()

    # Check hard-coded map first.
    if lower in _NPC_NAME_MAP:
        uid = _NPC_NAME_MAP[lower]
        if uid in npc_registry:
            return uid

    # Check registry names.
    for uid, npc in npc_registry.items():
        npc_name: str = npc.get("name", "")
        if npc_name.lower() == lower:
            return uid
        # Partial match (e.g. "jak" → "farmer_j4a1").
        if lower in npc_name.lower():
            return uid

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM tertiary analysis (stub)
# ═══════════════════════════════════════════════════════════════════════════════

async def _llm_parse_input(text: str) -> ParsedInput | None:
    """Placeholder for LLM-powered 3-dimension analysis.

    When implemented this will:
    1. Send the raw text to the local GGUF model with a structured prompt.
    2. Request JSON output with ``emotion``, ``action_id``, ``social``.
    3. Validate the response against the universal action catalog.
    4. Return a ``ParsedInput`` or ``None`` on failure.

    Currently returns ``None`` (no-op) — the caller falls through to
    template / keyword resolution.
    """
    # TODO: implement with llama-cpp-python via asyncio.to_thread()
    return None
