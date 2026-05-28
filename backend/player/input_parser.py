"""Input parser. Both button and free-text input produce a unified
ParsedInput dict.

Free-text uses a three-layer pipeline: keyword/synonym matching first
(deterministic), spaCy similarity as a tiebreaker, and the LLM only as
a last resort for low-confidence parses (driven from routes.py).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np  # type: ignore[import-unresolved]

from backend.config import (
    ACTION_SYNONYMS,
    EMOTION_KEYWORDS,
    LLM_MAX_RETRIES,
    LOCATION_IDS,
    SOCIAL_KEYWORDS,
    UNIVERSAL_ACTION_IDS,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.llm.guardrails import validate_input_analysis
from backend.llm.prompts import build_input_analysis_prompt

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

# spaCy is optional: when unavailable we fall back to keyword-only matching.
try:
    import spacy  # type: ignore[import-unresolved]

    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

_nlp: Language | None = None
ACTION_VECTORS: dict[str, np.ndarray] = {}


class ParsedInput(TypedDict, total=False):
    """Unified representation produced by both button and text input."""

    source: str            # "button" | "text"
    raw_text: str | None
    action_id: str | None
    target_npc: str | None
    target_item: str | None
    target_location: str | None
    confidence: float       # 1.0 for buttons, [0, 1] for text
    emotion: str
    intent: str
    social: str
    queued_action: dict | None  # second action of a compound utterance


_VALID_EMOTIONS: set[str] = {"neutral", "angry", "friendly", "fearful", "curious", "threatening"}
_VALID_SOCIALS: set[str] = {"neutral", "polite", "rude", "deceptive", "honest", "intimidating", "cooperative"}

# Hard-coded short names for the 6 MVP NPCs. Persists so the parser still
# resolves names if an NPC isn't yet in the live registry (e.g. tests).
_NPC_NAME_MAP: dict[str, str] = {
    "maren":  "elder_m8b2",
    "elder":  "elder_m8b2",
    "elder maren": "elder_m8b2",
    "the elder": "elder_m8b2",
    "jak":    "farmer_j4a1",
    "farmer": "farmer_j4a1",
    "farmer jak": "farmer_j4a1",
    "tessa":  "tavkeeper_t9c3",
    "tavkeeper": "tavkeeper_t9c3",
    "bartender": "tavkeeper_t9c3",
    "barmaid": "tavkeeper_t9c3",
    "innkeeper": "tavkeeper_t9c3",
    "aldric": "guard_a3f1",
    "aldrick": "guard_a3f1",
    "guard aldric": "guard_a3f1",
    "bryn":   "guard_b7e2",
    "brynn":  "guard_b7e2",
    "brin":   "guard_b7e2",
    "guard bryn": "guard_b7e2",
    "guard brynn": "guard_b7e2",
    "petra":  "villager_c1d4",
    "old petra": "villager_c1d4",
    "villager": "villager_c1d4",
}

# Names that could resolve to multiple NPCs — `guard` is the only one we
# need to break the tie for, using the player's current location.
_AMBIGUOUS_NPC_NAMES: dict[str, list[str]] = {
    "guard": ["guard_a3f1", "guard_b7e2"],
}

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

# Conjunctions / sequencers that introduce a follow-up action ("then X").
_COMPOUND_SPLIT_RE = re.compile(
    r"\b(?:then|and then|after that|afterwards|next|also)\b",
    re.IGNORECASE,
)


def init_nlp() -> None:
    """Load spaCy and warm action vectors. Idempotent; safe to skip if spaCy is absent."""
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
    if _nlp is None:
        return

    ACTION_VECTORS.clear()

    for action_id, meta in UNIVERSAL_ACTIONS.items():
        # Folding synonyms into the source string gives a richer vector
        # than embedding the bare label alone.
        synonym_text = " ".join(ACTION_SYNONYMS.get(action_id, []))
        combined = f"{meta['label']} {synonym_text}".strip()
        doc = _nlp(combined)
        if doc.has_vector and np.any(doc.vector):
            ACTION_VECTORS[action_id] = doc.vector / (np.linalg.norm(doc.vector) + 1e-10)
        else:
            ACTION_VECTORS[action_id] = doc.vector

    logger.info("Pre-computed action vectors for %d actions.", len(ACTION_VECTORS))


def parse_button_input(
    action_id: str,
    target_npc: str | None = None,
    target_item: str | None = None,
    target_location: str | None = None,
) -> ParsedInput:
    """Wrap an action-palette button click in a ParsedInput. Confidence = 1.0."""
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


def parse_text_input(
    text: str,
    npc_registry: dict[str, Any],
    player_location: str,
    player_inventory: list[dict] | None = None,
) -> ParsedInput:
    """Parse free-form text into a ParsedInput by extracting emotion, social
    register, action, and targets independently. `confidence` is 1.0 on a
    direct keyword hit, 0.4–0.99 on a vector-similarity match."""
    if not text or not text.strip():
        logger.debug("parse_text_input received empty text.")
        return _empty_parsed_input(text)

    text_clean = text.strip()
    text_lower = text_clean.lower()

    # Compound input: "do X then Y" — execute X now and queue Y for next turn.
    queued: dict | None = None
    parts = _COMPOUND_SPLIT_RE.split(text_clean, maxsplit=1)
    if len(parts) > 1:
        first_part = parts[0].strip()
        second_part = parts[1].strip()
        if first_part and second_part:
            text_clean = first_part
            text_lower = text_clean.lower()
            queued = dict(
                parse_text_input(second_part, npc_registry, player_location, player_inventory)
            )
            logger.debug(
                "Compound input detected — first=%r  queued=%r",
                first_part,
                second_part,
            )

    doc: Doc | None = None
    if _nlp is not None:
        doc = _nlp(text_clean)

    emotion = _extract_emotion(doc, text_lower)
    social = _extract_social(doc, text_lower)

    # Targets first, then action. Order matters: item names like
    # "travel papers" overlap action synonyms like "travel" — we mask
    # extracted entities out of the text before running action matching.
    target_npc = _extract_target_npc(doc, text_lower, npc_registry, player_location)
    target_item = _extract_target_item(doc, text_lower, player_inventory or [])
    target_location = _extract_target_location(doc, text_lower)

    masked_text = _mask_entities(text_lower, target_item, target_npc,
                                 target_location, npc_registry,
                                 player_inventory or [])
    masked_doc: Doc | None = None
    if _nlp is not None and masked_text != text_lower:
        masked_doc = _nlp(masked_text)

    action_id, confidence = _extract_action(
        masked_doc or doc, masked_text,
    )

    # Ambiguous / unparseable input defaults to "talk" so the player is never
    # silenced — the NPC layer can do something useful with conversational text.
    if action_id is None or confidence < 0.4:
        action_id = "talk"
        confidence = max(confidence, 0.35)

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


def _extract_emotion(doc: Doc | None, text_lower: str) -> str:
    """Highest emotion-keyword hit count wins; defaults to neutral."""
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

    Highest keyword-hit count wins; ties resolved by dict iteration order."""
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
    """Map text → (action_id, confidence). Returns (None, 0.0) below the 0.4 threshold."""
    match = _keyword_action_match(text_lower)
    if match is not None:
        return match, 1.0

    if doc is not None and ACTION_VECTORS:
        best_action, best_sim = _similarity_action_match(doc)
        if best_action is not None and best_sim >= 0.4:
            return best_action, round(float(best_sim), 4)

    return None, 0.0


_keyword_cache: list[tuple[re.Pattern, str]] | None = None


def _keyword_action_match(text_lower: str) -> str | None:
    """Deterministic action match via synonym dictionary.

    Patterns are sorted longest-first so "go to" wins over "go", and
    word-boundary anchors prevent substring false positives ("repeat"
    must not match "eat").
    """
    global _keyword_cache
    if _keyword_cache is None:
        candidates: list[tuple[str, str]] = []
        for action_id, synonyms in ACTION_SYNONYMS.items():
            for phrase in synonyms:
                candidates.append((phrase.lower(), action_id))
        candidates.sort(key=lambda t: len(t[0]), reverse=True)
        _keyword_cache = [
            (re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"), action_id)
            for phrase, action_id in candidates
        ]

    for pattern, action_id in _keyword_cache:
        if pattern.search(text_lower):
            return action_id

    return None


def _similarity_action_match(doc: Doc) -> tuple[str | None, float]:
    """Cosine-similarity nearest action vector. (None, 0.0) when the input has no vector."""
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


def _extract_target_npc(
    doc: Doc | None,
    text_lower: str,
    npc_registry: dict[str, Any],
    player_location: str,
) -> str | None:
    """Resolve a mentioned NPC to a UID. Prefers NPCs at the player's
    location, then falls back to spaCy PERSON NER, then the first
    candidate of an ambiguous name."""
    # Live-registry names get merged into the static map so newly-spawned
    # NPCs are addressable without a code change.
    dynamic_map: dict[str, str] = dict(_NPC_NAME_MAP)
    for uid, npc in npc_registry.items():
        name: str = getattr(npc, "name", "") if not isinstance(npc, dict) else npc.get("name", "")
        if name:
            dynamic_map[name.lower()] = uid
            parts = name.lower().split()
            for part in parts:
                if part not in {"old"}:
                    dynamic_map.setdefault(part, uid)

    # Longest-first so "elder maren" beats "elder".
    name_candidates = sorted(dynamic_map.keys(), key=len, reverse=True)
    matched_uid: str | None = None
    for name_fragment in name_candidates:
        if name_fragment in text_lower:
            matched_uid = dynamic_map[name_fragment]
            break

    if matched_uid is None:
        for ambig_name, candidate_uids in _AMBIGUOUS_NPC_NAMES.items():
            if ambig_name in text_lower:
                # Disambiguate by who's actually present, falling back to
                # the first candidate so the player still gets a target.
                for cuid in candidate_uids:
                    npc_c = npc_registry.get(cuid)
                    if npc_c is None:
                        continue
                    npc_c_loc = getattr(npc_c, "location", None) if not isinstance(npc_c, dict) else npc_c.get("location")
                    if npc_c_loc == player_location:
                        matched_uid = cuid
                        break
                if matched_uid is None and candidate_uids:
                    matched_uid = candidate_uids[0]
                break

    # ── 2. Fuzzy name fallback — check for close matches ─────────────────
    if matched_uid is None:
        matched_uid = _fuzzy_npc_match(text_lower, dynamic_map)

    # ── 3. spaCy NER fallback ────────────────────────────────────────────
    if matched_uid is None and doc is not None:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_lower = ent.text.lower()
                if ent_lower in dynamic_map:
                    matched_uid = dynamic_map[ent_lower]
                    break
                # Also try fuzzy on NER entities
                fuzzy_uid = _fuzzy_npc_match(ent_lower, dynamic_map)
                if fuzzy_uid is not None:
                    matched_uid = fuzzy_uid
                    break

    if matched_uid is None:
        return None

    # ── 4. Prefer NPCs at the player's location ─────────────────────────
    npc_data = npc_registry.get(matched_uid)
    npc_loc = getattr(npc_data, "location", None) if not isinstance(npc_data, dict) else npc_data.get("location")
    if npc_data and npc_loc == player_location:
        return matched_uid

    # Even when the NPC isn't at the player's location, return the UID so
    # the engine layer can produce a useful "they aren't here" message.
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
    """Resolve a mentioned item to an inventory ID. Name match wins over raw ID."""
    # Longest-first so "travel papers" beats "papers".
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

    return None


def _extract_target_location(doc: Doc | None, text_lower: str) -> str | None:
    """Resolve a mentioned location to a location ID."""
    # Longest-first so "the village center" beats "village".
    ordered = sorted(_LOCATION_NAME_MAP.keys(), key=len, reverse=True)
    for name in ordered:
        if name in text_lower:
            return _LOCATION_NAME_MAP[name]

    for loc_id in LOCATION_IDS:
        if loc_id.replace("_", " ") in text_lower or loc_id in text_lower:
            return loc_id

    return None


def _mask_entities(
    text_lower: str,
    target_item: str | None,
    target_npc: str | None,
    target_location: str | None,
    npc_registry: dict[str, Any],
    player_inventory: list[dict],
) -> str:
    """Strip recognised entity surfaces from the input so action matching
    doesn't collide with item/NPC/location names.

    Example: "show travel papers to aldric" → "show  to " after masking
    "travel papers" and "aldric", allowing "show" to resolve to
    `present_item` instead of "travel" → `move_to`.
    """
    masks: list[str] = []

    for item in player_inventory:
        item_name = item.get("name", "").lower()
        item_id = item.get("id", "").lower()
        if item_name and item_name in text_lower:
            masks.append(item_name)
        if item_id and item_id in text_lower and item_id != item_name:
            masks.append(item_id)

    for name_fragment in _NPC_NAME_MAP:
        if name_fragment in text_lower:
            masks.append(name_fragment)
    for uid, npc in npc_registry.items():
        name: str = getattr(npc, "name", "") if not isinstance(npc, dict) else npc.get("name", "")
        if name:
            name_low = name.lower()
            if name_low in text_lower:
                masks.append(name_low)
            for part in name_low.split():
                if part not in {"old"} and part in text_lower:
                    masks.append(part)

    for loc_name in _LOCATION_NAME_MAP:
        if loc_name in text_lower:
            masks.append(loc_name)
    for loc_id in LOCATION_IDS:
        readable = loc_id.replace("_", " ")
        if readable in text_lower:
            masks.append(readable)
        if loc_id in text_lower:
            masks.append(loc_id)

    if not masks:
        return text_lower

    # Longest-first so partial overlaps don't leave fragments behind.
    masks = sorted(set(masks), key=len, reverse=True)

    masked = text_lower
    for entity in masks:
        masked = masked.replace(entity, " ")

    return " ".join(masked.split())


def _fuzzy_npc_match(text_lower: str, name_map: dict[str, str]) -> str | None:
    """Approximate-match NPC names so common typos still resolve
    (e.g. "brynn" → "bryn")."""
    words = text_lower.split()
    for word in words:
        if len(word) < 3:
            continue
        for known_name, uid in name_map.items():
            if len(known_name) < 3 or " " in known_name:
                continue
            # Check if within 1-char edit distance (simple heuristic)
            if _is_close_match(word, known_name, max_dist=1):
                logger.debug("Fuzzy NPC match: '%s' → '%s' (%s)", word, known_name, uid)
                return uid
    return None


def _is_close_match(a: str, b: str, max_dist: int = 1) -> bool:
    """Approximate Levenshtein for short strings, char-set overlap for long ones."""
    if abs(len(a) - len(b)) > max_dist:
        return False
    if a == b:
        return True

    if len(a) <= 6 and len(b) <= 6:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        diffs = 0
        j = 0
        for i in range(len(longer)):
            if j < len(shorter) and longer[i] == shorter[j]:
                j += 1
            else:
                diffs += 1
        diffs += len(shorter) - j
        return diffs <= max_dist

    common = sum(1 for c in set(a) if c in b)
    return common >= max(len(a), len(b)) - max_dist


def _empty_parsed_input(raw_text: str | None = None) -> ParsedInput:
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
        npc_name: str = getattr(npc, "name", "") if not isinstance(npc, dict) else npc.get("name", "")
        if npc_name.lower() == lower:
            return uid
        # Partial match (e.g. "jak" → "farmer_j4a1").
        if lower in npc_name.lower():
            return uid

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM tertiary analysis (stub)
# ═══════════════════════════════════════════════════════════════════════════════

async def _llm_parse_input(
    text: str,
    location: str = "unknown",
    npcs_present: list[str] | None = None,
    highlighted_actions: list[str] | None = None,
    llm_service: object | None = None,
) -> ParsedInput | None:
    """LLM-powered tertiary 3-dimension analysis.

    Retries on LLM/validation failure up to LLM_MAX_RETRIES; returns None
    if the LLM is unavailable or every attempt fails the schema check.
    """
    if llm_service is None or not getattr(llm_service, "available", False):
        return None

    prompt = build_input_analysis_prompt(
        player_text=text,
        location=location,
        npcs_present=npcs_present or [],
        highlighted_actions=highlighted_actions or [],
    )

    for attempt in range(LLM_MAX_RETRIES):
        raw = await llm_service.async_generate(prompt, temperature=0.4)
        if raw is None:
            logger.debug("LLM input parse attempt %d returned None", attempt + 1)
            continue

        validated = validate_input_analysis(raw)
        if validated is None:
            logger.debug("LLM input parse validation failed on attempt %d", attempt + 1)
            continue

        action_id = validated["matched_action"]
        if action_id == "UNKNOWN":
            action_id = None

        logger.info("LLM input parse succeeded on attempt %d: action=%s", attempt + 1, action_id)
        return ParsedInput(
            source="text",
            raw_text=text,
            action_id=action_id,
            target_npc=None,
            target_item=None,
            target_location=None,
            confidence=validated["confidence"],
            emotion=validated["emotion"],
            intent=validated.get("interpreted_intent", validated.get("intent", "")),
            social=validated["social"],
            queued_action=None,
        )

    logger.info("LLM input parse exhausted %d attempts", LLM_MAX_RETRIES)
    return None
