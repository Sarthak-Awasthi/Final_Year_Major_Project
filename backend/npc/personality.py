"""
personality.py — Archetype registry and NPC factory.

Loads archetype / instance JSON files, validates reward weights,
and constructs the NPC registry used by the game engine.
"""

from __future__ import annotations

import json
import pathlib
import random

from backend.config import (
    MASTER_SEED,
    NPC_DIR,
    logger,
)
from backend.npc.npc import NPC


# ── Archetype loading ─────────────────────────────────────────────────────────

def load_archetypes() -> dict[str, dict]:
    """Load all archetype JSON files from ``NPC_DIR/archetypes/``.

    Returns:
        Mapping of archetype name → archetype data dict.
    """
    archetypes: dict[str, dict] = {}
    archetype_dir: pathlib.Path = NPC_DIR / "archetypes"

    if not archetype_dir.exists():
        logger.error("Archetype directory not found: %s", archetype_dir)
        return archetypes

    for path in sorted(archetype_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = data.get("archetype", path.stem)
            archetypes[key] = data
            logger.debug("Loaded archetype '%s' from %s", key, path.name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load archetype %s: %s", path, exc)

    return archetypes


def validate_archetype(data: dict) -> bool:
    """Check that an archetype's reward weights sum to 1.0.

    Args:
        data: A single archetype data dict.

    Returns:
        ``True`` if valid.
    """
    weights = data.get("reward_weights", {})
    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        logger.warning(
            "Archetype '%s' reward weights sum to %.4f (expected 1.0)",
            data.get("archetype", "?"),
            total,
        )
        return False
    return True


# ── Instance loading ──────────────────────────────────────────────────────────

def load_npc_instances() -> dict[str, dict]:
    """Load all NPC instance JSON files from ``NPC_DIR/instances/``.

    Returns:
        Mapping of ``npc_uid`` → instance data dict.
    """
    instances: dict[str, dict] = {}
    instance_dir: pathlib.Path = NPC_DIR / "instances"

    if not instance_dir.exists():
        logger.error("Instance directory not found: %s", instance_dir)
        return instances

    for path in sorted(instance_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            uid = data.get("npc_uid", path.stem)
            instances[uid] = data
            logger.debug("Loaded NPC instance '%s' from %s", uid, path.name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load NPC instance %s: %s", path, exc)

    return instances


# ── Registry construction ─────────────────────────────────────────────────────

def create_npc_registry(master_seed: int = MASTER_SEED) -> dict[str, NPC]:
    """Create the complete NPC registry from JSON data files.

    For each instance, the matching archetype is looked up, a
    per-NPC random seed is derived deterministically from *master_seed*
    and the NPC UID (avoiding non-deterministic hash()), and an :class:`NPC`
    object is built.

    Args:
        master_seed: Root seed for all per-NPC randomness.

    Returns:
        Mapping of ``npc_uid`` → :class:`NPC`.
    """
    archetypes = load_archetypes()
    instances = load_npc_instances()

    # Validate every archetype
    for key, arch in archetypes.items():
        validate_archetype(arch)

    registry: dict[str, NPC] = {}

    # Create a stable ordering of NPCs so seed derivation is deterministic
    sorted_uids = sorted(instances.keys())

    for npc_index, uid in enumerate(sorted_uids):
        inst_data = instances[uid]
        arch_key = inst_data.get("archetype")
        if arch_key not in archetypes:
            logger.error(
                "NPC '%s' references unknown archetype '%s' — skipping",
                uid,
                arch_key,
            )
            continue

        arch_data = archetypes[arch_key]

        npc = NPC(inst_data, arch_data)
        registry[uid] = npc
        logger.debug("Created NPC %s (%s)", uid, npc.name)

    logger.info("NPC registry created with %d NPCs", len(registry))
    return registry


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_npc_by_name(registry: dict[str, NPC], name: str) -> NPC | None:
    """Find an NPC in *registry* whose ``name`` matches (case-insensitive).

    Returns:
        The NPC, or ``None`` if not found.
    """
    name_lower = name.lower()
    for npc in registry.values():
        if npc.name.lower() == name_lower:
            return npc
    return None


def get_npcs_at_location(registry: dict[str, NPC], location: str) -> list[NPC]:
    """Return all active (non-incapacitated) NPCs at *location*.

    Args:
        registry: The NPC registry dict.
        location: A location ID string.

    Returns:
        List of :class:`NPC` objects at the given location.
    """
    return [
        npc
        for npc in registry.values()
        if npc.location == location and not npc.is_incapacitated()
    ]
