"""
combat.py — Combat resolution mechanic, damage calc, flee/defend.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from backend.config import (
    COMBAT_DAMAGE_VARIANCE,
    COMBAT_DEFEND_REDUCTION,
    COMBAT_FLEE_BASE_SUCCESS,
    COMBAT_FLEE_BONUS_HIT,
    COMBAT_FLEE_EXHAUSTED_SUCCESS,
    COMBAT_HIT_MAX,
    COMBAT_HIT_MIN,
    COMBAT_MIN_DAMAGE,
    logger,
)


@dataclass
class CombatResult:
    """Result of a single combat exchange."""

    hit: bool
    damage: int
    attacker_name: str
    defender_name: str
    attacker_stamina_cost: int
    defender_stamina_cost: int
    narrative: str
    effects: dict[str, Any]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high."""
    return max(low, min(high, value))


def compute_hit_probability(
    attacker_attack: int,
    weapon_mod: int,
    stamina_factor: float,
    defender_defense: int,
    armor_mod: int,
) -> float:
    """
    Compute probability of hitting in combat.

    P(hit) = clamp((atk + weapon + stamina_factor*10) / ((def + armor)*2 + 20), 0.1, 0.95)
    """
    numerator = attacker_attack + weapon_mod + (stamina_factor * 10)
    denominator = (defender_defense + armor_mod) * 2 + 20
    if denominator == 0:
        denominator = 1
    prob = numerator / denominator
    return clamp(prob, COMBAT_HIT_MIN, COMBAT_HIT_MAX)


def compute_damage(
    attacker_attack: int,
    weapon_mod: int,
    defender_defense: int,
    armor_mod: int,
    damage_multiplier: float = 1.0,
) -> int:
    """
    Compute damage dealt on a successful hit.

    damage = max(1, base_attack + weapon_mod - base_defense - armor_mod + randint(-3, 3))
    """
    base = attacker_attack + weapon_mod - defender_defense - armor_mod
    variance = random.randint(-COMBAT_DAMAGE_VARIANCE, COMBAT_DAMAGE_VARIANCE)
    raw_damage = base + variance
    return max(COMBAT_MIN_DAMAGE, int(raw_damage * damage_multiplier))


def resolve_attack(
    attacker: dict,
    defender: dict,
    difficulty_config: dict | None = None,
) -> CombatResult:
    """
    Resolve an attack action.

    attacker/defender are dicts with keys:
        name, base_attack, base_defense, weapon_modifier, armor_modifier,
        current_stamina, max_stamina, is_defending
    """
    diff = difficulty_config or {}

    stamina_factor = attacker.get("current_stamina", 25) / max(
        attacker.get("max_stamina", 50), 1
    )

    hit_prob = compute_hit_probability(
        attacker["base_attack"],
        attacker.get("weapon_modifier", 0),
        stamina_factor,
        defender["base_defense"],
        defender.get("armor_modifier", 0),
    )

    effects: dict[str, Any] = {}

    if random.random() < hit_prob:
        # Determine damage multiplier
        dmg_mult = 1.0
        if attacker.get("is_player", False):
            dmg_mult *= diff.get("combat_damage_from_player", 1.0)
        if defender.get("is_player", False):
            dmg_mult *= diff.get("combat_damage_to_player", 1.0)

        # Defending reduces damage by 50%
        if defender.get("is_defending", False):
            dmg_mult *= (1.0 - COMBAT_DEFEND_REDUCTION)

        damage = compute_damage(
            attacker["base_attack"],
            attacker.get("weapon_modifier", 0),
            defender["base_defense"],
            defender.get("armor_modifier", 0),
            dmg_mult,
        )

        effects["damage"] = damage
        effects["hit"] = True

        narrative = (
            f"{attacker['name']} strikes {defender['name']} for {damage} damage!"
        )
        return CombatResult(
            hit=True,
            damage=damage,
            attacker_name=attacker["name"],
            defender_name=defender["name"],
            attacker_stamina_cost=10,
            defender_stamina_cost=5,
            narrative=narrative,
            effects=effects,
        )
    else:
        effects["damage"] = 0
        effects["hit"] = False
        narrative = (
            f"{attacker['name']} swings at {defender['name']} but misses!"
        )
        return CombatResult(
            hit=False,
            damage=0,
            attacker_name=attacker["name"],
            defender_name=defender["name"],
            attacker_stamina_cost=10,
            defender_stamina_cost=2,
            narrative=narrative,
            effects=effects,
        )


def resolve_flee(
    fleeing_entity: dict,
    opponent: dict,
    difficulty_config: dict | None = None,
) -> dict:
    """
    Resolve a flee attempt.

    Returns dict with: success, narrative, free_attack (CombatResult|None)
    """
    diff = difficulty_config or {}
    base_success = diff.get("combat_flee_success_rate", COMBAT_FLEE_BASE_SUCCESS)

    # At 0 stamina, success drops to 40%
    if fleeing_entity.get("current_stamina", 25) <= 0:
        base_success = COMBAT_FLEE_EXHAUSTED_SUCCESS

    if random.random() < base_success:
        return {
            "success": True,
            "narrative": f"{fleeing_entity['name']} successfully flees from combat!",
            "free_attack": None,
        }
    else:
        # Failed flee — opponent gets free attack at +20% hit
        boosted_opponent = dict(opponent)
        boosted_opponent["base_attack"] = opponent.get("base_attack", 8)
        free_hit_prob = compute_hit_probability(
            boosted_opponent["base_attack"],
            boosted_opponent.get("weapon_modifier", 0),
            0.5,
            fleeing_entity.get("base_defense", 3),
            fleeing_entity.get("armor_modifier", 0),
        )
        # Boost by 20% for free attack
        free_hit_prob = clamp(free_hit_prob + COMBAT_FLEE_BONUS_HIT, COMBAT_HIT_MIN, COMBAT_HIT_MAX)

        if random.random() < free_hit_prob:
            damage = compute_damage(
                boosted_opponent["base_attack"],
                boosted_opponent.get("weapon_modifier", 0),
                fleeing_entity.get("base_defense", 3),
                fleeing_entity.get("armor_modifier", 0),
            )
            free_attack = CombatResult(
                hit=True,
                damage=damage,
                attacker_name=opponent["name"],
                defender_name=fleeing_entity["name"],
                attacker_stamina_cost=0,
                defender_stamina_cost=0,
                narrative=f"{fleeing_entity['name']} fails to flee! {opponent['name']} strikes for {damage} damage!",
                effects={"damage": damage, "hit": True},
            )
        else:
            free_attack = CombatResult(
                hit=False,
                damage=0,
                attacker_name=opponent["name"],
                defender_name=fleeing_entity["name"],
                attacker_stamina_cost=0,
                defender_stamina_cost=0,
                narrative=f"{fleeing_entity['name']} fails to flee! {opponent['name']} swings but misses!",
                effects={"damage": 0, "hit": False},
            )

        return {
            "success": False,
            "narrative": f"{fleeing_entity['name']} tries to flee but is blocked!",
            "free_attack": free_attack,
        }


def compute_skill_probability(
    action_id: str,
    reputation: int = 0,
    social_modifier: int = 0,
    time_bonus: int = 0,
    npcs_at_location: int = 0,
    search_count: int = 0,
) -> float:
    """Compute success probability for skill-check actions."""
    match action_id:
        case "persuade":
            return clamp(0.5 + reputation / 200 + social_modifier / 20, 0.1, 0.9)
        case "deceive":
            return clamp(0.4 - reputation / 200 + social_modifier / 20, 0.05, 0.85)
        case "sneak":
            return clamp(0.5 + time_bonus / 10 - npcs_at_location / 10, 0.1, 0.9)
        case "steal":
            sneak_prob = clamp(0.5 + time_bonus / 10 - npcs_at_location / 10, 0.1, 0.9)
            return sneak_prob * 0.8
        case "hide":
            return clamp(0.6 + time_bonus / 10 - npcs_at_location / 10, 0.15, 0.95)
        case "search":
            return clamp(0.3 + (search_count / 5) * 0.1, 0.2, 0.8)
        case _:
            return 0.5
