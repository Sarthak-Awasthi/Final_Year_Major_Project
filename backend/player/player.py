"""
player.py — Player state, stats, inventory, per-NPC reputation.
"""

from __future__ import annotations

from typing import Any

from backend.config import (
    PLAYER_BASE_ATTACK,
    PLAYER_BASE_DEFENSE,
    PLAYER_MAX_HEALTH,
    PLAYER_MAX_INVENTORY,
    PLAYER_MAX_STAMINA,
    REPUTATION_DECAY_AMOUNT,
    REPUTATION_DECAY_INTERVAL,
    REPUTATION_MAX,
    REPUTATION_MIN,
    logger,
)


class Player:
    """Player state and operations."""

    def __init__(self) -> None:
        self.name: str = "Traveler"
        self.health: int = PLAYER_MAX_HEALTH
        self.max_health: int = PLAYER_MAX_HEALTH
        self.stamina: int = PLAYER_MAX_STAMINA
        self.max_stamina: int = PLAYER_MAX_STAMINA
        self.combat_stats: dict[str, int] = {
            "base_attack": PLAYER_BASE_ATTACK,
            "base_defense": PLAYER_BASE_DEFENSE,
            "weapon_modifier": 0,
            "armor_modifier": 0,
        }
        self.equipped: dict[str, str | None] = {
            "weapon": None,
            "armor": None,
        }
        self.reputation: dict[str, int] = {}  # npc_uid -> reputation
        self.global_reputation: int = 0
        self.inventory: list[dict] = []
        self.max_inventory: int = PLAYER_MAX_INVENTORY
        self.location: str = "gate"
        self.quest_state: dict[str, Any] = {
            "quest_id": "main_quest_01",
            "current_stage": 1,
            "current_checkpoint": "1_1",
            "completed_checkpoints": [],
            "dynamic_checkpoints": [],
            "deviation_count": 0,
        }
        self.is_defending: bool = False
        self.in_combat: bool = False
        self.combat_target: str | None = None
        self.queued_action: dict | None = None
        self._greeted_npcs: set[str] = set()

        # Initialize starting inventory
        self._init_starting_inventory()

    def _init_starting_inventory(self) -> None:
        """Set up the player's starting items."""
        self.inventory = [
            {
                "id": "travel_papers",
                "name": "Travel Papers",
                "type": "quest",
                "quest_relevant": True,
                "description": "Official documents permitting travel to Thornhaven.",
                "effects": {},
                "slot": None,
                "stat_modifiers": None,
            },
            {
                "id": "bread",
                "name": "Stale Bread",
                "type": "consumable",
                "quest_relevant": False,
                "description": "A day-old loaf. Restores 5 HP.",
                "effects": {"heal": 5},
                "slot": None,
                "stat_modifiers": None,
            },
            {
                "id": "coin_pouch",
                "name": "Coin Pouch",
                "type": "misc",
                "quest_relevant": False,
                "description": "A small pouch with 10 copper coins. Useful for trading.",
                "effects": {},
                "slot": None,
                "stat_modifiers": None,
            },
        ]

    # ─── Health ──────────────────────────────────────────────────────────────
    def modify_health(self, amount: int) -> int:
        """Change health, clamped to [0, max_health]. Returns actual change."""
        old = self.health
        self.health = max(0, min(self.max_health, self.health + amount))
        return self.health - old

    def is_alive(self) -> bool:
        """Check if player is alive."""
        return self.health > 0

    # ─── Stamina ─────────────────────────────────────────────────────────────
    def modify_stamina(self, amount: int) -> int:
        """Change stamina, clamped to [0, max_stamina]. Returns actual change."""
        old = self.stamina
        self.stamina = max(0, min(self.max_stamina, self.stamina + amount))
        return self.stamina - old

    def can_afford_ap(self, cost: int) -> bool:
        """Check if player has enough AP for an action."""
        return self.stamina >= cost

    def regen_stamina(self, amount: int) -> int:
        """Passive stamina regeneration. Returns actual regen."""
        return self.modify_stamina(amount)

    # ─── Reputation ──────────────────────────────────────────────────────────
    def get_reputation(self, npc_uid: str) -> int:
        """Get reputation with a specific NPC."""
        return self.reputation.get(npc_uid, 0)

    def modify_reputation(self, npc_uid: str, amount: int) -> int:
        """Change reputation with an NPC. Returns actual change."""
        old = self.reputation.get(npc_uid, 0)
        new = max(REPUTATION_MIN, min(REPUTATION_MAX, old + amount))
        self.reputation[npc_uid] = new
        self._update_global_reputation()
        return new - old

    def _update_global_reputation(self) -> None:
        """Recompute global reputation as average of all per-NPC reps."""
        if not self.reputation:
            self.global_reputation = 0
            return
        self.global_reputation = sum(self.reputation.values()) // len(self.reputation)

    def get_reputation_label(self, npc_uid: str) -> str:
        """Get the reputation label for a specific NPC."""
        rep = self.get_reputation(npc_uid)
        if rep >= 50:
            return "trusted"
        elif rep >= 20:
            return "friendly"
        elif rep >= -19:
            return "neutral"
        elif rep >= -49:
            return "suspicious"
        else:
            return "hostile"

    def apply_reputation_decay(self, current_turn: int) -> dict[str, int]:
        """Decay all reputations toward 0 every N turns. Returns changes."""
        if current_turn % REPUTATION_DECAY_INTERVAL != 0:
            return {}
        changes = {}
        for npc_uid, rep in list(self.reputation.items()):
            if rep > 0:
                self.reputation[npc_uid] = max(0, rep - REPUTATION_DECAY_AMOUNT)
                changes[npc_uid] = -REPUTATION_DECAY_AMOUNT
            elif rep < 0:
                self.reputation[npc_uid] = min(0, rep + REPUTATION_DECAY_AMOUNT)
                changes[npc_uid] = REPUTATION_DECAY_AMOUNT
        self._update_global_reputation()
        return changes

    def has_greeted(self, npc_uid: str) -> bool:
        """Check if player has previously greeted an NPC."""
        return npc_uid in self._greeted_npcs

    def mark_greeted(self, npc_uid: str) -> None:
        """Mark an NPC as greeted."""
        self._greeted_npcs.add(npc_uid)

    # ─── Inventory ───────────────────────────────────────────────────────────
    def add_item(self, item: dict) -> bool:
        """Add item to inventory. Returns False if full."""
        if len(self.inventory) >= self.max_inventory:
            return False
        self.inventory.append(item)
        return True

    def remove_item(self, item_id: str) -> dict | None:
        """Remove and return item by ID. Returns None if not found."""
        for i, item in enumerate(self.inventory):
            if item["id"] == item_id:
                return self.inventory.pop(i)
        return None

    def has_item(self, item_id: str) -> bool:
        """Check if player has an item."""
        return any(it["id"] == item_id for it in self.inventory)

    def get_item(self, item_id: str) -> dict | None:
        """Get item by ID without removing it."""
        for item in self.inventory:
            if item["id"] == item_id:
                return item
        return None

    def get_droppable_items(self) -> list[dict]:
        """Get items that can be dropped (non-quest items)."""
        return [it for it in self.inventory if not it.get("quest_relevant", False)]

    def get_food_items(self) -> list[dict]:
        """Get consumable food items."""
        return [it for it in self.inventory if it.get("type") == "consumable" and "heal" in it.get("effects", {})]

    def get_equipment_items(self) -> list[dict]:
        """Get equipment items not currently equipped."""
        equipped_ids = {v for v in self.equipped.values() if v}
        return [it for it in self.inventory if it.get("type") == "equipment" and it["id"] not in equipped_ids]

    def inventory_full(self) -> bool:
        """Check if inventory is full."""
        return len(self.inventory) >= self.max_inventory

    # ─── Equipment ───────────────────────────────────────────────────────────
    def equip_item(self, item_id: str) -> dict | None:
        """
        Equip an item. Returns previous item in slot (or None).
        Updates combat stat modifiers.
        """
        item = self.get_item(item_id)
        if not item or item.get("type") != "equipment":
            return None

        slot = item.get("slot")  # "weapon" or "armor"
        if not slot or slot not in ("weapon", "armor"):
            return None

        # Unequip current item in that slot
        previous_id = self.equipped.get(slot)
        previous_item = None
        if previous_id:
            previous_item = self.get_item(previous_id)
            if previous_item and previous_item.get("stat_modifiers"):
                for stat, mod in previous_item["stat_modifiers"].items():
                    modifier_key = f"{stat}_modifier"
                    if modifier_key in self.combat_stats:
                        self.combat_stats[modifier_key] -= mod

        # Equip new item
        self.equipped[slot] = item_id
        if item.get("stat_modifiers"):
            for stat, mod in item["stat_modifiers"].items():
                modifier_key = f"{stat}_modifier"
                if modifier_key in self.combat_stats:
                    self.combat_stats[modifier_key] += mod

        return previous_item

    def get_combat_dict(self) -> dict:
        """Get combat stats as a dict for combat resolution."""
        return {
            "name": self.name,
            "base_attack": self.combat_stats["base_attack"],
            "base_defense": self.combat_stats["base_defense"],
            "weapon_modifier": self.combat_stats["weapon_modifier"],
            "armor_modifier": self.combat_stats["armor_modifier"],
            "current_stamina": self.stamina,
            "max_stamina": self.max_stamina,
            "is_player": True,
            "is_defending": self.is_defending,
        }

    # ─── Serialization ───────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Serialize player state."""
        return {
            "name": self.name,
            "health": self.health,
            "max_health": self.max_health,
            "stamina": self.stamina,
            "max_stamina": self.max_stamina,
            "combat_stats": self.combat_stats.copy(),
            "equipped": self.equipped.copy(),
            "reputation": self.reputation.copy(),
            "global_reputation": self.global_reputation,
            "inventory": [it.copy() for it in self.inventory],
            "max_inventory": self.max_inventory,
            "location": self.location,
            "quest_state": self.quest_state.copy(),
            "in_combat": self.in_combat,
            "combat_target": self.combat_target,
            "greeted_npcs": list(self._greeted_npcs),
        }

    def from_dict(self, data: dict) -> None:
        """Restore player state from dict."""
        self.name = data.get("name", "Traveler")
        self.health = data.get("health", PLAYER_MAX_HEALTH)
        self.max_health = data.get("max_health", PLAYER_MAX_HEALTH)
        self.stamina = data.get("stamina", PLAYER_MAX_STAMINA)
        self.max_stamina = data.get("max_stamina", PLAYER_MAX_STAMINA)
        self.combat_stats = data.get("combat_stats", {
            "base_attack": PLAYER_BASE_ATTACK,
            "base_defense": PLAYER_BASE_DEFENSE,
            "weapon_modifier": 0,
            "armor_modifier": 0,
        })
        self.equipped = data.get("equipped", {"weapon": None, "armor": None})
        self.reputation = data.get("reputation", {})
        self.global_reputation = data.get("global_reputation", 0)
        self.inventory = data.get("inventory", [])
        self.max_inventory = data.get("max_inventory", PLAYER_MAX_INVENTORY)
        self.location = data.get("location", "gate")
        self.quest_state = data.get("quest_state", {
            "quest_id": "main_quest_01",
            "current_stage": 1,
            "current_checkpoint": "1_1",
            "completed_checkpoints": [],
            "dynamic_checkpoints": [],
            "deviation_count": 0,
        })
        self.in_combat = data.get("in_combat", False)
        self.combat_target = data.get("combat_target", None)
        self._greeted_npcs = set(data.get("greeted_npcs", []))
