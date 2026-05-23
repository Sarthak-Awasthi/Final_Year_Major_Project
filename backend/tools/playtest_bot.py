"""
playtest_bot.py — Automated playtest bot for research / testing.

Runs the game engine autonomously using configurable strategies
to generate session data for analysis.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from backend.config import (
    MASTER_SEED,
    MAX_TURNS,
    UNIVERSAL_ACTION_IDS,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.engine.game_engine import GameEngine

# ─── Strategy definitions ────────────────────────────────────────────────────

STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "random": {cat: 1.0 for cat in ("navigation", "exploration", "social", "combat", "stealth", "utility")},
    "quest_focused": {
        "navigation": 2.0,
        "exploration": 2.5,
        "social": 2.0,
        "combat": 1.0,
        "stealth": 0.5,
        "utility": 1.5,
    },
    "aggressive": {
        "navigation": 1.0,
        "exploration": 0.5,
        "social": 0.3,
        "combat": 5.0,
        "stealth": 0.5,
        "utility": 0.5,
    },
    "social": {
        "navigation": 1.0,
        "exploration": 1.0,
        "social": 5.0,
        "combat": 0.1,
        "stealth": 0.3,
        "utility": 1.0,
    },
    "explorer": {
        "navigation": 3.0,
        "exploration": 5.0,
        "social": 1.0,
        "combat": 0.2,
        "stealth": 1.5,
        "utility": 1.0,
    },
}

VALID_STRATEGIES = set(STRATEGY_WEIGHTS.keys())

# Pre-build category → action_id lookup
_CATEGORY_ACTIONS: dict[str, list[str]] = {}
for _aid, _meta in UNIVERSAL_ACTIONS.items():
    _CATEGORY_ACTIONS.setdefault(_meta["category"], []).append(_aid)


class PlaytestBot:
    """AI agent that plays the game automatically for research/testing.

    Strategies:
        - ``"random"``: uniform random across all 27 actions.
        - ``"quest_focused"``: prefer exploration/social/navigation.
        - ``"aggressive"``: prefer combat and intimidation.
        - ``"social"``: prefer social interactions.
        - ``"explorer"``: prefer look/search/examine/movement.
    """

    def __init__(
        self,
        strategy: str = "random",
        seed: int = MASTER_SEED,
        difficulty: str = "normal",
    ) -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {sorted(VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self.seed = seed
        self.difficulty = difficulty
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # Session tracking
        self.actions_taken: list[dict[str, Any]] = []
        self.turn_results: list[dict[str, Any]] = []

    # ─── Main run loop ───────────────────────────────────────────────────

    async def run(self, max_turns: int = MAX_TURNS) -> dict[str, Any]:
        """Create a game, initialise it, and play autonomously.

        Args:
            max_turns: Maximum number of turns to play.

        Returns:
            Session summary dict containing strategy, seed, total_turns,
            game_result, actions_taken count, and a metrics snapshot.
        """
        self.engine = GameEngine(seed=self.seed, difficulty=self.difficulty, max_turns=max_turns)
        engine = self.engine
        init_result = await engine.initialize()
        logger.info(
            "PlaytestBot started: strategy=%s, seed=%d, max_turns=%d",
            self.strategy,
            self.seed,
            max_turns,
        )

        start_time = time.monotonic()

        while not engine.game_over and engine.turn < max_turns:
            state = engine.get_full_state()

            # Select and build parsed input
            parsed_input = self._select_action(state, self.strategy)
            self.actions_taken.append(parsed_input)

            try:
                result = await engine.process_turn(parsed_input)
                self.turn_results.append(result)
            except Exception as exc:
                logger.error(
                    "PlaytestBot turn %d error: %s", engine.turn, exc, exc_info=True
                )
                # Record error and continue
                self.turn_results.append({"error": str(exc), "turn": engine.turn})

        elapsed = time.monotonic() - start_time

        # Build summary
        final_state = engine.get_full_state()
        summary: dict[str, Any] = {
            "strategy": self.strategy,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "total_turns": engine.turn,
            "max_turns": max_turns,
            "game_over": engine.game_over,
            "game_result": engine.game_result,
            "elapsed_seconds": round(elapsed, 2),
            "actions_taken": len(self.actions_taken),
            "final_player": final_state.get("player", {}),
            "final_quest": final_state.get("quest", {}),
            "event_log_size": len(engine.event_log.entries),
            "metrics": engine.get_metrics(),
        }

        logger.info(
            "PlaytestBot finished: %d turns in %.1fs, result=%s",
            engine.turn,
            elapsed,
            engine.game_result,
        )
        return summary

    # ─── Action selection ────────────────────────────────────────────────

    def _select_action(self, state: dict[str, Any], strategy: str) -> dict[str, Any]:
        """Pick an action based on the current strategy and game state.

        Args:
            state: Full game state from ``GameEngine.get_full_state()``.
            strategy: One of the valid strategy names.

        Returns:
            A ``ParsedInput`` dict suitable for ``GameEngine.process_turn()``.
        """
        weights = STRATEGY_WEIGHTS[strategy]

        # Build weighted list of (action_id, weight) pairs
        action_weights: list[tuple[str, float]] = []
        for category, cat_actions in _CATEGORY_ACTIONS.items():
            w = weights.get(category, 1.0)
            for aid in cat_actions:
                action_weights.append((aid, w))

        action_ids = [aw[0] for aw in action_weights]
        raw_weights = [aw[1] for aw in action_weights]

        # Boost combat actions if player is already in combat
        player = state.get("player", {})
        if player.get("in_combat"):
            for i, aid in enumerate(action_ids):
                if UNIVERSAL_ACTIONS[aid]["category"] == "combat":
                    raw_weights[i] *= 3.0

        # Suppress move_to if no adjacency info (shouldn't happen, but safe)
        location_info = state.get("location", {})
        adjacent = location_info.get("adjacent", [])
        if not adjacent:
            for i, aid in enumerate(action_ids):
                if aid == "move_to":
                    raw_weights[i] = 0.0

        # Normalise weights
        total = sum(raw_weights)
        if total <= 0:
            # Fallback: uniform
            probs = [1.0 / len(action_ids)] * len(action_ids)
        else:
            probs = [w / total for w in raw_weights]

        chosen_action: str = self._rng.choices(action_ids, weights=probs, k=1)[0]

        # Build target info
        targets = self._pick_target(chosen_action, state)

        parsed_input: dict[str, Any] = {
            "source": "button",
            "raw_text": None,
            "action_id": chosen_action,
            "target_npc": targets.get("target_npc"),
            "target_item": targets.get("target_item"),
            "target_location": targets.get("target_location"),
            "confidence": 1.0,
            "emotion": "neutral",
            "intent": chosen_action,
            "social": "neutral",
        }

        return parsed_input

    # ─── Target picking ──────────────────────────────────────────────────

    def _pick_target(self, action_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """Choose appropriate targets based on action type and game context.

        Args:
            action_id: The chosen universal action ID.
            state: Full game state dict.

        Returns:
            Dict with optional keys ``target_npc``, ``target_item``,
            ``target_location``.
        """
        targets: dict[str, Any] = {
            "target_npc": None,
            "target_item": None,
            "target_location": None,
        }

        npcs_here: list[dict[str, Any]] = state.get("npcs_here", [])
        player: dict[str, Any] = state.get("player", {})
        inventory: list[dict[str, Any]] = player.get("inventory", [])
        location_info: dict[str, Any] = state.get("location", {})
        adjacent: list[str] = location_info.get("adjacent", [])
        items_on_ground: list[dict[str, Any]] = location_info.get("items_on_ground", [])

        category = UNIVERSAL_ACTIONS[action_id]["category"]

        # Navigation — pick a random adjacent location
        if action_id == "move_to" and adjacent:
            targets["target_location"] = self._rng.choice(adjacent)

        # Social / combat actions that need an NPC target
        elif category in ("social", "combat") and npcs_here:
            # Filter to active NPCs for combat
            if category == "combat":
                active = [n for n in npcs_here if n.get("status") == "active"]
                target_pool = active if active else npcs_here
            else:
                target_pool = npcs_here
            chosen_npc = self._rng.choice(target_pool)
            targets["target_npc"] = chosen_npc.get("npc_uid")

        # Stealth actions may target NPCs or items
        elif category == "stealth":
            if action_id == "steal" and npcs_here:
                targets["target_npc"] = self._rng.choice(npcs_here).get("npc_uid")

        # Item actions
        elif action_id in ("use_item", "eat", "drop_item", "equip"):
            if inventory:
                # Prefer consumables for eat, equipment for equip
                if action_id == "eat":
                    consumables = [i for i in inventory if i.get("type") == "consumable"]
                    pool = consumables if consumables else inventory
                elif action_id == "equip":
                    equipment = [i for i in inventory if i.get("type") == "equipment"]
                    pool = equipment if equipment else inventory
                elif action_id == "drop_item":
                    droppable = [i for i in inventory if not i.get("quest_relevant")]
                    pool = droppable if droppable else []
                else:
                    pool = inventory

                if pool:
                    targets["target_item"] = self._rng.choice(pool).get("id")

        elif action_id == "pick_up" and items_on_ground:
            targets["target_item"] = self._rng.choice(items_on_ground).get("id", None)

        elif action_id == "give_item" and npcs_here and inventory:
            non_quest = [i for i in inventory if not i.get("quest_relevant")]
            if non_quest:
                targets["target_item"] = self._rng.choice(non_quest).get("id")
                targets["target_npc"] = self._rng.choice(npcs_here).get("npc_uid")

        elif action_id == "trade" and npcs_here:
            targets["target_npc"] = self._rng.choice(npcs_here).get("npc_uid")

        return targets
