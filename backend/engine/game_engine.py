"""GameEngine — orchestrates world, player, NPCs, quest, events, and LLM.

One instance per game. `process_turn` drives the main loop: resolve player
action → quest progress → NPC turns → random events → time/regen → game-over check.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import backend.config as _cfg
from backend.config import (
    AUTO_SAVE_INTERVAL,
    GAME_VERSION,
    INCAPACITATION_TURNS,
    INDOOR_LOCATIONS,
    LOCATION_ADJACENCY,
    MASTER_SEED,
    MAX_AUTO_SAVES,
    MAX_MANUAL_SAVES,
    MAX_TURNS,
    METRICS_DIR,
    NPC_COLD_START_TURNS,
    NPC_INDOOR_REGEN,
    QUEST_DIR,
    SAVE_VERSION,
    SAVES_DIR,
    SHOCK_ENABLED,
    SOCIAL_MODIFIERS,
    STAMINA_REGEN_PER_TURN,
    UNIVERSAL_ACTION_IDS,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.engine.combat import (
    CombatResult,
    compute_skill_probability,
    resolve_attack,
    resolve_flee,
)
from backend.engine.difficulty import DifficultyConfig, assess_player_struggle
from backend.engine.event_log import EventLog, compute_importance, detect_witnesses
from backend.engine.events import RandomEventSystem
from backend.engine.narration import (
    add_context_modifiers,
    enhance_narration_with_llm,
    filter_npc_narration,
    get_template_narration,
    passive_perception_check,
)
from backend.engine.playthrough_logger import (
    PlaythroughLogger,
    build_world_snapshot,
)
from backend.engine.shock_manager import ShockManager
from backend.engine.world import World
from backend.llm.llm_service import LLMService
from backend.npc.dialogue import format_dialogue, resolve_dialogue
from backend.npc.interactions import (
    propagate_gossip,
    resolve_npc_npc_interaction,
    resolve_npc_target,
)
from backend.npc.knowledge import add_witnessed_event
from backend.npc.npc import NPC
from backend.npc.personality import (
    create_npc_registry,
    get_npcs_at_location,
    load_archetypes,
)
from backend.npc.rl_agent import (
    compute_reward,
    decay_epsilon,
    get_valid_actions,
    pretrain_npc,
    select_action,
    update_q_table,
)
from backend.npc.schedule import get_movement_destination, get_scheduled_action
from backend.player.player import Player
from backend.quest.checkpoint import generate_dynamic_checkpoint
from backend.quest.mdp import QuestMDP
from backend.quest.nudge import compute_nudge_reward, get_nudge_hint
from backend.quest.quest_manager import QuestManager


def _npc_names_map(registry: dict[str, NPC]) -> dict[str, str]:
    return {uid: npc.name for uid, npc in registry.items()}


def _npc_locations_map(registry: dict[str, NPC]) -> dict[str, str]:
    return {uid: npc.location for uid, npc in registry.items()}


class GameEngine:
    """Central game orchestrator. One instance per session."""

    def __init__(
        self,
        seed: int = MASTER_SEED,
        difficulty: str = "normal",
        max_turns: int = MAX_TURNS,
        restart_on_complete: bool = True,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.max_turns = max_turns
        self.turn: int = 0
        self.game_over: bool = False
        self.game_result: str | None = None  # "success" / "fail" / "turn_limit"
        self.pending_defeat_reason: str | None = None  # scripted defeat (e.g. elder betrayal)
        self.game_over_message: str | None = None  # narration for the defeat/victory screen
        # `restart_on_complete=True` loops the quest after S_success so RL
        # notebooks keep training to max_turns. The live demo sets it to
        # False so a successful quest ends the game cleanly.
        self.restart_on_complete: bool = restart_on_complete
        self.last_interacted_npc_uid: str | None = None
        self._last_dialogue: dict[str, str] = {}
        self.interacted_npc_uids: set[str] = set()

        self.world = World()
        self.player = Player()
        self.difficulty = DifficultyConfig(difficulty)
        self.event_log = EventLog()
        self.random_events = RandomEventSystem()
        self.llm = LLMService()  # safe no-op if no LLM is loaded
        self.shock_manager = ShockManager()
        self._prev_community_state: dict | None = None

        quest_path = QUEST_DIR / "main_quest.json"
        with open(quest_path, "r", encoding="utf-8") as f:
            quest_data = json.load(f)
        self.mdp = QuestMDP(quest_data)
        self.quest_manager = QuestManager(self.mdp)
        # Item registry indexed by id so reward strings can be resolved
        # back to the full item dict at give_item / pick_up time.
        self._quest_items: dict[str, dict] = quest_data.get("items", {})

        self.npc_registry: dict[str, NPC] = create_npc_registry(seed)
        for uid in self.npc_registry:
            self.player.reputation.setdefault(uid, 0)

        self._pretrained: bool = False
        self._auto_save_counter: int = 0
        self._auto_save_files: list[str] = []
        # Reset each turn — used to suppress duplicate gossip propagation.
        self._gossip_pairs_this_turn: set[tuple[str, str]] = set()

        self._metrics: dict[str, Any] = {
            "total_actions": 0,
            "actions_by_type": {},
            "combat_encounters": 0,
            "dynamic_checkpoints_created": 0,
            "npcs_incapacitated": 0,
            "reputation_changes": 0,
            "llm_calls": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }

        session_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.playthrough_logger = PlaythroughLogger(
            session_id=f"{session_ts}_{seed}"
        )

        logger.info(
            "GameEngine initialised: seed=%d, difficulty=%s, max_turns=%d",
            seed,
            difficulty,
            max_turns,
        )

    async def initialize(self) -> dict:
        """Pre-train NPCs, build the opening narration, return initial state."""
        self._pretrain_npcs()
        initial_state = self.get_full_state()

        quest_title = self.mdp.title
        stage_desc = ""
        cp_desc = ""
        try:
            stage = self.mdp.stages.get(self.quest_manager.current_stage)
            if stage:
                stage_desc = stage.description
            cp = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
            if cp:
                cp_desc = cp.description
        except Exception:
            pass

        opening = (
            f"You are {self.player.name}, a traveler who has journeyed far to reach "
            f"the village of Thornhaven. Word has spread of {quest_title}. "
        )
        if stage_desc:
            opening += f"{stage_desc} "
        if cp_desc:
            opening += f"\n\n{cp_desc}"

        initial_state["opening_narration"] = opening.strip()

        self.event_log.add_entry(
            turn=0,
            time_of_day=self.world.time_of_day,
            event_type="player_action",
            actor="player",
            action="arrive",
            target=None,
            location=self.player.location,
            outcome="success",
            effects={},
            witnesses=detect_witnesses(
                self.player.location,
                "player",
                _npc_locations_map(self.npc_registry),
            ),
            narration=opening,
            importance=3,
        )

        try:
            self.playthrough_logger.log_event("game_start", {
                "seed": self.seed,
                "difficulty": self.difficulty.preset,
                "max_turns": self.max_turns,
                "player_name": self.player.name,
                "opening_narration": opening.strip(),
                "initial_location": self.player.location,
                "npc_count": len(self.npc_registry),
            })
        except Exception as exc:
            logger.error("PlaythroughLogger game_start failed: %s", exc)

        return initial_state

    def _pretrain_npcs(self) -> None:
        """Q-learning warm-up for each NPC. Skips when RL is disabled (ablation C3)."""
        if self._pretrained:
            return
        if not _cfg.RL_ENABLED:
            self._pretrained = True
            logger.info("RL disabled — skipping NPC pre-training")
            return
        world_data = {
            "locations": LOCATION_ADJACENCY,
            "indoor": list(INDOOR_LOCATIONS),
        }
        # Sort for deterministic per-NPC seeding regardless of dict insertion order.
        sorted_uids = sorted(self.npc_registry.keys())
        for npc_index, uid in enumerate(sorted_uids):
            npc = self.npc_registry[uid]
            pretrain_npc(npc, world_data, npc_index=npc_index, master_seed=self.seed)
        self._pretrained = True
        logger.info("NPC pre-training complete for %d NPCs", len(self.npc_registry))

    async def process_turn(self, parsed_input: Mapping[str, Any]) -> dict:
        """Drive one full turn: player action → quest progress → NPC turns →
        random events → time advance → regen → game-over check."""
        if self.game_over:
            return {
                "error": "Game is over.",
                "game_result": self.game_result,
                "state": self.get_full_state(),
            }

        self.turn += 1
        self._gossip_pairs_this_turn.clear()

        # Snapshot inventory BEFORE the action runs. Action handlers like
        # _resolve_give_item and _resolve_present_item consume the item
        # before the quest progress check runs, which would otherwise
        # make `requires.item` checks fail for the very transition the
        # player just satisfied.
        self._pre_action_inventory = [dict(itm) for itm in self.player.inventory]

        # ── 1. Resolve player action ──────────────────────────────────
        # Offload to thread — _resolve_player_action is sync but may
        # invoke blocking LLM calls (dialogue, checkpoint generation).
        action_result = await asyncio.to_thread(
            self._resolve_player_action, parsed_input
        )
        action_id = action_result["action_id"]

        # Track metrics
        self._metrics["total_actions"] += 1
        self._metrics["actions_by_type"][action_id] = (
            self._metrics["actions_by_type"].get(action_id, 0) + 1
        )

        # ── 2. Log player action event ────────────────────────────────
        witnesses = detect_witnesses(
            self.player.location,
            "player",
            _npc_locations_map(self.npc_registry),
        )
        importance = compute_importance(
            "player_action",
            action_id,
            "success" if action_result["success"] else "fail",
            action_result.get("effects", {}),
        )

        # Add context modifiers to narration
        narration = action_result["narration"]
        weather = self._get_active_weather()

        # Layer 2: LLM narration enhancement (optional)
        # Skip LLM enhancement for social/dialogue actions — the dialogue
        # pipeline already produces the NPC's response.  LLM-enhancing the
        # narration template for these results in bloated text that drowns
        # out or duplicates the real dialogue.
        _social_actions = {"talk", "greet", "ask_info", "persuade", "deceive", "intimidate", "trade"}
        llm_enhanced = False
        if action_id not in _social_actions:
            raw_target = action_result.get("target")
            target_display = _npc_names_map(self.npc_registry).get(raw_target, raw_target) if raw_target else None

            enhanced = await asyncio.to_thread(
                enhance_narration_with_llm,
                template_narration=narration,
                action_id=action_id,
                actor_name=self.player.name,
                target_name=target_display,
                outcome_type="success" if action_result["success"] else "fail",
                location=self.player.location,
                time_of_day=self.world.time_of_day,
                weather=weather,
                emotion=parsed_input.get("emotion", "neutral"),
                social=parsed_input.get("social", "neutral"),
                witnesses=[_npc_names_map(self.npc_registry).get(w, w) for w in witnesses],
                llm_service=self.llm,
            )
            if enhanced != narration:
                llm_enhanced = True
                narration = enhanced

        # Layer 3: Context modifiers
        narration = add_context_modifiers(
            narration,
            self.world.time_of_day,
            weather=weather,
            witnesses=witnesses,
            npc_names=_npc_names_map(self.npc_registry),
            llm_enhanced=llm_enhanced,
        )
        action_result["narration"] = narration

        event_entry = self.event_log.add_entry(
            turn=self.turn,
            time_of_day=self.world.time_of_day,
            event_type="player_action",
            actor="player",
            action=action_id,
            target=action_result.get("target"),
            location=self.player.location,
            outcome="success" if action_result["success"] else "fail",
            effects=action_result.get("effects", {}),
            witnesses=witnesses,
            narration=narration,
            importance=importance,
        )

        # Notify NPC witnesses
        for w_uid in witnesses:
            npc_w = self.npc_registry.get(w_uid)
            if npc_w:
                add_witnessed_event(npc_w, event_entry, self.turn)

        # ── 3. Quest completion / deviation check ─────────────────────
        # Offload to thread — may invoke blocking LLM for dynamic checkpoints.
        quest_update = await asyncio.to_thread(
            self._check_quest_progress, action_id, parsed_input, action_result
        )

        # ── 4. Process NPC turns ──────────────────────────────────────
        npc_results = self._process_npc_turns()
        npc_narrations = self._filter_npc_narrations(npc_results)

        # ── 5. Random events ──────────────────────────────────────────
        new_events = self._check_random_events()

        # ── 5b. Shock engine tick ─────────────────────────────────────
        expired_shocks = []
        if SHOCK_ENABLED:
            expired_shocks = self.shock_manager.tick(self.turn)

            # P0 Fix: Apply ALL shock effects to NPC stats each turn
            if self.shock_manager.has_active_shocks:
                stat_drain = self.shock_manager.get_stat_drain()
                trust_mod = self.shock_manager.get_trust_modifier()

                for npc in self.npc_registry.values():
                    if npc.is_incapacitated():
                        continue

                    # Happiness drain (famine, plague, raids lower morale)
                    if stat_drain != 0.0:
                        npc.stats["happiness"] = max(0, min(10, npc.stats["happiness"] - stat_drain))

                    # Health drain (plague, harsh_winter damage HP directly)
                    if stat_drain > 0.3:
                        hp_loss = int(stat_drain * 2)  # Scale: drain 0.6 → 1 HP/turn
                        npc.modify_hp(-hp_loss)
                        npc.stats["health"] = max(0, min(10, npc.stats["health"] - stat_drain * 0.5))

                    # Trust/reputation modifier (bandit raids erode trust)
                    if trust_mod != 0.0:
                        for uid in self.npc_registry:
                            if uid != npc.npc_uid:
                                old_rel = npc.npc_relationships.get(uid, 0)
                                npc.npc_relationships[uid] = int(max(-100, min(100, old_rel + trust_mod)))
                        # Also affect player reputation toward this NPC
                        self.player.modify_reputation(npc.npc_uid, int(trust_mod))

        # ── 6. Advance time ───────────────────────────────────────────
        new_time = self.world.advance_turn()

        # Expire old events
        expired_events = self.world.expire_events()

        # ── 7. Passive stamina regen ──────────────────────────────────
        regen_amount = self.difficulty.get(
            "stamina_regen_per_turn", STAMINA_REGEN_PER_TURN
        )
        actual_regen = self.player.regen_stamina(regen_amount)

        # ── 8. Reputation decay ───────────────────────────────────────
        rep_decay = self.player.apply_reputation_decay(self.turn)

        # ── 9. Passive perception ─────────────────────────────────────
        perception = None
        loc = self.world.get_location(self.player.location)
        if loc:
            npcs_here = get_npcs_at_location(
                self.npc_registry, self.player.location
            )
            importance_map = {r["npc_uid"]: r.get("importance", 1) for r in npc_results}
            perception = passive_perception_check(
                self.player.location,
                self.world.is_social(self.player.location),
                loc.items_on_ground,
                [
                    {
                        "npc_uid": n.npc_uid,
                        "name": n.name,
                        "action_importance": importance_map.get(n.npc_uid, 1),
                    }
                    for n in npcs_here
                ],
            )

        # ── 10. Check game-over conditions ────────────────────────────
        result = self._check_game_over()
        if result is not None:
            self.game_over = True
            self.game_result = result
            if self.game_over_message:
                action_result["narration"] = (
                    (action_result.get("narration", "") + " " + self.game_over_message).strip()
                )
            self.save_game("auto")

        # ── 11. Auto-save ─────────────────────────────────────────────
        if not self.game_over:
            # Adaptive difficulty assessment (every 20 turns)
            if self.turn % 20 == 0 and self.turn > 0:
                entries = self.event_log.get_recent(20)
                adjustment = assess_player_struggle(entries, window=20)
                if adjustment != "maintain":
                    old_preset = self.difficulty.preset
                    presets_order = ["easy", "normal", "hard"]
                    current_idx = presets_order.index(old_preset) if old_preset in presets_order else 1
                    if adjustment == "decrease_difficulty" and current_idx > 0:
                        self.difficulty.apply_preset(presets_order[current_idx - 1])
                        logger.info("Adaptive difficulty: %s → %s", old_preset, self.difficulty.preset)
                    elif adjustment == "increase_difficulty" and current_idx < len(presets_order) - 1:
                        self.difficulty.apply_preset(presets_order[current_idx + 1])
                        logger.info("Adaptive difficulty: %s → %s", old_preset, self.difficulty.preset)

            self._auto_save()
            # Save before combat
            if action_id == "attack" or self.player.in_combat:
                self.save_game("auto")

        # Build turn result
        turn_result: dict[str, Any] = {
            "turn": self.turn,
            "time_period": new_time,
            "action_result": action_result,
            "quest_update": quest_update,
            "npc_narrations": npc_narrations,
            "new_events": new_events,
            "expired_events": expired_events,
            "stamina_regen": actual_regen,
            "reputation_decay": rep_decay,
            "perception": perception,
            "game_over": self.game_over,
            "game_result": self.game_result,
            "game_over_message": self.game_over_message,
            "state": self.get_full_state(),
        }

        # ── 12. Playthrough logging ───────────────────────────────────
        try:
            world_snapshot = build_world_snapshot(
                turn=self.turn,
                player=self.player,
                npc_registry=self.npc_registry,
                world=self.world,
                quest_manager=self.quest_manager,
                difficulty=self.difficulty,
            )
            self.playthrough_logger.log_turn(
                turn=self.turn,
                parsed_input=parsed_input,
                action_result=action_result,
                turn_result=turn_result,
                world_snapshot=world_snapshot,
            )
            # Phase 6: Log RL telemetry (reward decomposition, adaptation, shocks)
            self.playthrough_logger.log_rl_telemetry(
                turn=self.turn,
                npc_registry=self.npc_registry,
                shock_manager=self.shock_manager,
                community_state=self.compute_community_state(),
            )
        except Exception as exc:
            logger.error("PlaythroughLogger.log_turn failed: %s", exc)

        # Flush summary on game-over
        if self.game_over:
            try:
                self.playthrough_logger.flush_summary(turn_result.get("state"))
            except Exception as exc:
                logger.error("PlaythroughLogger.flush_summary failed: %s", exc)

        return turn_result

    # ── Player Action Resolution ──────────────────────────────────────────

    def _resolve_player_action(self, parsed_input: Mapping[str, Any]) -> dict:
        """Validate cost/preconditions, then dispatch to the action-specific resolver."""
        action_id: str = parsed_input.get("action_id") or "wait"
        target_npc: str | None = parsed_input.get("target_npc")
        target_item: str | None = parsed_input.get("target_item")
        target_location: str | None = parsed_input.get("target_location")
        emotion: str = parsed_input.get("emotion", "neutral")
        social: str = parsed_input.get("social", "neutral")

        # JS can send `target_npc: "undefined"` as a literal string — strip those.
        if target_npc and target_npc not in self.npc_registry:
            logger.warning("Invalid target_npc %r — clearing to None", target_npc)
            target_npc = None

        action_meta = UNIVERSAL_ACTIONS.get(action_id, {"base_ap": 0, "category": "utility"})
        base_ap: int = action_meta["base_ap"]
        ap_mult = self.difficulty.get("ap_cost_multiplier", 1.0)
        ap_cost = max(0, int(base_ap * ap_mult))

        # `defend` is the only action whose effect persists across turns;
        # any other action clears the standing-guard pose.
        if action_id != "defend":
            self.player.is_defending = False

        # Out of AP: still allow zero-cost actions, plus a free talk/greet
        # so the player isn't softlocked; in combat, defend/flee are free.
        if not self.player.can_afford_ap(ap_cost):
            free_allowed = ap_cost == 0
            if action_id in ("talk", "greet") and self.player.stamina <= 0:
                free_allowed = True
                ap_cost = 0
            if self.player.in_combat and action_id in ("defend", "flee"):
                free_allowed = True
                ap_cost = 0
            if not free_allowed:
                return {
                    "success": False,
                    "action_id": action_id,
                    "ap_cost": 0,
                    "narration": f"You're too exhausted to {action_meta.get('label', action_id).lower()}. (Not enough AP)",
                    "effects": {},
                    "target": None,
                    "perception": None,
                }

        self.player.modify_stamina(-ap_cost)

        narr_ctx: dict[str, Any] = {
            "target": target_npc or target_item or target_location or "them",
        }

        target_npc_obj: NPC | None = None
        if target_npc:
            target_npc_obj = self.npc_registry.get(target_npc)
            if target_npc_obj:
                narr_ctx["target"] = target_npc_obj.name

        effects: dict[str, Any] = {}

        match action_id:
            # ── Navigation ────────────────────────────────────────────
            case "move_to":
                return self._resolve_move_to(
                    target_location, ap_cost, narr_ctx, parsed_input
                )

            # ── Exploration ───────────────────────────────────────────
            case "look":
                return self._resolve_look(ap_cost, narr_ctx)

            case "search":
                return self._resolve_search(target_item, ap_cost, narr_ctx)

            case "examine":
                return self._resolve_examine(
                    target_npc_obj, target_item, ap_cost, narr_ctx
                )

            # ── Social ────────────────────────────────────────────────
            case "talk" | "greet" | "ask_info":
                return self._resolve_social_talk(
                    action_id, target_npc_obj, emotion, social,
                    parsed_input, ap_cost, narr_ctx
                )

            case "persuade":
                return self._resolve_persuade(
                    target_npc_obj, social, ap_cost, narr_ctx
                )

            case "deceive":
                return self._resolve_deceive(
                    target_npc_obj, social, ap_cost, narr_ctx
                )

            case "intimidate":
                return self._resolve_intimidate(
                    target_npc_obj, social, ap_cost, narr_ctx
                )

            case "trade":
                return self._resolve_trade(
                    target_npc_obj, ap_cost, narr_ctx
                )

            case "give_item":
                return self._resolve_give_item(
                    target_npc_obj, target_item, ap_cost, narr_ctx
                )

            case "present_item":
                return self._resolve_present_item(
                    target_npc_obj, target_item, ap_cost, narr_ctx
                )

            # ── Combat ────────────────────────────────────────────────
            case "attack":
                return self._resolve_attack(
                    target_npc_obj, ap_cost, narr_ctx
                )

            case "defend":
                return self._resolve_defend(ap_cost, narr_ctx)

            case "flee":
                return self._resolve_flee(ap_cost, narr_ctx)

            # ── Stealth ───────────────────────────────────────────────
            case "sneak":
                return self._resolve_sneak(ap_cost, narr_ctx)

            case "hide":
                return self._resolve_hide(ap_cost, narr_ctx)

            case "steal":
                return self._resolve_steal(
                    target_npc_obj, target_item, ap_cost, narr_ctx
                )

            # ── Utility ───────────────────────────────────────────────
            case "pick_up":
                return self._resolve_pick_up(target_item, ap_cost, narr_ctx)

            case "use_item":
                return self._resolve_use_item(target_item, ap_cost, narr_ctx)

            case "eat":
                return self._resolve_eat(target_item, ap_cost, narr_ctx)

            case "rest":
                return self._resolve_rest(ap_cost, narr_ctx)

            case "wait":
                return self._resolve_wait(ap_cost, narr_ctx)

            case "drop_item":
                return self._resolve_drop_item(target_item, ap_cost, narr_ctx)

            case "status":
                return self._resolve_status(ap_cost, narr_ctx)

            case "equip":
                return self._resolve_equip(target_item, ap_cost, narr_ctx)

            case "work":
                return self._resolve_work(ap_cost, narr_ctx)

            case _:
                return {
                    "success": False,
                    "action_id": action_id,
                    "ap_cost": ap_cost,
                    "narration": "You're unsure what to do.",
                    "effects": {},
                    "target": None,
                    "perception": None,
                }

    # ── Individual Action Resolvers ───────────────────────────────────────

    def _resolve_move_to(
        self,
        target_location: str | None,
        ap_cost: int,
        ctx: dict,
        parsed_input: Mapping[str, Any],
    ) -> dict:
        """Resolve movement action. Validates adjacency."""
        if not target_location:
            # Default to first adjacent location
            adj = self.world.get_adjacent(self.player.location)
            target_location = adj[0] if adj else None

        if not target_location:
            self.player.modify_stamina(ap_cost)  # refund AP on hard fail
            return self._hard_fail("move_to", ap_cost, "There's nowhere to go from here.")

        # Quest-driven "head back" / "fast travel": if the current CP has
        # a quest_transition whose effects.target_location matches the
        # requested destination, treat the move as a narratively-walked
        # multi-step journey. This covers CP 5_2's `move_to_elders_house`
        # and similar transitions where the JSON encodes a multi-hop
        # logical move as a single quest step.
        current_cp_for_move = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
        # Fall back to deviation_origin so fast-travel still works when the
        # player is on a dynamic CP that branched off a fast-travel-enabled
        # CP (e.g. chat with Tessa at CP 4_3 spawns 4_D1, then move_to fields
        # would otherwise be non-adjacent and blocked).
        fast_travel_cps = [current_cp_for_move]
        if self.quest_manager.deviation_origin is not None:
            origin_cp = self.mdp.get_checkpoint(self.quest_manager.deviation_origin)
            if origin_cp is not None and origin_cp not in fast_travel_cps:
                fast_travel_cps.append(origin_cp)
        quest_fast_travel = False
        for cp_ft in fast_travel_cps:
            if cp_ft is None or not cp_ft.completion_conditions:
                continue
            for _key, _tr in cp_ft.completion_conditions.items():
                if not _key.startswith("move_to"):
                    continue
                expected = (_tr.get("effects", {}) or {}).get("target_location")
                if expected == target_location:
                    quest_fast_travel = True
                    break
            if quest_fast_travel:
                break

        # Same-location move_to is a benign no-op success — it lets quest
        # transitions like CP 4_3's `move_to → 5_1` (both at fields) fire
        # without requiring the player to leave and come back.
        if target_location == self.player.location:
            loc_obj = self.world.get_location(target_location)
            loc_name = loc_obj.name if loc_obj else target_location
            ctx["target"] = loc_name
            narration = f"You take in {loc_name} a moment longer, eyes scanning for what to do next."
            return {
                "success": True,
                "action_id": "move_to",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"old_location": target_location, "new_location": target_location},
                "target": target_location,
                "perception": None,
            }

        if not self.world.is_adjacent(self.player.location, target_location):
            if not quest_fast_travel:
                self.player.modify_stamina(ap_cost)
                loc_obj = self.world.get_location(target_location)
                loc_name = loc_obj.name if loc_obj else target_location
                ctx["target"] = loc_name
                narration = get_template_narration("move_to", "blocked", ctx)
                return {
                    "success": False,
                    "action_id": "move_to",
                    "ap_cost": 0,
                    "narration": narration,
                    "effects": {},
                    "target": target_location,
                    "perception": None,
                }

        # Checkpoint-declared movement gate. Falls back to deviation_origin
        # so the player can't erase a gate by chatting (chat → dynamic CP,
        # dynamic CP has no gate, gate forgotten without this fallback).
        cp_for_gate = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
        gate = getattr(cp_for_gate, "movement_gate", None) if cp_for_gate else None
        if gate is None and self.quest_manager.deviation_origin is not None:
            origin_cp = self.mdp.get_checkpoint(self.quest_manager.deviation_origin)
            gate = getattr(origin_cp, "movement_gate", None) if origin_cp else None
            if gate is not None:
                cp_for_gate = origin_cp
        if gate and target_location in gate.get("blocked_targets", []):
            blocked = False
            # `requires_checkpoint_advance` gates lift naturally — the new CP
            # after advancement won't carry this gate, so we just block until
            # one of the CP's quest transitions (sneak/persuade/present_item) fires.
            if gate.get("requires_checkpoint_advance"):
                blocked = True
            else:
                required = gate.get("requires_interaction_with", [])
                if required and not any(uid in self.interacted_npc_uids for uid in required):
                    blocked = True
            if blocked:
                self.player.modify_stamina(ap_cost)
                block_msg = gate.get("block_message") or "Someone blocks your way."
                return {
                    "success": False,
                    "action_id": "move_to",
                    "ap_cost": 0,
                    "narration": block_msg,
                    "effects": {},
                    "target": target_location,
                    "perception": None,
                }

        # `flee` is the only legal way out of combat; refunding here ensures
        # players can still afford their flee action next turn.
        if self.player.in_combat:
            narration = "You can't simply walk away from combat! Try to flee instead."
            self.player.modify_stamina(ap_cost)
            return {
                "success": False,
                "action_id": "move_to",
                "ap_cost": 0,
                "narration": narration,
                "effects": {},
                "target": target_location,
                "perception": None,
            }

        old_location = self.player.location
        self.player.location = target_location
        loc_obj = self.world.get_location(target_location)
        loc_name = loc_obj.name if loc_obj else target_location
        ctx["target"] = loc_name
        narration = get_template_narration("move_to", "success", ctx)

        if loc_obj:
            narration += f" {loc_obj.description}"

        npcs_here = get_npcs_at_location(self.npc_registry, target_location)
        if npcs_here:
            npc_names = [n.name for n in npcs_here]
            if len(npc_names) == 1:
                narration += f" You see {npc_names[0]} here."
            else:
                narration += f" You see {', '.join(npc_names[:-1])} and {npc_names[-1]} here."

        return {
            "success": True,
            "action_id": "move_to",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {"old_location": old_location, "new_location": target_location},
            "target": target_location,
            "perception": None,
        }

    def _resolve_look(self, ap_cost: int, ctx: dict) -> dict:
        """Look around the current location."""
        loc = self.world.get_location(self.player.location)
        discovery_parts: list[str] = []

        if loc:
            discovery_parts.append(loc.description)
            if loc.objects:
                discovery_parts.append(f"Notable features: {', '.join(loc.objects)}.")
            if loc.items_on_ground:
                item_names = [i.get("name", "something") for i in loc.items_on_ground]
                discovery_parts.append(f"On the ground: {', '.join(item_names)}.")

            # List discovered POIs
            discovered_pois = self.world.get_discovered_pois(self.player.location)
            if discovered_pois:
                poi_names = [p.name for p in discovered_pois]
                discovery_parts.append(f"Points of interest: {', '.join(poi_names)}.")

        npcs_here = get_npcs_at_location(self.npc_registry, self.player.location)
        if npcs_here:
            npc_descs = [f"{n.name} ({n.archetype})" for n in npcs_here]
            discovery_parts.append(f"People here: {', '.join(npc_descs)}.")

        ctx["discovery"] = " ".join(discovery_parts) if discovery_parts else "Nothing stands out."
        narration = get_template_narration("look", "success" if discovery_parts else "fail", ctx)

        return {
            "success": True,
            "action_id": "look",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {},
            "target": None,
            "perception": None,
        }

    def _resolve_search(self, target_item: str | None, ap_cost: int, ctx: dict) -> dict:
        """Search the current location, optionally near a specific POI."""
        loc = self.world.get_location(self.player.location)
        if loc:
            loc.search_count += 1

        # Check if target_item is a POI ID
        target_poi = None
        if target_item and loc:
            target_poi = self.world.get_poi(self.player.location, target_item)

        # Base probability
        prob = compute_skill_probability(
            "search",
            search_count=loc.search_count if loc else 0,
        )

        # POI search bonus
        if target_poi and target_poi.discovered and target_poi.searchable:
            prob = min(0.95, prob + target_poi.search_bonus)
            ctx["target"] = target_poi.name

        if random.random() < prob:
            discovery = ""

            # POI-targeted search: check for hidden items at this POI
            if target_poi and target_poi.discovered and target_poi.items_hidden:
                hidden_item_id = target_poi.items_hidden[0]
                # Resolve from quest items
                item_def = self._quest_items.get(hidden_item_id)
                if item_def:
                    item_dict = dict(item_def)
                    self.player.add_item(item_dict)
                    target_poi.items_hidden.remove(hidden_item_id)
                    discovery = (
                        f"Searching near the {target_poi.name}, you discover "
                        f"{item_dict.get('name', hidden_item_id)} hidden in "
                        f"a hollow between the roots!"
                    )
                    ctx["discovery"] = discovery
                    narration = get_template_narration("search", "success", ctx)
                    return {
                        "success": True,
                        "action_id": "search",
                        "ap_cost": ap_cost,
                        "narration": narration,
                        "effects": {
                            "search_count": loc.search_count if loc else 0,
                            "picked_up": hidden_item_id,
                            "poi_searched": target_poi.poi_id,
                        },
                        "target": target_poi.poi_id,
                        "perception": None,
                    }
                else:
                    target_poi.items_hidden.remove(hidden_item_id)

            # POI-targeted search with no hidden items: contextual discovery
            if target_poi and target_poi.discovered:
                discovery = (
                    f"You search carefully near the {target_poi.name}. "
                    f"{target_poi.description}"
                )
            else:
                # Check for items on ground
                discovery = "You find something interesting amid the clutter."
                if loc and loc.items_on_ground:
                    found_item = loc.items_on_ground[0]
                    discovery = f"You discover {found_item.get('name', 'an item')}!"

            ctx["discovery"] = discovery
            narration = get_template_narration("search", "success", ctx)
            return {
                "success": True,
                "action_id": "search",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {
                    "search_count": loc.search_count if loc else 0,
                    "poi_searched": target_poi.poi_id if target_poi else None,
                },
                "target": target_poi.poi_id if target_poi else None,
                "perception": None,
            }
        else:
            if target_poi and target_poi.discovered:
                ctx["target"] = target_poi.name
            narration = get_template_narration("search", "fail", ctx)
            return {
                "success": False,
                "action_id": "search",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"search_count": loc.search_count if loc else 0},
                "target": target_poi.poi_id if target_poi else None,
                "perception": None,
            }

    def _resolve_examine(
        self,
        target_npc: NPC | None,
        target_item: str | None,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Examine a specific NPC, item, or the environment."""
        if target_npc:
            ctx["target"] = target_npc.name
            rep_label = self.player.get_reputation_label(target_npc.npc_uid)
            discovery = (
                f"{target_npc.name} is a {target_npc.archetype}. "
                f"Their demeanor toward you seems {rep_label}. "
                f"HP: {target_npc.current_hp}/{target_npc.max_hp}."
            )
            ctx["discovery"] = discovery
            narration = get_template_narration("examine", "success", ctx)
        elif target_item:
            # Check if target_item is a POI
            poi = self.world.get_poi(self.player.location, target_item)
            if poi and poi.discovered:
                ctx["target"] = poi.name
                examine_text = poi.examine_text or poi.description
                if poi.items_hidden:
                    examine_text += " Something might be hidden here..."
                ctx["discovery"] = examine_text
                narration = get_template_narration("examine", "success", ctx)
            else:
                # Check inventory items
                item = self.player.get_item(target_item)
                if item:
                    ctx["target"] = item["name"]
                    ctx["discovery"] = item.get("description", "Nothing remarkable.")
                    narration = get_template_narration("examine", "success", ctx)
                else:
                    narration = get_template_narration("examine", "blocked", ctx)
        else:
            # Examine environment
            loc = self.world.get_location(self.player.location)
            ctx["target"] = loc.name if loc else "the area"
            ctx["discovery"] = loc.description if loc else "Nothing stands out."
            narration = get_template_narration("examine", "success", ctx)

        return {
            "success": True,
            "action_id": "examine",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {},
            "target": target_npc.npc_uid if target_npc else target_item,
            "perception": None,
        }

    def _resolve_social_talk(
        self,
        action_id: str,
        target_npc: NPC | None,
        emotion: str,
        social: str,
        parsed_input: Mapping[str, Any],
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve talk, greet, or ask_info actions."""
        if not target_npc:
            npcs_here = get_npcs_at_location(self.npc_registry, self.player.location)
            # Prefer the last NPC the player interacted with, if still nearby
            if self.last_interacted_npc_uid and npcs_here:
                last_match = [n for n in npcs_here if n.npc_uid == self.last_interacted_npc_uid]
                target_npc = last_match[0] if last_match else npcs_here[0]
            elif npcs_here:
                target_npc = npcs_here[0]
            else:
                narration = get_template_narration(action_id, "blocked", ctx)
                return {
                    "success": False,
                    "action_id": action_id,
                    "ap_cost": ap_cost,
                    "narration": narration,
                    "effects": {},
                    "target": None,
                    "perception": None,
                }

        ctx["target"] = target_npc.name
        player_rep = self.player.get_reputation(target_npc.npc_uid)

        # Check if NPC is incapacitated
        if target_npc.is_incapacitated():
            narration = f"{target_npc.name} is unconscious and cannot respond."
            return {
                "success": False,
                "action_id": action_id,
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        # Detect "repeat" intent — replay last dialogue instead of generating new
        raw_text = (parsed_input.get("raw_text") or "").lower()
        _repeat_phrases = ("repeat", "say that again", "what did you say", "come again", "pardon", "say again", "one more time", "didn't catch")
        if any(p in raw_text for p in _repeat_phrases) and target_npc.npc_uid in self._last_dialogue:
            prev = self._last_dialogue[target_npc.npc_uid]
            narration = get_template_narration(action_id, "success", ctx)
            self.last_interacted_npc_uid = target_npc.npc_uid
            self.interacted_npc_uids.add(target_npc.npc_uid)
            return {
                "success": True,
                "action_id": action_id,
                "ap_cost": 0,
                "narration": f"{target_npc.name} repeats what they said.",
                "dialogue": prev,
                "dialogue_speaker": target_npc.name,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        # Resolve dialogue
        dialogue_ctx = {
            "player_reputation": player_rep,
            "quest_state": self.player.quest_state,
            "location": self.player.location,
            "turn": self.turn,
            "time_of_day": self.world.time_of_day,
            "quest_situation": self._build_quest_situation(target_npc),
        }
        dialogue_result = resolve_dialogue(
            target_npc,
            action_id,
            parsed_input.get("raw_text"),
            emotion,
            social,
            dialogue_ctx,
            llm_service=self.llm,
        )

        dialogue_text = format_dialogue(target_npc.name, dialogue_result["dialogue"])
        # Raw dialogue text (without speaker prefix) for separate frontend display
        raw_dialogue = dialogue_result["dialogue"]

        # Greet tracking
        if action_id == "greet":
            if not self.player.has_greeted(target_npc.npc_uid):
                self.player.mark_greeted(target_npc.npc_uid)
                self.player.modify_reputation(target_npc.npc_uid, 2)
                dialogue_result["reputation_change"] = dialogue_result.get("reputation_change", 0) + 2

        # Apply reputation change
        rep_change = dialogue_result.get("reputation_change", 0)
        if rep_change:
            mult = self.difficulty.get("reputation_gain_multiplier", 1.0) if rep_change > 0 else self.difficulty.get("reputation_loss_multiplier", 1.0)
            actual_rep = self.player.modify_reputation(
                target_npc.npc_uid, int(rep_change * mult)
            )
            if actual_rep:
                self._metrics["reputation_changes"] += 1

        # Add conversation to NPC history
        target_npc.add_conversation({
            "turn": self.turn,
            "action": action_id,
            "player_text": parsed_input.get("raw_text"),
            "npc_response": dialogue_result["dialogue"],
            "emotion": emotion,
            "social": social,
        })

        # Apply mood change
        mood_change = dialogue_result.get("mood_change", 0)
        if mood_change:
            target_npc.stats["happiness"] = max(
                0, min(10, target_npc.stats["happiness"] + mood_change)
            )

        narration = get_template_narration(action_id, "success", ctx)

        effects: dict[str, Any] = {
            "reputation_change": rep_change,
            "mood_change": mood_change,
            "reveals_info": dialogue_result.get("reveals_info", False),
        }

        # Check if this dialogue triggers POI discovery
        poi_discoveries = self.world.check_dialogue_discoveries(
            target_npc.npc_uid, raw_dialogue
        )
        if poi_discoveries:
            poi_names = [p.name for p in poi_discoveries]
            effects["poi_discovered"] = [p.poi_id for p in poi_discoveries]
            narration += (
                f" (New point of interest discovered: {', '.join(poi_names)})"
            )

        self.last_interacted_npc_uid = target_npc.npc_uid
        self.interacted_npc_uids.add(target_npc.npc_uid)
        self._last_dialogue[target_npc.npc_uid] = raw_dialogue

        return {
            "success": True,
            "action_id": action_id,
            "ap_cost": ap_cost,
            "narration": narration,
            "dialogue": raw_dialogue,
            "dialogue_speaker": target_npc.name,
            "effects": effects,
            "target": target_npc.npc_uid,
            "perception": None,
        }

    def _resolve_persuade(
        self,
        target_npc: NPC | None,
        social: str,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve a persuade attempt."""
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            return self._no_target("persuade", ap_cost, ctx)

        ctx["target"] = target_npc.name
        self.last_interacted_npc_uid = target_npc.npc_uid
        self.interacted_npc_uids.add(target_npc.npc_uid)
        rep = self.player.get_reputation(target_npc.npc_uid)
        social_mod = SOCIAL_MODIFIERS.get(social, 0)
        prob = compute_skill_probability("persuade", reputation=rep, social_modifier=social_mod)

        if random.random() < prob:
            rep_gain = 5
            mult = self.difficulty.get("reputation_gain_multiplier", 1.0)
            self.player.modify_reputation(target_npc.npc_uid, int(rep_gain * mult))
            narration = get_template_narration("persuade", "success", ctx)
            return {
                "success": True,
                "action_id": "persuade",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"reputation": {target_npc.npc_uid: int(rep_gain * mult)}},
                "target": target_npc.npc_uid,
                "perception": None,
            }
        else:
            rep_loss = -2
            mult = self.difficulty.get("reputation_loss_multiplier", 1.0)
            self.player.modify_reputation(target_npc.npc_uid, int(rep_loss * mult))
            narration = get_template_narration("persuade", "fail", ctx)
            return {
                "success": False,
                "action_id": "persuade",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"reputation": {target_npc.npc_uid: int(rep_loss * mult)}},
                "target": target_npc.npc_uid,
                "perception": None,
            }

    def _resolve_deceive(
        self,
        target_npc: NPC | None,
        social: str,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve a deception attempt."""
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            return self._no_target("deceive", ap_cost, ctx)

        ctx["target"] = target_npc.name
        rep = self.player.get_reputation(target_npc.npc_uid)
        social_mod = SOCIAL_MODIFIERS.get(social, 0)
        prob = compute_skill_probability("deceive", reputation=rep, social_modifier=social_mod)

        if random.random() < prob:
            narration = get_template_narration("deceive", "success", ctx)
            return {
                "success": True,
                "action_id": "deceive",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"deception_success": True},
                "target": target_npc.npc_uid,
                "perception": None,
            }
        else:
            rep_loss = -5
            mult = self.difficulty.get("reputation_loss_multiplier", 1.0)
            self.player.modify_reputation(target_npc.npc_uid, int(rep_loss * mult))
            # Witnesses also penalize
            witnesses = detect_witnesses(
                self.player.location, "player",
                _npc_locations_map(self.npc_registry),
            )
            for w_uid in witnesses:
                if w_uid != target_npc.npc_uid:
                    self.player.modify_reputation(w_uid, -2)

            narration = get_template_narration("deceive", "fail", ctx)
            return {
                "success": False,
                "action_id": "deceive",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {
                    "deception_success": False,
                    "reputation": {target_npc.npc_uid: int(rep_loss * mult)},
                },
                "target": target_npc.npc_uid,
                "perception": None,
            }

    def _resolve_intimidate(
        self,
        target_npc: NPC | None,
        social: str,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve an intimidation attempt."""
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            return self._no_target("intimidate", ap_cost, ctx)

        ctx["target"] = target_npc.name

        # Betrayal — threatening the quest-giver while still holding the amulet
        # (instead of returning it) ends the game.
        if target_npc.archetype == "elder" and self.player.has_item("jade_amulet"):
            msg = (
                f"You loom over {target_npc.name}, the jade amulet in your grip, and make "
                f"your demands. {target_npc.name}: 'You would threaten me — with our own "
                f"heirloom in your hand?' Her cry brings the village down on you, and you "
                f"are banished from Thornhaven."
            )
            self._trigger_defeat("elder_betrayal", msg)
            self.quest_manager.trigger_failure()  # quest registers as failed (graph -> S_fail)
            return {
                "success": True,
                "action_id": "intimidate",
                "ap_cost": ap_cost,
                "narration": msg,
                "effects": {"defeat": "elder_betrayal"},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        # Intimidation uses attack vs defense as a proxy
        player_atk = self.player.combat_stats["base_attack"] + self.player.combat_stats["weapon_modifier"]
        npc_def = target_npc.combat_stats["base_defense"]
        prob = min(0.9, max(0.1, 0.5 + (player_atk - npc_def) / 20))

        if random.random() < prob:
            rep_loss = -3
            mult = self.difficulty.get("reputation_loss_multiplier", 1.0)
            self.player.modify_reputation(target_npc.npc_uid, int(rep_loss * mult))
            narration = get_template_narration("intimidate", "success", ctx)
            return {
                "success": True,
                "action_id": "intimidate",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"reputation": {target_npc.npc_uid: int(rep_loss * mult)}},
                "target": target_npc.npc_uid,
                "perception": None,
            }
        else:
            narration = get_template_narration("intimidate", "fail", ctx)
            return {
                "success": False,
                "action_id": "intimidate",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

    def _resolve_trade(
        self,
        target_npc: NPC | None,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve trading with an NPC."""
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            return self._no_target("trade", ap_cost, ctx)

        ctx["target"] = target_npc.name
        rep = self.player.get_reputation(target_npc.npc_uid)

        # Hostile NPCs refuse trade
        if rep <= -50:
            narration = f"{target_npc.name} refuses to do business with you. Their expression is hostile."
            return {
                "success": False,
                "action_id": "trade",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        narration = get_template_narration("trade", "success", ctx)
        # Reputation bonus for successful trade
        self.player.modify_reputation(target_npc.npc_uid, 1)

        return {
            "success": True,
            "action_id": "trade",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {"reputation": {target_npc.npc_uid: 1}},
            "target": target_npc.npc_uid,
            "perception": None,
        }

    def _resolve_give_item(
        self,
        target_npc: NPC | None,
        target_item: str | None,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve giving an item to an NPC."""
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            return self._no_target("give_item", ap_cost, ctx)
        if not target_item:
            narration = get_template_narration("give_item", "blocked", ctx)
            return {
                "success": False,
                "action_id": "give_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        item = self.player.get_item(target_item)
        if not item:
            narration = "You don't have that item."
            return {
                "success": False,
                "action_id": "give_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        # Quest items cannot be given to non-quest NPCs unless relevant
        ctx["target"] = target_npc.name
        ctx["item"] = item["name"]

        # Track NPC interaction (mirrors present_item / social_talk behaviour)
        self.last_interacted_npc_uid = target_npc.npc_uid
        self.interacted_npc_uids.add(target_npc.npc_uid)

        # Remove from player inventory
        self.player.remove_item(target_item)
        # Remember what was just given (surfaced into quest_situation on next talk/greet)
        self._last_item_given = item["name"]
        # Reputation boost
        rep_gain = 3
        mult = self.difficulty.get("reputation_gain_multiplier", 1.0)
        self.player.modify_reputation(target_npc.npc_uid, int(rep_gain * mult))

        # ── Dialogue — NPC reacts to receiving the item ──────────────
        # Build context with a clear flag so the LLM / scripted layer
        # knows exactly what was just handed over.
        player_rep = self.player.get_reputation(target_npc.npc_uid)
        dialogue_ctx = {
            "player_reputation": player_rep,
            "quest_state": self.player.quest_state,
            "location": self.player.location,
            "turn": self.turn,
            "time_of_day": self.world.time_of_day,
            "item_given": item["name"],
            "item_id": target_item,
            "quest_situation": self._build_quest_situation(target_npc),
        }
        dialogue_result = resolve_dialogue(
            target_npc,
            "give_item",
            f"gives {item['name']}",
            "neutral",
            "neutral",
            dialogue_ctx,
            llm_service=self.llm,
        )
        raw_dialogue = dialogue_result["dialogue"]

        # Record in NPC history so subsequent talk/greet knows the
        # amulet was already handed over.
        target_npc.add_conversation({
            "turn": self.turn,
            "action": "give_item",
            "player_text": f"gives {item['name']}",
            "npc_response": raw_dialogue,
            "emotion": "neutral",
            "social": "neutral",
        })
        # Also cache in _last_dialogue so the repeat system works.
        if hasattr(self, "_last_dialogue"):
            self._last_dialogue[target_npc.npc_uid] = raw_dialogue

        narration = get_template_narration("give_item", "success", ctx)
        return {
            "success": True,
            "action_id": "give_item",
            "ap_cost": ap_cost,
            "narration": narration,
            "dialogue": raw_dialogue,
            "dialogue_speaker": target_npc.name,
            "effects": {
                "item_given": target_item,
                "reputation": {target_npc.npc_uid: int(rep_gain * mult)},
            },
            "target": target_npc.npc_uid,
            "perception": None,
        }

    def _resolve_present_item(
        self,
        target_npc: NPC | None,
        target_item: str | None,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve presenting/showing an item to an NPC without giving it away.

        The NPC inspects the item and reacts with dialogue.  The item stays
        in the player's inventory.  Reputation gain is smaller than
        ``give_item`` because nothing is actually surrendered.
        """
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            return self._no_target("present_item", ap_cost, ctx)

        if not target_item:
            narration = get_template_narration("present_item", "blocked", ctx)
            return {
                "success": False,
                "action_id": "present_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid if target_npc else None,
                "perception": None,
            }

        self.last_interacted_npc_uid = target_npc.npc_uid
        self.interacted_npc_uids.add(target_npc.npc_uid)

        item = self.player.get_item(target_item)
        if not item:
            narration = "You don't have that item to show."
            return {
                "success": False,
                "action_id": "present_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        # Check incapacitated
        if target_npc.is_incapacitated():
            narration = f"{target_npc.name} is unconscious and cannot see what you're showing."
            return {
                "success": False,
                "action_id": "present_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        ctx["target"] = target_npc.name
        ctx["item"] = item["name"]

        # ── Dialogue — NPC reacts to seeing the item ────────────────
        player_rep = self.player.get_reputation(target_npc.npc_uid)
        dialogue_ctx = {
            "player_reputation": player_rep,
            "quest_state": self.player.quest_state,
            "location": self.player.location,
            "turn": self.turn,
            "time_of_day": self.world.time_of_day,
            "item_presented": item["name"],
            "item_id": target_item,
        }
        dialogue_result = resolve_dialogue(
            target_npc,
            "present_item",
            f"shows {item['name']}",
            "neutral",
            "neutral",
            dialogue_ctx,
            llm_service=self.llm,
        )
        raw_dialogue = dialogue_result["dialogue"]

        # Add conversation to NPC history
        target_npc.add_conversation({
            "turn": self.turn,
            "action": "present_item",
            "player_text": f"shows {item['name']}",
            "npc_response": raw_dialogue,
            "emotion": "neutral",
            "social": "neutral",
        })

        # Reputation boost
        rep_gain = 2
        rep_change = dialogue_result.get("reputation_change", 0) + rep_gain
        mult = self.difficulty.get("reputation_gain_multiplier", 1.0)
        actual_rep = self.player.modify_reputation(target_npc.npc_uid, int(rep_change * mult))

        # Mood change
        mood_change = dialogue_result.get("mood_change", 0)
        if mood_change:
            target_npc.stats["happiness"] = max(
                0, min(10, target_npc.stats["happiness"] + mood_change)
            )

        narration = get_template_narration("present_item", "success", ctx)
        return {
            "success": True,
            "action_id": "present_item",
            "ap_cost": ap_cost,
            "narration": narration,
            "dialogue": raw_dialogue,
            "dialogue_speaker": target_npc.name,
            "effects": {
                "item_presented": target_item,
                "reputation": {target_npc.npc_uid: int(rep_change * mult)},
                "mood_change": mood_change,
            },
            "target": target_npc.npc_uid,
            "perception": None,
        }

    def _resolve_attack(
        self,
        target_npc: NPC | None,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Resolve a combat attack against an NPC."""
        if not target_npc:
            target_npc = self._auto_select_npc()
        if not target_npc:
            narration = get_template_narration("attack", "blocked", ctx)
            return {
                "success": False,
                "action_id": "attack",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        ctx["target"] = target_npc.name

        # Betrayal — raising a weapon against the quest-giver while still
        # carrying the recovered amulet ends the game.
        if target_npc.archetype == "elder" and self.player.has_item("jade_amulet"):
            msg = (
                f"You raise your weapon against {target_npc.name} with the jade amulet "
                f"still in your possession. Guards and villagers cry out at the betrayal. "
                f"{target_npc.name}: 'After everything... you would rob us of our heart?' "
                f"You are seized and cast out of Thornhaven for good."
            )
            self._trigger_defeat("elder_betrayal", msg)
            self.quest_manager.trigger_failure()  # quest registers as failed (graph -> S_fail)
            return {
                "success": True,
                "action_id": "attack",
                "ap_cost": ap_cost,
                "narration": msg,
                "effects": {"defeat": "elder_betrayal"},
                "target": target_npc.npc_uid,
                "perception": None,
            }

        self.player.in_combat = True
        self.player.combat_target = target_npc.npc_uid
        self._metrics["combat_encounters"] += 1

        # Resolve attack using combat system
        attacker = self.player.get_combat_dict()
        defender = target_npc.get_combat_dict()
        diff_cfg = self.difficulty.to_dict()

        combat_result: CombatResult = resolve_attack(attacker, defender, diff_cfg)

        # Apply damage to NPC
        effects: dict[str, Any] = {"combat": True}
        if combat_result.hit:
            target_npc.modify_hp(-combat_result.damage)
            ctx["damage"] = combat_result.damage
            effects["damage"] = combat_result.damage
            effects["target_hp"] = target_npc.current_hp

            # Check incapacitation
            if target_npc.current_hp <= 0:
                if target_npc.quest_critical:
                    # Quest-critical: floor at 1 HP, permanently hostile
                    target_npc.current_hp = 1
                    self.player.modify_reputation(target_npc.npc_uid, -80)
                    narration = (
                        f"{combat_result.narrative} "
                        f"{target_npc.name} collapses but clings to life. "
                        f"They will never forget this."
                    )
                    effects["quest_critical_downed"] = True
                else:
                    # Non-critical: incapacitate for 20 turns
                    target_npc.incapacitate(self.turn)
                    self._metrics["npcs_incapacitated"] += 1
                    self.player.modify_reputation(target_npc.npc_uid, -80)
                    # Witness penalty
                    witnesses = detect_witnesses(
                        self.player.location, "player",
                        _npc_locations_map(self.npc_registry),
                    )
                    for w_uid in witnesses:
                        from backend.config import INCAPACITATION_WITNESS_PENALTY
                        self.player.modify_reputation(w_uid, INCAPACITATION_WITNESS_PENALTY)
                    narration = (
                        f"{combat_result.narrative} "
                        f"{target_npc.name} falls unconscious."
                    )
                    effects["incapacitated"] = target_npc.npc_uid

                # Exit combat
                self.player.in_combat = False
                self.player.combat_target = None
            else:
                narration = combat_result.narrative
        else:
            narration = combat_result.narrative

        # Reputation loss for attacking (player's standing with the target)
        rep_loss = -10
        mult = self.difficulty.get("reputation_loss_multiplier", 1.0)
        self.player.modify_reputation(target_npc.npc_uid, int(rep_loss * mult))
        effects["reputation"] = {target_npc.npc_uid: int(rep_loss * mult)}

        # The target now regards the player as hostile, which drives it to fight
        # back on its own turns (see resolve_npc_target: relationship <= -50).
        prev_rel = target_npc.npc_relationships.get("player", 0)
        target_npc.npc_relationships["player"] = min(prev_rel - 60, -55)

        # Bystanders who witness the assault lose trust in the player too — both
        # the player's standing with them and their own view of the player.
        witness_uids = detect_witnesses(
            self.player.location, "player", _npc_locations_map(self.npc_registry),
        )
        witness_rep: dict[str, int] = {}
        for w_uid in witness_uids:
            if w_uid == target_npc.npc_uid:
                continue
            self.player.modify_reputation(w_uid, int(-8 * mult))
            witness_rep[w_uid] = int(-8 * mult)
            w_npc = self.npc_registry.get(w_uid)
            if w_npc is not None:
                w_prev = w_npc.npc_relationships.get("player", 0)
                w_npc.npc_relationships["player"] = max(-100, w_prev - 30)
        if witness_rep:
            effects["witness_reputation"] = witness_rep

        # Immediate self-defense: a conscious, un-incapacitated target strikes
        # back the same turn (this is what lets a guard actually kill the player).
        if target_npc.current_hp > 0 and not target_npc.is_incapacitated():
            self.player.in_combat = True
            self.player.combat_target = target_npc.npc_uid
            retaliation = resolve_attack(
                target_npc.get_combat_dict(),
                self.player.get_combat_dict(),
                self.difficulty.to_dict(),
            )
            if retaliation.hit:
                self.player.modify_health(-retaliation.damage)
                effects["retaliation_damage"] = retaliation.damage
                effects["player_hp"] = self.player.health
                narration = f"{narration} {target_npc.name} strikes back! {retaliation.narrative}"
            else:
                narration = f"{narration} {target_npc.name} lashes out in retaliation but misses."

        return {
            "success": combat_result.hit,
            "action_id": "attack",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": effects,
            "target": target_npc.npc_uid,
            "perception": None,
        }

    def _resolve_defend(self, ap_cost: int, ctx: dict) -> dict:
        """Set defending flag for damage reduction."""
        self.player.is_defending = True
        narration = get_template_narration("defend", "success", ctx)

        if not self.player.in_combat:
            narration = get_template_narration("defend", "blocked", ctx)

        return {
            "success": True,
            "action_id": "defend",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {"is_defending": True},
            "target": None,
            "perception": None,
        }

    def _resolve_flee(self, ap_cost: int, ctx: dict) -> dict:
        """Attempt to flee from combat."""
        if not self.player.in_combat or not self.player.combat_target:
            narration = "You aren't in combat. There's nothing to flee from."
            return {
                "success": False,
                "action_id": "flee",
                "ap_cost": 0,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        opponent_npc = self.npc_registry.get(self.player.combat_target)
        if not opponent_npc:
            self.player.in_combat = False
            self.player.combat_target = None
            return {
                "success": True,
                "action_id": "flee",
                "ap_cost": ap_cost,
                "narration": "Your opponent has vanished. You are no longer in combat.",
                "effects": {},
                "target": None,
                "perception": None,
            }

        fleeing = self.player.get_combat_dict()
        opponent = opponent_npc.get_combat_dict()
        diff_cfg = self.difficulty.to_dict()

        flee_result = resolve_flee(fleeing, opponent, diff_cfg)

        if flee_result["success"]:
            # Move player to random adjacent location
            adj = self.world.get_adjacent(self.player.location)
            if adj:
                new_loc = random.choice(adj)
                self.player.location = new_loc
            self.player.in_combat = False
            self.player.combat_target = None
            ctx["target"] = self.world.get_location(self.player.location)
            loc_name = ctx["target"].name if ctx["target"] else self.player.location
            ctx["target"] = loc_name
            narration = get_template_narration("flee", "success", ctx)
            return {
                "success": True,
                "action_id": "flee",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"fled_to": self.player.location},
                "target": None,
                "perception": None,
            }
        else:
            # Failed flee — take free attack
            narration = flee_result["narrative"]
            effects: dict[str, Any] = {"flee_failed": True}
            free_attack: CombatResult | None = flee_result.get("free_attack")
            if free_attack and free_attack.hit:
                self.player.modify_health(-free_attack.damage)
                narration += f" {free_attack.narrative}"
                effects["damage_taken"] = free_attack.damage
                effects["health_after"] = self.player.health

            return {
                "success": False,
                "action_id": "flee",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": effects,
                "target": self.player.combat_target,
                "perception": None,
            }

    def _resolve_sneak(self, ap_cost: int, ctx: dict) -> dict:
        """Attempt to sneak / move stealthily."""
        npcs_at_loc = len(get_npcs_at_location(self.npc_registry, self.player.location))
        time_bonus = self.world.get_time_bonus()
        prob = compute_skill_probability(
            "sneak", time_bonus=time_bonus, npcs_at_location=npcs_at_loc
        )

        if random.random() < prob:
            narration = get_template_narration("sneak", "success", ctx)
            return {
                "success": True,
                "action_id": "sneak",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"hidden": True},
                "target": None,
                "perception": None,
            }
        else:
            narration = get_template_narration("sneak", "fail", ctx)
            return {
                "success": False,
                "action_id": "sneak",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"hidden": False},
                "target": None,
                "perception": None,
            }

    def _resolve_hide(self, ap_cost: int, ctx: dict) -> dict:
        """Attempt to hide from view."""
        npcs_at_loc = len(get_npcs_at_location(self.npc_registry, self.player.location))
        time_bonus = self.world.get_time_bonus()
        prob = compute_skill_probability(
            "hide", time_bonus=time_bonus, npcs_at_location=npcs_at_loc
        )

        if random.random() < prob:
            narration = get_template_narration("hide", "success", ctx)
            return {
                "success": True,
                "action_id": "hide",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"hidden": True},
                "target": None,
                "perception": None,
            }
        else:
            narration = get_template_narration("hide", "fail", ctx)
            return {
                "success": False,
                "action_id": "hide",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"hidden": False},
                "target": None,
                "perception": None,
            }

    def _resolve_steal(
        self,
        target_npc: NPC | None,
        target_item: str | None,
        ap_cost: int,
        ctx: dict,
    ) -> dict:
        """Attempt to steal from an NPC or the environment."""
        if not target_npc:
            target_npc = self._auto_select_npc()

        npcs_at_loc = len(get_npcs_at_location(self.npc_registry, self.player.location))
        time_bonus = self.world.get_time_bonus()
        prob = compute_skill_probability(
            "steal", time_bonus=time_bonus, npcs_at_location=npcs_at_loc
        )

        if not target_npc and not target_item:
            narration = get_template_narration("steal", "blocked", ctx)
            return {
                "success": False,
                "action_id": "steal",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        if target_npc:
            ctx["target"] = target_npc.name

        if random.random() < prob:
            ctx["item"] = target_item or "some coins"
            narration = get_template_narration("steal", "success", ctx)
            return {
                "success": True,
                "action_id": "steal",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"stolen_item": target_item},
                "target": target_npc.npc_uid if target_npc else None,
                "perception": None,
            }
        else:
            # Caught! reputation penalty
            if target_npc:
                rep_loss = -10
                mult = self.difficulty.get("reputation_loss_multiplier", 1.0)
                self.player.modify_reputation(target_npc.npc_uid, int(rep_loss * mult))
                # Witnesses also penalize
                witnesses = detect_witnesses(
                    self.player.location, "player",
                    _npc_locations_map(self.npc_registry),
                )
                for w_uid in witnesses:
                    if w_uid != target_npc.npc_uid:
                        self.player.modify_reputation(w_uid, -5)

            narration = get_template_narration("steal", "fail", ctx)
            return {
                "success": False,
                "action_id": "steal",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"caught": True},
                "target": target_npc.npc_uid if target_npc else None,
                "perception": None,
            }

    def _resolve_pick_up(
        self, target_item: str | None, ap_cost: int, ctx: dict
    ) -> dict:
        """Pick an item off the ground, or accept a gift the current CP grants via `pick_up`.

        The gift path covers CP 7_2 — Elder Maren hands over the iron shield
        with no POI to physically pick up; the player still uses `pick_up`
        to accept it, and the quest manager's `gives` reward attaches the item.
        """
        loc = self.world.get_location(self.player.location)
        if not loc or not loc.items_on_ground:
            cp = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
            transitions = (cp.completion_conditions or {}) if cp else {}
            pickup_tr = transitions.get("pick_up", {}) if isinstance(transitions, dict) else {}
            gives = (pickup_tr.get("effects", {}) or {}).get("gives", []) or []
            wants = target_item or ""
            if pickup_tr and (not wants or wants in gives):
                gift = wants or (gives[0] if gives else "item")
                already_owns = any(
                    (it.get("id") == gift or it.get("name") == gift)
                    for it in self.player.inventory
                )
                if already_owns:
                    narration = get_template_narration("pick_up", "blocked", ctx)
                    return {
                        "success": False,
                        "action_id": "pick_up",
                        "ap_cost": ap_cost,
                        "narration": narration,
                        "effects": {},
                        "target": None,
                        "perception": None,
                    }
                ctx["item"] = gift
                narration = get_template_narration("pick_up", "success", ctx)
                return {
                    "success": True,
                    "action_id": "pick_up",
                    "ap_cost": ap_cost,
                    "narration": narration,
                    "effects": {"accepted_gift": gift},
                    "target": gift,
                    "perception": None,
                }
            narration = get_template_narration("pick_up", "blocked", ctx)
            return {
                "success": False,
                "action_id": "pick_up",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        # Find the item
        found_item = None
        for i, item in enumerate(loc.items_on_ground):
            if target_item and item["id"] == target_item:
                found_item = loc.items_on_ground.pop(i)
                break
        if not found_item and loc.items_on_ground:
            found_item = loc.items_on_ground.pop(0)

        if not found_item:
            narration = get_template_narration("pick_up", "blocked", ctx)
            return {
                "success": False,
                "action_id": "pick_up",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        # Check inventory capacity
        if self.player.inventory_full():
            if found_item.get("quest_relevant"):
                # Auto-prompt: find droppable item
                droppable = self.player.get_droppable_items()
                if droppable:
                    dropped = self.player.remove_item(droppable[0]["id"])
                    if dropped and loc:
                        loc.items_on_ground.append(dropped)
                    self.player.add_item(found_item)
                    ctx["item"] = found_item["name"]
                    narration = (
                        f"Your inventory is full, but this item is important. "
                        f"You drop {dropped['name'] if dropped else 'something'} "
                        f"and pick up {found_item['name']}."
                    )
                    return {
                        "success": True,
                        "action_id": "pick_up",
                        "ap_cost": ap_cost,
                        "narration": narration,
                        "effects": {
                            "picked_up": found_item["id"],
                            "auto_dropped": dropped["id"] if dropped else None,
                        },
                        "target": None,
                        "perception": None,
                    }
                else:
                    # All items are quest items — still pick it up (emergency)
                    self.player.add_item(found_item)
                    ctx["item"] = found_item["name"]
                    narration = f"You squeeze {found_item['name']} into your already full pack."
                    return {
                        "success": True,
                        "action_id": "pick_up",
                        "ap_cost": ap_cost,
                        "narration": narration,
                        "effects": {"picked_up": found_item["id"]},
                        "target": None,
                        "perception": None,
                    }
            else:
                # Not quest-relevant — put it back
                loc.items_on_ground.insert(0, found_item)
                narration = "Your inventory is full! You need to drop something first."
                return {
                    "success": False,
                    "action_id": "pick_up",
                    "ap_cost": ap_cost,
                    "narration": narration,
                    "effects": {"inventory_full": True},
                    "target": None,
                    "perception": None,
                }

        self.player.add_item(found_item)
        ctx["item"] = found_item["name"]
        narration = get_template_narration("pick_up", "success", ctx)
        return {
            "success": True,
            "action_id": "pick_up",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {"picked_up": found_item["id"]},
            "target": None,
            "perception": None,
        }

    def _resolve_use_item(
        self, target_item: str | None, ap_cost: int, ctx: dict
    ) -> dict:
        """Use an item from inventory."""
        if not target_item:
            narration = get_template_narration("use_item", "blocked", ctx)
            return {
                "success": False,
                "action_id": "use_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        item = self.player.get_item(target_item)
        if not item:
            narration = "You don't have that item."
            return {
                "success": False,
                "action_id": "use_item",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        ctx["item"] = item["name"]
        effects_dict: dict[str, Any] = {}

        # Apply item effects
        item_effects = item.get("effects", {})
        effect_desc_parts: list[str] = []

        if "heal" in item_effects:
            heal_amount = item_effects["heal"]
            actual = self.player.modify_health(heal_amount)
            effects_dict["healed"] = actual
            effect_desc_parts.append(f"+{actual} HP")

        if "stamina" in item_effects:
            stam_amount = item_effects["stamina"]
            actual = self.player.modify_stamina(stam_amount)
            effects_dict["stamina_restored"] = actual
            effect_desc_parts.append(f"+{actual} AP")

        ctx["effect"] = ", ".join(effect_desc_parts) if effect_desc_parts else "Nothing happens."

        # Consume if consumable
        if item.get("type") == "consumable":
            self.player.remove_item(target_item)
            effects_dict["consumed"] = True

        narration = get_template_narration("use_item", "success", ctx)
        return {
            "success": True,
            "action_id": "use_item",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": effects_dict,
            "target": None,
            "perception": None,
        }

    def _resolve_eat(
        self, target_item: str | None, ap_cost: int, ctx: dict
    ) -> dict:
        """Eat a food item to restore HP."""
        # Find food in inventory
        food_items = self.player.get_food_items()

        if target_item:
            food = self.player.get_item(target_item)
            if food and food.get("type") == "consumable" and "heal" in food.get("effects", {}):
                food_items = [food]
            else:
                food_items = []

        if not food_items:
            narration = get_template_narration("eat", "blocked", ctx)
            return {
                "success": False,
                "action_id": "eat",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        food = food_items[0]
        heal_amount = food.get("effects", {}).get("heal", 5)
        actual_heal = self.player.modify_health(heal_amount)
        self.player.remove_item(food["id"])

        ctx["item"] = food["name"]
        ctx["heal"] = actual_heal
        narration = get_template_narration("eat", "success", ctx)

        return {
            "success": True,
            "action_id": "eat",
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {"healed": actual_heal, "consumed": food["id"]},
            "target": None,
            "perception": None,
        }

    def _resolve_rest(self, ap_cost: int, ctx: dict) -> dict:
        """Rest to recover AP. Requires indoor/safe or no combat."""
        if self.player.in_combat:
            # In combat, rest becomes wait
            return self._resolve_wait(0, ctx)

        is_indoor = self.world.is_indoor(self.player.location)
        if not is_indoor:
            # Outdoor: rest resolves as wait with a message
            ctx_copy = dict(ctx)
            narration = get_template_narration("rest", "fail", ctx_copy)
            # Still give partial stamina bonus from wait
            return {
                "success": False,
                "action_id": "rest",
                "ap_cost": 0,
                "narration": narration,
                "effects": {"rested_as_wait": True},
                "target": None,
                "perception": None,
            }

        # Successful rest: +10 AP
        actual = self.player.modify_stamina(10)
        narration = get_template_narration("rest", "success", ctx)
        return {
            "success": True,
            "action_id": "rest",
            "ap_cost": 0,
            "narration": narration,
            "effects": {"stamina_restored": actual},
            "target": None,
            "perception": None,
        }

    def _resolve_wait(self, ap_cost: int, ctx: dict) -> dict:
        """Wait and observe."""
        narration = get_template_narration("wait", "success", ctx)

        # Passive perception from wait
        perception = None
        loc = self.world.get_location(self.player.location)
        if loc:
            npcs_here = get_npcs_at_location(self.npc_registry, self.player.location)
            perception = passive_perception_check(
                self.player.location,
                self.world.is_social(self.player.location),
                loc.items_on_ground,
                [
                    {"npc_uid": n.npc_uid, "name": n.name, "action_importance": 1}
                    for n in npcs_here
                ],
            )

        return {
            "success": True,
            "action_id": "wait",
            "ap_cost": 0,
            "narration": narration,
            "effects": {},
            "target": None,
            "perception": perception,
        }

    def _resolve_drop_item(
        self, target_item: str | None, ap_cost: int, ctx: dict
    ) -> dict:
        """Drop an item from inventory."""
        if not target_item:
            droppable = self.player.get_droppable_items()
            if not droppable:
                narration = get_template_narration("drop_item", "blocked", ctx)
                return {
                    "success": False,
                    "action_id": "drop_item",
                    "ap_cost": 0,
                    "narration": narration,
                    "effects": {},
                    "target": None,
                    "perception": None,
                }
            target_item = droppable[0]["id"]

        item = self.player.get_item(target_item)
        if not item:
            narration = "You don't have that item."
            return {
                "success": False,
                "action_id": "drop_item",
                "ap_cost": 0,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        # Quest items can't be dropped
        if item.get("quest_relevant"):
            narration = get_template_narration("drop_item", "blocked", ctx)
            return {
                "success": False,
                "action_id": "drop_item",
                "ap_cost": 0,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

        removed = self.player.remove_item(target_item)
        if removed:
            loc = self.world.get_location(self.player.location)
            if loc:
                loc.items_on_ground.append(removed)
            ctx["item"] = removed["name"]
            narration = get_template_narration("drop_item", "success", ctx)
            return {
                "success": True,
                "action_id": "drop_item",
                "ap_cost": 0,
                "narration": narration,
                "effects": {"dropped": removed["id"]},
                "target": None,
                "perception": None,
            }

        narration = "Something went wrong trying to drop that item."
        return {
            "success": False,
            "action_id": "drop_item",
            "ap_cost": 0,
            "narration": narration,
            "effects": {},
            "target": None,
            "perception": None,
        }

    def _resolve_status(self, ap_cost: int, ctx: dict) -> dict:
        """Check quest journal / status."""
        progress = self.quest_manager.get_quest_progress()
        cp = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
        quest_info = (
            f"Quest: {progress['title']} | "
            f"Stage {progress['current_stage']} | "
            f"Checkpoint: {progress['current_checkpoint']} | "
            f"Progress: {progress['completion_percent']}%"
        )
        if cp:
            quest_info += f" — {cp.description}"

        ctx["quest_info"] = quest_info
        narration = get_template_narration("status", "success", ctx)
        return {
            "success": True,
            "action_id": "status",
            "ap_cost": 0,
            "narration": narration,
            "effects": {"quest_progress": progress},
            "target": None,
            "perception": None,
        }

    def _resolve_equip(
        self, target_item: str | None, ap_cost: int, ctx: dict
    ) -> dict:
        """Equip a weapon or armor."""
        if not target_item:
            equipment = self.player.get_equipment_items()
            if not equipment:
                narration = get_template_narration("equip", "blocked", ctx)
                return {
                    "success": False,
                    "action_id": "equip",
                    "ap_cost": ap_cost,
                    "narration": narration,
                    "effects": {},
                    "target": None,
                    "perception": None,
                }
            target_item = equipment[0]["id"]

        prev = self.player.equip_item(target_item)
        item = self.player.get_item(target_item)

        if item:
            ctx["item"] = item["name"]
            narration = get_template_narration("equip", "success", ctx)
            return {
                "success": True,
                "action_id": "equip",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {
                    "equipped": target_item,
                    "previous": prev["id"] if prev else None,
                },
                "target": None,
                "perception": None,
            }
        else:
            narration = get_template_narration("equip", "blocked", ctx)
            return {
                "success": False,
                "action_id": "equip",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

    def _resolve_work(self, ap_cost: int, ctx: dict) -> dict:
        """Perform labor at appropriate locations."""
        # Work is most effective at fields, tavern, or village center
        location = self.player.location
        work_locations = {"fields", "tavern", "village_center"}

        if location in work_locations:
            # Reputation boost with NPCs at location
            npcs_here = get_npcs_at_location(self.npc_registry, location)
            rep_effects: dict[str, int] = {}
            for npc in npcs_here[:3]:
                gain = 2
                mult = self.difficulty.get("reputation_gain_multiplier", 1.0)
                actual = self.player.modify_reputation(npc.npc_uid, int(gain * mult))
                if actual:
                    rep_effects[npc.npc_uid] = actual

            narration = get_template_narration("work", "success", ctx)
            return {
                "success": True,
                "action_id": "work",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {"reputation": rep_effects},
                "target": None,
                "perception": None,
            }
        else:
            narration = get_template_narration("work", "fail", ctx)
            return {
                "success": False,
                "action_id": "work",
                "ap_cost": ap_cost,
                "narration": narration,
                "effects": {},
                "target": None,
                "perception": None,
            }

    # ── Action Resolution Helpers ─────────────────────────────────────────

    def _hard_fail(self, action_id: str, ap_cost: int, reason: str) -> dict:
        """Build a hard-fail result (0 AP cost, narrates why)."""
        return {
            "success": False,
            "action_id": action_id,
            "ap_cost": 0,
            "narration": reason,
            "effects": {},
            "target": None,
            "perception": None,
        }

    def _build_quest_situation(self, target_npc: NPC | None) -> str:
        """Summarize what the NPC currently expects from the player.

        Returns a short string describing the active checkpoint, what
        transitions advance it, and which items / outcomes the NPC is
        looking for. Surfaced into the LLM dialogue prompt so the NPC
        does not forget unresolved demands across turns.
        """
        cp = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
        # If we're on a dynamic offshoot, surface the original gated CP so
        # the NPC's outstanding demands stay in scope.
        origin = self.quest_manager.deviation_origin
        if origin is not None:
            origin_cp = self.mdp.get_checkpoint(origin)
            if origin_cp is not None and getattr(origin_cp, "movement_gate", None):
                cp = origin_cp
        if cp is None:
            return ""

        parts: list[str] = []
        parts.append(f"Active checkpoint: {cp.checkpoint_id} (stage {cp.stage_id}).")
        if cp.description:
            parts.append(f"Scene: {cp.description}")

        # Pending demands derived from quest_transitions
        demands: list[str] = []
        for action_key, trans in (cp.completion_conditions or {}).items():
            requires = trans.get("requires", {}) if isinstance(trans, dict) else {}
            required_item = requires.get("item")
            if required_item:
                # Check if player actually has it
                has_it = self.player.has_item(required_item) if hasattr(self.player, "has_item") else False
                state = "(player has it)" if has_it else "(player does NOT have it)"
                demands.append(f"{action_key} requires item '{required_item}' {state}")
            elif action_key in ("persuade", "sneak", "present_item", "give_item"):
                demands.append(f"{action_key} is an accepted way past")
        if demands:
            parts.append("Quest demands at this checkpoint:")
            for d in demands:
                parts.append(f"  - {d}")

        # Gate-specific reminder
        gate = getattr(cp, "movement_gate", None)
        if gate and gate.get("blocked_targets"):
            tgts = ", ".join(gate["blocked_targets"])
            parts.append(
                f"You ({target_npc.name if target_npc else 'NPC'}) are actively "
                f"blocking the player from leaving to: {tgts}. Idle chatter does "
                f"not earn passage — you only stand aside if the player meets "
                f"the quest demands listed above."
            )

        # Surface any item the player just handed over this turn so that a
        # subsequent talk/greet at the next checkpoint still has that context.
        last_given = getattr(self, "_last_item_given", None)
        if last_given:
            parts.append(
                f"The player JUST handed over '{last_given}' — acknowledge this "
                f"warmly and move the conversation forward to the reward."
            )

        return "\n".join(parts)

    def _no_target(self, action_id: str, ap_cost: int, ctx: dict) -> dict:
        """Build a 'no target present' blocked result."""
        narration = get_template_narration(action_id, "blocked", ctx)
        return {
            "success": False,
            "action_id": action_id,
            "ap_cost": ap_cost,
            "narration": narration,
            "effects": {},
            "target": None,
            "perception": None,
        }

    def _auto_select_npc(self) -> NPC | None:
        """Auto-select an NPC at the player's location, preferring the last interacted one."""
        npcs = get_npcs_at_location(self.npc_registry, self.player.location)
        if not npcs:
            return None
        if self.last_interacted_npc_uid:
            last_match = [n for n in npcs if n.npc_uid == self.last_interacted_npc_uid]
            if last_match:
                return last_match[0]
        return npcs[0]

    def _get_active_weather(self) -> str | None:
        """Get active weather event description, if any."""
        for ev in self.world.active_events:
            if ev.get("id", "").startswith("weather_"):
                return ev.get("description")
        return None

    # ── Quest Progression ─────────────────────────────────────────────────

    def _check_quest_progress(
        self,
        action_id: str,
        parsed_input: Mapping[str, Any],
        action_result: dict,
    ) -> dict | None:
        """Drive quest progression for this turn: completion → convergence →
        forward-completion → deviation. Returns a quest_update dict or None."""
        target_location = parsed_input.get("target_location")
        target_npc = parsed_input.get("target_npc")

        context = {
            "target_location": target_location or self.player.location,
            "target_npc": target_npc,
            "location": self.player.location,
            # Pre-action inventory snapshot so `requires.item` checks see the
            # state the player chose against — give_item/present_item consume
            # the item before we get here.
            "player_inventory": getattr(self, "_pre_action_inventory", self.player.inventory),
            # Lets quest_manager gate probability rolls (sneak, persuade)
            # and movement on actual success.
            "action_success": action_result.get("success"),
        }

        completion = self.quest_manager.check_completion(action_id, target_npc, context)

        # On a dynamic offshoot the action might still satisfy the original
        # static CP — that's convergence back to the main path.
        if completion is None and self.quest_manager.deviation_origin is not None:
            completion = self.quest_manager.check_convergence(
                action_id, target_npc, context
            )
            if completion is not None:
                self.event_log.add_entry(
                    turn=self.turn, time_of_day=self.world.time_of_day,
                    event_type="quest_convergence", actor="player", action=action_id,
                    target=None, location=self.player.location, outcome="natural",
                    effects={"forced": False},
                    witnesses=[], narration="", importance=3,
                )

        # Forward-completion lets a quest-critical action (e.g. handing the
        # amulet over directly) leap past intermediate travel checkpoints.
        if completion is None:
            completion = self.quest_manager.check_forward_completion(
                action_id, target_npc, context
            )

        if completion:
            next_cp = completion.get("next_checkpoint")
            rewards = completion.get("rewards", {})

            if "reputation" in rewards and isinstance(rewards["reputation"], dict):
                for npc_uid, delta in rewards["reputation"].items():
                    self.player.modify_reputation(npc_uid, delta)
            if "gives" in rewards:
                for item in rewards["gives"] if isinstance(rewards["gives"], list) else [rewards["gives"]]:
                    if isinstance(item, dict):
                        item_id = item.get("id", "")
                        if not self.player.has_item(item_id):
                            self.player.add_item(item)
                        else:
                            logger.debug(
                                "Quest gives reward: player already has '%s', skipping duplicate", item_id
                            )
                    elif isinstance(item, str):
                        # JSON encodes gives as item IDs — resolve to the full dict.
                        if self.player.has_item(item):
                            logger.debug(
                                "Quest gives reward: player already has '%s', skipping duplicate", item
                            )
                            continue
                        item_def = self._quest_items.get(item)
                        if item_def:
                            self.player.add_item(dict(item_def))
                            logger.info("Quest reward: gave item '%s'", item)
                        else:
                            logger.warning("Quest reward item '%s' not found in definitions", item)
            if "removes" in rewards:
                for item_id in rewards["removes"] if isinstance(rewards["removes"], list) else [rewards["removes"]]:
                    removed = self.player.remove_item(item_id)
                    if removed:
                        logger.info("Quest reward: removed item '%s'", item_id)
            if "stamina" in rewards:
                self.player.modify_stamina(rewards["stamina"])

            if next_cp:
                # Forward-scan matches skip CPs; mark the matched one done so
                # the graph reflects what actually completed.
                matched_cp = completion["checkpoint_completed"]
                if (
                    matched_cp != self.quest_manager.current_checkpoint
                    and matched_cp not in self.quest_manager.completed_checkpoints
                ):
                    self.quest_manager.completed_checkpoints.append(matched_cp)

                self.quest_manager.advance_checkpoint(next_cp)
                # Player.quest_state mirrors the manager — kept in sync so
                # save/load and the API state response don't drift.
                self.player.quest_state["current_stage"] = self.quest_manager.current_stage
                self.player.quest_state["current_checkpoint"] = self.quest_manager.current_checkpoint
                self.player.quest_state["completed_checkpoints"] = list(
                    self.quest_manager.completed_checkpoints
                )
                self.player.quest_state["deviation_count"] = self.quest_manager.deviation_count
                poi_discoveries = self.world.check_quest_stage_discoveries(
                    self.quest_manager.current_stage
                )
                if poi_discoveries:
                    poi_names = [p.name for p in poi_discoveries]
                    logger.info(
                        "Quest stage %d revealed POIs: %s",
                        self.quest_manager.current_stage,
                        poi_names,
                    )

            # Stage transitions are the rare high-importance events; intra-stage
            # progress is informative but not headline-worthy.
            importance = 5 if completion.get("stage_transition") else 3
            self.event_log.add_entry(
                turn=self.turn,
                time_of_day=self.world.time_of_day,
                event_type="quest_progress",
                actor="player",
                action=action_id,
                target=next_cp,
                location=self.player.location,
                outcome="checkpoint_completed",
                effects={
                    "checkpoint_completed": completion["checkpoint_completed"],
                    "next_checkpoint": next_cp,
                    "stage_transition": completion.get("stage_transition", False),
                },
                witnesses=[],
                narration=f"Quest progress: completed {completion['checkpoint_completed']}.",
                importance=importance,
            )

            self.save_game("auto")

            return {
                "type": "checkpoint_completed",
                "completion": completion,
                "quest_progress": self.quest_manager.get_quest_progress(),
            }

        # Deviation handling. Combat counts even when it misses — attacking is
        # an intentional choice, not a mechanical failure. Pure observation
        # actions (look/examine/rest/wait/status) never deviate.
        is_deliberate = action_id in ("attack",)
        action_ok = action_result.get("success", True) or is_deliberate
        exempt = action_id in ("look", "wait", "status", "rest", "examine")
        if action_ok and not exempt:
            deviation = self.quest_manager.handle_deviation(action_id, context)
            self.event_log.add_entry(
                turn=self.turn, time_of_day=self.world.time_of_day,
                event_type="quest_deviation", actor="player", action=action_id,
                target=None, location=self.player.location, outcome="deviation",
                effects={"deviation_count": deviation["deviation_count"]},
                witnesses=[], narration="", importance=2,
            )

            if deviation["force_convergence"]:
                # Force convergence: loop detection check
                looped = self.quest_manager.check_loop_detection(
                    action_id, self.quest_manager.current_stage
                )
                hint = get_nudge_hint(
                    deviation["deviation_count"],
                    self.quest_manager.current_checkpoint,
                    self.mdp,
                )

                if looped or deviation["force_convergence"]:
                    # Force back to main flow
                    convergence_cp = hint.get("target_cp")
                    if convergence_cp and convergence_cp != self.quest_manager.current_checkpoint:
                        self.quest_manager.advance_checkpoint(convergence_cp)
                        self.player.quest_state["current_checkpoint"] = self.quest_manager.current_checkpoint

                self.event_log.add_entry(
                    turn=self.turn, time_of_day=self.world.time_of_day,
                    event_type="quest_convergence", actor="player", action=action_id,
                    target=None, location=self.player.location, outcome="forced",
                    effects={"forced": True, "deviation_count": deviation["deviation_count"]},
                    witnesses=[], narration="", importance=3,
                )
                return {
                    "type": "forced_convergence",
                    "hint": hint,
                    "deviation_count": deviation["deviation_count"],
                }

            if deviation["needs_dynamic_cp"]:
                # Generate dynamic checkpoint
                cp_id = self.quest_manager.generate_dynamic_cp_id(
                    self.quest_manager.current_stage
                )
                npc_name = None
                if parsed_input.get("target_npc"):
                    npc_obj = self.npc_registry.get(parsed_input["target_npc"])
                    npc_name = npc_obj.name if npc_obj else None
                if not npc_name:
                    nearby = get_npcs_at_location(self.npc_registry, self.player.location)
                    if nearby:
                        npc_name = nearby[0].name

                # Resolve nudge_target: use the *origin* checkpoint's
                # nudge_target so dynamic CPs always point toward the
                # next main-path checkpoint, not back to themselves.
                origin_cp_id = (
                    self.quest_manager.deviation_origin
                    or self.quest_manager.current_checkpoint
                )
                origin_cp = self.mdp.get_checkpoint(origin_cp_id)
                proper_nudge_target = (
                    origin_cp.nudge_target if origin_cp else None
                )
                if proper_nudge_target is None:
                    # Fallback: first static CP in current/next stage
                    from backend.quest.nudge import get_convergence_checkpoint
                    proper_nudge_target = get_convergence_checkpoint(
                        self.quest_manager.current_stage, self.mdp
                    )

                cp_context = {
                    "checkpoint_id": cp_id,
                    "location": self.player.location,
                    "npc_name": npc_name or "a nearby villager",
                    "nudge_target": proper_nudge_target,
                }

                dynamic_cp = generate_dynamic_checkpoint(
                    self.quest_manager.current_stage,
                    action_id,
                    cp_context,
                    self.llm if self.llm.available else None,
                )
                self.quest_manager.add_dynamic_checkpoint(dynamic_cp)
                self._metrics["dynamic_checkpoints_created"] += 1

                # Move player to dynamic checkpoint
                self.quest_manager.current_checkpoint = cp_id
                self.player.quest_state["current_checkpoint"] = cp_id
                self.player.quest_state["dynamic_checkpoints"] = list(
                    self.quest_manager.dynamic_checkpoints
                )
                self.player.quest_state["deviation_count"] = self.quest_manager.deviation_count

                # Nudge hint
                hint = get_nudge_hint(
                    deviation["deviation_count"],
                    cp_id,
                    self.mdp,
                )

                # Nudge reward shaping
                nudge_reward = compute_nudge_reward(cp_id, action_id, self.mdp)

                return {
                    "type": "dynamic_checkpoint",
                    "checkpoint": {
                        "id": cp_id,
                        "description": dynamic_cp.description,
                        "highlighted_actions": dynamic_cp.highlighted_actions,
                    },
                    "hint": hint,
                    "nudge_reward": nudge_reward,
                    "deviation_count": deviation["deviation_count"],
                }

        return None

    # ── NPC Turn Processing ───────────────────────────────────────────────

    def compute_community_state(self) -> dict:
        """Aggregate village state into single dict for community reward calculation.

        Returns dict with:
        - avg_reputation: mean reputation across all NPCs (from player perspective)
        - total_health: sum of NPC health
        - avg_mood: mean happiness stat across NPCs
        """
        if not self.npc_registry:
            return {
                "avg_reputation": 0.0,
                "total_health": 0.0,
                "avg_mood": 0.0,
            }

        npcs = list(self.npc_registry.values())

        # Reputation: average of all per-NPC reputation values
        all_reps = [self.player.get_reputation(uid) for uid in self.npc_registry.keys()]
        avg_rep = float(np.mean(all_reps)) if all_reps else 0.0

        # Health: total across all NPCs
        total_hp = float(sum(npc.current_hp for npc in npcs))

        # Mood: average happiness stat
        moods = [npc.stats.get("happiness", 5) for npc in npcs]
        avg_mood = float(np.mean(moods)) if moods else 0.0

        return {
            "avg_reputation": avg_rep,
            "total_health": total_hp,
            "avg_mood": avg_mood,
        }

    def _process_npc_turns(self) -> list[dict]:
        """Process all NPC actions for this turn.

        For each active NPC:
        1. Check recovery from incapacitation
        2. Discretize state
        3. Select action (Q-learning if past cold start, else schedule)
        4. Resolve action
        5. Compute reward and update Q-table
        6. Indoor HP regen
        7. Decay epsilon

        Returns:
            List of NPC action result dicts for narration filtering.
        """
        results: list[dict] = []
        gossip_context = {"gossip_pairs": self._gossip_pairs_this_turn}

        for uid, npc in self.npc_registry.items():
            # 1. Check recovery
            if npc.is_incapacitated():
                recovered = npc.check_recovery(self.turn)
                if recovered:
                    results.append({
                        "npc_uid": uid,
                        "action": "recover",
                        "narration": f"{npc.name} regains consciousness, looking dazed and hostile.",
                        "location": npc.location,
                        "importance": 3,
                    })
                continue

            # 1b. Apply natural stat decay (Solution B: entropy prevents saturation)
            from backend.config import NPC_STAT_DECAY
            for stat_key, decay_rate in NPC_STAT_DECAY.items():
                if stat_key in npc.stats:
                    npc.stats[stat_key] = max(0.0, npc.stats[stat_key] - decay_rate)

            # 2. Discretize state
            state = npc.discretize_state(self.world.time_of_day)

            # Clear defending flag at start of NPC turn (unless this turn is defend)
            npc.is_defending = False

            # 3. Select action
            npcs_at_loc = [
                n for n_uid, n in self.npc_registry.items()
                if n.location == npc.location
                and n_uid != uid
                and not n.is_incapacitated()
            ]
            player_here = self.player.location == npc.location

            game_ctx: dict[str, Any] = {
                "npcs_at_location": npcs_at_loc,
                "player_location": self.player.location,
                "items_at_location": [],
            }
            loc = self.world.get_location(npc.location)
            if loc:
                game_ctx["items_at_location"] = loc.items_on_ground

            if not _cfg.RL_ENABLED or self.turn <= NPC_COLD_START_TURNS:
                scheduled = get_scheduled_action(npc, self.world.time_of_day)
                action_id = scheduled["action"]
                action_idx = UNIVERSAL_ACTION_IDS.index(action_id) if action_id in UNIVERSAL_ACTION_IDS else UNIVERSAL_ACTION_IDS.index("wait")
            else:
                valid_actions = get_valid_actions(npc, game_ctx)
                action_idx = select_action(npc, state, valid_actions)
                action_id = UNIVERSAL_ACTION_IDS[action_idx]
            
            # STEP 4: Track role telemetry for this action
            npc.update_role_telemetry(action_id)

            # Snapshot stats before action
            old_stats = dict(npc.stats)

            # 4. Resolve action
            npc_result = self._resolve_npc_action(npc, action_id, npcs_at_loc, player_here, gossip_context)

            # 5. Compute reward and update Q-table
            next_state = npc.discretize_state(self.world.time_of_day)
            # P1: Compute community state with delta from previous turn
            community_state = self.compute_community_state()
            reward_dict = compute_reward(npc, old_stats, npc.stats, community_state, self._prev_community_state)

            # STEP 5: Apply shock reward modifier to community component
            if SHOCK_ENABLED and self.shock_manager.has_active_shocks:
                shock_mod = self.shock_manager.get_reward_modifier()
                reward_dict["community"] *= shock_mod
                reward_dict["total"] = (
                    reward_dict["penalty"] + reward_dict["individual"]
                    + npc.lambda_coeff * reward_dict["community"]
                )

            npc.add_reward_sample(self.turn, reward_dict)
            if _cfg.RL_ENABLED:
                update_q_table(npc, state, action_idx, reward_dict["total"], next_state)
                shock_pressure = self.shock_manager.get_adaptation_pressure() if SHOCK_ENABLED else 0.0
                npc.update_adaptation(reward_dict, shock_pressure)
            npc.add_adaptation_sample(self.turn)

            if self.world.is_indoor(npc.location):
                npc.modify_hp(NPC_INDOOR_REGEN)

            if _cfg.RL_ENABLED and self.turn > NPC_COLD_START_TURNS:
                decay_epsilon(npc)

            # Log NPC action
            npc_importance = compute_importance(
                "npc_action",
                action_id,
                "success" if npc_result.get("success", True) else "fail",
                npc_result.get("effects", {}),
            )

            self.event_log.add_entry(
                turn=self.turn,
                time_of_day=self.world.time_of_day,
                event_type="npc_action",
                actor=uid,
                action=action_id,
                target=npc_result.get("target"),
                location=npc.location,
                outcome="success" if npc_result.get("success", True) else "fail",
                effects=npc_result.get("effects", {}),
                witnesses=[],
                narration=npc_result.get("narration", ""),
                importance=npc_importance,
            )

            npc_result["npc_uid"] = uid
            npc_result["action"] = action_id
            npc_result["location"] = npc.location
            npc_result["importance"] = npc_importance
            results.append(npc_result)

        # P1: Update prev community state for next turn's delta computation
        self._prev_community_state = self.compute_community_state()

        return results

    def _resolve_npc_action(
        self,
        npc: NPC,
        action_id: str,
        npcs_at_loc: list[NPC],
        player_here: bool,
        gossip_context: dict,
    ) -> dict:
        """Resolve a single NPC action.

        Returns:
            Dict with success, narration, target, effects.
        """
        target_uid = resolve_npc_target(npc, action_id, npcs_at_loc, player_here)

        match action_id:
            case "move_to":
                dest = get_movement_destination(npc, self.world.time_of_day, {})
                adjacent = LOCATION_ADJACENCY.get(npc.location, [])
                if dest in adjacent:
                    old_loc = npc.location
                    npc.location = dest
                    return {
                        "success": True,
                        "narration": f"{npc.name} moves to {dest}.",
                        "target": dest,
                        "effects": {"old_location": old_loc, "new_location": dest},
                    }
                else:
                    # Invalid location — penalty
                    npc.stats["happiness"] = max(0, npc.stats["happiness"] - 1)
                    return {
                        "success": False,
                        "narration": f"{npc.name} hesitates, unsure where to go.",
                        "target": None,
                        "effects": {"invalid_move": True},
                    }

            case "talk" | "greet" | "ask_info" | "persuade" | "trade" | "give_item" | "deceive" | "intimidate":
                if target_uid and target_uid == "player" and player_here:
                    # NPC → player social interaction
                    narration = f"{npc.name} approaches you to chat."
                    # Small reputation drift based on NPC mood
                    mood = npc.stats.get("happiness", 5)
                    if mood >= 7:
                        self.player.modify_reputation(npc.npc_uid, 1)
                    # P3: Social interactions boost mood
                    npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.3)
                    return {
                        "success": True,
                        "narration": narration,
                        "target": "player",
                        "effects": {},
                    }
                elif target_uid and target_uid != "player":
                    target_npc_obj = self.npc_registry.get(target_uid)
                    if target_npc_obj and not target_npc_obj.is_incapacitated():
                        result = resolve_npc_npc_interaction(npc, target_npc_obj, action_id, {})

                        # Gossip propagation after social interaction
                        if action_id in ("talk", "greet", "ask_info"):
                            recent_player_events = [
                                e for e in self.event_log.get_recent(5)
                                if e.get("actor") == "player"
                                and abs(e.get("effects", {}).get("reputation_change", 0)) >= 3
                            ]
                            for evt in recent_player_events[:1]:
                                propagate_gossip(
                                    npc, target_npc_obj, evt,
                                    self.turn, gossip_context,
                                )

                        return result
                # No target available — generic
                npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.1)
                return {
                    "success": False,
                    "narration": f"{npc.name} looks around for someone to talk to.",
                    "target": None,
                    "effects": {},
                }

            case "attack":
                if target_uid and target_uid == "player" and player_here:
                    # NPC attacks player
                    attacker = npc.get_combat_dict()
                    defender = self.player.get_combat_dict()
                    diff_cfg = self.difficulty.to_dict()
                    combat_result = resolve_attack(attacker, defender, diff_cfg)

                    self.player.in_combat = True
                    self.player.combat_target = npc.npc_uid

                    if combat_result.hit:
                        # Apply defending reduction
                        damage = combat_result.damage
                        self.player.modify_health(-damage)
                        narration = combat_result.narrative

                        # Last chance: HP = 0 during dynamic CP
                        if self.player.health <= 0:
                            cp = self.mdp.get_checkpoint(self.quest_manager.current_checkpoint)
                            if cp and cp.is_dynamic:
                                # Check if within 2 CPs of main flow
                                from backend.quest.nudge import compute_distance_to_main
                                dist = compute_distance_to_main(
                                    self.quest_manager.current_checkpoint, self.mdp
                                )
                                if dist <= 2:
                                    self.player.health = 1
                                    narration += " At the brink of death, you cling to life by sheer willpower."

                        return {
                            "success": True,
                            "narration": narration,
                            "target": "player",
                            "effects": {
                                "damage": damage,
                                "player_hp": self.player.health,
                            },
                        }
                    else:
                        return {
                            "success": False,
                            "narration": combat_result.narrative,
                            "target": "player",
                            "effects": {"damage": 0},
                        }
                elif target_uid and target_uid != "player":
                    target_npc_obj = self.npc_registry.get(target_uid)
                    if target_npc_obj and not target_npc_obj.is_incapacitated():
                        result = resolve_npc_npc_interaction(npc, target_npc_obj, "attack", {})
                        # Check incapacitation
                        if target_npc_obj.current_hp <= 0:
                            if target_npc_obj.quest_critical:
                                target_npc_obj.current_hp = 1
                            else:
                                target_npc_obj.incapacitate(self.turn)
                        return result
                return {
                    "success": False,
                    "narration": f"{npc.name} clenches their fists but finds no one to fight.",
                    "target": None,
                    "effects": {},
                }

            case "defend":
                npc.is_defending = True
                return {
                    "success": True,
                    "narration": f"{npc.name} takes a defensive stance.",
                    "target": None,
                    "effects": {},
                }

            case "flee":
                adjacent = LOCATION_ADJACENCY.get(npc.location, [])
                if adjacent:
                    dest = random.choice(adjacent)
                    npc.location = dest
                    return {
                        "success": True,
                        "narration": f"{npc.name} hurries away.",
                        "target": dest,
                        "effects": {"fled_to": dest},
                    }
                return {
                    "success": False,
                    "narration": f"{npc.name} looks around nervously.",
                    "target": None,
                    "effects": {},
                }

            case "rest":
                hp_gain = 5 if self.world.is_indoor(npc.location) else 2
                npc.modify_hp(hp_gain)
                npc.stats["health"] = min(10, npc.stats["health"] + 0.5)
                # P3: Resting restores mood
                npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.4)
                return {
                    "success": True,
                    "narration": f"{npc.name} takes a moment to rest.",
                    "target": None,
                    "effects": {"hp_restored": hp_gain},
                }

            case "work":
                income_gain = random.uniform(0.3, 1.0)
                npc.stats["income"] = min(10, npc.stats["income"] + income_gain)
                # P3: Working is tiring (slight mood loss) but satisfying (small boost if income high)
                if npc.stats.get("income", 0) > 7:
                    npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.1)  # Satisfaction
                else:
                    npc.stats["happiness"] = max(0, npc.stats["happiness"] - 0.15)  # Toil
                return {
                    "success": True,
                    "narration": f"{npc.name} busies themselves with work.",
                    "target": None,
                    "effects": {"income_gain": income_gain},
                }

            case "look" | "search" | "examine":
                npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.1)
                return {
                    "success": True,
                    "narration": f"{npc.name} looks around curiously.",
                    "target": None,
                    "effects": {},
                }

            case "eat":
                npc.modify_hp(3)
                npc.stats["health"] = min(10, npc.stats["health"] + 0.3)
                return {
                    "success": True,
                    "narration": f"{npc.name} eats a quick meal.",
                    "target": None,
                    "effects": {"hp_restored": 3},
                }

            case "sneak" | "hide":
                npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.05)
                return {
                    "success": True,
                    "narration": f"{npc.name} moves quietly.",
                    "target": None,
                    "effects": {},
                }

            case "steal":
                if target_uid and target_uid != "player":
                    target_npc_obj = self.npc_registry.get(target_uid)
                    if target_npc_obj:
                        npc.npc_relationships[target_uid] = max(
                            -100,
                            npc.npc_relationships.get(target_uid, 0) - 5,
                        )
                return {
                    "success": False,
                    "narration": f"{npc.name} eyes nearby belongings.",
                    "target": target_uid,
                    "effects": {},
                }

            case "wait":
                return {
                    "success": True,
                    "narration": f"{npc.name} waits patiently.",
                    "target": None,
                    "effects": {},
                }

            case _:
                return {
                    "success": True,
                    "narration": f"{npc.name} goes about their business.",
                    "target": None,
                    "effects": {},
                }

    def _filter_npc_narrations(self, npc_results: list[dict]) -> list[dict]:
        """Filter NPC action narrations based on player proximity and importance."""
        filtered: list[dict] = []
        for res in npc_results:
            uid = res.get("npc_uid", "")
            npc = self.npc_registry.get(uid)
            if not npc:
                continue

            display_info = filter_npc_narration(
                uid,
                npc.name,
                res.get("action", "wait"),
                res.get("location", ""),
                self.player.location,
                res.get("importance", 1),
            )
            res["display_type"] = display_info["display_type"]
            res["display_narration"] = display_info["narration"]
            res["name"] = npc.name
            filtered.append(res)
        return filtered

    # ── Random Events ─────────────────────────────────────────────────────

    def _check_random_events(self) -> list[dict]:
        """Check for and apply random events."""
        active_ids = [eid for e in self.world.active_events if (eid := e.get("id"))]
        freq_mult = self.difficulty.get("random_event_frequency", 1.0)

        new_events = self.random_events.check_events(
            turn=self.turn,
            time_of_day=self.world.time_of_day,
            active_event_ids=active_ids,
            player_reputation=self.player.reputation,
            global_reputation=self.player.global_reputation,
            npc_locations=_npc_locations_map(self.npc_registry),
            frequency_multiplier=freq_mult,
        )

        for event in new_events:
            self.world.add_event(event)
            # Log event
            self.event_log.add_entry(
                turn=self.turn,
                time_of_day=self.world.time_of_day,
                event_type="random_event",
                actor="world",
                action=event["id"],
                target=None,
                location=self.player.location,
                outcome="triggered",
                effects=event.get("effects", {}),
                witnesses=[],
                narration=event.get("description", "Something unexpected happens."),
                importance=3,
            )

        return new_events

    # ── Game Over Check ───────────────────────────────────────────────────

    def _trigger_defeat(self, reason: str, message: str) -> None:
        """Mark a scripted defeat to be finalized by _check_game_over this turn."""
        self.pending_defeat_reason = reason
        self.game_over_message = message

    def _check_game_over(self) -> str | None:
        """Check all game-ending conditions."""
        from backend.config import BANISHMENT_REPUTATION_THRESHOLD

        # Scripted defeats (e.g. betraying the elder) take precedence and are
        # not subject to quest auto-restart.
        if self.pending_defeat_reason:
            return "fail"
        if self.player.health <= 0:
            if not self.game_over_message:
                self.game_over_message = (
                    "You collapse, your strength spent. Darkness closes in, and your "
                    "journey ends here in Thornhaven."
                )
            return "fail"
        # Banishment — sustained havoc ruins the player's standing in the village.
        if self.player.global_reputation < BANISHMENT_REPUTATION_THRESHOLD:
            if not self.game_over_message:
                self.game_over_message = (
                    "Elder Maren's voice rings out across the square: 'You have been "
                    "wreaking havoc among us. Thornhaven will suffer it no longer — you "
                    "are banished.'"
                )
            return "fail"
        if self.quest_manager.quest_complete:
            if self.restart_on_complete and self.turn < self.max_turns:
                self._restart_quest()
                return None
            return "success"
        if self.quest_manager.quest_failed:
            if self.restart_on_complete and self.turn < self.max_turns:
                self._restart_quest()
                return None
            return "fail"
        if self.turn >= self.max_turns:
            return "turn_limit"
        return None

    def _restart_quest(self) -> None:
        """Reset quest state for a new episode. NPC learning (Q-tables, adaptation) is preserved."""
        self._metrics["quests_completed"] = self._metrics.get("quests_completed", 0) + 1
        self.quest_manager = QuestManager(self.mdp)
        self.player.quest_state = {
            "current_stage": 1,
            "current_checkpoint": "1_1",
            "completed_checkpoints": [],
        }
        self.player.location = "gate"
        self.player.health = self.player.max_health
        self.player.stamina = self.player.max_stamina
        logger.info("Quest restarted (episode %d), NPC learning preserved", self._metrics.get("quests_completed", 1))

    # ── Save / Load ───────────────────────────────────────────────────────

    def _auto_save(self) -> None:
        """Auto-save every AUTO_SAVE_INTERVAL turns, rotating MAX_AUTO_SAVES files."""
        if self.turn % AUTO_SAVE_INTERVAL != 0:
            return

        self._auto_save_counter += 1
        slot = f"auto_{(self._auto_save_counter - 1) % MAX_AUTO_SAVES + 1}"
        filepath = self.save_game(slot)
        self._auto_save_files.append(filepath)

        # Rotate: keep only MAX_AUTO_SAVES
        while len(self._auto_save_files) > MAX_AUTO_SAVES:
            old_file = self._auto_save_files.pop(0)
            old_path = Path(old_file)
            if old_path.exists():
                try:
                    old_path.unlink()
                except OSError:
                    pass
            backup_path = old_path.with_suffix(".json.backup")
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    pass

    def save_game(self, slot: str = "auto") -> str:
        """Save full game state to JSON file. Returns filepath.

        Creates a .backup copy for corruption recovery.
        """
        save_data: dict[str, Any] = {
            "save_version": SAVE_VERSION,
            "game_version": GAME_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "game_over": self.game_over,
            "game_result": self.game_result,
            "game_over_message": self.game_over_message,
            "pending_defeat_reason": self.pending_defeat_reason,
            "difficulty": self.difficulty.to_dict(),
            "world": self.world.to_dict(),
            "player": self.player.to_dict(),
            "npc_registry": {
                uid: npc.to_dict() for uid, npc in self.npc_registry.items()
            },
            "quest_manager": self.quest_manager.to_dict(),
            "event_log": self.event_log.to_list(),
            "metrics": self._metrics,
            "shock_state": self.shock_manager.to_dict(),
        }

        filename = f"save_{slot}.json"
        filepath = SAVES_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write save
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)

        # Atomic-ish recovery: keep a .backup so a torn write doesn't lose state.
        backup_path = filepath.with_suffix(".json.backup")
        shutil.copy2(str(filepath), str(backup_path))

        logger.info("Game saved to %s (turn %d)", filepath.name, self.turn)

        try:
            self.playthrough_logger.log_event("save", {
                "slot": slot,
                "filepath": str(filepath),
                "turn": self.turn,
            })
        except Exception:
            pass

        return str(filepath)

    def load_game(self, filepath: str) -> dict:
        """Restore full state from `filepath`, falling back to .backup on corruption."""
        path = Path(filepath)
        backup_path = path.with_suffix(".json.backup")

        save_data: dict | None = None
        for try_path in (path, backup_path):
            if not try_path.exists():
                continue
            try:
                with open(try_path, "r", encoding="utf-8") as f:
                    save_data = json.load(f)
                if try_path == backup_path:
                    logger.warning("Loaded from backup: %s", backup_path.name)
                break
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load %s: %s", try_path, exc)
                continue

        if save_data is None:
            raise FileNotFoundError(
                f"Save file not found or corrupted: {filepath}"
            )

        self.seed = save_data.get("seed", MASTER_SEED)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.turn = save_data.get("turn", 0)
        self.max_turns = save_data.get("max_turns", MAX_TURNS)
        self.game_over = save_data.get("game_over", False)
        self.game_result = save_data.get("game_result")
        self.game_over_message = save_data.get("game_over_message")
        self.pending_defeat_reason = save_data.get("pending_defeat_reason")

        self.difficulty.from_dict(save_data.get("difficulty", {}))
        self.world.from_dict(save_data.get("world", {}))
        self.player.from_dict(save_data.get("player", {}))

        # NPCs need their archetype definitions to be re-hydrated.
        archetypes = load_archetypes()
        npc_data = save_data.get("npc_registry", {})
        self.npc_registry = {}
        for uid, npc_dict in npc_data.items():
            arch_key = npc_dict.get("archetype")
            arch_data = archetypes.get(arch_key, {})
            self.npc_registry[uid] = NPC.from_dict(npc_dict, arch_data)

        self.quest_manager = QuestManager.from_dict(
            save_data.get("quest_manager", {}),
            self.mdp,
        )
        self.event_log.from_list(save_data.get("event_log", []))
        self._metrics = save_data.get("metrics", self._metrics)

        shock_data = save_data.get("shock_state")
        if shock_data:
            self.shock_manager.from_dict(shock_data)

        # Q-tables travel with the save, so skip re-warm-up on the next initialize().
        self._pretrained = True

        logger.info("Game loaded from %s (turn %d)", path.name, self.turn)

        try:
            self.playthrough_logger.log_event("load", {
                "filepath": str(path),
                "turn": self.turn,
            })
        except Exception:
            pass

        return self.get_full_state()

    def get_full_state(self) -> dict:
        """Complete game-state snapshot for the API / frontend."""
        npcs_here = get_npcs_at_location(self.npc_registry, self.player.location)
        loc = self.world.get_location(self.player.location)

        return {
            "turn": self.turn,
            "time_period": self.world.time_of_day,
            "player": self.player.to_dict(),
            "location": {
                "id": self.player.location,
                "name": loc.name if loc else self.player.location,
                "description": loc.description if loc else "",
                "type": loc.type if loc else "outdoor",
                "adjacent": self.world.get_adjacent(self.player.location),
                "objects": loc.objects if loc else [],
                "items_on_ground": loc.items_on_ground if loc else [],
                "discovered_pois": [
                    {
                        "poi_id": p.poi_id,
                        "name": p.name,
                        "description": p.description,
                        "searchable": p.searchable,
                        "has_hidden_items": bool(p.items_hidden),
                        "examine_text": p.examine_text or p.description,
                    }
                    for p in self.world.get_discovered_pois(self.player.location)
                ] if loc else [],
            },
            "npcs_here": [
                {
                    "npc_uid": n.npc_uid,
                    "name": n.name,
                    "archetype": n.archetype,
                    "status": n.status,
                    "reputation": self.player.get_reputation(n.npc_uid),
                    "reputation_label": self.player.get_reputation_label(n.npc_uid),
                }
                for n in npcs_here
            ],
            "quest": self.quest_manager.get_quest_progress(),
            "graph": self.mdp.to_graph_data(
                self.quest_manager.current_checkpoint,
                self.quest_manager.completed_checkpoints,
            ),
            "active_events": self.world.active_events,
            "game_over": self.game_over,
            "game_result": self.game_result,
            "game_over_message": self.game_over_message,
            "max_turns": self.max_turns,
        }

    def get_save_list(self) -> list[dict]:
        """List available save files with metadata."""
        saves: list[dict] = []
        if not SAVES_DIR.exists():
            return saves

        for path in sorted(SAVES_DIR.glob("save_*.json")):
            if path.suffix == ".backup":
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saves.append({
                    "filename": path.name,
                    "filepath": str(path),
                    "slot": path.stem.replace("save_", ""),
                    "turn": data.get("turn", 0),
                    "timestamp": data.get("timestamp", ""),
                    "game_version": data.get("game_version", ""),
                    "player_name": data.get("player", {}).get("name", "Unknown"),
                    "difficulty": data.get("difficulty", {}).get("preset", "normal"),
                })
            except (json.JSONDecodeError, OSError):
                saves.append({
                    "filename": path.name,
                    "filepath": str(path),
                    "slot": path.stem.replace("save_", ""),
                    "turn": -1,
                    "timestamp": "",
                    "error": "corrupted",
                })

        return saves

    def get_metrics(self) -> dict:
        """Return game metrics for the research dashboard."""
        npc_states = {}
        npc_adaptation = {}
        npc_rewards = {}
        npc_role_telemetry = {}
        
        for uid, npc in self.npc_registry.items():
            npc_states[uid] = {
                "name": npc.name,
                "location": npc.location,
                "status": npc.status,
                "hp": f"{npc.current_hp}/{npc.max_hp}",
                "happiness": npc.stats.get("happiness", 0),
                "epsilon": round(npc.epsilon, 4),
            }
            
            # STEP 3: Add adaptation telemetry
            npc_adaptation[uid] = {
                "cooperation_tendency": round(npc.adaptation_state["cooperation_tendency"], 4),
                "risk_aversion": round(npc.adaptation_state["risk_aversion"], 4),
                "social_sensitivity": round(npc.adaptation_state["social_sensitivity"], 4),
                "shock_resilience": round(npc.adaptation_state["shock_resilience"], 4),
            }
            
            # Add latest reward from trace if available
            if npc.reward_trace:
                latest_reward = npc.reward_trace[-1]
                npc_rewards[uid] = {
                    "penalty": round(latest_reward.get("penalty", 0.0), 4),
                    "individual": round(latest_reward.get("individual", 0.0), 4),
                    "community": round(latest_reward.get("community", 0.0), 4),
                    "total": round(latest_reward.get("total", 0.0), 4),
                }
            else:
                npc_rewards[uid] = {"penalty": 0.0, "individual": 0.0, "community": 0.0, "total": 0.0}
            
            # STEP 4: Add role telemetry
            npc_role_telemetry[uid] = {
                "role": npc.archetype,
                "actions_selected": npc.role_telemetry["actions_selected"],
                "role_aligned": npc.role_telemetry["role_aligned"],
                "role_misaligned": npc.role_telemetry["role_misaligned"],
                "role_coherence": (
                    round(npc.role_telemetry["role_aligned"] / max(1, npc.role_telemetry["actions_selected"]), 4)
                    if npc.role_telemetry["actions_selected"] > 0
                    else 0.0
                ),
            }
        
        return {
            **self._metrics,
            "current_turn": self.turn,
            "game_over": self.game_over,
            "game_result": self.game_result,
            "quest_progress": self.quest_manager.get_quest_progress(),
            "npc_states": npc_states,
            "npc_adaptation": npc_adaptation,
            "npc_rewards": npc_rewards,
            "npc_role_telemetry": npc_role_telemetry,
            "community_state": self.compute_community_state(),
            "shock_timeline": self.shock_manager.get_shock_timeline(),
            "active_shocks": self.shock_manager.get_active_shocks(),
            "player_health": self.player.health,
            "player_stamina": self.player.stamina,
            "player_location": self.player.location,
            "event_log_size": len(self.event_log),
        }
