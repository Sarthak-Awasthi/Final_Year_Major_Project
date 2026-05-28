"""Quest progression manager: tracks position in the MDP, matches actions to
transitions, handles deviations and convergence, and serialises for save/load."""

from __future__ import annotations

from collections import defaultdict

import backend.config as _cfg
from backend.config import (
    DYNAMIC_CP_LOOP_THRESHOLD,
    NUDGE_FORCE_CONVERGENCE_THRESHOLD,
    logger,
)
from backend.quest.mdp import Checkpoint, QuestMDP


class QuestManager:
    """One instance per game session. Owns mutable state; the MDP graph is shared."""

    def __init__(self, mdp: QuestMDP) -> None:
        self.mdp = mdp
        self.current_stage: int = 1
        self.current_checkpoint: str = "1_1"
        self.completed_checkpoints: list[str] = []
        self.dynamic_checkpoints: list[str] = []
        self.deviation_count: int = 0
        self.dynamic_counter: dict[int, int] = defaultdict(int)
        self.quest_complete: bool = False
        self.quest_failed: bool = False
        # Per-(action, stage) counter feeding loop detection.
        self._action_history: dict[tuple[str, int], int] = {}
        # Static CP the player last sat on before any deviation chain. Cleared
        # on advance; consulted by `check_convergence` so off-path actions can
        # still satisfy the original CP's transitions.
        self._deviation_origin: str | None = None
        logger.debug("QuestManager initialised at checkpoint %s", self.current_checkpoint)

    def _check_checkpoint_completion(
        self,
        checkpoint_id: str,
        action_id: str,
        target: str | None,
        context: dict,
    ) -> dict | None:
        """Shared completion logic for check_completion / check_convergence /
        check_forward_completion. Returns the transition result or None."""
        cp = self.mdp.get_checkpoint(checkpoint_id)
        if cp is None or cp.completion_conditions is None:
            return None

        matched = self._match_transition(action_id, target, context, cp)
        if matched is None:
            return None

        next_cp: str | None = matched.get("next")

        # Safety net for `requires_item` on the destination: block the
        # transition unless the player will hold that item after it fires
        # (already in inventory, or granted via this transition's `gives`).
        if next_cp and next_cp not in ("S_success", "S_fail"):
            next_cp_obj = self.mdp.get_checkpoint(next_cp)
            required_item = getattr(next_cp_obj, "requires_item", None) if next_cp_obj else None
            if required_item:
                effects_block = matched.get("effects", {}) or {}
                will_be_given = required_item in (effects_block.get("gives") or [])
                inventory: list[dict] = context.get("player_inventory", []) or []
                already_has = any(itm.get("id") == required_item for itm in inventory)
                if not (will_be_given or already_has):
                    return None

        effects: dict = matched.get("effects", {})
        rewards: dict = {}
        if "reputation" in effects:
            rewards["reputation"] = effects["reputation"]
        if "gives" in effects:
            rewards["gives"] = effects["gives"]
        if "removes" in effects:
            rewards["removes"] = effects["removes"]
        if "stamina" in effects:
            rewards["stamina"] = effects["stamina"]

        stage_transition = False
        if next_cp and next_cp not in ("S_success", "S_fail"):
            next_stage_id = QuestMDP.get_stage_for_checkpoint(next_cp)
            stage_transition = next_stage_id != self.current_stage
        elif next_cp in ("S_success", "S_fail"):
            stage_transition = True

        result = {
            "checkpoint_completed": checkpoint_id,
            "next_checkpoint": next_cp,
            "rewards": rewards,
            "stage_transition": stage_transition,
        }
        logger.info(
            "Checkpoint %s completed via '%s' → next: %s",
            checkpoint_id,
            action_id,
            next_cp,
        )
        return result

    def check_completion(
        self,
        action_id: str,
        target: str | None,
        context: dict,
    ) -> dict | None:
        return self._check_checkpoint_completion(
            self.current_checkpoint, action_id, target, context
        )

    def check_convergence(
        self,
        action_id: str,
        target: str | None,
        context: dict,
    ) -> dict | None:
        """If the player is on a dynamic CP, try to satisfy the static CP they came from."""
        if self._deviation_origin is None:
            return None
        result = self._check_checkpoint_completion(
            self._deviation_origin, action_id, target, context
        )
        if result:
            logger.info(
                "Convergence detected: action '%s' at dynamic CP %s "
                "satisfies origin %s → %s",
                action_id,
                self.current_checkpoint,
                self._deviation_origin,
                result.get("next_checkpoint"),
            )
        return result

    def check_forward_completion(
        self,
        action_id: str,
        target: str | None,
        context: dict,
    ) -> dict | None:
        """Allow a quest-critical action to skip ahead to a future static CP."""
        completed_set = set(self.completed_checkpoints)
        if _cfg.HIERARCHICAL_MDP:
            stages_to_scan = [self.mdp.stages.get(self.current_stage)]
        else:
            stages_to_scan = list(self.mdp.stages.values())

        # The immediate next CP is the job of `check_completion`; excluding it
        # here prevents trivial collisions where a bare action key matches a
        # neighbour (e.g. "talk" at 1_1 picking up 1_2's "talk" transition).
        current_cp_obj = self.mdp.get_checkpoint(self.current_checkpoint)
        nudge_target = getattr(current_cp_obj, "nudge_target", None) if current_cp_obj else None

        player_location = context.get("location")
        for stage in stages_to_scan:
            if stage is None:
                continue
            for cp_id, cp in stage.checkpoints.items():
                if cp.is_dynamic:
                    continue
                if cp_id in completed_set:
                    continue
                if cp_id == self.current_checkpoint:
                    continue
                if cp_id == nudge_target:
                    continue
                # Reject CPs whose scene is elsewhere — without this filter,
                # `move_to village_center` at CP 4_1 could leap to CP 5_1
                # (located at fields), teleporting the player.
                if cp.location and player_location and cp.location != player_location:
                    continue
                result = self._check_checkpoint_completion(
                    cp_id, action_id, target, context
                )
                if result is not None:
                    logger.info(
                        "Forward-scan match: action '%s' satisfies "
                        "future checkpoint %s (current: %s, stage %d)",
                        action_id,
                        cp_id,
                        self.current_checkpoint,
                        self.current_stage,
                    )
                    return result
        return None

    @staticmethod
    def _match_transition(
        action_id: str,
        target: str | None,
        context: dict,
        cp: Checkpoint,
    ) -> dict | None:
        """Resolve `action_id` against `cp.completion_conditions` keys.

        Two-stage match — exact key first, then compound suffix (e.g.
        "move_to_fields"). Both branches obey direction (target_location)
        and `requires` constraints. See feedback_transition_matching.md for
        the two traps this function navigates.
        """
        conditions = cp.completion_conditions
        if conditions is None:
            return None

        matched: dict | None = None

        if action_id in conditions:
            candidate = conditions[action_id]
            # Direction guard: if the bare-key transition declares an
            # expected target_location, only honor it when the player's
            # `target_location` agrees. Without this, CP 4_3's
            # `move_to → 5_1` (target_location: fields) would fire on
            # `move_to village_center` too, breaking forward-completion.
            expected_loc = (candidate.get("effects", {}) or {}).get("target_location")
            actual_loc = context.get("target_location")
            if not (expected_loc and actual_loc and expected_loc != actual_loc):
                matched = candidate

        if matched is None:
            # Compound key (e.g. "move_to_fields"). Only suffixed keys are
            # considered here — the bare-action key is already handled
            # above. Re-matching it would silently undo the direction guard.
            target_location = context.get("target_location", "")
            for key, transition in conditions.items():
                if not key.startswith(action_id) or key == action_id:
                    continue
                suffix = key[len(action_id) + 1:] if len(key) > len(action_id) else ""
                if suffix and target_location and suffix == target_location:
                    matched = transition
                    break
                if suffix and target and suffix == target:
                    matched = transition
                    break

        if matched is None:
            return None

        requires = matched.get("requires")
        if requires:
            inventory: list[dict] = context.get("player_inventory", [])
            required_item = requires.get("item")
            if required_item:
                has_item = any(itm.get("id") == required_item for itm in inventory)
                if not has_item:
                    return None
            required_loc = requires.get("location")
            if required_loc and context.get("location") != required_loc:
                return None

        # Action-success gating. Probability rolls (sneak, persuade) and
        # movement both demand a successful outcome — a blocked move_to
        # must not advance the quest even though its action_id matches a
        # transition key. Other actions stay lenient because their
        # resolvers report False for benign reasons (no target specified,
        # NPC asleep, etc.).
        if context.get("action_success") is False:
            if "success_prob" in matched or action_id == "move_to":
                return None

        return matched

    def advance_checkpoint(self, next_cp_id: str) -> None:
        """Step to `next_cp_id`; handle terminals and cross-stage hops."""
        old_cp = self.current_checkpoint
        if old_cp not in self.completed_checkpoints:
            self.completed_checkpoints.append(old_cp)

        # Mark the origin completed too when converging — otherwise the
        # main-path CP the player deviated from sits "unfinished" forever.
        if (
            self._deviation_origin is not None
            and self._deviation_origin != old_cp
            and self._deviation_origin not in self.completed_checkpoints
        ):
            self.completed_checkpoints.append(self._deviation_origin)

        if next_cp_id == "S_success":
            self._deviation_origin = None
            self.trigger_success()
            return
        if next_cp_id == "S_fail":
            self._deviation_origin = None
            self.trigger_failure()
            return

        self.current_checkpoint = next_cp_id
        self.deviation_count = 0
        self._deviation_origin = None

        new_stage = QuestMDP.get_stage_for_checkpoint(next_cp_id)
        if new_stage != self.current_stage:
            self.advance_stage(new_stage)

        logger.info("Advanced checkpoint: %s → %s", old_cp, next_cp_id)

    def advance_stage(self, next_stage: int) -> None:
        # Flat-MDP mode keeps everything in stage 1; the hierarchical stage
        # split is feature-flagged for ablation conditions.
        if not _cfg.HIERARCHICAL_MDP:
            return
        old_stage = self.current_stage
        self.current_stage = next_stage
        logger.info("Stage transition: %d → %d", old_stage, next_stage)

    def handle_deviation(self, action_id: str, context: dict) -> dict:
        """Record a deviation and report whether the engine should spawn a
        dynamic CP or force convergence."""
        self.deviation_count += 1

        if self._deviation_origin is None:
            self._deviation_origin = self.current_checkpoint
            logger.info("Deviation origin set to %s", self._deviation_origin)

        needs_dynamic = self.deviation_count >= 1
        force_convergence = self.deviation_count >= NUDGE_FORCE_CONVERGENCE_THRESHOLD
        logger.info(
            "Deviation #%d at checkpoint %s (origin: %s), action: %s",
            self.deviation_count,
            self.current_checkpoint,
            self._deviation_origin,
            action_id,
        )
        return {
            "needs_dynamic_cp": needs_dynamic,
            "deviation_count": self.deviation_count,
            "force_convergence": force_convergence,
        }

    def generate_dynamic_cp_id(self, stage_id: int) -> str:
        self.dynamic_counter[stage_id] += 1
        return f"{stage_id}_D{self.dynamic_counter[stage_id]}"

    def add_dynamic_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.mdp.add_dynamic_checkpoint(checkpoint.stage_id, checkpoint)
        self.dynamic_checkpoints.append(checkpoint.checkpoint_id)
        logger.info("Dynamic checkpoint %s tracked by manager", checkpoint.checkpoint_id)

    def check_loop_detection(self, action_id: str, stage_id: int) -> bool:
        """Detect spam: same action repeating in the same stage past the threshold."""
        key = (action_id, stage_id)
        self._action_history[key] = self._action_history.get(key, 0) + 1
        if self._action_history[key] >= DYNAMIC_CP_LOOP_THRESHOLD:
            logger.warning(
                "Loop detected: '%s' at stage %d repeated %d times",
                action_id,
                stage_id,
                self._action_history[key],
            )
            return True
        return False

    def get_quest_progress(self) -> dict:
        total = len(self.mdp.get_all_checkpoints())
        completed = len(self.completed_checkpoints)
        pct = (completed / max(total, 1)) * 100
        if self.quest_complete:
            pct = 100.0
        return {
            "quest_id": self.mdp.quest_id,
            "title": self.mdp.title,
            "current_stage": self.current_stage,
            "current_checkpoint": self.current_checkpoint,
            "completed_checkpoints": list(self.completed_checkpoints),
            "dynamic_checkpoints": list(self.dynamic_checkpoints),
            "deviation_count": self.deviation_count,
            "completion_percent": round(pct, 1),
            "quest_complete": self.quest_complete,
            "quest_failed": self.quest_failed,
            "total_checkpoints": total,
        }

    def trigger_failure(self) -> None:
        self.quest_failed = True
        logger.info("Quest '%s' FAILED", self.mdp.quest_id)

    def trigger_success(self) -> None:
        self.quest_complete = True
        if self.current_checkpoint not in self.completed_checkpoints:
            self.completed_checkpoints.append(self.current_checkpoint)
        logger.info("Quest '%s' COMPLETED successfully", self.mdp.quest_id)

    @property
    def deviation_origin(self) -> str | None:
        return self._deviation_origin

    def to_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "current_checkpoint": self.current_checkpoint,
            "completed_checkpoints": list(self.completed_checkpoints),
            "dynamic_checkpoints": list(self.dynamic_checkpoints),
            "deviation_count": self.deviation_count,
            "dynamic_counter": dict(self.dynamic_counter),
            "quest_complete": self.quest_complete,
            "quest_failed": self.quest_failed,
            "deviation_origin": self._deviation_origin,
            # Flatten the tuple key — JSON has no native tuple support.
            "action_history": {
                f"{act}|{stg}": cnt
                for (act, stg), cnt in self._action_history.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict, mdp: QuestMDP) -> QuestManager:
        manager = cls(mdp)
        manager.current_stage = data["current_stage"]
        manager.current_checkpoint = data["current_checkpoint"]
        manager.completed_checkpoints = list(data.get("completed_checkpoints", []))
        manager.dynamic_checkpoints = list(data.get("dynamic_checkpoints", []))
        manager.deviation_count = data.get("deviation_count", 0)
        manager.dynamic_counter = defaultdict(
            int,
            {int(k): v for k, v in data.get("dynamic_counter", {}).items()},
        )
        manager.quest_complete = data.get("quest_complete", False)
        manager.quest_failed = data.get("quest_failed", False)
        manager._deviation_origin = data.get("deviation_origin")

        manager._action_history = {}
        for key_str, count in data.get("action_history", {}).items():
            parts = key_str.split("|")
            if len(parts) == 2:
                manager._action_history[(parts[0], int(parts[1]))] = count

        logger.debug("QuestManager restored at checkpoint %s", manager.current_checkpoint)
        return manager
