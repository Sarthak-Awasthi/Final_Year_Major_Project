"""
quest_manager.py — Quest progression manager.

Tracks the player's position in the hierarchical MDP, checks checkpoint
completion conditions, handles deviations, and serialises / deserialises
quest state for save/load.
"""

from __future__ import annotations

from collections import defaultdict

from backend.config import (
    DYNAMIC_CP_LOOP_THRESHOLD,
    NUDGE_FORCE_CONVERGENCE_THRESHOLD,
    logger,
)
from backend.quest.mdp import Checkpoint, QuestMDP


class QuestManager:
    """Manages quest progression through the hierarchical MDP.

    One instance exists per game session.  It owns no data itself — it
    references a shared :class:`QuestMDP` for the graph topology.
    """

    def __init__(self, mdp: QuestMDP) -> None:
        """Initialise at the first checkpoint of stage 1.

        Args:
            mdp: The parsed quest MDP graph.
        """
        self.mdp = mdp
        self.current_stage: int = 1
        self.current_checkpoint: str = "1_1"
        self.completed_checkpoints: list[str] = []
        self.dynamic_checkpoints: list[str] = []
        self.deviation_count: int = 0
        self.dynamic_counter: dict[int, int] = defaultdict(int)
        self.quest_complete: bool = False
        self.quest_failed: bool = False
        # (action_id, stage_id) → count — used for loop detection
        self._action_history: dict[tuple[str, int], int] = {}
        # Tracks the last static checkpoint before deviation started.
        # Used to check convergence back to the main quest path.
        self._deviation_origin: str | None = None
        logger.debug("QuestManager initialised at checkpoint %s", self.current_checkpoint)

    # ── Completion check ──────────────────────────────────────────────────

    def _check_checkpoint_completion(
        self,
        checkpoint_id: str,
        action_id: str,
        target: str | None,
        context: dict,
    ) -> dict | None:
        """Core logic: check whether *action_id* satisfies a specific checkpoint.

        This is the shared implementation used by both :meth:`check_completion`
        (current checkpoint) and :meth:`check_convergence` (deviation origin).

        Returns:
            A result dict or ``None`` when conditions are not satisfied.
        """
        cp = self.mdp.get_checkpoint(checkpoint_id)
        if cp is None or cp.completion_conditions is None:
            return None

        matched = self._match_transition(action_id, target, context, cp)
        if matched is None:
            return None

        next_cp: str | None = matched.get("next")
        effects: dict = matched.get("effects", {})

        # Build rewards summary
        rewards: dict = {}
        if "reputation" in effects:
            rewards["reputation"] = effects["reputation"]
        if "gives" in effects:
            rewards["gives"] = effects["gives"]
        if "removes" in effects:
            rewards["removes"] = effects["removes"]
        if "stamina" in effects:
            rewards["stamina"] = effects["stamina"]

        # Detect cross-stage transition
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
        """Check whether *action_id* satisfies the current checkpoint.

        Matching logic:
          1. Exact key match in ``completion_conditions``.
          2. Compound key prefix match (e.g. ``move_to_fields``).

        Args:
            action_id: The action the player performed.
            target: Optional target NPC / item / location.
            context: Game context dict (must include ``target_location``
                when relevant).

        Returns:
            A result dict or ``None`` when conditions are not satisfied::

                {
                    "checkpoint_completed": str,
                    "next_checkpoint": str | None,
                    "rewards": dict,
                    "stage_transition": bool,
                }
        """
        return self._check_checkpoint_completion(
            self.current_checkpoint, action_id, target, context
        )

    def check_convergence(
        self,
        action_id: str,
        target: str | None,
        context: dict,
    ) -> dict | None:
        """Check if action satisfies the deviation origin, enabling convergence.

        When the player is on a dynamic checkpoint and performs an action that
        would satisfy the original static checkpoint they deviated from, this
        method detects it so the player can converge back to the main path.

        Returns:
            Completion dict or ``None``.
        """
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

    @staticmethod
    def _match_transition(
        action_id: str,
        target: str | None,
        context: dict,
        cp: Checkpoint,
    ) -> dict | None:
        """Find the matching transition dict inside *cp.completion_conditions*.

        Also validates ``requires`` constraints (e.g. the player must have
        a specific item).  The *context* dict should include
        ``player_inventory`` (list of item dicts) when callers want
        ``requires.item`` checks to be enforced.

        Returns the raw transition dict, or ``None``.
        """
        conditions = cp.completion_conditions
        if conditions is None:
            return None

        matched: dict | None = None

        # 1. Exact match
        if action_id in conditions:
            matched = conditions[action_id]
        else:
            # 2. Compound-key match (e.g. "move_to_fields")
            target_location = context.get("target_location", "")
            for key, transition in conditions.items():
                if not key.startswith(action_id):
                    continue
                suffix = key[len(action_id) + 1:] if len(key) > len(action_id) else ""
                if suffix and target_location and suffix == target_location:
                    matched = transition
                    break
                if suffix and target and suffix == target:
                    matched = transition
                    break
                if not suffix:
                    matched = transition
                    break

        if matched is None:
            return None

        # 3. Validate `requires` constraints
        requires = matched.get("requires")
        if requires:
            inventory: list[dict] = context.get("player_inventory", [])
            # requires.item — player must possess item with matching id
            required_item = requires.get("item")
            if required_item:
                has_item = any(
                    itm.get("id") == required_item for itm in inventory
                )
                if not has_item:
                    return None
            # requires.location — player must be at specified location
            required_loc = requires.get("location")
            if required_loc and context.get("location") != required_loc:
                return None

        return matched

    # ── State advancement ────────────────────────────────────────────────

    def advance_checkpoint(self, next_cp_id: str) -> None:
        """Move to the next checkpoint, recording the current one.

        Handles terminal states (``S_success`` / ``S_fail``) and
        cross-stage transitions automatically.  Also clears the deviation
        origin so convergence tracking resets.
        """
        old_cp = self.current_checkpoint
        self.completed_checkpoints.append(old_cp)

        # Also mark the deviation origin as completed when converging
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
        self.deviation_count = 0  # reset on forward progress
        self._deviation_origin = None  # clear deviation tracking

        new_stage = QuestMDP.get_stage_for_checkpoint(next_cp_id)
        if new_stage != self.current_stage:
            self.advance_stage(new_stage)

        logger.info("Advanced checkpoint: %s → %s", old_cp, next_cp_id)

    def advance_stage(self, next_stage: int) -> None:
        """Transition to a new quest stage."""
        old_stage = self.current_stage
        self.current_stage = next_stage
        logger.info("Stage transition: %d → %d", old_stage, next_stage)

    # ── Deviation handling ───────────────────────────────────────────────

    def handle_deviation(self, action_id: str, context: dict) -> dict:
        """Record and evaluate a player deviation from the expected path.

        On the first deviation, records the current checkpoint as the
        *deviation origin* so that convergence checks can later match
        the player's actions against the original checkpoint's conditions.

        Args:
            action_id: The off-path action taken.
            context: Current game context.

        Returns:
            ``{"needs_dynamic_cp": bool, "deviation_count": int,
              "force_convergence": bool}``
        """
        self.deviation_count += 1

        # Record the origin checkpoint on the first deviation so we can
        # check convergence back to the main path later.
        if self._deviation_origin is None:
            self._deviation_origin = self.current_checkpoint
            logger.info(
                "Deviation origin set to %s", self._deviation_origin
            )

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

    # ── Dynamic checkpoints ──────────────────────────────────────────────

    def generate_dynamic_cp_id(self, stage_id: int) -> str:
        """Generate a unique dynamic checkpoint ID.

        Format: ``"{stage}_D{counter}"`` with auto-incrementing counter
        per stage.
        """
        self.dynamic_counter[stage_id] += 1
        return f"{stage_id}_D{self.dynamic_counter[stage_id]}"

    def add_dynamic_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Insert a dynamic checkpoint into the MDP and track it."""
        self.mdp.add_dynamic_checkpoint(checkpoint.stage_id, checkpoint)
        self.dynamic_checkpoints.append(checkpoint.checkpoint_id)
        logger.info("Dynamic checkpoint %s tracked by manager", checkpoint.checkpoint_id)

    # ── Loop detection ───────────────────────────────────────────────────

    def check_loop_detection(self, action_id: str, stage_id: int) -> bool:
        """Detect repeated dynamic-CP generation for the same action/stage.

        Returns ``True`` when the same *action_id* has created a dynamic
        checkpoint at *stage_id* at least ``DYNAMIC_CP_LOOP_THRESHOLD``
        times (default 3), signalling forced convergence.
        """
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

    # ── Progress summary ─────────────────────────────────────────────────

    def get_quest_progress(self) -> dict:
        """Return a snapshot of current quest progress.

        Keys: quest_id, title, current_stage, current_checkpoint,
        completed_checkpoints, dynamic_checkpoints, deviation_count,
        completion_percent, quest_complete, quest_failed, total_checkpoints.
        """
        total = len(self.mdp.get_all_checkpoints())
        completed = len(self.completed_checkpoints)
        pct = (completed / max(total, 1)) * 100
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

    # ── Terminal states ──────────────────────────────────────────────────

    def trigger_failure(self) -> None:
        """Mark the quest as failed."""
        self.quest_failed = True
        logger.info("Quest '%s' FAILED", self.mdp.quest_id)

    def trigger_success(self) -> None:
        """Mark the quest as successfully completed."""
        self.quest_complete = True
        self.completed_checkpoints.append(self.current_checkpoint)
        logger.info("Quest '%s' COMPLETED successfully", self.mdp.quest_id)

    # ── Serialisation ────────────────────────────────────────────────────

    @property
    def deviation_origin(self) -> str | None:
        """The static checkpoint the player deviated from, or None."""
        return self._deviation_origin

    def to_dict(self) -> dict:
        """Serialise quest-manager state for save files."""
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
            "action_history": {
                f"{act}|{stg}": cnt
                for (act, stg), cnt in self._action_history.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict, mdp: QuestMDP) -> QuestManager:
        """Restore a QuestManager from previously saved state.

        Args:
            data: Dict produced by :meth:`to_dict`.
            mdp: The parsed quest MDP (must already include any dynamic CPs
                that were saved).

        Returns:
            A fully restored QuestManager instance.
        """
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

        # Restore action history
        manager._action_history = {}
        for key_str, count in data.get("action_history", {}).items():
            parts = key_str.split("|")
            if len(parts) == 2:
                manager._action_history[(parts[0], int(parts[1]))] = count

        logger.debug("QuestManager restored at checkpoint %s", manager.current_checkpoint)
        return manager
