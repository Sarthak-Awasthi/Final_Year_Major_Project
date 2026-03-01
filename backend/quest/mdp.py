"""
mdp.py — MDP data structures for the hierarchical quest system.

Macro MDP: Quest stages S1–S7 + terminal S_success / S_fail. γ = 1.0
Micro MDP: Checkpoints within each stage. γ = 0.95

Static checkpoint IDs:  "{stage}_{index}"  (e.g., "1_1", "3_2")
Dynamic checkpoint IDs: "{stage}_D{counter}" (e.g., "1_D1", "2_D3")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.config import logger


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """A single checkpoint (micro-state) within a quest stage.

    Attributes:
        checkpoint_id: Unique ID, e.g. '3_2' (static) or '3_D1' (dynamic).
        stage_id: Parent stage number.
        description: Narrative text shown to the player.
        location: World location where this checkpoint occurs.
        trigger: Optional dict describing the action that reveals this CP.
        completion_conditions: Dict mapping action keys to transition dicts
            (mirrors ``quest_transitions`` from the JSON data).
        rewards: Aggregated rewards from completing this checkpoint.
        highlighted_actions: Suggested action IDs for the player.
        next_checkpoint: Primary next checkpoint ID (from nudge_target).
        hint: Environmental / narrative hint text.
        is_dynamic: True if this checkpoint was generated at runtime.
        is_terminal: True if reaching this checkpoint ends the quest.
        nudge_target: Preferred next checkpoint for the nudging system.
    """

    checkpoint_id: str
    stage_id: int
    description: str
    location: str
    trigger: dict | None = None
    completion_conditions: dict | None = None
    rewards: dict = field(default_factory=dict)
    highlighted_actions: list[str] = field(default_factory=list)
    next_checkpoint: str | None = None
    hint: str = ""
    is_dynamic: bool = False
    is_terminal: bool = False
    nudge_target: str | None = None


@dataclass
class Stage:
    """A macro-state containing one or more checkpoints.

    Attributes:
        stage_id: Stage number (1-based).
        name: Human-readable stage title.
        description: Narrative overview of this stage.
        checkpoints: Ordered mapping of checkpoint_id → Checkpoint.
        next_stage: ID of the following stage, or None for the final stage.
    """

    stage_id: int
    name: str
    description: str
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    next_stage: int | None = None


# ─── QuestMDP ────────────────────────────────────────────────────────────────

class QuestMDP:
    """Hierarchical MDP representation of the quest system.

    Parses the quest JSON into stages and checkpoints and provides
    traversal, lookup, and graph-export helpers.
    """

    MACRO_GAMMA: float = 1.0
    MICRO_GAMMA: float = 0.95

    def __init__(self, quest_data: dict) -> None:
        """Parse stages and checkpoints from quest JSON data.

        Args:
            quest_data: Deserialized contents of ``main_quest.json``.
        """
        self.quest_id: str = quest_data["quest_id"]
        self.title: str = quest_data["title"]
        self.stages: dict[int, Stage] = {}
        self._parse_stages(quest_data)
        logger.debug(
            "QuestMDP initialised: %s (%d stages, %d checkpoints)",
            self.quest_id,
            len(self.stages),
            len(self.get_all_checkpoints()),
        )

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse_stages(self, quest_data: dict) -> None:
        """Build internal stage / checkpoint graph from raw JSON."""
        stages_list = quest_data.get("stages", [])
        for idx, stage_data in enumerate(stages_list):
            stage_id: int = stage_data["stage_id"]
            next_stage = stages_list[idx + 1]["stage_id"] if idx + 1 < len(stages_list) else None

            stage = Stage(
                stage_id=stage_id,
                name=stage_data.get("title", stage_data.get("name", f"Stage {stage_id}")),
                description=stage_data.get("description", ""),
                next_stage=next_stage,
            )

            for cp_data in stage_data.get("checkpoints", []):
                cp = self._parse_checkpoint(cp_data, stage_data)
                stage.checkpoints[cp.checkpoint_id] = cp

            self.stages[stage_id] = stage

    def _parse_checkpoint(self, cp_data: dict, stage_data: dict) -> Checkpoint:
        """Convert a single checkpoint dict from JSON into a Checkpoint."""
        cp_id: str = cp_data["cp_id"]
        stage_id = self.get_stage_for_checkpoint(cp_id)

        # Highlighted action IDs — JSON stores list[dict] or list[str]
        highlighted: list[str] = []
        for entry in cp_data.get("highlighted_actions", []):
            highlighted.append(entry["id"] if isinstance(entry, dict) else entry)

        # Aggregate rewards across all quest_transitions
        transitions: dict = cp_data.get("quest_transitions", {})
        rewards: dict = {}
        for _key, trans in transitions.items():
            effects = trans.get("effects", {})
            if "reputation" in effects:
                rewards.setdefault("reputation", {})
                for npc_uid, delta in effects["reputation"].items():
                    # Keep the max positive delta per NPC across transitions
                    prev = rewards["reputation"].get(npc_uid, 0)
                    rewards["reputation"][npc_uid] = max(prev, delta)

        # Primary next checkpoint — prefer nudge_target, else first transition
        next_cp: str | None = cp_data.get("nudge_target")
        if next_cp is None and transitions:
            first_trans = next(iter(transitions.values()))
            next_cp = first_trans.get("next")

        # Hint text from context.environment
        context_block = cp_data.get("context", {})
        hint = context_block.get("environment", "")

        return Checkpoint(
            checkpoint_id=cp_id,
            stage_id=stage_id,
            description=cp_data.get("description", ""),
            location=stage_data.get("location", ""),
            trigger=None,
            completion_conditions=transitions if transitions else None,
            rewards=rewards,
            highlighted_actions=highlighted,
            next_checkpoint=next_cp,
            hint=hint,
            is_dynamic=cp_data.get("is_dynamic", False),
            is_terminal=cp_data.get("is_terminal", False),
            nudge_target=cp_data.get("nudge_target"),
        )

    # ── Lookups ───────────────────────────────────────────────────────────

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Retrieve a checkpoint by its ID, or None if not found."""
        stage_id = self.get_stage_for_checkpoint(checkpoint_id)
        stage = self.stages.get(stage_id)
        if stage is None:
            return None
        return stage.checkpoints.get(checkpoint_id)

    def get_stage(self, stage_id: int) -> Stage | None:
        """Retrieve a stage by its numeric ID, or None if not found."""
        return self.stages.get(stage_id)

    def get_next_checkpoint(self, current_cp_id: str) -> str | None:
        """Return the primary next checkpoint ID following *current_cp_id*."""
        cp = self.get_checkpoint(current_cp_id)
        return cp.next_checkpoint if cp else None

    @staticmethod
    def get_stage_for_checkpoint(checkpoint_id: str) -> int:
        """Extract the stage number from a checkpoint ID.

        Works for both static (``'3_2'``) and dynamic (``'3_D1'``) formats.
        """
        return int(checkpoint_id.split("_")[0])

    # ── Mutation ──────────────────────────────────────────────────────────

    def add_dynamic_checkpoint(self, stage_id: int, checkpoint: Checkpoint) -> None:
        """Insert a dynamically generated checkpoint into a stage.

        Args:
            stage_id: Target stage number.
            checkpoint: The new Checkpoint to add.
        """
        stage = self.stages.get(stage_id)
        if stage is None:
            logger.warning("Cannot add dynamic CP: stage %d not found", stage_id)
            return
        stage.checkpoints[checkpoint.checkpoint_id] = checkpoint
        logger.info(
            "Dynamic checkpoint %s added to stage %d",
            checkpoint.checkpoint_id,
            stage_id,
        )

    # ── Iteration helpers ─────────────────────────────────────────────────

    def get_all_checkpoints(self) -> list[Checkpoint]:
        """Return a flat list of every checkpoint across all stages."""
        result: list[Checkpoint] = []
        for stage in self.stages.values():
            result.extend(stage.checkpoints.values())
        return result

    def get_checkpoint_ids_for_stage(self, stage_id: int) -> list[str]:
        """Return all checkpoint IDs belonging to *stage_id*."""
        stage = self.stages.get(stage_id)
        return list(stage.checkpoints.keys()) if stage else []

    # ── Visualisation ─────────────────────────────────────────────────────

    def to_graph_data(
        self,
        current_cp_id: str | None = None,
        completed_cps: list[str] | None = None,
    ) -> dict:
        """Return a hierarchical graph with stage nodes + checkpoint nodes.

        The graph has two tiers:
        - **Stage nodes** (``kind='stage'``): arranged in a horizontal row,
          connected sequentially with thick arrows.
        - **Checkpoint nodes** (``kind='checkpoint'``): positioned below their
          parent stage, connected with thinner edges.

        Args:
            current_cp_id: The checkpoint to highlight as ``'current'``.
            completed_cps: List of already-completed checkpoint IDs.

        Returns:
            ``{"nodes": [...], "edges": [...]}`` suitable for Cytoscape.js.
        """
        completed_set: set[str] = set(completed_cps or [])
        current_stage_id: int | None = None
        if current_cp_id:
            try:
                current_stage_id = self.get_stage_for_checkpoint(current_cp_id)
            except (ValueError, IndexError):
                current_stage_id = None

        nodes: list[dict] = []
        edges: list[dict] = []
        seen_edges: set[tuple[str, str]] = set()

        sorted_stages = sorted(self.stages.values(), key=lambda s: s.stage_id)

        # ── Layout constants ──────────────────────────────────────────
        stage_x_gap = 220       # horizontal spacing between stages
        cp_y_start = 140        # vertical offset for first checkpoint row
        cp_y_gap = 80           # vertical gap between checkpoint rows
        cp_x_spread = 80        # horizontal spread within a stage

        for si, stage in enumerate(sorted_stages):
            stage_x = si * stage_x_gap
            stage_node_id = f"stage_{stage.stage_id}"

            # Determine stage status
            all_cps = list(stage.checkpoints.keys())
            stage_completed = all(c in completed_set for c in all_cps) if all_cps else False
            stage_is_current = (current_stage_id == stage.stage_id) and not stage_completed

            if stage_completed:
                stage_type = "stage_completed"
            elif stage_is_current:
                stage_type = "stage_current"
            else:
                stage_type = "stage"

            nodes.append({
                "id": stage_node_id,
                "label": f"S{stage.stage_id}: {stage.name}",
                "kind": "stage",
                "type": stage_type,
                "stage_id": stage.stage_id,
                "position": {"x": stage_x, "y": 0},
            })

            # Stage-to-stage edge
            if si + 1 < len(sorted_stages):
                next_stage_node = f"stage_{sorted_stages[si + 1].stage_id}"
                edges.append({
                    "source": stage_node_id,
                    "target": next_stage_node,
                    "type": "stage_link",
                })

            # ── Checkpoint nodes beneath this stage ───────────────────
            cp_list = list(stage.checkpoints.values())
            num_cps = len(cp_list)

            for ci, cp in enumerate(cp_list):
                # Determine checkpoint type/status
                if current_cp_id and cp.checkpoint_id == current_cp_id:
                    cp_type = "current"
                elif cp.checkpoint_id in completed_set:
                    cp_type = "completed"
                elif cp.is_terminal:
                    cp_type = "terminal"
                elif cp.is_dynamic:
                    cp_type = "dynamic"
                else:
                    cp_type = "static"

                # Position: spread checkpoints under their stage node
                x_offset = (ci - (num_cps - 1) / 2) * cp_x_spread
                cp_x = stage_x + x_offset
                cp_y = cp_y_start + (ci // 3) * cp_y_gap  # wrap rows of 3

                nodes.append({
                    "id": cp.checkpoint_id,
                    "label": cp.checkpoint_id,
                    "kind": "checkpoint",
                    "type": cp_type,
                    "stage_id": cp.stage_id,
                    "parent_stage": stage_node_id,
                    "position": {"x": cp_x, "y": cp_y},
                })

                # Edge from stage node to first checkpoint
                if ci == 0:
                    edges.append({
                        "source": stage_node_id,
                        "target": cp.checkpoint_id,
                        "type": "stage_to_cp",
                    })

                # Checkpoint-to-checkpoint edges from transitions
                if cp.completion_conditions:
                    for _key, trans in cp.completion_conditions.items():
                        target = trans.get("next")
                        if target and target not in ("S_success", "S_fail"):
                            edge_key = (cp.checkpoint_id, target)
                            if edge_key not in seen_edges:
                                # Determine edge type
                                edge_type = "completed" if cp.checkpoint_id in completed_set else "default"
                                edges.append({
                                    "source": cp.checkpoint_id,
                                    "target": target,
                                    "type": edge_type,
                                })
                                seen_edges.add(edge_key)
                        elif target in ("S_success", "S_fail"):
                            edge_key = (cp.checkpoint_id, target)
                            if edge_key not in seen_edges:
                                edges.append({
                                    "source": cp.checkpoint_id,
                                    "target": target,
                                    "type": "terminal_link",
                                })
                                seen_edges.add(edge_key)

        # ── Terminal pseudo-nodes ─────────────────────────────────────
        last_x = (len(sorted_stages) - 1) * stage_x_gap
        nodes.append({
            "id": "S_success",
            "label": "Victory",
            "kind": "terminal",
            "type": "terminal_success",
            "stage_id": -1,
            "position": {"x": last_x + stage_x_gap, "y": cp_y_start},
        })
        nodes.append({
            "id": "S_fail",
            "label": "Defeat",
            "kind": "terminal",
            "type": "terminal_fail",
            "stage_id": -1,
            "position": {"x": last_x + stage_x_gap, "y": cp_y_start + cp_y_gap},
        })

        return {"nodes": nodes, "edges": edges}
