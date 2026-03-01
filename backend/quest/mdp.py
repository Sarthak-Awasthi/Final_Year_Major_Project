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

    def to_graph_data(self, current_cp_id: str | None = None) -> dict:
        """Return nodes and edges suitable for Cytoscape.js rendering.

        Args:
            current_cp_id: Optional current checkpoint to mark as ``'current'``
                in the node type field.

        Returns:
            ``{"nodes": [...], "edges": [...]}`` where each node is
            ``{id, label, type, stage_id}`` and each edge is
            ``{source, target}``.
        """
        nodes: list[dict] = []
        edges: list[dict] = []
        seen_edges: set[tuple[str, str]] = set()

        for stage in self.stages.values():
            for cp in stage.checkpoints.values():
                # Determine node type
                if current_cp_id and cp.checkpoint_id == current_cp_id:
                    cp_type = "current"
                elif cp.is_terminal:
                    cp_type = "terminal"
                elif cp.is_dynamic:
                    cp_type = "dynamic"
                else:
                    cp_type = "static"

                label = cp.checkpoint_id
                short_desc = cp.description[:50]
                if len(cp.description) > 50:
                    short_desc += "…"
                label = f"{cp.checkpoint_id}: {short_desc}"

                nodes.append({
                    "id": cp.checkpoint_id,
                    "label": label,
                    "type": cp_type,
                    "stage_id": cp.stage_id,
                })

                # Edges from completion_conditions (quest_transitions)
                if cp.completion_conditions:
                    for _key, trans in cp.completion_conditions.items():
                        target = trans.get("next")
                        if target:
                            edge = (cp.checkpoint_id, target)
                            if edge not in seen_edges:
                                edges.append({"source": cp.checkpoint_id, "target": target})
                                seen_edges.add(edge)

        # Terminal state pseudo-nodes
        nodes.append({"id": "S_success", "label": "Quest Complete", "type": "terminal", "stage_id": -1})
        nodes.append({"id": "S_fail", "label": "Quest Failed", "type": "terminal", "stage_id": -1})

        return {"nodes": nodes, "edges": edges}
