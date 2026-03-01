"""Quest system — hierarchical MDP, progression management, and dynamic checkpoints."""

from backend.quest.mdp import Checkpoint, QuestMDP, Stage
from backend.quest.quest_manager import QuestManager
from backend.quest.checkpoint import (
    CHECKPOINT_TEMPLATES,
    generate_dynamic_checkpoint,
)
from backend.quest.nudge import (
    compute_distance_to_main,
    compute_nudge_reward,
    get_convergence_checkpoint,
    get_nudge_hint,
)

__all__ = [
    "Checkpoint",
    "Stage",
    "QuestMDP",
    "QuestManager",
    "CHECKPOINT_TEMPLATES",
    "generate_dynamic_checkpoint",
    "compute_distance_to_main",
    "compute_nudge_reward",
    "get_convergence_checkpoint",
    "get_nudge_hint",
]
