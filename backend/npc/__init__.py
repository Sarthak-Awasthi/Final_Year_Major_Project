"""
npc — NPC subsystem for the MVP game.

Provides the NPC class, archetype/factory helpers, dialogue pipeline,
NPC–NPC interactions & gossip, knowledge base, Q-learning RL agent,
and fallback schedule system.
"""

from backend.npc.npc import NPC
from backend.npc.personality import (
    create_npc_registry,
    get_npcs_at_location,
    load_archetypes,
)
from backend.npc.dialogue import format_dialogue, resolve_dialogue
from backend.npc.interactions import (
    propagate_gossip,
    resolve_npc_npc_interaction,
    resolve_npc_target,
)
from backend.npc.knowledge import add_witnessed_event
from backend.npc.rl_agent import (
    compute_reward,
    decay_epsilon,
    get_valid_actions,
    pretrain_npc,
    select_action,
    update_q_table,
)
from backend.npc.schedule import (
    get_movement_destination,
    get_scheduled_action,
)

__all__ = [
    # npc.py
    "NPC",
    # personality.py
    "create_npc_registry",
    "get_npcs_at_location",
    "load_archetypes",
    # dialogue.py
    "format_dialogue",
    "resolve_dialogue",
    # interactions.py
    "propagate_gossip",
    "resolve_npc_npc_interaction",
    "resolve_npc_target",
    # knowledge.py
    "add_witnessed_event",
    # rl_agent.py
    "compute_reward",
    "decay_epsilon",
    "get_valid_actions",
    "pretrain_npc",
    "select_action",
    "update_q_table",
    # schedule.py
    "get_movement_destination",
    "get_scheduled_action",
]
