"""
npc — NPC subsystem for the MVP game.

Provides the NPC class, archetype/factory helpers, dialogue pipeline,
NPC–NPC interactions & gossip, knowledge base, Q-learning RL agent,
and fallback schedule system.
"""

from backend.npc.npc import NPC
from backend.npc.personality import (
    create_npc_registry,
    get_npc_by_name,
    get_npcs_at_location,
    load_archetypes,
    load_npc_instances,
    validate_archetype,
)
from backend.npc.dialogue import format_dialogue, resolve_dialogue
from backend.npc.interactions import (
    propagate_gossip,
    resolve_npc_npc_interaction,
    resolve_npc_target,
)
from backend.npc.knowledge import (
    NPCKnowledgeEntry,
    add_gossip_event,
    add_witnessed_event,
    get_player_opinion_summary,
    get_relevant_knowledge,
)
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
    is_schedule_time,
)

__all__ = [
    # npc.py
    "NPC",
    # personality.py
    "create_npc_registry",
    "get_npc_by_name",
    "get_npcs_at_location",
    "load_archetypes",
    "load_npc_instances",
    "validate_archetype",
    # dialogue.py
    "format_dialogue",
    "resolve_dialogue",
    # interactions.py
    "propagate_gossip",
    "resolve_npc_npc_interaction",
    "resolve_npc_target",
    # knowledge.py
    "NPCKnowledgeEntry",
    "add_gossip_event",
    "add_witnessed_event",
    "get_player_opinion_summary",
    "get_relevant_knowledge",
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
    "is_schedule_time",
]
