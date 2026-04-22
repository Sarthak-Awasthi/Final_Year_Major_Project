"""
rl_agent.py — Tabular Q-learning for NPC agents.

Each NPC owns a 2-D numpy Q-table (states × actions).
This module provides action selection (ε-greedy with masking),
Q-value updates, reward computation, epsilon decay, and pre-training.
"""

from __future__ import annotations

import random as _random

import numpy as np

from backend.config import (
    LOCATION_ADJACENCY,
    LOCATION_IDS,
    MASTER_SEED,
    NPC_ACTION_SPACE_SIZE,
    NPC_COLD_START_TURNS,
    NPC_EPSILON_DECAY_RATE,
    NPC_EPSILON_MIN,
    NPC_EPSILON_PRETRAIN,
    NPC_INVALID_LOCATION_PENALTY,
    NPC_PRETRAIN_EPISODES,
    NPC_PRETRAIN_TURNS,
    NPC_Q_LEARNING_ALPHA,
    NPC_Q_LEARNING_GAMMA,
    TIME_PERIODS,
    UNIVERSAL_ACTION_IDS,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.npc.npc import NPC


# ── Action selection ──────────────────────────────────────────────────────────

def select_action(
    npc: NPC,
    state: int,
    valid_actions: list[int] | None = None,
) -> int:
    """Choose an action index using ε-greedy with optional action masking.

    When *valid_actions* is provided, only those indices are considered.
    Invalid actions get ``-inf`` during argmax selection (the Q-table
    itself is never modified).

    STEP 4: If ROLE_MASK_ENABLED, applies soft-mask adjustment based on NPC role.

    Args:
        npc: The acting NPC (provides ε and Q-table).
        state: Discretized state index.
        valid_actions: Optional list of action indices that pass hard
                       pre-condition checks.

    Returns:
        The chosen action index (into ``UNIVERSAL_ACTION_IDS``).
    """
    from backend.config import ROLE_MASK_ENABLED, ROLE_ACTION_MASKS, ROLE_MASK_BONUS, ROLE_MASK_PENALTY

    if valid_actions is not None and len(valid_actions) == 0:
        # Degenerate case — no valid actions; fall back to wait
        return UNIVERSAL_ACTION_IDS.index("wait")

    # ε-greedy exploration
    if _module_rng.random() < npc.epsilon:
        if valid_actions is not None:
            return _module_rng.choice(valid_actions)
        return _module_rng.randint(0, NPC_ACTION_SPACE_SIZE - 1)

    # Exploit: argmax with masking
    q_row = npc.q_table[state].copy()

    # Hard masking: invalid actions (precondition-based)
    if valid_actions is not None:
        mask = np.full(NPC_ACTION_SPACE_SIZE, -np.inf)
        for idx in valid_actions:
            mask[idx] = 0.0
        q_row += mask

    # STEP 4: Soft masking by role (policy prioritization, not hard blocking)
    if ROLE_MASK_ENABLED and hasattr(npc, 'archetype'):
        role_actions = ROLE_ACTION_MASKS.get(npc.archetype, [])
        if role_actions:
            for idx, action_id in enumerate(UNIVERSAL_ACTION_IDS):
                if action_id in role_actions:
                    q_row[idx] += ROLE_MASK_BONUS  # Boost role-aligned actions
                else:
                    q_row[idx] += ROLE_MASK_PENALTY  # Reduce role-misaligned actions

    # Break ties randomly among max values
    max_val = np.max(q_row)
    candidates = np.where(np.isclose(q_row, max_val))[0]
    return int(_module_rng.choice(candidates))


# ── Q-value update ────────────────────────────────────────────────────────────

def update_q_table(
    npc: NPC,
    state: int,
    action: int,
    reward: float,
    next_state: int,
) -> None:
    """Apply one step of Q-learning to *npc*'s Q-table.

    ``Q[s,a] = Q[s,a] + α * (r + γ * max(Q[s']) − Q[s,a])``

    Args:
        npc: The NPC whose table is being updated.
        state: State index where the action was taken.
        action: Action index that was taken.
        reward: Immediate reward received.
        next_state: Resulting state index.
    """
    old_q = npc.q_table[state, action]
    best_next = np.max(npc.q_table[next_state])
    td_target = reward + NPC_Q_LEARNING_GAMMA * best_next
    npc.q_table[state, action] = old_q + NPC_Q_LEARNING_ALPHA * (td_target - old_q)


# ── Reward computation ────────────────────────────────────────────────────────

def compute_penalty_reward(npc: NPC, old_stats: dict, new_stats: dict) -> float:
    """Compute penalty rewards for invalid or harmful actions.

    Returns:
        Negative reward for penalties, 0.0 otherwise.
    """
    penalty = 0.0

    # Penalty for health drop below 20% (self-harm)
    hp_ratio = npc.current_hp / max(npc.max_hp, 1)
    if hp_ratio < 0.2:
        penalty -= 10.0

    # Penalty for severe reputation damage
    # (Note: reputation damage is tracked separately, this is a catch-all)

    return penalty


def compute_individual_reward(npc: NPC, old_stats: dict, new_stats: dict) -> float:
    """Compute individual NPC reward from weighted stat deltas.

    ``R_individual = Σ w_i · (new_i − old_i)``

    Args:
        npc: The NPC (provides ``reward_weights``).
        old_stats: Stats dict before the action.
        new_stats: Stats dict after the action.

    Returns:
        Weighted scalar reward.
    """
    reward = 0.0
    for key, weight in npc.reward_weights.items():
        delta = new_stats.get(key, 0) - old_stats.get(key, 0)
        reward += weight * delta
    return reward


def compute_community_reward(community_state: dict | None) -> float:
    """Compute village-level reward from aggregated state.

    If community_state is None, returns 0.0 (community reward disabled).

    Community reward aggregates:
    - Average reputation (higher = better)
    - Total health (higher = better)
    - Average mood (derived from happiness stats)

    Args:
        community_state: Dict with keys: avg_reputation, total_health,
                        total_stamina, avg_mood. Can be None.

    Returns:
        Scalar community reward.
    """
    if community_state is None:
        return 0.0

    reward = 0.0

    # Reputation component (range ~-100 to +100, normalize to ±1 scale)
    avg_rep = community_state.get("avg_reputation", 0)
    reward += (avg_rep / 100.0) * 0.5  # weight 0.5

    # Health component (aggregate; higher is better)
    total_hp = community_state.get("total_health", 0)
    # With 6 NPCs at ~30 HP each, typical = 180, normalize to 0-1
    hp_score = min(total_hp / 200.0, 1.0)
    reward += hp_score * 0.3  # weight 0.3

    # Mood component (average mood sentiment)
    avg_mood = community_state.get("avg_mood", 0)
    reward += (avg_mood / 10.0) * 0.2  # weight 0.2

    return reward


def compute_reward(
    npc: NPC,
    old_stats: dict,
    new_stats: dict,
    community_state: dict | None = None,
) -> dict:
    """Compute decomposed reward signal with penalty, individual, and community terms.

    Args:
        npc: The NPC (provides ``reward_weights`` and ``lambda_coeff``).
        old_stats: Stats dict before the action.
        new_stats: Stats dict after the action.
        community_state: Optional village-level state for community reward.

    Returns:
        Dict with keys: penalty, individual, community, total.
        Total = penalty + individual + lambda * community
    """
    penalty = compute_penalty_reward(npc, old_stats, new_stats)
    individual = compute_individual_reward(npc, old_stats, new_stats)
    community = compute_community_reward(community_state)

    # Combine: total = penalty + individual + lambda_coeff * community
    total = penalty + individual + npc.lambda_coeff * community

    return {
        "penalty": float(penalty),
        "individual": float(individual),
        "community": float(community),
        "total": float(total),
    }


# ── Epsilon decay ─────────────────────────────────────────────────────────────

def decay_epsilon(npc: NPC) -> None:
    """Decay *npc*'s exploration rate toward the minimum.

    ``ε = max(ε_min, ε × decay_rate)``
    """
    npc.epsilon = max(NPC_EPSILON_MIN, npc.epsilon * NPC_EPSILON_DECAY_RATE)


# ── Valid-action computation ──────────────────────────────────────────────────

def get_valid_actions(npc: NPC, game_context: dict) -> list[int]:
    """Return indices of actions whose hard preconditions are met.

    All actions are always "available", but some will inevitably
    hard-fail.  This function identifies which ones can actually
    produce an outcome so the Q-learner can mask the rest.

    **No** location or checkpoint restrictions are applied.

    Args:
        npc: The acting NPC.
        game_context: Dict with keys like ``"npcs_at_location"``,
                      ``"player_location"``, ``"items_at_location"``, etc.

    Returns:
        Sorted list of valid action indices.
    """
    valid: list[int] = []
    adjacent = LOCATION_ADJACENCY.get(npc.location, [])
    npcs_here: list = game_context.get("npcs_at_location", [])
    player_here: bool = game_context.get("player_location") == npc.location
    has_target: bool = len(npcs_here) > 0 or player_here
    has_items: bool = bool(game_context.get("items_at_location", []))

    for idx, action_id in enumerate(UNIVERSAL_ACTION_IDS):
        match action_id:
            # Navigation
            case "move_to":
                if adjacent:
                    valid.append(idx)

            # Combat — needs a target present
            case "attack":
                if has_target:
                    valid.append(idx)
            case "defend" | "flee":
                # Always valid (defensive / escape)
                valid.append(idx)

            # Social — needs someone to interact with
            case "talk" | "greet" | "ask_info" | "persuade" | "trade" | "give_item" | "deceive" | "intimidate":
                if has_target:
                    valid.append(idx)

            # Stealth
            case "sneak" | "hide":
                valid.append(idx)
            case "steal":
                if has_target:
                    valid.append(idx)

            # Exploration
            case "look" | "search" | "examine":
                valid.append(idx)

            # Utility — most are always valid
            case "pick_up":
                if has_items:
                    valid.append(idx)
            case "use_item" | "eat" | "drop_item" | "equip":
                # NPCs don't have a rich inventory, but the action is
                # still "valid" — it'll just produce a benign outcome.
                valid.append(idx)
            case "rest" | "wait" | "status" | "work":
                valid.append(idx)

            case _:
                # Unknown action — allow (will be handled gracefully)
                valid.append(idx)

    return sorted(valid)


# ── Pre-training ──────────────────────────────────────────────────────────────

def pretrain_npc(
    npc: NPC,
    world_data: dict,
    episodes: int = NPC_PRETRAIN_EPISODES,
    turns_per_ep: int = NPC_PRETRAIN_TURNS,
    npc_index: int = 0,
    master_seed: int = MASTER_SEED,
) -> None:
    """Pre-train an NPC's Q-table in lightweight simulation mode.

    No narration, no event log, no witnesses, no gossip.  Just state
    transitions → reward → Q-update for the specified number of
    episodes and turns.

    The NPC's epsilon is set to ``NPC_EPSILON_PRETRAIN`` during
    pre-training and restored to ``NPC_EPSILON_START`` afterward.

    Args:
        npc: The NPC to pre-train.
        world_data: Dict with ``"locations"`` and adjacency info.
        episodes: Number of training episodes.
        turns_per_ep: Turns per episode.
        npc_index: Deterministic index of this NPC in the registry.
        master_seed: Master seed for deriving per-NPC seed.
    """
    # Deterministic seed per NPC derived from index, not hash()
    seed = (master_seed + npc_index * 13) % (2**31)
    rng = _random.Random(seed)
    np_rng = np.random.RandomState(seed % (2**31))

    original_epsilon = npc.epsilon
    original_location = npc.location
    original_hp = npc.current_hp
    original_stats = dict(npc.stats)

    npc.epsilon = NPC_EPSILON_PRETRAIN

    for ep in range(episodes):
        # Reset to a random valid location
        npc.location = rng.choice(LOCATION_IDS)
        npc.current_hp = npc.max_hp
        npc.stats = dict(original_stats)

        time_idx = 0

        for turn in range(turns_per_ep):
            time_period = TIME_PERIODS[time_idx % len(TIME_PERIODS)]
            state = npc.discretize_state(time_period)

            # Build minimal context
            ctx: dict = {
                "npcs_at_location": [],
                "player_location": None,
                "items_at_location": [],
            }
            valid = get_valid_actions(npc, ctx)
            action_idx = select_action(npc, state, valid)
            action_id = UNIVERSAL_ACTION_IDS[action_idx]

            # Snapshot stats before
            old_stats = dict(npc.stats)

            # Simulate action effects (simplified)
            _simulate_pretrain_action(npc, action_id, rng, np_rng)

            # Compute next state
            time_idx += 1
            next_time = TIME_PERIODS[time_idx % len(TIME_PERIODS)]
            next_state = npc.discretize_state(next_time)

            # Reward
            reward_dict = compute_reward(npc, old_stats, npc.stats)
            # Use total reward for Q-table update
            update_q_table(npc, state, action_idx, reward_dict["total"], next_state)

    # Restore NPC to initial state, keep the trained Q-table
    from backend.config import NPC_EPSILON_START

    npc.epsilon = NPC_EPSILON_START
    npc.location = original_location
    npc.current_hp = original_hp
    npc.stats = dict(original_stats)
    npc.is_defending = False

    logger.info(
        "Pre-trained NPC %s: %d episodes × %d turns",
        npc.npc_uid,
        episodes,
        turns_per_ep,
    )


def _simulate_pretrain_action(
    npc: NPC,
    action_id: str,
    rng: _random.Random,
    np_rng: np.random.RandomState,
) -> None:
    """Apply simplified action effects during pre-training.

    No narration, no witnesses, no event log.
    """
    adjacent = LOCATION_ADJACENCY.get(npc.location, [])

    match action_id:
        case "move_to":
            if adjacent:
                # Pick destination weighted by movement_weights (among adjacent)
                weights = [npc.movement_weights.get(loc, 0.1) for loc in adjacent]
                total = sum(weights)
                if total > 0:
                    probs = [w / total for w in weights]
                    npc.location = rng.choices(adjacent, weights=probs, k=1)[0]
                else:
                    npc.location = rng.choice(adjacent)
            else:
                # Invalid location penalty
                npc.stats["happiness"] = max(0, npc.stats["happiness"] - 1)

        case "rest":
            hp_gain = 5
            npc.current_hp = min(npc.max_hp, npc.current_hp + hp_gain)
            npc.stats["health"] = min(10, npc.stats["health"] + 0.5)

        case "work":
            income_gain = rng.uniform(0.3, 1.0)
            npc.stats["income"] = min(10, npc.stats["income"] + income_gain)
            npc.stats["happiness"] = max(0, npc.stats["happiness"] - 0.2)

        case "look" | "search" | "examine":
            # Small happiness from exploration
            npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.1)

        case "talk" | "greet" | "ask_info":
            # Social actions give small happiness boost in pre-training
            npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.2)
            npc.stats["reputation"] = min(10, npc.stats["reputation"] + 0.1)

        case "trade":
            npc.stats["income"] = min(10, npc.stats["income"] + 0.5)

        case "eat":
            npc.current_hp = min(npc.max_hp, npc.current_hp + 3)
            npc.stats["health"] = min(10, npc.stats["health"] + 0.3)

        case "wait":
            pass  # No effect

        case "attack":
            # Self-inflicted mild negative in pre-training (no target)
            npc.stats["reputation"] = max(0, npc.stats["reputation"] - 0.5)

        case "defend":
            npc.is_defending = True

        case "flee":
            # Move to random adjacent if possible
            if adjacent:
                npc.location = rng.choice(adjacent)

        case "hide" | "sneak":
            # Minor positive
            npc.stats["happiness"] = min(10, npc.stats["happiness"] + 0.05)

        case _:
            pass  # No effect for other actions


# ── Module-level seeded RNG ───────────────────────────────────────────────────

_module_rng = _random.Random(MASTER_SEED)
