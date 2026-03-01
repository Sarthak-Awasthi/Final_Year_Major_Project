"""
nudge.py — Nudging system for guiding deviated players back on track.

Provides reward shaping (R_nudge), narrative hints of increasing urgency,
and convergence-target resolution.

Key constants (from config):
    NUDGE_LAMBDA = 0.3
    NUDGE_HINT_THRESHOLD = 3          → explicit hint
    NUDGE_FORCE_CONVERGENCE_THRESHOLD = 5  → forced convergence
"""

from __future__ import annotations

from collections import deque

from backend.config import (
    NUDGE_FORCE_CONVERGENCE_THRESHOLD,
    NUDGE_HINT_THRESHOLD,
    NUDGE_LAMBDA,
    logger,
)
from backend.quest.mdp import QuestMDP


# ─── Reward Shaping ──────────────────────────────────────────────────────────

def compute_nudge_reward(
    current_cp: str,
    action_id: str,
    mdp: QuestMDP,
) -> float:
    """Compute a reward-shaping bonus for the nudging system.

    ``R_nudge = λ × (distance_before − distance_after)``

    A positive value means the action moves the player closer to the main
    quest flow; negative means farther away.

    Args:
        current_cp: Checkpoint the player is currently at.
        action_id: Action performed by the player.
        mdp: The quest MDP graph.

    Returns:
        Float bonus (can be negative).
    """
    distance_before = compute_distance_to_main(current_cp, mdp)

    # Determine where the action leads
    cp = mdp.get_checkpoint(current_cp)
    next_cp: str | None = None
    if cp and cp.completion_conditions:
        transition = cp.completion_conditions.get(action_id)
        if transition:
            next_cp = transition.get("next")

    if next_cp is None:
        # No explicit transition — fall back to nudge_target as proxy
        next_cp = cp.nudge_target if cp else None

    if next_cp is None or next_cp in ("S_success", "S_fail"):
        # Heading toward terminal — always positive
        return NUDGE_LAMBDA * distance_before

    distance_after = compute_distance_to_main(next_cp, mdp)
    reward = NUDGE_LAMBDA * (distance_before - distance_after)

    logger.debug(
        "Nudge reward for '%s' at %s: %.2f (dist %d → %d)",
        action_id,
        current_cp,
        reward,
        distance_before,
        distance_after,
    )
    return reward


# ─── Hint Generation ─────────────────────────────────────────────────────────

def get_nudge_hint(
    deviation_count: int,
    current_cp: str,
    mdp: QuestMDP,
) -> dict:
    """Generate a narrative hint whose urgency scales with deviations.

    Hint tiers:
        * ``deviation_count < 3`` → *subtle*  (gentle flavour text)
        * ``3 ≤ deviation_count < 5`` → *explicit* (clear directional hint)
        * ``deviation_count ≥ 5`` → *convergence* (forced redirect)

    Args:
        deviation_count: Number of consecutive deviations.
        current_cp: Current checkpoint ID.
        mdp: The quest MDP graph.

    Returns:
        ``{"type": "subtle"|"explicit"|"convergence",
          "text": str, "target_cp": str}``
    """
    cp = mdp.get_checkpoint(current_cp)
    stage_id = QuestMDP.get_stage_for_checkpoint(current_cp) if current_cp else 1
    target_cp = cp.nudge_target if cp else None

    if target_cp is None:
        target_cp = get_convergence_checkpoint(stage_id, mdp)

    # ── Tier 3: forced convergence ────────────────────────────────────
    if deviation_count >= NUDGE_FORCE_CONVERGENCE_THRESHOLD:
        convergence_cp = get_convergence_checkpoint(stage_id, mdp)
        text = (
            "An overwhelming sense of urgency washes over you. "
            "You feel compelled to return to the path ahead — "
            "the quest cannot wait any longer."
        )
        logger.info(
            "Force convergence hint at deviation %d, target: %s",
            deviation_count,
            convergence_cp,
        )
        return {
            "type": "convergence",
            "text": text,
            "target_cp": convergence_cp or target_cp or current_cp,
        }

    # ── Tier 2: explicit hint ─────────────────────────────────────────
    if deviation_count >= NUDGE_HINT_THRESHOLD:
        hint_text = (
            cp.hint
            if cp and cp.hint
            else "You sense you should refocus on your objective."
        )
        text = f"A nagging feeling tugs at your mind. {hint_text}"
        logger.info("Explicit hint at deviation %d, target: %s", deviation_count, target_cp)
        return {
            "type": "explicit",
            "text": text,
            "target_cp": target_cp or current_cp,
        }

    # ── Tier 1: subtle ────────────────────────────────────────────────
    text = "Something about your surroundings reminds you of the task at hand."
    return {
        "type": "subtle",
        "text": text,
        "target_cp": target_cp or current_cp,
    }


# ─── Convergence Target ──────────────────────────────────────────────────────

def get_convergence_checkpoint(current_stage: int, mdp: QuestMDP) -> str | None:
    """Find the next main-flow checkpoint to redirect the player to.

    Search order:
        1. First static (non-dynamic) checkpoint in *current_stage*.
        2. First static checkpoint in the **next** stage.

    Args:
        current_stage: The stage the player is currently in.
        mdp: The quest MDP graph.

    Returns:
        A checkpoint ID string, or ``None`` if nothing suitable is found.
    """
    stage = mdp.get_stage(current_stage)
    if stage:
        for cp_id, cp in stage.checkpoints.items():
            if not cp.is_dynamic:
                return cp_id
        # Current stage exhausted — try next stage
        if stage.next_stage is not None:
            next_stage = mdp.get_stage(stage.next_stage)
            if next_stage:
                for cp_id, cp in next_stage.checkpoints.items():
                    if not cp.is_dynamic:
                        return cp_id

    return None


# ─── Distance Computation ────────────────────────────────────────────────────

def compute_distance_to_main(checkpoint_id: str, mdp: QuestMDP) -> int:
    """BFS distance from *checkpoint_id* to the nearest static checkpoint.

    Static (non-dynamic) checkpoints are on the main flow and have
    distance 0.  Dynamic checkpoints count as 1 hop each.

    The BFS follows ``nudge_target``, ``next_checkpoint``, and all
    ``completion_conditions`` edges.

    Args:
        checkpoint_id: Starting checkpoint ID.
        mdp: The quest MDP graph.

    Returns:
        Integer hop count (0 if already on main flow, 999 if unreachable).
    """
    cp = mdp.get_checkpoint(checkpoint_id)
    if cp is None:
        return 0

    # Static checkpoints are on the main flow by definition
    if not cp.is_dynamic:
        return 0

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(checkpoint_id, 0)])

    while queue:
        current_id, dist = queue.popleft()

        if current_id in visited:
            continue
        visited.add(current_id)

        current_cp = mdp.get_checkpoint(current_id)
        if current_cp is None:
            continue

        # Found a static checkpoint (skip the starting node itself)
        if not current_cp.is_dynamic and current_id != checkpoint_id:
            return dist

        # Collect reachable neighbours
        neighbours: list[str] = []
        if current_cp.nudge_target and current_cp.nudge_target not in ("S_success", "S_fail"):
            neighbours.append(current_cp.nudge_target)
        if (
            current_cp.next_checkpoint
            and current_cp.next_checkpoint not in ("S_success", "S_fail")
            and current_cp.next_checkpoint != current_cp.nudge_target
        ):
            neighbours.append(current_cp.next_checkpoint)

        if current_cp.completion_conditions:
            for trans in current_cp.completion_conditions.values():
                target = trans.get("next")
                if target and target not in ("S_success", "S_fail") and target not in neighbours:
                    neighbours.append(target)

        for neighbour in neighbours:
            if neighbour not in visited:
                queue.append((neighbour, dist + 1))

    # No static checkpoint reachable
    return 999
