"""
analytics.py — Research analytics and time-series computation for the RL Playground.

Produces interpretable metrics aligned to the project hypothesis:
  Early episodes favor individual gains → policies shift toward cooperation.

Computes:
  - Per-NPC reward series (individual, community, total, penalty)
  - Village-level social welfare index over turns
  - Cooperation index (global + per role)
  - Policy entropy (per NPC, from action distributions)
  - Action distribution shift (early vs late windows)
  - Shock response and recovery curves
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from backend.config import LLM_ENABLED, LLM_PROVIDER, SHOCK_ENABLED, logger


# ── Reward Time-Series ────────────────────────────────────────────────────────


def compute_reward_series(npc_registry: dict) -> dict[str, dict[str, list]]:
    """Extract per-NPC reward time-series from reward_trace.

    Returns:
        ``{npc_uid: {turns: [...], individual: [...], community: [...],
                     penalty: [...], total: [...]}}``
    """
    series: dict[str, dict[str, list]] = {}
    for uid, npc in npc_registry.items():
        trace = npc.reward_trace
        series[uid] = {
            "npc_name": npc.name,
            "role": npc.archetype,
            "turns": [r["turn"] for r in trace],
            "individual": [round(r["individual"], 4) for r in trace],
            "community": [round(r["community"], 4) for r in trace],
            "penalty": [round(r["penalty"], 4) for r in trace],
            "total": [round(r["total"], 4) for r in trace],
        }
    return series


def compute_community_reward_series(npc_registry: dict) -> dict[str, list]:
    """Compute village-level community reward aggregated per turn.

    Returns:
        ``{turns: [...], avg_community: [...], avg_total: [...]}``
    """
    # Collect all reward traces into per-turn buckets
    turn_community: dict[int, list[float]] = defaultdict(list)
    turn_total: dict[int, list[float]] = defaultdict(list)

    for npc in npc_registry.values():
        for r in npc.reward_trace:
            t = r["turn"]
            turn_community[t].append(r["community"])
            turn_total[t].append(r["total"])

    if not turn_community:
        return {"turns": [], "avg_community": [], "avg_total": []}

    sorted_turns = sorted(turn_community.keys())
    return {
        "turns": sorted_turns,
        "avg_community": [
            round(sum(turn_community[t]) / len(turn_community[t]), 4)
            for t in sorted_turns
        ],
        "avg_total": [
            round(sum(turn_total[t]) / len(turn_total[t]), 4)
            for t in sorted_turns
        ],
    }


# ── Social Welfare Index ──────────────────────────────────────────────────────


def compute_social_welfare_series(npc_registry: dict) -> dict[str, list]:
    """Compute village social welfare index per turn.

    Social welfare = average(cooperation_tendency) across all NPCs at each turn,
    weighted by their community reward performance.

    Returns:
        ``{turns: [...], welfare_index: [...]}``
    """
    # Use adaptation traces to build per-turn welfare
    turn_coop: dict[int, list[float]] = defaultdict(list)

    for npc in npc_registry.values():
        for sample in npc.adaptation_trace:
            t = sample["turn"]
            turn_coop[t].append(sample["cooperation_tendency"])

    if not turn_coop:
        return {"turns": [], "welfare_index": []}

    sorted_turns = sorted(turn_coop.keys())
    return {
        "turns": sorted_turns,
        "welfare_index": [
            round(sum(turn_coop[t]) / len(turn_coop[t]), 4)
            for t in sorted_turns
        ],
    }


# ── Cooperation Index ─────────────────────────────────────────────────────────


def compute_cooperation_index(npc_registry: dict) -> dict[str, Any]:
    """Compute cooperation index: global and per-role.

    Cooperation index = average cooperation_tendency across NPCs.
    Tracked at the latest adaptation state snapshot.

    Returns:
        ``{global: float, per_role: {role: float}, per_npc: {uid: float}}``
    """
    role_coop: dict[str, list[float]] = defaultdict(list)
    npc_coop: dict[str, float] = {}
    all_coop: list[float] = []

    for uid, npc in npc_registry.items():
        coop = npc.adaptation_state["cooperation_tendency"]
        all_coop.append(coop)
        role_coop[npc.archetype].append(coop)
        npc_coop[uid] = round(coop, 4)

    global_coop = round(sum(all_coop) / len(all_coop), 4) if all_coop else 0.0
    per_role = {
        role: round(sum(vals) / len(vals), 4)
        for role, vals in role_coop.items()
    }

    return {
        "global": global_coop,
        "per_role": per_role,
        "per_npc": npc_coop,
    }


def compute_cooperation_series(npc_registry: dict) -> dict[str, list]:
    """Compute cooperation index over turns from adaptation traces.

    Returns:
        ``{turns: [...], global_cooperation: [...]}``
    """
    turn_coop: dict[int, list[float]] = defaultdict(list)

    for npc in npc_registry.values():
        for sample in npc.adaptation_trace:
            turn_coop[sample["turn"]].append(sample["cooperation_tendency"])

    if not turn_coop:
        return {"turns": [], "global_cooperation": []}

    sorted_turns = sorted(turn_coop.keys())
    return {
        "turns": sorted_turns,
        "global_cooperation": [
            round(sum(turn_coop[t]) / len(turn_coop[t]), 4)
            for t in sorted_turns
        ],
    }


# ── Policy Entropy ────────────────────────────────────────────────────────────


def compute_policy_entropy(npc_registry: dict) -> dict[str, float]:
    """Compute per-NPC policy entropy from per-action counts.

    H = -Σ p(a) log2 p(a) over all actions taken.
    High entropy = exploratory, low = exploitative.

    Returns:
        ``{npc_uid: entropy_value}``
    """
    entropies: dict[str, float] = {}
    for uid, npc in npc_registry.items():
        total = sum(npc.action_counts.values()) if npc.action_counts else 0
        if total == 0:
            entropies[uid] = 0.0
            continue

        entropy = 0.0
        for count in npc.action_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        entropies[uid] = round(entropy, 4)

    return entropies


# ── Action Distribution ───────────────────────────────────────────────────────


def compute_action_distribution(
    event_log_entries: list[dict],
    npc_registry: dict,
    window_size: int = 20,
) -> dict[str, Any]:
    """Compute action distribution: global and per-role, with early/late comparison.

    Args:
        event_log_entries: Full event log.
        npc_registry: NPC registry for role lookups.
        window_size: Number of turns for early/late window.

    Returns:
        Dict with ``global``, ``per_role``, ``early_window``, ``late_window``,
        and ``distribution_shift`` keys.
    """
    # Build NPC uid → role mapping
    uid_to_role: dict[str, str] = {
        uid: npc.archetype for uid, npc in npc_registry.items()
    }

    # Separate NPC actions from event log
    npc_actions = [
        e for e in event_log_entries
        if e.get("event_type") == "npc_action" and e.get("action")
    ]

    if not npc_actions:
        return {
            "global": {},
            "per_role": {},
            "early_window": {},
            "late_window": {},
            "distribution_shift": {},
        }

    # Global distribution
    global_counter: Counter = Counter()
    role_counters: dict[str, Counter] = defaultdict(Counter)

    for entry in npc_actions:
        action = entry["action"]
        actor = entry.get("actor", "")
        global_counter[action] += 1
        role = uid_to_role.get(actor, "unknown")
        role_counters[role][action] += 1

    # Early vs late windows
    max_turn = max(e.get("turn", 0) for e in npc_actions)
    early_cutoff = min(window_size, max_turn // 2)
    late_start = max(max_turn - window_size, max_turn // 2)

    early_counter: Counter = Counter()
    late_counter: Counter = Counter()

    for entry in npc_actions:
        turn = entry.get("turn", 0)
        action = entry["action"]
        if turn <= early_cutoff:
            early_counter[action] += 1
        if turn >= late_start:
            late_counter[action] += 1

    # Distribution shift: difference in proportions
    shift: dict[str, float] = {}
    early_total = sum(early_counter.values()) or 1
    late_total = sum(late_counter.values()) or 1
    all_actions = set(early_counter.keys()) | set(late_counter.keys())
    for action in all_actions:
        early_pct = early_counter.get(action, 0) / early_total
        late_pct = late_counter.get(action, 0) / late_total
        shift[action] = round(late_pct - early_pct, 4)

    return {
        "global": dict(global_counter.most_common()),
        "per_role": {
            role: dict(counter.most_common())
            for role, counter in role_counters.items()
        },
        "early_window": dict(early_counter.most_common()),
        "late_window": dict(late_counter.most_common()),
        "distribution_shift": shift,
    }


# ── Shock Response Curves ─────────────────────────────────────────────────────


def compute_shock_response(
    npc_registry: dict,
    shock_timeline: list[dict],
) -> list[dict]:
    """Compute shock response and recovery curves.

    For each shock in the timeline, extracts the adaptation and reward
    trajectory of NPCs during and after the shock period.

    Returns:
        List of shock response records, each containing:
        - shock metadata
        - avg cooperation before/during/after
        - avg reward before/during/after
    """
    if not shock_timeline:
        return []

    responses: list[dict] = []

    for shock in shock_timeline:
        shock_start = shock.get("turn_started", 0)
        shock_duration = shock.get("duration", 0)
        shock_end = shock_start + shock_duration
        pre_start = max(0, shock_start - shock_duration)

        # Collect adaptation data across all NPCs for each period
        pre_coop: list[float] = []
        during_coop: list[float] = []
        post_coop: list[float] = []

        pre_reward: list[float] = []
        during_reward: list[float] = []
        post_reward: list[float] = []

        for npc in npc_registry.values():
            for sample in npc.adaptation_trace:
                t = sample["turn"]
                if pre_start <= t < shock_start:
                    pre_coop.append(sample["cooperation_tendency"])
                elif shock_start <= t < shock_end:
                    during_coop.append(sample["cooperation_tendency"])
                elif shock_end <= t < shock_end + shock_duration:
                    post_coop.append(sample["cooperation_tendency"])

            for r in npc.reward_trace:
                t = r["turn"]
                if pre_start <= t < shock_start:
                    pre_reward.append(r["total"])
                elif shock_start <= t < shock_end:
                    during_reward.append(r["total"])
                elif shock_end <= t < shock_end + shock_duration:
                    post_reward.append(r["total"])

        _avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else None

        responses.append({
            "shock_id": shock.get("shock_id", ""),
            "shock_type": shock.get("shock_type", ""),
            "turn_started": shock_start,
            "duration": shock_duration,
            "status": shock.get("status", ""),
            "avg_cooperation_before": _avg(pre_coop),
            "avg_cooperation_during": _avg(during_coop),
            "avg_cooperation_after": _avg(post_coop),
            "avg_reward_before": _avg(pre_reward),
            "avg_reward_during": _avg(during_reward),
            "avg_reward_after": _avg(post_reward),
        })

    return responses


# ── Narrative Coherence ───────────────────────────────────────────────────


def compute_narrative_coherence(event_log_entries: list[dict]) -> dict[str, Any]:
    """Compute average similarity between consecutive narration segments.

    Uses spaCy vectors when available, falls back to token overlap ratio.

    Returns:
        ``{avg_coherence: float, num_segments: int, method: str}``
    """
    narrations = [
        e.get("narration", "") or e.get("description", "")
        for e in event_log_entries
        if e.get("event_type") in ("player_action", "narration") and (e.get("narration") or e.get("description"))
    ]

    if len(narrations) < 2:
        return {"avg_coherence": 0.0, "num_segments": len(narrations), "method": "none"}

    try:
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        docs = [nlp(n[:500]) for n in narrations]
        similarities = []
        for i in range(len(docs) - 1):
            if docs[i].vector_norm and docs[i + 1].vector_norm:
                similarities.append(docs[i].similarity(docs[i + 1]))
        if similarities:
            return {
                "avg_coherence": round(sum(similarities) / len(similarities), 4),
                "num_segments": len(narrations),
                "method": "spacy_vectors",
            }
    except Exception:
        pass

    # Fallback: token overlap ratio (Jaccard similarity)
    similarities = []
    for i in range(len(narrations) - 1):
        tokens_a = set(narrations[i].lower().split())
        tokens_b = set(narrations[i + 1].lower().split())
        if tokens_a or tokens_b:
            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if (tokens_a | tokens_b) else 0
            similarities.append(jaccard)

    avg = round(sum(similarities) / len(similarities), 4) if similarities else 0.0
    return {"avg_coherence": avg, "num_segments": len(narrations), "method": "token_overlap"}


# ── Deviation Recovery ───────────────────────────────────────────────────


def compute_deviation_recovery(event_log_entries: list[dict]) -> dict[str, Any]:
    """Compute deviation recovery metrics from event log.

    Returns:
        ``{total_deviations: int, natural_convergences: int, forced_convergences: int,
           recovery_rate: float, dynamic_checkpoints: int, checkpoint_usage_rate: float}``
    """
    total_deviations = 0
    natural_convergences = 0
    forced_convergences = 0
    dynamic_cps = 0
    total_turns = 0

    for e in event_log_entries:
        total_turns = max(total_turns, e.get("turn", 0))

        event_type = e.get("event_type", "")
        if event_type == "quest_deviation":
            total_deviations += 1
        elif event_type == "quest_convergence":
            if e.get("outcome") == "forced" or e.get("effects", {}).get("forced", False):
                forced_convergences += 1
            else:
                natural_convergences += 1
        elif event_type == "dynamic_checkpoint":
            dynamic_cps += 1

    recovery_rate = (
        round(natural_convergences / total_deviations, 4)
        if total_deviations > 0 else 0.0
    )
    cp_usage_rate = (
        round(dynamic_cps / max(total_turns, 1), 4)
    )

    return {
        "total_deviations": total_deviations,
        "natural_convergences": natural_convergences,
        "forced_convergences": forced_convergences,
        "recovery_rate": recovery_rate,
        "dynamic_checkpoints": dynamic_cps,
        "checkpoint_usage_rate": cp_usage_rate,
    }


# ── Shock Recovery Time ──────────────────────────────────────────────────


def compute_shock_recovery_time(
    npc_registry: dict,
    shock_timeline: list[dict],
) -> list[dict]:
    """Compute recovery time for each shock — turns until K(t) returns to pre-shock level.

    Also computes correlation between shock_resilience and recovery speed.

    Returns:
        List of dicts with shock metadata plus recovery_turns and resilience data.
    """
    if not shock_timeline:
        return []

    coop_series = compute_cooperation_series(npc_registry)
    turns = coop_series.get("turns", [])
    values = coop_series.get("global_cooperation", [])
    if not turns:
        return []

    turn_to_coop = dict(zip(turns, values))
    results = []

    for shock in shock_timeline:
        shock_start = shock.get("turn_started", 0)
        shock_duration = shock.get("duration", 0)
        shock_end = shock_start + shock_duration

        # Pre-shock baseline: avg cooperation over 10 turns before shock
        pre_values = [turn_to_coop[t] for t in turns if shock_start - 10 <= t < shock_start]
        pre_baseline = sum(pre_values) / len(pre_values) if pre_values else 0.5

        # Find recovery: first turn after shock_end where cooperation >= pre_baseline
        recovery_turns = None
        for t in turns:
            if t >= shock_end:
                if turn_to_coop.get(t, 0) >= pre_baseline:
                    recovery_turns = t - shock_end
                    break

        # Avg shock_resilience across NPCs at time of shock
        resilience_values = []
        for npc in npc_registry.values():
            for sample in npc.adaptation_trace:
                if abs(sample["turn"] - shock_start) <= 2:
                    resilience_values.append(sample.get("shock_resilience", 0.5))
                    break

        avg_resilience = round(sum(resilience_values) / len(resilience_values), 4) if resilience_values else 0.5

        results.append({
            "shock_type": shock.get("shock_type", ""),
            "turn_started": shock_start,
            "duration": shock_duration,
            "pre_shock_cooperation": round(pre_baseline, 4),
            "recovery_turns": recovery_turns,
            "avg_resilience_at_shock": avg_resilience,
        })

    return results


# ── Experiment Bundle ─────────────────────────────────────────────────────────


def build_experiment_bundle(
    engine: Any,
    event_log_entries: list[dict],
) -> dict[str, Any]:
    """Build a complete experiment data bundle for export.

    Includes all time-series, indices, distributions, and metadata
    needed for offline research analysis.

    Args:
        engine: GameEngine instance.
        event_log_entries: Full event log.

    Returns:
        JSON-serializable experiment bundle dict.
    """
    from backend.config import GAME_VERSION, SHOCK_CATALOG

    npc_registry = engine.npc_registry
    shock_timeline = engine.shock_manager.get_shock_timeline()

    bundle: dict[str, Any] = {
        # Metadata
        "metadata": {
            "game_version": GAME_VERSION,
            "seed": engine.seed,
            "max_turns": engine.max_turns,
            "current_turn": engine.turn,
            "difficulty": engine.difficulty.preset,
            "game_over": engine.game_over,
            "game_result": engine.game_result,
            "llm_enabled": LLM_ENABLED,
            "llm_provider": LLM_PROVIDER if LLM_ENABLED else "disabled",
            "shock_enabled": SHOCK_ENABLED,
            "shock_catalog_types": list(SHOCK_CATALOG.keys()),
            "npc_count": len(npc_registry),
            "npc_roles": {uid: npc.archetype for uid, npc in npc_registry.items()},
        },

        # Time-series
        "reward_series": compute_reward_series(npc_registry),
        "community_reward_series": compute_community_reward_series(npc_registry),
        "social_welfare_series": compute_social_welfare_series(npc_registry),
        "cooperation_series": compute_cooperation_series(npc_registry),

        # Point-in-time indices
        "cooperation_index": compute_cooperation_index(npc_registry),
        "policy_entropy": compute_policy_entropy(npc_registry),

        # Distributions
        "action_distribution": compute_action_distribution(
            event_log_entries, npc_registry
        ),

        # Shock analysis
        "shock_timeline": shock_timeline,
        "shock_responses": compute_shock_response(npc_registry, shock_timeline),
        "shock_recovery": compute_shock_recovery_time(npc_registry, shock_timeline),

        # Narrative & quest metrics
        "narrative_coherence": compute_narrative_coherence(event_log_entries),
        "deviation_recovery": compute_deviation_recovery(event_log_entries),

        # Adaptation state (current snapshot)
        "adaptation_snapshot": {
            uid: dict(npc.adaptation_state)
            for uid, npc in npc_registry.items()
        },
    }

    return bundle
