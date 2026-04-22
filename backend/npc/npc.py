"""
npc.py — NPC base class for the MVP game.

Encapsulates all per-NPC state: identity, stats, combat, Q-table,
conversation history, knowledge, and serialization.
"""

from __future__ import annotations

import numpy as np

from backend.config import (
    LOCATION_IDS,
    MAX_CONVERSATION_HISTORY,
    NPC_ACTION_SPACE_SIZE,
    NPC_NUM_ENERGY_LEVELS,
    NPC_NUM_MOOD_LEVELS,
    NPC_NUM_TIME_SLOTS,
    NPC_STATE_SPACE_SIZE,
    REPUTATION_MAX,
    REPUTATION_MIN,
    TIME_PERIODS,
    UNIVERSAL_ACTION_IDS,
    logger,
)


class NPC:
    """Core NPC entity combining identity, stats, RL state, and dialogue data."""

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self, instance_data: dict, archetype_data: dict) -> None:
        """Initialise an NPC from instance JSON and archetype JSON data.

        Args:
            instance_data: Per-NPC instance fields (uid, stats, q_table, …).
            archetype_data: Shared archetype fields (schedule, dialogue, …).
        """
        # --- Identity ---
        self.npc_uid: str = instance_data["npc_uid"]
        self.name: str = instance_data["name"]
        self.archetype: str = instance_data["archetype"]
        self.location: str = instance_data["location"]
        self.personality: str = instance_data["personality"]
        self.quest_critical: bool = instance_data.get("quest_critical", False)

        # --- Stats (happiness / income / health / reputation, each 0-10) ---
        self.stats: dict[str, int | float] = dict(instance_data["stats"])

        # --- Reward weights (must sum to 1.0) ---
        self.reward_weights: dict[str, float] = dict(instance_data["reward_weights"])
        self._validate_reward_weights()

        # --- Combat ---
        cs = instance_data["combat_stats"]
        self.combat_stats: dict[str, int] = dict(cs)
        self.max_hp: int = cs["max_hp"]
        self.current_hp: int = instance_data.get("current_hp", self.max_hp)

        # --- Status ---
        self.status: str = instance_data.get("status", "active")
        self.epsilon: float = instance_data.get("epsilon", 0.15)
        self.is_defending: bool = False
        self.incapacitation_turn: int | None = None

        # --- Relationships ---
        self.npc_relationships: dict[str, int] = dict(
            instance_data.get("npc_relationships", {})
        )

        # --- Q-table (numpy 2D: states × actions) ---
        self.q_table: np.ndarray = self._init_q_table(
            instance_data.get("q_table", {})
        )

        # --- Dialogue / history ---
        self.conversation_history: list[dict] = list(
            instance_data.get("conversation_history", [])
        )
        self.known_events: list[dict] = list(
            instance_data.get("known_events", [])
        )

        # --- Archetype data ---
        self.movement_weights: dict[str, float] = dict(
            archetype_data.get("movement_weights", {})
        )
        self.fallback_schedule: list[dict] = list(
            archetype_data.get("fallback_schedule", [])
        )
        self.scripted_dialogue: dict[str, str] = dict(
            archetype_data.get("scripted_dialogue", {})
        )
        self.generic_responses: dict[str, list[str]] = {
            k: list(v)
            for k, v in archetype_data.get("generic_responses", {}).items()
        }

        # --- STEP 4: Role telemetry (for role-masked action analysis) ---
        self.role_telemetry: dict[str, int] = {
            "actions_selected": 0,
            "role_aligned": 0,
            "role_misaligned": 0,
        }
        self.role_telemetry_trace: list[dict] = []  # Per-turn snapshots
        self.max_role_telemetry_trace_len: int = 100

        # --- Reward tracing (for metrics/analysis) ---
        self.reward_trace: list[dict] = []  # List of {turn, penalty, individual, community, total}
        self.max_reward_trace_len: int = 100  # Keep only last 100 turns
        self.lambda_coeff: float = 0.0  # Community reward coefficient (0.0 = disabled by default)

        # --- Adaptation state (STEP 3: Adaptive Personality Dynamics) ---
        self.adaptation_state: dict[str, float] = {
            "cooperation_tendency": 0.5,      # 0.0–1.0: tendency to cooperate
            "risk_aversion": 0.5,             # 0.0–1.0: tendency to avoid risky actions
            "social_sensitivity": 0.5,        # 0.0–1.0: sensitivity to social feedback
            "shock_resilience": 0.5,          # 0.0–1.0: ability to adapt to shocks
        }
        self.adaptation_trace: list[dict] = []  # Track adaptation over time
        self.max_adaptation_trace_len: int = 100

    # ── Private helpers ───────────────────────────────────────────────────

    def _validate_reward_weights(self) -> None:
        """Log a warning if reward weights do not sum to 1.0."""
        total = sum(self.reward_weights.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(
                "Reward weights for %s sum to %.4f (expected 1.0)",
                self.npc_uid,
                total,
            )

    def _init_q_table(self, raw: dict) -> np.ndarray:
        """Build a 2-D numpy Q-table from an optional sparse dict.

        The sparse dict maps ``"state_action"`` strings (e.g. ``"12_7"``)
        to float Q-values.  Any missing entries default to 0.0.
        """
        q = np.zeros((NPC_STATE_SPACE_SIZE, NPC_ACTION_SPACE_SIZE), dtype=np.float64)
        for key, value in raw.items():
            parts = key.split("_")
            if len(parts) == 2:
                try:
                    s, a = int(parts[0]), int(parts[1])
                    if 0 <= s < NPC_STATE_SPACE_SIZE and 0 <= a < NPC_ACTION_SPACE_SIZE:
                        q[s, a] = float(value)
                except (ValueError, IndexError):
                    logger.warning(
                        "Skipping invalid Q-table key '%s' for %s",
                        key,
                        self.npc_uid,
                    )
        return q

    # ── State discretization ──────────────────────────────────────────────

    def discretize_state(self, time_period: str) -> int:
        """Encode the NPC's current state as a single integer index.

        State tuple: ``(location_idx, time_idx, energy_level, mood_level)``

        * **energy_level**: 0 if HP < 33 %, 1 if 33–66 %, 2 if > 66 %
        * **mood_level**: based on ``stats["happiness"]``:
          0 if < 4, 1 if 4–7, 2 if > 7

        Returns:
            Integer in ``[0, NPC_STATE_SPACE_SIZE)``.
        """
        location_idx = LOCATION_IDS.index(self.location) if self.location in LOCATION_IDS else 0
        time_idx = TIME_PERIODS.index(time_period) if time_period in TIME_PERIODS else 0

        # Energy level from HP ratio
        hp_ratio = self.current_hp / max(self.max_hp, 1)
        if hp_ratio < 0.33:
            energy_level = 0
        elif hp_ratio <= 0.66:
            energy_level = 1
        else:
            energy_level = 2

        # Mood level from happiness stat
        happiness = self.stats.get("happiness", 5)
        if happiness < 4:
            mood_level = 0
        elif happiness <= 7:
            mood_level = 1
        else:
            mood_level = 2

        # Pack into a flat index
        index = (
            location_idx * NPC_NUM_TIME_SLOTS * NPC_NUM_ENERGY_LEVELS * NPC_NUM_MOOD_LEVELS
            + time_idx * NPC_NUM_ENERGY_LEVELS * NPC_NUM_MOOD_LEVELS
            + energy_level * NPC_NUM_MOOD_LEVELS
            + mood_level
        )
        return int(index)

    # ── Combat helpers ────────────────────────────────────────────────────

    def get_combat_dict(self) -> dict:
        """Return a dict suitable for the combat resolution system."""
        return {
            "uid": self.npc_uid,
            "name": self.name,
            "base_attack": self.combat_stats["base_attack"],
            "base_defense": self.combat_stats["base_defense"],
            "weapon_modifier": self.combat_stats.get("weapon_modifier", 0),
            "armor_modifier": self.combat_stats.get("armor_modifier", 0),
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "is_defending": self.is_defending,
        }

    def modify_hp(self, amount: int) -> int:
        """Change current HP by *amount* (positive = heal, negative = damage).

        HP is clamped to ``[0, max_hp]``.  If the NPC is quest-critical,
        HP cannot drop below 1.

        Returns:
            The new ``current_hp`` value.
        """
        self.current_hp = max(0, min(self.max_hp, self.current_hp + amount))
        if self.quest_critical and self.current_hp <= 0:
            self.current_hp = 1
        return self.current_hp

    def is_incapacitated(self) -> bool:
        """Return ``True`` if the NPC is currently incapacitated."""
        return self.status == "incapacitated"

    def incapacitate(self, turn: int) -> None:
        """Mark the NPC as incapacitated starting on *turn*.

        Quest-critical NPCs can never actually reach 0 HP (floored at 1)
        but can still be incapacitated for narrative purposes, except
        we keep them at 1 HP.
        """
        if self.quest_critical:
            self.current_hp = max(1, self.current_hp)
        else:
            self.current_hp = 0
        self.status = "incapacitated"
        self.incapacitation_turn = turn
        logger.info(
            "NPC %s incapacitated on turn %d", self.npc_uid, turn
        )

    def check_recovery(self, current_turn: int) -> bool:
        """Return ``True`` (and recover) if 20+ turns have elapsed since incapacitation."""
        if self.status != "incapacitated" or self.incapacitation_turn is None:
            return False
        from backend.config import INCAPACITATION_TURNS

        if current_turn - self.incapacitation_turn >= INCAPACITATION_TURNS:
            self.status = "active"
            self.current_hp = max(1, self.max_hp // 4)  # recover to 25 %
            self.incapacitation_turn = None
            logger.info(
                "NPC %s recovered on turn %d", self.npc_uid, current_turn
            )
            return True
        return False

    # ── Conversation history ──────────────────────────────────────────────

    def add_conversation(self, entry: dict) -> None:
        """Append a conversation entry, capping at MAX_CONVERSATION_HISTORY."""
        self.conversation_history.append(entry)
        if len(self.conversation_history) > MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY:]

    # ── Reward tracing ───────────────────────────────────────────────────

    def add_reward_sample(self, turn: int, reward_dict: dict) -> None:
        """Store reward components for this turn.

        Keeps only the last ``max_reward_trace_len`` samples to prevent
        unbounded memory growth.

        Args:
            turn: Current game turn.
            reward_dict: Dict with keys: penalty, individual, community, total.
        """
        self.reward_trace.append({
            "turn": turn,
            "penalty": reward_dict.get("penalty", 0.0),
            "individual": reward_dict.get("individual", 0.0),
            "community": reward_dict.get("community", 0.0),
            "total": reward_dict.get("total", 0.0),
        })
        if len(self.reward_trace) > self.max_reward_trace_len:
            self.reward_trace = self.reward_trace[-self.max_reward_trace_len:]

    # ── Adaptation state (STEP 3) ──────────────────────────────────────────

    def update_adaptation(self, reward_dict: dict, shock_pressure: float = 0.0) -> None:
        """Update adaptation coefficients based on reward components.

        Adaptation rules:
        - High community reward → increase cooperation_tendency
        - Low individual reward → increase risk_aversion
        - Active shocks → adjust shock_resilience
        - Social feedback → adjust social_sensitivity

        Args:
            reward_dict: Dict with keys: penalty, individual, community, total.
            shock_pressure: Aggregate shock intensity in [0.0, 1.0]. 0.0 = no shocks.
        """
        # Extract reward components
        individual = reward_dict.get("individual", 0.0)
        community = reward_dict.get("community", 0.0)
        penalty = reward_dict.get("penalty", 0.0)

        # Adaptation update rates (tuned for slow, stable drift)
        adapt_rate = 0.02

        # Cooperation: increase when community reward is positive
        if community > 0.5:
            self.adaptation_state["cooperation_tendency"] = min(
                1.0,
                self.adaptation_state["cooperation_tendency"] + adapt_rate * (community / 1.5)
            )
        elif community < -0.5:
            self.adaptation_state["cooperation_tendency"] = max(
                0.0,
                self.adaptation_state["cooperation_tendency"] - adapt_rate * 0.5
            )

        # Risk aversion: increase when individual reward is low or penalties are high
        if individual < -0.5 or penalty < -2.0:
            self.adaptation_state["risk_aversion"] = min(
                1.0,
                self.adaptation_state["risk_aversion"] + adapt_rate * (abs(individual) / 2.0)
            )
        elif individual > 1.5:
            self.adaptation_state["risk_aversion"] = max(
                0.0,
                self.adaptation_state["risk_aversion"] - adapt_rate * 0.5
            )

        # Shock resilience: increase under sustained shock pressure
        if shock_pressure > 0.1:
            self.adaptation_state["shock_resilience"] = min(
                1.0,
                self.adaptation_state["shock_resilience"] + adapt_rate * shock_pressure
            )
        elif shock_pressure == 0.0 and self.adaptation_state["shock_resilience"] > 0.5:
            # Slowly decay back toward baseline when no shocks
            self.adaptation_state["shock_resilience"] = max(
                0.5,
                self.adaptation_state["shock_resilience"] - adapt_rate * 0.3
            )

        # Social sensitivity: drift toward middle (neutral sentiment by default)
        drift_toward_neutral = 0.01
        for key in ["cooperation_tendency", "risk_aversion", "social_sensitivity"]:
            self.adaptation_state[key] = (
                self.adaptation_state[key] * (1.0 - drift_toward_neutral) +
                0.5 * drift_toward_neutral
            )

        # Clamp all values to [0.0, 1.0]
        for key in self.adaptation_state:
            self.adaptation_state[key] = max(0.0, min(1.0, self.adaptation_state[key]))

    def add_adaptation_sample(self, turn: int) -> None:
        """Store adaptation state snapshot for this turn.

        Keeps only the last ``max_adaptation_trace_len`` samples.

        Args:
            turn: Current game turn.
        """
        self.adaptation_trace.append({
            "turn": turn,
            "cooperation_tendency": self.adaptation_state["cooperation_tendency"],
            "risk_aversion": self.adaptation_state["risk_aversion"],
            "social_sensitivity": self.adaptation_state["social_sensitivity"],
            "shock_resilience": self.adaptation_state["shock_resilience"],
        })
        if len(self.adaptation_trace) > self.max_adaptation_trace_len:
            self.adaptation_trace = self.adaptation_trace[-self.max_adaptation_trace_len:]

    def update_role_telemetry(self, action_id: str) -> None:
        """Track role-alignment of selected action.

        Args:
            action_id: The action ID that was selected.
        """
        from backend.config import ROLE_ACTION_MASKS

        self.role_telemetry["actions_selected"] += 1

        role_actions = ROLE_ACTION_MASKS.get(self.archetype, [])
        if action_id in role_actions:
            self.role_telemetry["role_aligned"] += 1
        else:
            self.role_telemetry["role_misaligned"] += 1

    def add_role_telemetry_sample(self, turn: int) -> None:
        """Store role telemetry snapshot for this turn.

        Args:
            turn: Current game turn.
        """
        total = self.role_telemetry["actions_selected"]
        aligned = self.role_telemetry["role_aligned"]
        coherence = (aligned / total) if total > 0 else 0.0

        self.role_telemetry_trace.append({
            "turn": turn,
            "actions_selected": self.role_telemetry["actions_selected"],
            "role_aligned": self.role_telemetry["role_aligned"],
            "role_misaligned": self.role_telemetry["role_misaligned"],
            "role_coherence": round(coherence, 4),
        })
        if len(self.role_telemetry_trace) > self.max_role_telemetry_trace_len:
            self.role_telemetry_trace = self.role_telemetry_trace[-self.max_role_telemetry_trace_len:]

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the NPC to a JSON-compatible dict.

        The Q-table is stored as a sparse dict of ``"state_action": value``
        for non-zero entries to keep file sizes manageable.
        """
        # Sparse Q-table
        sparse_q: dict[str, float] = {}
        non_zero = np.nonzero(self.q_table)
        for s, a in zip(non_zero[0], non_zero[1]):
            sparse_q[f"{int(s)}_{int(a)}"] = float(self.q_table[s, a])

        return {
            "npc_uid": self.npc_uid,
            "name": self.name,
            "archetype": self.archetype,
            "location": self.location,
            "personality": self.personality,
            "stats": dict(self.stats),
            "reward_weights": dict(self.reward_weights),
            "combat_stats": dict(self.combat_stats),
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "status": self.status,
            "epsilon": self.epsilon,
            "is_defending": self.is_defending,
            "incapacitation_turn": self.incapacitation_turn,
            "npc_relationships": dict(self.npc_relationships),
            "q_table": sparse_q,
            "conversation_history": list(self.conversation_history),
            "known_events": list(self.known_events),
            "quest_critical": self.quest_critical,
            "adaptation_state": dict(self.adaptation_state),
            "lambda_coeff": self.lambda_coeff,
        }

    @classmethod
    def from_dict(cls, data: dict, archetype_data: dict) -> NPC:
        """Reconstruct an NPC instance from a serialized dict.

        Args:
            data: Output of ``to_dict()`` or raw instance JSON.
            archetype_data: The archetype JSON for this NPC.

        Returns:
            A fully-initialized :class:`NPC`.
        """
        npc = cls(data, archetype_data)
        # Restore transient fields that are in saved data but not in
        # the raw instance JSON.
        npc.is_defending = data.get("is_defending", False)
        npc.incapacitation_turn = data.get("incapacitation_turn", None)
        if "max_hp" in data:
            npc.max_hp = data["max_hp"]
        # Restore adaptation state (backward compatible with old saves)
        if "adaptation_state" in data:
            npc.adaptation_state = dict(data["adaptation_state"])
        # Restore lambda coefficient (backward compatible)
        if "lambda_coeff" in data:
            npc.lambda_coeff = float(data["lambda_coeff"])
        return npc

    # ── Dunder ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NPC(uid={self.npc_uid!r}, name={self.name!r}, "
            f"loc={self.location!r}, hp={self.current_hp}/{self.max_hp}, "
            f"status={self.status!r})"
        )
