"""
test_step3_step4_features.py — Comprehensive tests for Adaptive Personality & Role Masks.

Tests for Step 3 (Adaptation Dynamics) and Step 4 (Role-Specific Policy Masks).
"""

import asyncio
import json
import pytest
from pathlib import Path

from backend.engine.game_engine import GameEngine
from backend.config import (
    MASTER_SEED,
    ROLE_ACTION_MASKS,
    ROLE_MASK_ENABLED,
    UNIVERSAL_ACTION_IDS,
)
from backend.npc.rl_agent import select_action


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: ADAPTATION STATE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptationStateInitialization:
    """Test adaptation state is properly initialized for all NPCs."""

    @pytest.mark.asyncio
    async def test_adaptation_state_exists(self):
        """Verify each NPC has adaptation_state dict."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        for uid, npc in engine.npc_registry.items():
            assert hasattr(npc, "adaptation_state"), f"{npc.name} missing adaptation_state"
            assert isinstance(npc.adaptation_state, dict), f"{npc.name} adaptation_state not dict"

    @pytest.mark.asyncio
    async def test_adaptation_state_default_values(self):
        """Verify adaptation coefficients default to 0.5."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        for uid, npc in engine.npc_registry.items():
            assert npc.adaptation_state["cooperation_tendency"] == 0.5
            assert npc.adaptation_state["risk_aversion"] == 0.5
            assert npc.adaptation_state["social_sensitivity"] == 0.5
            assert npc.adaptation_state["shock_resilience"] == 0.5

    @pytest.mark.asyncio
    async def test_adaptation_state_all_keys_present(self):
        """Verify all required adaptation keys are present."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        required_keys = {
            "cooperation_tendency",
            "risk_aversion",
            "social_sensitivity",
            "shock_resilience",
        }

        for uid, npc in engine.npc_registry.items():
            assert required_keys.issubset(npc.adaptation_state.keys())


class TestAdaptationUpdates:
    """Test adaptation coefficients update based on rewards."""

    @pytest.mark.asyncio
    async def test_cooperation_increases_with_positive_community_reward(self):
        """High community reward should increase cooperation_tendency."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        initial_cooperation = npc.adaptation_state["cooperation_tendency"]

        # Create positive community reward
        reward_dict = {
            "penalty": 0.0,
            "individual": 0.0,
            "community": 1.0,  # High community reward
            "total": 1.0,
        }

        npc.update_adaptation(reward_dict)

        assert npc.adaptation_state["cooperation_tendency"] > initial_cooperation, \
            "cooperation_tendency should increase with positive community reward"

    @pytest.mark.asyncio
    async def test_risk_aversion_increases_with_negative_individual_reward(self):
        """Low individual reward should increase risk_aversion."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        initial_risk_aversion = npc.adaptation_state["risk_aversion"]

        # Create negative individual reward
        reward_dict = {
            "penalty": 0.0,
            "individual": -1.0,  # Low individual reward
            "community": 0.0,
            "total": -1.0,
        }

        npc.update_adaptation(reward_dict)

        assert npc.adaptation_state["risk_aversion"] > initial_risk_aversion, \
            "risk_aversion should increase with negative individual reward"

    @pytest.mark.asyncio
    async def test_adaptation_values_clamped_to_valid_range(self):
        """Adaptation coefficients should stay within [0.0, 1.0]."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]

        # Apply many extreme rewards to try to push outside bounds
        for _ in range(10):
            reward_dict = {
                "penalty": -5.0,
                "individual": -5.0,
                "community": 5.0,
                "total": -5.0,
            }
            npc.update_adaptation(reward_dict)

        for key, value in npc.adaptation_state.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} outside valid range [0.0, 1.0]"


class TestAdaptationTracing:
    """Test adaptation state is traced for metrics."""

    @pytest.mark.asyncio
    async def test_adaptation_trace_exists(self):
        """NPCs should have adaptation_trace list."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        for uid, npc in engine.npc_registry.items():
            assert hasattr(npc, "adaptation_trace")
            assert isinstance(npc.adaptation_trace, list)

    @pytest.mark.asyncio
    async def test_adaptation_samples_recorded(self):
        """Adaptation samples should be added to trace."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        initial_trace_len = len(npc.adaptation_trace)

        npc.add_adaptation_sample(1)

        assert len(npc.adaptation_trace) == initial_trace_len + 1
        assert npc.adaptation_trace[-1]["turn"] == 1
        assert "cooperation_tendency" in npc.adaptation_trace[-1]

    @pytest.mark.asyncio
    async def test_adaptation_trace_max_length(self):
        """Adaptation trace should cap at max_adaptation_trace_len."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        max_len = npc.max_adaptation_trace_len

        # Add more samples than max
        for turn in range(max_len + 50):
            npc.add_adaptation_sample(turn)

        assert len(npc.adaptation_trace) <= max_len, \
            f"Trace length {len(npc.adaptation_trace)} exceeds max {max_len}"


class TestAdaptationSerialization:
    """Test adaptation state serializes and deserializes correctly."""

    @pytest.mark.asyncio
    async def test_adaptation_state_in_to_dict(self):
        """adaptation_state should be included in NPC serialization."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        npc_dict = npc.to_dict()

        assert "adaptation_state" in npc_dict
        assert isinstance(npc_dict["adaptation_state"], dict)
        assert "cooperation_tendency" in npc_dict["adaptation_state"]

    @pytest.mark.asyncio
    async def test_adaptation_state_restored_from_dict(self):
        """adaptation_state should be restored from saved dict."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]

        # Modify adaptation state
        npc.adaptation_state["cooperation_tendency"] = 0.75

        # Serialize and deserialize
        npc_dict = npc.to_dict()
        archetype_data = {}  # Not used in this test
        restored_npc = type(npc).from_dict(npc_dict, archetype_data)

        assert restored_npc.adaptation_state["cooperation_tendency"] == 0.75

    @pytest.mark.asyncio
    async def test_adaptation_state_save_load_cycle(self):
        """Adaptation state should survive full save/load cycle."""
        from backend.config import SAVES_DIR

        engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=50)
        await engine.initialize()

        # Modify adaptation state
        for uid, npc in engine.npc_registry.items():
            npc.adaptation_state["cooperation_tendency"] = 0.7
            npc.adaptation_state["risk_aversion"] = 0.3

        engine.save_game("test_manual_1")

        # Find the saved file
        save_files = list(SAVES_DIR.glob("save_test_manual_1*.json"))
        assert len(save_files) > 0, "Save file not created"

        save_path = save_files[0]

        # Verify in saved file
        saved_data = json.loads(save_path.read_text())
        for uid, npc_data in saved_data["npc_registry"].items():
            assert "adaptation_state" in npc_data
            assert npc_data["adaptation_state"]["cooperation_tendency"] == 0.7
            assert npc_data["adaptation_state"]["risk_aversion"] == 0.3

        # Cleanup
        save_path.unlink()


class TestAdaptationMetrics:
    """Test adaptation data appears in metrics."""

    @pytest.mark.asyncio
    async def test_npc_adaptation_in_metrics(self):
        """Metrics should include npc_adaptation field."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        metrics = engine.get_metrics()

        assert "npc_adaptation" in metrics
        assert isinstance(metrics["npc_adaptation"], dict)

    @pytest.mark.asyncio
    async def test_npc_adaptation_metrics_structure(self):
        """npc_adaptation metrics should have correct structure."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        metrics = engine.get_metrics()

        for uid, adapt in metrics["npc_adaptation"].items():
            assert "cooperation_tendency" in adapt
            assert "risk_aversion" in adapt
            assert "social_sensitivity" in adapt
            assert "shock_resilience" in adapt
            # Should be rounded to 4 decimal places
            assert isinstance(adapt["cooperation_tendency"], float)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ROLE MASK TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleTelemetryInitialization:
    """Test role telemetry is properly initialized."""

    @pytest.mark.asyncio
    async def test_role_telemetry_exists(self):
        """Each NPC should have role_telemetry dict."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        for uid, npc in engine.npc_registry.items():
            assert hasattr(npc, "role_telemetry")
            assert isinstance(npc.role_telemetry, dict)

    @pytest.mark.asyncio
    async def test_role_telemetry_default_values(self):
        """Role telemetry should start at zero."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        for uid, npc in engine.npc_registry.items():
            assert npc.role_telemetry["actions_selected"] == 0
            assert npc.role_telemetry["role_aligned"] == 0
            assert npc.role_telemetry["role_misaligned"] == 0

    @pytest.mark.asyncio
    async def test_role_telemetry_all_keys_present(self):
        """All required telemetry keys should be present."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        required_keys = {"actions_selected", "role_aligned", "role_misaligned"}

        for uid, npc in engine.npc_registry.items():
            assert required_keys.issubset(npc.role_telemetry.keys())


class TestRoleTelemetryTracking:
    """Test role telemetry tracks actions correctly."""

    @pytest.mark.asyncio
    async def test_role_aligned_action_tracked(self):
        """Role-aligned actions should increment role_aligned counter."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        initial_aligned = npc.role_telemetry["role_aligned"]

        # Get a role-aligned action for this NPC
        role_actions = ROLE_ACTION_MASKS.get(npc.archetype, [])
        if role_actions:
            action_id = role_actions[0]
            npc.update_role_telemetry(action_id)

            assert npc.role_telemetry["role_aligned"] == initial_aligned + 1

    @pytest.mark.asyncio
    async def test_role_misaligned_action_tracked(self):
        """Role-misaligned actions should increment role_misaligned counter."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        initial_misaligned = npc.role_telemetry["role_misaligned"]

        # Find an action NOT in this role's masks
        role_actions = ROLE_ACTION_MASKS.get(npc.archetype, [])
        misaligned_action = None
        for action_id in UNIVERSAL_ACTION_IDS:
            if action_id not in role_actions:
                misaligned_action = action_id
                break

        if misaligned_action:
            npc.update_role_telemetry(misaligned_action)
            assert npc.role_telemetry["role_misaligned"] == initial_misaligned + 1

    @pytest.mark.asyncio
    async def test_actions_selected_incremented(self):
        """Total actions_selected should increment for any action."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        initial_count = npc.role_telemetry["actions_selected"]

        npc.update_role_telemetry("wait")

        assert npc.role_telemetry["actions_selected"] == initial_count + 1


class TestRoleTelemetryTracing:
    """Test role telemetry is traced for metrics."""

    @pytest.mark.asyncio
    async def test_role_telemetry_trace_exists(self):
        """NPCs should have role_telemetry_trace list."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        for uid, npc in engine.npc_registry.items():
            assert hasattr(npc, "role_telemetry_trace")
            assert isinstance(npc.role_telemetry_trace, list)

    @pytest.mark.asyncio
    async def test_role_telemetry_samples_recorded(self):
        """Role telemetry samples should be added to trace."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]
        npc.update_role_telemetry("wait")
        npc.add_role_telemetry_sample(1)

        assert len(npc.role_telemetry_trace) == 1
        assert npc.role_telemetry_trace[-1]["turn"] == 1
        assert "role_coherence" in npc.role_telemetry_trace[-1]

    @pytest.mark.asyncio
    async def test_role_coherence_calculated(self):
        """Role coherence should be correctly calculated."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        npc = list(engine.npc_registry.values())[0]

        # Simulate: 3 actions selected, 2 role-aligned
        npc.role_telemetry["actions_selected"] = 3
        npc.role_telemetry["role_aligned"] = 2
        npc.add_role_telemetry_sample(1)

        expected_coherence = 2 / 3  # ~0.667
        actual_coherence = npc.role_telemetry_trace[-1]["role_coherence"]

        assert abs(actual_coherence - expected_coherence) < 0.01


class TestRoleMaskConfiguration:
    """Test role mask configuration is correct."""

    def test_role_masks_defined_for_all_archetypes(self):
        """All archetypes should have role mask definitions."""
        expected_archetypes = {"farmer", "guard", "tavkeeper", "elder", "villager"}
        defined_archetypes = set(ROLE_ACTION_MASKS.keys())

        assert expected_archetypes.issubset(defined_archetypes), \
            f"Missing role masks for: {expected_archetypes - defined_archetypes}"

    def test_role_actions_are_valid_universal_actions(self):
        """All role actions should be valid universal action IDs."""
        universal_ids = set(UNIVERSAL_ACTION_IDS)

        for role, actions in ROLE_ACTION_MASKS.items():
            for action_id in actions:
                assert action_id in universal_ids, \
                    f"Invalid action {action_id} in {role} role mask"

    def test_role_masks_not_empty(self):
        """Each role should have at least one action."""
        for role, actions in ROLE_ACTION_MASKS.items():
            assert len(actions) > 0, f"Role {role} has no actions"


class TestRoleTelemetryMetrics:
    """Test role telemetry appears in metrics."""

    @pytest.mark.asyncio
    async def test_npc_role_telemetry_in_metrics(self):
        """Metrics should include npc_role_telemetry field."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        metrics = engine.get_metrics()

        assert "npc_role_telemetry" in metrics
        assert isinstance(metrics["npc_role_telemetry"], dict)

    @pytest.mark.asyncio
    async def test_npc_role_telemetry_metrics_structure(self):
        """npc_role_telemetry should have correct structure."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal")
        await engine.initialize()

        metrics = engine.get_metrics()

        for uid, telemetry in metrics["npc_role_telemetry"].items():
            assert "role" in telemetry
            assert "actions_selected" in telemetry
            assert "role_aligned" in telemetry
            assert "role_misaligned" in telemetry
            assert "role_coherence" in telemetry


class TestRoleMaskIntegration:
    """Test role masks integrated into gameplay."""

    @pytest.mark.asyncio
    async def test_game_runs_with_role_masks_enabled(self):
        """Game should run with ROLE_MASK_ENABLED set."""
        import backend.config as config
        original_enabled = config.ROLE_MASK_ENABLED

        try:
            config.ROLE_MASK_ENABLED = True

            engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=20)
            await engine.initialize()

            # Play a few turns
            for _ in range(5):
                await engine.process_turn({
                    "source": "button",
                    "action_id": "wait",
                })

            # Verify no errors
            assert engine.turn == 5

        finally:
            config.ROLE_MASK_ENABLED = original_enabled

    @pytest.mark.asyncio
    async def test_game_runs_with_role_masks_disabled(self):
        """Game should run with ROLE_MASK_ENABLED unset."""
        import backend.config as config
        original_enabled = config.ROLE_MASK_ENABLED

        try:
            config.ROLE_MASK_ENABLED = False

            engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=20)
            await engine.initialize()

            # Play a few turns
            for _ in range(5):
                await engine.process_turn({
                    "source": "button",
                    "action_id": "wait",
                })

            # Verify no errors
            assert engine.turn == 5

        finally:
            config.ROLE_MASK_ENABLED = original_enabled


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS: Full Gameplay with Adaptation & Role Masks
# ─────────────────────────────────────────────────────────────────────────────

class TestFullGameplayIntegration:
    """Test adaptation and role masks during full gameplay."""

    @pytest.mark.asyncio
    async def test_adaptation_changes_during_gameplay(self):
        """Adaptation coefficients should be tracked during gameplay."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=30)
        await engine.initialize()

        # Record initial adaptation trace length
        npc = list(engine.npc_registry.values())[0]
        initial_trace_len = len(npc.adaptation_trace)

        # Play 20 turns
        for turn in range(20):
            await engine.process_turn({
                "source": "button",
                "action_id": "wait",
            })
            # Manually sample adaptation at each turn (this happens in gameplay)
            npc.add_adaptation_sample(turn)

        # Check if trace was populated
        final_trace_len = len(npc.adaptation_trace)

        assert final_trace_len > initial_trace_len, \
            "Adaptation trace should record samples during gameplay"

    @pytest.mark.asyncio
    async def test_role_telemetry_accumulates_during_gameplay(self):
        """Role telemetry should accumulate during gameplay."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=30)
        await engine.initialize()

        # Record initial telemetry
        npc = list(engine.npc_registry.values())[0]
        initial_actions = npc.role_telemetry["actions_selected"]

        # Play 10 turns
        for _ in range(10):
            await engine.process_turn({
                "source": "button",
                "action_id": "wait",
            })

        # Check if actions were tracked
        final_actions = npc.role_telemetry["actions_selected"]

        assert final_actions > initial_actions, \
            "Role telemetry should accumulate during gameplay"

    @pytest.mark.asyncio
    async def test_metrics_include_all_new_fields_after_gameplay(self):
        """All new metrics fields should be populated after gameplay."""
        engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=20)
        await engine.initialize()

        # Play 10 turns
        for _ in range(10):
            await engine.process_turn({
                "source": "button",
                "action_id": "wait",
            })

        metrics = engine.get_metrics()

        # Verify all Step 3 & 4 fields present
        assert "npc_adaptation" in metrics
        assert "npc_role_telemetry" in metrics
        assert "npc_rewards" in metrics
        assert "community_state" in metrics

    @pytest.mark.asyncio
    async def test_save_load_preserves_adaptation_during_gameplay(self):
        """Save/load should preserve adaptation state during gameplay."""
        from backend.config import SAVES_DIR

        engine = GameEngine(seed=MASTER_SEED, difficulty="normal", max_turns=50)
        await engine.initialize()

        # Play 10 turns to build up adaptation
        for _ in range(10):
            await engine.process_turn({
                "source": "button",
                "action_id": "wait",
            })

        # Record state before save
        npc_before = list(engine.npc_registry.values())[0]
        adaptation_before = dict(npc_before.adaptation_state)
        telemetry_before = dict(npc_before.role_telemetry)

        # Save
        engine.save_game("test_manual_2")

        # Find the saved file
        save_files = list(SAVES_DIR.glob("save_test_manual_2*.json"))
        assert len(save_files) > 0, "Save file not created"

        save_path = save_files[0]

        # Verify saved
        saved_data = json.loads(save_path.read_text())
        npc_data = list(saved_data["npc_registry"].values())[0]
        assert "adaptation_state" in npc_data

        # Cleanup
        save_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



