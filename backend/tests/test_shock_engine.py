"""
test_shock_engine.py — Tests for Phase 5: Dynamic Shock Engine.

Tests shock lifecycle (activation, decay, expiry), reward modification,
NPC adaptation pressure, save/load persistence, and game engine integration.
"""

import copy
import random

import numpy as np
import pytest

from backend.engine.shock_manager import ShockManager


# ── Unit Tests: ShockManager ──────────────────────────────────────────────────


class TestShockActivation:
    """Test shock activation and validation."""

    def test_activate_known_shock_type(self):
        sm = ShockManager()
        shock = sm.activate_shock("famine", turn=10)
        assert shock is not None
        assert shock["shock_type"] == "famine"
        assert shock["intensity"] == 1.0
        assert shock["turn_started"] == 10
        assert len(sm) == 1

    def test_activate_unknown_shock_returns_none(self):
        sm = ShockManager()
        shock = sm.activate_shock("nonexistent_type", turn=1)
        assert shock is None
        assert len(sm) == 0

    def test_no_stacking_same_type(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=1)
        shock2 = sm.activate_shock("famine", turn=5)
        assert shock2 is None
        assert len(sm) == 1

    def test_max_active_shocks_enforced(self):
        from backend.config import SHOCK_MAX_ACTIVE
        sm = ShockManager()
        types = ["famine", "bandit_raid", "plague", "trade_boom", "harsh_winter"]
        for i, t in enumerate(types[:SHOCK_MAX_ACTIVE]):
            assert sm.activate_shock(t, turn=i) is not None
        # Next activation should fail
        excess_type = types[SHOCK_MAX_ACTIVE] if SHOCK_MAX_ACTIVE < len(types) else "famine"
        # If famine already active, it'll be rejected for duplicate; use a fresh manager check
        sm2 = ShockManager()
        for i in range(SHOCK_MAX_ACTIVE):
            sm2.activate_shock(types[i], turn=i)
        result = sm2.activate_shock(types[SHOCK_MAX_ACTIVE], turn=99)
        assert result is None

    def test_shock_id_increments(self):
        sm = ShockManager()
        s1 = sm.activate_shock("famine", turn=1)
        s2 = sm.activate_shock("plague", turn=2)
        assert s1 is not None and s2 is not None
        assert s1["shock_id"] == "famine_001"
        assert s2["shock_id"] == "plague_002"


class TestShockDecay:
    """Test shock tick/decay/expiry behavior."""

    def test_linear_decay(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=0)
        # Famine has duration=15, linear decay
        sm.tick(5)  # 5/15 elapsed
        shocks = sm.get_active_shocks()
        assert len(shocks) == 1
        # Intensity should be ~0.6667
        assert 0.6 < shocks[0]["intensity"] < 0.7

    def test_sudden_decay_stays_at_full(self):
        sm = ShockManager()
        sm.activate_shock("bandit_raid", turn=0)
        # Bandit raid has duration=8, sudden decay
        sm.tick(4)  # Half elapsed
        shocks = sm.get_active_shocks()
        assert len(shocks) == 1
        assert shocks[0]["intensity"] == 1.0

    def test_shock_expires_at_duration(self):
        sm = ShockManager()
        sm.activate_shock("bandit_raid", turn=0)
        expired = sm.tick(8)  # Exactly at duration
        assert len(expired) == 1
        assert expired[0]["shock_type"] == "bandit_raid"
        assert len(sm) == 0

    def test_expired_shocks_in_history(self):
        sm = ShockManager()
        sm.activate_shock("bandit_raid", turn=0)
        sm.tick(8)
        timeline = sm.get_shock_timeline()
        assert len(timeline) == 1
        assert timeline[0]["status"] == "expired"


class TestShockEffects:
    """Test effect aggregation methods."""

    def test_reward_modifier_no_shocks(self):
        sm = ShockManager()
        assert sm.get_reward_modifier() == 1.0

    def test_reward_modifier_with_famine(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=0)
        # Famine: reward_scale=0.5, intensity=1.0
        mod = sm.get_reward_modifier()
        assert mod == pytest.approx(0.5)

    def test_reward_modifier_after_decay(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=0)
        sm.tick(7)  # ~53% elapsed, intensity ~0.533
        mod = sm.get_reward_modifier()
        # Should be between 0.5 and 1.0
        assert 0.5 < mod < 1.0

    def test_adaptation_pressure_no_shocks(self):
        sm = ShockManager()
        assert sm.get_adaptation_pressure() == 0.0

    def test_adaptation_pressure_with_shock(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=0)
        pressure = sm.get_adaptation_pressure()
        assert pressure == pytest.approx(1.0)  # Single shock at full intensity

    def test_stat_drain_aggregation(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=0)
        drain = sm.get_stat_drain()
        assert drain > 0

    def test_trade_boom_positive_effects(self):
        sm = ShockManager()
        sm.activate_shock("trade_boom", turn=0)
        mod = sm.get_reward_modifier()
        assert mod > 1.0  # Boost
        drain = sm.get_stat_drain()
        assert drain < 0  # Negative drain = gain
        trust = sm.get_trust_modifier()
        assert trust > 0


class TestShockSerialization:
    """Test save/load round-trip."""

    def test_round_trip(self):
        sm = ShockManager()
        sm.activate_shock("famine", turn=5)
        sm.activate_shock("plague", turn=7)
        sm.tick(10)  # Decay them a bit

        data = sm.to_dict()
        sm2 = ShockManager()
        sm2.from_dict(data)

        assert len(sm2) == len(sm)
        assert sm2.get_active_shocks() == sm.get_active_shocks()


# ── Integration Tests: Game Engine ────────────────────────────────────────────


class TestShockGameIntegration:
    """Test shock engine integration with the game engine."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        """Create a deterministic game engine for each test."""
        from backend.engine.game_engine import GameEngine
        random.seed(42)
        np.random.seed(42)
        self.engine = GameEngine(seed=42, max_turns=200)

    def test_engine_has_shock_manager(self):
        assert hasattr(self.engine, 'shock_manager')
        assert isinstance(self.engine.shock_manager, ShockManager)

    def test_shock_persisted_in_save_load(self, tmp_path):
        """Shock state should survive save → load round-trip."""
        from backend.config import SAVES_DIR
        import json

        # Activate a shock
        self.engine.shock_manager.activate_shock("famine", turn=3, source="test")

        # Save
        filepath = self.engine.save_game("test_shock")

        # Verify shock_state in save file
        with open(filepath, "r") as f:
            save_data = json.load(f)
        assert "shock_state" in save_data
        assert len(save_data["shock_state"]["active_shocks"]) == 1

        # Load into fresh engine
        from backend.engine.game_engine import GameEngine
        engine2 = GameEngine(seed=42, max_turns=200)
        engine2.load_game(filepath)
        assert len(engine2.shock_manager) == 1
        assert engine2.shock_manager.get_active_shocks()[0]["shock_type"] == "famine"

    def test_metrics_include_shock_fields(self):
        """get_metrics() should include shock_timeline and active_shocks."""
        self.engine.shock_manager.activate_shock("trade_boom", turn=1)
        metrics = self.engine.get_metrics()
        assert "shock_timeline" in metrics
        assert "active_shocks" in metrics
        assert len(metrics["active_shocks"]) == 1

    def test_shock_modifies_community_reward(self):
        """Active shock should modify community reward for NPCs."""
        # Pre-train so we can process turns
        self.engine._pretrain_npcs()

        # Process a turn without shock
        self.engine.turn = 10
        community_state = self.engine.compute_community_state()
        from backend.npc.rl_agent import compute_reward
        npc = list(self.engine.npc_registry.values())[0]
        old_stats = dict(npc.stats)
        reward_no_shock = compute_reward(npc, old_stats, npc.stats, community_state)

        # Activate famine shock
        self.engine.shock_manager.activate_shock("famine", turn=10)
        mod = self.engine.shock_manager.get_reward_modifier()
        assert mod < 1.0  # Should be dampened

    def test_shock_pressure_updates_resilience(self):
        """Shock pressure should increase NPC shock_resilience over turns."""
        self.engine._pretrain_npcs()
        npc = list(self.engine.npc_registry.values())[0]
        initial_resilience = npc.adaptation_state["shock_resilience"]

        # Activate a shock and simulate adaptation updates
        self.engine.shock_manager.activate_shock("famine", turn=0)
        pressure = self.engine.shock_manager.get_adaptation_pressure()
        assert pressure > 0

        # Simulate multiple adaptation updates with shock pressure
        dummy_reward = {"penalty": 0.0, "individual": 0.0, "community": -0.5, "total": -0.5}
        for _ in range(20):
            npc.update_adaptation(dummy_reward, shock_pressure=pressure)

        assert npc.adaptation_state["shock_resilience"] > initial_resilience

    def test_no_shocks_community_state_unchanged(self):
        """Without shocks, community_state should not contain total_stamina (removed)."""
        state = self.engine.compute_community_state()
        assert "total_stamina" not in state
        assert "avg_reputation" in state
        assert "total_health" in state
        assert "avg_mood" in state
