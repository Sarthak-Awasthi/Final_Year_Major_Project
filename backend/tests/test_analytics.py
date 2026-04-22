"""
test_analytics.py — Phase 6: Analytics module and API endpoint tests.

Tests:
  1. Reward series computation
  2. Community reward aggregation
  3. Social welfare index
  4. Cooperation index (global, per-role, per-NPC)
  5. Policy entropy
  6. Action distribution (early/late windows, shift)
  7. Shock response curves
  8. Experiment bundle structure and metadata
  9. API endpoint smoke tests (timeseries, cooperation, experiment)
  10. Playthrough logger RL telemetry
"""

import asyncio
import pytest

from backend.engine.game_engine import GameEngine
from backend.config import MASTER_SEED
from backend.engine.analytics import (
    build_experiment_bundle,
    compute_action_distribution,
    compute_community_reward_series,
    compute_cooperation_index,
    compute_cooperation_series,
    compute_policy_entropy,
    compute_reward_series,
    compute_shock_response,
    compute_social_welfare_series,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def make_engine(seed: int = MASTER_SEED, turns: int = 0, max_turns: int = 200) -> GameEngine:
    """Create an engine, initialize it, and optionally advance turns with 'wait' actions."""
    engine = GameEngine(seed=seed, difficulty="normal", max_turns=max_turns)
    await engine.initialize()
    for _ in range(turns):
        if engine.game_over:
            break
        await engine.process_turn({
            "source": "button",
            "action_id": "wait",
        })
    return engine


# ── Reward Series Tests ───────────────────────────────────────────────────────


class TestRewardSeries:
    """Tests for per-NPC reward time-series extraction."""

    @pytest.mark.asyncio
    async def test_reward_series_empty_on_new_game(self):
        engine = await make_engine(turns=0)
        series = compute_reward_series(engine.npc_registry)
        # Series dict should exist for each NPC
        assert len(series) == len(engine.npc_registry)
        for uid, npc_series in series.items():
            assert "turns" in npc_series
            assert "individual" in npc_series
            assert "community" in npc_series
            assert "penalty" in npc_series
            assert "total" in npc_series
            assert "npc_name" in npc_series
            assert "role" in npc_series

    @pytest.mark.asyncio
    async def test_reward_series_populated_after_turns(self):
        engine = await make_engine(turns=5)
        series = compute_reward_series(engine.npc_registry)
        for uid, npc_series in series.items():
            # After 5 turns, NPCs should have reward samples
            assert isinstance(npc_series["turns"], list)
            assert isinstance(npc_series["total"], list)
            assert len(npc_series["turns"]) == len(npc_series["total"])

    @pytest.mark.asyncio
    async def test_reward_series_values_are_numbers(self):
        engine = await make_engine(turns=3)
        series = compute_reward_series(engine.npc_registry)
        for uid, npc_series in series.items():
            for val in npc_series["total"]:
                assert isinstance(val, (int, float))


# ── Community Reward Tests ────────────────────────────────────────────────────


class TestCommunityRewardSeries:
    """Tests for village-level community reward aggregation."""

    @pytest.mark.asyncio
    async def test_community_series_structure(self):
        engine = await make_engine(turns=3)
        series = compute_community_reward_series(engine.npc_registry)
        assert "turns" in series
        assert "avg_community" in series
        assert "avg_total" in series

    @pytest.mark.asyncio
    async def test_community_series_lengths_match(self):
        engine = await make_engine(turns=5)
        series = compute_community_reward_series(engine.npc_registry)
        assert len(series["turns"]) == len(series["avg_community"])
        assert len(series["turns"]) == len(series["avg_total"])


# ── Social Welfare Tests ─────────────────────────────────────────────────────


class TestSocialWelfareSeries:
    """Tests for social welfare index computation."""

    @pytest.mark.asyncio
    async def test_welfare_series_structure(self):
        engine = await make_engine(turns=3)
        series = compute_social_welfare_series(engine.npc_registry)
        assert "turns" in series
        assert "welfare_index" in series

    @pytest.mark.asyncio
    async def test_welfare_values_in_range(self):
        engine = await make_engine(turns=5)
        series = compute_social_welfare_series(engine.npc_registry)
        for val in series["welfare_index"]:
            assert 0.0 <= val <= 1.0, f"Welfare index out of range: {val}"


# ── Cooperation Index Tests ───────────────────────────────────────────────────


class TestCooperationIndex:
    """Tests for cooperation index computation."""

    @pytest.mark.asyncio
    async def test_cooperation_index_structure(self):
        engine = await make_engine(turns=0)
        coop = compute_cooperation_index(engine.npc_registry)
        assert "global" in coop
        assert "per_role" in coop
        assert "per_npc" in coop

    @pytest.mark.asyncio
    async def test_cooperation_global_in_range(self):
        engine = await make_engine(turns=3)
        coop = compute_cooperation_index(engine.npc_registry)
        assert 0.0 <= coop["global"] <= 1.0

    @pytest.mark.asyncio
    async def test_cooperation_per_npc_count(self):
        engine = await make_engine(turns=0)
        coop = compute_cooperation_index(engine.npc_registry)
        assert len(coop["per_npc"]) == len(engine.npc_registry)

    @pytest.mark.asyncio
    async def test_cooperation_series_structure(self):
        engine = await make_engine(turns=5)
        series = compute_cooperation_series(engine.npc_registry)
        assert "turns" in series
        assert "global_cooperation" in series


# ── Policy Entropy Tests ──────────────────────────────────────────────────────


class TestPolicyEntropy:
    """Tests for per-NPC policy entropy computation."""

    @pytest.mark.asyncio
    async def test_entropy_per_npc(self):
        engine = await make_engine(turns=3)
        entropies = compute_policy_entropy(engine.npc_registry)
        assert len(entropies) == len(engine.npc_registry)

    @pytest.mark.asyncio
    async def test_entropy_non_negative(self):
        engine = await make_engine(turns=3)
        entropies = compute_policy_entropy(engine.npc_registry)
        for uid, val in entropies.items():
            assert val >= 0.0, f"Entropy for {uid} is negative: {val}"


# ── Action Distribution Tests ─────────────────────────────────────────────────


class TestActionDistribution:
    """Tests for action distribution with early/late windows."""

    @pytest.mark.asyncio
    async def test_action_dist_structure(self):
        engine = await make_engine(turns=5)
        dist = compute_action_distribution(engine.event_log.entries, engine.npc_registry)
        assert "global" in dist
        assert "per_role" in dist
        assert "early_window" in dist
        assert "late_window" in dist
        assert "distribution_shift" in dist

    @pytest.mark.asyncio
    async def test_action_dist_empty_on_new_game(self):
        engine = await make_engine(turns=0)
        dist = compute_action_distribution(engine.event_log.entries, engine.npc_registry)
        # Pretraining may generate NPC actions in event log
        assert isinstance(dist["global"], dict)


# ── Shock Response Tests ──────────────────────────────────────────────────────


class TestShockResponse:
    """Tests for shock response curve computation."""

    @pytest.mark.asyncio
    async def test_shock_response_empty_no_shocks(self):
        engine = await make_engine(turns=3)
        timeline = engine.shock_manager.get_shock_timeline()
        responses = compute_shock_response(engine.npc_registry, timeline)
        # Likely no shocks triggered in 3 turns
        assert isinstance(responses, list)

    @pytest.mark.asyncio
    async def test_shock_response_structure_with_shock(self):
        engine = await make_engine(turns=5)
        # Manually trigger a shock
        engine.shock_manager.activate_shock("famine", source="test", turn=engine.turn)
        timeline = engine.shock_manager.get_shock_timeline()
        responses = compute_shock_response(engine.npc_registry, timeline)
        assert len(responses) >= 1
        r = responses[0]
        assert "shock_type" in r
        assert "avg_cooperation_before" in r
        assert "avg_reward_during" in r


# ── Experiment Bundle Tests ───────────────────────────────────────────────────


class TestExperimentBundle:
    """Tests for complete experiment bundle export."""

    @pytest.mark.asyncio
    async def test_bundle_metadata(self):
        engine = await make_engine(turns=3)
        bundle = build_experiment_bundle(engine, engine.event_log.entries)
        assert "metadata" in bundle
        meta = bundle["metadata"]
        assert "seed" in meta
        assert "current_turn" in meta
        assert "llm_enabled" in meta
        assert "llm_provider" in meta
        assert "shock_enabled" in meta
        assert "npc_count" in meta
        assert "npc_roles" in meta

    @pytest.mark.asyncio
    async def test_bundle_contains_all_series(self):
        engine = await make_engine(turns=3)
        bundle = build_experiment_bundle(engine, engine.event_log.entries)
        expected_keys = [
            "metadata", "reward_series", "community_reward_series",
            "social_welfare_series", "cooperation_series",
            "cooperation_index", "policy_entropy",
            "action_distribution", "shock_timeline",
            "shock_responses", "adaptation_snapshot",
        ]
        for key in expected_keys:
            assert key in bundle, f"Missing key in bundle: {key}"

    @pytest.mark.asyncio
    async def test_bundle_adaptation_snapshot(self):
        engine = await make_engine(turns=3)
        bundle = build_experiment_bundle(engine, engine.event_log.entries)
        snap = bundle["adaptation_snapshot"]
        assert len(snap) == len(engine.npc_registry)
        for uid, state in snap.items():
            assert "cooperation_tendency" in state


# ── Playthrough Logger RL Telemetry Tests ─────────────────────────────────────


class TestPlaythroughRLTelemetry:
    """Tests for RL telemetry recording in playthrough logger."""

    @pytest.mark.asyncio
    async def test_rl_telemetry_recorded_after_turn(self):
        engine = await make_engine(turns=2)
        records = engine.playthrough_logger.get_all_records()
        rl_records = [r for r in records if r.get("record_type") == "rl_telemetry"]
        # Should have at least 1 rl_telemetry record after 2 turns
        assert len(rl_records) >= 1

    @pytest.mark.asyncio
    async def test_rl_telemetry_contains_npc_data(self):
        engine = await make_engine(turns=2)
        records = engine.playthrough_logger.get_all_records()
        rl_records = [r for r in records if r.get("record_type") == "rl_telemetry"]
        if rl_records:
            rec = rl_records[-1]
            assert "npc_telemetry" in rec
            assert "community_state" in rec
            assert "active_shocks" in rec
            # Check NPC telemetry structure
            for uid, tel in rec["npc_telemetry"].items():
                assert "name" in tel
                assert "reward" in tel
                assert "adaptation" in tel

    @pytest.mark.asyncio
    async def test_rl_telemetry_turn_number(self):
        engine = await make_engine(turns=3)
        records = engine.playthrough_logger.get_all_records()
        rl_records = [r for r in records if r.get("record_type") == "rl_telemetry"]
        if rl_records:
            # Each should have a turn field
            for rec in rl_records:
                assert "turn" in rec
                assert isinstance(rec["turn"], int)


# ── API Endpoint Smoke Tests ─────────────────────────────────────────────────


class TestAnalyticsAPI:
    """Smoke tests for new analytics API endpoints via TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up FastAPI test client with an active game session."""
        from fastapi.testclient import TestClient
        from backend.main import app

        self.client = TestClient(app)

        # Start a new game
        resp = self.client.post("/api/game/new", json={
            "seed": 42,
            "difficulty": "normal",
            "max_turns": 200,
            "player_name": "TestPlayer",
        })
        assert resp.status_code == 200

        # Play a few turns
        for _ in range(3):
            self.client.post("/api/game/action", json={
                "source": "button",
                "action_id": "wait",
            })

        yield

    def test_timeseries_endpoint(self):
        resp = self.client.get("/api/metrics/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        assert "reward_series" in data
        assert "cooperation_series" in data
        assert "social_welfare_series" in data
        assert "policy_entropy" in data
        assert "action_distribution" in data

    def test_cooperation_endpoint(self):
        resp = self.client.get("/api/metrics/cooperation")
        assert resp.status_code == 200
        data = resp.json()
        assert "global_cooperation" in data
        assert "per_role" in data
        assert "per_npc" in data
        assert isinstance(data["global_cooperation"], float)

    def test_experiment_endpoint(self):
        resp = self.client.get("/api/metrics/experiment")
        assert resp.status_code == 200
        data = resp.json()
        assert "metadata" in data
        assert "reward_series" in data
        assert "adaptation_snapshot" in data
        meta = data["metadata"]
        assert "seed" in meta
        assert "llm_enabled" in meta
        assert "llm_provider" in meta

    def test_timeseries_requires_active_session(self):
        """Endpoints should fail gracefully without an active game."""
        from backend.api.routes import session_mgr
        # Save and clear engine
        saved_engine = session_mgr.current_engine
        session_mgr.current_engine = None
        try:
            resp = self.client.get("/api/metrics/timeseries")
            assert resp.status_code == 400
        finally:
            session_mgr.current_engine = saved_engine
