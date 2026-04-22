"""
test_integration_smoke.py — Smoke test for MVP backend integration.

Tests basic game flow: new game → turns → save → load → more turns.
"""

import asyncio
import json
from pathlib import Path

import pytest

from backend.config import SAVES_DIR
from backend.engine.game_engine import GameEngine
from backend.player.input_parser import parse_button_input


@pytest.mark.asyncio
async def test_new_game_to_save_load():
    """
    Smoke test: 5 actions → save → load → 5 more actions.

    Verifies:
    - Game engine initializes
    - Player actions execute
    - Save creates valid JSON
    - Load restores state
    - Game continues normally
    """
    # Create new game
    engine = GameEngine(seed=42, difficulty="normal", max_turns=50)
    await engine.initialize()

    assert engine.turn == 0
    assert not engine.game_over
    assert engine.player.location == "gate"

    # 5 player turns
    for i in range(5):
        action = parse_button_input(action_id="look")
        result = await engine.process_turn(action)
        assert result is not None
        assert "turn" in result or engine.turn > 0

    assert engine.turn == 5

    # Manual save
    save_path = engine.save_game(slot="test_smoke_1")
    assert Path(save_path).exists()

    # Load and verify state
    engine2 = GameEngine(seed=42, difficulty="normal", max_turns=50)
    await engine2.initialize()

    engine2.load_game(filepath=save_path)

    # State should match
    assert engine2.turn == 5
    assert engine2.player.location == engine.player.location
    assert engine2.player.health == engine.player.health

    # 5 more turns on loaded game
    for i in range(5):
        action = parse_button_input(action_id="wait")
        result = await engine2.process_turn(action)
        assert result is not None

    assert engine2.turn == 10

    # Cleanup
    try:
        Path(save_path).unlink()
        backup = Path(f"{save_path}.backup")
        if backup.exists():
            backup.unlink()
    except OSError:
        pass


@pytest.mark.asyncio
async def test_game_state_response_shape():
    """Verify get_full_state() returns expected shape."""
    engine = GameEngine(seed=42)
    await engine.initialize()

    state = engine.get_full_state()

    # Check required top-level keys
    assert "turn" in state
    assert "time_period" in state
    assert "player" in state
    assert "location" in state
    assert "npcs_here" in state
    assert "quest" in state
    assert "graph" in state
    assert "active_events" in state
    assert "game_over" in state
    assert "game_result" in state
    assert "max_turns" in state

    # Check player dict shape
    player = state["player"]
    assert "name" in player
    assert "health" in player
    assert "stamina" in player
    assert "location" in player

    # Check location shape
    location = state["location"]
    assert "id" in location
    assert "name" in location
    assert "adjacent" in location

    # Check NPC list shape
    assert isinstance(state["npcs_here"], list)

    # Check quest shape
    quest = state["quest"]
    assert "current_stage" in quest
    assert "current_checkpoint" in quest


@pytest.mark.asyncio
async def test_metrics_consistency():
    """Verify get_metrics() returns consistent data."""
    engine = GameEngine(seed=42)
    await engine.initialize()

    # Take 5 actions
    for i in range(5):
        action = parse_button_input(action_id="wait")
        await engine.process_turn(action)

    metrics = engine.get_metrics()

    # Check required fields
    assert "current_turn" in metrics
    assert "total_actions" in metrics
    assert "actions_by_type" in metrics
    assert "combat_encounters" in metrics
    assert "quest_progress" in metrics
    assert "player_health" in metrics
    assert "player_stamina" in metrics
    assert "player_location" in metrics

    # Consistency checks
    assert metrics["current_turn"] == 5
    assert metrics["total_actions"] == 5
    assert metrics["actions_by_type"].get("wait", 0) == 5


@pytest.mark.asyncio
async def test_save_list_endpoint():
    """Verify get_save_list() returns properly formatted list."""
    engine = GameEngine(seed=42)
    await engine.initialize()

    # Create a save
    save_path = engine.save_game(slot="test_list_1")

    try:
        # Get list
        saves = engine.get_save_list()
        assert len(saves) > 0

        # Find our save
        test_save = next(
            (s for s in saves if "test_list_1" in s.get("slot", "")),
            None,
        )
        assert test_save is not None

        # Check structure
        assert "filename" in test_save
        assert "filepath" in test_save
        assert "slot" in test_save
        assert "turn" in test_save
        assert "timestamp" in test_save
        assert "player_name" in test_save

    finally:
        # Cleanup
        try:
            Path(save_path).unlink()
            backup = Path(f"{save_path}.backup")
            if backup.exists():
                backup.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

