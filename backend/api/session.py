"""
session.py — Simple session management for the single-player MVP.

Manages one active game session at a time.  Wraps :class:`GameEngine`
creation, initialization, turn processing, and teardown.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from backend.config import MASTER_SEED, MAX_TURNS, logger, apply_ablation_preset, reset_ablation_defaults
from backend.engine.game_engine import GameEngine


class SessionManager:
    """Single-session manager for the MVP research game.

    Only one game session is active at any time.  Creating a new session
    automatically tears down the previous one.
    """

    def __init__(self) -> None:
        self.current_engine: GameEngine | None = None
        self.session_id: str | None = None
        self.created_at: datetime | None = None

    # ── Session lifecycle ─────────────────────────────────────────────────

    async def create_session(
        self,
        seed: int = MASTER_SEED,
        difficulty: str = "normal",
        max_turns: int = MAX_TURNS,
        player_name: str = "Traveler",
        condition: str = "C1",
    ) -> dict:
        """Create a new game session, replacing any existing one.

        Instantiates and initialises a :class:`GameEngine`, sets the
        player name, and returns the initial game state dict.

        Args:
            seed: Master RNG seed.
            difficulty: Difficulty preset (``easy`` / ``normal`` / ``hard``).
            max_turns: Maximum turns before the session ends.
            player_name: Display name for the player character.
            condition: Ablation preset (``C1`` / ``C3`` / ``C4`` / ``C5`` / ``C6`` / ``C7``).

        Returns:
            Full initial game state dict from ``engine.initialize()``.
        """
        # Tear down previous session if any
        self.end_session()

        self.session_id = uuid.uuid4().hex[:12]
        self.created_at = datetime.now(timezone.utc)

        apply_ablation_preset(condition)

        engine = GameEngine(seed=seed, difficulty=difficulty, max_turns=max_turns)
        engine.player.name = player_name
        initial_state = await engine.initialize()

        self.current_engine = engine

        logger.info(
            "Session created: id=%s, seed=%d, difficulty=%s, player=%s, condition=%s",
            self.session_id,
            seed,
            difficulty,
            player_name,
            condition,
        )
        return initial_state

    async def process_action(self, parsed_input: dict) -> dict:
        """Forward *parsed_input* to the engine's turn processor.

        Args:
            parsed_input: A ``ParsedInput`` dict produced by the input parser.

        Returns:
            Turn result dict from ``engine.process_turn()``.

        Raises:
            RuntimeError: If no session is active.
        """
        if self.current_engine is None:
            raise RuntimeError("No active game session.")
        return await self.current_engine.process_turn(parsed_input)

    def get_state(self) -> dict:
        """Return full game state from the active engine.

        Raises:
            RuntimeError: If no session is active.
        """
        if self.current_engine is None:
            raise RuntimeError("No active game session.")
        return self.current_engine.get_full_state()

    def is_active(self) -> bool:
        """Check whether a game session is currently running."""
        return self.current_engine is not None

    def get_session_info(self) -> dict[str, Any]:
        """Return metadata about the current session (or None-safe defaults)."""
        if not self.is_active():
            return {
                "active": False,
                "session_id": None,
                "created_at": None,
                "turn": 0,
                "game_over": False,
            }
        assert self.current_engine is not None
        return {
            "active": True,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "turn": self.current_engine.turn,
            "game_over": self.current_engine.game_over,
            "game_result": self.current_engine.game_result,
        }

    def end_session(self) -> None:
        """Tear down the current session, releasing resources."""
        if self.current_engine is not None:
            logger.info(
                "Session ended: id=%s, turns=%d",
                self.session_id,
                self.current_engine.turn,
            )
        self.current_engine = None
        self.session_id = None
        self.created_at = None
