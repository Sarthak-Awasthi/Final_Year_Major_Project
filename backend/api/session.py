"""Single-session manager for the MVP. One game at a time; creating a new
session tears down the previous one."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from backend.config import MASTER_SEED, MAX_TURNS, logger, apply_ablation_preset
from backend.engine.game_engine import GameEngine


class SessionManager:
    def __init__(self) -> None:
        self.current_engine: GameEngine | None = None
        self.session_id: str | None = None
        self.created_at: datetime | None = None

    async def create_session(
        self,
        seed: int = MASTER_SEED,
        difficulty: str = "normal",
        max_turns: int = MAX_TURNS,
        player_name: str = "Traveler",
        condition: str = "C1",
    ) -> dict:
        """Spin up a fresh GameEngine and return its initial state.

        `condition` selects one of the ablation presets (C1/C3/.../C7).
        """
        self.end_session()

        self.session_id = uuid.uuid4().hex[:12]
        self.created_at = datetime.now(timezone.utc)

        apply_ablation_preset(condition)

        # Demo sessions terminate cleanly on quest completion. Notebooks
        # construct GameEngine directly and keep the default restart loop
        # for RL training across episodes.
        engine = GameEngine(
            seed=seed,
            difficulty=difficulty,
            max_turns=max_turns,
            restart_on_complete=False,
        )
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
        if self.current_engine is None:
            raise RuntimeError("No active game session.")
        return await self.current_engine.process_turn(parsed_input)

    def get_state(self) -> dict:
        if self.current_engine is None:
            raise RuntimeError("No active game session.")
        return self.current_engine.get_full_state()

    def is_active(self) -> bool:
        return self.current_engine is not None

    def get_session_info(self) -> dict[str, Any]:
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
        if self.current_engine is not None:
            logger.info(
                "Session ended: id=%s, turns=%d",
                self.session_id,
                self.current_engine.turn,
            )
        self.current_engine = None
        self.session_id = None
        self.created_at = None
