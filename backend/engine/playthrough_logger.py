"""
playthrough_logger.py — Per-turn structured playthrough logging.

Records every turn of a game session as a detailed JSON entry capturing:
  1. **Player Input** — raw text / button, parsed action, targets, 3-vector intent
  2. **System Response** — narration, dialogue, action result, quest updates, NPC actions
  3. **World Snapshot** — player state, NPC states, world time, active events, quest progress

Each playthrough is written to ``backend/data/logs/playthrough_<session_id>.jsonl``
as one JSON object per line (JSONL format) for easy streaming and analysis.

A session summary is flushed to ``playthrough_<session_id>_summary.json`` on
game-over or explicit flush.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import LOGS_DIR, GAME_VERSION, logger


class PlaythroughLogger:
    """Records structured per-turn data for research analysis.

    Usage::

        pt_log = PlaythroughLogger(session_id="abc123")
        pt_log.log_turn(turn, parsed_input, action_result, turn_result, world_snapshot)
        ...
        pt_log.flush_summary(final_state)
    """

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.start_time = datetime.now(timezone.utc).isoformat()
        self._turn_count = 0
        self._log_path = LOGS_DIR / f"playthrough_{session_id}.jsonl"
        self._summary_path = LOGS_DIR / f"playthrough_{session_id}_summary.json"

        # Ensure logs dir exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Open file handle for append-mode streaming writes
        self._file = open(self._log_path, "a", encoding="utf-8")

        # Write session header as first line
        header = {
            "record_type": "session_start",
            "session_id": session_id,
            "game_version": GAME_VERSION,
            "timestamp": self.start_time,
        }
        self._write_record(header)
        logger.info("PlaythroughLogger initialized: %s", self._log_path)

    # ── Public API ────────────────────────────────────────────────────────

    def log_turn(
        self,
        turn: int,
        parsed_input: dict[str, Any],
        action_result: dict[str, Any],
        turn_result: dict[str, Any],
        world_snapshot: dict[str, Any],
    ) -> None:
        """Record a single turn's full data.

        Args:
            turn: Current turn number.
            parsed_input: The ParsedInput dict (raw text, action_id, targets,
                emotion, intent, social, confidence, source).
            action_result: Result from ``_resolve_player_action`` (success,
                action_id, ap_cost, narration, effects, target, dialogue).
            turn_result: Full turn result dict returned by ``process_turn``
                (quest_update, npc_narrations, events, perception, etc.).
            world_snapshot: Complete world state snapshot (player, NPCs,
                world time, active events, quest progress, locations).
        """
        self._turn_count += 1

        record: dict[str, Any] = {
            "record_type": "turn",
            "session_id": self.session_id,
            "turn": turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # ── 1. Player Input ───────────────────────────────────────
            "player_input": {
                "source": parsed_input.get("source", "unknown"),
                "raw_text": parsed_input.get("raw_text"),
                "action_id": parsed_input.get("action_id"),
                "target_npc": parsed_input.get("target_npc"),
                "target_item": parsed_input.get("target_item"),
                "target_location": parsed_input.get("target_location"),
                "confidence": parsed_input.get("confidence", 0.0),
                "emotion": parsed_input.get("emotion", "neutral"),
                "intent": parsed_input.get("intent", ""),
                "social": parsed_input.get("social", "neutral"),
            },

            # ── 2. System Response ────────────────────────────────────
            "system_response": {
                "action_id": action_result.get("action_id"),
                "success": action_result.get("success", False),
                "ap_cost": action_result.get("ap_cost", 0),
                "narration": action_result.get("narration", ""),
                "dialogue": action_result.get("dialogue"),
                "effects": _safe_serialize(action_result.get("effects", {})),
                "target": action_result.get("target"),
                "perception": action_result.get("perception"),
                "quest_update": _safe_serialize(turn_result.get("quest_update")),
                "npc_narrations": turn_result.get("npc_narrations", []),
                "new_events": _safe_serialize(turn_result.get("new_events", [])),
                "expired_events": turn_result.get("expired_events", []),
                "stamina_regen": turn_result.get("stamina_regen", 0),
                "reputation_decay": _safe_serialize(turn_result.get("reputation_decay")),
                "perception_check": turn_result.get("perception"),
                "game_over": turn_result.get("game_over", False),
                "game_result": turn_result.get("game_result"),
            },

            # ── 3. World Snapshot (post-turn state) ───────────────────
            "world_state": world_snapshot,
        }

        self._write_record(record)

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a non-turn event (game start, save, load, error, etc.).

        Args:
            event_type: Category string (e.g. ``"game_start"``,
                ``"save"``, ``"load"``, ``"error"``, ``"npc_pretrain"``).
            data: Arbitrary data payload.
        """
        record = {
            "record_type": "event",
            "session_id": self.session_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": _safe_serialize(data),
        }
        self._write_record(record)

    def flush_summary(self, final_state: dict[str, Any] | None = None) -> None:
        """Write a session summary JSON file.

        Called on game-over, explicit save, or server shutdown.

        Args:
            final_state: Optional final game state snapshot.
        """
        summary: dict[str, Any] = {
            "session_id": self.session_id,
            "game_version": GAME_VERSION,
            "start_time": self.start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "total_turns": self._turn_count,
        }
        if final_state:
            summary["final_state"] = _safe_serialize(final_state)

        try:
            with open(self._summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("Playthrough summary written: %s", self._summary_path)
        except OSError as exc:
            logger.error("Failed to write playthrough summary: %s", exc)

    def get_log_path(self) -> str:
        """Return the path to the JSONL log file."""
        return str(self._log_path)

    def get_all_records(self) -> list[dict[str, Any]]:
        """Read and return all records from the JSONL log file.

        Useful for the API endpoint and research tooling.
        """
        records: list[dict[str, Any]] = []
        try:
            self._file.flush()
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass
        return records

    def get_turn_records(self) -> list[dict[str, Any]]:
        """Return only turn records (excludes session_start, events)."""
        return [r for r in self.get_all_records() if r.get("record_type") == "turn"]

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info("PlaythroughLogger closed: %s", self._log_path)

    # ── Internal ──────────────────────────────────────────────────────────

    def _write_record(self, record: dict[str, Any]) -> None:
        """Serialize and append a single JSON record to the JSONL file."""
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            self._file.write(line + "\n")
            self._file.flush()  # Ensure data is written immediately
        except (OSError, TypeError) as exc:
            logger.error("PlaythroughLogger write error: %s", exc)

    def __del__(self) -> None:
        """Ensure file is closed on garbage collection."""
        self.close()


# ─── Snapshot Helpers ─────────────────────────────────────────────────────────


def build_world_snapshot(
    turn: int,
    player: Any,
    npc_registry: dict[str, Any],
    world: Any,
    quest_manager: Any,
    difficulty: Any,
) -> dict[str, Any]:
    """Build a complete world-state snapshot for the playthrough log.

    Captures all mutable game state at the end of a turn so that
    any turn can be reconstructed from the log alone.

    Args:
        turn: Current turn number.
        player: Player instance (has ``to_dict()``).
        npc_registry: ``{npc_uid: NPC}`` mapping.
        world: World instance (has ``to_dict()``).
        quest_manager: QuestManager instance.
        difficulty: DifficultyConfig instance.

    Returns:
        A JSON-serializable dict of the full world state.
    """
    # NPC states — compact (no Q-tables, those are huge)
    npc_states: dict[str, dict[str, Any]] = {}
    for uid, npc in npc_registry.items():
        npc_states[uid] = {
            "name": npc.name,
            "archetype": npc.archetype,
            "location": npc.location,
            "status": npc.status,
            "current_hp": npc.current_hp,
            "max_hp": npc.max_hp,
            "happiness": npc.stats.get("happiness", 0),
            "income": npc.stats.get("income", 0),
            "energy": npc.stats.get("energy", 50),
            "mood": npc.stats.get("mood", "neutral") if isinstance(npc.stats.get("mood"), str) else npc.stats.get("mood", 50),
            "epsilon": round(npc.epsilon, 4),
            "is_defending": npc.is_defending,
            "conversation_count": len(npc.conversation_history),
        }

    # Quest progress
    quest_progress = {}
    if quest_manager:
        try:
            quest_progress = quest_manager.get_quest_progress()
        except Exception:
            quest_progress = {"error": "unavailable"}

    # Difficulty config
    diff_data = {}
    if difficulty:
        try:
            diff_data = difficulty.to_dict()
        except Exception:
            diff_data = {"preset": getattr(difficulty, "preset", "unknown")}

    return {
        "turn": turn,
        "time_of_day": getattr(world, "time_of_day", "unknown"),
        "active_events": getattr(world, "active_events", []),
        "player": _safe_serialize(player.to_dict()) if hasattr(player, "to_dict") else {},
        "npcs": npc_states,
        "quest": _safe_serialize(quest_progress),
        "difficulty": _safe_serialize(diff_data),
    }


def _safe_serialize(obj: Any) -> Any:
    """Ensure an object is JSON-serializable.

    Converts numpy arrays, sets, and other non-serializable types
    to basic Python types.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, set):
        return list(obj)
    # numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    # Fallback: convert to string
    return str(obj)
