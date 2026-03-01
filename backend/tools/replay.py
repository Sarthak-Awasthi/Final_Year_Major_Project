"""
replay.py — Session replay tool for research.

Reads a save file or event log and replays the game step by step,
allowing researchers to review sessions with timing and formatting.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from backend.config import SAVES_DIR, logger


async def replay_session(
    filepath: str,
    speed: float = 1.0,
) -> list[dict[str, Any]]:
    """Load a save file and replay event log entries with timing.

    Args:
        filepath: Path to a save JSON file or an event-log JSON file.
        speed: Playback speed multiplier (1.0 = real-time based on turn gaps,
               2.0 = double speed, 0 = no delay).

    Returns:
        List of event-log entry dicts in chronological order.
    """
    path = Path(filepath)
    if not path.exists():
        # Try resolving relative to SAVES_DIR
        path = SAVES_DIR / filepath
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    # Accept either a full save file or a bare event-log list
    if isinstance(data, list):
        entries: list[dict[str, Any]] = data
    elif isinstance(data, dict):
        entries = data.get("event_log", [])
    else:
        raise ValueError("Unrecognised replay file format")

    if not entries:
        logger.warning("Replay file contains no event log entries: %s", filepath)
        return []

    logger.info(
        "Replaying %d events from %s (speed=%.1fx)",
        len(entries),
        path.name,
        speed,
    )

    replayed: list[dict[str, Any]] = []
    prev_turn: int = 0

    for entry in entries:
        current_turn: int = entry.get("turn", prev_turn)
        turn_gap = max(0, current_turn - prev_turn)

        # Simulate pacing: 0.5s per turn gap at 1× speed
        if speed > 0 and turn_gap > 0:
            delay = (turn_gap * 0.5) / speed
            await asyncio.sleep(delay)

        replayed.append(entry)
        prev_turn = current_turn

    logger.info("Replay complete: %d events", len(replayed))
    return replayed


def format_replay_entry(entry: dict[str, Any]) -> str:
    """Format a single event-log entry as a human-readable string.

    Format: ``[Turn X | Time] Actor does Action → Outcome``

    Args:
        entry: A single event-log dict.

    Returns:
        Formatted one-line string.
    """
    turn = entry.get("turn", "?")
    time_of_day = entry.get("time_of_day", "???")
    actor = entry.get("actor", "unknown")
    action = entry.get("action", "unknown")
    target = entry.get("target")
    outcome = entry.get("outcome", "")
    importance = entry.get("importance", 1)
    narration = entry.get("narration", "")

    target_str = f" → {target}" if target else ""
    importance_marker = "!" * min(importance, 5)

    line = f"[Turn {turn} | {time_of_day}] {actor} {action}{target_str} → {outcome}"

    if narration:
        # Truncate long narration to keep replay lines scannable
        short_narration = narration[:120] + ("..." if len(narration) > 120 else "")
        line += f'  "{short_narration}"'

    if importance >= 3:
        line = f"{importance_marker} {line}"

    return line


def export_replay_log(
    entries: list[dict[str, Any]],
    output_path: str,
    format: str = "text",
) -> str:
    """Export replay entries to a file as plain text or JSON.

    Args:
        entries: List of event-log dicts.
        output_path: Destination file path.
        format: ``"text"`` for human-readable lines, ``"json"`` for raw JSON.

    Returns:
        The resolved output file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
    else:
        lines: list[str] = [
            "═" * 72,
            "  SESSION REPLAY LOG",
            "═" * 72,
            "",
        ]
        if entries:
            first_turn = entries[0].get("turn", 0)
            last_turn = entries[-1].get("turn", 0)
            lines.append(f"  Turns {first_turn}–{last_turn}  |  {len(entries)} events")
            lines.append("")
            lines.append("─" * 72)

        current_turn: int | None = None
        for entry in entries:
            turn = entry.get("turn", 0)
            if turn != current_turn:
                current_turn = turn
                lines.append("")
                lines.append(f"── Turn {turn} ──")
            lines.append(f"  {format_replay_entry(entry)}")

        lines.append("")
        lines.append("═" * 72)
        lines.append("  END OF REPLAY")
        lines.append("═" * 72)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    logger.info("Replay log exported to %s (%s format, %d entries)", path, format, len(entries))
    return str(path)
