"""
difficulty.py — Difficulty scaling config, presets, adaptive adjustment.
"""

from __future__ import annotations

from typing import Any

from backend.config import DIFFICULTY_PRESETS, logger


class DifficultyConfig:
    """Manages difficulty scaling parameters."""

    def __init__(self, preset: str = "normal") -> None:
        self.preset: str = preset
        self.params: dict[str, Any] = {}
        self.apply_preset(preset)

    def apply_preset(self, preset: str) -> None:
        """Apply a named difficulty preset."""
        if preset in DIFFICULTY_PRESETS:
            self.preset = preset
            self.params = DIFFICULTY_PRESETS[preset].copy()
        else:
            logger.warning(f"Unknown difficulty preset '{preset}', using 'normal'")
            self.preset = "normal"
            self.params = DIFFICULTY_PRESETS["normal"].copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a difficulty parameter."""
        return self.params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a single difficulty parameter (custom mode)."""
        self.params[key] = value
        self.preset = "custom"

    def to_dict(self) -> dict:
        """Serialize difficulty config."""
        return {"preset": self.preset, **self.params}

    def from_dict(self, data: dict) -> None:
        """Restore from dict."""
        self.preset = data.get("preset", "normal")
        self.params = {k: v for k, v in data.items() if k != "preset"}
        if not self.params:
            self.apply_preset(self.preset)


def assess_player_struggle(event_log_entries: list[dict], window: int = 20) -> str:
    """
    Analyze recent events for signs of player struggle.

    Returns: 'decrease_difficulty', 'increase_difficulty', or 'maintain'
    """
    recent = event_log_entries[-window:] if event_log_entries else []
    if not recent:
        return "maintain"

    death_count = sum(1 for e in recent if e.get("outcome") == "death")
    fail_count = sum(1 for e in recent if e.get("outcome") == "fail")
    fail_rate = fail_count / max(len(recent), 1)

    health_values = [
        e.get("effects", {}).get("health_after", 100) for e in recent
    ]
    avg_health = sum(health_values) / max(len(health_values), 1)

    deviation_count = sum(
        1 for e in recent if e.get("effects", {}).get("is_dynamic_cp", False)
    )

    struggle_score = (
        death_count * 3
        + fail_rate * 10
        + max(0, 30 - avg_health) / 10
        + deviation_count * 0.5
    )

    if struggle_score > 15:
        return "decrease_difficulty"
    elif struggle_score < 3:
        return "increase_difficulty"
    return "maintain"
