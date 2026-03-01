"""
export_narrative.py — Export game session as a readable narrative document.

Transforms raw event-log data into a structured story document
suitable for research review or presentation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from backend.config import logger


# ─── Narrative generation ────────────────────────────────────────────────────


def export_narrative(
    event_log_entries: list[dict[str, Any]],
    player_data: dict[str, Any],
    quest_data: dict[str, Any],
    format: str = "markdown",
) -> str:
    """Generate a narrative document from a game session.

    Sections produced:
    1. **Prologue** — player info, starting conditions
    2. **Per-turn narration** — events with importance >= 2
    3. **Quest milestones** — stage transitions and quest-critical events
    4. **Epilogue** — final outcome and summary statistics

    Args:
        event_log_entries: Full event log list of dicts.
        player_data: Player state dict (from ``player.to_dict()`` or save).
        quest_data: Quest progress dict (from ``quest_manager.get_quest_progress()``
                    or save ``quest_manager`` key).
        format: ``"markdown"`` or ``"text"``.

    Returns:
        The complete narrative as a string.
    """
    md = format == "markdown"
    sections: list[str] = []

    # ── Prologue ──────────────────────────────────────────────────────────
    if md:
        sections.append("# Session Narrative\n")
        sections.append("## Prologue\n")
    else:
        sections.append("SESSION NARRATIVE")
        sections.append("=" * 40)
        sections.append("\nPROLOGUE\n")

    player_name = player_data.get("name", "Traveler")
    location = player_data.get("location", "unknown")
    health = player_data.get("health", "?")
    max_health = player_data.get("max_health", "?")

    sections.append(
        f"**{player_name}** arrives at *{location}* with {health}/{max_health} HP."
        if md
        else f"{player_name} arrives at {location} with {health}/{max_health} HP."
    )
    sections.append("")

    # Starting inventory
    inventory = player_data.get("inventory", [])
    if inventory:
        item_names = [item.get("name", item.get("id", "?")) for item in inventory]
        inv_str = ", ".join(item_names)
        sections.append(f"Carrying: {inv_str}\n")

    # ── Per-turn narration ────────────────────────────────────────────────
    if md:
        sections.append("## The Story\n")
    else:
        sections.append("THE STORY")
        sections.append("-" * 40 + "\n")

    notable_entries = [e for e in event_log_entries if e.get("importance", 1) >= 2]

    if not notable_entries:
        sections.append("*No notable events were recorded.*\n" if md else "(No notable events recorded.)\n")
    else:
        current_turn: int | None = None
        current_time: str | None = None

        for entry in notable_entries:
            turn = entry.get("turn", 0)
            time_of_day = entry.get("time_of_day", "")

            # Turn / time header
            if turn != current_turn or time_of_day != current_time:
                current_turn = turn
                current_time = time_of_day
                if md:
                    sections.append(f"### Turn {turn} — {time_of_day.capitalize()}\n")
                else:
                    sections.append(f"--- Turn {turn} — {time_of_day.capitalize()} ---\n")

            narration = entry.get("narration", "")
            actor = entry.get("actor", "unknown")
            action = entry.get("action", "")
            outcome = entry.get("outcome", "")
            importance = entry.get("importance", 1)

            if narration:
                # Use the narration text directly
                prefix = "**!!!** " if importance >= 4 and md else ""
                sections.append(f"{prefix}{narration}\n")
            else:
                # Fallback: structured summary
                target = entry.get("target", "")
                target_str = f" targeting {target}" if target else ""
                sections.append(f"{actor} performs {action}{target_str} — {outcome}.\n")

    # ── Quest milestones ──────────────────────────────────────────────────
    if md:
        sections.append("## Quest Milestones\n")
    else:
        sections.append("\nQUEST MILESTONES")
        sections.append("-" * 40 + "\n")

    milestones = [
        e
        for e in event_log_entries
        if e.get("event_type") == "quest_progress" or e.get("importance", 1) >= 5
    ]
    if milestones:
        for m in milestones:
            turn = m.get("turn", "?")
            narration = m.get("narration", m.get("outcome", ""))
            if md:
                sections.append(f"- **Turn {turn}**: {narration}")
            else:
                sections.append(f"  Turn {turn}: {narration}")
        sections.append("")
    else:
        sections.append("No quest milestones reached.\n")

    # ── Epilogue ──────────────────────────────────────────────────────────
    if md:
        sections.append("## Epilogue\n")
    else:
        sections.append("EPILOGUE")
        sections.append("-" * 40 + "\n")

    current_stage = quest_data.get("current_stage", quest_data.get("stage", "?"))
    current_cp = quest_data.get("current_checkpoint", "?")
    completed = quest_data.get("completed_checkpoints", [])
    sections.append(f"Final quest stage: {current_stage}, checkpoint: {current_cp}")
    sections.append(f"Checkpoints completed: {len(completed)}")
    sections.append("")

    stats = generate_summary_stats(event_log_entries, player_data)
    if md:
        sections.append("### Session Statistics\n")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        for key, val in stats.items():
            label = key.replace("_", " ").title()
            sections.append(f"| {label} | {val} |")
    else:
        sections.append("Session Statistics:")
        for key, val in stats.items():
            label = key.replace("_", " ").title()
            sections.append(f"  {label}: {val}")

    sections.append("")
    return "\n".join(sections)


# ─── File I/O ────────────────────────────────────────────────────────────────


def save_narrative(content: str, filepath: str) -> str:
    """Write narrative content to a file.

    Args:
        content: The narrative string.
        filepath: Destination path.

    Returns:
        The resolved file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Narrative saved to %s (%d chars)", path, len(content))
    return str(path)


# ─── Summary statistics ──────────────────────────────────────────────────────


def generate_summary_stats(
    event_log: list[dict[str, Any]],
    player: dict[str, Any],
) -> dict[str, Any]:
    """Compute summary statistics for a game session.

    Args:
        event_log: Full list of event-log entry dicts.
        player: Player state dict.

    Returns:
        Dict with keys: total_turns, total_events, actions_by_type,
        combat_encounters, quests_completed, final_health, final_stamina,
        final_reputation, items_held, most_common_action,
        highest_importance_event.
    """
    if not event_log:
        return {
            "total_turns": 0,
            "total_events": 0,
            "actions_by_type": {},
            "combat_encounters": 0,
            "quests_completed": 0,
            "final_health": player.get("health", 0),
            "final_stamina": player.get("stamina", 0),
            "final_reputation": player.get("reputation", {}),
            "items_held": len(player.get("inventory", [])),
            "most_common_action": None,
            "highest_importance_event": None,
        }

    # Total turns
    turns_seen = {e.get("turn", 0) for e in event_log}
    total_turns = max(turns_seen) if turns_seen else 0

    # Actions by type
    action_counter: Counter[str] = Counter()
    for entry in event_log:
        action = entry.get("action")
        if action:
            action_counter[action] += 1

    # Combat encounters
    combat_encounters = sum(
        1
        for e in event_log
        if e.get("event_type") == "combat" or e.get("action") in ("attack", "defend", "flee")
    )

    # Quest completions (stage transitions marked as quest_progress)
    quests_completed = sum(
        1
        for e in event_log
        if e.get("event_type") == "quest_progress"
    )

    # Highest importance event
    max_importance_entry = max(event_log, key=lambda e: e.get("importance", 0))
    highest_importance = {
        "turn": max_importance_entry.get("turn"),
        "action": max_importance_entry.get("action"),
        "importance": max_importance_entry.get("importance"),
        "narration": (max_importance_entry.get("narration", ""))[:100],
    }

    most_common = action_counter.most_common(1)
    most_common_action = most_common[0][0] if most_common else None

    return {
        "total_turns": total_turns,
        "total_events": len(event_log),
        "actions_by_type": dict(action_counter.most_common()),
        "combat_encounters": combat_encounters,
        "quests_completed": quests_completed,
        "final_health": player.get("health", 0),
        "final_stamina": player.get("stamina", 0),
        "final_reputation": player.get("reputation", {}),
        "items_held": len(player.get("inventory", [])),
        "most_common_action": most_common_action,
        "highest_importance_event": highest_importance,
    }
