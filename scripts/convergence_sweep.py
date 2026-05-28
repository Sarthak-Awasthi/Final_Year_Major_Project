"""In-process convergence sweep over every static checkpoint.

For each main-path CP: speed-run to it, perform an off-path action,
then play the correct action and assert the quest still converges.
Catches "wanders → returns → completes" regressions that the happy-path
suite in test_quest_flow.py cannot see.

Note: `sneak` returns success=False at indoor CPs (no shadows to use),
so the deviation step won't always register at every CP — the
convergence assertion is the load-bearing one.
"""
import asyncio
import sys

from backend.engine.game_engine import GameEngine


PASS = 0
FAIL = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if ok:
        PASS += 1
    else:
        FAIL += 1


async def run_actions(eng: GameEngine, actions: list[dict]) -> None:
    for act in actions:
        await eng.process_turn(act)


async def speed_run_to(eng: GameEngine, target_cp: str) -> bool:
    """Replay the main-path action sequence until `current_checkpoint == target_cp`.

    The path covers every static CP transition end-to-end; the loop
    bails out early once the target is reached.
    """
    path: list[dict] = [
        {"action_id": "present_item", "target_npc": "guard_a3f1",
         "target_item": "travel_papers", "source": "button"},
        {"action_id": "move_to", "target_location": "village_center", "source": "button"},
        {"action_id": "ask_info", "source": "button"},
        {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
        {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
        {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
        {"action_id": "move_to", "target_location": "fields", "source": "button"},
        {"action_id": "talk", "target_npc": "farmer_j4a1", "source": "button"},
        {"action_id": "move_to", "target_location": "fields", "source": "button"},
        {"action_id": "search", "source": "button"},
        {"action_id": "move_to", "target_location": "elders_house", "source": "button"},
        {"action_id": "give_item", "target_npc": "elder_m8b2",
         "target_item": "jade_amulet", "source": "button"},
        {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
    ]
    for act in path:
        if eng.quest_manager.current_checkpoint == target_cp:
            return True
        await eng.process_turn(act)
    return eng.quest_manager.current_checkpoint == target_cp


async def test_cp(cp_id: str, deviation_action: dict,
                  correct_action: dict, expected_next: str) -> None:
    print(f"\n=== CP {cp_id} convergence (deviate → return → complete) ===")
    eng = GameEngine(seed=42, max_turns=200, restart_on_complete=False)
    arrived = await speed_run_to(eng, cp_id)
    report(arrived, f"Speed-run to {cp_id}",
           f"actual={eng.quest_manager.current_checkpoint}")
    if not arrived:
        return

    cp_before = eng.quest_manager.current_checkpoint
    loc_before = eng.player.location
    await eng.process_turn(deviation_action)
    cp_after_dev = eng.quest_manager.current_checkpoint
    deviated = cp_after_dev != cp_before or eng.quest_manager.deviation_count > 0
    report(deviated, "Deviation registered",
           f"cp {cp_before} → {cp_after_dev}, dev_count={eng.quest_manager.deviation_count}")

    # If the deviation moved the player elsewhere, snap them back — the
    # next assertion is "does the original CP still accept the right
    # action," not "can the player navigate back unaided."
    if eng.player.location != loc_before:
        eng.player.location = loc_before

    await eng.process_turn(correct_action)
    cp_final = eng.quest_manager.current_checkpoint
    report(cp_final == expected_next, f"Converged to {expected_next}",
           f"actual={cp_final}")


async def main() -> None:
    # (cp_id, deviation_action, correct_action, expected_next)
    cases = [
        ("2_1",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "ask_info", "source": "button"},
         "2_2"),
        ("2_2",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
         "3_1"),
        ("3_1",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
         "3_2"),
        ("3_2",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
         "4_1"),
        ("4_1",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "move_to", "target_location": "fields", "source": "button"},
         "4_2"),
        ("4_2",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "talk", "target_npc": "farmer_j4a1", "source": "button"},
         "4_3"),
        ("4_3",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "move_to", "target_location": "fields", "source": "button"},
         "5_1"),
        ("7_1",
         {"action_id": "sneak", "source": "button"},
         {"action_id": "talk", "target_npc": "elder_m8b2", "source": "button"},
         "7_2"),
    ]
    for c in cases:
        await test_cp(*c)

    print(f"\n{'='*60}\n  SUMMARY: {PASS} passed, {FAIL} failed\n{'='*60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
