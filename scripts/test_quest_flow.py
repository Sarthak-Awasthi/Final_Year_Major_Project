"""End-to-end MDP quest flow test against the running FastAPI server.

Covers all 7 stages / 15 checkpoints. Exercises stage transitions, compound
action keys, `requires` constraints, `success_prob` rolls, item give/remove
effects, and clean terminal exit in demo mode.
"""

import httpx
import sys
import json

BASE = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE, timeout=30.0)

passed = 0
failed = 0
warnings = []


def new_game(seed=42):
    r = client.post("/api/game/new", json={
        "seed": seed, "difficulty": "normal",
        "max_turns": 200, "player_name": "Traveler",
    })
    r.raise_for_status()
    return r.json()


def action(source="button", action_id=None, text=None,
           target_npc=None, target_item=None, target_location=None):
    payload = {"source": source}
    if action_id:
        payload["action_id"] = action_id
    if text:
        payload["text"] = text
    if target_npc:
        payload["target_npc"] = target_npc
    if target_item:
        payload["target_item"] = target_item
    if target_location:
        payload["target_location"] = target_location
    r = client.post("/api/game/action", json=payload)
    r.raise_for_status()
    return r.json()


def state():
    r = client.get("/api/game/state")
    r.raise_for_status()
    return r.json()


def get_cp(s=None):
    if s is None:
        s = state()
    q = s.get("quest_state") or s.get("quest") or {}
    return q.get("current_checkpoint", "?")


def get_stage(s=None):
    if s is None:
        s = state()
    q = s.get("quest_state") or s.get("quest") or {}
    return q.get("current_stage", -1)


def get_inventory_ids(s=None):
    if s is None:
        s = state()
    inv = s.get("inventory") or []
    if not inv:
        p = s.get("player", {})
        inv = p.get("inventory", [])
    return [i.get("id", "") for i in inv]


def check(label, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    mark = "  [+]" if condition else "  [X]"
    msg = f"{mark} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


section("STAGE 1: Arrival at the Gate (1_1 → 1_2)")

new_game(seed=42)
s = state()
check("Initial state: CP=1_1, Stage=1",
      get_cp(s) == "1_1" and get_stage(s) == 1,
      f"CP={get_cp(s)}, Stage={get_stage(s)}")
check("Player has travel_papers",
      "travel_papers" in get_inventory_ids(s))

# Greet alone shouldn't satisfy the gate — that's the whole point of movement_gate.
r = action(action_id="greet", target_npc="guard_a3f1")
rs = r.get("state", {})
check("Greet guard → quest does NOT advance",
      get_cp(rs) != "1_2",
      f"CP={get_cp(rs)}")

new_game(seed=42)
r = action(action_id="present_item", target_npc="guard_a3f1",
           target_item="travel_papers")
rs = r.get("state", {})
check("Present travel_papers → CP=1_2",
      get_cp(rs) == "1_2",
      f"CP={get_cp(rs)}")

# Seed 42 deterministically rolls a sneak failure here — keep this seed.
new_game(seed=42)
r = action(action_id="sneak")
rs = r.get("state", {})
success = r.get("action_result", {}).get("success")
check("Sneak (seed 42) fails → quest stays at 1_1",
      get_cp(rs) == "1_1" and success is False,
      f"success={success}, CP={get_cp(rs)}")

# Seed 1 → persuade rolls true.
new_game(seed=1)
r = action(action_id="persuade", target_npc="guard_a3f1")
rs = r.get("state", {})
success = r.get("action_result", {}).get("success")
if success:
    check("Persuade (seed 1) succeeds → CP=1_2",
          get_cp(rs) == "1_2",
          f"CP={get_cp(rs)}")
else:
    check("Persuade (seed 1) failed unexpectedly",
          False, f"success={success}")

new_game(seed=42)
action(action_id="present_item", target_npc="guard_a3f1",
       target_item="travel_papers")
r = action(action_id="move_to", target_location="village_center")
rs = r.get("state", {})
check("1_2: move_to village_center → CP=2_1, Stage=2",
      get_cp(rs) == "2_1" and get_stage(rs) == 2,
      f"CP={get_cp(rs)}, Stage={get_stage(rs)}")


section("STAGE 2: Seeking the Elder (2_1 → 2_2)")

# Continues from previous game — the server holds the session.
r = action(action_id="ask_info", target_npc="villager_c1d4")
rs = r.get("state", {})
check("2_1: ask_info → CP=2_2",
      get_cp(rs) == "2_2",
      f"CP={get_cp(rs)}")

r = action(action_id="talk", target_npc="elder_m8b2")
rs = r.get("state", {})
check("2_2: talk to Elder Maren → CP=3_1, Stage=3",
      get_cp(rs) == "3_1" and get_stage(rs) == 3,
      f"CP={get_cp(rs)}, Stage={get_stage(rs)}")


section("STAGE 3: The Missing Artifact (3_1 → 3_2)")

r = action(action_id="talk", target_npc="elder_m8b2")
rs = r.get("state", {})
check("3_1: talk → CP=3_2",
      get_cp(rs) == "3_2",
      f"CP={get_cp(rs)}")

r = action(action_id="talk", target_npc="elder_m8b2")
rs = r.get("state", {})
check("3_2: talk (accept quest) → CP=4_1, Stage=4",
      get_cp(rs) == "4_1" and get_stage(rs) == 4,
      f"CP={get_cp(rs)}, Stage={get_stage(rs)}")


section("STAGE 4: The Investigation (4_1 → 4_2 → 4_3)")

# Compound key: move_to_fields → 4_2.
r = action(action_id="move_to", target_location="fields")
rs = r.get("state", {})
check("4_1: move_to fields → CP=4_2",
      get_cp(rs) == "4_2",
      f"CP={get_cp(rs)}")

r = action(action_id="talk", target_npc="farmer_j4a1")
rs = r.get("state", {})
check("4_2: talk to Farmer Jak → CP=4_3",
      get_cp(rs) == "4_3",
      f"CP={get_cp(rs)}")

# Same-location move_to (fields → fields) — exercises the
# benign-success branch and crosses the stage boundary.
r = action(action_id="move_to", target_location="fields")
rs = r.get("state", {})
check("4_3: move_to fields → CP=5_1, Stage=5",
      get_cp(rs) == "5_1" and get_stage(rs) == 5,
      f"CP={get_cp(rs)}, Stage={get_stage(rs)}")


section("STAGE 5: The Old Oak Tree (5_1 → 5_2)")

s = state()
inv_before = get_inventory_ids(s)
check("Before search: no jade_amulet",
      "jade_amulet" not in inv_before,
      f"inventory={inv_before}")

r = action(action_id="search")
rs = r.get("state", {})
inv_after = get_inventory_ids(rs)
check("5_1: search → CP=5_2",
      get_cp(rs) == "5_2",
      f"CP={get_cp(rs)}")
check("5_1: search gives jade_amulet",
      "jade_amulet" in inv_after,
      f"inventory={inv_after}")

# Compound key move_to_elders_house → 6_2 (skipping 6_1).
r = action(action_id="move_to", target_location="elders_house")
rs = r.get("state", {})
check("5_2: move_to elders_house → CP=6_2 (direct) or 6_1, Stage=6",
      get_cp(rs) in ("6_1", "6_2") and get_stage(rs) == 6,
      f"CP={get_cp(rs)}, Stage={get_stage(rs)}")


section("STAGE 6: Return to the Elder (6_1 → 6_2)")

current_cp = get_cp(rs)
if current_cp == "6_1":
    r = action(action_id="move_to", target_location="elders_house")
    rs = r.get("state", {})
    check("6_1: move_to elders_house → CP=6_2",
          get_cp(rs) == "6_2",
          f"CP={get_cp(rs)}")
else:
    check("Skipped 6_1 (went directly to 6_2)", True)

inv_before = get_inventory_ids(rs)
check("Before give: has jade_amulet",
      "jade_amulet" in inv_before,
      f"inventory={inv_before}")

r = action(action_id="give_item", target_npc="elder_m8b2",
           target_item="jade_amulet")
rs = r.get("state", {})
check("6_2: give_item jade_amulet → CP=7_1, Stage=7",
      get_cp(rs) == "7_1" and get_stage(rs) == 7,
      f"CP={get_cp(rs)}, Stage={get_stage(rs)}")

inv_after = get_inventory_ids(rs)
check("6_2: jade_amulet removed from inventory",
      "jade_amulet" not in inv_after,
      f"inventory={inv_after}")


section("STAGE 7: The Reward (7_1 → 7_2 → Victory)")

r = action(action_id="talk", target_npc="elder_m8b2")
rs = r.get("state", {})
check("7_1: talk → CP=7_2",
      get_cp(rs) == "7_2",
      f"CP={get_cp(rs)}")

# Demo-mode terminal: the API constructs the engine with
# restart_on_complete=False, so S_success sets game_over instead of looping.
r = action(action_id="talk", target_npc="elder_m8b2")
rs = r.get("state", {})

inv_final = get_inventory_ids(rs)
check("7_2: iron_shield given",
      "iron_shield" in inv_final,
      f"inventory={inv_final}")

check("7_2: quest completes cleanly (game_over=True)",
      r.get("game_over") is True and r.get("game_result") == "success",
      f"game_over={r.get('game_over')}, result={r.get('game_result')}")


section("ADDITIONAL: Stage 4 alternate path (tavern)")

new_game(seed=42)
action(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")
action(action_id="move_to", target_location="village_center")
action(action_id="ask_info", target_npc="villager_c1d4")
action(action_id="talk", target_npc="elder_m8b2")
action(action_id="talk", target_npc="elder_m8b2")
action(action_id="talk", target_npc="elder_m8b2")
s = state()
check("Speed-run to 4_1",
      get_cp(s) == "4_1" and get_stage(s) == 4,
      f"CP={get_cp(s)}, Stage={get_stage(s)}")

# move_to_tavern is a CP 4_1 transition that jumps directly to 4_3.
r = action(action_id="move_to", target_location="tavern")
rs = r.get("state", {})
check("4_1: move_to tavern → CP=4_3 (skips 4_2)",
      get_cp(rs) == "4_3",
      f"CP={get_cp(rs)}")


section("ADDITIONAL: Stage 3_2 persuade (success_prob=0.7)")

new_game(seed=42)
action(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")
action(action_id="move_to", target_location="village_center")
action(action_id="ask_info", target_npc="villager_c1d4")
action(action_id="talk", target_npc="elder_m8b2")
action(action_id="talk", target_npc="elder_m8b2")
s = state()
check("At CP=3_2",
      get_cp(s) == "3_2",
      f"CP={get_cp(s)}")

r = action(action_id="persuade", target_npc="elder_m8b2")
rs = r.get("state", {})
success = r.get("action_result", {}).get("success")
if success:
    check("3_2: persuade succeeds → CP=4_1",
          get_cp(rs) == "4_1",
          f"CP={get_cp(rs)}")
else:
    check("3_2: persuade fails → quest stays at 3_2",
          get_cp(rs) == "3_2",
          f"success={success}, CP={get_cp(rs)}")


section("ADDITIONAL: Stage 5_2 compound key (village_center)")

new_game(seed=42)
action(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")
action(action_id="move_to", target_location="village_center")
action(action_id="ask_info", target_npc="villager_c1d4")
action(action_id="talk", target_npc="elder_m8b2")
action(action_id="talk", target_npc="elder_m8b2")
action(action_id="talk", target_npc="elder_m8b2")
action(action_id="move_to", target_location="fields")
action(action_id="talk", target_npc="farmer_j4a1")
action(action_id="move_to", target_location="fields")
action(action_id="search")
s = state()
check("At CP=5_2 with jade_amulet",
      get_cp(s) == "5_2" and "jade_amulet" in get_inventory_ids(s),
      f"CP={get_cp(s)}, inv={get_inventory_ids(s)}")

# Compound key move_to_village_center → 6_1 (alternate to elders_house → 6_2).
r = action(action_id="move_to", target_location="village_center")
rs = r.get("state", {})
check("5_2: move_to village_center → CP=6_1",
      get_cp(rs) == "6_1",
      f"CP={get_cp(rs)}")


section("SUMMARY")
total = passed + failed
print(f"\n  {passed}/{total} passed, {failed} failed")
if warnings:
    print(f"  Warnings:")
    for w in warnings:
        print(f"    - {w}")
if failed > 0:
    print("\n  SOME TESTS FAILED — see [X] markers above")
    sys.exit(1)
else:
    print("\n  ALL TESTS PASSED")
    sys.exit(0)
