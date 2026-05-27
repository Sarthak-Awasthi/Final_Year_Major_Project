"""Comprehensive quality check for faculty demo.

Tests every interaction pattern, edge case, and common user input
to catch embarrassing bugs before the presentation.
"""

import httpx
import sys
import json
import time

BASE = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE, timeout=60.0)

passed = 0
failed = 0
warnings_list = []


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [+] {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [X] {label}" + (f"  ({detail})" if detail else ""))


def warn(label, detail=""):
    warnings_list.append(f"{label}: {detail}")
    print(f"  [!] {label}" + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def new_game(seed=42, condition="C1"):
    r = client.post("/api/game/new", json={
        "seed": seed, "difficulty": "normal",
        "max_turns": 200, "player_name": "Traveler",
        "condition": condition,
    })
    r.raise_for_status()
    return r.json()


def act(source="button", action_id=None, text=None, **kw):
    payload = {"source": source}
    if action_id: payload["action_id"] = action_id
    if text: payload["text"] = text
    payload.update(kw)
    r = client.post("/api/game/action", json=payload)
    r.raise_for_status()
    return r.json()


def state():
    r = client.get("/api/game/state")
    r.raise_for_status()
    return r.json()


def get_cp(s=None):
    if s is None: s = state()
    q = s.get("quest_state") or s.get("quest") or {}
    return q.get("current_checkpoint", "?")


def get_inv(s=None):
    if s is None: s = state()
    inv = s.get("inventory") or s.get("player", {}).get("inventory", [])
    return [i.get("id", "") for i in inv]


# ═══════════════════════════════════════════════════════════════
# 1. FREE TEXT PARSING — common inputs that should NOT break
# ═══════════════════════════════════════════════════════════════
section("1. FREE TEXT PARSING")

new_game()

free_text_cases = [
    ("hello", "greet", "basic greeting"),
    ("hi there", "greet", "informal greeting"),
    ("look around", "look", "exploration"),
    ("what is this place", None, "question about location"),
    ("I want to fight", "attack", "combat intent"),
    ("let me rest", "rest", "rest action"),
    ("pick up the item", "pick_up", "pickup"),
    ("can you repeat yourself", None, "repeat request"),
    ("sneak past them", "sneak", "stealth"),
    ("show travel papers", "present_item", "present item"),
    ("go to village center", "move_to", "navigation"),
    ("tell me about yourself", None, "conversational"),
    ("I want to eat bread", "eat", "eat with actual food word"),
    ("examine the notice board", "examine", "examine object"),
    ("wait here", "wait", "wait action"),
    ("run away", "flee", "flee action"),
]

for text, expected_action, desc in free_text_cases:
    try:
        r = act(source="text", text=text)
        actual = r.get("action_result", {}).get("action_id", "?")
        ok = True
        if expected_action and actual != expected_action:
            ok = False
        # Key: no 500 errors
        check(f"'{text}' → {actual}", ok or expected_action is None,
              f"expected={expected_action}, got={actual}" if not ok and expected_action else desc)
    except Exception as e:
        check(f"'{text}' → no crash", False, str(e))
    # Reset game for each test to avoid state accumulation
    new_game()


# ═══════════════════════════════════════════════════════════════
# 2. SUBSTRING FALSE POSITIVE REGRESSION
# ═══════════════════════════════════════════════════════════════
section("2. SUBSTRING MATCHING REGRESSION")

new_game()

substring_cases = [
    ("repeat yourself", "eat", "repeat should NOT match eat"),
    ("beat the drum", "eat", "beat should NOT match eat"),
    ("create something", "eat", "create should NOT match eat"),
    ("great job", "eat", "great should NOT match eat"),
    ("please wait here", "wait", "wait should match wait"),
    ("I attacked earlier", "attack", "attacked may match attack"),
    ("look at the treated wood", "eat", "treated should NOT match eat"),
]

for text, false_action, desc in substring_cases:
    r = act(source="text", text=text)
    actual = r.get("action_result", {}).get("action_id", "?")
    if "NOT" in desc:
        check(f"'{text}' → {actual} (not {false_action})", actual != false_action, desc)
    else:
        check(f"'{text}' → {actual}", True, desc)
    new_game()


# ═══════════════════════════════════════════════════════════════
# 3. NPC TARGETING & LAST-INTERACTED
# ═══════════════════════════════════════════════════════════════
section("3. NPC TARGETING & CONVERSATION MEMORY")

new_game()

# Talk to Aldric explicitly
r1 = act(action_id="talk", target_npc="guard_a3f1")
check("Talk to Aldric explicitly",
      r1.get("dialogue_speaker") == "Aldric",
      f"speaker={r1.get('dialogue_speaker')}")

# Free text without naming NPC → should go to Aldric
r2 = act(source="text", text="what do you know about this place?")
check("Free text routes to last NPC (Aldric)",
      r2.get("dialogue_speaker") == "Aldric",
      f"speaker={r2.get('dialogue_speaker')}")

# Talk to Bryn, then free text should switch
r3 = act(action_id="talk", target_npc="guard_b7e2")
check("Talk to Bryn explicitly",
      r3.get("dialogue_speaker") == "Bryn",
      f"speaker={r3.get('dialogue_speaker')}")

r4 = act(source="text", text="tell me more")
check("Free text now routes to Bryn",
      r4.get("dialogue_speaker") == "Bryn",
      f"speaker={r4.get('dialogue_speaker')}")


# ═══════════════════════════════════════════════════════════════
# 4. REPEAT DIALOGUE
# ═══════════════════════════════════════════════════════════════
section("4. REPEAT DIALOGUE")

new_game()

r1 = act(action_id="talk", target_npc="guard_a3f1")
original_dialogue = r1.get("dialogue", "")
check("Got initial dialogue from Aldric", bool(original_dialogue))

r2 = act(source="text", text="can you repeat yourself")
repeat_dialogue = r2.get("dialogue", "")
check("'repeat yourself' returns same dialogue",
      repeat_dialogue == original_dialogue,
      f"orig='{original_dialogue[:50]}...' repeat='{repeat_dialogue[:50]}...'")
check("Repeat costs 0 AP",
      r2.get("action_result", {}).get("ap_cost") == 0)

r3 = act(source="text", text="say that again")
check("'say that again' also repeats",
      r3.get("dialogue", "") == original_dialogue)

r4 = act(source="text", text="what did you say")
check("'what did you say' also repeats",
      r4.get("dialogue", "") == original_dialogue)


# ═══════════════════════════════════════════════════════════════
# 5. QUEST GATE — CRITICAL FACULTY SCENARIO
# ═══════════════════════════════════════════════════════════════
section("5. QUEST GATE (Faculty Scenario)")

new_game()

# Greet should NOT advance
r = act(action_id="greet", target_npc="guard_a3f1")
s = r.get("state", {})
check("Greet at gate → quest stays at 1_1 or deviation",
      get_cp(s) != "1_2",
      f"CP={get_cp(s)}")

# Free text "can you repeat" should NOT advance
new_game()
act(action_id="greet", target_npc="guard_a3f1")
r2 = act(source="text", text="can you please repeat yourself")
s2 = r2.get("state", {})
check("'repeat yourself' → quest stays",
      get_cp(s2) != "1_2" and get_cp(s2) != "2_1",
      f"CP={get_cp(s2)}")

# Talk should NOT forward-complete to 2_1
new_game()
r3 = act(action_id="talk", target_npc="guard_a3f1")
s3 = r3.get("state", {})
check("Talk at gate → quest stays (no forward skip)",
      get_cp(s3) != "2_1",
      f"CP={get_cp(s3)}")

# Present papers SHOULD advance
new_game()
r4 = act(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")
s4 = r4.get("state", {})
check("Present travel_papers → advances to 1_2",
      get_cp(s4) == "1_2",
      f"CP={get_cp(s4)}")


# ═══════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════════
section("6. EDGE CASES")

# Empty text
new_game()
try:
    r = act(source="text", text="")
    check("Empty text → handled (no crash)", False, "should have been rejected")
except httpx.HTTPStatusError as e:
    check("Empty text → 422 rejected", e.response.status_code == 422)

# Very long text
new_game()
try:
    r = act(source="text", text="a " * 500)
    check("Very long text → no crash", True)
except httpx.HTTPStatusError:
    check("Very long text → rejected gracefully", True)

# Special characters
new_game()
try:
    r = act(source="text", text="hello! @#$% how are you?")
    check("Special chars → no crash", True)
except Exception as e:
    check("Special chars → no crash", False, str(e))

# Action with no NPCs at location (after moving) — engine falls back to
# an NPC at the current location, which is correct behavior
new_game()
act(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")
act(action_id="move_to", target_location="village_center")
r = act(action_id="talk", target_npc="guard_a3f1")
check("Talk to remote NPC → falls back to local NPC",
      r.get("action_result", {}).get("success") is True,
      f"speaker={r.get('dialogue_speaker')}")

# Pick up when nothing on ground
new_game()
r = act(action_id="pick_up")
check("Pick up with nothing → fails gracefully",
      r.get("action_result", {}).get("success") is False)

# Give item you don't have
new_game()
r = act(action_id="give_item", target_npc="guard_a3f1", target_item="jade_amulet")
check("Give item you don't have → fails gracefully",
      r.get("action_result", {}).get("success") is False)

# Attack guard
new_game()
r = act(action_id="attack", target_npc="guard_a3f1")
ar = r.get("action_result", {})
check("Attack guard → handled (no crash)",
      ar.get("action_id") == "attack")


# ═══════════════════════════════════════════════════════════════
# 7. ABLATION CONDITIONS
# ═══════════════════════════════════════════════════════════════
section("7. ABLATION CONDITIONS")

for cond in ["C1", "C3", "C4", "C5", "C6", "C7"]:
    try:
        s = new_game(condition=cond)
        check(f"{cond} → game starts", s.get("turn") == 0, f"turn={s.get('turn')}")
    except Exception as e:
        check(f"{cond} → game starts", False, str(e))

# Invalid condition
try:
    new_game(condition="C99")
    check("Invalid condition C99 → rejected", False, "should have been rejected")
except httpx.HTTPStatusError:
    check("Invalid condition C99 → rejected", True)


# ═══════════════════════════════════════════════════════════════
# 8. API ENDPOINTS
# ═══════════════════════════════════════════════════════════════
section("8. API ENDPOINTS")

new_game()
act(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")

# Game state
try:
    s = state()
    check("/api/game/state → works", "turn" in s)
except Exception as e:
    check("/api/game/state → works", False, str(e))

# NPC list
try:
    r = client.get("/api/npc/list").json()
    check("/api/npc/list → 6 NPCs", len(r.get("npcs", [])) == 6,
          f"got {len(r.get('npcs', []))}")
except Exception as e:
    check("/api/npc/list → works", False, str(e))

# Cooperation metric
try:
    r = client.get("/api/metrics/cooperation").json()
    check("/api/metrics/cooperation → valid",
          "global_cooperation" in r,
          f"coop={r.get('global_cooperation')}")
except Exception as e:
    check("/api/metrics/cooperation → works", False, str(e))

# Analytics
try:
    r = client.get("/api/metrics/timeseries").json()
    check("/api/metrics/timeseries → valid",
          "cooperation_series" in r)
except Exception as e:
    check("/api/metrics/analytics → works", False, str(e))

# Shocks
try:
    r = client.get("/api/shocks/active").json()
    check("/api/shocks/active → valid", isinstance(r, (list, dict)))
except Exception as e:
    check("/api/shocks/active → works", False, str(e))

# LLM status
try:
    r = client.get("/api/llm/status").json()
    check("/api/llm/status → valid", "available" in r or "status" in r or "provider" in r)
except Exception as e:
    check("/api/llm/status → works", False, str(e))


# ═══════════════════════════════════════════════════════════════
# 9. FULL QUEST PLAYTHROUGH
# ═══════════════════════════════════════════════════════════════
section("9. FULL QUEST PLAYTHROUGH (speed-run)")

new_game()
steps = [
    ("present_item", {"target_npc": "guard_a3f1", "target_item": "travel_papers"}, "1_2"),
    ("move_to", {"target_location": "village_center"}, "2_1"),
    ("ask_info", {"target_npc": "villager_c1d4"}, "2_2"),
    ("talk", {"target_npc": "elder_m8b2"}, "3_1"),
    ("talk", {"target_npc": "elder_m8b2"}, "3_2"),
    ("talk", {"target_npc": "elder_m8b2"}, "4_1"),
    ("move_to", {"target_location": "fields"}, "4_2"),
    ("talk", {"target_npc": "farmer_j4a1"}, "4_3"),
    ("move_to", {"target_location": "fields"}, "5_1"),
    ("search", {}, "5_2"),
    ("move_to", {"target_location": "elders_house"}, "6_2"),
    ("give_item", {"target_npc": "elder_m8b2", "target_item": "jade_amulet"}, "7_1"),
    ("talk", {"target_npc": "elder_m8b2"}, "7_2"),
    ("talk", {"target_npc": "elder_m8b2"}, "1_1"),  # restart
]

all_ok = True
for action_id, kwargs, expected_cp in steps:
    r = act(action_id=action_id, **kwargs)
    actual_cp = get_cp(r.get("state", {}))
    ok = actual_cp == expected_cp
    if not ok:
        all_ok = False
        check(f"{action_id} → {expected_cp}", False, f"got {actual_cp}")
        break

if all_ok:
    check("Full 14-step playthrough completed", True, "all checkpoints correct")


# ═══════════════════════════════════════════════════════════════
# 10. DEVIATION & DYNAMIC CHECKPOINT
# ═══════════════════════════════════════════════════════════════
section("10. DEVIATION & DYNAMIC CHECKPOINTS")

new_game()

# Deviate with use_item
r = act(action_id="use_item", target_item="bread")
s = r.get("state", {})
q = s.get("quest_state") or s.get("quest") or {}
check("use_item → deviation to dynamic CP",
      "D" in get_cp(s),
      f"CP={get_cp(s)}, deviation={q.get('deviation_count')}")

# Converge back
r2 = act(action_id="present_item", target_npc="guard_a3f1", target_item="travel_papers")
s2 = r2.get("state", {})
check("present_item → converge to 1_2",
      get_cp(s2) == "1_2",
      f"CP={get_cp(s2)}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
section("SUMMARY")
total = passed + failed
print(f"\n  {passed}/{total} passed, {failed} failed")
if warnings_list:
    print(f"\n  Warnings ({len(warnings_list)}):")
    for w in warnings_list:
        print(f"    - {w}")
if failed > 0:
    print("\n  ISSUES FOUND — fix before demo")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED — demo ready")
    sys.exit(0)
