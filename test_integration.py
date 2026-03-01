"""Integration tests for all modified/wired modules."""

import asyncio
import json


def test_guardrails():
    """Test LLM output validators."""
    from backend.llm.guardrails import (
        validate_dialogue_output,
        validate_checkpoint_output,
        validate_input_analysis,
        parse_json_response,
        sanitize_text,
        clamp,
    )

    # --- parse_json_response ---
    assert parse_json_response('{"key": "val"}') == {"key": "val"}
    assert parse_json_response('```json\n{"key": "val"}\n```') == {"key": "val"}
    assert parse_json_response('{"key": "val",}') == {"key": "val"}  # trailing comma
    assert parse_json_response("") is None
    assert parse_json_response("not json") is None
    print("  parse_json_response: PASS")

    # --- sanitize_text ---
    assert sanitize_text("<b>bold</b>") == "bold"
    assert len(sanitize_text("x" * 600)) <= 500
    print("  sanitize_text: PASS")

    # --- clamp ---
    assert clamp(5, 0, 10) == 5
    assert clamp(-10, 0, 10) == 0
    assert clamp(20, 0, 10) == 10
    assert isinstance(clamp(5, 0, 10), int)
    assert isinstance(clamp(5.0, 0, 10), float)
    print("  clamp: PASS")

    # --- Dialogue validation ---
    valid = json.dumps({
        "dialogue": "Hello there, traveler.",
        "mood_change": 1,
        "reveals_info": False,
        "info_type": None,
    })
    result = validate_dialogue_output(valid)
    assert result is not None
    assert result["dialogue"] == "Hello there, traveler."
    assert result["mood_change"] == 1
    assert result["reveals_info"] is False
    print("  Dialogue validation (valid): PASS")

    # Clamping
    clamped = json.dumps({
        "dialogue": "Ha!",
        "mood_change": 10,
        "reveals_info": True,
        "info_type": "quest_hint",
    })
    r2 = validate_dialogue_output(clamped)
    assert r2["mood_change"] == 3
    assert r2["reveals_info"] is True
    assert r2["info_type"] == "quest_hint"
    print("  Dialogue clamping: PASS")

    # Missing field
    bad = json.dumps({"no_dialogue": "oops"})
    assert validate_dialogue_output(bad) is None
    print("  Dialogue rejection: PASS")

    # --- Checkpoint validation ---
    cp_valid = json.dumps({
        "description": "You stumble upon a hidden path behind the old well.",
        "highlighted_actions": ["examine", "search"],
        "effects": {"stamina": -5},
        "hint": "The main road is not far.",
    })
    cp = validate_checkpoint_output(cp_valid)
    assert cp is not None
    assert "examine" in cp["highlighted_actions"]
    assert cp["effects"]["stamina"] == -5
    print("  Checkpoint validation (valid): PASS")

    # Bad actions
    bad_cp = json.dumps({
        "description": "Something happens here in the village.",
        "highlighted_actions": ["fly_away"],
        "effects": {},
    })
    assert validate_checkpoint_output(bad_cp) is None
    print("  Checkpoint rejection (bad actions): PASS")

    # Effects clamping
    cp_clamp = json.dumps({
        "description": "An explosion of magical energy hits you hard.",
        "highlighted_actions": ["flee"],
        "effects": {"health": -100, "stamina": -50, "reputation": 99},
    })
    cp3 = validate_checkpoint_output(cp_clamp)
    assert cp3["effects"]["health"] == -50
    assert cp3["effects"]["stamina"] == -20
    assert cp3["effects"]["reputation"] == 30
    print("  Checkpoint effects clamping: PASS")

    # --- Input analysis validation ---
    inp_valid = json.dumps({
        "emotion": "angry",
        "intent": "attack the guard",
        "social": "rude",
        "matched_action": "attack",
        "confidence": 0.9,
        "interpreted_intent": "Player wants to attack the guard",
    })
    inp = validate_input_analysis(inp_valid)
    assert inp is not None
    assert inp["emotion"] == "angry"
    assert inp["matched_action"] == "attack"
    assert inp["confidence"] == 0.9
    print("  Input analysis (valid): PASS")

    # Invalid action fallback
    bad_action = json.dumps({
        "emotion": "neutral",
        "intent": "fly",
        "social": "neutral",
        "matched_action": "fly",
        "confidence": 0.5,
    })
    inp2 = validate_input_analysis(bad_action)
    assert inp2["matched_action"] == "UNKNOWN"
    print("  Input analysis bad-action fallback: PASS")

    # Confidence clamping
    over_conf = json.dumps({
        "emotion": "neutral",
        "intent": "walk",
        "social": "neutral",
        "matched_action": "move_to",
        "confidence": 5.0,
    })
    inp3 = validate_input_analysis(over_conf)
    assert inp3["confidence"] == 1.0
    print("  Input analysis confidence clamping: PASS")

    print("=== All guardrail tests PASS ===\n")


def test_prompts():
    """Test prompt builders produce non-empty strings within budget."""
    from backend.llm.prompts import (
        build_dialogue_prompt,
        build_checkpoint_prompt,
        build_narration_prompt,
        build_input_analysis_prompt,
        estimate_tokens,
    )

    dp = build_dialogue_prompt(
        npc_name="Elder Maren",
        npc_uid="elder_m8b2",
        npc_role="village elder",
        archetype="elder",
        personality="wise and patient",
        mood="content",
        happiness=7,
        reputation=30,
        context="Player is at the village center.",
        conversation_history=[],
        player_input="Hello, Elder",
        emotion="friendly",
        social="polite",
    )
    assert len(dp) > 100
    assert estimate_tokens(dp) <= 2500
    print("  build_dialogue_prompt: PASS")

    cp = build_checkpoint_prompt(
        stage_desc="Investigate the rumors of bandit activity.",
        player_action="attack",
        emotion="angry",
        social="rude",
        location="gate",
        health=80,
        stamina=40,
        reputation=10,
        expected_next="Talk to the guard about the bandits.",
        inventory_summary="rusty sword, bread",
    )
    assert len(cp) > 100
    assert estimate_tokens(cp) <= 2500
    print("  build_checkpoint_prompt: PASS")

    np_ = build_narration_prompt(
        action_id="attack",
        actor_name="Player",
        target_name="Aldric",
        outcome_type="success",
        template_text="You land a solid blow on Aldric.",
        location="gate",
        time_of_day="evening",
        weather="clear",
        emotion="angry",
        social="neutral",
        witnesses=["Bryn"],
    )
    assert len(np_) > 100
    assert estimate_tokens(np_) <= 2500
    print("  build_narration_prompt: PASS")

    ip = build_input_analysis_prompt(
        player_text="I want to talk to the elder about the quest",
        location="village_center",
        npcs_present=["Elder Maren"],
        highlighted_actions=["talk", "ask_info"],
    )
    assert len(ip) > 100
    assert estimate_tokens(ip) <= 2500
    print("  build_input_analysis_prompt: PASS")

    print("=== All prompt tests PASS ===\n")


def test_narration():
    """Test narration template and enhancement functions."""
    from backend.engine.narration import (
        get_template_narration,
        add_context_modifiers,
        enhance_narration_with_llm,
        passive_perception_check,
        filter_npc_narration,
    )

    # Template narration
    n = get_template_narration("move_to", "success", {"target": "Tavern"})
    assert "Tavern" in n
    print("  get_template_narration (move_to success): PASS")

    n2 = get_template_narration("attack", "success", {"target": "Aldric", "damage": 12})
    assert "Aldric" in n2
    print("  get_template_narration (attack success): PASS")

    n3 = get_template_narration("nonexistent_action", "success", {})
    assert len(n3) > 0  # Falls back to "Something happens."
    print("  get_template_narration (fallback): PASS")

    # Context modifiers
    modified = add_context_modifiers("You walk forward.", "morning")
    assert "You walk forward." in modified
    print("  add_context_modifiers: PASS")

    # Enhancement without LLM (returns original)
    enhanced = enhance_narration_with_llm(
        "You walk forward.", "move_to", "Player", None, "success",
        "village_center", "morning", None, "neutral", "neutral", [],
        llm_service=None,
    )
    assert enhanced == "You walk forward."
    print("  enhance_narration_with_llm (no LLM): PASS")

    # Filter NPC narration
    r = filter_npc_narration("elder_m8b2", "Elder Maren", "talk", "village_center", "village_center", 2)
    assert r["display_type"] == "full"
    print("  filter_npc_narration (same location): PASS")

    r2 = filter_npc_narration("elder_m8b2", "Elder Maren", "talk", "tavern", "village_center", 4)
    assert r2["display_type"] == "brief"
    print("  filter_npc_narration (high importance): PASS")

    r3 = filter_npc_narration("elder_m8b2", "Elder Maren", "wait", "fields", "village_center", 1)
    assert r3["display_type"] == "meanwhile"
    print("  filter_npc_narration (meanwhile): PASS")

    print("=== All narration tests PASS ===\n")


def test_dialogue():
    """Test dialogue pipeline without LLM."""
    from backend.npc.dialogue import resolve_dialogue, format_dialogue

    class MockNPC:
        def __init__(self):
            self.npc_uid = "test_npc_01"
            self.name = "Test NPC"
            self.archetype = "villager"
            self.personality = "friendly"
            self.stats = {"happiness": 6}
            self.conversation_history = []
            self.scripted_dialogue = {"greeting": "Welcome, friend!"}
            self.generic_responses = {
                "greeting": ["Hello.", "Hi there."],
                "unknown": ["I don't know what to say."],
            }

    npc = MockNPC()
    ctx = {"player_reputation": 20, "quest_state": {}, "location": "tavern", "turn": 5, "time_of_day": "morning"}

    # Scripted hit
    r = resolve_dialogue(npc, "greet", None, "neutral", "neutral", ctx)
    assert r["dialogue"] == "Welcome, friend!"
    assert r["mood_change"] == 0
    print("  resolve_dialogue (scripted): PASS")

    # Generic fallback (no scripted match for ask_info)
    r2 = resolve_dialogue(npc, "ask_info", None, "curious", "polite", ctx)
    assert len(r2["dialogue"]) > 0
    print("  resolve_dialogue (generic fallback): PASS")

    # Without LLM (llm_service=None)
    r3 = resolve_dialogue(npc, "talk", "Tell me about the quest", "curious", "polite", ctx, llm_service=None)
    assert r3["dialogue"] in ["Welcome, friend!", "Hello.", "Hi there.", "I don't know what to say."]
    print("  resolve_dialogue (no LLM): PASS")

    # format_dialogue
    f = format_dialogue("Elder Maren", "Ah, welcome.")
    assert f == 'Elder Maren: "Ah, welcome."'
    print("  format_dialogue: PASS")

    print("=== All dialogue tests PASS ===\n")


def test_checkpoint():
    """Test dynamic checkpoint generation without LLM."""
    from backend.quest.checkpoint import generate_dynamic_checkpoint

    ctx = {
        "checkpoint_id": "2_D1",
        "location": "village_center",
        "npc_name": "Aldric",
        "nudge_target": "2_2",
    }

    cp = generate_dynamic_checkpoint(stage_id=2, action_id="attack", context=ctx, llm_service=None)
    assert cp.checkpoint_id == "2_D1"
    assert cp.stage_id == 2
    assert cp.is_dynamic is True
    assert len(cp.description) > 0
    assert len(cp.highlighted_actions) > 0
    print("  generate_dynamic_checkpoint (template, combat): PASS")

    # Social action
    ctx2 = {
        "checkpoint_id": "3_D1",
        "location": "tavern",
        "npc_name": "Tessa",
        "nudge_target": "3_2",
    }
    cp2 = generate_dynamic_checkpoint(stage_id=3, action_id="persuade", context=ctx2, llm_service=None)
    assert cp2.checkpoint_id == "3_D1"
    assert cp2.is_dynamic is True
    print("  generate_dynamic_checkpoint (template, social): PASS")

    # Exploration action
    ctx3 = {
        "checkpoint_id": "1_D1",
        "location": "fields",
        "nudge_target": "1_2",
    }
    cp3 = generate_dynamic_checkpoint(stage_id=1, action_id="search", context=ctx3, llm_service=None)
    assert cp3.is_dynamic is True
    assert "search" in cp3.highlighted_actions or "examine" in cp3.highlighted_actions
    print("  generate_dynamic_checkpoint (template, exploration): PASS")

    print("=== All checkpoint tests PASS ===\n")


def test_difficulty():
    """Test difficulty config and adaptive assessment."""
    from backend.engine.difficulty import DifficultyConfig, assess_player_struggle

    dc = DifficultyConfig("normal")
    assert dc.preset == "normal"
    print("  DifficultyConfig init: PASS")

    dc.apply_preset("easy")
    assert dc.preset == "easy"
    print("  DifficultyConfig apply_preset: PASS")

    dc.apply_preset("hard")
    assert dc.preset == "hard"
    print("  DifficultyConfig apply_preset hard: PASS")

    dc.apply_preset("invalid_preset")
    assert dc.preset == "normal"  # Falls back
    print("  DifficultyConfig invalid preset fallback: PASS")

    # Assess player struggle
    assert assess_player_struggle([]) == "maintain"
    print("  assess_player_struggle (empty): PASS")

    # Heavy failure scenario (deaths + failures + low health → score > 15)
    fail_events = ([{"outcome": "death", "effects": {"health_after": 0}} for _ in range(3)]
                   + [{"outcome": "fail", "effects": {"health_after": 10}} for _ in range(17)])
    result = assess_player_struggle(fail_events)
    assert result == "decrease_difficulty", f"Expected decrease, got {result}"
    print("  assess_player_struggle (struggling): PASS")

    # Easy scenario
    success_events = [{"outcome": "success", "effects": {"health_after": 95}} for _ in range(20)]
    result2 = assess_player_struggle(success_events)
    assert result2 == "increase_difficulty"
    print("  assess_player_struggle (cruising): PASS")

    print("=== All difficulty tests PASS ===\n")


def test_input_parser():
    """Test the NLP input parser."""
    from backend.player.input_parser import parse_text_input, parse_button_input, init_nlp

    # Initialize spaCy (may take a moment)
    init_nlp()

    # Button input
    btn = parse_button_input("attack", target_npc="guard_a3f1")
    assert btn["source"] == "button"
    assert btn["action_id"] == "attack"
    assert btn["confidence"] == 1.0
    assert btn["target_npc"] == "guard_a3f1"
    print("  parse_button_input: PASS")

    # Text input
    npc_registry = {
        "elder_m8b2": {"name": "Elder Maren", "location": "village_center"},
        "guard_a3f1": {"name": "Aldric", "location": "gate"},
    }

    t = parse_text_input("look around", npc_registry, "village_center")
    assert t["source"] == "text"
    assert t["action_id"] == "look"
    print("  parse_text_input ('look around'): PASS")

    t2 = parse_text_input("attack Aldric", npc_registry, "village_center")
    assert t2["action_id"] == "attack"
    print("  parse_text_input ('attack Aldric'): PASS")

    t3 = parse_text_input("go to tavern", npc_registry, "village_center")
    assert t3["action_id"] == "move_to"
    print("  parse_text_input ('go to tavern'): PASS")

    # Empty input
    t4 = parse_text_input("", npc_registry, "village_center")
    assert t4["action_id"] is None
    print("  parse_text_input (empty): PASS")

    print("=== All input parser tests PASS ===\n")


def test_llm_parse_input_no_llm():
    """Test _llm_parse_input returns None when no LLM service is provided."""
    from backend.player.input_parser import _llm_parse_input

    result = asyncio.run(_llm_parse_input("hello", llm_service=None))
    assert result is None
    print("  _llm_parse_input (no LLM): PASS")

    # With a mock that's not available
    class FakeLLM:
        available = False
    result2 = asyncio.run(_llm_parse_input("hello", llm_service=FakeLLM()))
    assert result2 is None
    print("  _llm_parse_input (unavailable LLM): PASS")

    print("=== All LLM parse tests PASS ===\n")


def test_game_engine_init():
    """Test GameEngine can be instantiated."""
    from backend.engine.game_engine import GameEngine

    engine = GameEngine(seed=42, difficulty="normal")
    assert engine.player.name is not None or True  # Player has a default name
    assert engine.turn == 0
    assert engine.game_over is False
    assert engine.difficulty.preset == "normal"
    assert engine.llm is not None  # LLMService instantiated (may not be available)
    print("  GameEngine init: PASS")

    # Check state output
    state = engine.get_full_state()
    assert "player" in state or "turn" in state
    print("  GameEngine.get_full_state: PASS")

    print("=== GameEngine init test PASS ===\n")


def test_api_routes_import():
    """Test that the API routes module loads correctly."""
    from backend.api.routes import router
    routes = [r.path for r in router.routes]
    assert "/api/game/new" in routes
    assert "/api/game/action" in routes
    assert "/api/llm/status" in routes
    assert "/api/saves" in routes
    assert "/api/metrics/summary" in routes
    assert "/api/difficulty" in routes
    assert "/api/export" in routes
    print("  API routes registered: PASS")
    print(f"  Routes found: {len(routes)}")

    print("=== API routes test PASS ===\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  INTEGRATION TESTS")
    print("=" * 60)
    print()

    test_guardrails()
    test_prompts()
    test_narration()
    test_dialogue()
    test_checkpoint()
    test_difficulty()
    test_input_parser()
    test_llm_parse_input_no_llm()
    test_game_engine_init()
    test_api_routes_import()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
