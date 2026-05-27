"""
routes.py — REST API routes for the MVP research game.

All endpoints are ``async def`` and tagged by category for the Swagger UI.
Pydantic models provide request/response validation with example values.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import (
    ACTION_CATEGORIES,
    GAME_VERSION,
    MASTER_SEED,
    MAX_TURNS,
    UNIVERSAL_ACTIONS,
    logger,
)
from backend.engine.game_engine import GameEngine
from backend.engine.analytics import (
    build_experiment_bundle,
    compute_action_distribution,
    compute_community_reward_series,
    compute_cooperation_index,
    compute_cooperation_series,
    compute_policy_entropy,
    compute_reward_series,
    compute_shock_response,
    compute_social_welfare_series,
)
from backend.player.input_parser import init_nlp, parse_button_input, parse_text_input
from backend.api.session import SessionManager
from backend.api.websocket import ws_manager


# ─── Module-level state ──────────────────────────────────────────────────────

session_mgr = SessionManager()


def get_engine(raise_on_missing: bool = True) -> GameEngine | None:
    """Return the active :class:`GameEngine` or raise ``400``.

    Args:
        raise_on_missing: If ``True`` (default), raise ``HTTPException``
            when no game is running.  When ``False``, silently return
            ``None`` (used by WebSocket reconnection).

    Returns:
        The current engine instance, or ``None``.
    """
    if session_mgr.current_engine is None:
        if raise_on_missing:
            raise HTTPException(
                status_code=400,
                detail="No active game session. Start a new game first via POST /api/game/new.",
            )
        return None
    return session_mgr.current_engine


# ─── Pydantic Request / Response models ──────────────────────────────────────


class NewGameRequest(BaseModel):
    """Parameters for starting a new game session."""

    seed: int = Field(default=MASTER_SEED, description="Master RNG seed.")
    difficulty: str = Field(
        default="normal",
        description="Difficulty preset: easy, normal, or hard.",
    )
    max_turns: int = Field(
        default=MAX_TURNS,
        description="Maximum turns before session ends.",
    )
    player_name: str = Field(
        default="Traveler",
        description="Display name for the player character.",
    )
    condition: str = Field(
        default="C1",
        description="Ablation condition: C1 (full), C3 (no RL), C4 (flat MDP), C5 (no shocks), C6 (no masking), C7 (static lambda).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "seed": 42,
                    "difficulty": "normal",
                    "max_turns": 200,
                    "player_name": "Traveler",
                    "condition": "C1",
                }
            ]
        }
    }


class ActionRequest(BaseModel):
    """Payload for submitting a player action (button or free-text)."""

    source: str = Field(
        ...,
        description="Input mode: 'button' or 'text'.",
    )
    text: str | None = Field(
        default=None,
        description="Free-text command (required when source='text').",
    )
    action_id: str | None = Field(
        default=None,
        description="Universal action ID (required when source='button').",
    )
    target_npc: str | None = Field(
        default=None,
        description="Target NPC UID, if applicable.",
    )
    target_item: str | None = Field(
        default=None,
        description="Target item ID, if applicable.",
    )
    target_location: str | None = Field(
        default=None,
        description="Target location ID, if applicable.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "button",
                    "action_id": "talk",
                    "target_npc": "elder_m8b2",
                },
                {
                    "source": "text",
                    "text": "ask the elder about the missing supplies",
                },
            ]
        }
    }


class SaveRequest(BaseModel):
    """Payload for manual save."""

    slot: str = Field(
        default="manual_1",
        description="Save slot name (e.g. manual_1, manual_2, manual_3).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"slot": "manual_1"}]
        }
    }


class LoadRequest(BaseModel):
    """Payload for loading a saved game."""

    filepath: str = Field(
        ...,
        description="Absolute or relative path to the save JSON file.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"filepath": "backend/data/saves/save_manual_1.json"}]
        }
    }


# ── Response models ───────────────────────────────────────────────────────────


class GameStateResponse(BaseModel):
    """Full game state returned on new game, load, or state query."""

    turn: int = Field(description="Current turn number.")
    time_period: str = Field(description="Current time of day.")
    player: dict = Field(description="Full player state dict.")
    location: dict = Field(description="Current location details.")
    npcs_here: list[dict] = Field(description="NPCs at the player's location.")
    quest: dict = Field(description="Quest progress summary.")
    graph: dict = Field(description="MDP graph data for Cytoscape.js.")
    active_events: list = Field(default_factory=list, description="Active random events.")
    game_over: bool = Field(description="Whether the game has ended.")
    game_result: str | None = Field(default=None, description="Game end result.")
    max_turns: int = Field(description="Maximum turns for the session.")
    opening_narration: str | None = Field(default=None, description="Backstory narration shown on game start.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "turn": 0,
                    "time_period": "morning",
                    "player": {"name": "Traveler", "health": 100, "stamina": 50},
                    "location": {"id": "gate", "name": "Village Gate"},
                    "npcs_here": [],
                    "quest": {"current_stage": 1, "current_checkpoint": "1_1"},
                    "graph": {"nodes": [], "edges": []},
                    "active_events": [],
                    "game_over": False,
                    "game_result": None,
                    "max_turns": 200,
                }
            ]
        }
    }


class TurnResultResponse(BaseModel):
    """Result of processing a single player turn."""

    turn: int = Field(description="Turn number after processing.")
    narration: str = Field(default="", description="Combined narration text.")
    dialogue: str | None = Field(default=None, description="NPC dialogue text, separate from narration.")
    dialogue_speaker: str | None = Field(default=None, description="Name of the NPC speaking.")
    action_result: dict = Field(default_factory=dict, description="Player action outcome.")
    npc_actions: list[dict] = Field(default_factory=list, description="NPC actions this turn.")
    events: list[dict] = Field(default_factory=list, description="Random events triggered.")
    state: dict = Field(default_factory=dict, description="Updated game state.")
    game_over: bool = Field(default=False, description="Whether the game ended this turn.")
    game_result: str | None = Field(default=None, description="End result if game_over.")
    quest_update: dict | None = Field(default=None, description="Quest change if any.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "turn": 1,
                    "narration": "You greet Elder Maren warmly.",
                    "action_result": {"action": "greet", "outcome": "success"},
                    "npc_actions": [],
                    "events": [],
                    "state": {},
                    "game_over": False,
                    "game_result": None,
                    "quest_update": None,
                }
            ]
        }
    }


class SaveListResponse(BaseModel):
    """List of available save files."""

    saves: list[dict] = Field(description="Save file metadata entries.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "saves": [
                        {
                            "filename": "save_manual_1.json",
                            "slot": "manual_1",
                            "turn": 15,
                            "timestamp": "2026-03-01T12:00:00+00:00",
                            "player_name": "Traveler",
                        }
                    ]
                }
            ]
        }
    }


class ActionCatalogResponse(BaseModel):
    """The 27 universal actions grouped by category."""

    categories: dict[str, list[dict]] = Field(
        description="Actions grouped by category key.",
    )
    total_actions: int = Field(description="Total number of actions (27).")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "categories": {
                        "navigation": [
                            {"id": "move_to", "label": "Move to location", "base_ap": 3}
                        ]
                    },
                    "total_actions": 27,
                }
            ]
        }
    }


class LLMStatusResponse(BaseModel):
    """LLM availability information."""

    available: bool = Field(description="Whether the configured provider is reachable.")
    enabled: bool = Field(default=True, description="Whether LLM usage is enabled by config.")
    provider: str = Field(default="", description="Configured LLM provider key.")
    model_name: str = Field(default="", description="Selected provider model identifier.")
    base_url: str = Field(default="", description="Provider base URL.")
    calls_this_minute: int = Field(default=0, description="Calls made in the current minute window.")
    max_calls: int = Field(default=20, description="Configured per-minute call cap.")
    model_path: str = Field(default="", description="Deprecated legacy field; retained for compatibility.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "available": True,
                    "enabled": True,
                    "provider": "ollama",
                    "model_name": "qwen3:4b",
                    "base_url": "http://127.0.0.1:11434",
                    "calls_this_minute": 2,
                    "max_calls": 20,
                    "model_path": "",
                }
            ]
        }
    }


class MetricsSummaryResponse(BaseModel):
    """Session metrics for the research dashboard."""

    current_turn: int = Field(description="Current turn number.")
    game_over: bool = Field(description="Whether the game has ended.")
    game_result: str | None = Field(default=None, description="Game end result.")
    total_actions: int = Field(default=0, description="Total player actions taken.")
    actions_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Action counts by action_id.",
    )
    combat_encounters: int = Field(default=0, description="Total combat encounters.")
    dynamic_checkpoints_created: int = Field(
        default=0,
        description="Number of dynamic checkpoints generated.",
    )
    quest_progress: dict = Field(
        default_factory=dict,
        description="Quest progress summary.",
    )
    player_health: int = Field(default=100, description="Player health.")
    player_stamina: int = Field(default=50, description="Player stamina/AP.")
    player_location: str = Field(default="gate", description="Player's location.")
    npc_states: dict[str, dict] = Field(
        default_factory=dict,
        description="NPC state summaries keyed by UID.",
    )
    event_log_size: int = Field(default=0, description="Number of event log entries.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "current_turn": 10,
                    "game_over": False,
                    "game_result": None,
                    "total_actions": 10,
                    "actions_by_type": {"talk": 3, "move_to": 4, "look": 3},
                    "combat_encounters": 0,
                    "dynamic_checkpoints_created": 0,
                    "quest_progress": {
                        "current_stage": 1,
                        "completion_percent": 14.3,
                    },
                    "player_health": 100,
                    "player_stamina": 37,
                    "player_location": "village_center",
                    "npc_states": {},
                    "event_log_size": 22,
                }
            ]
        }
    }


class NPCSummary(BaseModel):
    """Abbreviated NPC information for the list endpoint."""

    npc_uid: str = Field(description="Unique NPC identifier.")
    name: str = Field(description="Display name.")
    archetype: str = Field(description="NPC archetype key.")
    location: str = Field(description="Current world location.")
    status: str = Field(description="active / incapacitated.")
    reputation: int = Field(description="Player's reputation with this NPC.")
    reputation_label: str = Field(description="Reputation tier label.")


class NPCListResponse(BaseModel):
    """List of all NPCs."""

    npcs: list[NPCSummary] = Field(description="All NPC summaries.")


class NPCDetailResponse(BaseModel):
    """Detailed NPC information."""

    npc_uid: str
    name: str
    archetype: str
    location: str
    status: str
    personality: str
    reputation: int
    reputation_label: str
    quest_critical: bool
    current_hp: int
    max_hp: int
    stats: dict
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Recent dialogue exchanges.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "npc_uid": "elder_m8b2",
                    "name": "Elder Maren",
                    "archetype": "elder",
                    "location": "elders_house",
                    "status": "active",
                    "personality": "wise",
                    "reputation": 0,
                    "reputation_label": "neutral",
                    "quest_critical": True,
                    "current_hp": 30,
                    "max_hp": 30,
                    "stats": {"happiness": 5, "income": 3},
                    "conversation_history": [],
                }
            ]
        }
    }


class EventListResponse(BaseModel):
    """Recent event log entries."""

    events: list[dict] = Field(description="Event log entries, most recent last.")
    total: int = Field(description="Total entries in the event log.")


class SessionInfoResponse(BaseModel):
    """Information about the current session."""

    active: bool
    session_id: str | None = None
    created_at: str | None = None
    turn: int = 0
    game_over: bool = False
    game_result: str | None = None


class QuestProgressResponse(BaseModel):
    """Quest progress snapshot."""

    quest_id: str = ""
    title: str = ""
    current_stage: int = 1
    current_checkpoint: str = "1_1"
    completed_checkpoints: list[str] = Field(default_factory=list)
    dynamic_checkpoints: list[str] = Field(default_factory=list)
    deviation_count: int = 0
    completion_percent: float = 0.0
    quest_complete: bool = False
    quest_failed: bool = False
    total_checkpoints: int = 0


class GraphDataResponse(BaseModel):
    """MDP graph suitable for Cytoscape.js rendering."""

    nodes: list[dict] = Field(description="Graph nodes.")
    edges: list[dict] = Field(description="Graph edges.")


# ─── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api")


# ── Game endpoints ────────────────────────────────────────────────────────────


@router.post(
    "/game/new",
    response_model=GameStateResponse,
    tags=["game"],
    summary="Create new game session",
)
async def new_game(req: NewGameRequest) -> GameStateResponse:
    """Start a new game session, initialising the engine and returning
    the full initial state.  Replaces any existing session.
    """
    if req.difficulty not in ("easy", "normal", "hard"):
        raise HTTPException(status_code=422, detail="Invalid difficulty. Choose: easy, normal, hard.")
    if req.condition not in ("C1", "C3", "C4", "C5", "C6", "C7"):
        raise HTTPException(status_code=422, detail="Invalid condition. Choose: C1, C3, C4, C5, C6, C7.")

    state = await session_mgr.create_session(
        seed=req.seed,
        difficulty=req.difficulty,
        max_turns=req.max_turns,
        player_name=req.player_name,
        condition=req.condition,
    )

    # Broadcast initial state over WebSocket
    await ws_manager.broadcast({"type": "state_sync", "data": state})

    logger.info("New game started: player=%s, difficulty=%s, condition=%s", req.player_name, req.difficulty, req.condition)
    return GameStateResponse(**state)


@router.post(
    "/game/action",
    response_model=TurnResultResponse,
    tags=["game"],
    summary="Submit player action",
)
async def submit_action(req: ActionRequest) -> TurnResultResponse:
    """Submit a player action via button or free-text input.

    The request is parsed into a unified ``ParsedInput`` dict and
    forwarded to the engine for turn processing.
    """
    engine = get_engine()

    if engine.game_over:
        raise HTTPException(
            status_code=400,
            detail=f"Game is over. Result: {engine.game_result}",
        )

    # Parse input based on source
    if req.source == "text":
        if not req.text:
            raise HTTPException(status_code=422, detail="'text' field required when source='text'.")

        parsed_input = parse_text_input(
            text=req.text,
            npc_registry=engine.npc_registry,
            player_location=engine.player.location,
            player_inventory=engine.player.inventory if hasattr(engine.player, 'inventory') else None,
        )

        # Layer 3: LLM refinement for low-confidence parses
        if parsed_input.get("confidence", 1.0) < 0.5 and engine.llm and engine.llm.available:
            from backend.player.input_parser import _llm_parse_input
            from backend.npc.personality import get_npcs_at_location

            npcs_here = get_npcs_at_location(engine.npc_registry, engine.player.location)
            npc_names = [npc.name for npc in npcs_here]
            highlighted = []
            if hasattr(engine, 'quest_manager') and engine.quest_manager:
                cp = engine.quest_manager.mdp.get_checkpoint(engine.quest_manager.current_checkpoint)
                if cp and hasattr(cp, 'highlighted_actions'):
                    highlighted = cp.highlighted_actions or []

            llm_parsed = await _llm_parse_input(
                text=req.text,
                location=engine.player.location,
                npcs_present=npc_names,
                highlighted_actions=highlighted,
                llm_service=engine.llm,
            )
            if llm_parsed is not None and llm_parsed.get("confidence", 0) > parsed_input.get("confidence", 0):
                # Keep original target extraction, override action/emotion/social
                llm_parsed["target_npc"] = llm_parsed.get("target_npc") or parsed_input.get("target_npc")
                llm_parsed["target_item"] = llm_parsed.get("target_item") or parsed_input.get("target_item")
                llm_parsed["target_location"] = llm_parsed.get("target_location") or parsed_input.get("target_location")
                parsed_input = llm_parsed

    elif req.source == "button":
        if not req.action_id:
            raise HTTPException(
                status_code=422,
                detail="'action_id' field required when source='button'.",
            )
        parsed_input = parse_button_input(
            action_id=req.action_id,
            target_npc=req.target_npc,
            target_item=req.target_item,
            target_location=req.target_location,
        )

    else:
        raise HTTPException(
            status_code=422,
            detail="Invalid source. Must be 'button' or 'text'.",
        )

    # Process the turn
    result = await engine.process_turn(parsed_input)

    # Build response — accommodate varying engine result shapes
    action_result = result.get("action_result", {})
    narration = result.get("narration") or action_result.get("narration", "")
    dialogue = action_result.get("dialogue")
    dialogue_speaker = action_result.get("dialogue_speaker")
    npc_actions = result.get("npc_actions") or result.get("npc_narrations", [])
    events = result.get("events") or result.get("new_events", [])

    turn_response = TurnResultResponse(
        turn=result.get("turn", engine.turn),
        narration=narration,
        dialogue=dialogue,
        dialogue_speaker=dialogue_speaker,
        action_result=action_result,
        npc_actions=npc_actions,
        events=events,
        state=result.get("state", engine.get_full_state()),
        game_over=result.get("game_over", engine.game_over),
        game_result=result.get("game_result", engine.game_result),
        quest_update=result.get("quest_update"),
    )

    # Broadcast over WebSocket
    await ws_manager.broadcast_turn_result(result)

    state = engine.get_full_state()
    await ws_manager.broadcast_graph_update(state.get("graph", {}))

    if engine.game_over:
        await ws_manager.broadcast_game_over({
            "result": engine.game_result,
            "turn": engine.turn,
        })

    return turn_response


@router.get(
    "/game/state",
    response_model=GameStateResponse,
    tags=["game"],
    summary="Get current game state",
)
async def get_state() -> GameStateResponse:
    """Return the full current game state."""
    engine = get_engine()
    state = engine.get_full_state()
    return GameStateResponse(**state)


@router.get(
    "/game/actions",
    response_model=ActionCatalogResponse,
    tags=["game"],
    summary="Get universal action catalog",
)
async def get_action_catalog() -> ActionCatalogResponse:
    """Return all 27 universal actions grouped by category."""
    categories: dict[str, list[dict]] = {cat: [] for cat in ACTION_CATEGORIES}
    for action_id, info in UNIVERSAL_ACTIONS.items():
        cat = info["category"]
        categories.setdefault(cat, []).append({
            "id": action_id,
            "label": info["label"],
            "category": cat,
            "base_ap": info["base_ap"],
        })
    return ActionCatalogResponse(
        categories=categories,
        total_actions=len(UNIVERSAL_ACTIONS),
    )


@router.get(
    "/game/session",
    response_model=SessionInfoResponse,
    tags=["game"],
    summary="Get session info",
)
async def get_session_info() -> SessionInfoResponse:
    """Return metadata about the current session."""
    info = session_mgr.get_session_info()
    return SessionInfoResponse(**info)


# ── Quest endpoints ───────────────────────────────────────────────────────────


@router.get(
    "/quest/graph",
    response_model=GraphDataResponse,
    tags=["quest"],
    summary="Get MDP graph data",
)
async def get_quest_graph() -> GraphDataResponse:
    """Return MDP graph data formatted for Cytoscape.js rendering."""
    engine = get_engine()
    current_cp = engine.quest_manager.current_checkpoint
    completed = engine.quest_manager.completed_checkpoints
    graph = engine.mdp.to_graph_data(current_cp, completed)
    return GraphDataResponse(**graph)


@router.get(
    "/quest/progress",
    response_model=QuestProgressResponse,
    tags=["quest"],
    summary="Get quest progress",
)
async def get_quest_progress() -> QuestProgressResponse:
    """Return a quest progress snapshot."""
    engine = get_engine()
    progress = engine.quest_manager.get_quest_progress()
    return QuestProgressResponse(**progress)


# ── NPC endpoints ─────────────────────────────────────────────────────────────


@router.get(
    "/npc/list",
    response_model=NPCListResponse,
    tags=["npc"],
    summary="List all NPCs",
)
async def list_npcs() -> NPCListResponse:
    """Return a summary of every NPC in the game."""
    engine = get_engine()
    summaries: list[NPCSummary] = []
    for uid, npc in engine.npc_registry.items():
        summaries.append(
            NPCSummary(
                npc_uid=npc.npc_uid,
                name=npc.name,
                archetype=npc.archetype,
                location=npc.location,
                status=npc.status,
                reputation=engine.player.get_reputation(uid),
                reputation_label=engine.player.get_reputation_label(uid),
            )
        )
    return NPCListResponse(npcs=summaries)


@router.get(
    "/npc/{npc_uid}",
    response_model=NPCDetailResponse,
    tags=["npc"],
    summary="Get NPC detail",
)
async def get_npc_detail(npc_uid: str) -> NPCDetailResponse:
    """Return detailed information about a specific NPC."""
    engine = get_engine()
    npc = engine.npc_registry.get(npc_uid)
    if npc is None:
        raise HTTPException(status_code=404, detail=f"NPC not found: {npc_uid}")
    return NPCDetailResponse(
        npc_uid=npc.npc_uid,
        name=npc.name,
        archetype=npc.archetype,
        location=npc.location,
        status=npc.status,
        personality=npc.personality,
        reputation=engine.player.get_reputation(npc_uid),
        reputation_label=engine.player.get_reputation_label(npc_uid),
        quest_critical=npc.quest_critical,
        current_hp=npc.current_hp,
        max_hp=npc.max_hp,
        stats=npc.stats,
        conversation_history=npc.conversation_history[-5:],  # last 5 exchanges
    )


# ── LLM endpoint ─────────────────────────────────────────────────────────────


@router.get(
    "/llm/status",
    response_model=LLMStatusResponse,
    tags=["llm"],
    summary="Check LLM availability",
)
async def get_llm_status() -> LLMStatusResponse:
    """Return provider-aware LLM connectivity and config status."""
    engine = get_engine(raise_on_missing=False)
    if engine is not None:
        status = engine.llm.get_status()
        return LLMStatusResponse(**status)
    # No session — report based on config
    from backend.config import LLM_API_BASE_URL, LLM_ENABLED, LLM_MODEL_NAME, LLM_PROVIDER

    return LLMStatusResponse(
        available=False,
        enabled=LLM_ENABLED,
        provider=LLM_PROVIDER,
        model_name=LLM_MODEL_NAME,
        base_url=LLM_API_BASE_URL,
        calls_this_minute=0,
        max_calls=20,
        model_path="",
    )


# ── Save / Load endpoints ────────────────────────────────────────────────────


@router.post(
    "/save",
    tags=["save"],
    summary="Save game to slot",
)
async def save_game(req: SaveRequest) -> dict:
    """Save the current game state to the specified slot."""
    engine = get_engine()
    filepath = engine.save_game(slot=req.slot)
    return {
        "message": f"Game saved to slot '{req.slot}'.",
        "filepath": filepath,
        "turn": engine.turn,
    }


@router.post(
    "/load",
    response_model=GameStateResponse,
    tags=["save"],
    summary="Load game from file",
)
async def load_game(req: LoadRequest) -> GameStateResponse:
    """Load a previously saved game and return the restored state."""
    engine = get_engine()
    try:
        state = engine.load_game(filepath=req.filepath)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load save: {exc}")

    # Broadcast restored state
    await ws_manager.broadcast({"type": "state_sync", "data": state})

    return GameStateResponse(**state)


@router.get(
    "/saves",
    response_model=SaveListResponse,
    tags=["save"],
    summary="List save files",
)
async def list_saves() -> SaveListResponse:
    """Return metadata for all available save files."""
    engine = get_engine(raise_on_missing=False)
    if engine is not None:
        saves = engine.get_save_list()
    else:
        # No active session — scan directory directly
        import json
        from backend.config import SAVES_DIR

        saves: list[dict] = []
        if SAVES_DIR.exists():
            for path in sorted(SAVES_DIR.glob("save_*.json")):
                if path.suffix == ".backup":
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    saves.append({
                        "filename": path.name,
                        "filepath": str(path),
                        "slot": path.stem.replace("save_", ""),
                        "turn": data.get("turn", 0),
                        "timestamp": data.get("timestamp", ""),
                        "player_name": data.get("player", {}).get("name", "Unknown"),
                    })
                except (json.JSONDecodeError, OSError):
                    saves.append({
                        "filename": path.name,
                        "filepath": str(path),
                        "slot": path.stem.replace("save_", ""),
                        "error": "corrupted",
                    })
    return SaveListResponse(saves=saves)


# ── Metrics endpoint ──────────────────────────────────────────────────────────


@router.get(
    "/metrics/summary",
    response_model=MetricsSummaryResponse,
    tags=["metrics"],
    summary="Get session metrics",
)
async def get_metrics_summary() -> MetricsSummaryResponse:
    """Return aggregated session metrics for the research dashboard."""
    engine = get_engine()
    metrics = engine.get_metrics()
    return MetricsSummaryResponse(**metrics)


# ── Events endpoint ───────────────────────────────────────────────────────────


@router.get(
    "/events/recent",
    response_model=EventListResponse,
    tags=["game"],
    summary="Get recent events",
)
async def get_recent_events(
    limit: int = Query(default=20, ge=1, le=100, description="Number of recent events."),
) -> EventListResponse:
    """Return the most recent entries from the event log."""
    engine = get_engine()
    events = engine.event_log.get_recent(count=limit)
    return EventListResponse(
        events=events,
        total=len(engine.event_log),
    )


# ─── Difficulty endpoint ──────────────────────────────────────────────────────

class DifficultyRequest(BaseModel):
    """Payload for changing difficulty."""
    difficulty: str = Field(..., description="Preset name: 'easy', 'normal', or 'hard'.")


class DifficultyResponse(BaseModel):
    """Response after difficulty change."""
    preset: str
    message: str


@router.post(
    "/difficulty",
    response_model=DifficultyResponse,
    tags=["game"],
    summary="Change difficulty preset",
)
async def set_difficulty(req: DifficultyRequest) -> DifficultyResponse:
    """Change the game difficulty preset."""
    engine = get_engine()
    old_preset = engine.difficulty.preset
    engine.difficulty.apply_preset(req.difficulty)
    logger.info("Difficulty changed: %s → %s", old_preset, engine.difficulty.preset)
    return DifficultyResponse(
        preset=engine.difficulty.preset,
        message=f"Difficulty changed from {old_preset} to {engine.difficulty.preset}",
    )


# ─── Shock endpoints ─────────────────────────────────────────────────────────


@router.get(
    "/shocks/active",
    tags=["shocks"],
    summary="Get active shocks",
)
async def get_active_shocks() -> dict:
    """Return currently active shocks and their state."""
    engine = get_engine()
    return {
        "active_shocks": engine.shock_manager.get_active_shocks(),
        "shock_timeline": engine.shock_manager.get_shock_timeline(),
        "reward_modifier": round(engine.shock_manager.get_reward_modifier(), 4),
        "adaptation_pressure": round(engine.shock_manager.get_adaptation_pressure(), 4),
    }


class TriggerShockRequest(BaseModel):
    """Payload for manually triggering a shock."""
    shock_type: str = Field(..., description="Shock type key from SHOCK_CATALOG.")
    source: str = Field(default="player", description="Trigger source.")
    scope: str = Field(default="village", description="'village' or 'location'.")
    target_location: str | None = Field(default=None, description="Required if scope='location'.")
    duration: int | None = Field(default=None, description="Override default duration.")


@router.post(
    "/shocks/trigger",
    tags=["shocks"],
    summary="Trigger a shock",
)
async def trigger_shock(req: TriggerShockRequest) -> dict:
    """Manually activate a shock (for research experiments)."""
    engine = get_engine()
    shock = engine.shock_manager.activate_shock(
        shock_type=req.shock_type,
        source=req.source,
        scope=req.scope,
        target_location=req.target_location,
        duration=req.duration,
        turn=engine.turn,
    )
    if shock is None:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to activate shock '{req.shock_type}'. Check type, max active, or duplicates.",
        )
    return {
        "message": f"Shock '{req.shock_type}' activated.",
        "shock": shock,
    }


# ─── Export endpoint ──────────────────────────────────────────────────────────

@router.get(
    "/export",
    tags=["save"],
    summary="Export full game log as JSON",
)
async def export_game_log() -> dict:
    """Export the full event log and game state for research analysis."""
    engine = get_engine()
    return {
        "version": GAME_VERSION,
        "turn": engine.turn,
        "player": engine.player.to_dict() if hasattr(engine.player, 'to_dict') else {},
        "events": engine.event_log.entries,
        "metrics": engine.get_metrics() if hasattr(engine, 'get_metrics') else {},
        "difficulty": engine.difficulty.to_dict(),
    }


# ─── Playthrough Log endpoints ───────────────────────────────────────────────


class PlaythroughLogResponse(BaseModel):
    """Response containing playthrough log records."""
    session_id: str
    log_path: str
    total_records: int
    records: list[dict]


@router.get(
    "/playthrough-log",
    response_model=PlaythroughLogResponse,
    tags=["metrics"],
    summary="Get structured playthrough log",
)
async def get_playthrough_log(
    turns_only: bool = Query(False, description="If true, return only turn records"),
    last_n: int = Query(0, description="Return only the last N records (0 = all)"),
) -> PlaythroughLogResponse:
    """Retrieve the structured per-turn playthrough log for research analysis.

    Each record contains player input, system response, and world state snapshot.
    """
    engine = get_engine()
    pt = engine.playthrough_logger

    if turns_only:
        records = pt.get_turn_records()
    else:
        records = pt.get_all_records()

    if last_n > 0:
        records = records[-last_n:]

    return PlaythroughLogResponse(
        session_id=pt.session_id,
        log_path=pt.get_log_path(),
        total_records=len(records),
        records=records,
    )


@router.get(
    "/playthrough-log/turn/{turn_number}",
    tags=["metrics"],
    summary="Get playthrough log for a specific turn",
)
async def get_playthrough_turn(turn_number: int) -> dict:
    """Get the detailed log record for a specific turn number."""
    engine = get_engine()
    records = engine.playthrough_logger.get_turn_records()

    for record in records:
        if record.get("turn") == turn_number:
            return record

    raise HTTPException(status_code=404, detail=f"No log record for turn {turn_number}")


@router.get(
    "/playthrough-log/download",
    tags=["metrics"],
    summary="Download raw JSONL playthrough log",
)
async def download_playthrough_log() -> dict:
    """Return the raw JSONL content for download/archival."""
    engine = get_engine()
    pt = engine.playthrough_logger

    records = pt.get_all_records()
    return {
        "session_id": pt.session_id,
        "format": "jsonl",
        "log_path": pt.get_log_path(),
        "records": records,
    }


# ─── Phase 6: Analytics & Research Endpoints ─────────────────────────────────


class TimeseriesResponse(BaseModel):
    """Complete time-series data for the research analytics dashboard."""

    reward_series: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-NPC reward time-series (turns, individual, community, penalty, total).",
    )
    community_reward_series: dict[str, Any] = Field(
        default_factory=dict,
        description="Village-level community reward averaged per turn.",
    )
    social_welfare_series: dict[str, Any] = Field(
        default_factory=dict,
        description="Social welfare index over turns.",
    )
    cooperation_series: dict[str, Any] = Field(
        default_factory=dict,
        description="Global cooperation index over turns.",
    )
    cooperation_index: dict[str, Any] = Field(
        default_factory=dict,
        description="Current cooperation index: global, per-role, per-NPC.",
    )
    policy_entropy: dict[str, float] = Field(
        default_factory=dict,
        description="Per-NPC policy entropy (action distribution uniformity).",
    )
    action_distribution: dict[str, Any] = Field(
        default_factory=dict,
        description="Action distribution global, per-role, early/late windows, and shift.",
    )
    shock_timeline: list[dict] = Field(
        default_factory=list,
        description="Active and expired shocks with timing data.",
    )
    shock_responses: list[dict] = Field(
        default_factory=list,
        description="Shock response and recovery curves per shock event.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reward_series": {},
                    "community_reward_series": {"turns": [], "avg_community": []},
                    "social_welfare_series": {"turns": [], "welfare_index": []},
                    "cooperation_series": {"turns": [], "global_cooperation": []},
                    "cooperation_index": {"global": 0.5, "per_role": {}, "per_npc": {}},
                    "policy_entropy": {},
                    "action_distribution": {},
                    "shock_timeline": [],
                    "shock_responses": [],
                }
            ]
        }
    }


class CooperationResponse(BaseModel):
    """Lightweight cooperation index snapshot."""

    global_cooperation: float = Field(description="Village-wide average cooperation tendency.")
    per_role: dict[str, float] = Field(
        default_factory=dict,
        description="Average cooperation per NPC archetype.",
    )
    per_npc: dict[str, float] = Field(
        default_factory=dict,
        description="Cooperation tendency per individual NPC.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "global_cooperation": 0.5200,
                    "per_role": {"elder": 0.55, "farmer": 0.50},
                    "per_npc": {"elder_m8b2": 0.55},
                }
            ]
        }
    }


class ExperimentBundleResponse(BaseModel):
    """Complete experiment data bundle for offline research analysis."""

    metadata: dict[str, Any] = Field(description="Experiment metadata.")
    reward_series: dict = Field(default_factory=dict)
    community_reward_series: dict = Field(default_factory=dict)
    social_welfare_series: dict = Field(default_factory=dict)
    cooperation_series: dict = Field(default_factory=dict)
    cooperation_index: dict = Field(default_factory=dict)
    policy_entropy: dict = Field(default_factory=dict)
    action_distribution: dict = Field(default_factory=dict)
    shock_timeline: list = Field(default_factory=list)
    shock_responses: list = Field(default_factory=list)
    adaptation_snapshot: dict = Field(default_factory=dict)


@router.get(
    "/metrics/timeseries",
    response_model=TimeseriesResponse,
    tags=["analytics"],
    summary="Get research time-series data",
)
async def get_timeseries() -> TimeseriesResponse:
    """Return all time-series data for the research analytics dashboard.

    Includes per-NPC reward series, community reward, social welfare index,
    cooperation index, policy entropy, action distribution, and shock analysis.
    """
    engine = get_engine()
    npc_registry = engine.npc_registry
    shock_timeline = engine.shock_manager.get_shock_timeline()
    event_entries = engine.event_log.entries

    return TimeseriesResponse(
        reward_series=compute_reward_series(npc_registry),
        community_reward_series=compute_community_reward_series(npc_registry),
        social_welfare_series=compute_social_welfare_series(npc_registry),
        cooperation_series=compute_cooperation_series(npc_registry),
        cooperation_index=compute_cooperation_index(npc_registry),
        policy_entropy=compute_policy_entropy(npc_registry),
        action_distribution=compute_action_distribution(event_entries, npc_registry),
        shock_timeline=shock_timeline,
        shock_responses=compute_shock_response(npc_registry, shock_timeline),
    )


@router.get(
    "/metrics/cooperation",
    response_model=CooperationResponse,
    tags=["analytics"],
    summary="Get cooperation index",
)
async def get_cooperation() -> CooperationResponse:
    """Return the current cooperation index: global, per-role, and per-NPC."""
    engine = get_engine()
    coop = compute_cooperation_index(engine.npc_registry)
    return CooperationResponse(
        global_cooperation=coop["global"],
        per_role=coop["per_role"],
        per_npc=coop["per_npc"],
    )


@router.get(
    "/metrics/experiment",
    response_model=ExperimentBundleResponse,
    tags=["analytics"],
    summary="Export full experiment bundle",
)
async def get_experiment_bundle() -> ExperimentBundleResponse:
    """Build and return a complete experiment data bundle for offline analysis.

    Includes all time-series, indices, distributions, shock analysis,
    adaptation snapshots, and metadata (seed, difficulty, LLM provider mode).
    """
    engine = get_engine()
    event_entries = engine.event_log.entries
    bundle = build_experiment_bundle(engine, event_entries)
    return ExperimentBundleResponse(**bundle)
