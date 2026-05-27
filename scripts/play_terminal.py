"""Terminal game client for the Medieval Village RPG.

Plays the game via REST API calls so fixes can be verified
without a browser. Supports both button actions and free text.

Usage:
    uv run python scripts/play_terminal.py [--base-url http://localhost:8000]
"""

import argparse
import sys
import httpx

DEFAULT_BASE = "http://localhost:8000"


def create_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=30.0)


def start_game(client: httpx.Client, player_name: str = "Traveler") -> dict:
    resp = client.post("/api/game/new", json={
        "seed": 42,
        "difficulty": "normal",
        "max_turns": 200,
        "player_name": player_name,
    })
    resp.raise_for_status()
    return resp.json()


def get_state(client: httpx.Client) -> dict:
    resp = client.get("/api/game/state")
    resp.raise_for_status()
    return resp.json()


def get_all_npcs(client: httpx.Client) -> list:
    resp = client.get("/api/npc/list")
    resp.raise_for_status()
    return resp.json().get("npcs", [])


def send_action(client: httpx.Client, payload: dict) -> dict:
    resp = client.post("/api/game/action", json=payload)
    resp.raise_for_status()
    return resp.json()


def print_status(state: dict):
    quest = state.get("quest_state") or state.get("quest") or {}
    location = state.get("location") or {}
    loc_name = location.get("name") or location.get("id") or state.get("current_location", "?")
    npcs = state.get("npcs_here") or []
    inventory = state.get("inventory") or []
    player = state.get("player") or {}

    print(f"\n{'=' * 60}")
    print(f"  Turn: {state.get('turn', '?')}  |  Location: {loc_name}")
    print(f"  HP: {player.get('health', '?')}/{player.get('max_health', '?')}  |  "
          f"AP: {player.get('stamina', '?')}/{player.get('max_stamina', '?')}  |  "
          f"Rep: {player.get('global_reputation', '?')}")
    print(f"  Quest: Stage {quest.get('current_stage', '?')} / "
          f"CP {quest.get('current_checkpoint', '?')}")
    if quest.get("description") or quest.get("stage_description"):
        desc = quest.get("description") or quest.get("stage_description", "")
        if len(desc) > 100:
            desc = desc[:100] + "..."
        print(f"  Desc: {desc}")
    if npcs:
        names = [f"{n.get('name', '?')} ({n.get('reputation', 0):+d})" for n in npcs]
        print(f"  NPCs here: {', '.join(names)}")
    if inventory:
        items = [f"{i.get('name', i.get('id', '?'))}" for i in inventory]
        print(f"  Inventory: {', '.join(items)}")
    print(f"{'=' * 60}")


def print_result(result: dict):
    narration = result.get("narration")
    dialogue = result.get("dialogue")
    speaker = result.get("dialogue_speaker")
    events = result.get("events") or result.get("new_events") or []
    quest_update = result.get("quest_update")
    game_over = result.get("game_over")
    game_result = result.get("game_result")
    action_result = result.get("action_result") or {}

    if narration:
        print(f"\n  >> {narration}")
    if dialogue:
        label = speaker or "NPC"
        print(f"\n  [{label}]: \"{dialogue}\"")
    if action_result.get("success") is False:
        reason = action_result.get("reason") or action_result.get("message") or ""
        print(f"\n  [FAILED] {reason}")
    for evt in events:
        text = evt if isinstance(evt, str) else evt.get("text", evt.get("outcome", ""))
        if text:
            print(f"  * {text}")
    if quest_update:
        print(f"  [QUEST] {quest_update}")
    if game_over:
        print(f"\n  === GAME OVER: {game_result} ===")


def parse_command(raw: str) -> dict | None:
    """Parse user input into an API payload.

    /action_id [target_npc] [--item item_id]  →  button action
    /state                                     →  print state
    /npcs                                      →  list all NPCs
    /quit                                      →  exit
    anything else                              →  free text
    """
    stripped = raw.strip()
    if not stripped:
        return None

    if stripped.startswith("/"):
        parts = stripped[1:].split()
        cmd = parts[0].lower()

        if cmd in ("quit", "q", "exit"):
            return {"_cmd": "quit"}
        if cmd in ("state", "s"):
            return {"_cmd": "state"}
        if cmd in ("npcs", "n"):
            return {"_cmd": "npcs"}
        if cmd in ("help", "h", "?"):
            return {"_cmd": "help"}

        payload = {"source": "button", "action_id": cmd}
        i = 1
        while i < len(parts):
            if parts[i] == "--item" and i + 1 < len(parts):
                payload["target_item"] = parts[i + 1]
                i += 2
            elif parts[i] == "--loc" and i + 1 < len(parts):
                payload["target_location"] = parts[i + 1]
                i += 2
            elif "target_npc" not in payload:
                payload["target_npc"] = parts[i]
                i += 1
            else:
                i += 1
        return payload

    return {"source": "text", "text": stripped}


def print_help():
    print("""
  Commands:
    /greet <npc_uid>              Greet an NPC
    /talk <npc_uid>               Talk to an NPC
    /present_item <npc> --item X  Present item to NPC
    /sneak                        Sneak past
    /persuade <npc_uid>           Persuade an NPC
    /move_to --loc <location_id>  Move to location
    /pick_up --item <item_id>     Pick up item
    /look                         Look around
    /wait                         Wait a turn
    /rest                         Rest
    /attack <npc_uid>             Attack an NPC
    <any text>                    Free-text input (sent to LLM parser)

  Meta:
    /state  or /s     Print current game state
    /npcs   or /n     List all NPCs (all locations)
    /help   or /h     Show this help
    /quit   or /q     Exit
""")


def main():
    parser = argparse.ArgumentParser(description="Terminal RPG client")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--player-name", default="Traveler")
    args = parser.parse_args()

    client = create_client(args.base_url)

    try:
        client.get("/api/game/state")
    except httpx.ConnectError:
        print(f"Cannot connect to {args.base_url}. Is the server running?")
        print("Start it with: uv run uvicorn backend.main:app --reload")
        sys.exit(1)

    print("Starting new game...")
    state = start_game(client, args.player_name)
    print_status(state)
    print_help()

    while True:
        try:
            raw = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        payload = parse_command(raw)
        if payload is None:
            continue

        if "_cmd" in payload:
            cmd = payload["_cmd"]
            if cmd == "quit":
                print("Bye!")
                break
            elif cmd == "state":
                state = get_state(client)
                print_status(state)
                continue
            elif cmd == "npcs":
                npcs = get_all_npcs(client)
                print(f"\n  All NPCs ({len(npcs)}):")
                for npc in npcs:
                    rep = npc.get("reputation", 0)
                    print(f"    {npc.get('name', '?'):20s}  "
                          f"{npc.get('archetype', ''):12s}  "
                          f"@ {npc.get('location', '?'):20s}  "
                          f"rep: {rep:+d}")
                continue
            elif cmd == "help":
                print_help()
                continue

        try:
            result = send_action(client, payload)
            print_result(result)
            state = result.get("state") or get_state(client)
            print_status(state)
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            print(f"  [ERROR] {e.response.status_code}: {detail or e}")


if __name__ == "__main__":
    main()
