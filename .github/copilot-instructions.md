# Copilot Instructions — MVP Research Project (Game/Simulation with MDP + RL + LLM)

> These instructions provide full context for the project. Always consult `Plan.md` for the canonical specification when in doubt.

---

## Project Overview

This is a **single-player, single-session MVP** research game combining a hierarchical MDP quest system, NPC reinforcement learning agents, and optional local LLM integration. The server supports one active game session at a time. Concurrent players/sessions are out of scope.

### Core Features (Priority Order)

1. Quest modeled as a **hierarchical MDP** (macro stages → micro checkpoints)
2. **Core systems**: Health, Stamina/AP, Inventory, Per-NPC Reputation, Combat, Equipment
3. **Visual MDP graph** (Cytoscape.js nodal view)
4. **Adaptive system**: dynamic checkpoint generation + player nudging
5. **NPC RL agents**: 6 NPCs with tabular Q-learning, all sharing a universal action space
6. **Local LLM integration** (GGUF, qwen3-4B) for quest/checkpoint/dialogue generation (optional, with graceful fallback)

---

## Tech Stack — Do NOT Deviate

| Layer | Technology | Notes |
|-------|-----------|-------|
| Backend / Engine | **Python 3.11+ / FastAPI** | Async support required; all LLM calls use `asyncio.to_thread()` |
| Frontend | **HTML + CSS + Vanilla JS + Cytoscape.js** | No frameworks (React, Vue, etc.). Modern minimal dark theme. |
| NLP | **spaCy** (`en_core_web_md`) | Tokenization, lemmatization, word vectors, NER |
| RL Engine | **NumPy-based tabular Q-learning** | No deep RL libraries. 180 states × 27 actions = 4,860 Q-table entries per NPC. |
| LLM Runtime | **llama-cpp-python** | Runs GGUF models locally |
| LLM Model | **qwen3-4B-q4_k_m.gguf** | 4096 token context window |
| Persistence | **JSON files** | No database |
| Communication | **REST API + WebSocket** | REST for actions/save/load; WebSocket for real-time MDP graph updates |

---

## Architecture Principles

### Universal Action Space

The game uses a **single universal action catalog of 27 actions** shared by ALL entities (player and NPCs). There are zero per-checkpoint or per-location restrictions on action availability.

**Actions are always available.** Preconditions and context modifiers determine outcomes, not availability. The player is never told "you can't do that here" — instead, actions produce contextually appropriate outcomes. Hard-fail preconditions (e.g., no target present) narrate why and cost 0 AP.

The 27 universal actions (by category):
- **Navigation**: `move_to` (3 AP)
- **Exploration**: `look` (1), `search` (5), `examine` (2)
- **Social**: `talk` (1), `greet` (1), `ask_info` (1), `persuade` (2), `trade` (2), `give_item` (1), `deceive` (2), `intimidate` (2)
- **Combat**: `attack` (10), `defend` (5), `flee` (5)
- **Stealth**: `sneak` (5), `hide` (3), `steal` (5)
- **Utility**: `pick_up` (1), `use_item` (2), `eat` (1), `rest` (0), `wait` (0), `drop_item` (0), `status` (0), `equip` (1), `work` (5)

### Action Resolution Pipeline

Every action goes through: **Precondition Check → Context Evaluation → Outcome Resolution**. This applies uniformly to both player and NPC actions.

### Two Input Modes

Both modes produce the same `ParsedInput` object:
1. **Action Palette** — categorized buttons, direct 1:1 action mapping, defaults to neutral emotion/social
2. **Free-Text Input** — NLP pipeline extracts 3 dimensions (emotion, intent, social), maps to universal action catalog

### Layered NLP Strategy

1. **Primary**: Keyword/synonym dictionaries (fast, deterministic)
2. **Secondary**: spaCy `doc.similarity()` as tiebreaker
3. **Tertiary** (optional): LLM for highest-accuracy 3D analysis

Pre-compute and cache action vectors at startup (`ACTION_VECTORS: dict[str, np.ndarray]`).

---

## Coding Conventions

### Python (Backend)

- **Python 3.11+** — use modern syntax (`match/case`, type hints, `|` union types)
- **Type hints on all functions** — parameters and return types
- **Pydantic models** for request/response schemas and data validation
- **Async endpoints** — all FastAPI endpoints that may invoke LLM must be `async def`
- **LLM calls** MUST be wrapped with `asyncio.to_thread()` to avoid blocking the event loop
- **Structured JSON logging** — use Python `logging` with `JSONFormatter`; never `print()`
- **All randomness seeded** — use `random` and `np.random` seeded from `MASTER_SEED`; never `os.urandom`
- **Constants in `config.py`** — all tunable parameters centralized, not scattered in code
- **Docstrings** on all public functions
- **No circular imports** — respect the module hierarchy: `engine/` → `quest/`, `npc/`, `player/`; `llm/` is called by others but never imports game modules

### FastAPI Specifics

- Tag all endpoints by category: `game`, `quest`, `npc`, `llm`, `save`, `metrics`
- Every endpoint must have a docstring and Pydantic response model
- Add example values to Pydantic models via `model_config` / `json_schema_extra`
- Swagger at `/docs`, ReDoc at `/redoc`

### JavaScript (Frontend)

- **Vanilla JS only** — no frameworks, no build tools, no npm
- **ES6+ modules** — use `<script type="module">`
- **CSS custom properties** for theming (all colors from the defined palette)
- **Cytoscape.js** for graph visualization — no D3 or other graph libraries
- **WebSocket** for real-time updates; REST for player actions and save/load
- Exponential backoff reconnection (1s, 2s, 4s, 8s, 16s — max 5 attempts)

### CSS / UI Design

Follow **modern minimal** design:
- Dark theme: background `#0a0a0f`, surface `#14141b`, surface-2 `#1e1e28`
- Text: primary `#e8e8ed`, secondary `#8b8b96`, muted `#5a5a66`
- Borders: `rgba(255, 255, 255, 0.06)` only
- Color as signal ONLY: green `#2ECC71` = completed, amber `#F39C12` = current, orange `#E67E22` = dynamic, blue `#4A90D9` = static, red `#E74C3C` = danger
- No gradients, no heavy shadows, no rounded corners > 8px
- CSS Grid for layout; generous whitespace replaces borders/dividers
- Transitions: 150ms ease; pulse animation only for current MDP node
- System font stack: `Inter` / `-apple-system` / `system-ui`; monospace for stats/data

---

## Data Model Contracts

### ParsedInput (unified for both button and text input)

```python
ParsedInput = {
    "source": "button" | "text",
    "raw_text": str | None,
    "action_id": str | None,          # from universal catalog
    "target_npc": str | None,         # resolved NPC UID
    "target_item": str | None,
    "target_location": str | None,
    "confidence": float,              # 1.0 for buttons, 0.0–1.0 for text
    "emotion": str,                   # neutral/angry/friendly/fearful/curious/threatening
    "intent": str,
    "social": str,                    # neutral/polite/rude/deceptive/honest/intimidating
}
```

### Player State

```python
Player = {
    "name": str,
    "health": int,            # 0–100
    "max_health": 100,
    "stamina": int,           # 0–50
    "max_stamina": 50,
    "combat_stats": {
        "base_attack": 8, "base_defense": 3,
        "weapon_modifier": 0, "armor_modifier": 0
    },
    "equipped": {"weapon": str | None, "armor": str | None},
    "reputation": dict,       # {npc_uid: int} each -100 to +100
    "global_reputation": int, # weighted avg (read-only, display)
    "inventory": list,        # max 10 items
    "max_inventory": 10,
    "location": str,
    "quest_state": dict,
}
```

### NPC Instance

Every NPC has a **unique UID** (`{archetype}_{hex}`), deterministic from master seed. The UID is the primary key used everywhere (game state, event log, reputation, save files).

6 MVP NPCs:
- **Elder Maren** (`elder_m8b2`) — elder archetype
- **Farmer Jak** (`farmer_j4a1`) — farmer archetype
- **Tessa** (`tavkeeper_t9c3`) — tavern_keeper archetype
- **Aldric** (`guard_a3f1`) — guard archetype
- **Bryn** (`guard_b7e2`) — guard archetype
- **Old Petra** (`villager_c1d4`) — villager archetype

### Item

```python
Item = {
    "id": str, "name": str,
    "type": "quest" | "consumable" | "equipment" | "misc",
    "description": str, "effects": dict,
    "quest_relevant": bool,   # cannot be discarded if True
    "slot": str | None,       # "weapon" or "armor" for equipment
    "stat_modifiers": dict | None,
}
```

### EventLogEntry

```python
EventLogEntry = {
    "event_id": str, "turn": int, "time_of_day": str,
    "event_type": str,  # player_action/npc_action/random_event/quest_progress/combat/dialogue
    "actor": str, "action": str, "target": str | None,
    "location": str, "outcome": str,
    "effects": dict, "witnesses": list[str],
    "narration": str, "importance": int,  # 1–5
}
```

---

## Key System Specifications

### Hierarchical MDP (Quest)

- **Macro MDP**: Quest stages S1–S7 + terminal S_success / S_fail. γ = 1.0.
- **Micro MDP**: Checkpoints within each stage. γ = 0.95.
- Static checkpoint IDs: `{stage}_{index}` (e.g., `1_1`, `3_2`)
- Dynamic checkpoint IDs: `{stage}_D{counter}` (e.g., `1_D1`, `2_D3`)

### NPC Q-Learning

- State: `(location × time_slot × energy_level × mood_level)` = ~180 states
- Actions: 27 universal actions
- Q-table: 180 × 27 = 4,860 entries
- α = 0.1, γ = 0.9
- ε: 0.5 during pre-training → 0.15 at live play start (turn 20) → decay to 0.05
- **Cold-start**: first 20 turns use fallback schedule; ε-greedy disabled
- **Pre-training**: 100 episodes × 50 turns per NPC, no player present, lightweight mode (narration/logging disabled)
- **Action masking**: hard-fail preconditions → Q-value = -∞ during selection (Q-table retains learned values)
- **Movement resolution**: Q-table selects `move_to`; destination chosen by secondary heuristic (schedule → goal → weighted random by personality)
- **Reward**: R = w_h·Δhappiness + w_i·Δincome + w_hp·Δhealth + w_r·Δreputation (weights must sum to 1.0 per archetype)

### Per-NPC Reputation

Reputation is tracked **per-NPC**, not globally. Range: -100 to +100.
- Direct changes apply to target NPC + witnesses at same location
- **Gossip propagation**: 40% chance per social interaction, 50% decay per hop, max 3 hops, min delta 2
- **Gossip cascade limit**: 1 gossip event per NPC pair per turn
- **Reputation decay**: every 20 turns, all reputations nudge ±1 toward 0
- Thresholds: 50+ = Trusted, 20–49 = Friendly, -19–19 = Neutral, -49–-20 = Suspicious, ≤-50 = Hostile

### Combat

- Hit probability: P(hit) = clamp((atk + weapon + stamina_factor×10) / ((def + armor)×2 + 20), 0.1, 0.95)
- Damage: max(1, base_attack + weapon_mod - base_defense - armor_mod + randint(-3, 3))
- `defend`: next incoming attack -50% damage (5 AP)
- `flee`: 70% success (modified by stamina); failure = free attack at +20% hit
- **NPCs are never permanently killed**: non-critical → incapacitated 20 turns, return hostile; quest-critical → floor at 1 HP, permanently hostile

### Skill Check Formulas

- P(persuade) = clamp(0.5 + rep/200 + social_mod/20, 0.1, 0.9)
- P(deceive) = clamp(0.4 − rep/200 + social_mod/20, 0.05, 0.85)
- P(sneak) = clamp(0.5 + time_bonus/10 − npcs_at_location/10, 0.1, 0.9)
- P(steal) = P(sneak) × 0.8
- P(hide) = clamp(0.6 + time_bonus/10 − npcs_at_location/10, 0.15, 0.95)
- P(search_discovery) = clamp(0.3 + search_count/5 × 0.1, 0.2, 0.8)

### Stamina

- Passive regen: +5 AP/turn, capped at max_stamina (50)
- At 0 AP: only 0-AP actions + `talk`/`greet` at 0 AP (anti-soft-lock); in combat `defend`/`flee` cost 0 AP but `flee` success drops to 40%
- `rest`: +10 AP, requires safe/indoor location or no combat; else resolves as `wait`
- `wait`: advances time, passive perception check (40% base, +20% at social locations)

### Time System

- 1 turn = 1 player action; NPCs also act each turn
- 4 turns = 1 time period; cycle: morning → midday → afternoon → evening → night → morning

### World Map — 5 Locations

```
            [Fields]
               |
[Gate] — [Village Center] — [Elder's House]
               |
            [Tavern]
```

Adjacency strictly enforced; Village Center is the hub. `move_to` to non-adjacent = hard fail, 0 AP.

**Indoor**: Elder's House, Tavern (NPC passive regen +2 HP/turn, player `rest` works, shelter from weather)
**Outdoor**: Gate, Village Center, Fields (no passive regen, weather effects apply)

### Random Events

8 event types (weather_storm, weather_fog, wandering_merchant, theft_event, npc_accident, festival_prep, supply_shortage, lost_item) with trigger conditions, probabilities, and durations. Events inject unpredictability; they modify NPC reward signals temporarily.

### Event Importance Scoring (1–5)

- 5 = Quest-critical (stage transitions, completion/failure)
- 4 = Major (combat, rep changes ≥10, NPC incapacitation)
- 3 = Significant (social with rep ≥5, quest discoveries, random events)
- 2 = Minor (routine social, movement, trade)
- 1 = Trivial (routine NPC actions, look/wait/rest)

### Difficulty Scaling

12 tunable parameters with Easy/Normal/Hard presets. All in `config.py` / `difficulty_presets.json`. Optional adaptive difficulty based on event log analytics.

---

## LLM Integration Rules

### Always Optional

The entire game must work without LLM loaded. Every LLM call has a template/rules fallback.

### Fallback Chain

1. Try LLM with timeout (10s) and validation
2. On failure: retry up to 2 times
3. On exhaustion: use template-based fallback

### Temperature Per Use Case

| Use Case | Temperature |
|----------|------------|
| Free-text input parsing | 0.3–0.5 |
| NPC dialogue | 0.7 |
| Narration enhancement | 0.7 |
| Checkpoint generation | 0.8–0.9 |

### Context Window Budget

Hard rule: **no prompt may exceed 2500 tokens** (leaves ≥1500 for generation). Truncation priority: (1) conversation history to last 3 exchanges, (2) event log to last 3 events, (3) omit inventory details.

### Output Validation (Mandatory)

All LLM outputs MUST be validated before use:
1. Parse JSON (handle markdown code block wrapping)
2. Schema validation (required fields, enum values, length constraints)
3. Clamp numeric values (health: -50..+50, stamina: -20..+20, reputation: -30..+30, mood_change: -3..+3)
4. Validate action references against universal catalog
5. Content filter (strip HTML, system prompt leakage, truncate to 500 chars)

### Rate Limiting

Min 2s between LLM calls, max 20 calls/min. When rate-limited, fall back to templates immediately.

### NPC Dialogue Pipeline

`talk`/`greet`/`ask_info` → resolve target NPC → check scripted dialogue → if no match and LLM loaded: LLM generates in-character → else: archetype generic template fallback. Log to `conversation_history` (cap: 10 per NPC; include last 5 in LLM prompts).

---

## Narration System

### 4-Layer Pipeline

1. **Template narration** — pre-written for all 27 actions × outcome types (success/partial/fail/blocked). Always available.
2. **LLM enhancement** — enriches template with context-aware details if LLM loaded.
3. **Context modifiers** — append time-of-day flavor, weather, NPC mood, witness notes.
4. **Passive perception** — auto-roll on any action at a location or new location entry.

### NPC Action Narration Filtering

- NPC at player's location → full narration
- NPC action importance ≥ 3 → brief notification
- NPC elsewhere, routine → collapsed "Meanwhile..." section
- Gossip about player with delta ≥ 5 → brief notification

---

## Dynamic Checkpoints & Nudging

- **Triggers**: unexpected action outcome, unmatched free text, deviation from expected path, NPC-created situation
- **Generation**: LLM → template fallback (4 templates: unexpected_combat, unexpected_explore, unexpected_social, unexpected_stealth)
- **Nudging**: narrative hints + reward shaping (λ = 0.3); 3 consecutive deviations = explicit hint; 5 = forced convergence
- **Loop detection**: same action_id generates dynamic CP 3 times at same stage → force convergence

---

## Save/Load System

- **Auto-save**: every 5 turns + immediately after quest-relevant actions + before combat
- **Manual save**: up to 3 slots; API: `POST /api/save`, `POST /api/load`, `GET /api/saves`
- **Save file**: single JSON with ALL state (player, NPC registry with Q-tables, event log, active events, dynamic checkpoints, difficulty config, metrics)
- **Backup**: `.backup` copy of last good save; auto-restore on corruption
- **Max auto-saves**: 3 (rotating); max manual saves: 3
- **WebSocket reconnection**: on disconnect, client `GET /api/state` to rebuild UI

---

## Edge Cases — Handle These Explicitly

| Case | Resolution |
|------|-----------|
| Inventory full + quest item | Auto-prompt to drop non-quest item; quest items never lost |
| 0 stamina + forced combat | defend/flee at 0 AP; flee success drops to 40% |
| All NPCs hostile + quest needs help | "Mercy" mechanic after 3 failed attempts: dynamic CP for rep recovery |
| Player-triggered CP loops | Same action 3× at same stage → force convergence |
| Health = 0 during dynamic CP | S_fail; "last chance" if within 2 CPs of main flow (HP → 1) |
| Simultaneous contradictory NPC actions | Priority: combat first, then social; ties broken by reputation then random |
| NPC at invalid location | Hard-fail, replace with `wait`, Q-table penalty -5 |
| LLM generates invalid action | Validate against catalog; unknown → closest spaCy match; if <0.4 → `wait` |
| Save file corruption | Auto-restore from `.backup`; if backup fails → new game with notification |
| Compound text input | Parse first action immediately, queue second for next turn (max queue: 1) |

---

## Project Structure

The canonical directory structure is defined in Plan.md §11. Key modules:

```
MVP/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # All configuration constants
│   ├── engine/              # Game loop, world, events, combat, narration, difficulty
│   ├── quest/               # MDP, quest manager, checkpoints, nudging
│   ├── npc/                 # NPC class, personality, dialogue, interactions, knowledge, RL, schedule
│   ├── player/              # Player state, actions, input parser (spaCy NLP)
│   ├── llm/                 # LLM service, prompts, guardrails, fallback
│   ├── api/                 # REST routes, WebSocket, session management
│   ├── data/                # JSON data files (quests, NPCs, world, config, saves, metrics, logs)
│   └── tools/               # Replay, export, playtest bot
├── frontend/
│   ├── index.html
│   ├── css/                 # style.css, components.css, graph.css
│   └── js/                  # app.js, game.js, graph.js, stats.js, metrics.js, history.js, npc-panel.js, audio.js, api.js
├── models/                  # GGUF files (gitignored)
├── Plan.md
└── requirements.txt
```

---

## Implementation Phase Order

1. **Phase 1a** — Backend Core (player, world, actions, combat, narration, event log, difficulty, input parser)
2. **Phase 1b** — Frontend + API (REST/WS, dark theme UI, action palette, text input)
3. **Phase 2** — Quest MDP (stages, checkpoints, transitions, quest manager, quest JSON)
4. **Phase 3** — MDP Visualization (Cytoscape.js graph, real-time updates)
5. **Phase 4** — Dynamic Checkpoints + Nudging + Random Events
6. **Phase 5** — NPC RL Agents (archetypes, Q-learning, NPC-NPC interactions, gossip, pre-training)
7. **Phase 6** — LLM Integration (llama-cpp-python, prompts, guardrails, retry logic)
8. **Phase 7** — Polish & Demo (metrics dashboard, save/load UI, keyboard shortcuts, research tooling)
9. **Phase 8** — Research Tooling (replay, export, playtest bot, undo, A/B profiles)

---

## Critical Constraints — Never Violate

1. **All 27 actions always available** — never restrict the action catalog per location/checkpoint
2. **NPC UIDs are the primary key** — never reference NPCs by name alone in code; always use UID
3. **Archetype reward weights must sum to 1.0** — validate at load time
4. **LLM is always optional** — every code path that calls LLM must have a working template fallback
5. **No blocking LLM calls** — always `asyncio.to_thread()` or `run_in_executor()`
6. **All randomness from seeded sources** — `random` and `np.random` only; seed from `MASTER_SEED`
7. **Per-NPC reputation, not global** — global_reputation is a derived read-only display value
8. **NPCs never permanently die** — incapacitation only (see §5.6)
9. **Validate all LLM output** — JSON schema + value clamping + content filter before applying to game state
10. **Event log max 500 detailed entries** — prune routine events; keep importance ≥ 2 forever
11. **Conversation history max 10 per NPC** — include last 5 in LLM prompts
12. **No npm, no React, no Vue** — vanilla JS + Cytoscape.js only
13. **Session turn limit** — default MAX_TURNS = 200; configurable; auto-save before ending
