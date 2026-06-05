# Copilot Instructions

## Tech Stack

| Layer         | Technology                                                                       |
|---------------|----------------------------------------------------------------------------------|
| Backend       | Python 3.12, FastAPI, async throughout                                           |
| Frontend      | HTML + CSS + Vanilla JS + Cytoscape.js (no frameworks)                           |
| NLP           | spaCy (`en_core_web_md`)                                                         |
| RL            | NumPy tabular Q-learning (no deep RL libraries)                                  |
| LLM           | httpx HTTP adapter to external providers (OpenRouter, Ollama, OpenAI-compatible) |
| Persistence   | JSON files (no database)                                                         |
| Deps          | uv + pyproject.toml                                                              |
| Communication | REST API + WebSocket                                                             |

## Project Structure

```
backend/
  main.py           # FastAPI entry point
  config.py         # All tunable constants, env-var overrides
  engine/           # Game loop, world, events, combat, narration, shocks
  quest/            # Hierarchical MDP, quest manager, checkpoints, nudging
  npc/              # NPC class, personality, dialogue, Q-learning, schedule
  player/           # Player state, input parser (spaCy NLP)
  llm/              # HTTP adapter, prompts, guardrails, template fallback
  api/              # REST routes, WebSocket, session management
  tools/            # Playtest bot, replay, narrative export
  data/             # JSON data files (quests, NPCs, world, config, saves)
  tests/            # Integration and feature tests
frontend/
  index.html        # Single-page UI
  css/              # Dark theme
  js/               # Vanilla JS + Chart.js + Cytoscape.js
```

## Coding Conventions

### Python

- Python 3.12 syntax: `match/case`, `|` union types, full type hints on all functions
- Pydantic models for request/response schemas
- All FastAPI endpoints that may call LLM must be `async def`
- LLM calls wrapped with `asyncio.to_thread()` -- never block the event loop
- Structured JSON logging via `logging` -- never `print()`
- All randomness seeded from `MASTER_SEED` via `random` and `np.random`
- Constants centralized in `config.py`
- Docstrings on all public functions
- No circular imports: `engine/` may import `quest/`, `npc/`, `player/`; `llm/` never imports game modules
- Tag all FastAPI endpoints by category: `game`, `quest`, `npc`, `llm`, `save`, `metrics`

### JavaScript

- Vanilla JS only -- no npm, no React, no Vue, no build tools
- ES6+ modules with `<script type="module">`
- CSS custom properties for theming
- Cytoscape.js for graph visualization
- WebSocket with exponential backoff reconnection

## Critical Constraints

1. **All 28 actions always available** -- never restrict the action catalog per location/checkpoint
2. **NPC UIDs are the primary key** -- never reference NPCs by name alone in code
3. **LLM is always optional** -- every code path that calls LLM must have a working template fallback
4. **No blocking LLM calls** -- always `asyncio.to_thread()` or `run_in_executor()`; HTTP-only via httpx
5. **All randomness from seeded sources** -- `random` and `np.random` only; seed from `MASTER_SEED`
6. **Per-NPC reputation, not global** -- `global_reputation` is a derived read-only display value
7. **NPCs never permanently die** -- incapacitation only
8. **Validate all LLM output** -- JSON schema + value clamping + content filter before applying to game state
9. **No npm, no React, no Vue** -- vanilla JS + Cytoscape.js only
10. **Archetype reward weights must sum to 1.0** -- validate at load time

## LLM Integration Rules

- Every LLM call: try provider (10s timeout) -> retry up to 2x -> template fallback
- No prompt may exceed 2500 tokens
- Rate limit: min 2s between calls, max 20 calls/min
- All LLM output validated (JSON parse, schema check, value clamping, content filter)
- Conversation history: max 10 per NPC, include last 5 in prompts

## Key Data Contracts

- **Universal action space**: 28 actions shared by player and all NPCs
- **NPC state space**: 225 states (5 dimensions x 3-5 bins) x 28 actions = 6,300 Q-values per NPC
- **Reward**: `R_i(t) = P_i(t) + G_i(t) + lambda_i(t) * C(t)` with dynamic lambda driven by cooperation_tendency
- **6 NPCs**: elder_m8b2, farmer_j4a1, tavkeeper_t9c3, guard_a3f1, guard_b7e2, villager_c1d4
