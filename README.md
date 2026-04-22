# MVP Research Game

Single-session RL playground where NPC agents learn cooperative behavior through tabular Q-learning, adaptive personalities, and role-grounded action selection.

## Quick Start

```bash
uv sync                                          # install deps
uv run uvicorn backend.main:app --reload         # start server
uv run pytest backend/tests/ -v                  # run tests (41 passing)
```

## Project Structure

```
backend/
├── main.py              # FastAPI entry point
├── config.py            # All tunable constants
├── engine/              # Turn loop, world, events, combat, narration
├── npc/                 # NPC class, RL agent, dialogue, personality
├── player/              # Player state & input parsing
├── quest/               # Hierarchical MDP & quest graph
├── llm/                 # HTTP API adapter (ollama/llamacpp/openai)
├── api/                 # REST routes & session management
├── tests/               # Integration + feature tests
└── data/                # Saves, metrics, logs, quest/NPC/world JSON
frontend/
├── index.html
├── css/                 # Dark theme stylesheets
└── js/                  # Vanilla JS modules + Cytoscape.js
models/                  # GGUF files (gitignored)
docs.archive/            # Historical design docs & LaTeX paper
```

