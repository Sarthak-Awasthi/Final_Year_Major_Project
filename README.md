# Cooperative Non-Player Character Behavior via Multi-Agent Reinforcement Learning in a Role-Playing Game Environment

A medieval village RPG research playground for studying emergent cooperative behavior in multi-agent systems using tabular Q-learning with decomposed rewards.

## Research Hypothesis

Early game episodes favor individual NPC gains; as agents learn that community welfare improves their total return through dynamic lambda weighting, policies shift toward cooperative strategies.

## Quick Start

```bash
uv sync

# Set LLM provider (optional -- game works without LLM)
# export OPENROUTER_API_KEY=your_key

uv run uvicorn backend.main:app --reload
# Open http://localhost:8000
```

## Architecture

The system is organized in three layers:

1. **Hierarchical MDP** -- A 7-stage quest graph with 16 static + dynamic checkpoints drives game progression. Macro stages govern quest flow; micro checkpoints handle granular objectives with nudging and forced convergence for deviating players.
2. **Q-Learning NPCs** -- Six NPCs run tabular Q-learning over a 225-state x 28-action space with role-specific action masking. Each NPC maintains a decomposed reward signal that blends individual and community welfare via a dynamic lambda coefficient.
3. **Optional LLM Narration** -- An HTTP adapter connects to OpenRouter, Ollama, or any OpenAI-compatible provider for atmospheric narration. Every LLM call has a template fallback, so the game runs identically without a model.

## Key Features

- **Hierarchical MDP**: 7-stage quest with 16 static + dynamic checkpoints
- **6 Q-Learning NPCs**: 225-state x 28-action tabular Q-learning with role-specific action masking
- **Decomposed Rewards**: R_i(t) = P_i(t) + G_i(t) + lambda_i(t) * C(t) with dynamic lambda
- **Adaptive Personalities**: 4 coefficients evolve per turn (cooperation, risk, social, resilience)
- **System Shocks**: 5 shock types (famine, raid, plague, boom, winter) for non-stationary perturbation
- **Optional LLM**: OpenRouter/Ollama/OpenAI-compatible for narration with template fallbacks
- **Analytics Dashboard**: Real-time cooperation curves, reward decomposition, shock response

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `true` | Enable/disable LLM integration |
| `LLM_PROVIDER` | `openrouter` | LLM provider (`openrouter`, `ollama`, `openai_compatible`) |
| `LLM_API_KEY` | | API key for provider |
| `LLM_MODEL_NAME` | `google/gemma-3-1b-it:free` | Model identifier |
| `ROLE_MASK_ENABLED` | `true` | Enable role-specific action masking |
| `SHOCK_ENABLED` | `true` | Enable system shock engine |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/game/new` | Start new game session |
| POST | `/api/game/action` | Submit player action |
| GET | `/api/game/state` | Get current game state |
| GET | `/api/game/actions` | Get 28-action catalog |
| GET | `/api/quest/graph` | Get MDP graph for visualization |
| GET | `/api/quest/progress` | Quest progress snapshot |
| GET | `/api/npc/list` | List all NPCs |
| GET | `/api/npc/{uid}` | Detailed NPC info |
| GET | `/api/llm/status` | Check LLM connectivity |
| POST | `/api/save` | Save game to slot |
| POST | `/api/load` | Load game from file |
| GET | `/api/metrics/summary` | Session metrics |
| GET | `/api/metrics/timeseries` | Time-series analytics |
| GET | `/api/metrics/experiment` | Full experiment bundle |
| POST | `/api/shocks/trigger` | Trigger system shock |
| GET | `/api/shocks/active` | Active shocks and effects |
| WS | `/ws` | WebSocket for real-time updates |

## Running Experiments

```bash
# Install dev dependencies (includes Jupyter)
uv sync

# Open ablation study notebook (cooperation graphs, 5 conditions, no LLM needed)
uv run jupyter notebook demo_ablation.ipynb

# Open narrative quality notebook (LLM vs template comparison, needs LM Studio)
uv run jupyter notebook demo_narrative.ipynb

# Run full automated ablation study (7 conditions x 20 runs each)
uv run python -m backend.tools.run_experiment

# Run specific conditions with custom parameters
uv run python -m backend.tools.run_experiment --conditions C1,C2,C5 --runs 50 --turns 100
```

The system supports 7 ablation conditions for controlled experiments:

| Condition | Description |
|-----------|-------------|
| **C1** | Everything on (full system baseline) |
| **C2** | No LLM (template fallbacks only) |
| **C3** | No RL (schedule-only NPCs, no Q-learning) |
| **C4** | Flat MDP (no hierarchical quest decomposition) |
| **C5** | No shocks (`SHOCK_ENABLED=false`) |
| **C6** | No role masking (`ROLE_MASK_ENABLED=false`) |
| **C7** | Static lambda (fixed at 0.325, no cooperation feedback loop) |

Each condition isolates a subsystem so its contribution to emergent cooperation can be measured independently.

## Project Structure

```
backend/
  main.py         # FastAPI entry point
  config.py       # All tunable constants and env-var overrides
  engine/         # Game engine, combat, analytics, shock manager, narration
  npc/            # NPC agents, Q-learning, dialogue, personality, schedule
  quest/          # MDP quest system, checkpoints, nudging
  llm/            # LLM service adapter, guardrails, prompts, fallback
  player/         # Player state, input parsing (spaCy NLP)
  api/            # FastAPI routes, WebSocket, session management
  tools/          # Playtest bot, replay, narrative export
  data/           # Game data (NPCs, quests, world, config, saves, metrics)
  tests/          # Integration and feature tests
frontend/
  index.html      # Single-page game UI
  css/            # Dark theme stylesheets
  js/             # Game UI (app.js) and analytics dashboard (analytics.js)
docs/
  Journal/        # LaTeX research report (IEEE format)
```

## Testing

```bash
uv run pytest
```

## License

University research project. All rights reserved.
