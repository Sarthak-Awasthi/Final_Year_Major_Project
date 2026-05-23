# Launch & Configuration Guide

> Complete reference for running, configuring, and experimenting with the RL Research Playground.
> For formula rationale and academic references, see [README.md](README.md#reward-model--formula-rationale).

---

## Quick Start

```bash
uv run uvicorn backend.main:app --reload
```

The server starts at **http://localhost:8000**. Open this in your browser to play.

---

## Launch Command

### Basic Launch (No LLM)

```bash
LLM_ENABLED=false uv run uvicorn backend.main:app --reload
```

### Launch with Ollama (Default)

```bash
# Make sure Ollama is running on port 11434
uv run uvicorn backend.main:app --reload
```

### Launch with llama.cpp Server

```bash
LLM_PROVIDER=llamacpp_server \
LLM_API_BASE_URL=http://localhost:8080 \
LLM_MODEL_NAME=Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
uv run uvicorn backend.main:app --reload
```

### Launch with Role Masking Disabled

```bash
ROLE_MASK_ENABLED=false uv run uvicorn backend.main:app --reload
```

### Launch with Shocks Disabled

```bash
SHOCK_ENABLED=false uv run uvicorn backend.main:app --reload
```

### Full Research Configuration (All Features Enabled)

```bash
LLM_PROVIDER=ollama \
LLM_API_BASE_URL=http://127.0.0.1:11434 \
LLM_MODEL_NAME=qwen3:4b \
LLM_ENABLED=true \
ROLE_MASK_ENABLED=true \
SHOCK_ENABLED=true \
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

### Run Automated Playtest Bot

```bash
uv run python -m backend.tools.playtest_bot
```

---

## Environment Variables

Set these **before** the `uv run` command to override defaults.

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `true` | Enable/disable all LLM calls. Set `false` to run without any AI text generation. Accepts: `1`, `true`, `yes`, `on` |
| `LLM_PROVIDER` | `ollama` | LLM backend to use. Options: `ollama`, `llamacpp_server` |
| `LLM_API_BASE_URL` | `http://127.0.0.1:11434` | Base URL for the LLM server. For llama.cpp: `http://localhost:8080` |
| `LLM_API_KEY` | `""` | API key for authenticated LLM providers (optional) |
| `LLM_MODEL_NAME` | `qwen3:4b` | Model identifier. For Ollama: model tag. For llama.cpp: GGUF filename |
| `LLM_HEALTH_ENDPOINT` | Auto-detected | Custom health check endpoint path (auto-detected per provider) |
| `LLM_CHAT_ENDPOINT` | Auto-detected | Custom chat/completion endpoint path (auto-detected per provider) |

### Game Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `ROLE_MASK_ENABLED` | `true` | **Role-specific action masking.** When enabled, each NPC archetype (farmer, guard, elder, etc.) gets Q-value bonuses for role-aligned actions and penalties for misaligned ones. This constrains each agent to prefer actions fitting their role. |
| `SHOCK_ENABLED` | `true` | **Dynamic shock engine.** When enabled, random economic/environmental shocks (famine, bandit raids, plague, trade boom, harsh winter) can fire during gameplay, affecting community rewards and NPC adaptation. |

---

## Uvicorn Server Arguments

These go **after** `uvicorn backend.main:app`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to allow external connections |
| `--port` | `8000` | Port number to listen on |
| `--reload` | off | Auto-reload server when source files change (development mode) |
| `--log-level` | `info` | Logging verbosity: `debug`, `info`, `warning`, `error`, `critical` |
| `--workers` | `1` | Number of worker processes (don't use with `--reload`) |

---

## API Game Session Parameters

When creating a new game via `POST /api/game/new`, you can pass:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | `42` | Master RNG seed. Same seed = same NPC pre-training, same world initialization. Use different seeds for different experimental runs. |
| `difficulty` | string | `"normal"` | Difficulty preset: `easy`, `normal`, or `hard`. Affects AP costs, combat damage, reputation gains, and deviation thresholds. |
| `max_turns` | int | `200` | Maximum turns before the session automatically ends. Increase for longer experiments. |
| `player_name` | string | `"Traveler"` | Display name for the player character shown in narrations. |

### Difficulty Presets

| Setting | Easy | Normal | Hard |
|---------|------|--------|------|
| AP cost multiplier | ×0.75 | ×1.0 | ×1.5 |
| Combat damage to player | ×0.7 | ×1.0 | ×1.4 |
| Combat damage from player | ×1.3 | ×1.0 | ×0.8 |
| Reputation gain multiplier | ×1.5 | ×1.0 | ×0.7 |
| Max deviations before force convergence | 7 | 5 | 3 |
| Stamina regen per turn | 8 | 5 | 3 |
| Combat flee success rate | 80% | 70% | 55% |

---

## Config Constants Reference

These are set in `backend/config.py` and typically don't need env var overrides, but can be edited directly for research tuning.

### RL Agent Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `MASTER_SEED` | `42` | Global RNG seed for reproducibility |
| `NPC_Q_LEARNING_ALPHA` | `0.1` | Q-learning rate (α) — how fast agents learn from new experiences |
| `NPC_Q_LEARNING_GAMMA` | `0.9` | Discount factor (γ) — how much agents value future vs. immediate reward |
| `NPC_EPSILON_START` | `0.15` | Initial exploration rate — probability of random action |
| `NPC_EPSILON_MIN` | `0.05` | Minimum exploration rate — agents always explore at least this much |
| `NPC_EPSILON_DECAY_RATE` | `0.995` | Per-turn epsilon decay — controls how fast exploration decreases |
| `NPC_COLD_START_TURNS` | `20` | Turns using fallback schedule before Q-learning kicks in |
| `NPC_PRETRAIN_EPISODES` | `100` | Pre-training episodes at game initialization |
| `NPC_PRETRAIN_TURNS` | `50` | Turns per pre-training episode |

### Role Masking (STEP 4)

| Constant | Value | Description |
|----------|-------|-------------|
| `ROLE_MASK_BONUS` | `+0.5` | Q-value bonus added to role-aligned actions during action selection |
| `ROLE_MASK_PENALTY` | `-0.3` | Q-value penalty for role-misaligned actions |

**Role → Action Mapping:**

| Role | Preferred Actions |
|------|-------------------|
| **farmer** | work, give_item, trade, talk, greet, pick_up, drop_item, eat, rest, wait |
| **guard** | defend, attack, talk, intimidate, greet, look, examine, move_to |
| **tavkeeper** | trade, talk, greet, give_item, present_item, ask_info, persuade, rest, wait |
| **elder** | talk, persuade, ask_info, greet, give_item, look, examine, rest, wait |
| **villager** | talk, greet, work, trade, give_item, look, move_to, rest, wait |

### Adaptation & Cooperation Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `lambda_coeff` (per NPC) | `0.05 → 0.60` (dynamic) | Community reward coefficient. **Now driven by cooperation_tendency** via feedback loop: `λ = 0.05 + coop × 0.55`. Starts at 0.325 (coop=0.5) and grows as cooperation increases. |
| `_lambda_min` | `0.05` | Floor — agents always retain minimal community awareness |
| `_lambda_max` | `0.60` | Ceiling — agents never become fully altruistic |
| `adapt_rate` | `0.04` | Rate at which cooperation/risk/sensitivity evolve per turn |
| `drift_toward_neutral` | `0.003` | Gentle pull back toward 0.5 baseline — prevents runaway drift |

### Shock Engine (STEP 5)

| Constant | Value | Description |
|----------|-------|-------------|
| `SHOCK_MAX_ACTIVE` | `3` | Maximum concurrent shocks |

**Available Shock Types:**

| Shock | Duration | Decay | Stat Effects (per turn) |
|-------|----------|-------|---------|
| `famine` | 15 turns | linear | −happiness (drain 0.3), −trust (−0.5/turn), ×0.5 reward scale |
| `bandit_raid` | 8 turns | sudden | −happiness (drain 0.5), −HP (drain > 0.3), −trust (−1.0/turn), ×0.6 reward scale |
| `plague` | 12 turns | linear | −happiness (drain 0.6), −HP (drain > 0.3), −health stat, −trust (−0.3/turn), ×0.4 reward scale |
| `trade_boom` | 10 turns | linear | +happiness (gain 0.2), +trust (+0.5/turn), ×1.5 reward scale |
| `harsh_winter` | 20 turns | linear | −happiness (drain 0.4), −HP (drain > 0.3), −trust (−0.2/turn), ×0.7 reward scale |

### Quest / Nudging

| Constant | Value | Description |
|----------|-------|-------------|
| `NUDGE_LAMBDA` | `0.3` | Nudge reward shaping weight |
| `NUDGE_HINT_THRESHOLD` | `3` | Deviations before hints appear to guide the player |
| `NUDGE_FORCE_CONVERGENCE_THRESHOLD` | `5` | Deviations before forced convergence back to main quest |
| `DYNAMIC_CP_LOOP_THRESHOLD` | `3` | Dynamic checkpoint loop detection limit |

### LLM Tuning

| Constant | Value | Description |
|----------|-------|-------------|
| `LLM_MAX_PROMPT_TOKENS` | `2500` | Maximum tokens per prompt |
| `LLM_DEFAULT_TEMPERATURE` | `0.7` | Generation temperature (higher = more creative) |
| `LLM_TIMEOUT_SECONDS` | `10` | Request timeout before fallback to template narration |
| `LLM_MAX_RETRIES` | `2` | Retry count on LLM failure |
| `LLM_MIN_INTERVAL_MS` | `2000` | Minimum milliseconds between consecutive LLM calls |
| `LLM_MAX_CALLS_PER_MINUTE` | `20` | Rate limit — max LLM calls per minute |

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/game/new` | Start a new game session |
| POST | `/api/game/action` | Submit a player action (text or button) |
| GET | `/api/game/state` | Get full current game state |
| GET | `/api/game/actions` | Get the 27-action catalog |
| GET | `/api/game/session` | Get session metadata |
| GET | `/api/quest/graph` | Get MDP graph data for visualization |
| GET | `/api/quest/progress` | Get quest progress snapshot |
| GET | `/api/npc/list` | List all NPCs |
| GET | `/api/npc/{uid}` | Get detailed NPC info |
| GET | `/api/llm/status` | Check LLM connectivity |
| POST | `/api/save` | Save game to slot |
| POST | `/api/load` | Load game from file |
| GET | `/api/saves` | List available save files |
| GET | `/api/metrics/summary` | Get session metrics |
| GET | `/api/metrics/analytics/rewards/{uid}` | NPC reward decomposition series |
| GET | `/api/metrics/analytics/cooperation` | Cooperation progression series |
| GET | `/api/metrics/analytics/community-rewards` | Community reward series |
| GET | `/api/metrics/analytics/shock-response` | Shock impact analysis |
| GET | `/api/metrics/analytics/social-welfare` | Social welfare metrics |
| POST | `/api/shocks/trigger` | Manually trigger a system shock (body: `{shock_type, source}`) |
| GET | `/api/shocks/active` | Get currently active shocks and their effects |
| WS | `/ws` | WebSocket for real-time updates |

---

## In-Game Shock Trigger (Debug Panel)

Press **`D`** during gameplay to open the **Debug / Research** panel. The **⚡ System Shocks** section allows you to:

1. **View active shocks** — intensity %, remaining turns, reward modifier, and adaptation pressure
2. **Trigger shocks** — click any of the 5 shock type buttons to inject a shock immediately
3. **Custom duration** — set a custom duration (in turns) before triggering, or leave empty for defaults
4. **Observe impact** — watch cooperation curves in the Analytics panel (`A` key) respond to the shock

**Workflow:** Start a game → play 10-15 turns to establish a cooperation baseline → press `D` → trigger a **Famine** → continue playing → observe cooperation drop in the Analytics dashboard → optionally trigger a **Trade Boom** to see recovery.

## Example Research Workflows

### Experiment: Compare Role Masking On vs. Off

```bash
# Run 1: Masking enabled (default)
ROLE_MASK_ENABLED=true uv run uvicorn backend.main:app --port 8000

# Run 2: Masking disabled
ROLE_MASK_ENABLED=false uv run uvicorn backend.main:app --port 8001
```

### Experiment: Shock Impact Analysis

```bash
# Run with shocks
SHOCK_ENABLED=true uv run uvicorn backend.main:app --port 8000

# Run without shocks (control)
SHOCK_ENABLED=false uv run uvicorn backend.main:app --port 8001
```

### Experiment: LLM vs. Template Narration

```bash
# With LLM
LLM_ENABLED=true LLM_PROVIDER=ollama uv run uvicorn backend.main:app --port 8000

# Without LLM (template-only)
LLM_ENABLED=false uv run uvicorn backend.main:app --port 8001
```
