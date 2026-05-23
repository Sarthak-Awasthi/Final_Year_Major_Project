# MVP — Multi-Agent RL Research Playground

> A single-session research environment where NPC agents learn cooperative behavior through tabular Q-learning, adaptive personalities, role-grounded action masking, and system shocks.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-91%20passing-brightgreen.svg)](#testing)

---

## Research Hypothesis

> *Early turns should favor short-term individual gains. As agents learn, their policies should shift toward stronger community contribution due to higher long-run combined reward — especially under system shocks that make cooperation critical for survival.*

---

## Quick Start

```bash
uv sync                                          # install deps
uv run uvicorn backend.main:app --reload         # start server @ http://localhost:8000
uv run pytest backend/tests/ -v                  # run tests (91 passing)
```

See **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** for full configuration, environment variables, and experiment presets.

---

## Architecture Overview

```
backend/
├── main.py              # FastAPI entry point
├── config.py            # All tunable constants & env-var overrides
├── engine/
│   ├── game_engine.py   # Main turn loop & orchestration
│   ├── shock_manager.py # System shock lifecycle & effect aggregation
│   ├── world.py         # World state, locations, time
│   ├── combat.py        # Combat resolution
│   └── events.py        # Random event system
├── npc/
│   ├── npc.py           # NPC class, adaptation state, cooperation dynamics
│   ├── rl_agent.py      # Q-learning, reward computation, action selection
│   ├── dialogue.py      # Dialogue & gossip system
│   └── personality.py   # Archetype loading & personality traits
├── player/              # Player state & input parsing
├── quest/               # Hierarchical MDP & quest graph
├── llm/                 # HTTP API adapter (Ollama / llama.cpp / OpenAI-compat)
├── api/                 # REST routes & session management
├── tests/               # 91 integration + feature tests
└── data/                # Saves, metrics, logs, quest/NPC/world JSON
frontend/
├── index.html           # Single-page game UI
├── css/                 # Dark theme stylesheets
└── js/                  # Vanilla JS + Chart.js + Cytoscape.js
```

---

## Reward Model & Formula Rationale

### Total Reward (Per NPC, Per Turn)

```
R_i(t) = P_i(t) + G_i(t) + λ_i(t) × C(t)
```

| Symbol | Term | Description |
|--------|------|-------------|
| `P_i(t)` | **Penalty** | Large negative for catastrophic states (HP < 20%). Prevents degenerate policies. |
| `G_i(t)` | **Individual** | Weighted sum of per-NPC stat deltas: `Σ w_k × (stat_k(t) - stat_k(t-1))` |
| `C(t)` | **Community** | Village-level welfare signal (hybrid absolute + delta, non-linear) |
| `λ_i(t)` | **Lambda** | Dynamic prosociality coefficient driven by `cooperation_tendency` |

**Why this structure:** This is a standard *mixed-motive reward decomposition* from cooperative MARL. The individual term captures selfish utility (did my stats improve?), while the community term captures prosocial utility (is the village better off?). The additive structure allows each component to be independently measured, debugged, and ablated — critical for research prototypes.

> **References:**
> - Lowe et al., *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG), 2017 — [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)
> - Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL*, 2018 — [arXiv:1803.11485](https://arxiv.org/abs/1803.11485)

---

### Community Reward (Hybrid Absolute + Delta)

```python
R_community = 0.6 × R_absolute + 0.4 × R_delta

R_absolute = sigmoid(avg_rep, center=25, scale=0.08) × 0.5    # reputation
           + (hp_norm ^ 0.7) × 0.3                             # health
           + sigmoid(avg_mood, center=5, scale=0.6) × 0.2      # mood

R_delta    = Δrep/20 × 0.5 + Δhp/50 × 0.3 + Δmood/3 × 0.2   # turn-over-turn change
```

**Why hybrid:** A purely absolute formula saturates (health at 90% = constant 0.27), causing reward plateaus. A purely delta formula is noisy and has zero baseline. The 60/40 hybrid provides a stable baseline plus a gradient signal that responds to improvement or decline each turn.

**Why non-linear (sigmoid + sub-linear power):**
- **Reputation sigmoid:** Diminishing returns at high reputation prevents reward ceiling; amplified sensitivity near crisis (low rep).
- **Health sub-linear (power 0.7):** Crisis hits harder — losing HP from 50 → 30 is more impactful than 180 → 160.
- **Mood sigmoid:** S-curve centered at the default (5) provides symmetric sensitivity to mood swings.

> **References:**
> - Ng, Harada & Russell, *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*, ICML 1999 — [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
> - The sigmoid shaping draws from potential-based reward shaping theory, ensuring the added structure doesn't alter the optimal policy under the base MDP.

---

### Individual Reward (Delta-Based)

```
G_i(t) = Σ w_k × (new_stat_k - old_stat_k)
```

**Why delta-based:** Delta rewards provide clean temporal credit assignment — the reward is directly tied to what changed *this turn*. An agent that eats and gains +2 energy gets a positive signal; one at max energy that eats gets 0. This is the standard approach in RL for avoiding reward hacking on absolute values.

> **Reference:**
> - Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018, Ch. 3 (Reward Hypothesis) — [Full text](http://incompleteideas.net/book/the-book-2nd.html)

---

### Cooperation Feedback Loop (Dynamic λ)

```
λ_i = λ_min + cooperation_tendency × (λ_max - λ_min)
    = 0.05  + coop × 0.55

# cooperation_tendency evolves based on community reward:
#   community > 0.2  →  cooperation ↑
#   community < -0.2 →  cooperation ↓
```

| cooperation_tendency | λ | Behavior |
|---------------------|---|----------|
| 0.0 (selfish) | 0.05 | Nearly ignores community welfare |
| 0.5 (default start) | 0.325 | Balanced individual/community |
| 1.0 (fully cooperative) | 0.60 | Community reward dominates decisions |

**Why dynamic λ:** A static λ forces a fixed cooperation level. Dynamic λ creates an emergent feedback loop: positive community outcomes → cooperation rises → λ increases → agent weights community more → more cooperative actions → community improves further (virtuous cycle). Under shocks, the reverse occurs (vicious cycle), creating the rich dynamics needed for research.

> **References:**
> - Leibo et al., *Multi-agent Reinforcement Learning in Sequential Social Dilemmas*, AAMAS 2017 — [arXiv:1702.03037](https://arxiv.org/abs/1702.03037)
> - This paper demonstrates that cooperation is an emergent property of the environment structure and reward design, not a hard-coded behavior — exactly the principle behind our dynamic λ approach.

---

### Q-Learning (Tabular)

```
Q[s,a] = Q[s,a] + α × (r + γ × max(Q[s']) - Q[s,a])
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| α (learning rate) | 0.1 | Moderate — stable convergence without forgetting |
| γ (discount) | 0.9 | Agents value future rewards (long-horizon cooperation) |
| ε (exploration) | 0.15 → 0.05 | Decaying ε-greedy; starts exploratory, becomes exploitative |
| State space | 225 states | 5 dimensions × 3-5 bins per dimension |
| Action space | 27 actions | Universal action catalog with role masking |

**Why tabular Q-learning over deep RL:** With 225 states × 27 actions = 6,075 Q-values per NPC, tabular Q-learning is deterministic, reproducible, interpretable (you can inspect every Q-value), and converges with theoretical guarantees. For a research prototype studying cooperation dynamics, this isolates the cooperation signal from neural network training noise.

> **References:**
> - Watkins & Dayan, *Q-learning*, Machine Learning 8:279–292, 1992 — [Springer](https://link.springer.com/article/10.1007/BF00992698)
> - Watkins, *Learning from Delayed Rewards*, PhD Thesis, University of Cambridge, 1989

---

### System Shocks

Shocks create non-stationary environments that stress-test agent adaptation:

| Shock | Duration | Effects | Purpose |
|-------|----------|---------|---------|
| Famine 🌾 | 15 turns | −health, −mood, −trust | Tests resilience under resource scarcity |
| Bandit Raid ⚔️ | 8 turns | −income, −trust, −mood | Tests response to sudden external threat |
| Plague ☠️ | 12 turns | −health, −HP, −mood | Tests sustained welfare decline |
| Trade Boom 📈 | 10 turns | +income, +mood, +trust | Tests positive shock recovery |
| Harsh Winter ❄️ | 20 turns | −health, −income, −mood | Tests long-duration stress |

Shocks apply `resource_drain` to NPC stats (HP, health, happiness) and `trust_modifier` to reputation each turn, creating organic community reward decline that requires real behavioral adaptation to recover from.

---

## Key Features

| Feature | Description | Keyboard |
|---------|-------------|----------|
| **RL Analytics Dashboard** | Cooperation curves, reward decomposition, shock response charts | `A` |
| **Debug / Research Panel** | NPC Q-values, system shock trigger controls, game metrics | `D` |
| **Quest Graph** | Cytoscape.js MDP visualization with dynamic checkpoints | `Q` |
| **System Shock Triggers** | Inject famine/raid/plague/boom/winter from Debug panel | `D` → click |
| **Role Action Masking** | Each NPC archetype has weighted action preferences | Config |
| **LLM Integration** | Optional Ollama/llama.cpp narration (graceful fallback) | Config |

---

## Testing

```bash
uv run pytest backend/tests/ -v    # 91 tests across 3 suites
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| Core gameplay | 34 | Engine, combat, saves, quest, world |
| Shock engine | 23 | Lifecycle, effects, integration, serialization |
| Adaptation & masking | 34 | Cooperation, role telemetry, metrics |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** | Complete reference for launch commands, env vars, API endpoints, experiment presets |
| **[Plan.md](Plan.md)** | Implementation roadmap and architecture specification |
| `docs.archive/` | Historical design docs and early LaTeX paper drafts |

---

## References

1. Watkins, C.J.C.H. & Dayan, P. (1992). *Q-learning*. Machine Learning 8:279–292. [Springer](https://link.springer.com/article/10.1007/BF00992698)
2. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. [Full text](http://incompleteideas.net/book/the-book-2nd.html)
3. Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)
4. Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL*. [arXiv:1803.11485](https://arxiv.org/abs/1803.11485)
5. Leibo, J.Z. et al. (2017). *Multi-agent Reinforcement Learning in Sequential Social Dilemmas*. [arXiv:1702.03037](https://arxiv.org/abs/1702.03037)
6. Ng, A.Y., Harada, D. & Russell, S.J. (1999). *Policy Invariance Under Reward Transformations*. ICML. [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
