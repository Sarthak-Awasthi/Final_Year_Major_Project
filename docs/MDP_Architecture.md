# MDP Architecture — Quest System Documentation

> Reference document for the hierarchical MDP quest system used in the MVP Research Game.  
> Last updated: March 2, 2026

---

## 1. What Is a Traditional MDP?

A **Markov Decision Process** is defined as a tuple $(S, A, T, R, \gamma)$:

| Symbol | Definition |
|--------|-----------|
| $S$ | Finite set of states |
| $A$ | Finite set of actions |
| $T(s, a, s')$ | Transition function — probability of reaching state $s'$ from state $s$ after taking action $a$. For all $(s, a)$: $\sum_{s'} T(s, a, s') = 1$ |
| $R(s, a)$ | Reward function — immediate reward for taking action $a$ in state $s$ |
| $\gamma \in [0, 1]$ | Discount factor — how much future rewards are valued relative to immediate ones |

### Key Properties

- **Markov Property**: The next state depends *only* on the current state and chosen action, not on the history of states visited. Formally: $P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$.
- **Transition Matrix**: In a traditional MDP with $|S|$ states and $|A|$ actions, the full transition function is a 3D tensor of shape $|S| \times |A| \times |S|$, where each $|S|$-slice (for a given $s, a$) sums to 1.
- **Stationarity**: The transition and reward functions do not change over time.
- **Policy**: A mapping $\pi: S \to A$ that specifies which action to take in each state. The goal is to find an **optimal policy** $\pi^*$ that maximises expected cumulative discounted reward.

### Transition Matrix Example (Traditional)

For a simple 3-state, 2-action MDP:

$$T(\text{action}_1) = \begin{bmatrix} 0.8 & 0.2 & 0.0 \\ 0.1 & 0.7 & 0.2 \\ 0.0 & 0.3 & 0.7 \end{bmatrix}$$

Each row sums to 1. Entry $(i, j)$ gives $P(s' = j \mid s = i, a = \text{action}_1)$.

---

## 2. Our Implementation — Formal Mapping

Our quest system **is a valid MDP**. Here is the formal mapping:

| Formal MDP Component | Our Implementation |
|---|---|
| $S$ (States) | Checkpoint IDs: `1_1`, `1_2`, `2_1`, ..., `7_2`, `S_success`, `S_fail` — finite discrete states |
| $A$ (Actions) | 28 universal actions (same set available at every state) |
| $T(s, a, s')$ (Transitions) | `quest_transitions` dict per checkpoint — sparse, mostly deterministic |
| $R(s, a)$ (Rewards) | `effects` dict: reputation deltas, stamina costs, items given/removed |
| $\gamma$ (Discount) | 1.0 at macro level (stages), 0.95 at micro level (checkpoints) |

### 2.1 State Space $S$

States are **checkpoint IDs** — string identifiers like `"1_1"`, `"3_2"`, `"3_D1"`.

- **Static checkpoints**: Pre-defined in `main_quest.json`. Format: `"{stage}_{index}"` (e.g., `"1_1"`, `"4_3"`).
- **Dynamic checkpoints**: Generated at runtime when the player deviates. Format: `"{stage}_D{counter}"` (e.g., `"1_D1"`, `"2_D3"`).
- **Terminal states**: `S_success` (quest victory) and `S_fail` (quest failure) — absorbing states with no outgoing transitions.

Total base state count: ~20 static checkpoints + 2 terminals = ~22 states. Dynamic checkpoints expand this at runtime.

### 2.2 Action Space $A$

28 universal actions, always available regardless of state:

| Category | Actions |
|----------|---------|
| Navigation | `move_to` |
| Exploration | `look`, `search`, `examine` |
| Social | `talk`, `greet`, `ask_info`, `persuade`, `trade`, `give_item`, `present_item`, `deceive`, `intimidate` |
| Combat | `attack`, `defend`, `flee` |
| Stealth | `sneak`, `hide`, `steal` |
| Utility | `pick_up`, `use_item`, `eat`, `rest`, `wait`, `drop_item`, `status`, `equip`, `work` |

Actions are **never restricted** per state. Instead, most actions produce self-loops (no quest state change) while still having gameplay effects (combat, dialogue, etc.).

### 2.3 Transition Function $T(s, a, s')$

The transition function is **well-defined for every $(s, a)$ pair**:

$$T(s, a, s') = \begin{cases} 1.0 & \text{if } a \in \text{transitions}(s) \text{ and } s' = \text{next}(s, a) \text{ and requirements met} \\ 1.0 & \text{if } a \notin \text{transitions}(s) \text{ and } s' = s \quad \text{(self-loop)} \\ 1.0 & \text{if } a \in \text{transitions}(s) \text{ but requirements not met and } s' = s \end{cases}$$

For the rare **stochastic** case (e.g., `sneak` with `success_prob: 0.6`):

$$T(s, \text{sneak}, s_{\text{next}}) = 0.6, \quad T(s, \text{sneak}, s) = 0.4$$

#### Concrete Example — Checkpoint `1_1`

```json
"quest_transitions": {
    "greet":        { "next": "1_2", "effects": {"reputation": {"guard_a3f1": 2, "guard_b7e2": 2}} },
    "sneak":        { "next": "1_2", "effects": {"reputation": {"guard_a3f1": -5, "guard_b7e2": -5}}, "success_prob": 0.6 },
    "present_item": { "next": "1_2", "effects": {"reputation": {"guard_a3f1": 5, "guard_b7e2": 5}}, "requires": {"item": "travel_papers"} }
}
```

The effective transition table for state `1_1` (28 actions × possible next states):

| Action | $P(s' = \text{1\_2})$ | $P(s' = \text{1\_1})$ | Notes |
|--------|---|---|---|
| `greet` | 1.0 | 0.0 | Deterministic advance |
| `sneak` | 0.6 | 0.4 | Stochastic (success_prob) |
| `present_item` (with papers) | 1.0 | 0.0 | Deterministic, requires item |
| `present_item` (without papers) | 0.0 | 1.0 | Requirement not met → self-loop |
| `attack` | 0.0 | 1.0 | Self-loop (no quest transition) |
| `look` | 0.0 | 1.0 | Self-loop |
| ... (other 22 actions) | 0.0 | 1.0 | Self-loop |

Every row sums to 1.0. The Markov property holds.

### 2.4 Reward Function $R(s, a)$

Rewards are the `effects` dict attached to each transition:

- **Reputation changes**: `{"guard_a3f1": +5, "guard_b7e2": +5}` — per-NPC
- **Stamina costs**: `{"stamina": -3}` — action point expenditure
- **Item grants**: `{"gives": [{"id": "artifact_clue", ...}]}`
- **Item removal**: `{"removes": ["travel_papers"]}`

Actions that produce self-loops (no quest transition) still have gameplay effects (combat damage, dialogue, AP costs), but these are **gameplay rewards**, not MDP quest rewards.

### 2.5 Discount Factors $\gamma$

| Level | $\gamma$ | Rationale |
|-------|----------|-----------|
| Macro MDP (stages) | 1.0 | Undiscounted — reaching the goal matters, not speed |
| Micro MDP (checkpoints) | 0.95 | Encourages efficiency within a stage |

---

## 3. Hierarchical Structure

The system uses a **Hierarchical MDP** consistent with the **Options framework** (Sutton, Precup & Singh, 1999):

### Macro MDP (Stage Level)

```
S1 → S2 → S3 → S4 → S5 → S6 → S7 → S_success
                                    ↘ S_fail
```

- **States**: 7 stages + 2 terminal states
- **Transitions**: A stage completes when its last checkpoint transitions to a checkpoint in the next stage
- **γ = 1.0**: No discounting at the strategic level

### Micro MDP (Checkpoint Level)

Within each stage, checkpoints form a small directed graph:

```
Stage 4 example:
    4_1 → 4_2 → 4_3    (linear path)
    4_1 → 4_3           (skip path via alternative action)
```

- **States**: 2–3 checkpoints per stage
- **Transitions**: Action-keyed, sparse, mostly deterministic
- **γ = 0.95**: Slight discounting encourages direct paths

### Analogy

Each stage acts as an **option** (temporally extended action) in the macro MDP. The micro MDP governs behavior within an option. Completing a stage's checkpoints is equivalent to terminating an option and transitioning in the macro MDP.

---

## 4. The Markov Property — Why It Holds

The Markov property requires: the next state depends only on the current state and action, not on history.

The transition matching function (`_match_transition()` in `quest_manager.py`) inspects:

1. **Current checkpoint** (the state)
2. **Action ID** (the action)
3. **Player context** (inventory, location)

Point 3 means the **effective state is augmented**:

$$s_{\text{effective}} = (\text{checkpoint\_id}, \text{inventory}, \text{location})$$

This is standard in applied MDPs — the checkpoint ID alone is a **partial state**; the full Markov state includes context variables that affect transition eligibility. This is analogous to how in robotics MDPs, the state includes both discrete task phase and continuous sensor readings.

The crucial point: **no transition depends on which checkpoints were previously visited or how many turns have elapsed**. The history of completed checkpoints is tracked for UI/save purposes but is never consulted by the transition function. Only the current checkpoint + current context matters. The Markov property holds.

---

## 5. How It Differs from a Traditional MDP

| Aspect | Traditional MDP | Our Implementation |
|--------|----------------|-------------------|
| **State space** | Fixed $\|S\|$ | Expandable at runtime (dynamic checkpoints) |
| **Transition representation** | Dense $\|S\| \times \|A\| \times \|S\|$ tensor | Sparse dict: only 1–3 actions per state cause non-self-loop transitions |
| **Stochasticity** | Fully stochastic $T$ | Mostly deterministic; rare stochastic transitions via `success_prob` |
| **Matrix sparsity** | Typically moderate | Extreme: of 28 actions × ~20 states = 560 $(s,a)$ pairs, only ~30 cause state changes (>94% identity/self-loop) |
| **Stationarity** | Fixed $T$ and $R$ | Base transitions are fixed; dynamic CPs extend $S$ and add new transitions |
| **State definition** | Single state variable | Augmented state: (checkpoint_id, inventory, location) |

### Why These Differences Are Acceptable

1. **Dynamic state space**: Can be framed as **state-space augmentation** triggered by player deviation, analogous to option creation in hierarchical RL. The base MDP has a fixed topology; extensions are conservative additions.

2. **Extreme sparsity**: A valid design choice. Storing a full $22 \times 28 \times 22 = 13{,}552$-entry tensor where 94%+ entries are zero would be wasteful. The dict representation is equivalent but compact.

3. **Deterministic dominance**: A deterministic MDP is a valid subclass where $T(s, a, s') \in \{0, 1\}$. Deterministic MDPs reduce to **shortest-path problems**, which is exactly what the quest is — find the optimal action sequence from `1_1` to `S_success`.

---

## 6. Code Architecture

### Data Structures (`backend/quest/mdp.py`)

```python
@dataclass
class Checkpoint:
    checkpoint_id: str          # "1_1", "3_D1"
    stage_id: int               # Parent stage number
    description: str            # Narrative text
    completion_conditions: dict  # action_key → {next, effects, requires}
    # ... other fields

@dataclass
class Stage:
    stage_id: int
    name: str
    checkpoints: dict[str, Checkpoint]
    next_stage: int | None

class QuestMDP:
    stages: dict[int, Stage]    # The complete graph
    MACRO_GAMMA = 1.0
    MICRO_GAMMA = 0.95
```

### Transition Evaluation (`backend/quest/quest_manager.py`)

```python
class QuestManager:
    current_checkpoint: str      # Current state
    completed_checkpoints: list  # History (for UI, not transitions)
    
    def check_completion(action_id, target, context) -> dict | None:
        # 1. Get current checkpoint's completion_conditions
        # 2. Match action against transition keys
        # 3. Validate requires constraints
        # 4. Return {next_checkpoint, rewards, stage_transition}
    
    def _match_transition(action_id, target, context, cp) -> dict | None:
        # Priority 1: Exact key match (action_id in conditions)
        # Priority 2: Compound key match (e.g., "move_to_fields")
        # Priority 3: Validate requires (item, location)
```

### Runtime Flow (per turn)

```
Player action
    → game_engine._check_quest_progress()
        → quest_manager.check_completion(action_id, target, context)
            → _match_transition(action_id, target, context, current_checkpoint)
                → Match found? → Return {next_cp, rewards}
                → No match?   → Return None (self-loop, no quest change)
        → If matched: quest_manager.advance_checkpoint(next_cp)
            → Update current_checkpoint
            → Record old checkpoint in completed_checkpoints
            → Detect stage transitions
            → Handle terminal states (S_success / S_fail)
```

### JSON Source (`backend/data/quests/main_quest.json`)

```json
{
  "quest_id": "main_quest_01",
  "stages": [
    {
      "stage_id": 1,
      "title": "Arrival at the Gate",
      "checkpoints": [
        {
          "cp_id": "1_1",
          "quest_transitions": {
            "greet": { "next": "1_2", "effects": {...} },
            "sneak": { "next": "1_2", "effects": {...}, "success_prob": 0.6 },
            "present_item": { "next": "1_2", "effects": {...}, "requires": {"item": "travel_papers"} }
          }
        },
        {
          "cp_id": "1_2",
          "quest_transitions": {
            "move_to": { "next": "2_1", "effects": {...} }
          }
        }
      ]
    }
  ]
}
```

---

## 7. Summary Pitch

> The quest system is modeled as a **hierarchical deterministic MDP** with two levels. The macro level has 7 stage-states plus two terminal absorbers (success/fail), with γ = 1.0. Each stage decomposes into a micro-MDP of 2–3 checkpoint-states with γ = 0.95. The action space is a universal catalog of 28 actions shared across all states. Transitions are defined as a sparse action-keyed function: at each checkpoint, 1–3 actions advance the quest while the remaining 25+ actions produce self-loops with gameplay effects (combat, dialogue, etc.) but no quest state change. The augmented state includes checkpoint ID plus player context (inventory, location) to handle conditional transitions. The system supports runtime state-space expansion via dynamically generated checkpoints for handling player deviations, making it a practical hybrid between a fixed MDP and an options-based hierarchical framework.

---

## References

- Sutton, R. S., Precup, D., & Singh, S. (1999). *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence, 112(1-2), 181–211.
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming.* Wiley.
- Bellman, R. (1957). *Dynamic Programming.* Princeton University Press.
