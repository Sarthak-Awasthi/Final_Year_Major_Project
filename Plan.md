# MVP Plan v2 – Research Project (Game/Simulation with MDP + RL + LLM)

> **Improved & formalized version of Plan.md**  
> All original MVP requirements preserved; technical gaps filled.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Architecture & Tech Stack](#2-architecture--tech-stack)
3. [Formal MDP Specification – Quest System](#3-formal-mdp-specification--quest-system)
4. [NPC RL Agent Specification](#4-npc-rl-agent-specification)
   - 4.6 [NPC-to-NPC Interactions](#46-npc-to-npc-interactions)
5. [Core Game Systems](#5-core-game-systems)
   - 5.1 [Player Stats](#51-player-stats)
   - 5.2 [Health System](#52-health-system)
   - 5.3 [Stamina / Action Points](#53-stamina--action-points)
   - 5.4 [Inventory](#54-inventory)
   - 5.5 [Per-NPC Reputation System](#55-per-npc-reputation-system)
   - 5.6 [Combat Resolution Mechanic](#56-combat-resolution-mechanic)
   - 5.7 [Random Events System](#57-random-events-system)
   - 5.8 [Action Outcome Narration](#58-action-outcome-narration)
   - 5.9 [World Memory & Event Log](#59-world-memory--event-log)
   - 5.10 [Edge Case Handling](#510-edge-case-handling)
   - 5.11 [Victory & Defeat](#511-victory--defeat)
   - 5.12 [Difficulty Scaling](#512-difficulty-scaling)
6. [Dynamic Checkpoint Generation & Nudging](#6-dynamic-checkpoint-generation--nudging)
7. [LLM Integration Specification](#7-llm-integration-specification)
   - 7.5 [LLM Output Validation & Guardrails](#75-llm-output-validation--guardrails)
8. [Visual MDP Representation](#8-visual-mdp-representation)
   - 8.5 [Metrics & Analytics Dashboard](#85-metrics--analytics-dashboard)
9. [Data Models](#9-data-models)
   - 9.5 [Save/Load Specification](#95-saveload-specification)
10. [Game World Design](#10-game-world-design)
11. [Project Structure](#11-project-structure)
12. [Implementation Phases](#12-implementation-phases)
13. [Example Playthrough](#13-example-playthrough)
14. [Additional Features & Research Tooling](#14-additional-features--research-tooling)

- 14.1 [Reproducibility & Debugging](#141-reproducibility--debugging-high-priority)
- 14.2 [UI Enhancements](#142-ui-enhancements-high-priority)
- 14.3 [Research Tooling](#143-research-tooling-medium-priority--stretch-goals)
- 14.4 [Quality of Life](#144-quality-of-life-medium-priority)

---

## 1. Objective

Build an **MVP** that demonstrates:

| # | Feature                                                                    | Priority   |
|---|----------------------------------------------------------------------------|------------|
| 1 | Quest modeled as a **hierarchical MDP** (major stages → checkpoints)       | **Must**   |
| 2 | **Core systems**: Health, Stamina/AP, Inventory, Reputation                | **Must**   |
| 3 | **Visual MDP graph** (nodal view, like neural network diagrams)            | **Must**   |
| 4 | **Adaptive system**: dynamic checkpoint generation + player nudging        | **Must**   |
| 5 | **NPC RL agents** (6 NPCs with tabular Q-learning, all using the same universal action space) | **Must**   |
| 6 | **Local LLM integration** (GGUF, qwen3-4B) for quest/checkpoint generation | **Should** |

---

## 2. Architecture & Tech Stack

### Stack

| Layer                 | Technology                                    | Why                                                          |
|-----------------------|-----------------------------------------------|--------------------------------------------------------------|
| Backend / Game Engine | **Python 3.11+ / FastAPI**                    | Fast prototyping, async support, rich ecosystem              |
| Frontend              | **HTML + CSS (modern minimal) + Vanilla JS + Cytoscape.js** | Clean, minimal dark-themed UI; best-in-class graph visualization for MDP nodal view |
| NLP                   | **spaCy** (`en_core_web_md`)                  | Tokenization, lemmatization, word vectors for similarity matching, entity recognition |
| RL Engine             | **NumPy-based tabular Q-learning**            | Simple, interpretable, sufficient for 6 NPCs (4,860 entries each) |
| LLM Runtime           | **llama-cpp-python** (bindings for llama.cpp) | Runs GGUF models locally, Python-native API                  |
| LLM Model             | **qwen3-4B-q4_k_m.gguf**                      | Small enough for laptop, capable enough for text generation  |
| Data Persistence      | **JSON files** (MVP)                          | No DB overhead; easy to inspect/debug                        |
| Communication         | **REST API + WebSocket**                      | REST for actions, WebSocket for real-time MDP graph updates  |

> **Scope:** This is a **single-player, single-session** MVP. The server supports one active game session at a time. Concurrent players/sessions are out of scope.

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   WEB FRONTEND (Modern Minimal)          │
│  ┌────────────────────┐  ┌─────────────────────────┐     │
│  │  Narrative Log      │  │   MDP Visualization     │     │
│  │  (scrollable)       │  │   (Cytoscape.js graph)  │     │
│  ├────────────────────┤  └──────────┬──────────────┘     │
│  │  Action Palette     │             │ WS updates         │
│  │  (categorized)      │             │                    │
│  ├────────────────────┤             │                    │
│  │  Free-Text Input    │             │                    │
│  └──────┬─────────────┘             │                    │
│         │ REST/WS                   │                    │
└─────────┼───────────────────────────┼────────────────────┘
          │                           │
┌─────────▼───────────────────────────▼────────────────────┐
│                   FASTAPI BACKEND                        │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Input Parser / Action Resolver            │    │
│  │  (palette click → action) OR (text → NLP → action)│    │
│  │  Action Resolution: preconditions + context eval  │    │
│  └──────────────────┬───────────────────────────────┘    │
│                     │                                    │
│  ┌─────────────┐ ┌──▼───────┐ ┌────────────────┐        │
│  │ Game Engine │ │ Quest MDP│ │ NPC RL Engine  │        │
│  │ (turn loop) │ │ Manager  │ │ (Q-learning)   │        │
│  └─────┬───────┘ └────┬─────┘ └───────┬────────┘        │
│        │              │               │                  │
│  ┌─────▼──────────────▼───────────────▼────────┐        │
│  │           LLM Service (optional)            │        │
│  │        llama-cpp-python / qwen3-4B          │        │
│  └─────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### Universal Action Space

The game uses a single **universal action catalog** shared by all entities (player and NPCs). Every action is theoretically available at every game state — there are no per-checkpoint or per-location restrictions on what actions exist. Instead, actions have **preconditions** and **context modifiers** that determine their outcome:

| Concept | Description |
|---------|-------------|
| **Always available** | Every action in the catalog can be attempted at any time by any entity |
| **Preconditions** | Determine success probability (e.g., `attack` without a weapon = low success, `trade` with no merchant = no effect) |
| **Context modifiers** | Location, NPCs present, time of day, entity stats, and inventory modify the action's effect and outcome |
| **Outcome spectrum** | Actions don't just succeed/fail — they produce contextual outcomes based on the full game state |

#### Global Action Catalog

| Action ID | Label | Category | Base AP | Precondition Examples |
|-----------|-------|----------|---------|----------------------|
| `move_to` | Move to location | Navigation | 3 | Adjacent location exists |
| `look` | Look around | Exploration | 1 | — (always succeeds, results depend on location) |
| `search` | Search area thoroughly | Exploration | 5 | — (findings depend on location + time) |
| `examine` | Examine object / person | Exploration | 2 | Object or NPC present |
| `talk` | Talk to NPC | Social | 1 | NPC present at location. Opens general conversation; can discuss any topic. Used for extended dialogue and information exchange. |
| `greet` | Greet someone | Social | 1 | NPC present at location. First-contact action; sets the tone for subsequent interactions. Affects first impression. If the player has never interacted with this NPC before, `greet` gives a +2 bonus to the first reputation change. Subsequent `greet` actions have no bonus. |
| `ask_info` | Ask for information | Social | 1 | NPC present; reputation affects willingness |
| `persuade` | Persuade / convince | Social | 2 | NPC present |
| `trade` | Trade items | Social | 2 | NPC present; willingness depends on NPC role + reputation |
| `give_item` | Give item to NPC | Social | 1 | Has item, NPC present |
| `deceive` | Lie / bluff | Social | 2 | NPC present; success depends on reputation + NPC personality |
| `intimidate` | Threaten / intimidate | Social | 2 | NPC present; outcome depends on NPC strength vs. player |
| `attack` | Attack / fight | Combat | 10 | Target exists (NPC, creature, object) |
| `defend` | Defend / block | Combat | 5 | In active combat |
| `flee` | Flee / run away | Combat/Nav | 5 | — (success depends on stamina + context) |
| `sneak` | Sneak / move stealthily | Stealth | 5 | — (success depends on context, time, NPCs nearby) |
| `hide` | Hide from view | Stealth | 3 | Hiding spots contextually determined |
| `steal` | Steal item | Stealth | 5 | Target has items; detection risk based on context |
| `pick_up` | Pick up item | Utility | 1 | Item exists at location |
| `use_item` | Use inventory item | Utility | 2 | Has usable item in inventory |
| `eat` | Eat food | Utility | 1 | Has food item |
| `rest` | Rest / wait | Utility | 0 | — (always available, recovers +10 AP; requires safe/indoor location or no active combat) |
| `wait` | Wait and observe | Utility | 0 | — (always available, advances time; may overhear nearby NPC conversations or notice environmental details — passive perception check) |
| `drop_item` | Drop an item | Utility | 0 | Has non-quest item in inventory |
| `status` | Check quest journal | Utility | 0 | — (always available; displays current objectives, clues, and quest progress) |
| `equip` | Equip weapon or armor | Utility | 1 | Has equipment-type item in inventory |
| `work` | Perform labor | Utility | 5 | Work context available at location |

#### Why Universal Action Space

1. **No artificial gating** — the player is never told "you can't do that here"; instead, actions produce contextually appropriate outcomes (attempting `steal` at the Elder's house has different consequences than at the tavern, but both are valid)
2. **Emergent gameplay** — the combination of universal actions + context-dependent outcomes creates situations the designer didn't explicitly script
3. **Richer MDP** — the transition function $T(s'|s,a)$ encodes the full complexity; actions that seem "wrong" for a context lead to interesting dynamic checkpoint generation
4. **Entity symmetry** — NPCs share the same universal action catalog, making their RL agents operate in the same space as the player
5. **Large action space, smart resolution** — instead of restricting the action space, the game resolves any action through precondition checks + context evaluation, generating dynamic outcomes for novel combinations

#### Action Resolution Pipeline

```
Entity selects action from universal catalog
        │
        ▼
┌─────────────────────────────────────┐
│  1. Precondition Check              │
│     • Hard preconditions (target    │
│       exists? has required item?)   │
│     → If hard-fail: narrate why     │
│       ("There's no one here to      │
│        talk to.") — costs 0 AP      │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  2. Context Evaluation              │
│     • Soft modifiers: location,     │
│       time, NPC mood, reputation,   │
│       player stats, inventory       │
│     → Compute success probability   │
│     → Compute outcome severity      │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  3. Outcome Resolution              │
│     • Roll against success prob     │
│     • Apply effects (HP, AP, rep)   │
│     • If action is quest-relevant:  │
│       transition to next checkpoint │
│     • If action is novel/unexpected:│
│       trigger dynamic CP generation │
│     • Narrate result                │
└─────────────────────────────────────┘
```

#### UI Presentation of Actions

The frontend groups all universal actions by **category** (Navigation, Social, Combat, Stealth, Exploration, Utility) in a compact, always-visible **action palette**. Context-**relevant** actions (those with high success probability given current state) are visually emphasized; others are **dimmed but always clickable**. This gives the player full agency while guiding attention.

### Player Input System

The player interacts through **two input modes**, both mapped to the universal action space:

| Input Mode | How It Works | Example |
|--------------------|-------------|--------|
| **Action Palette** | Categorized action buttons for the entire universal action catalog. Context-relevant actions are highlighted; all are always clickable. Direct 1:1 mapping to action IDs. | Player clicks **[Greet]** → action `greet` resolved against current context |
| **Free-Text Input** | Player types natural language in a chat box. Backend parses and maps to the closest matching action in the universal action catalog. | Player types *"I want to say hello to the guards"* → parsed → mapped to `greet` |

#### Text Input Parsing Pipeline

Every free-text input is analyzed along **3 dimensions** before action mapping:

| Dimension | What It Captures | Example Values | Effect on Game |
|-----------|-----------------|----------------|----------------|
| **Emotion** | The emotional tone of the player's input | `neutral`, `angry`, `fearful`, `curious`, `friendly`, `threatening` | Influences NPC reaction, can shift reputation, affects generated checkpoint tone |
| **Intent** | The gameplay action the player wants to perform | `greet`, `attack`, `explore`, `trade`, `ask_info`, `flee`, `unknown` | Maps to an action in the action space (or triggers dynamic checkpoint) |
| **Social** | The social posture toward NPCs / the world | `polite`, `rude`, `deceptive`, `honest`, `intimidating`, `neutral` | Modifies reputation delta, changes NPC dialogue, can unlock/lock options |

```
Player types free text
        │
        ▼
┌──────────────────────────────────────────┐
│  1. spaCy NLP pipeline                   │
│     • Tokenize, lemmatize, POS-tag       │
│     • Extract doc.vector for similarity  │
│     • Named entity recognition (NER)     │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  2. 3-Dimensional Analysis                           │
│     ┌────────────┐                                   │
│     │  Emotion   │ → angry / friendly / neutral / ...│
│     ├────────────┤                                   │
│     │  Intent    │ → greet / attack / explore / ...  │
│     ├────────────┤                                   │
│     │  Social    │ → polite / rude / deceptive / ... │
│     └────────────┘                                   │
│  (spaCy vectors + keyword rules OR LLM)              │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│  3. Intent → Action mapping              │
│     spaCy doc.similarity() against action│  Compare input vector to
│     labels + synonym map as fallback     │  each action's label vector
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────┐
│  4. Confidence check             │  score ≥ threshold?
└──────┬───────────┬───────────────┘
       │Yes        │No
       ▼           ▼
  Map to action   ┌─────────────────────────────────┐
  in action       │ 5a. If LLM available:           │
  space           │     LLM classifies intent       │
       │          │ 5b. Else:                       │
       │          │     Treat as unexpected action  │
       │          │     → trigger dynamic CP gen    │
       │          └─────────────────────────────────┘
       ▼
  Attach emotion + social
  metadata to the resolved
  action → affects NPC response,
  reputation delta, checkpoint tone
```

#### 3-Dimensional Analysis: Rules-Based Fallback

When LLM is not available, use keyword dictionaries:

```python
EMOTION_KEYWORDS = {
    "angry":       ["furious", "angry", "rage", "damn", "hate", "kill"],
    "friendly":    ["please", "kindly", "hello", "hi", "friend", "thanks"],
    "fearful":     ["scared", "afraid", "nervous", "run", "hide", "danger"],
    "curious":     ["wonder", "what", "why", "how", "interesting", "tell me"],
    "threatening": ["or else", "better", "warn", "threat", "regret"],
    "neutral":     []  # default if no keywords match
}

SOCIAL_KEYWORDS = {
    "polite":       ["please", "excuse me", "thank you", "sir", "ma'am", "kindly"],
    "rude":         ["idiot", "fool", "shut up", "get lost", "stupid", "out of my way"],
    "deceptive":    ["trick", "lie", "pretend", "disguise", "fool them", "bluff"],
    "honest":       ["truth", "honest", "truly", "really", "sincerely"],
    "intimidating": ["threaten", "scare", "force", "make them", "demand", "or else"],
    "neutral":      []  # default
}
```

> **Disambiguation rule:** Some keywords appear in both `EMOTION_KEYWORDS` and `SOCIAL_KEYWORDS` (e.g., "please", "kindly" appear in both `friendly` and `polite`). Both dimensions are evaluated **independently** — a single input can be classified as `emotion: friendly` AND `social: polite` simultaneously. The dimensions are not mutually exclusive; they capture different aspects of the same input.

#### How Dimensions Affect Gameplay

| Dimension Value | NPC Reaction | Reputation Modifier | Checkpoint Tone |
|----------------|-------------|--------------------|-----------------|
| **Emotion: angry** | NPC becomes defensive or scared | -2 to -5 bonus penalty | Tense, confrontational |
| **Emotion: friendly** | NPC is more open, shares extra info | +1 to +3 bonus | Warm, cooperative |
| **Social: polite** | NPC responds warmly, may give discount | +2 to +5 bonus | Welcoming |
| **Social: rude** | NPC is offended, may refuse help | -3 to -8 bonus penalty | Hostile, curt |
| **Social: deceptive** | Skill check — if detected: large rep loss; if undetected: advantage | -10 if caught, 0 if not | Suspenseful |
| **Social: intimidating** | Weak NPCs comply, strong NPCs resist or fight | -5 to -15 | Threatening |

#### Intent → Action Synonym Map (examples)

```python
ACTION_SYNONYMS = {
    "greet_guards": ["greet", "hello", "talk to guards", "say hi", "approach guards", "wave"],
    "sneak_past":   ["sneak", "stealth", "creep", "slip past", "avoid guards", "hide"],
    "search":       ["look around", "explore", "examine", "investigate", "search area"],
    "fight":        ["attack", "fight", "hit", "strike", "combat", "punch"],
    "trade":        ["buy", "sell", "trade", "barter", "shop", "purchase"],
    "rest":         ["rest", "sleep", "sit down", "take a break", "recover"],
    "wait":         ["wait", "observe", "watch", "hang around", "stay put"],
    "flee":         ["run", "flee", "escape", "run away", "retreat"],
}
```

#### Parsed Input Object

Every player input (button or text) produces a unified `ParsedInput` that the game engine consumes:

```python
ParsedInput = {
    "source": "button" | "text",        # how the input was provided
    "raw_text": str | None,              # original text (None for button clicks)
    "action_id": str | None,             # resolved action ID (None if unmatched)
    "target_npc": str | None,            # resolved NPC UID (None if no NPC target)
    "target_item": str | None,           # resolved item ID (None if no item target)
    "target_location": str | None,       # resolved location ID (for move_to; None otherwise)
    "confidence": float,                 # 1.0 for buttons, 0.0-1.0 for text
    "emotion": str,                      # "neutral", "angry", "friendly", "fearful", "curious", "threatening"
    "intent": str,                       # "greet", "attack", "explore", etc.
    "social": str,                       # "neutral", "polite", "rude", "deceptive", "honest", "intimidating"
}
# For button clicks: emotion="neutral", intent=action_type, social="neutral" (defaults)
#   Target resolved via: (1) NPC selection sub-menu if >1 valid target at location,
#   (2) auto-selected if exactly 1 valid target, (3) None if action needs no target.
# For text input: all 3 dimensions extracted; target resolved via NER + name matching
#   against NPCs at current location.
```

#### Key Design Decisions

1. **Both inputs produce the same `ParsedInput`** — the game engine sees a unified object; button clicks default to `neutral` emotion/social
2. **Universal action space, not limited choices** — every action in the global catalog is always available to every entity; the action resolution pipeline determines outcomes based on preconditions and context rather than restricting what actions can be attempted
3. **Unmatched text triggers dynamic generation** — if free-text intent doesn't map to any action in the universal catalog, it's treated as a truly novel action, triggering dynamic checkpoint generation (Section 6)
4. **Layered NLP strategy** — Keyword/synonym dictionaries are the **primary** matching strategy (fast, deterministic, reliable). spaCy `doc.similarity()` is used as a **secondary** signal when keyword matching is ambiguous. Note: `en_core_web_md` vectors are trained on general English; cosine similarity between similar action labels (e.g., "greet" vs "talk") will be very high (~0.8+), making vector-only disambiguation unreliable. The synonym map should be the first-pass matcher, with spaCy vectors as a tiebreaker. LLM (when loaded) provides the highest-accuracy 3D analysis as an optional third layer.
5. **Confidence threshold** — intent matching uses a similarity score; below the threshold (e.g., 0.5), the input is treated as a novel action
6. **Emotion & social ride along** — even when an action is successfully matched, the emotion and social dimensions still modify the outcome (NPC reactions, reputation delta, narrative tone)
7. **Multi-action compound inputs** — if free-text contains a compound intent (e.g., "go to the tavern and talk to Tessa"), the parser detects multiple intents and queues them as sequential actions. The first action is resolved immediately; the second is queued for the next turn with a notification: "Next action queued: talk to Tessa." Maximum queue depth: 1 (only 2-action compounds supported).
8. **Pre-computed action vectors** — at startup, spaCy vectors for all action labels in the universal catalog are pre-computed and cached (`ACTION_VECTORS: dict[str, np.ndarray]`). Free-text similarity matching compares the input vector against this cache, avoiding repeated `nlp()` calls. Cache is rebuilt only if the action catalog changes.

---

## 3. Formal MDP Specification – Quest System

### 3.1 Two-Level Hierarchical MDP

The quest system uses a **two-level hierarchy**:

#### **Level 1 – Macro MDP (Quest Stages)**

$$\mathcal{M}_{macro} = (S_{macro}, A_{macro}, T_{macro}, R_{macro}, \gamma_{macro})$$

| Component                 | Definition                                                                                                            |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------|
| $S_{macro}$               | Major quest stages: $\{S_1, S_2, \ldots, S_K\}$ + terminal states $\{S_{success}, S_{fail}\}$                         |
| $A_{macro}$               | Emergent from micro-level: a stage completes when the player reaches $Cp_{i,last}$ through any sequence of universal actions; `abandon_quest` is implicit (player stops progressing) |
| $T_{macro}(s' \mid s, a)$ | Transition: completing all checkpoints in $S_i$ moves to $S_{i+1}$ with $p=1$; deviation at micro level triggers dynamic checkpoint generation |
| $R_{macro}(s, a)$         | Reward for stage completion (reputation gains, item rewards)                                                          |
| $\gamma_{macro}$          | Discount factor (1.0 for quest — all stages matter equally)                                                           |

#### **Level 2 – Micro MDP (Checkpoints within a Stage)**

$$\mathcal{M}_{micro}^{(i)} = (S_{micro}^{(i)}, A_{micro}^{(i)}, T_{micro}^{(i)}, R_{micro}^{(i)}, \gamma_{micro})$$

| Component                         | Definition                                                                                                          |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------|
| $S_{micro}^{(i)}$                 | Checkpoints within stage $S_i$: $\{Cp_{i,1}, Cp_{i,2}, \ldots, Cp_{i,n_i}\}$ + dynamically generated $Cp_{i,n_i+k}$ |
| $A_{micro}^{(i)}$                 | **Universal action catalog** $A_{global}$: all actions available at every checkpoint; preconditions + context determine outcome, not availability |
| $T_{micro}^{(i)}(cp' \mid cp, a)$ | Transition probabilities between checkpoints based on action + game state                                           |
| $R_{micro}^{(i)}(cp, a)$          | Immediate reward/cost (stamina cost, health change, item gain, reputation change)                                   |
| $\gamma_{micro}$                  | 0.95 (slight discount to prefer efficient paths)                                                                    |

### 3.2 Transition Rules

```
Standard transition:
  Cp_i_j --[expected_action]--> Cp_i_(j+1)     probability: 0.8-1.0

Deviation transition:
  Cp_i_j --[unexpected_action]--> Cp_i_(n+1)   (dynamically generated)

Stage completion:
  Cp_i_last --[complete]--> S_(i+1), Cp_(i+1)_1

Failure:
  Cp_i_j --[fatal_action]--> S_fail             (e.g., health reaches 0)
```

### 3.3 State Representation

Each checkpoint state is a tuple:

```python
CheckpointState = {
    "stage_id": int,           # which major stage (1-K)
    "checkpoint_id": str,      # checkpoint ID within stage (e.g., "1_1", "1_D1") — string to support dynamic IDs
    "is_dynamic": bool,        # was this auto-generated?
    "description": str,        # narrative text
    "highlighted_actions": list,  # quest-relevant actions to emphasize in UI (subset of universal catalog)
    "context": dict,           # NPCs present, objects, environmental details — used by action resolution
    "requirements": dict,      # e.g., {"min_health": 10, "has_item": "key"}
    "effects": dict,           # what happens on arrival (e.g., stamina_cost: 5)
    "next_expected": str,      # next checkpoint ID in standard flow
    "nudge_target": str        # which checkpoint ID to nudge toward if deviated
}
# NOTE: All universal actions are always available. `highlighted_actions` only controls
# which buttons are visually emphasized in the UI — the player can always choose any action.
```

---

## 4. NPC RL Agent Specification

### 4.1 NPC MDP

Each NPC operates on its own MDP:

$$\mathcal{M}_{npc} = (S_{npc}, A_{npc}, T_{npc}, R_{npc}, \gamma_{npc})$$

| Component                         | Definition                                                                                                   |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------|
| $S_{npc}$                         | Tuple: `(location, time_of_day, energy, mood)` — discretized for Q-table (see §4.3). Additional context (`has_item`, `recent_event`) is used by the action resolution pipeline but not part of the Q-table state to keep the state space tractable. |
| $A_{npc}$                         | **Universal action catalog** $A_{global}$ (same as player): all 27 universal actions — preconditions + NPC personality determine which actions are chosen. For `move_to`, a secondary location-selection heuristic chooses the destination (see §4.3 Movement Resolution). |
| $T_{npc}$                         | Stochastic: actions succeed with probability based on energy/health; random events can occur                 |
| $R_{npc}$                         | **Multi-objective**: weighted sum of happiness, income, health, reputation changes                           |
| $\gamma_{npc}$                    | 0.9 (NPCs care about near-future more)                                                                       |

### 4.2 Reward Function

$$R_{npc}(s, a) = w_h \cdot \Delta happiness + w_i \cdot \Delta income + w_{hp} \cdot \Delta health + w_r \cdot \Delta reputation$$

Default weights (tunable per NPC personality — full table in §4.4):

| NPC Type | $w_h$  | $w_i$  | $w_{hp}$ | $w_r$  |
|----------|--------|--------|----------|--------|
| Elder    | 0.1    | 0.1    | 0.3      | 0.5    |
| Farmer   | 0.2    | 0.4    | 0.3      | 0.1    |
| Merchant | 0.1    | 0.5    | 0.2      | 0.2    |
| Tavern Keeper | 0.5 | 0.2  | 0.1      | 0.2    |
| Guard    | 0.1    | 0.2    | 0.3      | 0.4    |
| Villager | 0.3    | 0.2    | 0.3      | 0.2    |

### 4.3 Q-Learning Implementation

```
Algorithm: Tabular Q-Learning
─────────────────────────────
State space:  Discretized (location × time_slot × energy_level × mood_level)
              ~5 locations × 4 time_slots × 3 energy_levels × 3 mood_levels = 180 states
Action space: 27 universal actions (shared catalog, same for all entities)
              NPC personality weights bias action selection but do not restrict it
Q-table size: 180 × 27 = 4,860 entries per NPC (still trivially small)

Hyperparameters:
  α (learning rate):    0.1
  γ (discount):         0.9
  ε (exploration):      0.5 during pre-training → 0.15 at live-play start (turn 20)
                        → decay to 0.05 over episodes (see Cold-start below)

Cold-start mitigation:
  For the first 20 game turns, NPCs use their fallback schedule as the
  primary policy (ε-greedy is disabled). After turn 20, ε-greedy is enabled
  starting at ε=0.15 (not 0.3) and decays to 0.05. This prevents NPCs from
  behaving erratically (guards randomly attacking, farmers stealing) during
  early gameplay when Q-tables are empty.

Update rule:
  Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]

Action masking (soft):
  Actions with hard-failed preconditions get Q-value penalty of -∞ during
  action selection (argmax), but the Q-table still stores learned values.
  This lets NPCs learn that certain actions are useless in certain states
  without shrinking the formal action space.

Fallback policy:
  If Q-table has no data for state → use preprogrammed schedule

Movement Resolution (for `move_to` action):
  The Q-table selects `move_to` as an action, but does NOT encode the
  destination. A secondary heuristic selects where to move:
    1. If fallback schedule specifies a location for current time → go there
    2. Else if NPC has a goal location (e.g., knows player is at X) → go there
    3. Else choose a random adjacent location weighted by personality:
       - Guards prefer gate/village_center
       - Farmers prefer fields
       - Tavern keeper prefers tavern
    4. Hard constraint: destination must be adjacent (see location adjacency graph)
  This keeps the Q-table compact while still producing sensible movement.
```

#### Pre-Training Specification

Before the game starts, each NPC is pre-trained with **100 simulated episodes** to seed reasonable Q-values:

```
Pre-training procedure:
  1. For each NPC independently:
     a. Initialize Q-table to zeros
     b. Simulate 100 episodes of 50 turns each
     c. Each episode: NPC starts at its default location, time cycles normally
     d. NPC selects actions via ε-greedy (ε=0.5 during pre-training)
     e. No player is present — NPC interacts only with the world and other
        NPCs (who also use fallback schedules during pre-training)
     f. Rewards are computed as normal (multi-objective weighted sum)
  2. After pre-training, Q-tables are saved to NPC instance JSON files
  3. Pre-training runs once at first launch; subsequent launches load saved Q-tables
  4. Pre-training is deterministic given the random seed (see §2 config)

Lightweight Pre-Training Mode:
  During pre-training, the following systems are DISABLED to reduce startup time:
  - Narration generation (no template rendering or LLM calls)
  - Event logging (no EventLogEntry creation)
  - Witness detection (no NPC knowledge updates)
  - Gossip propagation (no cross-NPC reputation changes)
  Only state transitions + reward computation + Q-table updates run.
  Expected startup time: <5 seconds for 6 NPCs × 100 episodes × 50 turns.
  A loading indicator is shown in the UI during pre-training.
```

> **Design Note:** Pre-training against a world with no player ensures NPCs learn sensible daily routines (farmer works in fields, guard patrols gate) without learning player-specific strategies. Live gameplay then refines these tables.

### 4.4 NPC Personality Archetypes & Unique Identity

The game uses **generic personality archetypes** — reusable personality templates that can be instantiated into individual NPCs. Every NPC instance receives a **unique ID** at creation time so it can be tracked, referenced, and remembered throughout the game.

#### Architecture

```
Personality Archetype (template)         NPC Instance (runtime)
┌───────────────────────────┐           ┌────────────────────────────────┐
│ archetype: "guard"        │           │ npc_uid: "guard_04a7"          │
│ base_personality: ...     │  ──spawn──▶ │ name: "Aldric"                 │
│ reward_weights: ...       │           │ archetype: "guard"             │
│ dialogue_templates: ...   │           │ personality: (from archetype)  │
│ fallback_schedule: ...    │           │ stats: (initialized from arch) │
└───────────────────────────┘           │ q_table: {}                    │
                                        │ conversation_history: []       │
                                        └────────────────────────────────┘
```

#### Unique ID Generation

Every NPC instance gets a **unique identifier** at creation:

```python
import random as _random

def create_npc(archetype: str, name: str, location: str, master_seed: int, index: int) -> dict:
    """Create NPC with deterministic UID derived from master seed + index."""
    npc_seed = master_seed + hash(f"{archetype}_{name}_{index}") & 0xFFFF
    uid_hex = f"{npc_seed:04x}"
    return {
        "npc_uid": f"{archetype}_{uid_hex}",  # e.g., "guard_a3f1" — deterministic given seed
        "name": name,
        "archetype": archetype,
        # ... initialized from archetype template
    }
```

- `npc_uid` is the **primary key** — all references in game state, conversation logs, reputation tracking, and quest data use this UID
- UIDs are **deterministic** given the master seed, so the same seed always produces the same UIDs (important for reproducibility)
- Allows multiple NPCs of the same archetype (e.g., two guards at the gate, three villagers in the center)
- UIDs persist across save/load so references remain consistent

#### Personality Archetypes

| Archetype    | Base Personality               | $w_h$ | $w_i$ | $w_{hp}$ | $w_r$ | Typical Locations |
|--------------|--------------------------------|-------|-------|----------|-------|-------------------|
| `elder`      | Wise, cautious, knowledgeable  | 0.1   | 0.1   | 0.3      | 0.5   | Elder's House     |
| `farmer`     | Hardworking, honest, practical | 0.2   | 0.4   | 0.3      | 0.1   | Fields, Barn      |
| `merchant`   | Shrewd, opportunistic, social  | 0.1   | 0.5   | 0.2      | 0.2   | Village Center, Gate |
| `tavern_keeper` | Friendly, gossipy, observant | 0.5   | 0.2   | 0.1      | 0.2   | Tavern            |
| `guard`      | Dutiful, suspicious, loyal     | 0.1   | 0.2   | 0.3      | 0.4   | Gate, Village Center |
| `villager`   | Curious, nervous, cooperative  | 0.3   | 0.2   | 0.3      | 0.2   | Village Center, Fields |

> **Constraint:** Reward weights for each archetype **must sum to 1.0**. This is validated at archetype load time.

Each archetype defines:

- **Base personality traits** — text description used in LLM prompts and dialogue tone
- **Reward weights** — for Q-learning behavior (must sum to 1.0)
- **Combat stats** — `base_attack` and `base_defense` values for combat resolution (see §5.6)
- **Dialogue templates** — scripted responses for common interactions (greeting, quest hints, hostility)
- **Fallback schedule** — daily routine when Q-table has no learned policy
- **Movement weights** — per-location preference weights for `move_to` destination selection

#### Archetype Combat Stats

| Archetype | base_attack | base_defense | max_hp | Notes |
|-----------|-------------|-------------|--------|-------|
| `elder`   | 2 | 2 | 40 | Non-combatant; flees at first sign of danger |
| `farmer`  | 6 | 4 | 80 | Farm tools as improvised weapons; sturdy |
| `merchant`| 3 | 3 | 60 | Avoids combat; relies on guards |
| `tavern_keeper` | 5 | 5 | 70 | Bar brawler; can hold their own |
| `guard`   | 12 | 10 | 120 | Trained combatant; armored |
| `villager` | 3 | 2 | 50 | Untrained; panics in combat |

> **Note:** `max_hp` is the NPC's hit-point pool for combat resolution (see §5.6). This is **separate** from the Q-learning `stats.health` dimension, which tracks the NPC's general well-being for reward calculations (see §4.2). When `current_hp` reaches 0, the NPC is incapacitated (see §5.6 NPC Death / Incapacitation).

#### NPC Instance Tracking

The game state maintains a registry of all NPC instances:

```python
npc_registry = {
    "elder_m8b2": {"name": "Elder Maren", "archetype": "elder", ...},
    "farmer_j4a1": {"name": "Farmer Jak", "archetype": "farmer", ...},
    "tavkeeper_t9c3": {"name": "Tessa", "archetype": "tavern_keeper", ...},
    "guard_a3f1": {"name": "Aldric", "archetype": "guard", ...},
    "guard_b7e2": {"name": "Bryn", "archetype": "guard", ...},
    "villager_c1d4": {"name": "Old Petra", "archetype": "villager", ...},
    # ... more instances as needed
}
```

When the player references an NPC (via text input or action palette), the system resolves the target using:

1. Exact name match against NPCs at current location
2. Archetype match if ambiguous ("talk to guard" → pick nearest/first guard)
3. UID if referenced from conversation history or quest data

### 4.5 MVP NPCs

| NPC                       | UID (example)      | Archetype      | Personality                  | Key Interactions                               |
|---------------------------|--------------------|-----------------|-----------------------------|------------------------------------------------|
| **Elder Maren**           | `elder_m8b2`       | `elder`         | Wise, cautious ($w_r = 0.5$) | Gives main quest, provides hints               |
| **Farmer Jak**            | `farmer_j4a1`      | `farmer`        | Hardworking ($w_i = 0.4$)    | Can find/hide items, helps or hinders player   |
| **Tessa (Tavern keeper)** | `tavkeeper_t9c3`   | `tavern_keeper` | Social ($w_h = 0.5$)         | Shares rumors, trades, overhears conversations |
| **Aldric (Gate guard)**   | `guard_a3f1`       | `guard`         | Dutiful ($w_r = 0.4$)        | Guards the gate, questions visitors            |
| **Bryn (Gate guard)**     | `guard_b7e2`       | `guard`         | Suspicious ($w_r = 0.4$)     | Guards the gate, patrols                       |
| **Old Petra (Villager)**  | `villager_c1d4`    | `villager`      | Curious ($w_h = 0.3$)        | Background NPC, shares village gossip          |

### 4.6 NPC-to-NPC Interactions

NPCs don't just react to the player — they **interact with each other** during their turns. Each game turn, after the player acts, every NPC selects an action from the universal catalog. When NPCs are at the same location, they can target each other for social, trade, or even combat actions.

#### NPC-to-NPC Action Selection

```python
def npc_select_action(npc, game_state):
    """NPC chooses action using Q-learning policy (or fallback schedule)."""
    state = discretize_npc_state(npc, game_state)
    
    # Check if other NPCs are at same location
    colocated_npcs = [n for n in game_state.npcs.values() 
                      if n.location == npc.location and n.npc_uid != npc.npc_uid]
    player_present = game_state.player.location == npc.location
    
    # Q-learning selects action (with ε-greedy exploration)
    action = epsilon_greedy(npc.q_table, state, epsilon=npc.epsilon)
    
    # Resolve target: player if present, else colocated NPC, else self/environment
    target = resolve_npc_target(action, npc, colocated_npcs, player_present)
    
    return action, target
```

#### Interaction Types

| Interaction | When It Happens | Effect |
|-------------|----------------|--------|
| **NPC talks to NPC** | Two NPCs at same location during social time slots | Exchange gossip about player (reputation propagation), share event knowledge |
| **NPC trades with NPC** | Merchant + any NPC at same location | Items exchange between NPCs, adjusting their inventories and income stats |
| **NPC gossips** | Any social interaction between NPCs | Player reputation propagates: if NPC_A knows about player's deed, NPC_B may learn it (see §5.5 Gossip Propagation) |
| **NPC warns NPC** | NPC has hostile reputation with player | Spreads distrust — nearby NPCs pre-adjust their disposition toward player |
| **NPC conflict** | Two NPCs with conflicting goals (rare) | Resolved via combat resolution (see §5.6), can create dynamic events |

#### Gossip as Information Network

```
                    Turn 5: Player steals from Farmer Jak
                                │
                    ┌───────────┴──────────────┐
                    │ Farmer Jak (direct)       │
                    │ rep: -20 (theft)          │
                    │ knows_event: ✓            │
                    └───────────┬──────────────┘
                                │ Turn 7: Jak talks to Tessa at Tavern
                    ┌───────────┴──────────────┐
                    │ Tessa (gossip, 40% prob)  │
                    │ rep: -10 (50% decay)      │
                    │ knows_event: ✓ (via Jak)  │
                    └───────────┬──────────────┘
                                │ Turn 10: Tessa chats with Aldric
                    ┌───────────┴──────────────┐
                    │ Aldric (2nd-hand gossip)  │
                    │ rep: -5 (50% decay again) │
                    │ knows_event: ✓ (via Tessa)│
                    └──────────────────────────┘
```

> **Design Note:** NPC-to-NPC interactions create an organic information network. The player's actions have ripple effects through the village. A theft witnessed by one NPC eventually reaches others, creating consequences that feel natural rather than scripted.

---

## 5. Core Game Systems

### 5.1 Player Stats

#### Initial Player State

Every new game starts with the following player state:

```python
INITIAL_PLAYER = {
    "name": "Traveler",
    "health": 100,
    "max_health": 100,
    "stamina": 50,
    "max_stamina": 50,
    "combat_stats": {
        "base_attack": 8,
        "base_defense": 3,
        "weapon_modifier": 0,   # 0 = unarmed; updated when player equips a weapon
        "armor_modifier": 0    # 0 = unarmored; updated when player equips armor
    },
    "equipped": {
        "weapon": None,         # item ID of equipped weapon (or None)
        "armor": None           # item ID of equipped armor (or None)
    },
    "reputation": {},         # empty dict — will be populated with NPC UIDs as interactions happen
    "global_reputation": 0,
    "inventory": [
        {"id": "travel_papers", "name": "Travel Papers", "type": "quest", "quest_relevant": True,
         "description": "Official documents permitting travel to Thornhaven.",
         "effects": {}, "slot": None, "stat_modifiers": None},
        {"id": "bread", "name": "Stale Bread", "type": "consumable", "quest_relevant": False,
         "description": "A day-old loaf. Restores 5 HP.", "effects": {"heal": 5},
         "slot": None, "stat_modifiers": None},
        {"id": "coin_pouch", "name": "Coin Pouch", "type": "misc", "quest_relevant": False,
         "description": "A small pouch with 10 copper coins. Useful for trading.",
         "effects": {}, "slot": None, "stat_modifiers": None}
    ],
    "max_inventory": 10,
    "location": "gate",       # game starts at the village gate
    "quest_state": {
        "quest_id": "main_quest_01",
        "current_stage": 1,
        "current_checkpoint": "1_1",
        "completed_checkpoints": [],
        "dynamic_checkpoints": [],
        "deviation_count": 0
    }
}
```

#### Player Data Model

```python
Player = {
    "name": str,
    "health": int,          # 0–100, death at 0
    "max_health": 100,
    "stamina": int,         # 0–50, actions cost stamina
    "max_stamina": 50,
    "combat_stats": dict,   # {base_attack, base_defense, weapon_modifier, armor_modifier}
    "equipped": dict,       # {weapon: str|None, armor: str|None} — item IDs
    "reputation": dict,     # per-NPC reputation: {npc_uid: int} — each -100 to +100
    "global_reputation": int,  # weighted average of all per-NPC reps (read-only, for display)
    "inventory": list,      # list of Item objects
    "max_inventory": 10,    # carry limit
    "location": str,        # current location ID
    "quest_state": dict     # current stage + checkpoint
}

# Per-NPC Reputation example:
# player.reputation = {
#     "elder_m8b2": 25,      # Elder Maren thinks well of you
#     "farmer_j4a1": 10,     # Farmer Jak is warming up
#     "guard_a3f1": -15,     # Aldric is suspicious of you
#     "tavkeeper_t9c3": 0,   # Tessa is neutral
#     "guard_b7e2": -15,     # Bryn mirrors Aldric (gossip propagation)
#     "villager_c1d4": 5,    # Old Petra heard good things
# }
# global_reputation = weighted_avg(all per-NPC reps)  # for quick UI display
```

### 5.2 Health System

| Event         | Effect               |
|---------------|----------------------|
| Combat (win)  | -5 to -20 HP         |
| Combat (lose) | -30 to -50 HP        |
| Rest at home  | +20 HP               |
| Eat food      | +5 to +15 HP         |
| Poison/trap   | -10 to -30 HP        |
| Health = 0    | Quest fails (S_fail) |

### 5.3 Stamina / Action Points

All universal actions have a base AP cost (see Global Action Catalog in Section 2). Summary:

| Action Category | Actions | Base Cost |
|-----------------|---------|----------|
| Navigation | `move_to` | 3 AP |
| Exploration | `look` | 1 AP |
| Exploration | `examine` | 2 AP |
| Exploration | `search` | 5 AP |
| Social | `talk`, `greet`, `ask_info`, `give_item` | 1 AP |
| Social | `trade`, `persuade`, `deceive`, `intimidate` | 2 AP |
| Combat | `attack` | 10 AP |
| Combat | `defend`, `flee` | 5 AP |
| Stealth | `sneak`, `steal` | 5 AP |
| Stealth | `hide` | 3 AP |
| Utility | `use_item`, `eat` | 1-2 AP |
| Utility | `work` | 5 AP |
| Utility | `rest` | 0 AP (recovers +10 AP; requires safe location or no active combat) |
| Utility | `wait` | 0 AP (advances time, passive perception check) |
| Utility | `drop_item` (0 AP), `status` (0 AP), `equip` (1 AP) | 0–1 AP |
| Utility | `pick_up` | 1 AP |

Stamina regenerates +5 per game-turn passively. Stamina cannot exceed `max_stamina` (50); any regen beyond the cap is lost.

**At 0 stamina:** The player can only perform 0-AP actions (`rest`, `wait`, `status`, `drop_item`) plus `talk` and `greet` at 0 AP (social actions become free when exhausted to prevent soft-locks). In combat situations, `defend` costs 0 AP when at 0 stamina, and `flee` costs 0 AP but success rate drops to 40%. The player cannot `attack` at 0 AP.

**`wait` vs. `rest`:** These are distinct actions with different purposes:

- `rest` recovers +10 AP but requires a safe/indoor location or no active combat. If attempted during combat or at an unsafe location, it resolves as `wait` instead.
- `wait` advances time with 0 AP cost and triggers a **passive perception check**: the player may automatically notice important environmental details, overhear nearby NPC conversations, or detect approaching events. Success probability: 40% base, +20% if at a social location (tavern, village center).

### 5.4 Inventory

```python
Item = {
    "id": str,
    "name": str,
    "type": "quest" | "consumable" | "equipment" | "misc",
    "description": str,
    "effects": dict,       # e.g., {"heal": 10} or {"reputation": +5}
    "quest_relevant": bool, # cannot be discarded if True
    "slot": str | None,     # for equipment: "weapon" or "armor" (None for non-equipment)
    "stat_modifiers": dict | None  # for equipment: {"attack": +N} or {"defense": +N}
}
```

#### Equipment Items

Equipment-type items can be equipped via the `equip` action. Only one weapon and one armor can be equipped at a time. Equipping a new item in an occupied slot automatically unequips the previous item back to inventory.

| Item ID | Name | Slot | Stat Modifier | Source |
|---------|------|------|--------------|--------|
| `rusty_sword` | Rusty Sword | weapon | attack +3 | Found at fields (old oak) |
| `kitchen_knife` | Kitchen Knife | weapon | attack +1 | Trade from Tessa |
| `guard_sword` | Guard's Sword | weapon | attack +5 | Looted from incapacitated guard |
| `leather_vest` | Leather Vest | armor | defense +2 | Trade from merchant or quest reward |
| `iron_shield` | Iron Shield | armor | defense +4 | Quest reward (Stage 7) |

### 5.5 Per-NPC Reputation System

Reputation is tracked **per-NPC**, not globally. Each NPC maintains its own opinion of the player based on direct interactions and **gossip propagation** from other NPCs.

#### Direct Reputation Changes

When the player acts, reputation changes apply to the **target NPC** and any **witness NPCs** at the same location:

| Action               | Target NPC | Witnesses (same location) |
|----------------------|------------|---------------------------|
| Greet NPC (first time)| +2        | +1                        |
| Greet NPC (subsequent)| +0        | +0                        |
| Help NPC             | +5 to +15  | +2 to +5                  |
| Complete quest stage | +10        | +3                        |
| Steal from NPC       | -10 to -20 | -5 to -10                 |
| Attack NPC           | -20 to -30 | -10 to -20                |
| Lie (detected)       | -5 to -15  | -3 to -8                  |
| Share info           | +3 to +8   | +1 to +3                  |
| Give item            | +5 to +10  | +2 to +4                  |
| Trade (fair)         | +2 to +5   | +1                        |
| Intimidate           | -5 to -15  | -3 to -8                  |

**Dialogue Dimension Modifiers** (applied on top of base reputation change to **target NPC only**):

| Dimension:Value | Reputation Modifier |
|-----------------|--------------------|
| Emotion: friendly | +1 to +3 |
| Emotion: angry | -2 to -5 |
| Emotion: threatening | -3 to -8 |
| Social: polite | +2 to +5 |
| Social: rude | -3 to -8 |
| Social: deceptive (detected) | -10 |
| Social: intimidating | -5 to -15 |
| Social: honest | +1 to +3 |

> Example: Player greets guard Aldric (`guard_a3f1`, base: +2 rep) with `emotion: friendly` (+2) and `social: polite` (+3) → Aldric gets **+7 reputation**. Bryn (`guard_b7e2`) witnesses this and gets **+3 reputation** (witness bonus).

#### Gossip Propagation

NPCs who **were not present** at an event can learn about it through **gossip** during NPC-to-NPC social interactions (see Section 4.6):

```python
def propagate_gossip(source_npc, target_npc, event):
    """When NPCs interact socially, they may share knowledge about the player."""
    gossip_probability = 0.4  # 40% chance per social interaction
    decay_factor = 0.5        # gossip carries half the original rep change
    max_gossip_hops = 3       # gossip stops after 3 hops from original witness
    min_gossip_delta = 2      # don't bother gossiping about trivial rep changes
    
    # Check termination conditions
    hop_count = event.get("gossip_hops", 0)
    if hop_count >= max_gossip_hops:
        return  # gossip chain exhausted
    
    original_delta = event["reputation_change"]
    gossip_delta = int(original_delta * decay_factor)
    
    if abs(gossip_delta) < min_gossip_delta:
        return  # reputation change too small to be worth gossiping about
    
    if random() < gossip_probability:
        target_npc.reputation_toward_player += gossip_delta
        target_npc.known_events.append({
            "event": event,
            "source": "gossip",
            "from_npc": source_npc.npc_uid,
            "decayed_delta": gossip_delta,
            "gossip_hops": hop_count + 1  # track hop count for termination
        })
```

> **Gossip cascade limit:** To prevent reputation floods when multiple NPCs are co-located, each NPC may propagate gossip about the player **at most once per turn**. If 4 NPCs are at Village Center and all choose `talk`, only 1 gossip event fires per NPC pair (not 6 cascading events).

#### Global Reputation (Derived)

For display purposes and coarse-grained checks, a **global reputation** is computed as the weighted average of all per-NPC reputations:

```python
def compute_global_reputation(player):
    if not player.reputation:
        return 0
    return sum(player.reputation.values()) // len(player.reputation)
```

#### Per-NPC Reputation Thresholds

Each NPC individually responds based on their specific reputation with the player:

| Range        | Label      | NPC Reaction                       |
|--------------|------------|------------------------------------|
| 50+          | Trusted    | Shares secrets, gives discounts, offers help proactively |
| 20–49        | Friendly   | Normal helpful behavior, answers questions willingly |
| -19–19       | Neutral    | Standard transactional responses   |
| -49–-20      | Suspicious | Withholds info, charges more, watches player closely |
| -50 or below | Hostile    | Refuses to help, may alert others, may attack |

> **Design Note:** Per-NPC reputation creates natural consequences — helping Farmer Jak doesn't automatically make guard Aldric trust you. The player must build relationships individually, though gossip provides indirect reputation transfer over time.

#### Reputation Decay

To make sustained relationship maintenance meaningful, all per-NPC reputations **decay toward neutral** over time:

```python
REPUTATION_DECAY_INTERVAL = 20   # every 20 turns
REPUTATION_DECAY_AMOUNT = 1      # ±1 toward 0

def apply_reputation_decay(player, current_turn):
    """Every REPUTATION_DECAY_INTERVAL turns, nudge all reputations toward 0."""
    if current_turn % REPUTATION_DECAY_INTERVAL != 0:
        return
    for npc_uid, rep in player.reputation.items():
        if rep > 0:
            player.reputation[npc_uid] = max(0, rep - REPUTATION_DECAY_AMOUNT)
        elif rep < 0:
            player.reputation[npc_uid] = min(0, rep + REPUTATION_DECAY_AMOUNT)
```

> This prevents permanent first-impression locks and rewards ongoing engagement with NPCs.

### 5.6 Combat Resolution Mechanic

When `attack`, `defend`, or `flee` actions are triggered (by player or NPC), combat is resolved through a **stat-based probabilistic model**.

#### Combat Stats

```python
CombatStats = {
    "base_attack": int,      # base damage output (5-20 depending on entity)
    "base_defense": int,     # damage reduction (0-10)
    "weapon_modifier": int,  # bonus from equipped weapon (0 = unarmed)
    "armor_modifier": int,   # bonus from equipped armor (0 = unarmored)
    "stamina_factor": float, # current_stamina / max_stamina (affects all rolls)
    "reputation_factor": float  # NPC willingness to fight (hostile NPCs fight harder)
}
```

#### Success Probability Formula

$$P(\text{hit}) = \text{clamp}\left(\frac{\text{attacker\_attack} + \text{weapon\_mod} + \text{stamina\_factor} \times 10}{(\text{defender\_defense} + \text{armor\_mod}) \times 2 + 20}, \ 0.1, \ 0.95\right)$$

#### Damage Calculation

```python
def resolve_combat(attacker, defender):
    """Resolve a single combat exchange."""
    hit_prob = compute_hit_probability(attacker, defender)
    
    if random() < hit_prob:
        # Hit — compute damage
        base_damage = attacker.base_attack + attacker.weapon_modifier
        defense = defender.base_defense + defender.armor_modifier
        damage = max(1, base_damage - defense + randint(-3, 3))  # min 1 damage on hit
        
        return {
            "hit": True,
            "damage": damage,
            "attacker_stamina_cost": 10,
            "defender_stamina_cost": 5,
            "narrative": f"{attacker.name} strikes {defender.name} for {damage} damage!"
        }
    else:
        # Miss
        return {
            "hit": False,
            "damage": 0,
            "attacker_stamina_cost": 10,
            "defender_stamina_cost": 2,
            "narrative": f"{attacker.name} swings at {defender.name} but misses!"
        }
```

#### Combat Flow

```
Player selects `attack` targeting NPC (or NPC selects `attack`)
        │
        ▼
┌─────────────────────────────────────┐
│  1. Compute hit probability          │
│     (attacker stats vs defender)     │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  2. Roll against P(hit)              │
│     → Hit: compute damage            │
│     → Miss: narrate miss             │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  3. Apply effects                    │
│     • Subtract HP from defender      │
│     • Subtract AP from both          │
│     • Major reputation penalty for   │
│       attacking non-hostile NPCs     │
│     • Witnesses get rep penalty too  │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  4. Check combat end conditions      │
│     • Defender HP = 0 → defeated     │
│     • Attacker can choose `flee`     │
│     • NPC may choose `flee` if low   │
│       HP/stamina (Q-learning policy) │
└─────────────────────────────────────┘
```

#### Defend & Flee Actions

| Action | Effect |
|--------|--------|
| `defend` | Next incoming attack has -50% damage; costs 5 AP |
| `flee` | 70% success chance (modified by stamina). Success: exit combat, move to adjacent location. Failure: enemy gets free attack at +20% hit chance |

> **Design Note:** Combat is intentionally simple (single roll per turn) to keep the game's focus on social interaction and quest progression. NPCs should rarely initiate combat — it primarily results from player aggression or extreme reputation penalties.

#### NPC Death & Incapacitation

When an NPC's health reaches 0, they are **incapacitated**, not permanently killed:

| Scenario | Resolution |
|----------|------------|
| **Non-quest-critical NPC** (Petra, Bryn) | NPC is incapacitated: removed from location, marked `status: "incapacitated"`. Returns after 20 turns with full health at their home location. Reputation with player set to -80 (hostile). All witness NPCs get -15 rep. |
| **Quest-critical NPC** (Elder Maren, Farmer Jak) | NPC is incapacitated at 1 HP instead of 0 ("They collapse but cling to life..."). NPC becomes permanently hostile (`reputation: -100`). A dynamic checkpoint is generated offering a redemption path. The quest can still be completed but becomes significantly harder. |
| **Guard NPCs** (Aldric, Bryn) | If both guards are incapacitated, a `guard_reinforcement` random event triggers within 5 turns (new temporary guard NPC arrives). Gate access rules remain unchanged. |

> **Design Note:** Permanent NPC death is avoided in the MVP to prevent unrecoverable quest states. Incapacitation provides consequences without dead ends.

#### NPC Health Recovery

NPCs recover HP passively and through their own actions:

| Recovery Method | HP Restored | Condition |
|----------------|-------------|-----------|
| Passive regen | +2 HP/turn | NPC is at an indoor location (tavern, elder's house — see §10.1 Type column) |
| NPC `rest` action | +10 HP | NPC selects `rest` via Q-learning or fallback schedule |
| NPC `eat` action | +5 HP | NPC selects `eat` (available during midday schedule) |
| Incapacitation return | Full HP | NPC returns after 20-turn incapacitation cooldown |

> **Note:** NPC HP recovery is intentionally slower than the player's to make combat consequences meaningful. NPCs at outdoor locations (gate, village center, fields — see §10.1) do not receive passive regen. NPC Q-learning naturally learns to `rest` when HP is low because `health` is a reward dimension (see §4.2).

#### Non-Combat Success Probability Formulas

Actions that involve skill checks (not just combat) use the following probability formulas:

$$P(\text{persuade}) = \text{clamp}\left(0.5 + \frac{\text{reputation}}{200} + \frac{\text{social\_modifier}}{20}, \ 0.1, \ 0.9\right)$$

$$P(\text{deceive}) = \text{clamp}\left(0.4 - \frac{\text{reputation}}{200} + \frac{\text{social\_modifier}}{20}, \ 0.05, \ 0.85\right)$$

$$P(\text{sneak}) = \text{clamp}\left(0.5 + \frac{\text{time\_bonus}}{10} - \frac{\text{npcs\_at\_location}}{10}, \ 0.1, \ 0.9\right)$$

$$P(\text{steal}) = P(\text{sneak}) \times 0.8$$

$$P(\text{hide}) = \text{clamp}\left(0.6 + \frac{\text{time\_bonus}}{10} - \frac{\text{npcs\_at\_location}}{10}, \ 0.15, \ 0.95\right)$$

$$P(\text{search\_discovery}) = \text{clamp}\left(0.3 + \frac{\text{search\_count\_here}}{5} \times 0.1, \ 0.2, \ 0.8\right)$$

Where:

- `reputation` = player's per-NPC reputation with the target (range: -100 to +100)
- `social_modifier` = bonus from social dimension: polite = +5, honest = +3, rude = -5, deceptive = -3, intimidating = varies
- `time_bonus` = +3 at night, +1 at evening, 0 at other times
- `npcs_at_location` = number of NPCs at current location
- `search_count_here` = number of times this location has been searched this session

### 5.7 Random Events System

Random events inject **unpredictability** into the game world, creating emergent situations that aren't tied to quest progression.

#### Event Types

| Event ID | Event | Trigger Condition | Probability/Turn | Duration | Effects |
|----------|-------|-------------------|-------------------|----------|---------|
| `weather_storm` | Thunderstorm | Turn > 10, time = evening/night | 5% | 4-8 turns | All outdoor `search` costs +3 AP; NPCs prefer indoor locations |
| `weather_fog` | Dense Fog | Turn > 5, time = morning | 8% | 4 turns | `sneak` success +20%; `look` effectiveness reduced |
| `wandering_merchant` | Traveling Merchant arrives | Turn > 8, no merchant at Gate | 3% | 8 turns | New temporary NPC at Gate; buys/sells rare items |
| `theft_event` | NPC reports theft | Player reputation < -10 with any NPC | 5% | — | Guards increase patrols; `steal` detection chance +15% |
| `npc_accident` | NPC gets injured | Random NPC in Fields/Gate | 2% | — | NPC health drops; player can help (+rep) or ignore |
| `festival_prep` | Village festival preparations | Turn > 20, global_rep > 10 | 4% | 12 turns | NPCs gather at Village Center; social actions give +50% reputation |
| `supply_shortage` | Supply shortage at tavern | Random, turn > 15 | 3% | 6 turns | Tessa can't serve food; `eat` at tavern unavailable; trade prices +25% |
| `lost_item` | NPC loses an item | Random NPC | 3% | — | New mini-objective: find and return item for reputation boost |

#### Event Importance Scoring

Every event logged to the event log receives an `importance` score (1–5) based on the following rubric:

| Importance | Criteria | Examples |
|------------|----------|----------|
| **5** (Quest-critical) | Quest stage transitions, quest completion/failure | Player completes Stage 3, quest fails |
| **4** (Major) | Combat events, major reputation changes (≥10), NPC incapacitation | Player attacks Aldric, Farmer Jak incapacitated |
| **3** (Significant) | Social interactions with rep change ≥5, quest-relevant discoveries, random events that affect gameplay | Player persuades Elder, theft event triggered |
| **2** (Minor) | Routine social interactions, movement, trade | Player greets guard, NPC trades with NPC |
| **1** (Trivial) | Routine NPC actions, look/wait/rest, background activity | Farmer Jak works in fields, player rests |

#### Event Resolution

```python
def check_random_events(game_state):
    """Called every turn. Check if any random event triggers."""
    active_events = game_state.active_events
    
    for event_def in EVENT_CATALOG:
        if event_def.id in [e.id for e in active_events]:
            continue  # don't stack same event
        
        if not event_def.check_conditions(game_state):
            continue  # conditions not met
        
        if random() < event_def.probability:
            event = Event(
                id=event_def.id,
                turn_started=game_state.turn,
                duration=event_def.roll_duration(),
                effects=event_def.effects
            )
            active_events.append(event)
            game_state.event_log.append(event.to_log_entry())
            
            # Narrate event to player
            narrate_event(event, game_state)
            
            # Notify affected NPCs
            for npc in get_affected_npcs(event, game_state):
                npc.known_events.append(event.to_npc_knowledge())
    
    # Expire old events
    active_events[:] = [e for e in active_events 
                         if game_state.turn < e.turn_started + e.duration]
```

#### Event Impact on NPC Behavior

Random events modify NPC Q-learning by temporarily adjusting reward signals:

| Event | NPC Behavior Change |
|-------|-------------------|
| Storm | NPCs prioritize `move_to` indoor locations; `rest` reward increased |
| Wandering Merchant | Merchant archetype NPCs increase `trade` action selection |
| Theft Report | Guard archetype NPCs increase `examine` and patrol frequency |
| NPC Accident | Nearby NPCs may choose `talk` or `give_item` toward injured NPC |

### 5.8 Action Outcome Narration

Every action produces a **narrative description** for the player. The narration system uses a layered approach to generate contextually appropriate text.

#### NPC Action Narration Filtering

With 6 NPCs each acting every turn, the UI would flood with NPC narration. NPC actions are filtered as follows:

| Condition | Display | Example |
|-----------|---------|---------|
| NPC at **player's location** | Full narration in main narrative log | "Aldric patrols past you, eyeing your pack." |
| NPC action with **importance ≥ 3** (quest-relevant) | Brief notification in main log | "📣 Farmer Jak found something in the fields." |
| NPC at **different location**, routine action | Collapsed under "Meanwhile..." section (expandable) | "Meanwhile: Tessa cleans the bar. Petra gossips." |
| NPC-to-NPC gossip about player | Brief notification if gossip delta ≥ 5 | "🗣 Rumor: Tessa told Aldric about your theft." |

#### Narration Layers

```
Action resolved (effects computed)
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 1: Template Narration          │
│  Pre-written templates for all 27     │
│  universal actions × outcome types    │
│  (success, partial, fail, blocked)    │
│  → Fast, always available             │
└──────────┬────────────────────────────┘
           │ If LLM available & context is rich
           ▼
┌───────────────────────────────────────┐
│  Layer 2: LLM-Enhanced Narration      │
│  LLM enriches the template with       │
│  context-aware details, NPC reactions, │
│  emotional tone matching player input  │
│  → Richer, but optional               │
└──────────┬────────────────────────────┘
           ▼
┌───────────────────────────────────────┐
│  Layer 3: Context Modifiers           │
│  Append environmental details:        │
│  time-of-day flavor, weather effects, │
│  NPC mood indicators, witness notes   │
└───────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│  Layer 4: Passive Perception          │
│  On `wait`, `look`, or any action at  │
│  a new location, auto-roll for nearby │
│  NPC behavior, quest items, ambient   │
│  details. Ensures quest-critical info │
│  isn't missed without explicit search │
└───────────────────────────────────────┘
```

#### Passive Perception

Whenever the player takes **any action** at a location (including `wait`), or enters a **new location**, a passive perception check fires:

```python
def passive_perception_check(player, game_state):
    """Auto-notice important things without requiring explicit `look` or `search`."""
    base_chance = 0.4  # 40% base chance
    
    # Bonus at social locations
    if game_state.current_location in ["tavern", "village_center"]:
        base_chance += 0.2
    
    # Check for quest-critical items at this location
    for item in game_state.location_items.get(player.location, []):
        if item.quest_relevant and random() < base_chance:
            return {"type": "item_noticed", "item": item,
                    "narration": f"Something catches your eye: {item.description}"}
    
    # Check for notable NPC behavior
    for npc in game_state.npcs_at(player.location):
        if npc.current_action and npc.action_importance >= 3 and random() < base_chance:
            return {"type": "npc_noticed", "npc": npc,
                    "narration": f"You notice {npc.name} {npc.current_action_description}."}
    
    return None  # nothing noticed
```

> **Design Note:** Passive perception prevents the player from missing quest-critical discoveries just because they didn't explicitly `look` or `search`. It fires subtly — the player still benefits from active exploration, which has higher discovery rates.

#### Template Structure

```python
ACTION_NARRATION_TEMPLATES = {
    "attack": {
        "success": [
            "You land a solid blow on {target}. They stagger back, taking {damage} damage.",
            "Your strike connects with {target}! {damage} damage dealt.",
        ],
        "fail": [
            "You swing at {target}, but they dodge aside.",
            "{target} deflects your attack effortlessly.",
        ],
        "blocked": [
            "There's no one here to attack.",
            "You clench your fists, but think better of it.",
        ],
    },
    "talk": {
        "success": [
            "You approach {target} and strike up a conversation.",
            "{target} turns to listen.",
        ],
        "fail": [
            "{target} ignores you completely.",
            "{target} turns away, clearly uninterested.",
        ],
        "blocked": [
            "There's nobody around to talk to.",
        ],
    },
    "search": {
        "success": [
            "You search the area thoroughly. {discovery}",
            "After careful examination, you find {discovery}.",
        ],
        "fail": [
            "You search around but find nothing of interest.",
            "Despite a thorough search, nothing stands out.",
        ],
    },
    # ... templates for all 27 actions × outcome types
}
```

#### LLM Narration Enhancement Prompt

```
You are the narrator for a fantasy RPG. Enhance this action outcome with vivid, contextual detail.

ACTION: {action_id} by {actor_name}
TARGET: {target_name}
OUTCOME: {outcome_type} (success/fail/partial)
BASE NARRATION: "{template_text}"
CONTEXT: {location}, {time_of_day}, {weather}
PLAYER EMOTION: {emotion}
PLAYER SOCIAL TONE: {social}
NEARBY NPCs: {witness_list}

Rewrite the narration in 2-3 sentences. Match the emotional tone.
Keep it concise and atmospheric. Do not add mechanical details.

OUTPUT: Just the narration text, no JSON.
```

#### Narration Output Structure

The player sees a structured output each turn:

```
┌─────────────────────────────────────────────────┐
│  [Turn 5 | Midday | Village Center]             │
│                                                 │
│  > You search the notice board carefully.       │  ← Action narration
│    Several faded postings flutter in the wind.  │  ← Context modifier
│    Among them, one catches your eye — a report  │  ← Discovery detail
│    of strange lights near the old oak.          │
│                                                 │
│  ⚔ Stamina: -5 AP                              │  ← Mechanical effects
│  📜 New clue: "Strange lights at old oak"       │  ← Discovery logged
│  👁 Old Petra watches you curiously.             │  ← Witness note
└─────────────────────────────────────────────────┘
```

### 5.9 World Memory & Event Log

The game maintains a **central event log** — a chronological record of all significant events that serves as the world's "memory." This feeds into NPC knowledge, gossip propagation, LLM context, and player recap.

#### Event Log Structure

```python
EventLogEntry = {
    "event_id": str,          # unique event ID (auto-generated)
    "turn": int,              # game turn when event occurred
    "time_of_day": str,       # morning/midday/afternoon/evening/night
    "event_type": str,        # "player_action", "npc_action", "random_event", "quest_progress", "combat", "dialogue"
    "actor": str,             # npc_uid or "player"
    "action": str,            # action_id from universal catalog
    "target": str | None,     # npc_uid, item_id, location, or None
    "location": str,          # where it happened
    "outcome": str,           # "success", "fail", "partial", "blocked"
    "effects": dict,          # {"health": -5, "reputation": {"guard_a3f1": -20}, "item_gained": "old_map"}
    "witnesses": list[str],   # list of npc_uids at the location when event occurred
    "narration": str,         # the narration text shown to the player
    "importance": int,        # 1-5 scale (1=trivial, 5=quest-critical) — affects gossip priority
}
```

#### Event Log Capacity & Pruning

```python
MAX_EVENT_LOG_SIZE = 500      # keep last 500 events in full detail
SUMMARY_AFTER = 500           # older events are summarized
# NOTE: With 6 NPCs + player + random events ≈ 7-8 events/turn, this cap covers
# ~60-70 turns of full detail. Pruning only compresses importance ≤ 1 (routine) events;
# events with importance ≥ 2 are always kept in full detail for replay and analytics.

def prune_event_log(event_log):
    """Keep recent events detailed; compress old ones into summaries."""
    if len(event_log) > MAX_EVENT_LOG_SIZE:
        old_events = event_log[:-MAX_EVENT_LOG_SIZE]
        summary = summarize_events(old_events)  # e.g., "Turns 1-50: Player completed Stage 1, befriended Farmer Jak, fought guard Aldric"
        event_log[:] = [summary] + event_log[-MAX_EVENT_LOG_SIZE:]
```

#### NPC Witnessed Events & Knowledge

Each NPC maintains a **personal knowledge base** derived from events they directly witnessed or learned about via gossip:

```python
NPCKnowledge = {
    "known_events": [
        {
            "event_id": str,         # reference to event log
            "source": str,           # "witnessed" | "gossip"
            "from_npc": str | None,  # npc_uid of gossip source (None if witnessed)
            "turn_learned": int,     # when this NPC learned about it
            "importance": int,       # inherited from event, decayed for gossip
            "decayed_effects": dict  # reputation effects after gossip decay
        }
    ],
    "player_opinion_factors": [
        # Aggregated reasons for current reputation toward player
        {"reason": "Helped me find lost tool", "rep_change": +10, "turn": 15},
        {"reason": "Was rude to Elder Maren (gossip)", "rep_change": -5, "turn": 22}
    ]
}
```

#### Witness Detection

When an event occurs, all NPCs at the same location are automatically added as witnesses:

```python
def detect_witnesses(event, game_state):
    """Find all NPCs who witness an event at its location."""
    witnesses = []
    for npc_uid, npc in game_state.npcs.items():
        if npc.location == event["location"] and npc_uid != event.get("actor"):
            witnesses.append(npc_uid)
            # Direct witness: full reputation effect
            npc.known_events.append({
                "event_id": event["event_id"],
                "source": "witnessed",
                "from_npc": None,
                "turn_learned": game_state.turn,
                "importance": event["importance"],
                "decayed_effects": event["effects"]  # no decay for direct witness
            })
    return witnesses
```

#### World Memory Usage

| Consumer | How It Uses Event Log |
|----------|----------------------|
| **LLM Prompts** | Last 5-10 relevant events included as context for checkpoint generation and NPC dialogue |
| **NPC Gossip** | Events with importance ≥ 3 are eligible for gossip propagation during NPC-to-NPC interactions |
| **Player Recap** | UI can display a "journal" of major events (importance ≥ 3) |
| **Quest Logic** | Quest transitions can check event log for prerequisites (e.g., "has the player talked to Jak?") |
| **Difficulty Scaling** | Event log analytics inform difficulty adjustment (see §5.12) |

### 5.10 Edge Case Handling

Specific edge cases that can arise during gameplay, with defined resolution behavior:

| Edge Case | Detection | Resolution |
|-----------|-----------|------------|
| **Inventory full + quest item** | Player tries to `pick_up` a quest-critical item with full inventory | Auto-prompt: "Your inventory is full. Drop an item to make room?" → show droppable (non-quest) items. Quest items are never lost. |
| **0 Stamina + forced combat** | NPC attacks player when player has 0 AP | Player can always `defend` (costs 0 AP when at 0 stamina) or `flee` (costs 0 AP, but success rate drops to 40%). Cannot `attack` at 0 AP. |
| **All NPCs hostile + quest requires help** | Every NPC's reputation is below -50, but quest needs NPC cooperation | Introduce a "mercy" mechanic: after 3 failed attempts to interact with hostile NPCs, a dynamic checkpoint offers a path to reputation recovery (e.g., complete a random event task for +15 rep). |
| **Player-triggered CP loops** | Player repeatedly takes the same unexpected action, generating identical dynamic checkpoints | Loop detection: if same action_id generates a dynamic CP 3 times in a row at the same quest stage, force convergence to next main-flow checkpoint with narration: "The world shifts around you..." |
| **Health = 0 during dynamic CP** | Player dies while in a dynamic (non-scripted) checkpoint | Same as scripted death: transition to $S_{fail}$. However, offer a "last chance" if player is within 2 CPs of a main-flow checkpoint: reduce HP to 1, add narration "You barely survive..." |
| **Simultaneous contradictory NPC actions** | Two NPCs select actions that conflict (e.g., both try to `trade` the same item, or one attacks while another befriends the player) | Priority resolution: (1) Combat actions resolve first, (2) Social actions second, (3) If true conflict (same item), NPC with higher reputation wins; tie broken by random roll |
| **NPC at invalid location** | Q-learning directs NPC to a non-adjacent location | Hard-fail precondition catches this. NPC's action is replaced with `wait` (fallback). Q-table penalized with $Q(s, a) \leftarrow Q(s, a) - 5$ for this state-action pair. |
| **LLM generates invalid action** | LLM checkpoint/dialogue references actions not in universal catalog | Validate against action catalog. Unknown actions mapped to closest match via spaCy similarity. If no match > 0.4, replace with `wait`. |
| **Save file corruption** | Save JSON fails to parse on load | Maintain a `.backup` copy of last known good save. Auto-restore from backup. If backup also fails, start new game with notification. |
| **NPC incapacitated during quest** | Quest-critical NPC is reduced to 0 HP by player or NPC combat | Quest-critical NPCs cannot die (see §5.6 NPC Death & Incapacitation): their HP floors at 1, they become permanently hostile, and a redemption dynamic checkpoint is generated. Non-critical NPCs respawn after 20 turns. |
| **Compound text input** | Player types multi-action input like "go to tavern and talk to Tessa" | Parser detects compound intent, resolves first action immediately, queues second action for next turn. Max queue depth: 1. Notification shown: "Next action queued: talk to Tessa." |

### 5.11 Victory & Defeat

The quest has two terminal states: $S_{success}$ and $S_{fail}$.

#### Victory ($S_{success}$)

Triggered when the player completes the final checkpoint of Stage 7 ("Get Reward").

**UI flow:**

1. Final narration plays: Elder Maren's gratitude, artifact restored, village celebrations
2. Screen transitions to **Session Summary** panel:
   - Total turns played
   - Quest path visualization (MDP graph with completed path highlighted)
   - Checkpoints visited: static vs. dynamic count
   - Key NPC relationships (final per-NPC reputation values)
   - Notable events (importance ≥ 4)
   - Difficulty-adjusted score (optional)
3. Options: **[Export Session]** (JSON/Markdown), **[Play Again]** (new seed), **[Review Metrics]** (opens analytics dashboard)

#### Defeat ($S_{fail}$)

Triggered when player HP reaches 0.

**UI flow:**

1. Death narration plays (contextual: combat death, trap, exhaustion)
2. Screen shows **Defeat Summary**:
   - Turn of death
   - Cause of death (last event that reduced HP to 0)
   - Quest progress at time of death (stage, checkpoint)
   - Per-NPC reputation at death
3. Options: **[Load Last Save]** (most recent auto-save or manual save), **[Restart]** (new game, same seed), **[Restart New Seed]** (new game, random seed)

> **Note:** The session turn limit (see Session Turn Limit below) is a separate session-ending condition — not a defeat. Reaching `MAX_TURNS` triggers a summary screen with options to save, export, or extend.

> **Design Note:** The player is never stuck without options. Auto-save every 5 turns (plus saves on every quest-relevant action) ensures minimal lost progress.

#### Session Turn Limit

Each session has a configurable turn limit (default `MAX_TURNS = 200`, adjustable via URL parameter `max_turns`; see §14.3.3). When the limit is reached:

1. The current turn completes normally (no mid-turn interruption).
2. **Session End** screen displays:
   - "Session limit reached (200/200 turns)"
   - Quest progress summary (stage, checkpoint, percentage)
   - Per-NPC reputation snapshot
   - Option buttons: **[Save & Quit]**, **[Export Session]**, **[Extend +50 Turns]** (research mode only)
3. The game auto-saves before ending.

> **Rationale:** A turn limit prevents unbounded sessions during automated playtesting (§14.3) and provides a natural stopping point for research data collection. Human players can set `max_turns=0` to disable the limit.

### 5.12 Difficulty Scaling

The game supports tunable difficulty through a set of **difficulty parameters** that modify core mechanics. These can be set as presets (Easy/Normal/Hard) or individually configured.

#### Difficulty Presets

| Parameter | Easy | Normal | Hard |
|-----------|------|--------|------|
| `ap_cost_multiplier` | 0.75 | 1.0 | 1.5 |
| `combat_damage_to_player` | 0.7 | 1.0 | 1.4 |
| `combat_damage_from_player` | 1.3 | 1.0 | 0.8 |
| `npc_hostility_threshold` | -60 | -50 | -35 |
| `reputation_gain_multiplier` | 1.5 | 1.0 | 0.7 |
| `reputation_loss_multiplier` | 0.7 | 1.0 | 1.5 |
| `nudge_aggressiveness` | High | Medium | Low |
| `max_deviations_before_convergence` | 7 | 5 | 3 |
| `stamina_regen_per_turn` | 8 | 5 | 3 |
| `gossip_propagation_rate` | 0.2 | 0.4 | 0.6 |
| `random_event_frequency` | 0.5× | 1.0× | 1.5× |
| `combat_flee_success_rate` | 80% | 70% | 55% |

#### Difficulty Configuration

```python
DifficultyConfig = {
    "preset": "easy" | "normal" | "hard" | "custom",
    "ap_cost_multiplier": float,        # multiplied against all AP costs
    "combat_damage_to_player": float,   # scales incoming damage
    "combat_damage_from_player": float, # scales outgoing damage
    "npc_hostility_threshold": int,     # reputation below this = hostile
    "reputation_gain_multiplier": float,
    "reputation_loss_multiplier": float,
    "nudge_aggressiveness": str,        # "low", "medium", "high"
    "max_deviations_before_convergence": int,
    "stamina_regen_per_turn": int,
    "gossip_propagation_rate": float,
    "random_event_frequency": float,    # multiplier on event probabilities
    "combat_flee_success_rate": float,
}
```

#### Adaptive Difficulty (Optional)

The system can **auto-adjust** difficulty based on event log analytics:

```python
def assess_player_struggle(event_log, window=20):
    """Analyze last N events for signs of player struggle."""
    recent = event_log[-window:]
    
    death_count = sum(1 for e in recent if e["outcome"] == "death")
    fail_rate = sum(1 for e in recent if e["outcome"] == "fail") / max(len(recent), 1)
    avg_health = mean([e["effects"].get("health_after", 100) for e in recent])
    deviation_count = sum(1 for e in recent if e.get("is_dynamic_cp"))
    
    struggle_score = (death_count * 3 + fail_rate * 10 + 
                      max(0, 30 - avg_health) / 10 + deviation_count * 0.5)
    
    if struggle_score > 15:
        return "decrease_difficulty"  # auto-shift toward Easy
    elif struggle_score < 3:
        return "increase_difficulty"  # auto-shift toward Hard
    return "maintain"
```

> **Design Note:** Difficulty scaling is metadata-driven — all parameters are stored in `config.py` and can be adjusted without code changes. The adaptive system is optional and off by default; research mode may want fixed difficulty for controlled experiments.

---

## 6. Dynamic Checkpoint Generation & Nudging

### 6.1 When Checkpoints Are Generated

#### Dynamic Checkpoint ID Format

Dynamic checkpoints use the format `{stage_id}_D{counter}` where `counter` is an auto-incrementing integer per stage:

```python
def generate_dynamic_cp_id(stage_id: int, existing_dynamic_cps: list) -> str:
    """Generate next dynamic checkpoint ID for this stage."""
    stage_dynamics = [cp for cp in existing_dynamic_cps if cp.startswith(f"{stage_id}_D")]
    next_counter = len(stage_dynamics) + 1
    return f"{stage_id}_D{next_counter}"  # e.g., "1_D1", "1_D2", "3_D1"
```

A new dynamic checkpoint $Cp_{i,(n+1)}$ is created when:

1. **Action produces an unexpected outcome** — a universal action is attempted with preconditions that create a novel situation not covered by static checkpoints (e.g., `attack` at a peaceful checkpoint, `steal` from the quest giver)
2. **Player types free text that doesn't match any action in the universal catalog** — text input parsed with low confidence, treated as a truly novel player intention outside the known action space
3. **Player deviates from expected path** — doesn't proceed to `next_expected` checkpoint  
4. **Environmental trigger** — NPC RL agent does something that creates a new situation  

### 6.2 Generation Pipeline

```
Player input (button click from action palette OR free text)
        │
        ▼
┌──────────────────────────────────┐
│  Input Parser / Action Resolver  │
│  (palette → action) OR           │
│  (text → NLP → universal action) │
└───────┬───────────────┬──────────┘
        │ Known action  │ Unknown / truly novel
        ▼               ▼
   Resolve against  Player intent outside
   context + pre-   universal catalog
   conditions              │
        │                  ▼
        │         ┌───────────────────────┐
        │         │ Check: Is LLM loaded? │
        │         └───────┬───────┬───────┘
        │                 │Yes    │No
        │                 ▼       ▼
        │            ┌─────────┐ ┌──────────────┐
        │            │ LLM Gen │ │ Template Gen │
        │            └────┬────┘ └──────┬───────┘
        │                 │             │
        ▼                 ▼             ▼
   Quest-relevant? ┌───────────────────────────┐
        │          │  Validate checkpoint:     │
   Yes: transition │  - Has description        │
   to next CP      │  - Has highlighted_actions│
        │          │  - Has nudge_target set   │
   No: dynamic     │  - Effects are reasonable │
   CP generated    └───────────┬───────────────┘
                               ▼
                ┌───────────────────────────┐
                │  Insert into MDP graph    │
                │  Update visualization     │
                └───────────────────────────┘
```

### 6.3 Template-Based Generation (Fallback)

When LLM is unavailable, use predefined templates:

```python
CHECKPOINT_TEMPLATES = {
    "unexpected_combat": {
        "description": "Your aggressive action draws attention. {npc_name} confronts you.",
        "highlighted_actions": ["attack", "persuade", "flee"],
        "nudge": "The commotion settles. You notice {nudge_hint}.",
        "effects": {"stamina": -5, "reputation": -3}
    },
    "unexpected_explore": {
        "description": "You wander off the path and discover {discovery}.",
        "highlighted_actions": ["examine", "search", "pick_up"],
        "nudge": "In the distance, you can see {next_landmark}.",
        "effects": {"stamina": -3}
    },
    "unexpected_social": {
        "description": "You strike up a conversation with {npc_name}.",
        "highlighted_actions": ["ask_info", "trade", "talk", "move_to"],
        "nudge": "{npc_name} mentions something about {quest_hint}.",
        "effects": {"stamina": -1}
    },
    "unexpected_stealth": {
        "description": "You try to sneak around. {outcome}.",
        "highlighted_actions": ["sneak", "hide", "wait"],
        "nudge": "From your hiding spot, you overhear talk about {quest_hint}.",
        "effects": {"stamina": -4}
    }
}
```

### 6.4 Nudging Mechanism

**Nudging** = gently guiding the player back toward the quest's expected progression without forcing them.

**How it works:**

1. Every dynamic checkpoint has a `nudge_target` — the next expected checkpoint in the main quest flow
2. Each dynamic checkpoint's description/actions contain **narrative hints** pointing toward the nudge target
3. Actions that move toward the nudge target have **slightly higher rewards** (reward shaping)
4. After **3 consecutive dynamic checkpoints** without returning to main flow: provide an explicit hint
5. After **5 consecutive dynamic checkpoints**: create a "forced convergence" checkpoint that connects back to main flow

**Reward shaping for nudging:**

$$R_{nudge}(cp, a) = R_{base}(cp, a) + \lambda \cdot (\text{distance\_to\_main\_path}_{before} - \text{distance\_to\_main\_path}_{after})$$

Where $\lambda = 0.3$ (subtle) and distance = number of checkpoints to reach the next main-flow checkpoint.

---

## 7. LLM Integration Specification

### 7.1 Setup

```
Model:    qwen3-4B-q4_k_m.gguf
Runtime:  llama-cpp-python
Context:  4096 tokens (sufficient for checkpoint generation)
GPU:      Optional (CPU fallback)
```

#### Temperature Settings Per Use Case

| Use Case | Temperature | Rationale |
|----------|------------|----------|
| **Checkpoint generation** | 0.8–0.9 | Higher creativity for novel scenarios |
| **Free-text input parsing** | 0.3–0.5 | Low temperature → deterministic, accurate classification |
| **NPC dialogue** | 0.7 | Balanced: personality-consistent but varied |
| **Narration enhancement** | 0.7 | Balanced: atmospheric but coherent |

> **Note:** Temperature is passed per-call to `llm_service.generate()`. The global default of 0.7 (in `config.py`) applies only when no per-call value is specified.

#### Async LLM Inference

LLM inference is CPU/GPU-bound and **must not block the FastAPI event loop**. All LLM calls are wrapped with `asyncio.to_thread()` (or `run_in_executor()` for finer control):

```python
import asyncio

async def llm_generate_async(prompt: str, **kwargs) -> str:
    """Non-blocking LLM inference wrapper."""
    return await asyncio.to_thread(
        llm_service.generate, prompt, **kwargs
    )
```

All endpoints that may invoke LLM generation (`/api/action`, `/api/talk`) are `async def` and await `llm_generate_async`.

#### Context Window Token Budget

With a 4096-token context window, tokens must be carefully allocated. Target budgets per prompt type:

| Prompt Type | System/Instruction | Game Context | Conversation History | Generation Headroom | Total |
|-------------|-------------------|--------------|---------------------|-------------------|-------|
| **Checkpoint generation** | ~200 tokens | ~400 tokens (stage, player stats, location, inventory, last 5 events) | N/A | ~500 tokens | ~1100 |
| **NPC dialogue** | ~250 tokens (personality, reputation, mood) | ~300 tokens (situation, location) | ~400 tokens (last 5 exchanges) | ~300 tokens | ~1250 |
| **Input analysis** | ~150 tokens (action catalog list) | ~200 tokens (location, NPCs present) | N/A | ~200 tokens | ~550 |
| **Narration enhancement** | ~100 tokens | ~300 tokens (action, outcome, context) | N/A | ~300 tokens | ~700 |

> **Hard rule:** No prompt should exceed **2500 tokens** to guarantee at least 1500 tokens for generation. If context exceeds budget, truncate in priority order: (1) trim conversation history to last 3 exchanges, (2) trim event log to last 3 events, (3) omit inventory details.

### 7.2 Usage Points

| Use Case                      | When Triggered                                         | Fallback (no LLM)                |
|-------------------------------|--------------------------------------------------------|----------------------------------|
| **Free-text input parsing**   | Player types ambiguous text                            | Keyword/fuzzy match + synonyms   |
| Dynamic checkpoint generation | Player deviates                                        | Template-based generation        |
| **NPC dialogue (unscripted)** | **Player talks to NPC in a context with no scripted response** | **Generic personality-based template response** |
| NPC dialogue enrichment       | Player talks to NPC (scripted context)                 | Predefined dialogue lines        |
| Quest description elaboration | Quest accepted                                         | Static quest text                |
| Rumor generation              | NPC in tavern                                          | Predefined rumor pool            |

### 7.3 Prompt Templates

**Checkpoint Generation Prompt:**

```
You are a game master for a fantasy RPG. Generate a brief checkpoint for the quest.

CONTEXT:
- Current quest stage: {stage_description}
- Player just did: {player_action}
- Player's emotional tone: {emotion}
- Player's social manner: {social}
- Player location: {location}
- Player stats: Health={health}, Stamina={stamina}, Reputation={reputation}
- Expected next checkpoint: {expected_next}
- Key items in inventory: {inventory_summary}

TASK: Generate a checkpoint that:
1. Acknowledges the player's action AND their emotional/social tone
2. Describes what happens next (2-3 sentences) — match the narrative tone to the player's demeanor
3. Provides 3-4 highlighted actions from the universal catalog that are contextually relevant
4. Subtly hints toward returning to the main quest

OUTPUT FORMAT (JSON):
{
  "description": "...",
  "highlighted_actions": ["action1", "action2", "action3"],
  "effects": {"health": 0, "stamina": -2, "reputation": 0},
  "hint": "..."
}
```

**Free-Text 3-Dimensional Analysis + Action Classification Prompt (LLM-assisted):**

```
You are an input parser for a fantasy RPG game. Analyze the player's text input along 3 dimensions and classify it.

PLAYER INPUT: "{player_text}"
CURRENT LOCATION: {location}
NPCs PRESENT: {npcs_present}

UNIVERSAL ACTION CATALOG:
{universal_actions_list}

HIGHLIGHTED ACTIONS (quest-relevant at current checkpoint):
{highlighted_actions_list}

TASK:
1. Determine the EMOTION conveyed: neutral, angry, friendly, fearful, curious, or threatening
2. Determine the INTENT (gameplay action): match to one of the universal actions, or "UNKNOWN" if truly novel
3. Determine the SOCIAL posture: neutral, polite, rude, deceptive, honest, or intimidating

OUTPUT FORMAT (JSON):
{
  "emotion": "friendly",
  "intent": "greet_guards",
  "social": "polite",
  "matched_action": "action_id" or "UNKNOWN",
  "confidence": 0.0-1.0,
  "interpreted_intent": "brief description of what player wants"
}
```

**NPC Dialogue Prompt (for unscripted contexts):**

When the player interacts with an NPC in a situation where no scripted dialogue exists, the LLM generates the response in character. This is the **primary dialogue generation path** — scripted lines are reserved only for critical quest moments and common greetings.

```
You are {npc_name} (ID: {npc_uid}), a {npc_role} ({archetype}) in a small village.
Your personality: {personality}
Your current mood: {mood} (happiness={happiness}/10)
Player reputation with you: {reputation}
Current situation: {context}
Conversation history with this player: {recent_conversation_history}

The player says: "{player_input}"
The player's tone is {emotion}, their social manner is {social}.

React appropriately to their demeanor:
- If they are rude/threatening: be defensive, less helpful
- If they are polite/friendly: be warmer, share more
- If they are deceptive: be suspicious if your reputation with them is low
- Stay consistent with your personality archetype ({archetype})
- Reference previous conversation if relevant

Respond in character. Keep it under 3 sentences.
If reputation < -20, be unhelpful. If reputation > 40, share a secret or hint.

OUTPUT FORMAT (JSON):
{
  "dialogue": "...",
  "mood_change": 0,
  "reveals_info": false,
  "info_type": null
}
```

#### NPC Dialogue Resolution Pipeline

```
Player talks to NPC (via action palette or free text)
        │
        ▼
┌───────────────────────────────────────┐
│  1. Resolve target NPC               │
│     name/archetype → npc_uid lookup  │
└───────────┬───────────────────────────┘
            ▼
┌───────────────────────────────────────┐
│  2. Check scripted dialogue           │
│     Does (context + action) match a  │
│     scripted dialogue key?           │
└─────┬──────────────┬─────────────────┘
      │ Yes           │ No (unscripted)
      ▼               ▼
  Return scripted  ┌──────────────────────────┐
  dialogue line    │ 3. Check: Is LLM loaded? │
      │            └────┬──────────┬──────────┘
      │                 │ Yes       │ No
      │                 ▼           ▼
      │           LLM generates  Use archetype's
      │           in-character   generic template:
      │           response via   "{name} {generic_response
      │           prompt above   _for_archetype}"
      │                 │           │
      ▼                 ▼           ▼
┌───────────────────────────────────────┐
│  4. Log to conversation_history      │
│     (npc_uid, turn, player_input,    │
│      npc_response, emotion, social)  │
│  5. Apply mood/reputation effects    │
└───────────────────────────────────────┘
```

#### Generic Template Responses (Fallback when no LLM)

Each archetype has generic responses for common interaction types:

```python
ARCHETYPE_GENERIC_RESPONSES = {
    "guard": {
        "greeting": ["{name} grunts. 'State your business.'", "{name} nods curtly."],
        "ask_info": ["{name} shrugs. 'I just stand watch.'", "{name} eyes you. 'Not my concern.'"],
        "unknown":  ["{name} stares at you blankly.", "{name} ignores you."],
    },
    "villager": {
        "greeting": ["{name} smiles nervously. 'Hello there.'", "{name} waves."],
        "ask_info": ["{name} looks uncertain. 'I don't know much about that.'"],
        "unknown":  ["{name} seems confused.", "{name} shrugs."],
    },
    "merchant": {
        "greeting": ["{name} grins. 'Looking to buy?'", "{name} beckons you over."],
        "ask_info": ["{name} taps the counter. 'Information costs coin, friend.'"],
        "unknown":  ["{name} frowns. 'That's not how we do business.'"],
    },
    "elder": {
        "greeting": ["{name} regards you thoughtfully.", "{name} nods slowly."],
        "ask_info": ["{name} strokes their chin. 'That is a long story...'"],
        "unknown":  ["{name} sighs. 'The young always rush.'"],
    },
    "farmer": {
        "greeting": ["{name} wipes their brow. 'Hey there.'", "{name} leans on a fence."],
        "ask_info": ["{name} scratches their head. 'Can't say I know.'"],
        "unknown":  ["{name} shrugs and goes back to work."],
    },
    "tavern_keeper": {
        "greeting": ["{name} polishes a glass. 'What'll it be?'", "{name} waves from behind the bar."],
        "ask_info": ["{name} leans in. 'I hear things...' They glance around."],
        "unknown":  ["{name} chuckles. 'That's a new one.'"],
    },
}
```

### 7.4 Fallback Strategy

```python
async def generate_checkpoint(context):
    if llm_available:
        try:
            result = await llm_generate_async(context, timeout=10)
            if validate_checkpoint(result):
                return result
        except (TimeoutError, ParseError):
            pass  # fall through to template
    
    # Template fallback
    template = select_template(context.player_action_type)
    return fill_template(template, context)
```

### 7.5 LLM Output Validation & Guardrails

All LLM outputs are validated before being applied to the game state. This prevents hallucinated values, out-of-bound effects, and malformed responses from corrupting the simulation.

#### JSON Schema Validation

Every LLM response must conform to one of the expected JSON schemas:

```python
CHECKPOINT_SCHEMA = {
    "type": "object",
    "required": ["description", "highlighted_actions", "effects", "hint"],
    "properties": {
        "description": {"type": "string", "minLength": 10, "maxLength": 500},
        "highlighted_actions": {
            "type": "array",
            "items": {"type": "string", "enum": UNIVERSAL_ACTION_IDS},  # must be valid action IDs
            "minItems": 2,
            "maxItems": 5
        },
        "effects": {
            "type": "object",
            "properties": {
                "health":     {"type": "integer", "minimum": -50, "maximum": 50},
                "stamina":    {"type": "integer", "minimum": -20, "maximum": 20},
                "reputation": {"type": "integer", "minimum": -30, "maximum": 30}
            }
        },
        "hint": {"type": "string", "maxLength": 200}
    }
}

DIALOGUE_SCHEMA = {
    "type": "object",
    "required": ["dialogue", "mood_change", "reveals_info"],
    "properties": {
        "dialogue":    {"type": "string", "minLength": 5, "maxLength": 300},
        "mood_change": {"type": "integer", "minimum": -3, "maximum": 3},
        "reveals_info": {"type": "boolean"},
        "info_type":   {"type": "string", "enum": ["quest_hint", "location_info", "npc_gossip", "item_location", None]}
    }
}

INPUT_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": ["emotion", "intent", "social", "matched_action", "confidence"],
    "properties": {
        "emotion":         {"type": "string", "enum": ["neutral", "angry", "friendly", "fearful", "curious", "threatening"]},
        "intent":          {"type": "string"},
        "social":          {"type": "string", "enum": ["neutral", "polite", "rude", "deceptive", "honest", "intimidating"]},
        "matched_action":  {"type": "string"},
        "confidence":      {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "interpreted_intent": {"type": "string", "maxLength": 200}
    }
}
```

#### Value Bounds (Hard Limits)

Even if the LLM output passes schema validation, **clamp all numeric values** to prevent extreme effects:

| Field | Min | Max | Rationale |
|-------|-----|-----|-----------|
| `effects.health` | -50 | +50 | Prevents one-shot kills or full heals from a single checkpoint |
| `effects.stamina` | -20 | +20 | Prevents stamina drain beyond what's recoverable in a few turns |
| `effects.reputation` | -30 | +30 | Prevents instant hostility or instant trust from a single event |
| `dialogue` length | 5 chars | 300 chars | Prevents empty or excessively long NPC responses |
| `mood_change` | -3 | +3 | NPC mood changes should be gradual, not dramatic |
| `highlighted_actions` count | 2 | 5 | Always offer choices, but don't overwhelm |

#### Validation Pipeline

```python
def validate_llm_output(raw_response: str, schema: dict, context: str) -> dict | None:
    """Validate and sanitize LLM output. Returns clean dict or None on failure."""
    
    # Step 1: Parse JSON
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        match = re.search(r'```json?\s*(.*?)\s*```', raw_response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                log_warning(f"LLM JSON parse failed for {context}")
                return None
        else:
            return None
    
    # Step 2: Schema validation
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        log_warning(f"LLM schema validation failed: {e.message}")
        return None
    
    # Step 3: Clamp numeric values
    if "effects" in data:
        data["effects"]["health"] = clamp(data["effects"].get("health", 0), -50, 50)
        data["effects"]["stamina"] = clamp(data["effects"].get("stamina", 0), -20, 20)
        data["effects"]["reputation"] = clamp(data["effects"].get("reputation", 0), -30, 30)
    
    # Step 4: Validate action references
    if "highlighted_actions" in data:
        data["highlighted_actions"] = [a for a in data["highlighted_actions"] 
                                        if a in UNIVERSAL_ACTION_IDS]
        if len(data["highlighted_actions"]) < 2:
            data["highlighted_actions"] = ["look", "talk", "wait"]  # safe defaults
    
    # Step 5: Content filter (basic)
    if "dialogue" in data:
        data["dialogue"] = sanitize_text(data["dialogue"])
    if "description" in data:
        data["description"] = sanitize_text(data["description"])
    
    return data
```

#### Retry Logic

```python
MAX_LLM_RETRIES = 2
LLM_TIMEOUT_SECONDS = 10

async def llm_generate_with_retry(prompt, schema, context_label, temperature=0.7):
    """Generate LLM output with retry and validation."""
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            raw = await llm_generate_async(
                prompt, 
                max_tokens=512, 
                temperature=temperature,  # per-use-case; see §7.1 table
                timeout=LLM_TIMEOUT_SECONDS
            )
            result = validate_llm_output(raw, schema, context_label)
            if result is not None:
                return result
            
            log_info(f"LLM attempt {attempt + 1} failed validation for {context_label}")
        
        except TimeoutError:
            log_warning(f"LLM timeout on attempt {attempt + 1} for {context_label}")
        except Exception as e:
            log_error(f"LLM error on attempt {attempt + 1}: {e}")
    
    return None  # all retries exhausted → caller uses template fallback
```

#### Content Filtering

```python
def sanitize_text(text: str) -> str:
    """Basic content filter for LLM-generated text."""
    # Remove any system prompt leakage
    text = re.sub(r'(system|assistant|user):', '', text, flags=re.IGNORECASE)
    # Truncate excessive length
    if len(text) > 500:
        text = text[:497] + "..."
    # Remove potentially harmful content markers
    text = re.sub(r'<[^>]+>', '', text)  # strip HTML tags
    return text.strip()
```

> **Design Note:** The guardrails are intentionally conservative for MVP. In research mode, validation failures and fallback usage are logged to `metrics.json` for analysis (see §8.2). The retry count (2) and timeout (10s) are tunable in `config.py`.

---

## 8. Visual MDP Representation

### 8.0 UI Design Philosophy: Modern Minimal

The entire frontend follows a **modern minimal** design language:

| Principle | Implementation |
|-----------|----------------|
| **Dark theme** | Near-black background (`#0a0a0f`), high-contrast text, subtle surface elevation with dark grays |
| **Typography-first** | System font stack (`Inter` / `-apple-system` / `system-ui`); monospace for stats and data; clear hierarchy via weight and size, not decoration |
| **Whitespace as structure** | Generous padding and margins replace borders and dividers; content breathes |
| **Minimal chrome** | No gradients, no shadows heavier than `0 1px 3px`, no rounded corners > 8px; flat surfaces with subtle 1px borders (`rgba(255,255,255,0.06)`) |
| **Color as signal** | Color is used sparingly and only for meaning: green = completed, amber = current, orange = dynamic, blue = static; all other UI elements are grayscale |
| **Micro-interactions** | Subtle CSS transitions (150ms ease) on hover/focus; pulse animation only for current node; no bouncing, sliding, or attention-grabbing effects |
| **Responsive layout** | CSS Grid for main layout (game panel │ graph │ stats); collapses cleanly on smaller viewports |
| **Information density** | Show essential info at a glance; use progressive disclosure (expandable panels) for secondary info like full inventory or NPC status |

#### Color Palette

```
Background:    #0a0a0f (near-black)
Surface:       #14141b (card/panel bg)
Surface-2:     #1e1e28 (elevated elements)
Border:        rgba(255, 255, 255, 0.06)
Text-primary:  #e8e8ed
Text-secondary:#8b8b96
Text-muted:    #5a5a66
Accent-blue:   #4A90D9
Accent-green:  #2ECC71
Accent-amber:  #F39C12
Accent-orange: #E67E22
Accent-red:    #E74C3C
```

#### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Top Bar: Turn │ Time │ Location              (minimal)     │
├────────────────────┬─────────────────────┬──────────────────┤
│                    │                     │                  │
│   Narrative Log    │   MDP Graph         │  Stats Sidebar   │
│   (scrollable)     │   (Cytoscape.js)    │  HP / AP / Rep   │
│                    │                     │  Inventory       │
│                    │                     │  Quest Status    │
│                    │                     │  NPC Activity    │
├────────────────────┴─────────────────────┴──────────────────┤
│  Action Palette (categorized universal actions)             │
│  [Nav] [Social] [Combat] [Stealth] [Explore] [Util]         │
├─────────────────────────────────────────────────────────────┤
│  > Free-text input...                                       │
└─────────────────────────────────────────────────────────────┘
```

### 8.1 Requirements

- Display quest as a **directed graph** (nodal view)
- **Major stages** = large nodes in a horizontal flow
- **Checkpoints** = smaller nodes clustered under their parent stage
- **Dynamic checkpoints** = differently colored (e.g., orange vs. blue for static)
- **Current position** = highlighted node (green glow)
- **Completed path** = highlighted edges (green)
- **Available transitions** = dashed edges
- Real-time updates when new checkpoints are generated

### 8.2 Graph Layout

```
[S1: Go to Village] ──── [S2: Talk to Elder] ──── [S3: Get Quest] ──── ...
       │                        │                        │
   ┌───┼───┐               ┌───┼───┐               ┌───┼───┐
   │   │   │               │   │   │               │   │   │
 Cp1.1 Cp1.2 Cp1.3       Cp2.1 Cp2.2            Cp3.1 Cp3.2 Cp3.3
              │
           Cp1.4 (dynamic, player-generated)
```

### 8.3 Technology: Cytoscape.js

Why Cytoscape.js:

- Designed for graph/network visualization
- Supports hierarchical layouts (compound nodes)
- Great performance with 50-100 nodes
- Built-in styling, animation, and interaction
- Active open-source project

### 8.4 Node Styling

| Element                  | Shape             | Color                  | Size           |
|--------------------------|-------------------|------------------------|----------------|
| Major Stage (incomplete) | Rounded rectangle | `#4A90D9` (blue)       | Large          |
| Major Stage (complete)   | Rounded rectangle | `#2ECC71` (green)      | Large          |
| Major Stage (current)    | Rounded rectangle | `#F39C12` (amber)      | Large + glow   |
| Checkpoint (static)      | Circle            | `#85C1E9` (light blue) | Medium         |
| Checkpoint (dynamic)     | Diamond           | `#E67E22` (orange)     | Medium         |
| Checkpoint (current)     | Circle/Diamond    | `#F1C40F` (yellow)     | Medium + pulse |
| Checkpoint (completed)   | Circle/Diamond    | `#27AE60` (green)      | Medium         |

### 8.5 Metrics & Analytics Dashboard

A **debug/research panel** (collapsible, hidden by default) that tracks key simulation metrics in real-time. Essential for research analysis and gameplay tuning.

#### Tracked Metrics

| Metric | Description | Visualization |
|--------|-------------|---------------|
| **Deviation Rate** | % of player actions that triggered dynamic CP generation vs. followed expected path | Line chart (per-turn rolling average) |
| **LLM vs Template Usage** | Ratio of LLM-generated content vs template fallback (for CPs, dialogue, narration) | Stacked bar chart |
| **NPC Q-Table Convergence** | Average Q-value change per episode for each NPC; indicates learning stability | Sparkline per NPC |
| **Quest Completion Path** | Sequence of checkpoints visited (static vs dynamic) | Path overlay on MDP graph |
| **Action Distribution** | Frequency histogram of all 27 universal actions selected by player | Horizontal bar chart |
| **NPC Action Distribution** | Frequency histogram of actions selected by each NPC | Grouped bar chart |
| **Reputation Timeline** | Per-NPC reputation over turns | Multi-line chart (one line per NPC) |
| **Event Log Rate** | Events per turn by type (player_action, npc_action, random_event, etc.) | Stacked area chart |
| **Gossip Network** | Which NPCs have shared information with which others | Directed graph overlay |
| **Combat Statistics** | Hit rate, average damage, flee success rate | Summary cards |
| **Difficulty Adjustments** | If adaptive difficulty is on: when and how parameters changed | Timeline markers |
| **LLM Performance** | Average inference time, retry count, validation failure rate | Summary stats |

#### Metrics Data Model

```json
{
  "session_id": "uuid",
  "started_at": "ISO-8601",
  "total_turns": 0,
  "metrics": {
    "deviation_rate": {
      "total_actions": 0,
      "deviations": 0,
      "rate": 0.0,
      "per_turn": []
    },
    "llm_usage": {
      "checkpoint_gen": {"llm": 0, "template": 0},
      "dialogue_gen": {"llm": 0, "template": 0, "scripted": 0},
      "narration_gen": {"llm": 0, "template": 0},
      "input_analysis": {"llm": 0, "rules": 0}
    },
    "npc_convergence": {
      "elder_m8b2": {"avg_delta": [], "episodes": 0},
      "farmer_j4a1": {"avg_delta": [], "episodes": 0}
    },
    "action_distribution": {
      "player": {"greet": 5, "talk": 12, "search": 3, "...": "..."},
      "npc": {"elder_m8b2": {"talk": 8, "rest": 4}, "...": "..."}
    },
    "llm_performance": {
      "avg_inference_ms": 0,
      "total_calls": 0,
      "retries": 0,
      "validation_failures": 0,
      "timeouts": 0
    }
  }
}
```

#### UI Integration

```
┌────────────────────────────────────────────────────────┐
│  [📊 Research Metrics]   (click to expand)             │
├────────────────────────────────────────────────────────┤
│  Deviation Rate: ████████░░ 78%  |  LLM/Template: 3:1 │
│  NPC Convergence: Maren ✓  Jak ✓  Tessa ~            │
│  Actions: social 45% | explore 30% | combat 5% | ...  │
│  LLM: avg 2.3s | retries 2 | fails 1                  │
│  [Export Session JSON]                                 │
└────────────────────────────────────────────────────────┘
```

> **Design Note:** The metrics dashboard is primarily a research tool. In "player mode" it's hidden entirely. In "research mode" it can be toggled with a keyboard shortcut. All metrics are also written to `data/metrics/session_{id}.json` for post-hoc analysis.

---

## 9. Data Models

### 9.1 Quest Definition (JSON)

```json
{
  "quest_id": "main_quest_01",
  "title": "The Elder's Lost Artifact",
  "description": "Elder Maren's ancestral artifact has gone missing...",
  "stages": [
    {
      "stage_id": 1,
      "title": "Go to Village",
      "description": "Travel to the village of Thornhaven.",
      "checkpoints": [
        {
          "cp_id": "1_1",
          "description": "You approach the village gate. Two guards stand watch.",
          "context": {
            "npcs_present": ["guard_a3f1", "guard_b7e2"],
            "objects": ["village_gate", "notice_board"],
            "environment": "stone gatehouse, dirt road"
          },
          "highlighted_actions": [
            {"id": "greet", "label": "Greet the guards", "type": "social"},
            {"id": "sneak", "label": "Try to sneak past", "type": "stealth"},
            {"id": "examine", "label": "Show travel papers", "type": "social"}
          ],
          "quest_transitions": {
            "greet": {"next": "1_2", "effects": {"stamina": -1, "reputation": 2}},
            "sneak": {"next": "1_2", "effects": {"stamina": -5, "reputation": -5}, "success_prob": 0.6},
            "examine": {"next": "1_2", "effects": {"stamina": -1, "reputation": 5}, "requires": {"item": "travel_papers"}}
          },
          "is_dynamic": false,
          "nudge_target": "1_2"
        }
      ]
    }
  ]
}
```

> **Note:** `highlighted_actions` is a hint for the UI to emphasize quest-relevant actions in the action palette. All 27 universal actions remain available and can be attempted — actions outside `quest_transitions` are resolved dynamically by the action resolution pipeline.

### 9.2 NPC Personality Archetype (JSON)

```json
{
  "archetype": "farmer",
  "base_personality": "hardworking, honest, practical, cautious",
  "reward_weights": {"happiness": 0.2, "income": 0.4, "health": 0.3, "reputation": 0.1},
  "combat_stats": {"base_attack": 6, "base_defense": 4, "max_hp": 80},
  "base_stats": {
    "happiness": 5,
    "income": 3,
    "health": 8,
    "reputation": 5
  },
  "fallback_schedule": [
    {"time": "morning", "action": "work", "location": "fields"},
    {"time": "midday", "action": "eat", "location": "tavern"},
    {"time": "afternoon", "action": "work", "location": "fields"},
    {"time": "evening", "action": "move_to", "location": "tavern"},
    {"time": "night", "action": "rest", "location": "tavern"}
  ],
  "scripted_dialogue": {
    "greeting": "Mornin'. Busy day in the fields.",
    "quest_hint": "I found something strange near the old oak...",
    "hostile": "Don't have time for folks like you."
  },
  "movement_weights": {
    "fields": 0.5,
    "tavern": 0.25,
    "village_center": 0.15,
    "gate": 0.05,
    "elders_house": 0.05
  },
  "generic_responses": {
    "greeting": ["{name} wipes their brow. 'Hey there.'"],
    "ask_info": ["{name} scratches their head. 'Can't say I know.'"],
    "unknown": ["{name} shrugs and goes back to work."]
  }
}
```

> **NPC Home Locations:** Fallback schedules use actual game location IDs (see §10.1), not a generic "home" placeholder. Each NPC's nighttime/rest location is their thematic home: Elder Maren → `elders_house`, Farmer Jak → `tavern` (no farmhouse on map; eats and sleeps at tavern), Tessa → `tavern`, Aldric & Bryn → `gate` (gatehouse), Old Petra → `village_center`. These defaults are overridden by Q-learning policies once NPCs have trained.

### 9.3 NPC Instance (JSON)

```json
{
  "npc_uid": "farmer_j4a1",
  "name": "Farmer Jak",
  "archetype": "farmer",
  "location": "fields",
  "personality": "hardworking, honest, cautious",
  "stats": {
    "happiness": 6,
    "income": 4,
    "health": 8,
    "reputation": 5
  },
  "reward_weights": {"happiness": 0.2, "income": 0.4, "health": 0.3, "reputation": 0.1},
  "combat_stats": {"base_attack": 6, "base_defense": 4, "max_hp": 80},
  "current_hp": 80,
  "status": "active",
  "epsilon": 0.15,
  "npc_relationships": {
    "elder_m8b2": 10,
    "tavkeeper_t9c3": 15,
    "guard_a3f1": 0,
    "guard_b7e2": 0,
    "villager_c1d4": 5
  },
  "q_table": {},
  "conversation_history": [
    {
      "turn": 12,
      "player_input": "Have you seen anything strange lately?",
      "npc_response": "I found something strange near the old oak...",
      "source": "scripted",
      "emotion": "curious",
      "social": "polite"
    }
  ]
}
```

> **Note:** The `npc_uid` is the persistent unique identifier. `archetype` links back to the personality template. `conversation_history` tracks all interactions with this specific NPC instance, enabling the LLM to generate context-aware responses for unscripted situations. `status` is `"active"` (default), `"incapacitated"` (0 HP, returns after 20 turns), or `"hostile_wounded"` (quest-critical NPC at 1 HP floor). `epsilon` tracks the NPC's current Q-learning exploration rate (see §4.3); it decays over time and is persisted across saves.

#### Conversation History Cap

Each NPC's `conversation_history` is capped at the **last 10 exchanges** to prevent unbounded growth and to stay within the LLM's 4096-token context budget:

```python
MAX_CONVERSATION_HISTORY = 10  # per NPC

def add_to_conversation_history(npc, entry):
    """Append new exchange and prune if over cap."""
    npc.conversation_history.append(entry)
    if len(npc.conversation_history) > MAX_CONVERSATION_HISTORY:
        npc.conversation_history = npc.conversation_history[-MAX_CONVERSATION_HISTORY:]
```

> When building LLM prompts, only the last 5 exchanges are included (half the cap) to leave room for other context. The full 10 are persisted in save files for richer history.

### 9.4 Game State (JSON)

```json
{
  "turn": 0,
  "time_of_day": "morning",
  "master_seed": 42,
  "player": { "...player stats + quest_state (see §5.1)..." },
  "npcs": { "...npc states (see §9.3)..." },
  "event_log": [],
  "active_events": [],
  "dynamic_checkpoints_data": [],
  "difficulty_config": {
    "preset": "normal",
    "ap_cost_multiplier": 1.0,
    "...see §5.12 for all parameters..."
  },
  "metrics": { "...session metrics (see §8.5)..." }
}
```

### 9.5 Save/Load Specification

The game supports **saving and loading** the complete simulation state so gameplay can be resumed and research sessions can be replayed or analyzed.

#### Save File Format

A single JSON file containing the complete game state:

```json
{
  "save_version": "1.0",
  "save_timestamp": "2025-01-15T14:30:00Z",
  "game_version": "mvp-1.0",
  "turn": 25,
  "time_of_day": "afternoon",
  
  "player": {
    "name": "Player",
    "health": 75,
    "max_health": 100,
    "stamina": 32,
    "max_stamina": 50,
    "reputation": {
      "elder_m8b2": 25,
      "farmer_j4a1": -10,
      "tavkeeper_t9c3": 15,
      "guard_a3f1": 5,
      "guard_b7e2": 0,
      "villager_c1d4": 8
    },
    "global_reputation": 7,
    "inventory": ["...array of Item objects (see §5.1)..."],
    "equipped": {"weapon": null, "armor": null},
    "location": "village_center",
    "quest_state": {
      "quest_id": "main_quest_01",
      "current_stage": 3,
      "current_checkpoint": "3_2",
      "completed_checkpoints": ["1_1", "1_2", "2_1", "2_2", "3_1"],
      "dynamic_checkpoints": ["1_D1", "2_D1"],
      "deviation_count": 2
    }
  },
  
  "npc_registry": {
    "elder_m8b2": {
      "name": "Elder Maren",
      "archetype": "elder",
      "location": "elders_house",
      "stats": {"happiness": 7, "income": 2, "health": 6, "reputation": 8},
      "combat_stats": {"base_attack": 2, "base_defense": 2, "max_hp": 40},
      "current_hp": 40,
      "npc_relationships": {"farmer_j4a1": 15, "tavkeeper_t9c3": 10, "guard_a3f1": 20, "guard_b7e2": 20, "villager_c1d4": 25},
      "q_table": {"...serialized Q-table...": "..."},
      "conversation_history": ["..."],
      "known_events": ["..."]
    }
  },
  
  "event_log": ["...array of EventLogEntry objects..."],
  "active_events": ["...currently active random events..."],
  
  "quest_definition": "main_quest_01",
  "dynamic_checkpoints_data": ["...generated CP definitions..."],
  
  "difficulty_config": {
    "preset": "normal",
    "ap_cost_multiplier": 1.0
  },
  
  "metrics": {"...session metrics snapshot..."}
}
```

#### Save Triggers

| Trigger | Type | When |
|---------|------|------|
| **Auto-save** | Automatic | Every 5 turns (configurable via `AUTO_SAVE_INTERVAL`), or immediately after any quest-relevant action (checkpoint transition, stage completion) |
| **Stage completion** | Automatic | When player completes a quest stage |
| **Manual save** | Player-initiated | Via API endpoint `POST /save` or UI button |
| **Pre-combat save** | Automatic | Before combat resolution begins |

#### Save/Load API

```python
# Save
POST /api/save
Body: {"slot": "auto" | "manual_1" | "manual_2" | "manual_3"}
Response: {"success": true, "save_id": "save_20250115_143000", "file": "saves/save_20250115_143000.json"}

# Load
POST /api/load
Body: {"save_id": "save_20250115_143000"}
Response: {"success": true, "turn": 25, "stage": 3}

# List saves
GET /api/saves
Response: {"saves": [{"save_id": "...", "timestamp": "...", "turn": 25, "stage": 3, "slot": "auto"}]}
```

#### WebSocket Reconnection Handling

When the browser tab refreshes or the WebSocket connection drops, the client must be able to recover the current session:

```
WebSocket disconnect detected (client-side)
        │
        ▼
┌─────────────────────────────────────────┐
│  1. Client attempts reconnect           │
│     Exponential backoff: 1s, 2s, 4s,   │
│     8s, 16s — max 5 attempts            │
└──────────┬──────────────────────────────┘
           ▼
┌─────────────────────────────────────────┐
│  2. On reconnect: GET /api/state        │
│     Server returns full current state   │
│     (player, quest, NPCs, active events)│
└──────────┬──────────────────────────────┘
           ▼
┌─────────────────────────────────────────┐
│  3. Client rebuilds UI from state       │
│     - Re-render MDP graph               │
│     - Restore narrative log (last 20)   │
│     - Refresh stats sidebar             │
│     - Resume WebSocket event stream     │
└─────────────────────────────────────────┘
```

The server maintains game state in memory; the WebSocket is stateless — it only pushes updates. On reconnect, `GET /api/state` provides the full snapshot needed to reconstruct the UI. No session token is required (single-session MVP).

#### State Restoration Flow

```
Load save file
        │
        ▼
┌─────────────────────────────────┐
│  1. Validate save_version       │
│     Compatible? Continue.       │ 
│     Incompatible? Reject.       │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  2. Restore Player state        │
│     Stats, inventory, location, │
│     per-NPC reputation, quest   │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  3. Restore NPC Registry        │
│     Stats, Q-tables, known      │
│     events, conversation history│
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  4. Restore World State         │
│     Event log, active events,   │
│     dynamic checkpoints, time   │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  5. Rebuild MDP Visualization   │
│     Reconstruct graph with all  │
│     static + dynamic CPs        │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  6. Resume game loop            │
│     WebSocket reconnect         │
│     UI state refresh            │
└─────────────────────────────────┘
```

#### Save File Management

```python
MAX_AUTO_SAVES = 3          # rotate auto-saves (keep last 3)
MAX_MANUAL_SAVES = 3        # player can have up to 3 manual slots
SAVE_DIRECTORY = "data/saves/"

# Backup: every save creates a .backup copy of the previous save
# If primary save is corrupted, auto-restore from .backup
```

> **Design Note:** Save files intentionally include the full Q-table state so NPC learning persists across sessions. For research, save files can be used to reproduce exact game states and test different player strategies from the same starting point.

---

## 10. Game World Design

### 10.1 Locations

```
                    ┌──────────┐
                    │  Fields  │
                    └────┬─────┘
                         │
┌───────────┐    ┌───────┴──────┐    ┌────────────────┐
│   Gate    │────│Village Center│────│ Elder's House  │
└───────────┘    └───────┬──────┘    └────────────────┘
                         │
                    ┌────┴─────┐
                    │  Tavern  │
                    └──────────┘
```

| Location           | Type | Context (NPCs, objects, environment)          | Default NPCs (starting position)             |
|--------------------|------|-----------------------------------------------|----------------------------------------------|
| **Gate**           | Outdoor | Guards, notice board, merchant cart, road      | Aldric `guard_a3f1`, Bryn `guard_b7e2`      |
| **Village Center** | Outdoor | Stalls, well, villagers, crates, buildings     | Old Petra `villager_c1d4`                    |
| **Elder's House**  | Indoor | Bookshelves, artifact display, fireplace       | Elder Maren `elder_m8b2`                     |
| **Fields**         | Outdoor | Crops, old oak tree, barn, tools, creek        | Farmer Jak `farmer_j4a1`                     |
| **Tavern**         | Indoor | Tables, bar, fireplace, kitchen, notice board  | Tessa `tavkeeper_t9c3`                       |

> **Indoor vs Outdoor:** The `type` column determines which mechanics apply per location. **Indoor** locations (Elder's House, Tavern) grant NPC passive HP regen (+2 HP/turn, see §5.6), allow the player's `rest` action, and provide shelter from weather events. **Outdoor** locations (Gate, Village Center, Fields) do not grant passive NPC regen and are affected by weather events (see §5.7).

> All 6 NPCs use tabular Q-learning (see §4). NPCs may move between locations based on their learned policies and fallback schedules.

> **Note:** All 27 universal actions are available at every location. The context (NPCs, objects, environment) determines how each action resolves — e.g., `trade` at the Gate resolves differently than `trade` at the Tavern.

#### Location Adjacency Graph

The `move_to` action requires adjacency. The formal adjacency list (stored in `locations.json`):

```python
LOCATION_ADJACENCY = {
    "gate":            ["village_center"],
    "village_center":  ["gate", "elders_house", "fields", "tavern"],
    "elders_house":    ["village_center"],
    "fields":          ["village_center"],
    "tavern":          ["village_center"]
}
```

> `move_to` to a non-adjacent location is a hard-fail precondition: narrate "You can't reach there from here" and cost 0 AP. The player must navigate through `village_center` as a hub.

### 10.2 Time System

```
1 game turn = 1 action by player
4 turns = 1 time period transition

Time periods:  morning → midday → afternoon → evening → night → morning
               (4 turns)  (4 turns) (4 turns)   (4 turns)  (4 turns)
```

NPC actions happen simultaneously with player actions (1 NPC action per player turn).

### 10.3 Main Quest: "The Elder's Lost Artifact"

| Stage | Title               | Description                     | Key Checkpoints                                              |
|-------|---------------------|---------------------------------|--------------------------------------------------------------|
| S1    | Go to Village       | Arrive at Thornhaven            | Approach gate → Handle guards → Enter village                |
| S2    | Talk to Elder       | Find and speak with Elder Maren | Navigate to Elder's house → Enter → Learn about artifact     |
| S3    | Get Quest           | Accept the quest, learn details | Hear full story → Examine clues → Accept quest               |
| S4    | Investigate         | Gather clues about the artifact | Talk to Farmer Jak → Visit tavern → Hear rumors              |
| S5    | Get the Artifact    | Find and retrieve the item      | Go to fields → Search old oak → Retrieve artifact            |
| S6    | Return the Artifact | Bring it back to Elder Maren    | Return to village → Go to Elder's house → Hand over artifact |
| S7    | Get Reward          | Receive quest reward            | Elder thanks you → Choose reward → Quest complete            |

---

## 11. Project Structure

```
MVP/
├── Plan.md                    # This document (comprehensive MVP plan)
│
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Game configuration constants
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_engine.py     # Main game loop, turn processing, action outcome narration
│   │   ├── world.py           # World state, locations, time system
│   │   ├── events.py          # Random events system, event catalog, trigger conditions
│   │   ├── event_log.py       # World memory / event log, witness detection, pruning
│   │   ├── combat.py          # Combat resolution mechanic, damage calc, flee/defend
│   │   ├── difficulty.py      # Difficulty scaling config, presets, adaptive adjustment
│   │   └── narration.py       # Action outcome narration templates + LLM enhancement
│   │
│   ├── quest/
│   │   ├── __init__.py
│   │   ├── mdp.py             # MDP data structures, transitions
│   │   ├── quest_manager.py   # Quest state tracking, stage progression
│   │   ├── checkpoint.py      # Checkpoint generation (template + LLM)
│   │   └── nudge.py           # Nudging system, reward shaping
│   │
│   ├── npc/
│   │   ├── __init__.py
│   │   ├── npc.py             # NPC base class, UID generation, archetype loading, stats, dialogue resolution
│   │   ├── personality.py     # Personality archetype registry, generic response templates, NPC instance factory
│   │   ├── dialogue.py        # Dialogue pipeline: scripted check → LLM generation → archetype fallback
│   │   ├── interactions.py    # NPC-to-NPC interaction system, gossip propagation, target resolution
│   │   ├── knowledge.py       # NPC knowledge base, witnessed events, gossip memory
│   │   ├── rl_agent.py        # Q-learning implementation
│   │   └── schedule.py        # Fallback schedule system
│   │
│   ├── player/
│   │   ├── __init__.py
│   │   ├── player.py          # Player state, stats, inventory, per-NPC reputation
│   │   ├── actions.py         # Action processing, validation, edge case handling
│   │   └── input_parser.py    # spaCy NLP + 3D analysis (emotion/intent/social), vector similarity against universal action catalog, action mapping
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_service.py     # LLM loading, inference, retry logic
│   │   ├── prompts.py         # Prompt templates (checkpoint, dialogue, narration, analysis)
│   │   ├── guardrails.py      # Output validation, JSON schema checks, value clamping, content filter
│   │   └── fallback.py        # Template-based fallback generation
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # REST API endpoints (incl. save/load)
│   │   ├── websocket.py       # WebSocket for real-time updates + reconnection handling
│   │   └── session.py         # Session management, state recovery on reconnect
│   │
│   ├── data/
│   │   ├── quests/
│   │   │   └── main_quest.json
│   │   ├── npcs/
│   │   │   ├── archetypes/
│   │   │   │   ├── elder.json
│   │   │   │   ├── farmer.json
│   │   │   │   ├── merchant.json
│   │   │   │   ├── tavern_keeper.json
│   │   │   │   ├── guard.json
│   │   │   │   └── villager.json
│   │   │   └── instances/
│   │   │       ├── elder_maren.json
│   │   │       ├── farmer_jak.json
│   │   │       ├── tessa.json
│   │   │       ├── guard_aldric.json
│   │   │       ├── guard_bryn.json
│   │   │       └── villager_petra.json
│   │   ├── world/
│   │   │   └── locations.json
│   │   ├── config/
│   │   │   ├── difficulty_presets.json  # Easy/Normal/Hard parameter sets
│   │   │   ├── event_catalog.json      # Random event definitions
│   │   │   └── profiles/
│   │   │       └── default.json        # A/B config profiles for experiments
│   │   ├── saves/
│   │   │   └── .gitkeep
│   │   ├── metrics/
│   │   │   └── .gitkeep                # Session metrics JSONs written here
│   │   └── logs/
│   │       └── .gitkeep                # Structured JSON logs (session_*.jsonl)
│   │
│   └── tools/
│       ├── __init__.py
│       ├── replay.py              # Replay/playback engine for post-session review
│       ├── export_narrative.py    # Export event log as Markdown/JSON narrative
│       └── playtest_bot.py        # Automated playtesting bot (random/quest/explorer/combat)
│
├── frontend/
│   ├── index.html             # Main page (modern minimal dark theme)
│   ├── css/
│   │   ├── style.css          # Base layout, CSS Grid, dark theme variables
│   │   ├── components.css     # Action palette, stats cards, narrative log
│   │   └── graph.css          # Cytoscape.js container + overlays
│   └── js/
│       ├── app.js             # Main application logic, layout orchestration
│       ├── game.js            # Narrative log + action palette (categorized universal actions) + text input
│       ├── graph.js           # Cytoscape.js MDP visualization
│       ├── stats.js           # Player stats sidebar (progressive disclosure, per-NPC reputation)
│       ├── metrics.js         # Research metrics dashboard (collapsible panel)
│       ├── history.js         # Turn history navigation (click past turns for read-only snapshots)
│       ├── npc-panel.js       # NPC location indicator + conversation review panel
│       ├── audio.js           # Web Audio API procedural sound cues (muted by default)
│       └── api.js             # API client (REST + WebSocket + save/load)
│
├── models/                    # GGUF model files (gitignored)
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 12. Implementation Phases

### Phase 1a — Backend Core (Data + Game Loop) `~4-5 days`

**Goal:** Backend game engine: player can move, act, fight, and the world responds with full narration.

- [ ] Set up project structure + FastAPI skeleton
- [ ] Implement `Player` class with health, stamina, inventory, **per-NPC reputation dict**, combat stats, equipped slots
- [ ] Implement `World` with locations and connections
- [ ] Implement time system (turns → time periods)
- [ ] Implement basic action processing (resolve any universal action against current context)
- [ ] Implement action resolution pipeline: precondition check → context evaluation → outcome
- [ ] Implement **combat resolution mechanic** (hit probability, damage calc, defend/flee)
- [ ] Implement **action outcome narration** (template layer + context modifiers)
- [ ] Implement **world memory / event log** (log all actions, detect witnesses)
- [ ] Implement **difficulty scaling config** (presets: easy/normal/hard + parameter loading)
- [ ] Implement **edge case handling** for common scenarios (inventory full, 0 stamina, NPC incapacitation, etc.)
- [ ] Implement **equipment system** (`equip`/`drop_item` actions, weapon/armor modifiers to combat stats)
- [ ] Implement `InputParser` — spaCy pipeline (`en_core_web_md`), 3D analysis (emotion/intent/social), vector similarity matching against universal action catalog, synonym map fallback, confidence scoring

### Phase 1b — Frontend + API `~3-4 days`

**Goal:** Frontend connected to backend; player can interact through browser.

- [ ] Create REST API: `/action`, `/state`, `/locations`, **`/save`, `/load`, `/saves`**
- [ ] Build modern minimal frontend: dark theme, CSS Grid layout, narrative log + categorized action palette + free-text input + stats sidebar
- [ ] Implement WebSocket for real-time updates
- [ ] End-to-end integration testing (frontend → API → engine → response)

### Phase 2 — Quest MDP System `~3-4 days`

**Goal:** Quest with major stages and static checkpoints is playable.

- [ ] Implement MDP data structures (Stage, Checkpoint, Transition)
- [ ] Load quest from JSON definition
- [ ] Implement `QuestManager` — track current stage/checkpoint
- [ ] Implement checkpoint transitions (action → next checkpoint)
- [ ] Implement stage completion detection
- [ ] Add quest-related actions to action processing
- [ ] Create `main_quest.json` with all 7 stages and static checkpoints
- [ ] Author all narration/dialogue templates for 27 actions × outcome types (~1-2 days content work)
- [ ] Design and write quest checkpoint descriptions, scripted dialogue lines, and NPC flavor text

### Phase 3 — MDP Visualization `~2-3 days`

**Goal:** Live graph showing quest MDP with current position.

- [ ] Set up Cytoscape.js in frontend
- [ ] Create graph layout: stages as compound nodes, checkpoints as child nodes
- [ ] Style nodes by type (major/checkpoint/dynamic) and state (current/complete/locked)
- [ ] Implement WebSocket for real-time graph updates
- [ ] Highlight current position and completed path
- [ ] Animate transitions on player action

### Phase 4 — Dynamic Checkpoints + Nudging + Events `~3-4 days`

**Goal:** System handles unexpected player actions gracefully; world has emergent events.

- [ ] Implement checkpoint template system (4-5 templates)
- [ ] Detect quest-irrelevant action outcomes (actions from universal catalog that produce novel situations at a checkpoint)
- [ ] Generate dynamic checkpoints from templates
- [ ] Insert dynamic checkpoints into MDP graph
- [ ] Implement nudging: hints in descriptions, reward shaping
- [ ] Implement escalation (3 deviations = strong hint, 5 = forced convergence)
- [ ] Update visualization in real-time for new dynamic nodes
- [ ] Implement **random events system** (event catalog, trigger conditions, duration, effects)
- [ ] Create `event_catalog.json` with all event definitions
- [ ] Random event narration integration with action outcome narration system

### Phase 5 — NPC RL Agents `~4-5 days`

**Goal:** All 6 NPCs make autonomous, learning decisions and interact with each other.

- [ ] Implement personality archetype system: load archetype templates, NPC instance factory with UID generation
- [ ] Implement `NPC` base class: stats, archetype link, conversation history, UID-based identity, **per-NPC reputation tracking**
- [ ] Implement dialogue resolution pipeline: scripted check → LLM generation → archetype generic fallback
- [ ] Implement **NPC-to-NPC interaction system** (gossip propagation, social interactions, target resolution)
- [ ] Implement **NPC knowledge base** (witnessed events, gossip memory, player opinion factors)
- [ ] Implement tabular Q-learning agent
- [ ] Discretize NPC state space (location × time × energy × mood)
- [ ] Implement NPC action execution and reward calculation
- [ ] NPCs act every player turn (simultaneous); **resolve NPC-NPC interactions + gossip**
- [ ] Q-table persistence (save/load)
- [ ] Pre-train NPCs with ~100 simulated episodes before game start
- [ ] Create archetype JSONs: elder, farmer, merchant, tavern_keeper, guard, villager
- [ ] Create NPC instances: Elder Maren, Farmer Jak, Tessa, Aldric, Bryn, Old Petra
- [ ] NPC behavior affects quest (Jak finds item, Tessa shares rumor)

### Phase 6 — LLM Integration (Optional Enhancement) `~2-3 days`

**Goal:** LLM generates richer checkpoints, dialogue, and narration with robust guardrails.

- [ ] Set up llama-cpp-python with qwen3-4B GGUF
- [ ] Implement `LLMService` with load/generate/validate
- [ ] Implement **LLM output validation & guardrails** (JSON schema validation, value clamping, content filtering)
- [ ] Implement **retry logic** (max 2 retries with timeout)
- [ ] Create prompt templates for checkpoint generation
- [ ] Create prompt templates for NPC dialogue (unscripted context generation)
- [ ] Create prompt templates for **action outcome narration enhancement**
- [ ] Integrate LLM into dialogue resolution pipeline: generate in-character responses for any unscripted interaction
- [ ] Integrate LLM into checkpoint generation pipeline (with fallback)
- [ ] Integrate LLM into **narration pipeline** (Layer 2 enhancement)
- [ ] Verify conversation_history is passed to LLM for contextual dialogue
- [ ] Test with and without LLM (graceful degradation: scripted → LLM → archetype generic template)

### Phase 7 — Polish & Demo `~5-7 days`

**Goal:** Clean, demonstrable MVP with research tooling and §14.1–14.2 features.

- [ ] End-to-end playtesting
- [ ] UI polish: dark theme refinement, typography, micro-interactions, responsive breakpoints
- [ ] Action palette UX: context-relevance highlighting, category collapse/expand, dimmed-but-clickable styling
- [ ] Add game log / narrative history panel with smooth scroll
- [ ] Add NPC panel (show what NPCs are doing, collapsible)
- [ ] Implement **metrics & analytics dashboard** (research panel, collapsible, tracks all metrics)
- [ ] Implement **save/load UI** (save slots, auto-save indicator, load menu)
- [ ] Implement **per-NPC reputation display** in stats sidebar
- [ ] Performance optimization (LLM inference caching)
- [ ] **Adaptive difficulty** tuning (optional: auto-adjust based on event log analytics)
- [ ] Export session metrics to `data/metrics/session_{id}.json`
- [ ] Implement **master random seed** (`config.py` MASTER_SEED, seeded at session start)
- [ ] Implement **structured JSON logging** (`JSONFormatter`, log all action resolution steps)
- [ ] Configure **FastAPI auto-docs** (Swagger at `/docs`, ReDoc at `/redoc`, tagged endpoints)
- [ ] Implement **keyboard shortcuts** (1-9 palette, Enter/Esc, M/S/H toggles, / for text)
- [ ] Implement **NPC location indicator** sidebar widget (updated every turn, click-to-tooltip)
- [ ] Implement **turn history navigation** (clickable past turns, read-only state snapshots)
- [ ] Implement **conversation review panel** (per-NPC chronological interaction history)
- [ ] Implement **LLM rate limiter** (2s min interval, 20 calls/min max, auto-fallback)
- [ ] README with setup and demo instructions
- [ ] Record demo playthrough scenarios

### Phase 8 — Research Tooling & Stretch Goals (Post-MVP) `~5-7 days`

**Goal:** Advanced research tooling for systematic experiments and data collection (§14.3–14.4).

- [ ] Implement **replay/playback engine** (`tools/replay.py`: step forward/backward, turn slider, API endpoints)
- [ ] Implement **narrative export** (`tools/export_narrative.py`: Markdown and JSON export from event log)
- [ ] Implement **parameter override via URL** (`?seed=42&difficulty=hard&llm=off`)
- [ ] Implement **NPC "thought bubble" debug mode** (Q-values overlay, toggle with D key)
- [ ] Implement **A/B configuration profiles** (named profiles in `data/config/profiles/`, loaded via URL)
- [ ] Implement **post-session analytics export** (auto-generate summary JSON at session end)
- [ ] Implement **NPC relationship matrix visualization** (color-coded heatmap, gossip drill-down)
- [ ] Implement **automated playtesting bot** (`tools/playtest_bot.py`: 4 strategies, batch CLI)
- [ ] Implement **basic sound cues** (`audio.js`: Web Audio API procedural tones, muted by default)
- [ ] Implement **undo/rewind** (snapshot every turn, `POST /api/undo`, research mode only)
- [ ] Implement **multi-quest groundwork** (`QuestRegistry` pattern, single quest via registry interface)
- [ ] Cross-session analytics comparison tooling
- [ ] Stress-test with 100+ automated sessions via playtest bot

---

## 13. Example Playthrough

```
═══════════════════════════════════════════════════
  TURN 1 | Morning | Gate
═══════════════════════════════════════════════════
  
  [Quest: The Elder's Lost Artifact — Stage 1: Go to Village]
  [Checkpoint 1.1: Approach the village gate]

  You stand before the wooden gates of Thornhaven. 
  Two guards lean against their posts, eyeing you warily.

  Actions:
    1. Greet the guards          (Social)     [-1 AP]
    2. Show travel papers        (Social)     [-1 AP, requires: travel_papers]
    3. Sneak past the guards     (Stealth)    [-5 AP, 60% success]
    4. Look around               (Explore)    [-1 AP]

  Stats: HP 100/100 | AP 50/50 | Rep 0 | Items: travel_papers, bread, coin_pouch

  ┌─────────────────────────────────────────┐
  │ [Greet guards] [Show papers] [Sneak]    │  ← Choice Buttons
  │                                         │
  │ > Type something...                     │  ← Text Input
  └─────────────────────────────────────────┘

  > Player types: "I want to look around first"
  → 3D Analysis: emotion=curious, intent=look, social=neutral
  → Intent match: "look" → resolved from universal action catalog
  → Action `look` is valid but not in checkpoint's highlighted_actions
  → Treated as deviation from expected path → dynamic CP generated...
  
═══════════════════════════════════════════════════
  TURN 2 | Morning | Gate
═══════════════════════════════════════════════════

  [Quest: Stage 1 | Checkpoint 1.D1 (DYNAMIC)]

  You look around the gate area. There's a worn notice board 
  with faded postings, and a merchant's cart parked nearby.
  In the distance, you can hear the bustle of the village.

  Actions:
    1. Read the notice board      (Explore)    [-1 AP]
    2. Talk to the merchant       (Social)     [-1 AP]  
    3. Go greet the guards        (Social)     [-1 AP]  ← nudge
    4. Examine the cart           (Explore)    [-2 AP]

  💡 Hint: The guards seem approachable. Getting into 
     the village is your first priority.

  Stats: HP 100/100 | AP 50/50 | Rep 0 | Items: travel_papers, bread, coin_pouch

  ℹ️ AP note: look costs 1 AP, but +5 passive regen per turn
     caps at max. Net: 50 − 1 + 5 = 50/50

  ┌──────────────────────────────────────────────┐
  │ [Read board] [Talk merchant] [Greet guards]  │
  │ [Examine cart]                               │
  │                                              │
  │ > Type something...                          │
  └──────────────────────────────────────────────┘

  ── Meanwhile ──
  🧑‍🌾 Farmer Jak wakes up and heads to the fields.
  🍺 Tessa opens the tavern and starts preparing.

  > Player clicks: [Greet guards]  (button click → direct action mapping)
  → 3D defaults: emotion=neutral, intent=greet, social=neutral
  → Returns to main flow: Checkpoint 1.1 → 1.2

═══════════════════════════════════════════════════
  [MDP GRAPH UPDATED]
  Node Cp_1_D1 (orange diamond) added between Cp_1_1 and Cp_1_2
  Current position: Cp_1_2 (yellow pulse)
═══════════════════════════════════════════════════
```

---

## 14. Additional Features & Research Tooling

> **Scope note:** These features enhance the MVP's research value and developer experience. Items in §14.1–14.2 should be implemented during Phase 7; items in §14.3–14.4 are stretch goals for post-MVP or Phase 8.

---

### 14.1 Reproducibility & Debugging (High Priority)

#### 14.1.1 Random Seed for Reproducibility

All sources of randomness must be seeded from a single master seed so that any session can be exactly replayed.

```python
# In config.py
MASTER_SEED = 42  # Override via URL param ?seed=N or config file

# At session start (game_engine.py)
import random, numpy as np
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
```

**Rules:**

- The seed is stored in save files (`game_state.master_seed`).
- Every random call (combat rolls, event triggers, NPC Q-learning exploration) uses `random` or `np.random` — never `os.urandom`.
- The seed is logged at session start for reproducibility: `logger.info(f"Session seed: {MASTER_SEED}")`.
- LLM outputs are NOT seeded (non-deterministic by nature); the system logs the raw LLM output for each call so it can be replayed with template fallback.

#### 14.1.2 Structured Logging

Use Python's `logging` module with JSON-formatted output for machine-readable logs.

```python
# In config.py
import logging, json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "turn": getattr(record, "turn", None),
            "event": getattr(record, "event", None),
            "message": record.getMessage()
        }
        return json.dumps(log_entry)

LOG_FILE = "data/logs/session_{session_id}.jsonl"
LOG_LEVEL = logging.DEBUG  # DEBUG in dev, INFO in demo
```

**What to log (minimum):**

| Category | Events | Level |
|----------|--------|-------|
| Action Resolution | Every step: raw input → NLP parse → action match → precondition check → outcome roll → result | DEBUG |
| NPC Decisions | Q-value lookup, ε-roll result, chosen action, reward received | DEBUG |
| Quest Progression | Checkpoint transitions, dynamic CP generation, nudge triggers | INFO |
| LLM Calls | Prompt (truncated to 200 chars), raw output, validation result, retry count | INFO |
| Combat | Hit roll, damage calc, defend/flee resolution | DEBUG |
| Gossip | Source → target, delta, hop count | DEBUG |
| Random Events | Event triggered, effects applied, duration | INFO |
| Save/Load | Save created, save loaded, auto-save triggered | INFO |
| Errors | Any exception, validation failure, LLM timeout | ERROR |

#### 14.1.3 Auto-Generated API Documentation

FastAPI provides this for free — ensure it's properly configured:

```python
# In main.py
app = FastAPI(
    title="MVP Game/Simulation API",
    description="MDP + RL + LLM Research Game",
    version="1.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
)
```

**Rules:**

- Every endpoint must have a docstring and response model.
- Add example values to Pydantic models using `model_config` / `json_schema_extra`.
- Tag endpoints by category: `game`, `quest`, `npc`, `llm`, `save`, `metrics`.

---

### 14.2 UI Enhancements (High Priority)

#### 14.2.1 Keyboard Shortcuts

| Key | Action | Context |
|-----|--------|---------|
| `1`–`9` | Select Nth action from currently visible palette | Always |
| `Enter` | Focus free-text input field | When not already focused |
| `Escape` | Unfocus text input / close open panel | Always |
| `Tab` | Cycle through action palette categories | Action palette focused |
| `M` | Toggle MDP graph panel | Always |
| `S` | Toggle stats sidebar | Always |
| `H` | Toggle turn history panel | Always |
| `D` | Toggle NPC debug/thought bubble overlay (§14.3.4) | Research mode only |
| `N` | Toggle audio cues (§14.4.2) | Always |
| `/` | Focus free-text input (vim-style) | When not already focused |

```javascript
// In app.js
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;  // Don't intercept typing
    const key = e.key;
    if (key >= '1' && key <= '9') selectPaletteAction(parseInt(key) - 1);
    else if (key === 'Enter') document.getElementById('text-input').focus();
    else if (key === 'Escape') handleEscape();
    else if (key === 'm' || key === 'M') togglePanel('mdp-graph');
    else if (key === 's' || key === 'S') togglePanel('stats');
    else if (key === 'h' || key === 'H') togglePanel('history');
    else if (key === 'd' || key === 'D') togglePanel('npc-debug');  // Research mode: NPC debug overlay
    else if (key === 'n' || key === 'N') toggleAudioCues();          // Toggle audio cues on/off
    else if (key === 'Tab') { e.preventDefault(); cycleActionCategory(); }  // Cycle palette categories
    else if (key === '/') { e.preventDefault(); document.getElementById('text-input').focus(); }
});
```

#### 14.2.2 Turn History Navigation

Players and researchers can click on past turns to inspect historical state.

- **Narrative log entries** are clickable: clicking a past turn entry loads a **read-only snapshot** of that turn's state.
- Read-only mode shows: player stats at that turn, NPC locations, quest state, inventory.
- A "Return to Current" button brings back the live view.
- Implementation: the `event_log` already stores per-turn state; the UI reads from it.

#### 14.2.3 NPC Location Indicator

A sidebar widget shows which NPCs are at each location, updated every turn.

```
📍 Gate:          Guard Aldric, Guard Bryn
📍 Village Center: Old Petra
📍 Fields:        Farmer Jak
📍 Tavern:        Tessa
📍 Elder's House: Elder Maren
```

- Shows only NPC names (no stats) to avoid clutter.
- NPCs at the player's current location are **highlighted**.
- Clicking an NPC name opens a tooltip: current mood, reputation with player, last action.

#### 14.2.4 Conversation Review Panel

Players can review full conversation history with any NPC they've spoken to.

- Accessible from the NPC location indicator (click NPC → "View Conversation History").
- Shows chronological list of all `talk`, `greet`, `ask_info`, `persuade` interactions with that NPC.
- Each entry shows: turn number, player action, NPC response, reputation change.
- **Read-only** — no interaction from this panel.

---

### 14.3 Research Tooling (Medium Priority — Stretch Goals)

#### 14.3.1 Replay / Playback Mode

Step through a completed session's `event_log` turn-by-turn to study player-NPC interactions.

```python
# In backend/tools/replay.py
class ReplayEngine:
    def __init__(self, event_log_path: str):
        self.events = load_json(event_log_path)
        self.current_turn = 0
    
    def step_forward(self) -> dict:
        """Return the state snapshot for the next turn."""
        self.current_turn += 1
        return self.events[self.current_turn]
    
    def step_backward(self) -> dict:
        """Return the state snapshot for the previous turn."""
        self.current_turn = max(0, self.current_turn - 1)
        return self.events[self.current_turn]
    
    def jump_to(self, turn: int) -> dict:
        """Jump to a specific turn."""
        self.current_turn = clamp(turn, 0, len(self.events) - 1)
        return self.events[self.current_turn]
```

- **API:** `GET /api/replay/load?file=session_001.json`, `GET /api/replay/step?direction=forward`
- **Frontend:** Play/pause button, step forward/backward arrows, turn slider, playback speed selector.
- The MDP graph, narrative log, and stats all animate during playback.

#### 14.3.2 Export Playthrough as Narrative

Auto-generate a Markdown story document from the event log — useful for research papers and demos.

```python
# In backend/tools/export_narrative.py
def export_as_markdown(event_log: list, output_path: str):
    """Convert event log to a readable narrative Markdown file."""
    md = "# Session Playthrough Narrative\n\n"
    for event in event_log:
        if event["type"] == "player_action":
            md += f"**Turn {event['turn']}** — *{event['location']}*\n\n"
            md += f"> {event['narration']}\n\n"
        elif event["type"] == "npc_action":
            md += f"*Meanwhile, {event['npc_name']} {event['action']}...*\n\n"
        elif event["type"] == "quest_progress":
            md += f"---\n\n🏁 **{event['description']}**\n\n---\n\n"
    write_file(output_path, md)
```

- **API:** `GET /api/export/narrative?format=markdown` → returns `.md` file download.
- Also supports JSON export: `GET /api/export/narrative?format=json`.

#### 14.3.3 Parameter Override via URL

Allow researchers to configure sessions without editing config files:

```
http://localhost:8000/?seed=42&difficulty=hard&llm=off&max_turns=100
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Master random seed |
| `difficulty` | string | "normal" | Difficulty preset name |
| `llm` | string | "on" | `on` / `off` / `template_only` |
| `max_turns` | int | 200 | Maximum turns before session auto-ends |
| `npc_pretrain` | int | 100 | Number of pre-training episodes per NPC |
| `profile` | string | — | Named config profile (see §14.3.5) |

- URL params override `config.py` defaults.
- Applied at session initialization; logged in the session metadata.

#### 14.3.4 NPC "Thought Bubble" Debug Mode

A toggleable overlay showing NPC decision-making in real-time.

When enabled (toggle with `D` key or checkbox in metrics panel), each NPC action shows:

```
🧠 Farmer Jak's Decision:
   State: (fields, morning, 80, content)
   Q-values: {work_field: 2.3, rest: 1.1, move_to_tavern: 0.8, talk: 0.5, ...}
   ε-roll: 0.12 (exploit)  →  Chose: work_field (Q=2.3)
   Reward received: +1.2
```

- **Frontend-only feature** — the backend already returns Q-values in debug state; this just renders them.
- Shows top 5 Q-values (sorted descending) to avoid clutter.
- Hidden by default; only visible in debug/research mode.

#### 14.3.5 A/B Configuration Profiles

Named config profiles for systematic experiments:

```json
// data/config/profiles/
{
  "profile_name": "high_exploration",
  "description": "NPCs explore more, player gets less nudging",
  "overrides": {
    "NPC_EPSILON_START": 0.4,
    "NPC_EPSILON_MIN": 0.15,
    "NUDGE_ESCALATION_THRESHOLD": 8,
    "DYNAMIC_CP_COOLDOWN": 2,
    "LLM_ENABLED": true
  }
}
```

- Stored in `data/config/profiles/*.json`.
- Loaded via URL param `?profile=high_exploration` or config selection in UI.
- Session metadata records which profile was active for analysis.

#### 14.3.6 Post-Session Analytics Export

Auto-generate a summary JSON at session end with key metrics for cross-session comparison.

```python
# Auto-generated at session end → data/metrics/summary_{session_id}.json
{
    "session_id": "2025-01-15_14-30-00",
    "seed": 42,
    "difficulty": "normal",
    "profile": "default",
    "total_turns": 87,
    "quest_completed": true,
    "completion_time_turns": 87,
    "dynamic_checkpoints_generated": 3,
    "nudges_triggered": 2,
    "forced_convergences": 0,
    "unique_actions_used": 18,
    "combat_encounters": 4,
    "npc_interactions": {
        "elder_maren": {"talks": 5, "final_reputation": 32},
        "farmer_jak": {"talks": 8, "final_reputation": 45},
        "tessa": {"talks": 3, "final_reputation": 20}
    },
    "llm_calls": {"total": 23, "successes": 21, "fallbacks": 2, "avg_latency_ms": 1200},
    "q_table_convergence": {"jak": 0.85, "tessa": 0.72, "maren": 0.91}
}
```

#### 14.3.7 NPC Relationship Matrix Visualization

A grid visualization showing NPC-to-NPC relationships and opinions.

```
              Maren   Jak    Tessa   Aldric  Bryn   Petra
Maren           —     +15     +10     +20    +20    +25
Jak            +15      —     +30      +5     +5    +10
Tessa          +10    +30       —      +8     +5    +15
Aldric         +20     +5      +8       —    +40     +5
Bryn           +20     +5      +5     +40      —     +5
Petra          +25    +10     +15      +5     +5      —
```

- Rendered as a color-coded heatmap (green = positive, red = negative).
- Cell click shows the gossip chain and events that led to the current value.
- Updates live as gossip propagates.
- Accessible from the metrics dashboard.

#### 14.3.8 Automated Playtesting Bot

A bot that plays through sessions automatically for stress-testing and data collection.

```python
# In backend/tools/playtest_bot.py
class PlaytestBot:
    STRATEGIES = {
        "random": lambda actions: random.choice(actions),
        "quest_focused": lambda actions: pick_quest_relevant(actions),
        "explorer": lambda actions: pick_exploration_biased(actions),
        "combat_heavy": lambda actions: pick_combat_biased(actions),
    }
    
    def __init__(self, strategy: str = "random", max_turns: int = 200):
        self.strategy = self.STRATEGIES[strategy]
        self.max_turns = max_turns
    
    def run_session(self, seed: int) -> dict:
        """Run a full automated session, return analytics summary."""
        game = GameEngine(seed=seed)
        for turn in range(self.max_turns):
            available_actions = game.get_available_actions()
            chosen = self.strategy(available_actions)
            game.process_action(chosen)
            if game.quest_completed:
                break
        return game.export_analytics()
```

- **CLI usage:** `python -m backend.tools.playtest_bot --strategy=random --sessions=100 --seed-start=1`
- Outputs: per-session summaries + aggregate statistics (completion rate, avg turns, action distribution).
- Useful for: finding edge cases, measuring quest completion difficulty, testing NPC convergence.

---

### 14.4 Quality of Life (Medium Priority)

#### 14.4.1 Rate Limiting on LLM Calls

Prevent LLM call spam during rapid interactions:

```python
# In llm_service.py
LLM_MIN_INTERVAL_MS = 2000  # Minimum 2 seconds between LLM calls
LLM_MAX_CALLS_PER_MINUTE = 20

class LLMRateLimiter:
    def __init__(self):
        self.last_call_time = 0
        self.calls_this_minute = 0
        self.minute_start = time.time()
    
    def can_call(self) -> bool:
        now = time.time()
        if now - self.minute_start > 60:
            self.calls_this_minute = 0
            self.minute_start = now
        if now - self.last_call_time < LLM_MIN_INTERVAL_MS / 1000:
            return False
        if self.calls_this_minute >= LLM_MAX_CALLS_PER_MINUTE:
            return False
        return True
    
    def record_call(self):
        self.last_call_time = time.time()
        self.calls_this_minute += 1
```

- When rate limited, fall back to template generation immediately (no queue).
- Log rate-limit events for analysis.

#### 14.4.2 Basic Sound / Notification Cues

Minimal audio feedback using Web Audio API (no external files needed):

| Event | Sound | Implementation |
|-------|-------|----------------|
| Turn processed | Soft click (50ms sine wave, 440Hz) | `oscillator.frequency.value = 440` |
| Quest checkpoint reached | Two-tone chime (C5 → E5) | Two sequential oscillators |
| Combat hit | Short noise burst (100ms) | White noise buffer |
| NPC approaches | Low tone (200Hz, 200ms) | `oscillator.frequency.value = 200` |
| Session saved | Confirmation beep (880Hz, 100ms) | Single high tone |

```javascript
// In frontend/js/audio.js
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playTone(frequency, duration, type = 'sine') {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = frequency;
    gain.gain.value = 0.1;  // Low volume
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration / 1000);
}
```

- Muted by default; toggle with 🔇 button or `N` key.
- Volume stored in `localStorage`.

#### 14.4.3 Undo / Rewind in Research Mode

Allow researchers to undo the last N turns for experimentation:

- **Snapshot system:** every turn, save a lightweight state snapshot (player stats, NPC states, quest state, Q-tables). Store last 20 snapshots in memory.
- **Undo action:** `POST /api/undo` → restore the previous snapshot, decrement turn counter.
- **Rewind to turn N:** `POST /api/rewind?turn=N` → restore snapshot N, discard all later entries from event log.
- **Constraint:** undo does NOT revert LLM-generated text (it was already generated). The narration stays in the log but is marked `[UNDONE]`.
- **UI:** "Undo" button in research mode (hidden in normal play).

#### 14.4.4 Multi-Quest Groundwork

Prepare the data model for future multi-quest support without implementing it:

```python
# In quest_manager.py
class QuestRegistry:
    """Registry of all available quests. MVP: contains exactly one quest."""
    def __init__(self):
        self.quests: dict[str, Quest] = {}  # quest_id → Quest
        self.active_quest_id: str = "main_quest_01"  # MVP: hardcoded
    
    def get_active_quest(self) -> Quest:
        return self.quests[self.active_quest_id]

# In game_state
"quest_registry": {
    "quests": {"main_quest_01": { ... }},
    "active_quest_id": "main_quest_01"
}
```

- The save file stores a `quest_registry` instead of a flat quest object.
- All quest lookups go through `QuestRegistry.get_active_quest()`.
- Future expansion: add side quests by registering them in the registry and tracking `active_quests: list[str]`.

---

## Summary of Improvements Over Plan v1

| Area              | v1 (Original)             | v2 (Improved)                                                            |
|-------------------|---------------------------|--------------------------------------------------------------------------|
| MDP               | Mentioned, not formalized | Full formal spec with $(S, A, T, R, \gamma)$ for both quest and NPC      |
| **Action Space**  | **Per-checkpoint limited** | **Universal action catalog (27 actions) available to all entities at all times; preconditions + context determine outcomes** |
| Tech Stack        | Not specified             | Python + FastAPI + Cytoscape.js + llama-cpp-python                       |
| **UI Design**     | **Not specified**         | **Modern minimal: dark theme, CSS Grid, typography-first, color-as-signal, progressive disclosure** |
| **NPC Identity**  | **Not specified**         | **Generic personality archetypes (6 types) instantiated with unique UIDs; conversation history per NPC; LLM generates dialogue for unscripted contexts** |
| **NPC-NPC**       | **Not specified**         | **NPCs interact with each other: gossip propagation, social exchanges, information network creating organic reputation spread** |
| NPC RL            | "RL agent" vague          | Tabular Q-learning, universal action space (4,860 Q-table entries), soft action masking |
| **Reputation**    | **Single global integer** | **Per-NPC reputation dict: direct changes + witness bonuses + gossip propagation with decay; each NPC has individual opinion of player** |
| **Combat**        | **Not specified**         | **Stat-based probabilistic combat: hit probability formula, damage calc, defend (-50% dmg), flee (70% success), AP costs** |
| **Random Events** | **Not specified**         | **8 event types (weather, merchant, theft, accident, festival, etc.) with trigger conditions, probabilities, durations, and NPC behavior impact** |
| **Narration**     | **Not specified**         | **3-layer narration: template strings → LLM enhancement → context modifiers; structured output with mechanical effects + witness notes** |
| **World Memory**  | **Not specified**         | **Central event log with witness detection; NPC knowledge bases; gossip as information propagation; pruning for log size management** |
| **Edge Cases**    | **Not specified**         | **11 defined edge cases with specific resolution: inventory full + quest item, 0 stamina combat, all NPCs hostile, CP loops, save corruption, NPC incapacitation, compound input, etc.** |
| **Difficulty**    | **Not specified**         | **12 tunable parameters with Easy/Normal/Hard presets; optional adaptive difficulty based on event log analytics** |
| **LLM Guardrails**| **Not specified**         | **JSON schema validation, value bounds clamping, retry logic (2 retries, 10s timeout), content filtering, structured validation pipeline** |
| **Metrics**       | **Not specified**         | **Research analytics dashboard: deviation rate, LLM vs template usage, Q-table convergence, action distribution, reputation timeline, gossip network, LLM performance stats** |
| **Save/Load**     | **Not specified**         | **Full save specification: single JSON with all state (player, NPCs, Q-tables, event log, metrics); auto-save every 5 turns + on quest events; manual saves; backup/restore; API endpoints** |
| Nudging           | "Gently nudge" undefined  | Reward shaping formula, escalation levels, convergence rules             |
| LLM               | "Use GGUF"                | Full prompt templates, validation, graceful fallback pipeline            |
| Data Models       | None                      | Complete JSON schemas for quest, NPC, game state                         |
| Visualization     | "Nodal view"              | Cytoscape.js with specific node styles, colors, layout rules             |
| World             | Not designed              | 5 locations with connections, time system, NPC placement                 |
| Game Systems      | Listed                    | Full mechanics with numbers (HP values, AP costs, reputation thresholds) |
| Implementation    | No plan                   | 9 phases with dependencies and time estimates (Phases 1a–7: ~26-35 days core; Phase 8: ~5-7 days stretch) |
| Project Structure | None                      | Full directory tree with every file's responsibility                     |
| Player Input      | Not defined               | Dual input: action palette (categorized universal actions) + free-text chat, 3D analysis (emotion/intent/social) |
| **Reproducibility**| **Not addressed**        | **Master random seed, structured JSON logging, all randomness seeded and logged for exact session replay** |
| **API Docs**      | **Not addressed**         | **Auto-generated Swagger/ReDoc via FastAPI; tagged endpoints with example values** |
| **Keyboard UX**   | **Mouse-only**            | **Full keyboard shortcut set (1-9 palette, Enter/Escape, M/S/H toggles, / for text input)** |
| **Turn History**   | **Not addressed**        | **Clickable past turns in narrative log; read-only state snapshots; "Return to Current" button** |
| **NPC Locations**  | **Not visible**          | **Sidebar widget showing NPC-to-location mapping, updated every turn, with click-to-tooltip** |
| **Conversation Log**| **Not addressed**       | **Per-NPC conversation review panel: full chronological history of all social interactions** |
| **Replay Mode**   | **Not addressed**         | **Post-session replay engine: step forward/backward, turn slider, playback speed, animate MDP graph** |
| **Narrative Export**| **Not addressed**        | **Auto-generate Markdown story from event log; also JSON export; API endpoint for download** |
| **URL Config**    | **Not addressed**         | **Parameter override via URL (?seed=42&difficulty=hard&llm=off); logged in session metadata** |
| **Debug Mode**    | **Not addressed**         | **NPC "thought bubble" overlay: shows Q-values, ε-roll, chosen action, reward in real-time** |
| **Config Profiles**| **Not addressed**        | **Named A/B config profiles for systematic experiments; loaded via URL or UI** |
| **Analytics Export**| **Not addressed**        | **Auto-generated summary JSON at session end with key metrics for cross-session comparison** |
| **NPC Relationships**| **Not addressed**      | **Color-coded NPC-to-NPC relationship matrix heatmap with gossip chain drill-down** |
| **Playtest Bot**  | **Not addressed**         | **Automated playtesting bot: random/quest/explorer/combat strategies; batch 100 sessions via CLI** |
| **LLM Rate Limit**| **Not addressed**        | **Min 2s interval, max 20 calls/min; auto-fallback to templates when limited** |
| **Audio Cues**    | **Not addressed**         | **Web Audio API procedural sounds: click, chime, combat, NPC approach; muted by default** |
| **Undo/Rewind**   | **Not addressed**         | **Research-mode undo: last 20 snapshots in memory; rewind to any turn; POST /api/undo endpoint** |
| **Multi-Quest Prep**| **Not addressed**       | **QuestRegistry pattern: single quest via registry interface; future-proofed for side quests** |

### New Systems Added in This Revision

| System | Section | Description |
|--------|---------|-------------|
| **Equipment** | §2, §5.3 | `equip`/`drop_item` actions, weapon/armor modifier slots, combat stat bonuses |
| **Passive Perception** | §5.8 | Layer 4 narration: `wait` action triggers automatic hidden object/threat detection |
| **NPC Death & Incapacitation** | §5.6 | Non-quest NPCs incapacitated → return after 20 turns; quest-critical NPCs floor at 1 HP |
| **Victory & Defeat Flow** | §5.11 | Complete UI flow for quest success (S_success) and player defeat (S_fail) |
| **Success Probability Formulas** | §5.6 | Formal probability equations for persuade, deceive, sneak, steal, hide, search |
| **Event Importance Scoring** | §5.7 | 5-level rubric (trivial→quest-critical) for event log prioritization and pruning |
| **Per-Use-Case LLM Temperature** | §7.1 | Different temperature settings for parsing (0.3-0.5), dialogue (0.7), checkpoint gen (0.8-0.9) |
| **Async LLM Inference** | §7.1 | `asyncio.to_thread()` wrapper to prevent blocking FastAPI event loop |
| **NPC Relationships** | §9.3 | Per-NPC opinion dict tracking NPC-to-NPC relationship values |
| **Deterministic UIDs** | §4.4 | Seed-based UID generation for full session reproducibility |

---
