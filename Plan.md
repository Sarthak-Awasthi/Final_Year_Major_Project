# MVP Plan v3 - RL Playground Transition Plan

> Updated: 2026-04-22
> 
> This document replaces the previous MVP plan with an implementation-first roadmap for evolving the current game into an RL Playground where player actions reshape village dynamics.
> 
> **LLM Runtime Decision (Updated):** remove direct `llama-cpp-python` integration. All LLM access will be done via API calls to a provider endpoint (self-hosted `llama.cpp` server, Ollama, or other compatible providers) through a single backend adapter layer.

---

## 1) Vision and Outcome

Build a robust **single-session RL playground** where:

- NPCs have persistent, role-grounded personalities and wellness states.
- NPCs optimize both **self-interest** and **community-level welfare**.
- Player actions and narrative events trigger **system shocks** that force adaptation.
- Learning progression is measurable and visualized over time (individualism -> cooperation shift).

Primary research hypothesis:

- Early episodes/turns should favor short-term individual gains.
- As agents learn and shocks are introduced, policies should move toward stronger community contribution due to higher long-run combined reward.

---

## 2) Scope and Constraints

### In Scope

- FastAPI backend + vanilla JS frontend (no framework migration).
- NumPy tabular Q-learning (no deep RL libraries).
- JSON persistence for saves, metrics, replay logs.
- Provider-agnostic LLM access over HTTP API (self-hosted `llama.cpp` server, Ollama, or external provider), with strict fallback behavior.

### Out of Scope (Current Plan)

- Multi-session multiplayer support.
- Database migration.
- Replacing Cytoscape.js or frontend stack.
- Device-specific local binding management (ROCm/CUDA/Vulkan-specific `llama-cpp-python` build paths).

### Compatibility Principle

We will preserve current runtime and API stability while transitioning:

- Keep `UNIVERSAL_ACTION_IDS` as canonical storage IDs.
- Introduce **role-specific policy masks** gradually (selection constraints, not hard API break).
- Gate RL Playground features behind config flags until validated.

---

## 3) Current Baseline (What Exists Today)

### Engine and Session

- Single active session model via `backend/api/session.py`.
- Main turn orchestration in `backend/engine/game_engine.py`.
- Save/load implemented with backup restore behavior.

### Quest and MDP

- Hierarchical quest graph with static + dynamic checkpoints.
- Cytoscape-compatible graph payloads already available.

### NPC RL

- Tabular Q-learning with epsilon-greedy selection and pretraining.
- Cold-start schedule then learned action selection.
- Reward currently based on NPC stat deltas only (individual-centric).

### LLM

- LLM remains optional and non-blocking for gameplay-critical paths.
- All model calls route through an API adapter (no in-process `llama-cpp-python`).
- Template fallback remains mandatory and first-class.

### Known Contract Risks to Resolve First

- Save/load UI contract drift vs backend payload shape.
- Metrics field-name mismatches between backend and frontend.
- Parsed compound `queued_action` is created but not executed.
- Action-catalog drift (`present_item` exists in backend but not consistently surfaced across all consumers).

---

## 4) Target Architecture for RL Playground

## 4.1 Core Runtime Loop

Each turn should explicitly execute:

1. Player action resolution.
2. Shock-state update (activation, decay, propagation).
3. NPC decision phase (role-masked actions + adaptive parameters).
4. Environment transition.
5. Reward computation per NPC:
   - Penalty term
   - Individual gain term
   - Community reward term
6. Q-update + adaptation update.
7. Metrics logging and snapshot emission.

## 4.2 Reward Model (Dual Reward)

For NPC `i` at turn `t`:

- `R_i(t) = P_i(t) + G_i(t) + lambda_i(t) * C(t)`

Where:

- `P_i(t)`: penalties (invalid actions, harmful instability, severe social damage, etc.).
- `G_i(t)`: individual utility delta (wealth/health/reputation/role target).
- `C(t)`: community welfare reward from village aggregates.
- `lambda_i(t)`: role/personality-specific community weight (static baseline + adaptive component).

Community welfare base vector:

- Village goodwill/trust aggregate.
- Village economic health/resources.
- Population wellness and safety.

## 4.3 Role-Specific Action Spaces (Backward Compatible)

- Keep universal catalog for API compatibility and persistence.
- Add role masks to policy selection layer:
  - `farmer`: higher priors for work/resource/social aid loops.
  - `guard`: safety, patrol, mediation, de-escalation.
  - `tavkeeper`: trade, social cohesion, rumor routing.
  - etc.
- Non-role actions remain resolvable but policy-deprioritized or masked at selection time.

## 4.4 Adaptive Personality Layer

Add adaptive coefficients on top of archetypes:

- Cooperation tendency.
- Risk aversion.
- Social sensitivity.
- Shock resilience.

These coefficients update from experienced outcomes (individual + community + shock context).

## 4.5 System Shocks

Shocks are explicit world modifiers with lifecycle:

- Trigger source: player choice, quest event, random narrative event.
- Scope: location-level or village-wide.
- Duration + decay profile.
- Effect channels: action cost/probability, reward scaling, resource availability, trust dynamics.

## 4.6 LLM Access Layer (API-First)

All LLM-dependent features (input parsing refinement, dialogue, narration enhancement, dynamic checkpoint generation) use a unified provider client:

- **Transport:** async HTTP calls from backend.
- **Providers:** self-hosted `llama.cpp` server, Ollama, or compatible external providers.
- **Abstraction:** one interface with provider-specific adapters.
- **Controls:** timeout, retries, rate limits, response validation, and automatic template fallback.
- **Policy:** if provider is unavailable, system degrades gracefully with no gameplay break.

Required adapter capabilities:

- Health/status check.
- Text/chat generation request.
- Temperature/max-token controls per use case.
- Standardized error mapping for fallback routing.

---

## 5) Data Contracts and Schema Evolution

## 5.1 API Stability Rules

- Existing endpoints remain available.
- Additive response fields only during transition.
- Breaking payload changes require versioned route or migration shim.

## 5.2 New Metrics Contract (Additive)

Extend metrics responses with:

- `individual_reward_series` (per NPC, per turn).
- `community_reward_series` (global per turn).
- `total_reward_series` (per NPC, per turn).
- `cooperation_index` (global + per role).
- `policy_entropy` (per NPC).
- `shock_timeline` (active shocks and magnitudes).
- `action_distribution` (per role and global).

## 5.3 Save/Load Schema Additions

Add fields without removing existing keys:

- `shock_state`.
- `npc_adaptation_state`.
- `reward_trace` (bounded rolling window).
- `schema_version` increment with backward loader.

## 5.4 Replay/Export Extensions

Enhance research exports to include:

- Turn-level reward decomposition.
- Shock context at decision time.
- Policy drift markers (phase changes).

## 5.5 Configuration Migration (LLM)

Introduce provider-oriented configuration in `backend/config.py`:

- `LLM_ENABLED`
- `LLM_API_BASE_URL`
- `LLM_API_KEY` (optional, env-sourced)
- `LLM_PROVIDER` (`llamacpp_server`, `ollama`, `openai_compatible`, etc.)
- `LLM_MODEL_NAME`
- Existing timeout/retry/rate-limit settings retained

Deprecate and remove local-runtime-only settings tied to in-process model loading.

---

## 6) Implementation Phases

## Phase 1 - Contract and Runtime Stabilization

### Goals

- Eliminate frontend/backend payload drift.
- Ensure deterministic and reproducible baseline behavior.
- Close known turn-processing gaps before introducing new RL complexity.
- Remove `llama-cpp-python` dependency path and switch to API-based LLM access.

### Work Items

- Align `/api/saves` and frontend modal consumption in `frontend/js/app.js` + `backend/api/routes.py`.
- Align `/api/load` payload (`filepath`) and client-side usage.
- Align metrics field names and dashboard mapping.
- Implement `queued_action` execution pipeline in `backend/engine/game_engine.py`.
- Normalize action catalog consumption from backend endpoint in frontend.
- Fix deterministic pretraining seed derivation (avoid process-randomized hash behavior).
- Replace in-process LLM service internals with provider-agnostic HTTP adapter in `backend/llm/llm_service.py`.
- Keep guardrails and fallback chain unchanged functionally while changing transport.
- Update dependency manifest to remove `llama-cpp-python`.

### Deliverables

- Contract matrix document.
- Passing integration smoke test for new game -> action -> save -> load -> continue.
- LLM provider adapter supporting at least one self-hosted endpoint (llama.cpp server or Ollama).

### Acceptance Criteria

- No schema mismatch errors in standard UI flow.
- Compound text actions execute as first action + one queued action next turn.
- Repeated same-seed runs produce stable pretraining summary metrics.
- Gameplay works with LLM disabled and with LLM enabled via API provider, without local binding dependencies.

---

## Phase 2 - Dual Reward Foundation

### Goals

- Introduce explicit individual/community reward decomposition.
- Preserve existing game behavior while enabling measurable RL experiments.

### Work Items

- Refactor reward function in `backend/npc/rl_agent.py`.
- Add community-state aggregator in engine layer.
- Introduce configurable `lambda` weights per archetype/role.
- Persist and expose reward components in metrics/export endpoints.

### Deliverables

- Reward decomposition logs and API fields.
- Validation plots for baseline runs (individual vs community reward traces).

### Acceptance Criteria

- Each NPC turn records `penalty`, `individual`, `community`, and `total` reward.
- Community term is non-null and responds to aggregate village state changes.

---

## Phase 3 - Adaptive Personality Dynamics

### Goals

- Make agents behaviorally plastic under changing world consequences.

### Work Items

- Add adaptation state object to `NPC` model (`backend/npc/npc.py`).
- Update decision logic with adaptation coefficients.
- Add bounded update rules and clamping for stability.
- Add per-role adaptation defaults in NPC data/archetypes.

### Deliverables

- Adaptation trace in metrics and replay data.
- Behavior-drift dashboard panel.

### Acceptance Criteria

- Coefficients evolve over time in non-trivial scenarios.
- Behavior shifts under repeated shocks are visible in action distribution.

---

## Phase 4 - Role-Specific Policy Masks (Backward Compatible)

### Goals

- Introduce role-grounded action spaces without breaking universal action compatibility.

### Work Items

- Add role-to-action-mask config in `backend/config.py` and archetype data.
- Enforce masks in action selection (`get_valid_actions` path).
- Keep parser/API universal; only policy selection is role constrained.
- Add debug telemetry: masked actions count + fallback behavior.

### Deliverables

- Role mask configuration set and documentation.
- Comparative experiments: universal-only vs role-masked.

### Acceptance Criteria

- No endpoint-breaking changes.
- NPC policy respects role masks while engine still resolves universal actions safely.

---

## Phase 5 - Dynamic Shock Engine

### Goals

- Create non-stationary environment pressures requiring adaptation.

### Work Items

- Extend `backend/engine/events.py` and event catalog schema with shock metadata.
- Add shock manager in engine: activation, propagation, decay, expiry.
- Apply shock modifiers into rewards, action probabilities, and resource channels.
- Add player-triggered shock hooks from action outcomes.

### Deliverables

- Shock timeline in state/metrics API.
- Scenario presets for repeatable experiments.

### Acceptance Criteria

- Shocks measurably alter policy and reward trajectories.
- Recovery behavior appears after shock decay.

---

## Phase 6 - Analytics, Curves, and Research Outputs

### Goals

- Produce interpretable metrics and plots aligned to project hypothesis.

### Work Items

- Extend `backend/engine/playthrough_logger.py` for structured RL telemetry.
- Add analytics endpoints for downloadable series.
- Add frontend charts/tables in `frontend/js/app.js` + dedicated UI module(s).
- Implement "individualism -> cooperation" progression curve computation.

### Required Plots

- Per-NPC: individual reward, community reward, total reward over turns.
- Village: social welfare index over turns.
- Action distribution shift (early vs late windows).
- Cooperation index vs episode/turn.
- Shock response and recovery curves.

### Acceptance Criteria

- One command/API flow can export complete experiment bundle.
- Curves clearly show phase shift in at least one reference scenario.
- Experiment metadata includes active LLM provider mode (disabled / llama.cpp server / Ollama / other).

---

## 7) API and Frontend Change Plan

## 7.1 Backend Endpoints

Maintain existing endpoints and add:

- `GET /api/metrics/timeseries` for turn-wise reward and welfare arrays.
- `GET /api/shocks/active` for current shock state.
- `POST /api/experiments/run` (optional Phase 7+) for scripted benchmark runs.

LLM status endpoint behavior:

- `GET /api/llm/status` reports provider connectivity and selected model, not local in-process model load state.

## 7.2 Frontend

- Replace hardcoded action assumptions with server action catalog source.
- Fix save/load integration with backend contract.
- Add metrics panels for reward decomposition and policy drift.
- Add shock timeline visualization.
- Update LLM indicator labels to provider-aware status (reachable/unreachable + provider name).

---

## 8) Testing and Validation Strategy

## 8.1 Automated Tests

- Unit tests: reward decomposition, mask logic, shock application, adaptation updates.
- Integration tests: full turn loop with save/load continuity.
- Regression tests: existing gameplay paths still function with RL playground flags off.
- LLM transport tests: provider adapter contract tests + fallback behavior on timeout/HTTP failures.

## 8.2 Determinism and Reproducibility

- Seed all random streams from `MASTER_SEED` with stable derivation.
- Log seed, config profile, scenario id, and LLM provider mode in each run artifact.

## 8.3 Performance Guardrails

- Keep per-turn processing budget within practical interactive limits.
- Avoid blocking calls in async routes.
- Keep JSON outputs bounded with configurable window sizes.
- Use strict request timeouts to prevent provider latency from stalling gameplay.

---

## 9) Risk Register and Mitigations

- **Contract drift risk** -> Phase 1 contract tests and endpoint fixtures.
- **Reward instability** -> bounded/clamped reward channels + staged rollout.
- **Behavior collapse to selfish/local optimum** -> adaptive `lambda` schedules + shock scenarios.
- **Overfitting to one scenario** -> scenario rotation and profile testing.
- **Provider instability / network failures** -> retry + timeout + immediate template fallback.
- **Dependency/build friction** -> remove `llama-cpp-python`; rely on external provider APIs.

---

## 10) Milestone Summary

- **M1:** Stable contracts + deterministic baseline + API-based LLM adapter.
- **M2:** Dual reward live and exported.
- **M3:** Adaptive personality live and observable.
- **M4:** Role masks live (backward compatible).
- **M5:** Shock engine live with measurable adaptation.
- **M6:** Full research analytics and progression curves.

---

## 11) Definition of Done (Project-Level)

Project is considered complete when:

- Core gameplay remains stable in single-session mode.
- RL Playground mode can be toggled on with no API breakage.
- NPC learning behavior demonstrates measurable shift from individualistic to community-aware strategies under at least one reference scenario.
- Exported artifacts are sufficient for reproducible analysis and plotting.
- LLM integration runs through provider APIs (self-hosted or external) with no required `llama-cpp-python` runtime dependency.
- Documentation and acceptance tests are updated for each phase.
