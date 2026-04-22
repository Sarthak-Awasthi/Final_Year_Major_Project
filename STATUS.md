# Implementation Status

**Updated:** 2026-04-22 | **Tests:** 91/91 ✅

## Phase Completion

| # | Phase | Status | Key Deliverables |
|---|-------|--------|-----------------|
| 1 | Contract Stabilization | ✅ | Save/load alignment, queued actions, snake_case metrics, deterministic seeding |
| 2 | Dual Reward Foundation | ✅ | 3-component reward (penalty+individual+community), community aggregator, reward tracing |
| 3 | Adaptive Personality | ✅ | Per-NPC adaptation state (cooperation, risk_aversion, social_sensitivity, shock_resilience), reward-driven updates |
| 4 | Role-Specific Masks | ✅ | Soft-mask Q-value adjustment for 6 archetypes, role telemetry & coherence tracking |
| — | LLM HTTP Migration | ✅ | httpx adapter, provider abstraction (ollama/llamacpp_server/openai_compatible), llama-cpp-python removed |
| 5 | Dynamic Shock Engine | ✅ | ShockManager with 5 shock types, linear/sudden decay, reward/adaptation modifiers, API endpoints |
| 6 | Analytics & Curves | ✅ | Chart.js research dashboard, timeseries/cooperation/experiment API endpoints, RL telemetry logging, experiment bundle export |

## Config Flags

| Flag | Default | Effect |
|------|---------|--------|
| `SHOCK_ENABLED` | `True` | Enables Dynamic Shock Engine |
| `SHOCK_MAX_ACTIVE` | `3` | Maximum concurrent active shocks |
| `ROLE_MASK_ENABLED` | `False` | Enables role-specific soft-masking in action selection |
| `LLM_ENABLED` | `True` | Enables LLM provider calls (falls back to templates if unavailable) |
| `LLM_PROVIDER` | `ollama` | Provider type: `ollama`, `llamacpp_server`, `openai_compatible` |

## Shock Types (Phase 5)

| Type | Duration | Decay | Reward Scale | Notes |
|------|----------|-------|-------------|-------|
| `famine` | 15 | linear | 0.5× | Stat drain, trust drop |
| `bandit_raid` | 8 | sudden | 0.6× | High drain, full intensity until expiry |
| `plague` | 12 | linear | 0.4× | Heavy drain, moderate trust impact |
| `trade_boom` | 10 | linear | 1.5× | Positive: stat gain, trust boost |
| `harsh_winter` | 20 | linear | 0.7× | Long duration, moderate effects |

## Analytics Endpoints (Phase 6)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics/timeseries` | GET | Per-NPC reward curves, welfare index, cooperation index, entropy, action distribution, shock responses |
| `/api/metrics/cooperation` | GET | Lightweight cooperation snapshot (global, per-role, per-NPC) |
| `/api/metrics/experiment` | GET | Complete experiment bundle for offline analysis (all series + metadata) |

## Tests

```bash
uv run pytest backend/tests/ -v
```

- `test_integration_smoke.py` — 7 tests (new game → save → load, response shapes, metrics)
- `test_seed_reproducibility.py` — 14 tests (deterministic pretraining, Q-table convergence)
- `test_step3_step4_features.py` — 20 tests (adaptation state, role masks, telemetry, persistence)
- `test_shock_engine.py` — 23 tests (activation, decay, effects, serialization, game integration)
- `test_analytics.py` — 27 tests (reward series, cooperation index, welfare, entropy, action distribution, shock response, experiment bundle, RL telemetry, API endpoints)
