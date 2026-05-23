"""
run_experiment.py — Automated ablation study runner.

Runs 7 experimental conditions × N playthroughs each, collects experiment
bundles, and produces a summary report. Fully automated — no human input.

Usage:
    uv run python -m backend.tools.run_experiment
    uv run python -m backend.tools.run_experiment --runs 50 --turns 100
    uv run python -m backend.tools.run_experiment --conditions C1,C2,C5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import backend.config as cfg
from backend.engine.analytics import build_experiment_bundle
from backend.tools.playtest_bot import PlaytestBot, VALID_STRATEGIES

RESULTS_DIR = cfg.DATA_DIR / "experiments"

# ── Ablation Conditions ─────────────────────────────────────────────────────

CONDITIONS: dict[str, dict[str, Any]] = {
    "C1": {
        "name": "Full System",
        "description": "All subsystems enabled (baseline)",
        "env": {},
    },
    "C2": {
        "name": "No LLM",
        "description": "LLM disabled — template narration only",
        "env": {"LLM_ENABLED": False},
    },
    "C3": {
        "name": "No RL (Schedules Only)",
        "description": "Q-learning disabled — NPCs use deterministic schedules",
        "env": {"RL_ENABLED": False, "ROLE_MASK_ENABLED": False},
    },
    "C4": {
        "name": "Flat MDP",
        "description": "No hierarchical quest stages — single flat stage",
        "env": {"HIERARCHICAL_MDP": False},
    },
    "C5": {
        "name": "No Shocks",
        "description": "System shock engine disabled — stationary environment",
        "env": {"SHOCK_ENABLED": False},
    },
    "C6": {
        "name": "No Role Masking",
        "description": "Role-specific action masking disabled",
        "env": {"ROLE_MASK_ENABLED": False},
    },
    "C7": {
        "name": "Static Lambda",
        "description": "Dynamic λ disabled — fixed λ=0.3 for all NPCs",
        "env": {"DYNAMIC_LAMBDA": False},
    },
}

STRATEGIES_CYCLE = ["quest_focused", "aggressive", "social", "explorer", "random"]


def _apply_condition(condition_id: str) -> dict[str, Any]:
    """Set config flags for a given condition. Returns previous values."""
    cond = CONDITIONS[condition_id]
    prev: dict[str, Any] = {}
    for key, val in cond["env"].items():
        prev[key] = getattr(cfg, key)
        setattr(cfg, key, val)
    return prev


def _restore_condition(prev: dict[str, Any]) -> None:
    """Restore config flags to their previous values."""
    for key, val in prev.items():
        setattr(cfg, key, val)


async def run_single(
    condition_id: str,
    run_index: int,
    max_turns: int,
    base_seed: int,
) -> dict[str, Any]:
    """Run a single playthrough and return its experiment bundle."""
    seed = base_seed + run_index
    strategy = STRATEGIES_CYCLE[run_index % len(STRATEGIES_CYCLE)]

    bot = PlaytestBot(strategy=strategy, seed=seed, difficulty="normal")
    summary = await bot.run(max_turns=max_turns)

    engine = bot.engine
    experiment_bundle = build_experiment_bundle(engine, engine.event_log.entries)

    bundle = {
        "condition": condition_id,
        "condition_name": CONDITIONS[condition_id]["name"],
        "run_index": run_index,
        "seed": seed,
        "strategy": strategy,
        "total_turns": summary["total_turns"],
        "game_over": summary["game_over"],
        "game_result": summary["game_result"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "metrics": summary.get("metrics", {}),
        "final_player": summary.get("final_player", {}),
        "final_quest": summary.get("final_quest", {}),
        "cooperation_index": experiment_bundle.get("cooperation_index", {}),
        "cooperation_series": experiment_bundle.get("cooperation_series", {}),
        "social_welfare_series": experiment_bundle.get("social_welfare_series", {}),
        "action_distribution": experiment_bundle.get("action_distribution", {}),
        "shock_responses": experiment_bundle.get("shock_responses", []),
        "adaptation_snapshot": experiment_bundle.get("adaptation_snapshot", {}),
        "policy_entropy": experiment_bundle.get("policy_entropy", {}),
    }
    return bundle


async def run_condition(
    condition_id: str,
    num_runs: int,
    max_turns: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Run all playthroughs for one condition."""
    cond = CONDITIONS[condition_id]
    print(f"\n{'='*60}")
    print(f"  {condition_id}: {cond['name']}")
    print(f"  {cond['description']}")
    print(f"  Runs: {num_runs} | Max turns: {max_turns}")
    print(f"{'='*60}")

    prev = _apply_condition(condition_id)
    results: list[dict[str, Any]] = []

    for i in range(num_runs):
        t0 = time.monotonic()
        try:
            bundle = await run_single(condition_id, i, max_turns, base_seed)
            elapsed = time.monotonic() - t0
            results.append(bundle)
            status = bundle.get("game_result", "unknown")
            print(
                f"  [{condition_id}] Run {i+1:3d}/{num_runs} "
                f"| {bundle['strategy']:14s} "
                f"| turns={bundle['total_turns']:3d} "
                f"| result={status:10s} "
                f"| {elapsed:.1f}s"
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"  [{condition_id}] Run {i+1:3d}/{num_runs} | ERROR: {exc} | {elapsed:.1f}s")
            results.append({
                "condition": condition_id,
                "run_index": i,
                "error": str(exc),
            })

    _restore_condition(prev)
    return results


def _compute_condition_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate stats for a condition's results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"runs": len(results), "errors": len(results), "valid": 0}

    total_turns = [r["total_turns"] for r in valid]
    completion = sum(1 for r in valid if r.get("game_result") == "success")
    turn_limits = sum(1 for r in valid if r.get("game_result") == "turn_limit")
    failures = sum(1 for r in valid if r.get("game_result") == "fail")
    elapsed = [r["elapsed_seconds"] for r in valid]

    return {
        "runs": len(results),
        "valid": len(valid),
        "errors": len(results) - len(valid),
        "quest_completion_rate": round(completion / len(valid), 4) if valid else 0,
        "turn_limit_rate": round(turn_limits / len(valid), 4) if valid else 0,
        "failure_rate": round(failures / len(valid), 4) if valid else 0,
        "avg_turns": round(sum(total_turns) / len(total_turns), 1),
        "min_turns": min(total_turns),
        "max_turns": max(total_turns),
        "avg_time_seconds": round(sum(elapsed) / len(elapsed), 2),
        "total_time_seconds": round(sum(elapsed), 2),
    }


def _format_report(
    all_results: dict[str, list[dict]],
    summaries: dict[str, dict],
    total_elapsed: float,
) -> str:
    """Generate a plain-text summary report."""
    lines = [
        "=" * 70,
        "  ABLATION STUDY RESULTS",
        f"  Generated: {datetime.now(timezone.utc).isoformat()}",
        f"  Total runtime: {total_elapsed:.1f}s",
        "=" * 70,
        "",
    ]

    for cid in sorted(summaries.keys()):
        cond = CONDITIONS[cid]
        s = summaries[cid]
        lines.append(f"  {cid}: {cond['name']}")
        lines.append(f"  {'-'*50}")

        if s.get("valid", 0) == 0:
            lines.append(f"    All {s['runs']} runs failed.")
        else:
            lines.append(f"    Runs: {s['valid']}/{s['runs']} valid ({s.get('errors',0)} errors)")
            lines.append(f"    Quest completion: {s.get('quest_completion_rate', 0)*100:.1f}%")
            lines.append(f"    Turn limit exits: {s.get('turn_limit_rate', 0)*100:.1f}%")
            lines.append(f"    Deaths/failures:  {s.get('failure_rate', 0)*100:.1f}%")
            lines.append(f"    Turns: avg={s.get('avg_turns',0):.1f}, "
                         f"min={s.get('min_turns',0)}, max={s.get('max_turns',0)}")
            lines.append(f"    Time: {s.get('total_time_seconds',0):.1f}s total, "
                         f"{s.get('avg_time_seconds',0):.2f}s/run")
        lines.append("")

    # Comparison table
    lines.append("  COMPARISON TABLE")
    lines.append(f"  {'Cond':<6} {'Name':<25} {'Compl%':>7} {'AvgTurns':>9} {'Deaths%':>8}")
    lines.append(f"  {'-'*6} {'-'*25} {'-'*7} {'-'*9} {'-'*8}")
    for cid in sorted(summaries.keys()):
        s = summaries[cid]
        name = CONDITIONS[cid]["name"][:25]
        comp = f"{s.get('quest_completion_rate', 0)*100:.1f}" if s.get("valid") else "N/A"
        turns = f"{s.get('avg_turns', 0):.1f}" if s.get("valid") else "N/A"
        deaths = f"{s.get('failure_rate', 0)*100:.1f}" if s.get("valid") else "N/A"
        lines.append(f"  {cid:<6} {name:<25} {comp:>7} {turns:>9} {deaths:>8}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated ablation study")
    parser.add_argument("--runs", type=int, default=20, help="Playthroughs per condition (default: 20)")
    parser.add_argument("--turns", type=int, default=200, help="Max turns per playthrough (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    parser.add_argument(
        "--conditions", type=str, default=None,
        help="Comma-separated condition IDs to run (default: all). Example: C1,C2,C5",
    )
    args = parser.parse_args()

    if args.conditions:
        condition_ids = [c.strip().upper() for c in args.conditions.split(",")]
        invalid = [c for c in condition_ids if c not in CONDITIONS]
        if invalid:
            parser.error(f"Unknown conditions: {invalid}. Valid: {list(CONDITIONS.keys())}")
    else:
        condition_ids = list(CONDITIONS.keys())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_dir = RESULTS_DIR / f"ablation_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAblation Study: {len(condition_ids)} conditions × {args.runs} runs × {args.turns} max turns")
    print(f"Output: {experiment_dir}\n")

    all_results: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    t_start = time.monotonic()

    for cid in condition_ids:
        results = await run_condition(cid, args.runs, args.turns, args.seed)
        all_results[cid] = results
        summaries[cid] = _compute_condition_summary(results)

        # Save per-condition results
        cond_path = experiment_dir / f"{cid}_results.json"
        cond_path.write_text(
            json.dumps(results, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    total_elapsed = time.monotonic() - t_start

    # Save summaries
    summary_path = experiment_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Generate report
    report = _format_report(all_results, summaries, total_elapsed)
    report_path = experiment_dir / "REPORT.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    asyncio.run(main())
