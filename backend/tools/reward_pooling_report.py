"""
reward_pooling_report.py — Generate reward decomposition metrics for NPCs.

Runs a single playthrough and generates detailed reward breakdown:
- Individual vs Community reward evolution
- Cooperation tendency growth
- Lambda coefficient progression
- Reward traces per NPC
- Matplotlib graphs for visualization

NOTE: LLM is always disabled for this analysis (focuses on RL reward dynamics only).

Usage:
    uv run python -m backend.tools.reward_pooling_report --turns 500 --seed 42 --alpha 1.0
    uv run python -m backend.tools.reward_pooling_report --turns 300 --seed 123 --alpha 0.8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
from datetime import datetime, timezone

import backend.config as cfg
from backend.tools.playtest_bot import PlaytestBot

# Output directory
METRICS_DIR = cfg.METRICS_DIR
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Skipping graph generation.")
    print("Install with: pip install matplotlib")


def _generate_graphs(metrics_per_turn: list[dict], seed: int) -> None:
    """Generate matplotlib graphs from metrics data.
    
    Creates:
    - Reward decomposition over time (individual, community, total)
    - Cooperation and Lambda evolution
    - Phase analysis summary
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    turns = [m["turn"] for m in metrics_per_turn]
    individual_sum = [m["individual_sum"] for m in metrics_per_turn]
    community_sum = [m["community_sum"] for m in metrics_per_turn]
    total_sum = [m["total_sum"] for m in metrics_per_turn]
    cooperation_avg = [m["cooperation_avg"] for m in metrics_per_turn]
    lambda_avg = [m["lambda_avg"] for m in metrics_per_turn]
    
    # Figure 1: Reward Decomposition
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(turns, individual_sum, label="Individual Reward", linewidth=2, color="#4472C4")
    ax.plot(turns, community_sum, label="Community Reward (λ-weighted)", linewidth=2, color="#70AD47")
    ax.plot(turns, total_sum, label="Total Reward", linewidth=2, color="#FFC000", linestyle="--")
    
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Summed Reward (All NPCs)", fontsize=12)
    ax.set_title(f"Reward Decomposition Over Time (Seed={seed})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    graph_path = METRICS_DIR / f"reward_decomposition_seed{seed}.png"
    plt.savefig(graph_path, dpi=150)
    print(f"✓ Graph 1 saved: {graph_path}")
    plt.close()
    
    # Figure 2: Cooperation and Lambda Evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(turns, cooperation_avg, label="Cooperation Tendency", linewidth=2, color="#E7298A")
    ax1.axhline(y=0.5, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Mid-point (0.5)")
    ax1.fill_between(turns, 0, cooperation_avg, alpha=0.2, color="#E7298A")
    ax1.set_ylabel("Cooperation Tendency", fontsize=12)
    ax1.set_title(f"NPC Adaptation Over Time (Seed={seed})", fontsize=14, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(turns, lambda_avg, label="λ (Community Weight)", linewidth=2, color="#66C2A5")
    ax2.axhline(y=0.05, color="blue", linestyle=":", linewidth=1, alpha=0.5, label="Min (0.05)")
    ax2.axhline(y=0.60, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Max (0.60)")
    ax2.fill_between(turns, 0.05, lambda_avg, alpha=0.2, color="#66C2A5")
    ax2.set_xlabel("Turn", fontsize=12)
    ax2.set_ylabel("Lambda Coefficient", fontsize=12)
    ax2.set_ylim([0, 0.7])
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    graph_path = METRICS_DIR / f"adaptation_evolution_seed{seed}.png"
    plt.savefig(graph_path, dpi=150)
    print(f"✓ Graph 2 saved: {graph_path}")
    plt.close()
    
    # Figure 3: Reward Ratio (Individual % vs Community %)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ind_pct = []
    comm_pct = []
    for m in metrics_per_turn:
        tot = m["total_sum"]
        if tot > 0:
            ind_pct.append((m["individual_sum"] / tot) * 100)
            comm_pct.append((m["community_sum"] / tot) * 100)
        else:
            ind_pct.append(0)
            comm_pct.append(0)
    
    ax.fill_between(turns, 0, ind_pct, label="Individual %", alpha=0.6, color="#4472C4")
    ax.fill_between(turns, ind_pct, 100, label="Community %", alpha=0.6, color="#70AD47")
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Reward Percentage", fontsize=12)
    ax.set_title(f"Individual vs Community Reward Ratio (Seed={seed})", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 100])
    ax.legend(loc="center right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    graph_path = METRICS_DIR / f"reward_ratio_seed{seed}.png"
    plt.savefig(graph_path, dpi=150)
    print(f"✓ Graph 3 saved: {graph_path}\n")
    plt.close()


def aggregate_npc_rewards(game_engine) -> dict:
    """Aggregate reward metrics across all NPCs at current state.
    
    The community reward stored in reward_trace is the *raw* (unweighted)
    value.  The actual contribution to total is ``lambda * community``.
    We derive the weighted community contribution as:
        weighted_community = total - penalty - individual
    so that: individual + weighted_community + penalty == total.
    
    Returns:
        Dict with keys:
        - individual_sum: Sum of all NPCs' individual rewards this turn
        - community_sum: Sum of all NPCs' λ-weighted community rewards
        - total_sum: Sum of all NPCs' total rewards this turn
        - cooperation_rates: Per-NPC cooperation tendency [0, 1]
        - lambda_coeffs: Per-NPC lambda coefficients
    """
    npcs = list(game_engine.npc_registry.values())
    
    # Most recent reward sample per NPC
    individual_sum = 0.0
    community_sum = 0.0
    total_sum = 0.0
    cooperation_rates = {}
    lambda_coeffs = {}
    
    for npc in npcs:
        if npc.reward_trace:
            latest = npc.reward_trace[-1]
            ind = latest.get("individual", 0.0)
            penalty = latest.get("penalty", 0.0)
            total = latest.get("total", 0.0)
            # Derive the λ-weighted community contribution
            weighted_comm = total - penalty - ind
            individual_sum += ind
            community_sum += weighted_comm
            total_sum += total
        
        cooperation_rates[npc.npc_uid] = npc.adaptation_state.get("cooperation_tendency", 0.0)
        lambda_coeffs[npc.npc_uid] = npc.lambda_coeff
    
    return {
        "individual_sum": individual_sum,
        "community_sum": community_sum,
        "total_sum": total_sum,
        "cooperation_rates": cooperation_rates,
        "lambda_coeffs": lambda_coeffs,
    }


async def run_reward_pooling_report(
    turns: int = 500,
    seed: int = 42,
    alpha: float = 1.0,
    llm_enabled: bool = False,
) -> None:
    """Run a playthrough and generate reward decomposition report.
    
    Args:
        turns: Number of turns to simulate
        seed: Random seed
        alpha: Difficulty alpha (1.0 = normal)
        llm_enabled: Whether to enable LLM (default False for this analysis)
    """
    print(f"\n{'='*70}")
    print(f"REWARD POOLING ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"Turns: {turns}")
    print(f"Seed: {seed}")
    print(f"Difficulty Alpha: {alpha}")
    print(f"LLM: DISABLED (analysis focuses on RL reward dynamics)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*70}\n")
    
    # Configure: Disable LLM for analysis (test focuses on RL reward dynamics)
    original_llm = cfg.LLM_ENABLED
    cfg.LLM_ENABLED = False  # Always disabled for this analysis
    
    try:
        # Run playthrough
        bot = PlaytestBot(strategy="social", seed=seed, difficulty="normal")
        summary = await bot.run(max_turns=turns)
        engine = bot.engine
        
        print(f"Playthrough completed. Processing metrics...\n")
        
        # Collect metrics from reward traces per NPC
        metrics_per_turn = []
        
        # Get max turn number to iterate through
        max_turn = engine.turn
        
        # Iterate through turns and collect reward data
        for turn_num in range(1, max_turn + 1):
            individual_sum = 0.0
            community_sum = 0.0
            total_sum = 0.0
            cooperation_rates = {}
            lambda_coeffs = {}
            
            # Aggregate from all NPCs' reward traces
            for npc in engine.npc_registry.values():
                # Find reward entry for this turn
                for reward_entry in npc.reward_trace:
                    if reward_entry.get("turn") == turn_num:
                        ind = reward_entry.get("individual", 0.0)
                        penalty = reward_entry.get("penalty", 0.0)
                        total = reward_entry.get("total", 0.0)
                        # Derive λ-weighted community: total = penalty + ind + λ*comm
                        weighted_comm = total - penalty - ind
                        individual_sum += ind
                        community_sum += weighted_comm
                        total_sum += total
                        break
                
                # Get cooperation/lambda from adaptation trace
                for adapt_entry in npc.adaptation_trace:
                    if adapt_entry.get("turn") == turn_num:
                        cooperation_rates[npc.npc_uid] = adapt_entry.get("cooperation_tendency", 0.0)
                        lambda_coeffs[npc.npc_uid] = adapt_entry.get("lambda_coeff", 0.05)
                        break
            
            # Compute averages
            num_npcs = len(engine.npc_registry) if engine.npc_registry else 1
            coop_avg = sum(cooperation_rates.values()) / len(cooperation_rates) if cooperation_rates else 0.0
            lambda_avg = sum(lambda_coeffs.values()) / len(lambda_coeffs) if lambda_coeffs else 0.05
            
            metrics_per_turn.append({
                "turn": turn_num,
                "individual_sum": individual_sum,
                "community_sum": community_sum,
                "total_sum": total_sum,
                "cooperation_avg": coop_avg,
                "lambda_avg": lambda_avg,
            })
            
            # Progress indicator
            if turn_num % 50 == 0:
                latest = metrics_per_turn[-1]
                print(f"Turn {turn_num:3d}: "
                      f"Individual {latest['individual_sum']:7.3f} | "
                      f"Community {latest['community_sum']:7.3f} | "
                      f"Total {latest['total_sum']:7.3f} | "
                      f"Coop {latest['cooperation_avg']:.2f} | "
                      f"λ {latest['lambda_avg']:.3f}")
        
        print(f"\nCompleted {turns} turns.\n")
        
        # Write CSV
        csv_path = METRICS_DIR / "reward_pooling_summary.csv"
        csv_path.write_text("")  # Clear any existing file
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["turn", "individual_sum", "community_sum", "total_sum", 
                           "cooperation_avg", "lambda_avg"]
            )
            writer.writeheader()
            writer.writerows(metrics_per_turn)
        
        print(f"✓ CSV written to: {csv_path}\n")
        
        # Generate graphs if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            print("Generating graphs...\n")
            _generate_graphs(metrics_per_turn, seed)
        
        # Generate summary statistics
        early_turns = [m for m in metrics_per_turn if m["turn"] <= 50]
        mid_turns = [m for m in metrics_per_turn if 150 <= m["turn"] <= 250]
        late_turns = [m for m in metrics_per_turn if m["turn"] >= 400]
        
        def avg_dict(turns_list, key):
            if not turns_list:
                return 0.0
            return sum(m[key] for m in turns_list) / len(turns_list)
        
        def pct(value, total):
            if total == 0:
                return 0.0
            return (value / total) * 100 if total > 0 else 0.0
        
        print(f"\n{'='*70}")
        print(f"PHASE ANALYSIS")
        print(f"{'='*70}\n")
        
        phases = [
            ("EARLY (1-50)", early_turns),
            ("MID (150-250)", mid_turns),
            ("LATE (400+)", late_turns),
        ]
        
        for phase_name, phase_data in phases:
            if not phase_data:
                print(f"{phase_name}: [No data]")
                continue
            
            ind = avg_dict(phase_data, "individual_sum")
            comm = avg_dict(phase_data, "community_sum")
            tot = avg_dict(phase_data, "total_sum")
            coop = avg_dict(phase_data, "cooperation_avg")
            lam = avg_dict(phase_data, "lambda_avg")
            
            ind_pct = pct(ind, tot)
            comm_pct = pct(comm, tot)
            
            print(f"{phase_name}:")
            print(f"  Individual Reward: {ind:7.3f} ({ind_pct:5.1f}%)")
            print(f"  Community Reward:  {comm:7.3f} ({comm_pct:5.1f}%)")
            print(f"  Total Reward:      {tot:7.3f}")
            print(f"  Cooperation:       {coop:7.3f}")
            print(f"  Lambda (λ):        {lam:7.3f}")
            print()
        
        # Validation checklist
        print(f"{'='*70}")
        print(f"VALIDATION CHECKLIST")
        print(f"{'='*70}\n")
        
        checks = []
        
        # Check 1: Early individual dominance
        if early_turns:
            early_ind = avg_dict(early_turns, "individual_sum")
            early_tot = avg_dict(early_turns, "total_sum")
            if early_tot > 0:
                early_ind_pct = pct(early_ind, early_tot)
                check1 = early_ind_pct >= 70.0
                checks.append(check1)
                symbol = "✓" if check1 else "✗"
                print(f"{symbol} Early Individual % >= 70%: {early_ind_pct:.1f}% {'[PASS]' if check1 else '[FAIL]'}")
        
        # Check 2: Phase transition (mid > early)
        if early_turns and mid_turns:
            early_comm = avg_dict(early_turns, "community_sum")
            mid_comm = avg_dict(mid_turns, "community_sum")
            check2 = mid_comm > early_comm
            checks.append(check2)
            symbol = "✓" if check2 else "✗"
            print(f"{symbol} Mid Community > Early Community: {mid_comm:.3f} > {early_comm:.3f} {'[PASS]' if check2 else '[FAIL]'}")
        
        # Check 3: Late cooperation
        if late_turns:
            late_coop = avg_dict(late_turns, "cooperation_avg")
            check3 = late_coop >= 0.6
            checks.append(check3)
            symbol = "✓" if check3 else "✗"
            print(f"{symbol} Late Cooperation >= 0.6: {late_coop:.3f} {'[PASS]' if check3 else '[FAIL]'}")
        
        # Check 4: Lambda growth
        if early_turns and late_turns:
            early_lam = avg_dict(early_turns, "lambda_avg")
            late_lam = avg_dict(late_turns, "lambda_avg")
            check4 = late_lam > early_lam
            checks.append(check4)
            symbol = "✓" if check4 else "✗"
            print(f"{symbol} Late Lambda > Early Lambda: {late_lam:.3f} > {early_lam:.3f} {'[PASS]' if check4 else '[FAIL]'}")
        
        passed = sum(checks)
        total = len(checks)
        print(f"\n{'='*70}")
        print(f"RESULT: {passed}/{total} checks passed")
        print(f"{'='*70}\n")
        
        if passed == total:
            print("✓ All checks passed! Reward system behaving as designed.")
        else:
            print(f"⚠ {total - passed} check(s) failed. Review results above.")
    
    finally:
        # Restore config
        cfg.LLM_ENABLED = original_llm


def main() -> None:
    """Parse arguments and run report."""
    parser = argparse.ArgumentParser(
        description="Generate reward decomposition report for NPCs."
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=500,
        help="Number of turns to simulate (default: 500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Difficulty alpha coefficient (default: 1.0)"
    )
    # Removed --llm-enabled flag since LLM is always disabled for this analysis
    
    args = parser.parse_args()
    
    asyncio.run(run_reward_pooling_report(
        turns=args.turns,
        seed=args.seed,
        alpha=args.alpha,
        llm_enabled=False,  # Always disabled for this analysis
    ))


if __name__ == "__main__":
    main()
