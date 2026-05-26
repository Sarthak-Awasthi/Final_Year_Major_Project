"""Generate all report figures (light mode, C1-C7) and save to report/."""
import os, sys, asyncio, logging

os.environ["LLM_ENABLED"] = "false"
logging.getLogger("mvp").setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({"font.size": 11})

import backend.config as cfg
from backend.engine.analytics import (
    compute_action_distribution,
    compute_cooperation_index,
    compute_cooperation_series,
    compute_reward_series,
    compute_social_welfare_series,
)
from backend.engine.game_engine import GameEngine
from backend.tools.playtest_bot import PlaytestBot

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "report")
MAX_TURNS = 500
SEED = 42

CONDITIONS = {
    "C1": {"label": "C1: Full System", "color": "#2ecc71", "env": {}},
    "C3": {"label": "C3: No RL (Schedules Only)", "color": "#e74c3c", "env": {"RL_ENABLED": False, "ROLE_MASK_ENABLED": False}},
    "C4": {"label": "C4: Flat MDP", "color": "#9b59b6", "env": {"HIERARCHICAL_MDP": False}},
    "C5": {"label": "C5: No Shocks", "color": "#3498db", "env": {"SHOCK_ENABLED": False}},
    "C6": {"label": "C6: No Role Masking", "color": "#f39c12", "env": {"ROLE_MASK_ENABLED": False}},
    "C7": {"label": "C7: Static Lambda", "color": "#e91e63", "env": {"DYNAMIC_LAMBDA": False}},
}


def _apply(env):
    prev = {}
    for k, v in env.items():
        prev[k] = getattr(cfg, k)
        setattr(cfg, k, v)
    return prev


def _restore(prev):
    for k, v in prev.items():
        setattr(cfg, k, v)


async def run_condition(cid):
    cond = CONDITIONS[cid]
    prev = _apply(cond["env"])
    engine = GameEngine(seed=SEED, difficulty="normal", max_turns=MAX_TURNS)
    await engine.initialize()
    bot = PlaytestBot(strategy="quest_focused", seed=SEED)
    shock_turn = MAX_TURNS // 3 if cfg.SHOCK_ENABLED else None

    while not engine.game_over and engine.turn < MAX_TURNS:
        if shock_turn and engine.turn == shock_turn:
            engine.shock_manager.activate_shock("famine", engine.turn)
        state = engine.get_full_state()
        pi = bot._select_action(state, bot.strategy)
        await engine.process_turn(pi)

    coop_series = compute_cooperation_series(engine.npc_registry)
    welfare_series = compute_social_welfare_series(engine.npc_registry)
    reward_series = compute_reward_series(engine.npc_registry)
    action_dist = compute_action_distribution(engine.event_log.entries, engine.npc_registry)
    final_coop = compute_cooperation_index(engine.npc_registry)

    npc_states = {}
    for uid, npc in engine.npc_registry.items():
        npc_states[uid] = {
            "name": npc.name, "role": npc.archetype, "lambda": npc.lambda_coeff,
            **npc.adaptation_state,
        }

    all_ind, all_comm, all_total = [], [], []
    for uid, rs in reward_series.items():
        for i in range(len(rs["turns"])):
            while len(all_ind) <= i:
                all_ind.append([])
                all_comm.append([])
                all_total.append([])
            all_ind[i].append(rs["individual"][i])
            all_comm[i].append(rs["community"][i])
            all_total[i].append(rs["total"][i])

    _restore(prev)
    return {
        "cid": cid, "label": cond["label"], "color": cond["color"],
        "coop_turns": coop_series.get("turns", []),
        "coop_values": coop_series.get("global_cooperation", []),
        "welfare_turns": welfare_series.get("turns", []),
        "welfare_values": welfare_series.get("welfare_index", []),
        "reward_turns": list(range(len(all_ind))),
        "avg_individual_reward": [sum(v) / len(v) for v in all_ind] if all_ind else [],
        "avg_community_reward": [sum(v) / len(v) for v in all_comm] if all_comm else [],
        "avg_total_reward": [sum(v) / len(v) for v in all_total] if all_total else [],
        "action_dist": action_dist,
        "final_coop": final_coop,
        "npc_states": npc_states,
        "shock_turn": shock_turn,
    }


def save(fig, name):
    fig.savefig(os.path.join(REPORT_DIR, f"{name}.png"), dpi=200)
    fig.savefig(os.path.join(REPORT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}")


async def main():
    results = {}
    for cid in CONDITIONS:
        print(f"  Running {CONDITIONS[cid]['label']}...", end="", flush=True)
        r = await run_condition(cid)
        results[cid] = r
        print(f" K={r['final_coop']['global']:.3f}")

    # 1 — Cooperation curves
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in results.values():
        ax.plot(r["coop_turns"], r["coop_values"], color=r["color"], linewidth=2.5, label=r["label"], alpha=0.9)
    st = results["C1"].get("shock_turn")
    if st:
        ax.axvline(x=st, color="#e74c3c", linestyle="--", alpha=0.5, label="Famine shock")
        ax.axvspan(st, st + 15, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Turn"); ax.set_ylabel("Cooperation Index K(t)")
    ax.set_title("Cooperation Emergence: Ablation Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10); ax.set_ylim(-0.05, 1.05); ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); fig.tight_layout()
    save(fig, "fig_cooperation")

    # 2 — Final bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    cids = list(results.keys())
    labels = [results[c]["label"] for c in cids]
    final_coops = [results[c]["final_coop"]["global"] for c in cids]
    colors = [results[c]["color"] for c in cids]
    bars = ax.bar(labels, final_coops, color=colors, alpha=0.85, width=0.5, edgecolor="#333", linewidth=0.5)
    for bar, val in zip(bars, final_coops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Cooperation Index")
    ax.set_title("Final Cooperation: Ablation Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.2); ax.grid(True, axis="y"); fig.tight_layout()
    save(fig, "fig_final_bar")

    # 3 — Reward decomposition (C1)
    c1 = results["C1"]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(c1["reward_turns"], c1["avg_individual_reward"], color="#2ecc71", linewidth=2, label="Individual G(t)", alpha=0.8)
    ax.plot(c1["reward_turns"], c1["avg_community_reward"], color="#3498db", linewidth=2, label="Community C(t)", alpha=0.8)
    ax.plot(c1["reward_turns"], c1["avg_total_reward"], color="#f1c40f", linewidth=2.5, label="Total R(t)", alpha=0.9)
    ax.axhline(y=0, color="#888", linewidth=0.5)
    if c1.get("shock_turn"):
        ax.axvline(x=c1["shock_turn"], color="#e74c3c", linestyle="--", alpha=0.5, label="Famine shock")
    ax.set_xlabel("Turn"); ax.set_ylabel("Average Reward")
    ax.set_title("Reward Decomposition (Full System C1)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10); ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); fig.tight_layout()
    save(fig, "fig_reward")

    # 4 — NPC adaptation (C1 vs C3)
    coefficients = ["cooperation_tendency", "risk_aversion", "social_sensitivity", "shock_resilience"]
    coeff_labels = ["Cooperation", "Risk Aversion", "Social Sens.", "Shock Resil."]
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax_idx, cid in enumerate(["C1", "C3"]):
        r = results[cid]; ax = axes[ax_idx]
        npc_names = [s["name"] for s in r["npc_states"].values()]
        x = range(len(npc_names)); width = 0.18
        for i, (coeff, clabel) in enumerate(zip(coefficients, coeff_labels)):
            vals = [s[coeff] for s in r["npc_states"].values()]
            ax.bar([xi + i * width for xi in x], vals, width, label=clabel, color=bar_colors[i], alpha=0.85)
        ax.set_xticks([xi + 1.5 * width for xi in x])
        ax.set_xticklabels(npc_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient Value")
        ax.set_title(r["label"], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1); ax.legend(fontsize=8, loc="upper right"); ax.grid(True, axis="y")
    fig.suptitle("NPC Adaptation: Full System vs No RL", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_npc_adaptation")

    # 5 — Action distribution shift
    shift = results["C1"]["action_dist"].get("distribution_shift", {})
    sorted_actions = sorted(shift.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    if sorted_actions:
        fig, ax = plt.subplots(figsize=(14, 7))
        actions = [a[0] for a in sorted_actions]
        values = [a[1] for a in sorted_actions]
        colors_s = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
        ax.barh(actions, values, color=colors_s, alpha=0.85, height=0.6)
        ax.axvline(x=0, color="#888", linewidth=0.8)
        ax.set_xlabel("Proportion Change (Late - Early)")
        ax.set_title("NPC Action Distribution Shift: Early vs Late Game (C1)", fontsize=14, fontweight="bold")
        ax.grid(True, axis="x"); fig.tight_layout()
        save(fig, "fig_action_shift")

    # 6 — Social welfare
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in results.values():
        ax.plot(r["welfare_turns"], r["welfare_values"], color=r["color"], linewidth=2.5, label=r["label"], alpha=0.9)
    ax.set_xlabel("Turn"); ax.set_ylabel("Social Welfare Index")
    ax.set_title("Village Social Welfare Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10); ax.set_ylim(-0.05, 1.05); ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); fig.tight_layout()
    save(fig, "fig_welfare")

    print("\nAll 6 figures saved to report/")


if __name__ == "__main__":
    asyncio.run(main())
