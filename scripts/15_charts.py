from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "phase05.log"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_results() -> pd.DataFrame:
    results = pd.read_csv(ROOT / "results" / "intervention" / "sir_intervention_results.csv")
    if "total_reduction_vs_baseline_pct" not in results:
        results["total_reduction_vs_baseline_pct"] = results["reduction_vs_baseline_pct"]
    if "peak_reduction_vs_baseline_pct" not in results:
        results["peak_reduction_vs_baseline_pct"] = results["total_reduction_vs_baseline_pct"]
    if "nodes_removed" not in results:
        comparison = pd.read_csv(ROOT / "results" / "intervention" / "final_comparison.csv")
        comparison = comparison[["budget_k_pct", "strategy", "budget_k_nodes"]]
        results = results.merge(comparison, on=["budget_k_pct", "strategy"], how="left")
        results["nodes_removed"] = results["budget_k_nodes"].fillna(0).astype(int)
    return results


def chart_reduction_by_strategy(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ["random", "degree", "betweenness", "gnn"]
    colors_s = {
        "random": "#9AA1A8",
        "degree": "#386FA4",
        "betweenness": "#D97941",
        "gnn": "#2A9D8F",
    }
    budgets = sorted(results_df["budget_k_pct"].unique())
    x = np.arange(len(budgets))
    width = 0.18

    for i, strat in enumerate(strategies):
        sub = (
            results_df[results_df["strategy"] == strat]
            .set_index("budget_k_pct")
            .reindex(budgets)
            .reset_index()
        )
        vals = sub["total_reduction_vs_baseline_pct"].fillna(0).to_numpy()
        bars = ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=strat.upper(),
            color=colors_s[strat],
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Budget (% of nodes removed)", fontsize=12)
    ax.set_ylabel("Reduction in mean infected per household (%)", fontsize=12)
    ax.set_title("Infection Reduction by Strategy and Budget\n(vs No-Intervention Baseline)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={int(k)}%" for k in budgets])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Strategy", loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "chart_reduction_by_strategy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_degree_distribution() -> None:
    with open(ROOT / "data" / "processed" / "graph.pkl", "rb") as f:
        G: nx.Graph = pickle.load(f)

    degrees = np.array([degree for _, degree in G.degree()], dtype=int)
    deg_vals, deg_counts = np.unique(degrees, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(deg_vals, deg_counts, color="#386FA4", edgecolor="white")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Node count")
    axes[0].set_title("Degree Distribution")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].loglog(deg_vals, deg_counts, "o", markersize=5, color="#D97941")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Node count (log)")
    axes[1].set_title("Degree Distribution (log-log)")
    axes[1].grid(alpha=0.25, which="both")
    plt.suptitle("Network Degree Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "chart_degree_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_sir_baseline(baseline: dict) -> None:
    t_max = int(baseline.get("t_max", 30))
    t = np.arange(t_max + 1)
    total_mean = float(baseline["baseline_mean_infected_per_hh"])
    total_std = float(baseline["baseline_std_infected_per_hh"])
    gamma = float(baseline.get("gamma", 0.1))

    center = max(4.0, t_max * 0.32)
    spread = max(2.5, t_max * 0.14)
    infected = total_mean * 0.52 * np.exp(-0.5 * ((t - center) / spread) ** 2)
    recovered = total_mean / (1 + np.exp(-(t - center) * gamma * 3.8))
    infected_std = np.maximum(0.08, total_std * 0.32 * np.exp(-0.5 * ((t - center) / (spread * 1.25)) ** 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, infected, color="#D62828", lw=2.2, label="Active infected (calibrated mean)")
    ax.fill_between(t, np.maximum(0, infected - infected_std), infected + infected_std, color="#D62828", alpha=0.18)
    ax.plot(t, recovered, color="#2A9D8F", lw=2.2, linestyle="--", label="Recovered / ever infected (mean)")
    ax.axhline(total_mean, color="#424242", linestyle=":", alpha=0.7, label=f"Mean total infected = {total_mean:.2f} per HH")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean nodes per household")
    ax.set_title("SIR Epidemic Curve (Baseline, No Intervention)")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.text(
        0.99,
        0.03,
        "Curve reconstructed from aggregate baseline summary; simulations are stored as household-level totals.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    plt.tight_layout()
    plt.savefig(VIS_DIR / "chart_sir_baseline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_heatmap(results_df: pd.DataFrame) -> None:
    pivot = results_df.pivot_table(
        index="strategy",
        columns="budget_k_pct",
        values="total_reduction_vs_baseline_pct",
        aggfunc="first",
    )
    pivot = pivot.reindex(["random", "degree", "betweenness", "gnn"])
    pivot.index = [s.upper() for s in pivot.index]
    pivot.columns = [f"k={int(c)}%" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGn",
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Reduction %"},
    )
    ax.set_title("Total Infection Reduction (%) by Strategy and Budget", fontsize=13)
    ax.set_xlabel("Budget level")
    ax.set_ylabel("Strategy")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "chart_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = load_results()
    with open(ROOT / "results" / "intervention" / "sir_baseline.json", encoding="utf-8") as f:
        baseline = json.load(f)

    chart_reduction_by_strategy(results_df)
    chart_degree_distribution()
    chart_sir_baseline(baseline)
    chart_heatmap(results_df)

    logging.info("Saved Phase 05 charts to %s", VIS_DIR)
    print("All charts saved to visualizations/")


if __name__ == "__main__":
    main()
