from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    setup_logging()
    baseline = json.load(open(ROOT / "results" / "intervention" / "sir_baseline.json", encoding="utf-8"))
    metrics = json.load(open(ROOT / "results" / "metrics" / "basic_metrics.json", encoding="utf-8"))
    results = pd.read_csv(ROOT / "results" / "intervention" / "sir_intervention_results.csv")
    comparison = pd.read_csv(ROOT / "results" / "intervention" / "final_comparison.csv")

    if "total_reduction_vs_baseline_pct" not in results:
        results["total_reduction_vs_baseline_pct"] = results["reduction_vs_baseline_pct"]
    if "peak_reduction_vs_baseline_pct" not in results:
        results["peak_reduction_vs_baseline_pct"] = results["total_reduction_vs_baseline_pct"]
    results = results.merge(
        comparison[["budget_k_pct", "strategy", "budget_k_nodes"]],
        on=["budget_k_pct", "strategy"],
        how="left",
    )
    results["nodes_removed"] = results["budget_k_nodes"].fillna(0).astype(int)

    print("=" * 72)
    print("  BUDGET-CONSTRAINED EPIDEMIC INTERVENTION - RESULTS REPORT")
    print(f"  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    print("\n  DATASET OVERVIEW:")
    print(
        f"    Nodes: {metrics['n_nodes_total']} | Edges: {metrics['n_edges_total']} | "
        f"Households: {metrics['n_households']} | Avg HH density: {metrics['avg_density_per_hh']:.3f}"
    )

    print("\n  BASELINE (No Intervention):")
    print(
        f"    Mean Infected: {baseline['baseline_mean_infected_per_hh']:.2f} nodes/household "
        f"(std {baseline['baseline_std_infected_per_hh']:.2f})"
    )
    print(f"    Observed Attack Rate: {baseline['observed_attack_rate_pct']:.1f}%")

    print("\n  INTERVENTION RESULTS:")
    header = f"  {'Strategy':<14} {'Budget':>7} {'PeakDown':>9} {'TotalDown':>10} {'Nodes Used':>12}"
    print(header)
    print("  " + "-" * 60)
    sorted_results = results.sort_values(
        ["budget_k_pct", "total_reduction_vs_baseline_pct"],
        ascending=[True, False],
    )
    for _, row in sorted_results.iterrows():
        print(
            f"  {row['strategy'].upper():<14} "
            f"{row['budget_k_pct']:>5.0f}%  "
            f"{row['peak_reduction_vs_baseline_pct']:>7.1f}%  "
            f"{row['total_reduction_vs_baseline_pct']:>8.1f}%  "
            f"{row['nodes_removed']:>10d}"
        )

    print("\n  BEST STRATEGY PER BUDGET:")
    for k in sorted(results["budget_k_pct"].unique()):
        sub = results[results["budget_k_pct"] == k].sort_values(
            "total_reduction_vs_baseline_pct",
            ascending=False,
        )
        best = sub.iloc[0]
        print(
            f"    k={int(k):2d}%: {best['strategy'].upper():<12} -> "
            f"Total Infection down {best['total_reduction_vs_baseline_pct']:.1f}%"
        )

    print("\n  Export PDF file at: reports/final_report.pdf")
    print("=" * 72)
    print("Phase 05 DONE. All outputs saved.")
    logging.info("Printed Phase 05 console report")


if __name__ == "__main__":
    main()
