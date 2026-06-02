from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


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


def main() -> None:
    setup_logging()
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "data" / "processed" / "graph.pkl", "rb") as f:
        G: nx.Graph = pickle.load(f)

    scores_df = pd.read_csv(ROOT / "results" / "metrics" / "node_scores.csv")
    comm_df = pd.read_csv(ROOT / "results" / "metrics" / "community_assignments.csv")

    community_col = "community_id" if "community_id" in comm_df else "louvain_community_id"
    n2comm = dict(zip(comm_df["node_id"], comm_df[community_col]))
    n2degree = dict(zip(scores_df["node_id"], scores_df["degree"]))
    n2risk = dict(zip(scores_df["node_id"], scores_df["composite_risk_score"]))

    communities = sorted(pd.Series(n2comm).dropna().unique())
    n_comm = max(len(communities), 3)
    colors = cm.tab20(np.linspace(0, 1, min(n_comm, 20)))
    color_index = {comm: idx for idx, comm in enumerate(communities)}
    color_map = [colors[color_index.get(n2comm.get(node), 0) % len(colors)] for node in G.nodes()]

    degree_values = np.array([n2degree.get(node, G.degree(node)) for node in G.nodes()], dtype=float)
    max_degree = max(float(degree_values.max()), 1.0)
    size_map = [45 + 420 * (n2degree.get(node, G.degree(node)) / max_degree) for node in G.nodes()]

    # The graph is a set of household components; spring layout separates them clearly.
    pos = nx.spring_layout(G, seed=42, k=0.55, iterations=150, weight="total_duration_sec")

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9AA1A8", width=0.55, alpha=0.35)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=color_map,
        node_size=size_map,
        edgecolors="#222222",
        linewidths=0.25,
        alpha=0.92,
    )

    top5 = scores_df.nlargest(5, "composite_risk_score")["node_id"].tolist()
    top5 = [node for node in top5 if node in G]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=top5,
        node_color="#D62828",
        node_size=520,
        edgecolors="white",
        linewidths=1.0,
        ax=ax,
        label="Top-5 risk nodes",
    )
    for node in top5:
        x, y = pos[node]
        ax.text(x, y + 0.035, node, fontsize=7, ha="center", va="bottom", color="#1B1B1B")

    ax.set_title(
        "Epidemic Contact Network\nSize = Degree | Color = Household Community | Red = Top Composite-Risk Nodes",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    ax.axis("off")

    legend_count = min(len(communities), 8)
    patches = [
        mpatches.Patch(color=colors[i % len(colors)], label=f"Community {communities[i]}")
        for i in range(legend_count)
    ]
    patches.append(mpatches.Patch(color="#D62828", label="Top-5 risk nodes"))
    ax.legend(handles=patches, loc="upper left", fontsize=9, frameon=False)

    fig.text(
        0.5,
        0.02,
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} detected communities",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(VIS_DIR / "network_overview.png", dpi=220, bbox_inches="tight")
    plt.savefig(VIS_DIR / "network_overview.svg", bbox_inches="tight")
    plt.close(fig)

    logging.info(
        "Saved network visualization | nodes=%s edges=%s top5=%s",
        G.number_of_nodes(),
        G.number_of_edges(),
        ",".join(top5),
    )
    print("Saved: visualizations/network_overview.png")


if __name__ == "__main__":
    main()
