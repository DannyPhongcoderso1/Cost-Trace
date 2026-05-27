from __future__ import annotations

import logging
from pathlib import Path

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


def gephi_title(name: str) -> str:
    return name.replace("_", " ").title()


def main() -> None:
    setup_logging()
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    scores_path = ROOT / "results" / "metrics" / "node_scores.csv"
    community_path = ROOT / "results" / "metrics" / "community_assignments.csv"
    edges_path = ROOT / "data" / "processed" / "edgelist.csv"

    scores_df = pd.read_csv(scores_path)
    comm_df = pd.read_csv(community_path)
    node_attrs = scores_df.merge(comm_df, on="node_id", how="left", suffixes=("", "_community"))
    node_attrs = node_attrs.rename(columns={col: gephi_title(col) for col in node_attrs.columns})
    node_attrs = node_attrs.rename(columns={"Node Id": "Id"})
    node_attrs.to_csv(VIS_DIR / "gephi_nodes.csv", index=False)

    edges_df = pd.read_csv(edges_path)
    gephi_edges = edges_df.rename(columns={"source": "Source", "target": "Target", "weight": "Weight"})
    keep_cols = [col for col in ["Source", "Target", "Weight", "n_contacts", "transmission"] if col in gephi_edges]
    gephi_edges = gephi_edges[keep_cols]
    gephi_edges["Type"] = "Undirected"
    gephi_edges = gephi_edges.rename(columns={"n_contacts": "Contact Count", "transmission": "Transmission"})
    gephi_edges.to_csv(VIS_DIR / "gephi_edges.csv", index=False)

    logging.info("Exported Gephi CSV files to %s", VIS_DIR)
    print("Gephi files exported:")
    print("  visualizations/gephi_nodes.csv")
    print("  visualizations/gephi_edges.csv")
    print("\nGEPHI INSTRUCTIONS:")
    print("1. File -> Import Spreadsheet -> gephi_nodes.csv (Node table)")
    print("2. File -> Import Spreadsheet -> gephi_edges.csv (Edge table)")
    print("3. Layout: ForceAtlas2 (run about 2000 steps)")
    print("4. Appearance -> Nodes -> Size -> Ranking -> Degree Centrality")
    print("5. Appearance -> Nodes -> Color -> Partition -> Louvain Community Id")
    print("6. Filters -> Giant Component if you need a cleaner single-component view")
    print("7. Export: File -> Export -> SVG/PNG (2000x2000px)")


if __name__ == "__main__":
    main()
