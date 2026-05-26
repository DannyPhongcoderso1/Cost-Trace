import json
import logging
import pickle
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
EPOCHS = 500
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

LOG_PATH = Path("logs/modeling.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_weighted_mean_adjacency(G: nx.Graph, nodes: list[str]) -> torch.Tensor:
    """Build row-normalized weighted adjacency for mean-neighbor aggregation."""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    adj = torch.zeros((len(nodes), len(nodes)), dtype=torch.float32)

    for u, v, attrs in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = float(attrs.get("total_duration_sec", 1.0))
        weight = np.log1p(max(weight, 0.0))
        adj[i, j] = weight
        adj[j, i] = weight

    row_sum = adj.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, adj / row_sum.clamp_min(1e-12), adj)


def build_features_and_labels(scores_df: pd.DataFrame, nodes: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    scores = scores_df.set_index("node_id")
    feature_names = [
        "degree_centrality_norm",
        "weighted_degree_sec_norm",
        "betweenness_centrality_norm",
        "closeness_centrality_norm",
        "sleep_room_enc",
        "cared_by_enc",
        "age_enc_scaled",
        "sex_enc",
        "sus_enc",
        "site_enc",
        "is_index",
    ]

    rows = []
    labels = []
    for node in nodes:
        if node not in scores.index:
            rows.append([0.0] * len(feature_names))
            labels.append(0.0)
            continue

        row = scores.loc[node]
        rows.append(
            [
                float(row.get("degree_centrality_norm", 0.0)),
                float(row.get("weighted_degree_sec_norm", 0.0)),
                float(row.get("betweenness_centrality_norm", 0.0)),
                float(row.get("closeness_centrality_norm", 0.0)),
                float(row.get("sleep_room_enc", 0.0)),
                float(row.get("cared_by_enc", 0.0)),
                float(row.get("age_enc", 3.0)) / 5.0,
                float(row.get("sex_enc", 0.0)),
                float(row.get("sus_enc", 0.0)),
                float(row.get("site_enc", 0.0)),
                float(row.get("is_index", 0.0)),
            ]
        )
        labels.append(float(row.get("sars_label", 0.0)))

    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), feature_names


def stratified_masks(labels: torch.Tensor, seed: int = SEED) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    n = labels.numel()
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for label in [0.0, 1.0]:
        idx = np.where(labels.numpy() == label)[0]
        rng.shuffle(idx)
        n_train = int(0.70 * len(idx))
        n_val = int(0.15 * len(idx))

        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
        test_mask[idx[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


class WeightedGraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.self_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.neigh_1 = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        self.self_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.neigh_2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.self_1(x) + self.neigh_1(adj @ x))
        h = F.dropout(h, p=0.35, training=self.training)
        h = F.relu(self.self_2(h) + self.neigh_2(adj @ h))
        h = F.dropout(h, p=0.20, training=self.training)
        return self.out(h).squeeze(-1)


def metric_dict(true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(true, probs)) if len(np.unique(true)) > 1 else 0.0,
        "ap": float(average_precision_score(true, probs)),
        "f1": float(f1_score(true, preds, zero_division=0)),
        "precision": float(precision_score(true, preds, zero_division=0)),
        "recall": float(recall_score(true, preds, zero_division=0)),
    }
    return out


def best_f1_threshold(true: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(true, probs >= thr, zero_division=0) for thr in thresholds]
    return float(thresholds[int(np.argmax(scores))])


@torch.no_grad()
def predict_probs(model: torch.nn.Module, x: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(x, adj)
    return torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    set_seed(SEED)
    logging.info("GraphSAGE proxy training start")

    Path("models").mkdir(exist_ok=True)
    Path("results/model").mkdir(parents=True, exist_ok=True)

    G = pickle.load(open("data/processed/graph.pkl", "rb"))
    scores_df = pd.read_csv("results/metrics/node_scores.csv")
    nodes = sorted(G.nodes())

    adj = build_weighted_mean_adjacency(G, nodes)
    x, y, feature_names = build_features_and_labels(scores_df, nodes)
    train_mask, val_mask, test_mask = stratified_masks(y)

    n_pos = int(y[train_mask].sum().item())
    n_neg = int(train_mask.sum().item() - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    model = WeightedGraphSAGEClassifier(in_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best = {"val_auc": -1.0, "val_ap": -1.0, "epoch": 0, "state": None}
    print("=== GRAPH SAGE NODE CLASSIFICATION ===")
    print(f"Nodes: {len(nodes)} | Edges: {G.number_of_edges()} | Features: {x.shape[1]}")
    print(
        f"Train: {int(train_mask.sum())} | Val: {int(val_mask.sum())} | "
        f"Test: {int(test_mask.sum())}"
    )
    print(f"SARS+ in train: {n_pos} / {int(train_mask.sum())}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            probs = predict_probs(model, x, adj)
            val_true = y[val_mask].numpy().astype(int)
            val_probs = probs[val_mask.numpy()]
            val_metrics = metric_dict(val_true, val_probs, threshold=0.5)

            is_better = (
                val_metrics["auc"] > best["val_auc"]
                or (
                    np.isclose(val_metrics["auc"], best["val_auc"])
                    and val_metrics["ap"] > best["val_ap"]
                )
            )
            if is_better:
                best = {
                    "val_auc": val_metrics["auc"],
                    "val_ap": val_metrics["ap"],
                    "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }

            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"  Ep {epoch:3d} | Loss {loss.item():.4f} | "
                    f"Val AUC {val_metrics['auc']:.3f} AP {val_metrics['ap']:.3f} "
                    f"F1 {val_metrics['f1']:.3f}"
                )

    if best["state"] is None:
        best["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best["epoch"] = EPOCHS

    model.load_state_dict(best["state"])
    torch.save(model.state_dict(), "models/gnn_best.pt")

    all_probs = predict_probs(model, x, adj)
    val_true = y[val_mask].numpy().astype(int)
    test_true = y[test_mask].numpy().astype(int)
    val_probs = all_probs[val_mask.numpy()]
    test_probs = all_probs[test_mask.numpy()]
    threshold = best_f1_threshold(val_true, val_probs)

    train_metrics = metric_dict(y[train_mask].numpy().astype(int), all_probs[train_mask.numpy()], threshold)
    val_metrics = metric_dict(val_true, val_probs, threshold)
    test_metrics = metric_dict(test_true, test_probs, threshold)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"  Best epoch: {best['epoch']}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  AUC-ROC   : {test_metrics['auc']:.4f}")
    print(f"  Avg Prec  : {test_metrics['ap']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")

    gnn_df = pd.DataFrame(
        {
            "node_id": nodes,
            "gnn_infection_prob": all_probs,
            "gnn_pred_sars": (all_probs >= threshold).astype(int),
            "sars_label": y.numpy().astype(int),
        }
    )
    gnn_df.to_csv("results/model/gnn_risk_scores.csv", index=False)

    metrics = {
        "model": "WeightedGraphSAGEClassifier",
        "implementation": "pure_pytorch_weighted_mean_aggregation",
        "seed": SEED,
        "epochs": EPOCHS,
        "best_epoch": int(best["epoch"]),
        "threshold": round(threshold, 4),
        "feature_names": feature_names,
        "train": {k: round(v, 4) for k, v in train_metrics.items()},
        "validation": {k: round(v, 4) for k, v in val_metrics.items()},
        "test": {k: round(v, 4) for k, v in test_metrics.items()},
    }
    with open("results/model/gnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("models/gnn_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "node_order": nodes,
                "feature_names": feature_names,
                "threshold": threshold,
                "weighted_adjacency": "log1p(total_duration_sec), row-normalized",
            },
            f,
            indent=2,
        )

    print("\nExported: results/model/gnn_risk_scores.csv")
    print("Metrics : results/model/gnn_metrics.json")
    print("Model   : models/gnn_best.pt")

    logging.info(
        "GraphSAGE proxy training done | test_auc=%.4f test_ap=%.4f threshold=%.2f",
        test_metrics["auc"],
        test_metrics["ap"],
        threshold,
    )


if __name__ == "__main__":
    main()
