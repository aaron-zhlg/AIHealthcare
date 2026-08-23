"""Train a simple GCN baseline on ABIDE FC graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from aihealthcare.fc_dataset import AbideFCDataset, collate_graphs
from aihealthcare.gcn import SimpleGCN

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "gcn_baseline"


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/GPU exists")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: SimpleGCN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for batch in loader:
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["label"].to(device)

        logits = model(node_features, adjacency)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds)),
        "auc": float(roc_auc_score(all_labels, all_probs)),
    }


def train_one_epoch(
    model: SimpleGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(node_features, adjacency)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(force_cpu=args.cpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AbideFCDataset(args.data_dir)
    labels = np.array([int(dataset[i]["label"]) for i in range(len(dataset))])

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=args.seed
    )
    train_idx, val_idx = next(split.split(np.zeros(len(labels)), labels))

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    num_nodes = int(dataset[0]["node_features"].shape[0])
    model = SimpleGCN(
        in_features=num_nodes,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Subjects: {len(dataset)} (train={len(train_idx)}, val={len(val_idx)})")
    print(f"Nodes per graph: {num_nodes}")
    print()

    best_auc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "num_nodes": num_nodes,
                    "best_auc": best_auc,
                },
                args.output_dir / "best_model.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.3f} | "
                f"val_auc={val_metrics['auc']:.3f} | "
                f"val_f1={val_metrics['f1']:.3f}"
            )

    summary = {
        "device": str(device),
        "num_subjects": len(dataset),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "best_auc": best_auc,
        "final_metrics": history[-1],
        "checkpoint": str(args.output_dir / "best_model.pt"),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"Best val AUC: {best_auc:.3f}")
    print(f"Saved checkpoint: {args.output_dir / 'best_model.pt'}")
    print(f"Saved metrics:  {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
