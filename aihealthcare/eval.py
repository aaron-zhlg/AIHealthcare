"""Evaluate a trained GCN checkpoint on ABIDE FC graphs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from aihealthcare.fc_dataset import AbideFCDataset, collate_graphs
from aihealthcare.gcn import SimpleGCN
from aihealthcare.train import TrainLogger, evaluate, pick_device

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "gcn_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint (default: <output-dir>/final_model.pt)",
    )
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="Which subjects to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/GPU exists")
    return parser.parse_args()


def load_split_indices(output_dir: Path) -> dict[str, list[int]]:
    split_path = output_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run training first to create it."
        )
    return json.loads(split_path.read_text(encoding="utf-8"))


def load_model(checkpoint_path: Path, device: torch.device) -> SimpleGCN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    num_nodes = int(checkpoint["num_nodes"])

    model = SimpleGCN(
        in_features=num_nodes,
        hidden_dim=int(train_args.get("hidden_dim", 64)),
        num_classes=2,
        dropout=float(train_args.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(
    model: SimpleGCN,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for batch in loader:
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["label"].to(device)

        logits = model(node_features, adjacency)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for index in range(labels.size(0)):
            rows.append(
                {
                    "file_id": batch["file_id"][index],
                    "site_id": batch["site_id"][index],
                    "label": int(labels[index].item()),
                    "pred": int(preds[index].item()),
                    "prob_control": float(probs[index, 1].item()),
                    "prob_asd": float(probs[index, 0].item()),
                    "correct": int(preds[index].item() == labels[index].item()),
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or (args.output_dir / "final_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.output_dir / "eval.log")

    device = pick_device(force_cpu=args.cpu)
    split = load_split_indices(args.output_dir)
    dataset = AbideFCDataset(args.data_dir)

    if args.split == "train":
        indices = split["train_idx"]
    elif args.split == "val":
        indices = split["val_idx"]
    else:
        indices = list(range(len(dataset)))

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    logger.log(f"Started at {datetime.now(timezone.utc).isoformat()}")
    logger.log(f"Device: {device}")
    logger.log(f"Checkpoint: {checkpoint_path}")
    logger.log(f"Split: {args.split} ({len(indices)} subjects)")
    logger.log("")

    model = load_model(checkpoint_path, device)
    metrics = evaluate(model, loader, device)
    predictions = collect_predictions(model, loader, device)

    labels = [int(row["label"]) for row in predictions]
    preds = [int(row["pred"]) for row in predictions]
    probs = [float(row["prob_control"]) for row in predictions]

    report = classification_report(
        labels,
        preds,
        target_names=["asd", "control"],
        digits=4,
    )
    matrix = confusion_matrix(labels, preds).tolist()

    logger.log(
        f"Accuracy={metrics['accuracy']:.4f} | "
        f"AUC={metrics['auc']:.4f} | F1={metrics['f1']:.4f}"
    )
    logger.log("")
    logger.log("Confusion matrix [rows=true, cols=pred] (0=ASD, 1=control):")
    logger.log(str(matrix))
    logger.log("")
    logger.log("Classification report:")
    logger.log(report)

    predictions_path = args.output_dir / f"predictions_{args.split}.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_id",
                "site_id",
                "label",
                "pred",
                "prob_asd",
                "prob_control",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)

    summary = {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "model_selection": "final epoch (no selection on the evaluation set)",
        "split": args.split,
        "num_subjects": len(indices),
        "metrics": metrics,
        "confusion_matrix": matrix,
        "predictions_csv": str(predictions_path),
        "log_file": str(args.output_dir / "eval.log"),
    }
    eval_metrics_path = args.output_dir / f"eval_{args.split}.json"
    eval_metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.log("")
    logger.log(f"Saved predictions: {predictions_path}")
    logger.log(f"Saved metrics:     {eval_metrics_path}")
    logger.log(f"Saved log:         {args.output_dir / 'eval.log'}")
    logger.close()


if __name__ == "__main__":
    main()
