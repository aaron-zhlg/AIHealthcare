"""Leave-one-site-out cross-validation for GCN on ABIDE FC graphs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from aihealthcare.fc_dataset import AbideFCDataset, collate_graphs
from aihealthcare.gcn import SimpleGCN
from aihealthcare.train import TrainLogger, evaluate, pick_device, train_one_epoch

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "loso_cv_gcn_v1"
DEFAULT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "loso_cv_gcn_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Write paper-ready run_config.json and results.json here",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--sites",
        type=str,
        default="",
        help="Optional comma-separated SITE_ID list to run (default: all sites)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log training progress every N epochs within each fold",
    )
    return parser.parse_args()


def group_indices_by_site(dataset: AbideFCDataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        site_id = str(dataset[index]["site_id"])
        groups[site_id].append(index)
    return dict(sorted(groups.items()))


def train_and_evaluate_fold(
    dataset: AbideFCDataset,
    train_indices: list[int],
    test_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
    logger: TrainLogger,
) -> dict[str, float | int | str]:
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
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

    best_auc = -1.0
    final_metrics: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, device)
        final_metrics = test_metrics
        best_auc = max(best_auc, test_metrics["auc"])

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            logger.log(
                f"    epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"test_acc={test_metrics['accuracy']:.3f} | "
                f"test_auc={test_metrics['auc']:.3f}"
            )

    # Keep the last epoch, not the best one: the held-out site is the score, so
    # selecting on it would leak the test fold into model selection.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "num_nodes": num_nodes,
            "final_metrics": final_metrics,
        },
        fold_dir / "final_model.pt",
    )

    return {
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "diagnostic_best_epoch_auc": float(best_auc),
        **final_metrics,
    }


def aggregate_metrics(folds: list[dict]) -> dict[str, float]:
    keys = ("accuracy", "auc", "f1")
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(fold[key]) for fold in folds]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(force_cpu=args.cpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.output_dir / "loso_cv.log")

    dataset = AbideFCDataset(args.data_dir)
    site_groups = group_indices_by_site(dataset)

    if args.sites.strip():
        selected = [site.strip() for site in args.sites.split(",") if site.strip()]
        site_groups = {site: site_groups[site] for site in selected if site in site_groups}

    logger.log(f"Started at {datetime.now(timezone.utc).isoformat()}")
    logger.log(f"Device: {device}")
    logger.log(f"Output dir: {args.output_dir}")
    logger.log(f"Experiment dir: {args.experiment_dir}")
    logger.log(f"Args: {vars(args)}")
    logger.log(f"Sites to evaluate: {len(site_groups)}")
    logger.log("")

    fold_results: list[dict] = []

    for fold_index, (site_id, test_indices) in enumerate(site_groups.items(), start=1):
        train_indices = [
            index
            for other_site, indices in site_groups.items()
            if other_site != site_id
            for index in indices
        ]
        fold_dir = args.output_dir / "folds" / site_id

        logger.log(
            f"[{fold_index}/{len(site_groups)}] held-out site={site_id} | "
            f"train={len(train_indices)} test={len(test_indices)}"
        )

        metrics = train_and_evaluate_fold(
            dataset=dataset,
            train_indices=train_indices,
            test_indices=test_indices,
            args=args,
            device=device,
            fold_dir=fold_dir,
            logger=logger,
        )
        fold_record = {
            "site_id": site_id,
            **metrics,
        }
        fold_results.append(fold_record)
        logger.log(
            f"    fold result | acc={metrics['accuracy']:.3f} | "
            f"auc={metrics['auc']:.3f} | f1={metrics['f1']:.3f}"
        )
        logger.log("")

    summary = aggregate_metrics(fold_results)

    logger.log("LOSO-CV summary (mean ± std across sites):")
    logger.log(
        f"  Accuracy: {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}"
    )
    logger.log(f"  AUC:      {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
    logger.log(f"  F1:       {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")

    output_summary = {
        "method": "leave-one-site-out cross-validation",
        "model": "SimpleGCN (2-layer GCN, global mean pooling)",
        "num_folds": len(fold_results),
        "folds": fold_results,
        "aggregate": summary,
        "output_dir": str(args.output_dir),
        "log_file": str(args.output_dir / "loso_cv.log"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "folds.json").write_text(
        json.dumps(fold_results, indent=2) + "\n", encoding="utf-8"
    )

    run_config = {
        "name": "loso_cv_gcn_v1",
        "validation": "leave-one-site-out (LOSO-CV)",
        "model_selection": "final epoch (no selection on the held-out site)",
        "model": "SimpleGCN (2-layer GCN, global mean pooling)",
        "data": {
            "dataset": "ABIDE Preprocessed rois_ho",
            "num_subjects": len(dataset),
            "num_rois": 111,
            "fc_method": "Pearson correlation",
            "num_sites": len(fold_results),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "optimizer": "Adam",
            "weight_decay": 0.0001,
            "seed": args.seed,
        },
        "reproduce": "./scripts/run_loso_cv.sh",
    }
    results = {
        "name": "loso_cv_gcn_v1",
        "evaluated_at": datetime.now(timezone.utc).date().isoformat(),
        "validation": "leave-one-site-out across acquisition sites",
        "model_selection": "final epoch (no selection on the held-out site)",
        "aggregate_metrics": {
            "accuracy_mean": round(summary["accuracy_mean"], 4),
            "accuracy_std": round(summary["accuracy_std"], 4),
            "auc_mean": round(summary["auc_mean"], 4),
            "auc_std": round(summary["auc_std"], 4),
            "f1_mean": round(summary["f1_mean"], 4),
            "f1_std": round(summary["f1_std"], 4),
        },
        "folds": fold_results,
        "notes": (
            "Each fold reports the last epoch. Fold entries also carry "
            "diagnostic_best_epoch_auc, which is what selecting the best epoch on the "
            "held-out site would have given; it is optimistic and must not be reported."
        ),
        "local_artifacts": {
            "summary": "outputs/loso_cv_gcn_v1/summary.json",
            "folds": "outputs/loso_cv_gcn_v1/folds.json",
            "log": "outputs/loso_cv_gcn_v1/loso_cv.log",
        },
    }
    (args.experiment_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    (args.experiment_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    logger.log(f"Saved summary: {args.output_dir / 'summary.json'}")
    logger.log(f"Saved experiment results: {args.experiment_dir / 'results.json'}")
    logger.close()


if __name__ == "__main__":
    main()
