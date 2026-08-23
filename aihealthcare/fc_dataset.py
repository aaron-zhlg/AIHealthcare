"""Load ABIDE FC matrices for GNN training."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"


def load_manifest(data_dir: Path) -> list[dict[str, str]]:
    manifest_path = data_dir / "processed" / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preprocess_fc(fc: np.ndarray) -> np.ndarray:
    """Replace NaNs and clip extreme correlations."""
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(fc, -1.0, 1.0).astype(np.float32)


class AbideFCDataset(Dataset):
    """Each sample is one subject's functional connectivity graph."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.records = load_manifest(self.data_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        fc = np.load(self.data_dir / record["fc_path"])
        fc = preprocess_fc(fc)

        # Node features: each ROI's connectivity profile (111-dim vector).
        node_features = torch.from_numpy(fc)
        adjacency = torch.from_numpy(np.abs(fc))
        label = torch.tensor(int(record["label"]), dtype=torch.long)

        return {
            "node_features": node_features,
            "adjacency": adjacency,
            "label": label,
            "file_id": record["FILE_ID"],
            "site_id": record["SITE_ID"],
        }


def collate_graphs(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "node_features": torch.stack([item["node_features"] for item in batch]),
        "adjacency": torch.stack([item["adjacency"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "file_id": [item["file_id"] for item in batch],
        "site_id": [item["site_id"] for item in batch],
    }
