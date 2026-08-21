#!/usr/bin/env python3
"""Align ABIDE phenotypic CSV with downloaded rois_ho time series.

Keeps only subjects present in both the phenotypic table and local .1D files,
writes a filtered manifest, and optionally precomputes functional connectivity
(FC) matrices for GNN / ML pipelines.

Usage:
    python3 build_aligned_dataset.py
    python3 build_aligned_dataset.py --no-fc
    python3 build_aligned_dataset.py --base-dir data/abide --out-dir data/abide/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numpy is required. Install with: pip install numpy"
    ) from exc

DEFAULT_BASE = Path(__file__).resolve().parents[1]
PHENO_NAME = "Phenotypic_V1_0b_preprocessed1.csv"
ROIS_REL = Path("raw/Outputs/cpac/filt_global/rois_ho")
MANIFEST_FIELDS = [
    "FILE_ID",
    "SUB_ID",
    "SITE_ID",
    "DX_GROUP",
    "label",
    "AGE_AT_SCAN",
    "SEX",
    "FIQ",
    "func_mean_fd",
    "roi_path",
    "fc_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="ABIDE data root (contains phenotypic/ and raw/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <base-dir>/processed)",
    )
    parser.add_argument(
        "--no-fc",
        action="store_true",
        help="Skip FC matrix computation; only write manifest",
    )
    return parser.parse_args()


def load_phenotypic(pheno_path: Path) -> dict[str, dict[str, str]]:
    with pheno_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["FILE_ID"]: row for row in rows if row.get("FILE_ID")}


def discover_roi_files(roi_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(roi_dir.glob("*_rois_ho.1D")):
        file_id = path.name[: -len("_rois_ho.1D")]
        mapping[file_id] = path
    return mapping


def load_roi_timeseries(path: Path) -> np.ndarray:
    # First line is ROI header (#10, #11, ...); comments='#' skips it.
    return np.loadtxt(path, comments="#")


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    return np.corrcoef(timeseries, rowvar=False)


def dx_group_to_label(dx_group: str) -> int:
    """Map DX_GROUP to 0-indexed class label: ASD=0, control=1."""
    value = int(dx_group)
    if value == 1:
        return 0
    if value == 2:
        return 1
    raise ValueError(f"Unexpected DX_GROUP={dx_group!r}; expected 1 (ASD) or 2 (control)")


def rel_path(path: Path, anchor: Path) -> str:
    """Return a POSIX path relative to anchor (for portable manifests)."""
    return path.resolve().relative_to(anchor.resolve()).as_posix()


def build_manifest_row(
    row: dict[str, str], roi_rel_path: str, fc_rel_path: str | None = None
) -> dict[str, str]:
    manifest = {
        "FILE_ID": row["FILE_ID"],
        "SUB_ID": row.get("SUB_ID", ""),
        "SITE_ID": row.get("SITE_ID", ""),
        "DX_GROUP": row["DX_GROUP"],
        "label": str(dx_group_to_label(row["DX_GROUP"])),
        "AGE_AT_SCAN": row.get("AGE_AT_SCAN", ""),
        "SEX": row.get("SEX", ""),
        "FIQ": row.get("FIQ", ""),
        "func_mean_fd": row.get("func_mean_fd", ""),
        "roi_path": roi_rel_path,
    }
    if fc_rel_path is not None:
        manifest["fc_path"] = fc_rel_path
    return manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    out_dir = (args.out_dir or base_dir / "processed").resolve()

    pheno_path = base_dir / "phenotypic" / PHENO_NAME
    roi_dir = base_dir / ROIS_REL

    if not pheno_path.exists():
        raise SystemExit(f"Phenotypic file not found: {pheno_path}")
    if not roi_dir.is_dir():
        raise SystemExit(f"ROI directory not found: {roi_dir}")

    pheno_by_id = load_phenotypic(pheno_path)
    roi_by_id = discover_roi_files(roi_dir)

    csv_ids = set(pheno_by_id) - {"no_filename"}
    roi_ids = set(roi_by_id)
    aligned_ids = sorted(csv_ids & roi_ids)

    csv_only = sorted(csv_ids - roi_ids)
    roi_only = sorted(roi_ids - csv_ids)

    print(f"Phenotypic rows with FILE_ID: {len(pheno_by_id)}")
    print(f"Phenotypic rows (excluding no_filename): {len(csv_ids)}")
    print(f"Local rois_ho files: {len(roi_ids)}")
    print(f"Aligned subjects: {len(aligned_ids)}")
    print(f"CSV only (no .1D): {len(csv_only)}")
    print(f".1D only (no CSV row): {len(roi_only)}")

    manifest_rows: list[dict[str, str]] = []
    fc_dir = out_dir / "fc"
    fc_paths: list[str] = []
    labels: list[int] = []
    file_ids: list[str] = []

    if not args.no_fc:
        fc_dir.mkdir(parents=True, exist_ok=True)

    for index, file_id in enumerate(aligned_ids, start=1):
        pheno_row = pheno_by_id[file_id]
        roi_path = roi_by_id[file_id]
        roi_rel = rel_path(roi_path, base_dir)
        fc_rel: str | None = None

        if args.no_fc:
            manifest_rows.append(build_manifest_row(pheno_row, roi_rel))
        else:
            fc_rel = rel_path(fc_dir / f"{file_id}_fc.npy", base_dir)
            manifest_rows.append(build_manifest_row(pheno_row, roi_rel, fc_rel))

        label = dx_group_to_label(pheno_row["DX_GROUP"])
        labels.append(label)
        file_ids.append(file_id)

        if args.no_fc:
            continue

        ts = load_roi_timeseries(roi_path)
        fc = compute_fc(ts)
        fc_path = fc_dir / f"{file_id}_fc.npy"
        np.save(fc_path, fc.astype(np.float32))
        fc_paths.append(fc_rel)

        if index % 100 == 0 or index == len(aligned_ids):
            print(f"  FC computed: {index}/{len(aligned_ids)}")

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    labels_arr = np.array(labels, dtype=np.int64)
    file_ids_path = out_dir / "file_ids.txt"
    file_ids_path.write_text("\n".join(file_ids) + "\n", encoding="utf-8")
    np.save(out_dir / "labels.npy", labels_arr)

    if not args.no_fc:
        np.save(out_dir / "fc_paths.npy", np.array(fc_paths, dtype=object))

    summary = {
        "n_aligned": len(aligned_ids),
        "n_asd": int((labels_arr == 0).sum()),
        "n_control": int((labels_arr == 1).sum()),
        "n_csv_only": len(csv_only),
        "n_roi_only": len(roi_only),
        "base_dir": ".",
        "manifest": rel_path(manifest_path, base_dir),
        "labels": rel_path(out_dir / "labels.npy", base_dir),
        "file_ids": rel_path(file_ids_path, base_dir),
        "fc_dir": None if args.no_fc else rel_path(fc_dir, base_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote manifest: {manifest_path}")
    print(f"Wrote labels:   {out_dir / 'labels.npy'}  (0=ASD, 1=control)")
    print(f"Wrote file IDs: {file_ids_path}")
    if not args.no_fc:
        print(f"Wrote FC mats:  {fc_dir}/ ({len(fc_paths)} files)")
    print(f"Wrote summary:  {summary_path}")
    print(
        f"\nClass balance: ASD={summary['n_asd']}, "
        f"control={summary['n_control']}"
    )


if __name__ == "__main__":
    main()
