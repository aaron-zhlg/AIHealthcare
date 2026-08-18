#!/usr/bin/env python3
"""Download ABIDE func_minimal (4D resting-state fMRI volumes, minimally preprocessed)."""

from __future__ import annotations

import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

S3_PREFIX = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENO_URL = f"{S3_PREFIX}/Phenotypic_V1_0b_preprocessed1.csv"
MEAN_FD_THRESH = 0.2


def load_file_ids() -> list[str]:
    with urllib.request.urlopen(PHENO_URL, timeout=60) as response:
        rows = list(csv.DictReader(line.decode() for line in response))

    file_ids: list[str] = []
    for row in rows:
        file_id = row.get("FILE_ID", "")
        if not file_id or file_id == "no_filename":
            continue
        try:
            if float(row["func_mean_fd"]) >= MEAN_FD_THRESH:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        file_ids.append(file_id)
    return file_ids


def download_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_ids = load_file_ids()
    total = len(file_ids)
    print(f"Subjects to download: {total}")

    for index, file_id in enumerate(file_ids, start=1):
        rel_path = Path("Outputs/cpac/func_minimal") / f"{file_id}_func_minimal.nii.gz"
        local_path = out_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"[{index}/{total}] skip existing {file_id}")
            continue

        url = f"{S3_PREFIX}/{rel_path.as_posix()}"
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        print(f"[{index}/{total}] downloading {file_id}")

        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(local_path)
        except urllib.error.HTTPError as exc:
            print(f"HTTP error for {file_id}: {exc}", file=sys.stderr)
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError as exc:
            print(f"Stopped at {file_id}: {exc}", file=sys.stderr)
            if tmp_path.exists():
                tmp_path.unlink()
            raise SystemExit(1) from exc

    print("Done!")


if __name__ == "__main__":
    target = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/zhangligao/Documents/Codex/2026-07-17/ban/AIHealthcare/data/abide/raw"
    )
    download_all(target)
