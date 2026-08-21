# ABIDE Processed Dataset

This directory contains the **aligned ABIDE preprocessed data**, ready for ASD vs. healthy control classification, brain network analysis, and GNN modeling.

> **Path convention:** All relative paths in this document and in `manifest.csv` are relative to `data/abide/`.  
> **Generation script:** `../scripts/build_aligned_dataset.py`  
> **Last updated:** 2026-08-21

---

## 1. What Is This?

**ABIDE** (Autism Brain Imaging Data Exchange) is a public neuroimaging dataset for autism research. We downloaded PCP-preprocessed **ROI time series** (`rois_ho`) and applied two processing steps:

1. **Alignment & filtering** — keep only subjects present in both the phenotypic CSV and local `.1D` files
2. **Functional connectivity precomputation** — compute Pearson correlation matrices from ROI time series and save as `.npy`

Original `.1D` files remain in `../raw/`. This directory does not copy them; it stores indices and derived features only.

---

## 2. Dataset Summary

| Metric | Value |
|--------|-------|
| Aligned subjects | **884** |
| ASD | 408 |
| Healthy controls | 476 |
| ROIs (brain regions) | 111 (Harvard-Oxford atlas) |
| FC matrix size | 111 × 111 |
| Original CSV rows | 1,112 |
| Excluded (no `.1D` or failed QC) | 151 + 77 (`no_filename`, etc.) |

**QC criteria (same as download script):**

- `FILE_ID != "no_filename"`
- `func_mean_fd < 0.2` (mean framewise displacement < 0.2 mm)

---

## 3. Directory Layout

```text
processed/
├── README.md           ← this document
├── summary.json        ← dataset statistics
├── manifest.csv        ← main index (884 rows, one per subject)
├── labels.npy          ← label array, shape (884,), same order as manifest
├── file_ids.txt        ← one FILE_ID per line
├── fc_paths.npy        ← relative FC paths (same order as manifest)
└── fc/
    ├── CMU_a_0050649_fc.npy
    ├── Caltech_0051476_fc.npy
    └── ...             ← 884 FC matrices total
```

---

## 4. manifest.csv Fields

Each row is one subject — the **primary entry point** for modeling.

| Field | Example | Description |
|-------|---------|-------------|
| `FILE_ID` | `Caltech_0051476` | Unique subject identifier |
| `SUB_ID` | `51476` | Original ABIDE subject ID |
| `SITE_ID` | `CALTECH` | Acquisition site |
| `DX_GROUP` | `2` | Original diagnosis: `1`=ASD, `2`=healthy control |
| `label` | `1` | Modeling label: `0`=ASD, `1`=control |
| `AGE_AT_SCAN` | `39.3` | Age at scan (years) |
| `SEX` | `1` | Sex: `1`=male, `2`=female |
| `FIQ` | `123` | Full Scale IQ |
| `func_mean_fd` | `0.070` | Mean head motion (mm); lower is better |
| `roi_path` | `raw/Outputs/.../Caltech_0051476_rois_ho.1D` | ROI time series (relative path) |
| `fc_path` | `processed/fc/Caltech_0051476_fc.npy` | Functional connectivity matrix (relative path) |

### Label Mapping

| Original `DX_GROUP` | Modeling `label` | Meaning |
|---------------------|------------------|---------|
| 1 | 0 | Autism Spectrum Disorder (ASD) |
| 2 | 1 | Typically Developing Control |

---

## 5. File Formats

### 5.1 FC Matrix — `fc/{FILE_ID}_fc.npy`

- **Format:** NumPy array, `float32`
- **Shape:** `(111, 111)`
- **Content:** Pearson functional connectivity between 111 ROIs (values in [-1, 1])
- **Computation:** `np.corrcoef(ts, rowvar=False)` on ROI time series
- **Use case:** Adjacency matrix for brain graphs; input to GNN or traditional ML

### 5.2 ROI Time Series — `.1D` files via `roi_path`

- **Location:** `../raw/Outputs/cpac/filt_global/rois_ho/`
- **Format:** AFNI-style plain-text matrix
- **Shape:** ~`(T, 111)` — timepoints × ROIs (T varies slightly by site)
- **Use case:** Custom FC methods, dynamic connectivity, node feature engineering

### 5.3 labels.npy

- **Shape:** `(884,)`
- **dtype:** `int64`
- **Values:** `0`=ASD, `1`=control
- **Order:** Matches row order in `manifest.csv` and `file_ids.txt`

---

## 6. Usage

### 6.1 Load a Single Subject

```python
import csv
import numpy as np
from pathlib import Path

base = Path("data/abide")

rows = list(csv.DictReader(open(base / "processed/manifest.csv")))
row = rows[0]

label = int(row["label"])        # 0=ASD, 1=control
site = row["SITE_ID"]
age = float(row["AGE_AT_SCAN"])

fc = np.load(base / row["fc_path"])   # shape: (111, 111)
ts = np.loadtxt(base / row["roi_path"], comments="#")  # shape: (T, 111)
```

### 6.2 Batch Load All Subjects

```python
import csv
import numpy as np
from pathlib import Path

base = Path("data/abide")
rows = list(csv.DictReader(open(base / "processed/manifest.csv")))

graphs, labels = [], []
for row in rows:
    graphs.append(np.load(base / row["fc_path"]))
    labels.append(int(row["label"]))

X = np.stack(graphs)   # (884, 111, 111)
y = np.array(labels)   # (884,)
print(f"Samples: {X.shape[0]}, ASD: {(y==0).sum()}, Control: {(y==1).sum()}")
```

### 6.3 Quick Label Access

```python
import numpy as np
from pathlib import Path

base = Path("data/abide")
y = np.load(base / "processed/labels.npy")
file_ids = (base / "processed/file_ids.txt").read_text().strip().split("\n")
# file_ids[i] corresponds to y[i]
```

---

## 7. Processing Pipeline

```text
Phenotypic_V1_0b_preprocessed1.csv (1112 rows)
        +
raw/.../rois_ho/*.1D (884 files)
        │
        ▼  build_aligned_dataset.py
        │  · intersect on FILE_ID → 884 subjects
        │  · write manifest.csv (labels + metadata + paths)
        │  · compute FC from .1D → fc/*.npy
        ▼
processed/ (this directory)
```

---

## 8. Regenerate

From the project root:

```bash
uv run python data/abide/scripts/build_aligned_dataset.py
```

Update manifest only (skip FC recomputation):

```bash
uv run python data/abide/scripts/build_aligned_dataset.py --no-fc
```

---

## 9. Modeling Notes

| Task | Recommended input | Label |
|------|-------------------|-------|
| ASD vs. control classification | `fc/*.npy` → brain graph | `label` |
| Control for site effects | same + group by `SITE_ID` | `label` |
| Control for covariates | same + `AGE_AT_SCAN`, `SEX`, `FIQ` | `label` |
| Custom connectivity | `roi_path` → `.1D` time series | `label` |
| Dynamic FC | `.1D` + sliding window | `label` |

**Important:** Subjects come from 20 acquisition sites. Use **leave-one-site-out cross-validation (LOSO-CV)** to avoid models learning site-specific artifacts instead of ASD-related patterns.

---

## 10. Related Docs

- Raw data documentation: `../README.md`
- Alignment script: `../scripts/build_aligned_dataset.py`
- ABIDE official site: https://preprocessed-connectomes-project.org/abide/
