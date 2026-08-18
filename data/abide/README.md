# ABIDE Downloaded Data Documentation

This document describes the **ABIDE data currently downloaded** in the `AIHealthcare` project: sources, contents, formats, field definitions, and recommended use cases.

> **Source:** [ABIDE Preprocessed — Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/download.html)  
> **Local path:** `data/abide/`  
> **Last updated:** 2026-08-18

---

## 1. Background: What Is ABIDE?

**ABIDE** (Autism Brain Imaging Data Exchange) is a multi-site, publicly shared neuroimaging dataset released through INDI (International Neuroimaging Data-sharing Initiative).

| Item | Description |
|------|-------------|
| Full name | Autism Brain Imaging Data Exchange |
| Subjects | ~1,112 total (539 ASD + 573 typically developing controls) |
| Modalities | Structural MRI + **resting-state fMRI** + extensive phenotypic data |
| Sites | 16+ international imaging centers |
| Research use | Autism (ASD) mechanisms, brain network analysis, disease classification, connectomics |

What we downloaded is the **ABIDE Preprocessed** release — derivatives uniformly preprocessed by PCP (Preprocessed Connectomes Project) and hosted on a public Amazon S3 bucket (HTTP download, no account required).

**Official resources:**

- Project homepage: http://preprocessed-connectomes-project.org/abide/
- Download instructions: http://preprocessed-connectomes-project.org/abide/download.html
- Official download script: https://github.com/preprocessed-connectomes-project/abide/blob/master/download_abide_preproc.py

**Raw (unpreprocessed) ABIDE data** must be obtained separately from INDI (not downloaded here):

- https://fcon_1000.projects.nitrc.org/indi/abide/

---

## 2. Current Download Summary

| Data | Status | Files | Size | Local path |
|------|--------|-------|------|------------|
| Phenotypic + QC CSV | ✅ Downloaded | 1 | ~438 KB | `phenotypic/Phenotypic_V1_0b_preprocessed1.csv` |
| ROI time series `rois_ho` | ✅ Downloaded | **884** | **~178 MB** | `raw/Outputs/cpac/filt_global/rois_ho/*.1D` |
| 4D fMRI volumes `func_minimal` | ❌ Not downloaded | 0 | — | Attempted, then stopped; full set ≈ 220 GB |
| 4D fMRI volumes `func_preproc` | ❌ Not downloaded | 0 | — | — |
| Scanner-level raw DICOM/BIDS | ❌ Not downloaded | 0 | — | Requires ABIDE/INDI original release |

**Subject composition (884 subjects after QC filtering):**

| Metric | Value |
|--------|-------|
| ASD (`DX_GROUP=1`) | 408 |
| Typically developing controls (`DX_GROUP=2`) | 476 |
| Acquisition sites | 20 |
| Age range | 6.47 – 64.0 years |
| Male / Female (`SEX=1/2`) | 746 / 138 |

**QC criteria (applied automatically by the download script):**

- `FILE_ID != "no_filename"` (preprocessed file exists)
- `func_mean_fd < 0.2` (mean framewise displacement < 0.2 mm; excludes excessive head motion)

The original phenotypic file has 1,112 rows; after the above QC, **884** subjects remain.

---

## 3. fMRI Data Hierarchy

To understand what counts as "raw fMRI," it helps to see the full processing chain:

```text
┌─────────────────────────────────────────────────────────────┐
│  Level 0: Scanner-level raw data                             │
│  DICOM / BIDS NIfTI, no software preprocessing               │
│  Source: ABIDE/INDI original release (❌ not in this repo)   │
└───────────────────────────┬─────────────────────────────────┘
                            │ Minimal preprocessing (motion, registration, skull-stripping…)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 1: func_minimal                                       │
│  Minimally preprocessed 4D resting-state fMRI volumes        │
│  Format: {FILE_ID}_func_minimal.nii.gz                       │
│  ~250 MB/subject, ~220 GB for 884 subjects (❌ not downloaded)│
└───────────────────────────┬─────────────────────────────────┘
                            │ Full preprocessing (band-pass filter, global signal regression…)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 2: func_preproc                                       │
│  Fully preprocessed 4D resting-state fMRI volumes            │
│  e.g. cpac/filt_global/func_preproc                          │
│  ~103 MB/subject, ~89 GB for 884 subjects (❌ not downloaded) │
└───────────────────────────┬─────────────────────────────────┘
                            │ Parcellation + signal extraction
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 3: rois_ho  ← ✅ currently downloaded                 │
│  Harvard-Oxford atlas regional BOLD time series              │
│  Format: {FILE_ID}_rois_ho.1D                                │
│  111 ROIs × ~196 timepoints, ~206 KB/subject                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ Correlation / partial correlation / etc.
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 4: Functional connectivity matrix / brain graph       │
│  Nodes = brain regions, edges = connectivity strength        │
│  Must be computed from rois_ho (❌ not precomputed here)      │
└───────────────────────────┬─────────────────────────────────┘
                            │ GNN / ML
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 5: Model outputs                                      │
│  ASD vs control classification, key regions, virtual intervention │
└─────────────────────────────────────────────────────────────┘
```

**Takeaways:**

- What you have is **Level 3 derived data** (regional time series), not 4D voxel-level imaging.
- It is **resting-state fMRI after preprocessing and ROI extraction**, ready for connectome construction and GNN modeling without running a full preprocessing pipeline yourself.
- For **voxel-level fMRI volumes**, download `func_minimal` or `func_preproc` separately (89–220 GB total).

---

## 4. File-Level Details

### 4.1 Phenotypic file — `phenotypic/Phenotypic_V1_0b_preprocessed1.csv`

**Download URL:**

```
https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
```

**Contents:** Original ABIDE phenotypes + PCP preprocessing QC metrics + manual QC annotations.

**Key fields:**

| Field | Meaning | Use |
|-------|---------|-----|
| `FILE_ID` | Unique file ID, e.g. `KKI_0050822` | Maps 1:1 to `rois_ho` filenames |
| `SUB_ID` | Subject ID | Deduplication, tracking |
| `SITE_ID` | Acquisition site, e.g. `KKI`, `Yale` | Site-effect analysis, domain generalization |
| `DX_GROUP` | `1` = ASD, `2` = healthy control | **Classification label** |
| `AGE_AT_SCAN` | Age at scan | Covariate, stratified analysis |
| `SEX` | `1` = male, `2` = female | Covariate |
| `FIQ` / `VIQ` / `PIQ` | IQ subscales | Cognitive covariates |
| `ADOS_*` / `ADI_R_*` | Autism diagnostic scales | Symptom severity regression |
| `func_mean_fd` | Mean framewise displacement | Head-motion QC (used in filtering) |
| `func_dvars` / `func_outlier` | Time-series quality metrics | Additional QC |
| `anat_*` / `func_*` | Structural and functional QA metrics | Exclude low-quality scans |
| `qc_*` | Manual QC ratings | Optional stricter filtering |
| `SUB_IN_SMP` | Included in Di Martino et al. 2014 sample | Align with published benchmarks |

**Use cases:**

- Match **ASD / control labels** to each `.1D` file
- Control for covariates (age, sex, IQ, head motion)
- Multi-site generalization (leave-one-site-out cross-validation)
- Symptom severity prediction (regress on ADOS scores, not just binary classification)

---

### 4.2 ROI time series — `raw/Outputs/cpac/filt_global/rois_ho/*.1D`

**Download command (official script):**

```bash
python3 scripts/download_abide_preproc.py \
  -d rois_ho -p cpac -s filt_global \
  -o data/abide/raw
```

**S3 URL template:**

```
https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_ho/{FILE_ID}_rois_ho.1D
```

**Example:**

```
https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_ho/KKI_0050822_rois_ho.1D
```

**Preprocessing parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pipeline | **C-PAC** | Configurable Pipeline for the Analysis of Connectomes |
| Strategy | **filt_global** | Band-pass filtering + global signal regression (GSR) |
| Atlas | **Harvard-Oxford (HO)** | 110 cortical + 1 other = **111 ROIs** |
| Source modality | Resting-state fMRI | Resting-state BOLD |

**File format:**

- Extension: `.1D` (AFNI-style plain-text matrix)
- Row 1: Tab-separated ROI labels (`#10`, `#11`, …, `#4802`)
- Rows 2+: One timepoint per row, one BOLD value per ROI per column
- Typical shape: **111 ROIs × ~196 timepoints** (varies slightly by site TR and scan duration)

**Example structure:**

```text
#10    #11    #12    ...    #4802
-3.45  7.08   0.21   ...    10.04     ← timepoint 1
-2.47  5.44  -4.55   ...    25.08     ← timepoint 2
...
```

**Use cases:**

| Scenario | How |
|----------|-----|
| **ASD vs control classification (GNN)** | Compute Pearson correlation from time series → 111×111 FC matrix → graph → GNN |
| **Key brain region identification** | GNN attention / GNNExplainer → map back to HO atlas regions |
| **Graph representation learning** | Node features = time-series statistics (mean, variance, ALFF, etc.); edges = connectivity |
| **Dynamic functional connectivity** | Sliding-window time series → temporal graphs / dynamic GNN |
| **Virtual node intervention** | Perturb node features or edge weights → re-run inference → observe probability shift |
| **Rapid prototyping** | Full pipeline runnable at ~178 MB without downloading GB-scale 4D data |

**Not suitable for:**

- Custom preprocessing (different filter strategy, no GSR) → download `func_preproc` or `func_minimal` and process yourself
- Voxel-level analysis (ICA, MVPA, 3D CNN) → requires 4D `.nii.gz`
- Alternative atlases (AAL, CC200, etc.) → download `rois_aal`, `rois_cc200`, or similar derivatives

---

## 5. Available but Not Downloaded

All of the following come from the same S3 bucket and can be downloaded on demand.

### 5.1 func_minimal — minimally preprocessed 4D fMRI

```
https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/{FILE_ID}_func_minimal.nii.gz
```

| Item | Description |
|------|-------------|
| Format | 4D NIfTI (`.nii.gz`) |
| Size | ~255 MB/subject; 884 subjects ≈ **220 GB** |
| Use case | Custom preprocessing, re-extract ROIs with a different atlas, voxel-level analysis, 3D deep learning |

Local script (supports resume): `scripts/download_func_minimal.py`

### 5.2 func_preproc — fully preprocessed 4D fMRI

```
https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/{FILE_ID}_func_preproc.nii.gz
```

| Item | Description |
|------|-------------|
| Size | ~103 MB/subject; 884 subjects ≈ **89 GB** |
| Use case | Same pipeline/strategy as `rois_ho`; reproduce PCP results; voxel-level ALFF/ReHo, etc. |

### 5.3 Other ROI atlas time series

Same URL structure; swap the derivative name:

| Derivative | Atlas | Use case |
|------------|-------|----------|
| `rois_aal` | AAL (116 ROIs) | Classic connectome analysis |
| `rois_cc200` | Craddock 200 | Widely used in FC studies |
| `rois_cc400` | Craddock 400 | Finer parcellation |
| `rois_dosenbach160` | Dosenbach 160 | Pediatric / adolescent networks |
| `rois_ez` | Eickhoff-Zilles | Cytoarchitectonic parcellation |
| `rois_tt` | Talairach-Tournoux | Classic anatomical atlas |

### 5.4 Graph-theory voxel maps

Derivatives such as `alff`, `reho`, `degree_weighted`, `eigenvector_weighted`, `vmhc` — each a 3D `.nii.gz` (~256 KB–5 MB/subject) computed directly from preprocessed fMRI.

### 5.5 Structural MRI derivatives

Cortical thickness and surface metrics from ANTS, CIVET, and FreeSurfer (see the official download page).

---

## 6. Directory Layout

```text
data/abide/
├── README.md                          ← this document
├── download.log                       ← rois_ho download log
├── download_func_minimal.log          ← func_minimal attempt log (stopped)
│
├── phenotypic/
│   └── Phenotypic_V1_0b_preprocessed1.csv
│
├── raw/
│   └── Outputs/
│       └── cpac/
│           └── filt_global/
│               └── rois_ho/
│                   ├── KKI_0050822_rois_ho.1D
│                   ├── Yale_0050607_rois_ho.1D
│                   └── ... (884 total)
│
└── scripts/
    ├── download_abide_preproc.py      ← official PCP script
    └── download_func_minimal.py       ← local func_minimal downloader (not completed)
```

---

## 7. File ID Alignment Example

Mapping between the phenotypic CSV and ROI files:

```text
Phenotypic CSV                    ROI time series file
─────────────────────────────────────────────────────────────
FILE_ID = KKI_0050822      →      KKI_0050822_rois_ho.1D
DX_GROUP = 1 (ASD)
SITE_ID = KKI
AGE_AT_SCAN = 10.2
func_mean_fd = 0.08
```

Python loading example:

```python
import csv
import numpy as np
from pathlib import Path

base = Path("data/abide")
pheno = {
    r["FILE_ID"]: r
    for r in csv.DictReader(open(base / "phenotypic/Phenotypic_V1_0b_preprocessed1.csv"))
}

file_id = "KKI_0050822"
label = int(pheno[file_id]["DX_GROUP"])  # 1=ASD, 2=control

ts = np.loadtxt(base / f"raw/Outputs/cpac/filt_global/rois_ho/{file_id}_rois_ho.1D")
# ts shape: (n_timepoints, n_rois)

fc = np.corrcoef(ts, rowvar=False)  # functional connectivity matrix (111, 111)
```

---

## 8. Mapping to This Project's Research Pipeline

Planned pipeline:

```text
Resting-state fMRI regional time series
       ↓
Functional connectivity → brain graph
       ↓
GNN embedded in an agent → ASD vs healthy classification
       ↓
Identify key brain regions
       ↓
LLM literature retrieval for evidence
       ↓
Virtual node modulation → test whether network state shifts toward normal
```

**What the current data supports:**

| Step | Data | Status |
|------|------|--------|
| Regional time series | `rois_ho/*.1D` | ✅ Available |
| Disease labels | `DX_GROUP` in CSV | ✅ Available |
| Functional connectivity / brain graph | Compute from `rois_ho` | Needs implementation |
| GNN classification | `rois_ho` + labels | ✅ Data ready |
| Key brain regions | GNN interpretability methods | ✅ Data ready |
| Virtual node intervention | In-silico graph perturbation | ✅ Data ready |
| Custom preprocessing | `func_minimal` / `func_preproc` | ❌ Requires separate download |

---

## 9. How to Download More

### Other derivatives (e.g. `rois_cc200`)

```bash
python3 data/abide/scripts/download_abide_preproc.py \
  -d rois_cc200 -p cpac -s filt_global \
  -o data/abide/raw
```

### ASD or controls only

```bash
# ASD only
python3 ... -a -d rois_ho -p cpac -s filt_global -o ...

# Controls only
python3 ... -c -d rois_ho -p cpac -s filt_global -o ...
```

### 4D fMRI volumes (requires large disk)

```bash
# func_minimal, ~220 GB, supports resume
python3 data/abide/scripts/download_func_minimal.py data/abide/raw
```

---

## 10. Citations

If you use ABIDE Preprocessed data, please cite:

> Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). **The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives.** Neuroinformatics 2013, Stockholm, Sweden.

For the original ABIDE dataset:

> Adriana Di Martino et al. (2014). **The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism.** Molecular Psychiatry, 19(6), 659–667.

---

## 11. FAQ

**Q: Is this raw fMRI data?**  
A: No. It is **regional time series extracted from preprocessed resting-state fMRI**. For raw 4D volumes, download `func_minimal`/`func_preproc`; for scanner-level raw data, use the ABIDE/INDI original release.

**Q: Why 884 subjects instead of 1,112?**  
A: The download script excludes subjects with `no_filename` (no preprocessed file) and `func_mean_fd >= 0.2` (excessive head motion).

**Q: What does `filt_global` mean?**  
A: C-PAC preprocessing with **band-pass filtering (filt)** and **global signal regression (global)**. This is one of the most common ABIDE strategies, though GSR remains debated; alternative strategies (`filt_noglobal`, `nofilt_global`, etc.) can be downloaded for comparison.

**Q: What are the 111 ROIs?**  
A: Harvard-Oxford cortical and subcortical atlas regions. ROI label definitions: http://preprocessed-connectomes-project.org/abide/Pipelines.html#regions_of_interest

**Q: Should this data be pushed to GitHub?**  
A: Not recommended. The 884 `.1D` files total ~178 MB, and ABIDE data use is subject to its data-use terms. Prefer adding `data/abide/raw/` to `.gitignore` and keeping only this documentation and download scripts in version control.
