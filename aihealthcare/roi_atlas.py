"""Harvard-Oxford ROI names for ABIDE `rois_ho` (111 parcels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"

# FreeSurfer aseg codes used by C-PAC for HO subcortical parcels.
# (full name, hemisphere, network, abbreviation)
_SUBCORTICAL = {
    10: ("Left Thalamus", "L", "Subcortical", "THA"),
    11: ("Left Caudate", "L", "Subcortical", "CAU"),
    12: ("Left Putamen", "L", "Subcortical", "PUT"),
    13: ("Left Pallidum", "L", "Subcortical", "PAL"),
    17: ("Left Hippocampus", "L", "Limbic", "HIP"),
    18: ("Left Amygdala", "L", "Limbic", "AMYG"),
    26: ("Left Accumbens", "L", "Subcortical", "NAC"),
    49: ("Right Thalamus", "R", "Subcortical", "THA"),
    50: ("Right Caudate", "R", "Subcortical", "CAU"),
    51: ("Right Putamen", "R", "Subcortical", "PUT"),
    52: ("Right Pallidum", "R", "Subcortical", "PAL"),
    53: ("Right Hippocampus", "R", "Limbic", "HIP"),
    54: ("Right Amygdala", "R", "Limbic", "AMYG"),
    58: ("Right Accumbens", "R", "Subcortical", "NAC"),
}

# FSL Harvard-Oxford cortical atlas, regions 1–48.
# (full name, network, abbreviation)
_CORTICAL = {
    1: ("Frontal Pole", "Frontoparietal", "FP"),
    2: ("Insular Cortex", "Salience", "INS"),
    3: ("Superior Frontal Gyrus", "Frontoparietal", "SFG"),
    4: ("Middle Frontal Gyrus", "Frontoparietal", "MFG"),
    5: ("IFG pars triangularis", "Language", "IFGtr"),
    6: ("IFG pars opercularis", "Language", "IFGop"),
    7: ("Precentral Gyrus", "Somatomotor", "PreCG"),
    8: ("Temporal Pole", "Limbic", "TP"),
    9: ("STG anterior", "Auditory", "aSTG"),
    10: ("STG posterior", "Auditory", "pSTG"),
    11: ("MTG anterior", "Default", "aMTG"),
    12: ("MTG posterior", "Default", "pMTG"),
    13: ("MTG temporooccipital", "Default", "toMTG"),
    14: ("ITG anterior", "Limbic", "aITG"),
    15: ("ITG posterior", "Limbic", "pITG"),
    16: ("ITG temporooccipital", "Visual", "toITG"),
    17: ("Postcentral Gyrus", "Somatomotor", "PostCG"),
    18: ("Superior Parietal Lobule", "DorsalAttention", "SPL"),
    19: ("SMG anterior", "Salience", "aSMG"),
    20: ("SMG posterior", "Salience", "pSMG"),
    21: ("Angular Gyrus", "Default", "ANG"),
    22: ("LOC superior", "Visual", "sLOC"),
    23: ("LOC inferior", "Visual", "iLOC"),
    24: ("Intracalcarine Cortex", "Visual", "CALC"),
    25: ("Frontal Medial Cortex", "Default", "mPFC"),
    26: ("SMA / Juxtapositional", "Somatomotor", "SMA"),
    27: ("Subcallosal Cortex", "Limbic", "SCC"),
    28: ("Paracingulate Gyrus", "Salience", "PaCiG"),
    29: ("Cingulate anterior", "Salience", "ACC"),
    30: ("Cingulate posterior", "Default", "PCC"),
    31: ("Precuneous Cortex", "Default", "PCUN"),
    32: ("Cuneal Cortex", "Visual", "CUN"),
    33: ("Frontal Orbital Cortex", "Limbic", "OFC"),
    34: ("Parahippocampal anterior", "Limbic", "aPHG"),
    35: ("Parahippocampal posterior", "Limbic", "pPHG"),
    36: ("Lingual Gyrus", "Visual", "LING"),
    37: ("Temporal Fusiform anterior", "Limbic", "aTFus"),
    38: ("Temporal Fusiform posterior", "Limbic", "pTFus"),
    39: ("Temporal Occipital Fusiform", "Visual", "TOFus"),
    40: ("Occipital Fusiform Gyrus", "Visual", "OFus"),
    41: ("Frontal Operculum", "Salience", "FOpe"),
    42: ("Central Opercular Cortex", "Somatomotor", "COpe"),
    43: ("Parietal Operculum", "Somatomotor", "POpe"),
    44: ("Planum Polare", "Auditory", "PP"),
    45: ("Heschl's Gyrus", "Auditory", "HES"),
    46: ("Planum Temporale", "Auditory", "PT"),
    47: ("Supracalcarine Cortex", "Visual", "SCLC"),
    48: ("Occipital Pole", "Visual", "OP"),
}

# C-PAC inserts one extra parcel id between HO 34 and 35.
_EXTRA = {
    3455: ("HO extra parcel 3455", "L", "Limbic", "X"),
}

# Ring order on the left hemisphere, clockwise from 12 o'clock.
# The right hemisphere is the reverse, so the same network meets at 6 o'clock:
#   … SCN — SMN — FPN | FPN — SMN — SCN …
# That is the usual connectome chord layout (CCN/MN/SCN in the reference figure).
NETWORK_COLORS = {
    "Visual": "#E07A3D",
    "Auditory": "#C9A227",
    "Limbic": "#8FBC8F",
    "Default": "#2A9D8F",
    "Salience": "#C75B7A",
    "DorsalAttention": "#6B8E23",
    "Language": "#E09F3E",
    "Subcortical": "#3D5A80",
    "Somatomotor": "#5B8FA8",
    "Frontoparietal": "#7A5EA8",
    "Unknown": "#888888",
}

NETWORK_ABBR = {
    "Visual": "VN",
    "Somatomotor": "SMN",
    "Auditory": "AN",
    "Salience": "SN",
    "DorsalAttention": "DAN",
    "Frontoparietal": "FPN",
    "Default": "DMN",
    "Language": "LN",
    "Limbic": "BLN",
    "Subcortical": "SCN",
    "Unknown": "?",
}


def read_roi_ids(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[int]:
    """Read the 111 HO ids from the first `.1D` header in the raw download."""
    roi_dir = Path(data_dir) / "raw" / "Outputs" / "cpac" / "filt_global" / "rois_ho"
    files = sorted(roi_dir.glob("*_rois_ho.1D"))
    if not files:
        raise FileNotFoundError(f"No rois_ho .1D files under {roi_dir}")
    header = files[0].read_text(encoding="utf-8").splitlines()[0]
    return [int(token.lstrip("#")) for token in header.split() if token]


def describe_roi(roi_id: int) -> dict[str, str | int]:
    if roi_id in _SUBCORTICAL:
        name, hemi, network, abbr = _SUBCORTICAL[roi_id]
    elif roi_id in _EXTRA:
        name, hemi, network, abbr = _EXTRA[roi_id]
    else:
        region = roi_id // 100
        suffix = roi_id % 100
        hemi = "L" if suffix == 1 else "R" if suffix == 2 else "U"
        label, network, abbr = _CORTICAL.get(
            region, (f"HO {roi_id}", "Unknown", str(roi_id))
        )
        name = f"{hemi} {label}" if hemi in {"L", "R"} else label
    token = f"{hemi}.{abbr}" if hemi in {"L", "R"} else abbr
    return {
        "roi_id": roi_id,
        "name": name,
        "hemi": hemi,
        "network": network,
        "network_abbr": NETWORK_ABBR.get(network, "?"),
        "abbr": token,
        "short": str(name).replace("Left ", "L ").replace("Right ", "R "),
    }


def roi_table(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[dict[str, str | int]]:
    rows = []
    for index, roi_id in enumerate(read_roi_ids(data_dir)):
        row = describe_roi(roi_id)
        row["index"] = index
        rows.append(row)
    return rows


def sort_order(table: list[dict[str, str | int]]) -> np.ndarray:
    """Mirror layout: left half clockwise, right half reversed.

    Same networks touch at 6 o'clock (FPN next to FPN, then SMN, then SCN),
    instead of dumping every left ROI and then every right ROI.
    """
    network_rank = {name: rank for rank, name in enumerate(NETWORK_COLORS)}

    def key(row: dict[str, str | int]) -> tuple:
        rank = network_rank.get(str(row["network"]), 99)
        if row["hemi"] == "L":
            return (0, rank, int(row["index"]))
        if row["hemi"] == "R":
            return (1, -rank, int(row["index"]))
        return (2, rank, int(row["index"]))

    return np.array([row["index"] for row in sorted(table, key=key)], dtype=int)
