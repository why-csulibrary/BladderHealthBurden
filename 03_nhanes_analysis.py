# -*- coding: utf-8 -*-
r"""
NHANES bladder-health burden: Part 3 biomarker contextualization + Part 4 short-term / mortality analyses.

Scope
-----
1) Reads local NHANES XPT files arranged as:
   data/nhanes/1999-2000/Demographics/*.xpt
   data/nhanes/1999-2000/Questionnaire/*.xpt
   data/nhanes/1999-2000/Examination/*.xpt
   data/nhanes/1999-2000/Laboratory/*.xpt
   ...
2) Builds a supportive NHANES bladder-health burden phenotype.
3) Runs survey-weighted biomarker gradient models.
4) Runs survey-weighted short-term health/service-use burden models.
5) Runs optional linked-mortality Cox models if a public-use mortality file is found.

No figures are produced. Outputs are CSV only. No Excel workbook is produced.

Important interpretation
------------------------
NHANES is used here as a biomarker / short-term burden / mortality contextualization cohort.
It is not pooled with HRS/KLoSA/MHAS/LASI/ELSA/SHARE, and it is not used for cross-country comparison.
"""

from __future__ import annotations

import os
import json
import logging
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =========================================================
# 0) Project paths and user settings
# =========================================================
# Place NHANES XPT files under data/nhanes, or set BHB_NHANES_DATA.
# NHANES outputs are written to outputs/nhanes unless BHB_OUTPUT_ROOT is set.
PROJECT_ROOT = Path(__file__).resolve().parent
NHANES_ROOT = Path(os.environ.get("BHB_NHANES_DATA", str(PROJECT_ROOT / "data" / "nhanes")))
OUT_ROOT = Path(os.environ.get("BHB_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))) / "nhanes"
RUN_DIR = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

# Main analysis uses late-life participants.
AGE_MIN = 60

# Main NHANES continuous cycles.
# For 2017-March 2020, NHANES released a special pre-pandemic combined file.
# This script uses it when P_*.xpt files are found and replaces the ordinary 2017-2018 cycle
# to avoid double counting. It does not analyze the incomplete 2019-March 2020 sample alone.
USE_2017_MARCH2020_PREPANDEMIC = True

# Skip very large repeated-minute accelerometer files that are irrelevant to this study and
# can exhaust memory when read from XPT.
SKIP_XPT_PREFIXES = ["PAXMIN", "PAXDAY", "PAXHD"]

# Use examination weights for KIQ005-KIQ480 and biomarker analyses.
WEIGHT_CANDIDATES = ["WTMEC2YR", "WTMECPRP"]

# Public-use NHANES linked mortality DAT files can be stored in a dedicated folder
# such as D:\\科研\\NHANES\\NHANES生存数据. The script searches these folders first.
MORTALITY_SEARCH_DIRS = [
    NHANES_ROOT / "NHANES生存数据",
    NHANES_ROOT / "mortality",
    NHANES_ROOT / "Mortality",
    NHANES_ROOT,
]

MIN_TOTAL_N_FOR_MODEL = 100
MIN_EVENTS_FOR_LOGISTIC = 10
MIN_DEATHS_FOR_COX = 20

SCRIPT_VERSION = "NHANES_parts3_4_V11_2_dual_burden_interaction_added"

RUN_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(RUN_DIR / "run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# =========================================================
# 1) Variable rules
# =========================================================
COMPONENTS = ["Demographics", "Questionnaire", "Examination", "Laboratory"]

PREFIX_KEEP = {
    "Questionnaire": [
        "KIQ", "HUQ", "HSD", "DPQ", "PFQ", "SLQ", "DIQ", "BPQ", "SMQ", "MCQ", "ALQ", "PAQ"
    ],
    "Examination": ["BMX"],
    "Laboratory": [
        "LBX", "LBD", "URX", "URD", "LBDH", "LBDL", "LBDT", "LBDG", "LBDI"
    ],
}

DEMOGRAPHIC_KEEP = {
    "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "DMDEDUC3",
    "DMDMARTL", "INDFMPIR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR", "WTMECPRP",
    "WTINTPRP", "WTSAF2YR", "WTSAFPRP", "WTDRD1", "WTDR2D"
}

MODEL_SPECS = {
    "M1_age_sex": ["age", "sex"],
    "M2_sociodemographic": ["age", "sex", "race_ethnicity", "education", "poverty_ratio", "smoking"],
}

BIOMARKER_OUTCOMES = {
    "egfr": "linear",
    "uacr_log": "linear",
    "ckd": "logistic",
    "hba1c": "linear",
    "diabetes_biomarker": "logistic",
    "bmi": "linear",
    "waist": "linear",
    "crp_log": "linear",
    "high_crp": "logistic",
    "albumin": "linear",
    "hemoglobin": "linear",
    "biomarker_vulnerability_high": "logistic",
}

SHORT_TERM_OUTCOMES = {
    "poor_self_rated_health": "logistic",
    "high_healthcare_use": "logistic",
    "any_healthcare_use": "logistic",
    "depressive_symptoms": "logistic",
    "physical_limitation": "logistic",
    "sleep_problem": "logistic",
}

# Dual-burden module: combines the bladder-health burden phenotype with
# biomarker vulnerability to identify a joint high-risk group.
DUAL_BURDEN_LEVELS = [
    "Low burden",
    "Bladder-only",
    "Biomarker-only",
    "Dual burden",
]

DUAL_BURDEN_SHORT_TERM_OUTCOMES = {
    "poor_self_rated_health": "logistic",
    "high_healthcare_use": "logistic",
    "sleep_problem": "logistic",
}

# =========================================================
# 2) General helpers
# =========================================================
def normalize_colname(x: str) -> str:
    return re.sub(r"\s+", "", str(x).strip().upper())


def clean_special_codes(s: pd.Series, extra: Sequence[float] = ()) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    special = set([7, 9, 77, 99, 777, 999, 7777, 9999, 6666, 8888]) | set(extra)
    out = out.mask(out.isin(special), np.nan)
    return out


def recode_yes_no(s: pd.Series) -> pd.Series:
    x = clean_special_codes(s)
    return pd.Series(np.where(x == 1, 1, np.where(x == 2, 0, np.nan)), index=s.index, dtype="float")


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coalesce_numeric(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Row-wise first non-missing numeric value across candidate columns."""
    out = pd.Series(np.nan, index=df.index, dtype="float")
    for c in candidates:
        if c in df.columns:
            x = clean_special_codes(df[c])
            out = out.where(out.notna(), x)
    return out


def coalesce_category(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Row-wise first non-missing categorical code across candidate columns.

    NHANES categorical codes are usually numeric, but some imported columns can contain
    non-integer float values or mixed encodings. Avoid pandas nullable-integer casting
    because it can raise: cannot safely cast non-equivalent float64 to int64.
    """
    x = coalesce_numeric(df, candidates)

    def fmt_category(v):
        if pd.isna(v):
            return np.nan
        try:
            fv = float(v)
            if np.isfinite(fv) and abs(fv - round(fv)) < 1e-8:
                return str(int(round(fv)))
            return str(fv)
        except Exception:
            return str(v).strip()

    return x.map(fmt_category)


def combine_binary_max(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype="float")
    arr = df[available].copy()
    return arr.max(axis=1, skipna=True)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def safe_divide(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    return a / b


def cycle_start_year(cycle_name: str) -> int:
    m = re.match(r"(\d{4})", str(cycle_name))
    return int(m.group(1)) if m else 9999


@dataclass
class CycleSpec:
    label: str
    path: Path
    p_file_only: bool = False
    represented_years: float = 2.0
    note: str = "standard_2yr"


def is_prepandemic_file(path: Path) -> bool:
    return path.stem.upper().startswith("P_")


def should_skip_xpt(path: Path) -> bool:
    stem = path.stem.upper()
    return any(stem.startswith(prefix) for prefix in SKIP_XPT_PREFIXES)


def list_xpt_files(directory: Path) -> List[Path]:
    """Return unique XPT files in a directory.

    On Windows, Path.glob("*.xpt") can match upper-case .XPT files, so using
    glob("*.xpt") + glob("*.XPT") may read the same file twice. This helper
    avoids duplicated reads and skips non-files.
    """
    if not directory.exists():
        return []
    files = []
    seen = set()
    for item in directory.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() != ".xpt":
            continue
        key = str(item.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        files.append(item)
    return sorted(files, key=lambda x: x.name.upper())


def find_cycle_specs(root: Path) -> List[CycleSpec]:
    all_dirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{4}", p.name)]
    if not all_dirs:
        return []

    # Search for the 2017-March 2020 pre-pandemic public-use files. They are usually named P_*.xpt.
    prep_sources = []
    if USE_2017_MARCH2020_PREPANDEMIC:
        for p in all_dirs:
            if p.name in {"2017-2018", "2017-2020", "2019-2020"}:
                n_p = 0
                for component in COMPONENTS:
                    comp_dir = p / component
                    if comp_dir.exists():
                        n_p += len([x for x in list_xpt_files(comp_dir) if is_prepandemic_file(x)])
                if n_p > 0:
                    prep_sources.append((n_p, p))

    selected_prep_dir = None
    if prep_sources:
        selected_prep_dir = sorted(prep_sources, key=lambda x: (-x[0], cycle_start_year(x[1].name)))[0][1]
        logger.info("Using NHANES 2017-March 2020 pre-pandemic source: %s", selected_prep_dir)

    specs: List[CycleSpec] = []
    for p in all_dirs:
        if selected_prep_dir is not None:
            # Replace ordinary 2017-2018 by the combined pre-pandemic release.
            if p.name == "2017-2018":
                continue
            # Do not analyze incomplete 2019-March 2020 alone.
            if p.name in {"2017-2020", "2019-2020"}:
                continue
        else:
            # No pre-pandemic public-use files found; skip incomplete 2019-2020 / 2017-2020 folders.
            if p.name in {"2017-2020", "2019-2020"}:
                logger.warning("Skipping %s because no P_*.xpt pre-pandemic files were found for a valid combined release.", p)
                continue
        specs.append(CycleSpec(label=p.name, path=p, p_file_only=False, represented_years=2.0, note="standard_2yr"))

    if selected_prep_dir is not None:
        specs.append(CycleSpec(
            label="2017-2020",
            path=selected_prep_dir,
            p_file_only=True,
            represented_years=3.2,
            note="2017_March2020_prepandemic"
        ))

    return sorted(specs, key=lambda spec: cycle_start_year(spec.label))


def read_xpt(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_sas(path, format="xport", encoding="latin1")
    except TypeError:
        df = pd.read_sas(path, format="xport")
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def keep_relevant_columns(df: pd.DataFrame, component: str) -> pd.DataFrame:
    if "SEQN" not in df.columns:
        return pd.DataFrame()
    keep = {"SEQN"}
    if component == "Demographics":
        keep |= set(c for c in df.columns if c in DEMOGRAPHIC_KEEP)
    else:
        prefixes = PREFIX_KEEP.get(component, [])
        keep |= set(c for c in df.columns if any(c.startswith(prefix) for prefix in prefixes))
        # Always keep common design variables if they appear outside Demographics.
        keep |= set(c for c in df.columns if c in DEMOGRAPHIC_KEEP)
    keep = [c for c in df.columns if c in keep]
    return df[keep].copy()


def first_non_missing(s: pd.Series):
    x = s.dropna()
    return x.iloc[0] if len(x) else np.nan


def collapse_to_one_row_per_seqn(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "SEQN" not in df.columns:
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    df = df[df["SEQN"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    if not df["SEQN"].duplicated().any():
        return df
    value_cols = [c for c in df.columns if c != "SEQN"]
    if not value_cols:
        return df[["SEQN"]].drop_duplicates()
    return df.groupby("SEQN", as_index=False, sort=False)[value_cols].agg(first_non_missing)


def merge_on_seqn(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [collapse_to_one_row_per_seqn(f) for f in frames if f is not None and not f.empty and "SEQN" in f.columns]
    frames = [f for f in frames if f is not None and not f.empty and "SEQN" in f.columns]
    if not frames:
        return pd.DataFrame()

    used_cols = set(["SEQN"])
    indexed = []
    for df in frames:
        df = df.loc[:, ~df.columns.duplicated()].copy()
        keep_cols = ["SEQN"] + [c for c in df.columns if c != "SEQN" and c not in used_cols]
        if len(keep_cols) <= 1:
            continue
        used_cols.update(keep_cols)
        indexed.append(df[keep_cols].set_index("SEQN"))

    if not indexed:
        return frames[0][["SEQN"]].drop_duplicates().copy()

    base = pd.concat(indexed, axis=1, join="outer", copy=False).reset_index()
    return base

# =========================================================
# 3) Load NHANES XPT files
# =========================================================
def load_all_nhanes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests = []
    inventories = []
    cycle_frames = []

    cycle_specs = find_cycle_specs(NHANES_ROOT)
    if not cycle_specs:
        raise FileNotFoundError(f"No NHANES cycle folders found under {NHANES_ROOT}")

    for spec in cycle_specs:
        cycle_dir = spec.path
        cycle = spec.label
        logger.info("Loading NHANES cycle %s from %s | %s", cycle, cycle_dir, spec.note)
        frames = []
        for component in COMPONENTS:
            comp_dir = cycle_dir / component
            if not comp_dir.exists():
                continue
            for xpt in list_xpt_files(comp_dir):
                if should_skip_xpt(xpt):
                    manifests.append({
                        "cycle": cycle, "component": component, "file": str(xpt),
                        "status": "skipped_large_repeated_file", "error": "",
                        "n_rows": np.nan, "n_cols": np.nan,
                        "cycle_note": spec.note, "represented_years": spec.represented_years,
                    })
                    continue
                if spec.p_file_only and not is_prepandemic_file(xpt):
                    manifests.append({
                        "cycle": cycle, "component": component, "file": str(xpt),
                        "status": "skipped_replaced_by_prepandemic", "error": "",
                        "n_rows": np.nan, "n_cols": np.nan,
                        "cycle_note": spec.note, "represented_years": spec.represented_years,
                    })
                    continue
                if (not spec.p_file_only) and is_prepandemic_file(xpt):
                    manifests.append({
                        "cycle": cycle, "component": component, "file": str(xpt),
                        "status": "skipped_prepandemic_file_in_standard_cycle", "error": "",
                        "n_rows": np.nan, "n_cols": np.nan,
                        "cycle_note": spec.note, "represented_years": spec.represented_years,
                    })
                    continue
                try:
                    raw = read_xpt(xpt)
                except Exception as exc:
                    logger.warning("Read failed | cycle=%s | component=%s | file=%s | error=%s", cycle, component, xpt.name, repr(exc))
                    manifests.append({
                        "cycle": cycle, "component": component, "file": str(xpt),
                        "status": "read_failed", "error": str(exc),
                        "n_rows": np.nan, "n_cols": np.nan,
                        "cycle_note": spec.note, "represented_years": spec.represented_years,
                    })
                    continue

                inv = pd.DataFrame({
                    "cycle": cycle,
                    "component": component,
                    "file": xpt.name,
                    "column": raw.columns,
                    "non_missing": [raw[c].notna().sum() for c in raw.columns],
                    "dtype": [str(raw[c].dtype) for c in raw.columns],
                })
                inventories.append(inv)

                kept = keep_relevant_columns(raw, component)
                kept_rows_before = kept.shape[0]
                kept_dup_seqn = int(kept["SEQN"].duplicated().sum()) if (not kept.empty and "SEQN" in kept.columns) else 0
                kept = collapse_to_one_row_per_seqn(kept)
                frames.append(kept)
                manifests.append({
                    "cycle": cycle, "component": component, "file": str(xpt),
                    "status": "ok", "error": "", "n_rows": raw.shape[0],
                    "n_cols": raw.shape[1], "kept_cols": kept.shape[1],
                    "cycle_note": spec.note, "represented_years": spec.represented_years,
                    "kept_rows_before_collapse": kept_rows_before,
                    "kept_rows_after_collapse": kept.shape[0],
                    "duplicate_seqn_before_collapse": kept_dup_seqn,
                })

        cdf = merge_on_seqn(frames)
        if cdf.empty:
            logger.warning("No usable data for cycle %s", cycle)
            continue
        cdf["cycle"] = cycle
        cdf["cycle_start"] = cycle_start_year(cycle)
        cdf["cycle_note"] = spec.note
        cdf["represented_years"] = spec.represented_years
        cycle_frames.append(cdf)
        logger.info("Cycle %s merged shape=%s", cycle, cdf.shape)

    if not cycle_frames:
        raise RuntimeError("No usable NHANES cycle data were loaded.")

    data = pd.concat(cycle_frames, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifests)
    inventory = pd.concat(inventories, ignore_index=True, sort=False) if inventories else pd.DataFrame()
    return data, manifest, inventory

# =========================================================
# 4) Phenotype and biomarkers
# =========================================================
def derive_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["age"] = clean_special_codes(out["RIDAGEYR"]) if "RIDAGEYR" in out else np.nan
    if "RIAGENDR" in out:
        sex_raw = clean_special_codes(out["RIAGENDR"])
        out["sex"] = np.where(sex_raw == 1, "Male", np.where(sex_raw == 2, "Female", np.nan))
        out["female"] = np.where(sex_raw == 2, 1, np.where(sex_raw == 1, 0, np.nan))
    else:
        out["sex"] = np.nan
        out["female"] = np.nan

    # RIDRETH3 is unavailable before 2011-2012; use RIDRETH1 as row-wise fallback.
    out["race_ethnicity"] = coalesce_category(out, ["RIDRETH3", "RIDRETH1"])

    # DMDEDUC2 and DMDEDUC3 have different age ranges/cycles; use row-wise fallback.
    out["education"] = coalesce_category(out, ["DMDEDUC2", "DMDEDUC3"])

    out["poverty_ratio"] = clean_special_codes(out["INDFMPIR"]) if "INDFMPIR" in out else np.nan

    if "SMQ020" in out:
        out["smoking"] = np.where(clean_special_codes(out["SMQ020"]) == 1, "Ever", np.where(clean_special_codes(out["SMQ020"]) == 2, "Never", np.nan))
    else:
        out["smoking"] = np.nan

    out["diabetes_self_report"] = recode_yes_no(out["DIQ010"]) if "DIQ010" in out else np.nan
    out["hypertension_self_report"] = recode_yes_no(out["BPQ020"]) if "BPQ020" in out else np.nan

    return out


def derive_bladder_phenotype(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Core leakage frequency: KIQ005 in modern NHANES has 1=Never, 2=Less than once/month,
    # 3=few times/month, 4=few times/week, 5=every day/night.
    if "KIQ005" in out:
        kiq005 = clean_special_codes(out["KIQ005"])
        out["any_ui_from_kiq005"] = np.where(kiq005.isin([2, 3, 4, 5]), 1, np.where(kiq005 == 1, 0, np.nan))
        out["frequent_ui_from_kiq005"] = np.where(kiq005.isin([4, 5]), 1, np.where(kiq005.isin([1, 2, 3]), 0, np.nan))
        # Sandvik frequency component uses 1-4 among non-never responses.
        out["ui_frequency_component"] = np.where(kiq005.isin([2, 3, 4, 5]), kiq005 - 1, np.nan)
    else:
        out["any_ui_from_kiq005"] = np.nan
        out["frequent_ui_from_kiq005"] = np.nan
        out["ui_frequency_component"] = np.nan

    if "KIQ010" in out:
        kiq010 = clean_special_codes(out["KIQ010"])
        out["ui_amount_component"] = np.where(kiq010.isin([1, 2, 3]), kiq010, np.nan)
        out["large_amount_ui"] = np.where(kiq010.isin([2, 3]), 1, np.where(kiq010 == 1, 0, np.nan))
    else:
        out["ui_amount_component"] = np.nan
        out["large_amount_ui"] = np.nan

    out["incontinence_severity_index"] = out["ui_frequency_component"] * out["ui_amount_component"]
    out["isi_moderate_or_worse"] = np.where(
        out["incontinence_severity_index"] >= 3,
        1,
        np.where(out["incontinence_severity_index"].notna(), 0, np.nan),
    )

    out["stress_like_ui"] = recode_yes_no(out["KIQ042"]) if "KIQ042" in out else np.nan
    out["urgency_like_ui"] = recode_yes_no(out["KIQ044"]) if "KIQ044" in out else np.nan
    out["other_ui"] = recode_yes_no(out["KIQ046"]) if "KIQ046" in out else np.nan

    # Frequency of subtype leakage. Coding differs by era:
    # KIQ430/450/470: 1 less than monthly ... 4 daily/night.
    # KIQ043/045/047 in older files may use 1 daily ... 4 yearly.
    modern_freq = []
    for c in ["KIQ430", "KIQ450", "KIQ470"]:
        if c in out:
            x = clean_special_codes(out[c])
            modern_freq.append(pd.Series(np.where(x.isin([3, 4]), 1, np.where(x.isin([1, 2]), 0, np.nan)), index=out.index))
    old_freq = []
    for c in ["KIQ043", "KIQ045", "KIQ047"]:
        if c in out:
            x = clean_special_codes(out[c])
            old_freq.append(pd.Series(np.where(x.isin([1, 2]), 1, np.where(x.isin([3, 4]), 0, np.nan)), index=out.index))
    freq_cols = modern_freq + old_freq
    out["frequent_subtype_ui"] = pd.concat(freq_cols, axis=1).max(axis=1, skipna=True) if freq_cols else np.nan

    if "KIQ050" in out:
        bother = clean_special_codes(out["KIQ050"])
        out["high_ui_bother"] = np.where(bother >= 4, 1, np.where(bother.isin([1, 2, 3]), 0, np.nan))
    else:
        out["high_ui_bother"] = np.nan

    if "KIQ052" in out:
        affected = clean_special_codes(out["KIQ052"])
        out["high_daily_activity_impact"] = np.where(affected >= 4, 1, np.where(affected.isin([1, 2, 3]), 0, np.nan))
    else:
        out["high_daily_activity_impact"] = np.nan

    if "KIQ480" in out:
        noct = clean_special_codes(out["KIQ480"])
        out["nocturia_2plus"] = np.where(noct >= 2, 1, np.where(noct.isin([0, 1]), 0, np.nan))
        out["nocturia_3plus"] = np.where(noct >= 3, 1, np.where(noct.isin([0, 1, 2]), 0, np.nan))
    else:
        out["nocturia_2plus"] = np.nan
        out["nocturia_3plus"] = np.nan

    out["any_ui"] = combine_binary_max(out, ["any_ui_from_kiq005", "stress_like_ui", "urgency_like_ui", "other_ui"])
    out["frequent_ui"] = combine_binary_max(out, ["frequent_ui_from_kiq005", "frequent_subtype_ui"])
    out["any_bladder_marker"] = combine_binary_max(out, ["any_ui", "nocturia_2plus"])
    # Severity-oriented high-burden marker. Stress-like/urgency-like indicators are used to
    # define any UI/subtype, but not by themselves to define Tier 2 in NHANES. This avoids
    # overclassifying subtype-only UI as high burden.
    out["high_bladder_burden_marker"] = combine_binary_max(out, [
        "frequent_ui", "isi_moderate_or_worse",
        "high_ui_bother", "high_daily_activity_impact", "nocturia_3plus"
    ])

    tier = pd.Series(np.nan, index=out.index, dtype="object")
    tier = tier.mask(out["any_bladder_marker"] == 0, "Tier 0")
    tier = tier.mask((out["any_bladder_marker"] == 1) & (out["high_bladder_burden_marker"] != 1), "Tier 1")
    tier = tier.mask(out["high_bladder_burden_marker"] == 1, "Tier 2")
    out["nhanes_bladder_tier"] = pd.Categorical(tier, categories=["Tier 0", "Tier 1", "Tier 2"], ordered=True)
    out["tier2_binary"] = np.where(tier == "Tier 2", 1, np.where(tier.isin(["Tier 0", "Tier 1"]), 0, np.nan))

    return out


def derive_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Creatinine: use row-wise fallback because some cycles have LBDSCR/LB2SCR or SI units
    # while LBXSCR may exist globally but be missing within a cycle.
    scr_mgdl = coalesce_numeric(out, ["LBXSCR", "LBDSCR", "LB2SCR"])
    scr_si = coalesce_numeric(out, ["LBDSCRSI", "LB2SCRSI"]) / 88.4
    scr = scr_mgdl.where(scr_mgdl.notna(), scr_si)
    out["serum_creatinine_mg_dl"] = scr

    age = pd.to_numeric(out.get("age"), errors="coerce")
    female = pd.to_numeric(out.get("female"), errors="coerce")
    kappa = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    egfr = 142 * np.minimum(scr / kappa, 1) ** alpha * np.maximum(scr / kappa, 1) ** (-1.200) * (0.9938 ** age) * np.where(female == 1, 1.012, 1.0)
    out["egfr"] = np.where((scr > 0) & age.notna() & pd.Series(female).notna(), egfr, np.nan)

    if "URXUMA" in out and "URXUCR" in out:
        urine_albumin = clean_special_codes(out["URXUMA"])
        urine_creat = clean_special_codes(out["URXUCR"])
        # URXUMA mg/L, URXUCR mg/dL => mg/g = mg/L / mg/dL * 100.
        out["uacr"] = np.where(urine_creat > 0, urine_albumin / urine_creat * 100, np.nan)
    else:
        out["uacr"] = np.nan
    out["uacr_log"] = np.log(out["uacr"].where(out["uacr"] > 0))
    out["albuminuria"] = np.where(out["uacr"] >= 30, 1, np.where(out["uacr"].notna(), 0, np.nan))
    out["ckd"] = np.where((out["egfr"] < 60) | (out["albuminuria"] == 1), 1, np.where(out["egfr"].notna() | out["albuminuria"].notna(), 0, np.nan))

    out["hba1c"] = clean_special_codes(out["LBXGH"]) if "LBXGH" in out else np.nan
    out["diabetes_biomarker"] = np.where(
        (out["diabetes_self_report"] == 1) | (out["hba1c"] >= 6.5),
        1,
        np.where((out["diabetes_self_report"] == 0) | out["hba1c"].notna(), 0, np.nan),
    )

    out["bmi"] = clean_special_codes(out["BMXBMI"]) if "BMXBMI" in out else np.nan
    out["waist"] = clean_special_codes(out["BMXWAIST"]) if "BMXWAIST" in out else np.nan
    out["obesity"] = np.where(out["bmi"] >= 30, 1, np.where(out["bmi"].notna(), 0, np.nan))

    # hs-CRP is mg/L; older standard CRP is often mg/dL, converted to mg/L.
    # Use row-wise fallback because both columns may exist after stacking cycles.
    crp_hs = coalesce_numeric(out, ["LBXHSCRP"])
    crp_std = coalesce_numeric(out, ["LBXCRP"]) * 10
    crp = crp_hs.where(crp_hs.notna(), crp_std)
    out["crp_mg_l"] = crp
    out["crp_log"] = np.log(out["crp_mg_l"].where(out["crp_mg_l"] > 0))
    out["high_crp"] = np.where(out["crp_mg_l"] >= 3, 1, np.where(out["crp_mg_l"].notna(), 0, np.nan))

    out["albumin"] = clean_special_codes(out["LBXSAL"]) if "LBXSAL" in out else np.nan
    out["hemoglobin"] = clean_special_codes(out["LBXHGB"]) if "LBXHGB" in out else np.nan

    vuln_components = pd.concat([
        out["ckd"], out["diabetes_biomarker"], out["obesity"], out["high_crp"]
    ], axis=1)
    out["biomarker_vulnerability_score"] = vuln_components.sum(axis=1, skipna=False)
    # If at least two components are non-missing, define high vulnerability as >=2 positive domains.
    out["biomarker_vulnerability_high"] = np.where(
        vuln_components.notna().sum(axis=1) >= 2,
        np.where(vuln_components.sum(axis=1, skipna=True) >= 2, 1, 0),
        np.nan,
    )

    return out


def derive_short_term_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    health_col = first_existing(out, ["HUQ010", "HSD010"])
    if health_col:
        health = clean_special_codes(out[health_col])
        out["poor_self_rated_health"] = np.where(health.isin([4, 5]), 1, np.where(health.isin([1, 2, 3]), 0, np.nan))
    else:
        out["poor_self_rated_health"] = np.nan

    if "HUQ051" in out:
        visits = clean_special_codes(out["HUQ051"])
        out["any_healthcare_use"] = np.where(visits > 0, 1, np.where(visits == 0, 0, np.nan))
        out["high_healthcare_use"] = np.where(visits >= 4, 1, np.where(visits.notna(), 0, np.nan))
    else:
        out["any_healthcare_use"] = np.nan
        out["high_healthcare_use"] = np.nan

    dpq_cols = [c for c in ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"] if c in out]
    if len(dpq_cols) >= 7:
        dpq = out[dpq_cols].apply(clean_special_codes)
        dpq = dpq.where(dpq.isin([0, 1, 2, 3]))
        out["phq9_score"] = dpq.sum(axis=1, min_count=7)
        out["depressive_symptoms"] = np.where(out["phq9_score"] >= 10, 1, np.where(out["phq9_score"].notna(), 0, np.nan))
    else:
        out["phq9_score"] = np.nan
        out["depressive_symptoms"] = np.nan

    pfq_candidates = ["PFQ020", "PFQ049"] + [f"PFQ061{x}" for x in list("ABCDEFGHIJKLMNO")]
    pfq_binary = []
    for c in pfq_candidates:
        if c in out:
            pfq_binary.append(recode_yes_no(out[c]))
    out["physical_limitation"] = pd.concat(pfq_binary, axis=1).max(axis=1, skipna=True) if pfq_binary else np.nan

    if "SLQ050" in out:
        out["sleep_problem"] = recode_yes_no(out["SLQ050"])
    else:
        out["sleep_problem"] = np.nan

    return out



def derive_dual_burden_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create a joint bladder-health/biomarker vulnerability group.

    Reference group: Tier 0/1 with low biomarker vulnerability.
    High bladder burden: Tier 2.
    High biomarker vulnerability: biomarker_vulnerability_high == 1.
    """
    out = df.copy()
    tier = out["nhanes_bladder_tier"].astype(str) if "nhanes_bladder_tier" in out.columns else pd.Series(np.nan, index=out.index)
    biomarker_high = pd.to_numeric(out.get("biomarker_vulnerability_high", pd.Series(np.nan, index=out.index)), errors="coerce")

    bladder_high = np.where(tier == "Tier 2", 1, np.where(tier.isin(["Tier 0", "Tier 1"]), 0, np.nan))
    group = pd.Series(np.nan, index=out.index, dtype="object")
    group = group.mask((bladder_high == 0) & (biomarker_high == 0), "Low burden")
    group = group.mask((bladder_high == 1) & (biomarker_high == 0), "Bladder-only")
    group = group.mask((bladder_high == 0) & (biomarker_high == 1), "Biomarker-only")
    group = group.mask((bladder_high == 1) & (biomarker_high == 1), "Dual burden")

    out["bladder_high_burden"] = bladder_high
    out["biomarker_high_burden"] = biomarker_high
    out["dual_burden_group"] = pd.Categorical(group, categories=DUAL_BURDEN_LEVELS, ordered=True)
    return out


def dual_burden_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "dual_burden_group" not in df.columns:
        return pd.DataFrame(rows)
    for group in DUAL_BURDEN_LEVELS:
        mask = df["dual_burden_group"].astype(str).eq(group)
        rows.append({
            "module": "dual_burden",
            "group": group,
            "n": int(mask.sum()),
            "weighted_percent": weighted_mean(mask.astype(float), df["survey_weight"]) * 100,
        })
    return pd.DataFrame(rows)


def dual_burden_mortality_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "dual_burden_group" not in df.columns or "followup_years" not in df.columns:
        return pd.DataFrame(rows)
    linked = df["followup_years"].notna() & df["dual_burden_group"].notna()
    for group in DUAL_BURDEN_LEVELS:
        g = df[linked & df["dual_burden_group"].astype(str).eq(group)].copy()
        row = {
            "module": "dual_burden_mortality",
            "group": group,
            "n_linked": int(g.shape[0]),
            "weighted_percent_among_linked": weighted_mean(pd.Series(1.0, index=g.index), g["survey_weight"]) * 100 if g.shape[0] else np.nan,
            "mean_followup_years": float(pd.to_numeric(g.get("followup_years", pd.Series(dtype=float)), errors="coerce").mean()) if g.shape[0] else np.nan,
            "median_followup_years": float(pd.to_numeric(g.get("followup_years", pd.Series(dtype=float)), errors="coerce").median()) if g.shape[0] else np.nan,
            "max_followup_years": float(pd.to_numeric(g.get("followup_years", pd.Series(dtype=float)), errors="coerce").max()) if g.shape[0] else np.nan,
        }
        for outcome in ["all_cause_death", "cvd_death", "cancer_death"]:
            if outcome in g.columns and g.shape[0]:
                y = pd.to_numeric(g[outcome], errors="coerce")
                row[f"{outcome}_events"] = int(y.fillna(0).sum())
                row[f"{outcome}_weighted_prevalence"] = weighted_mean(y, g["survey_weight"])
            else:
                row[f"{outcome}_events"] = np.nan
                row[f"{outcome}_weighted_prevalence"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_dual_design_matrix(d: pd.DataFrame, covars_existing: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    group = pd.Categorical(d["dual_burden_group"], categories=DUAL_BURDEN_LEVELS, ordered=True)
    group_dummies = pd.get_dummies(group, prefix="dual", drop_first=True, dtype=float)
    group_dummies.index = d.index
    x_parts = [group_dummies]
    for c in covars_existing:
        if c not in d.columns:
            continue
        if d[c].dtype == "object" or str(d[c].dtype).startswith("category"):
            tmp = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
            tmp.index = d.index
            x_parts.append(tmp)
        else:
            tmp = pd.to_numeric(d[c], errors="coerce").to_frame(c)
            tmp.index = d.index
            x_parts.append(tmp)
    X = pd.concat(x_parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()].copy()
    variable_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[variable_cols]
    dual_terms = [c for c in X.columns if c.startswith("dual_") and X[c].nunique(dropna=True) > 1]
    X = sm.add_constant(X, has_constant="add")
    return X, dual_terms


def prepare_dual_weighted_matrix(df: pd.DataFrame, outcome: str, outcome_type: str, covars: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], pd.Series, pd.DataFrame]:
    needed = [outcome, "dual_burden_group", "survey_weight", "SDMVPSU"] + list(covars)
    existing = [c for c in needed if c in df.columns]
    d = df[existing].copy()
    d = d.dropna(subset=[outcome, "dual_burden_group", "survey_weight"]).copy()
    d = d[(pd.to_numeric(d["survey_weight"], errors="coerce") > 0)].copy()
    if "SDMVPSU" not in d.columns:
        d["SDMVPSU"] = np.arange(len(d))
    y = pd.to_numeric(d[outcome], errors="coerce")
    if outcome_type == "logistic":
        d = d[y.isin([0, 1])].copy()
    else:
        d = d[y.notna()].copy()
    y = pd.to_numeric(d[outcome], errors="coerce")

    covars_existing = [c for c in covars if c in d.columns]
    if covars_existing:
        d = d.dropna(subset=covars_existing).copy()
        y = pd.to_numeric(d[outcome], errors="coerce")

    X, dual_terms = make_dual_design_matrix(d, covars_existing)
    w = pd.to_numeric(d["survey_weight"], errors="coerce")
    groups = d["SDMVPSU"].copy()
    complete = y.notna() & w.notna() & (w > 0) & X.notna().all(axis=1)
    d = d.loc[complete].copy()
    y = y.loc[complete]
    w = w.loc[complete]
    groups = groups.loc[complete]
    X = X.loc[complete].copy()
    dual_terms = [c for c in dual_terms if c in X.columns and X[c].nunique(dropna=True) > 1]
    return X, y, w, dual_terms, groups, d


def run_dual_burden_weighted_models(df: pd.DataFrame, outcomes: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for outcome, outcome_type in outcomes.items():
        if outcome not in df.columns:
            continue
        for model_name, covars in MODEL_SPECS.items():
            try:
                X, y, w, dual_terms, groups, d_model = prepare_dual_weighted_matrix(df, outcome, outcome_type, covars)
            except Exception as exc:
                rows.append({"module": "dual_burden_short_term", "outcome": outcome, "model": model_name, "status": "prepare_failed", "error": str(exc)})
                continue
            n = int(len(y))
            events = float(y.sum()) if outcome_type == "logistic" else np.nan
            if n < MIN_TOTAL_N_FOR_MODEL:
                rows.append({"module": "dual_burden_short_term", "outcome": outcome, "model": model_name, "status": "skipped_small_n", "n": n, "events": events})
                continue
            if outcome_type == "logistic" and (events < MIN_EVENTS_FOR_LOGISTIC or (n - events) < MIN_EVENTS_FOR_LOGISTIC):
                rows.append({"module": "dual_burden_short_term", "outcome": outcome, "model": model_name, "status": "skipped_few_events", "n": n, "events": events})
                continue
            if not dual_terms:
                rows.append({"module": "dual_burden_short_term", "outcome": outcome, "model": model_name, "status": "skipped_no_dual_contrast", "n": n, "events": events})
                continue
            try:
                fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200
                )
            except Exception:
                try:
                    fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(maxiter=200)
                except Exception as exc:
                    rows.append({"module": "dual_burden_short_term", "outcome": outcome, "model": model_name, "status": "fit_failed", "error": str(exc), "n": n, "events": events})
                    continue
            for term in dual_terms:
                est = float(fit.params.get(term, np.nan))
                se = float(fit.bse.get(term, np.nan))
                p = float(fit.pvalues.get(term, np.nan))
                # Marginal difference against Low burden reference.
                X0 = X.copy()
                X1 = X.copy()
                for t in dual_terms:
                    if t in X0.columns:
                        X0[t] = 0
                        X1[t] = 0
                X1[term] = 1
                rd = weighted_mean(pd.Series(fit.predict(X1) - fit.predict(X0), index=X.index), w)
                rows.append({
                    "module": "dual_burden_short_term",
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "model": model_name,
                    "term": term.replace("dual_", ""),
                    "reference": "Low burden",
                    "effect_name": "OR",
                    "effect": math.exp(est),
                    "lcl": math.exp(est - 1.96 * se),
                    "ucl": math.exp(est + 1.96 * se),
                    "p": p,
                    "marginal_difference": rd,
                    "excess_per_1000": rd * 1000 if pd.notna(rd) else np.nan,
                    "n": n,
                    "events": events,
                    "low_burden_n": int((d_model["dual_burden_group"] == "Low burden").sum()),
                    "bladder_only_n": int((d_model["dual_burden_group"] == "Bladder-only").sum()),
                    "biomarker_only_n": int((d_model["dual_burden_group"] == "Biomarker-only").sum()),
                    "dual_burden_n": int((d_model["dual_burden_group"] == "Dual burden").sum()),
                    "status": "ok",
                    "error": "",
                })
    return pd.DataFrame(rows)


def prepare_dual_cox_matrix(df: pd.DataFrame, outcome: str, covars: Sequence[str]) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    needed = ["followup_years", outcome, "dual_burden_group", "survey_weight", "SDMVPSU"] + list(covars)
    existing = [c for c in needed if c in df.columns]
    d = df[existing].copy()
    d = d.dropna(subset=["followup_years", outcome, "dual_burden_group", "survey_weight"]).copy()
    d = d[(pd.to_numeric(d["followup_years"], errors="coerce") > 0) & (pd.to_numeric(d["survey_weight"], errors="coerce") > 0)].copy()
    if "SDMVPSU" not in d.columns:
        d["SDMVPSU"] = np.arange(len(d))
    covars_existing = [c for c in covars if c in d.columns]
    if covars_existing:
        d = d.dropna(subset=covars_existing).copy()
    X, dual_terms = make_dual_design_matrix(d, covars_existing)
    # Cox PH models should not include an intercept/constant column. The main tier Cox
    # model happened to converge with a constant in earlier versions, but the joint
    # dual-burden design can fail with delta=nan if the constant is retained.
    if "const" in X.columns:
        X = X.drop(columns=["const"])
    complete = X.notna().all(axis=1)
    d = d.loc[complete].copy()
    X = X.loc[complete].copy()
    # Remove no-variance columns after complete-case filtering.
    variable_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[variable_cols]
    dual_terms = [c for c in dual_terms if c in X.columns and X[c].nunique(dropna=True) > 1]
    return X, dual_terms, d


def run_dual_burden_cox_models(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        return pd.DataFrame([{"module": "dual_burden_mortality", "status": "skipped_lifelines_not_installed", "error": str(exc)}])
    rows = []
    for outcome in ["all_cause_death", "cvd_death", "cancer_death"]:
        if outcome not in df.columns:
            continue
        for model_name, covars in MODEL_SPECS.items():
            try:
                X, dual_terms, d = prepare_dual_cox_matrix(df, outcome, covars)
            except Exception as exc:
                rows.append({"module": "dual_burden_mortality", "outcome": outcome, "model": model_name, "status": "prepare_failed", "error": str(exc)})
                continue
            deaths = int(pd.to_numeric(d[outcome], errors="coerce").fillna(0).sum()) if outcome in d.columns else 0
            if len(d) < MIN_TOTAL_N_FOR_MODEL or deaths < MIN_DEATHS_FOR_COX:
                rows.append({"module": "dual_burden_mortality", "outcome": outcome, "model": model_name, "status": "skipped_small_events", "n": len(d), "events": deaths})
                continue
            if not dual_terms:
                rows.append({"module": "dual_burden_mortality", "outcome": outcome, "model": model_name, "status": "skipped_no_dual_contrast", "n": len(d), "events": deaths})
                continue
            model_df = pd.concat([d[["followup_years", outcome, "survey_weight", "SDMVPSU"]], X], axis=1).dropna().copy()
            deaths_model = int(pd.to_numeric(model_df[outcome], errors="coerce").fillna(0).sum())
            if model_df.shape[0] < MIN_TOTAL_N_FOR_MODEL or deaths_model < MIN_DEATHS_FOR_COX:
                rows.append({"module": "dual_burden_mortality", "outcome": outcome, "model": model_name, "status": "skipped_small_events_after_design", "n": int(model_df.shape[0]), "events": deaths_model})
                continue
            try:
                fit_status = "ok"
                fit_note = "unpenalized"
                try:
                    cph = CoxPHFitter()
                    cph.fit(
                        model_df,
                        duration_col="followup_years",
                        event_col=outcome,
                        weights_col="survey_weight",
                        cluster_col="SDMVPSU",
                        robust=True,
                    )
                except Exception as first_exc:
                    # Fallback for occasional convergence failures in the expanded
                    # dual-burden design. A small ridge penalizer stabilizes Cox fitting
                    # without changing the estimand qualitatively.
                    cph = CoxPHFitter(penalizer=0.01)
                    cph.fit(
                        model_df,
                        duration_col="followup_years",
                        event_col=outcome,
                        weights_col="survey_weight",
                        cluster_col="SDMVPSU",
                        robust=True,
                    )
                    fit_status = "ok_penalized_fallback"
                    fit_note = f"penalizer=0.01 after convergence failure: {str(first_exc)[:160]}"
                for term in dual_terms:
                    if term not in cph.summary.index:
                        continue
                    srow = cph.summary.loc[term]
                    rows.append({
                        "module": "dual_burden_mortality",
                        "outcome": outcome,
                        "model": model_name,
                        "term": term.replace("dual_", ""),
                        "reference": "Low burden",
                        "effect_name": "HR",
                        "effect": float(np.exp(srow["coef"])),
                        "lcl": float(np.exp(srow["coef lower 95%"])),
                        "ucl": float(np.exp(srow["coef upper 95%"])),
                        "p": float(srow["p"]),
                        "n": int(model_df.shape[0]),
                        "events": deaths_model,
                        "low_burden_n": int((d.loc[model_df.index, "dual_burden_group"] == "Low burden").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "bladder_only_n": int((d.loc[model_df.index, "dual_burden_group"] == "Bladder-only").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "biomarker_only_n": int((d.loc[model_df.index, "dual_burden_group"] == "Biomarker-only").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "dual_burden_n": int((d.loc[model_df.index, "dual_burden_group"] == "Dual burden").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "status": fit_status,
                        "fit_note": fit_note,
                        "error": "",
                    })
            except Exception as exc:
                rows.append({"module": "dual_burden_mortality", "outcome": outcome, "model": model_name, "status": "fit_failed", "n": len(d), "events": deaths, "error": str(exc)})
    return pd.DataFrame(rows)


# =========================================================
# 6A) Multiplicative interaction module for bladder burden x biomarker vulnerability
# =========================================================
def make_interaction_design_matrix(d: pd.DataFrame, covars_existing: Sequence[str], add_const: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Build aligned design matrix for multiplicative interaction analyses.

    The interaction is defined on two binary indicators:
    bladder_high_burden = Tier 2 vs Tier 0/1;
    biomarker_high_burden = biomarker_vulnerability_high == 1.
    """
    X0 = pd.DataFrame(index=d.index)
    X0["bladder_high_burden"] = pd.to_numeric(d["bladder_high_burden"], errors="coerce")
    X0["biomarker_high_burden"] = pd.to_numeric(d["biomarker_high_burden"], errors="coerce")
    X0["bladder_x_biomarker"] = X0["bladder_high_burden"] * X0["biomarker_high_burden"]

    x_parts = [X0]
    for c in covars_existing:
        if c not in d.columns:
            continue
        if d[c].dtype == "object" or str(d[c].dtype).startswith("category"):
            tmp = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
            tmp.index = d.index
            x_parts.append(tmp)
        else:
            tmp = pd.to_numeric(d[c], errors="coerce").to_frame(c)
            tmp.index = d.index
            x_parts.append(tmp)
    X = pd.concat(x_parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()].copy()
    variable_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[variable_cols]
    interaction_terms = [c for c in ["bladder_high_burden", "biomarker_high_burden", "bladder_x_biomarker"] if c in X.columns]
    if add_const:
        X = sm.add_constant(X, has_constant="add")
    return X, interaction_terms


def prepare_interaction_weighted_matrix(df: pd.DataFrame, outcome: str, outcome_type: str, covars: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], pd.Series, pd.DataFrame]:
    needed = [outcome, "bladder_high_burden", "biomarker_high_burden", "survey_weight", "SDMVPSU"] + list(covars)
    existing = [c for c in needed if c in df.columns]
    d = df[existing].copy()
    d = d.dropna(subset=[outcome, "bladder_high_burden", "biomarker_high_burden", "survey_weight"]).copy()
    d = d[(pd.to_numeric(d["survey_weight"], errors="coerce") > 0)].copy()
    if "SDMVPSU" not in d.columns:
        d["SDMVPSU"] = np.arange(len(d))

    y = pd.to_numeric(d[outcome], errors="coerce")
    if outcome_type == "logistic":
        d = d[y.isin([0, 1])].copy()
    else:
        d = d[y.notna()].copy()
    y = pd.to_numeric(d[outcome], errors="coerce")

    covars_existing = [c for c in covars if c in d.columns]
    if covars_existing:
        d = d.dropna(subset=covars_existing).copy()
        y = pd.to_numeric(d[outcome], errors="coerce")

    X, interaction_terms = make_interaction_design_matrix(d, covars_existing, add_const=True)
    w = pd.to_numeric(d["survey_weight"], errors="coerce")
    groups = d["SDMVPSU"].copy()

    complete = y.notna() & w.notna() & (w > 0) & X.notna().all(axis=1)
    d = d.loc[complete].copy()
    y = y.loc[complete]
    w = w.loc[complete]
    groups = groups.loc[complete]
    X = X.loc[complete].copy()

    # Recheck variance after complete-case filtering.
    nonconst_cols = [c for c in X.columns if c == "const" or X[c].nunique(dropna=True) > 1]
    X = X[nonconst_cols]
    interaction_terms = [c for c in interaction_terms if c in X.columns and X[c].nunique(dropna=True) > 1]
    return X, y, w, interaction_terms, groups, d


def interaction_group_counts(d: pd.DataFrame) -> Dict[str, int]:
    b = pd.to_numeric(d.get("bladder_high_burden", pd.Series(np.nan, index=d.index)), errors="coerce")
    m = pd.to_numeric(d.get("biomarker_high_burden", pd.Series(np.nan, index=d.index)), errors="coerce")
    return {
        "B0M0_n": int(((b == 0) & (m == 0)).sum()),
        "B1M0_n": int(((b == 1) & (m == 0)).sum()),
        "B0M1_n": int(((b == 0) & (m == 1)).sum()),
        "B1M1_n": int(((b == 1) & (m == 1)).sum()),
    }


def predict_interaction_marginal_risks(fit, X: pd.DataFrame, w: pd.Series, outcome: str, model_name: str, module_name: str) -> pd.DataFrame:
    rows = []
    combos = [
        ("B0M0_low", 0, 0),
        ("B1M0_bladder_only", 1, 0),
        ("B0M1_biomarker_only", 0, 1),
        ("B1M1_dual_burden", 1, 1),
    ]
    ref_risk = np.nan
    for label, bval, mval in combos:
        Xp = X.copy()
        if "bladder_high_burden" in Xp.columns:
            Xp["bladder_high_burden"] = bval
        if "biomarker_high_burden" in Xp.columns:
            Xp["biomarker_high_burden"] = mval
        if "bladder_x_biomarker" in Xp.columns:
            Xp["bladder_x_biomarker"] = bval * mval
        pred = fit.predict(Xp)
        risk = weighted_mean(pd.Series(pred, index=Xp.index), w)
        if label == "B0M0_low":
            ref_risk = risk
        rows.append({
            "module": module_name,
            "outcome": outcome,
            "model": model_name,
            "group": label,
            "adjusted_risk": risk,
            "adjusted_excess_vs_B0M0": risk - ref_risk if pd.notna(ref_risk) and pd.notna(risk) else np.nan,
            "adjusted_excess_per_1000_vs_B0M0": (risk - ref_risk) * 1000 if pd.notna(ref_risk) and pd.notna(risk) else np.nan,
        })
    return pd.DataFrame(rows)


def run_interaction_weighted_models(df: pd.DataFrame, outcomes: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    risk_tables = []
    for outcome, outcome_type in outcomes.items():
        if outcome not in df.columns:
            continue
        for model_name, covars in MODEL_SPECS.items():
            try:
                X, y, w, interaction_terms, groups, d_model = prepare_interaction_weighted_matrix(df, outcome, outcome_type, covars)
            except Exception as exc:
                model_rows.append({"module": "dual_burden_interaction_short_term", "outcome": outcome, "model": model_name, "status": "prepare_failed", "error": str(exc)})
                continue
            n = int(len(y))
            events = float(y.sum()) if outcome_type == "logistic" else np.nan
            counts = interaction_group_counts(d_model)
            if n < MIN_TOTAL_N_FOR_MODEL:
                model_rows.append({"module": "dual_burden_interaction_short_term", "outcome": outcome, "model": model_name, "status": "skipped_small_n", "n": n, "events": events, **counts})
                continue
            if outcome_type == "logistic" and (events < MIN_EVENTS_FOR_LOGISTIC or (n - events) < MIN_EVENTS_FOR_LOGISTIC):
                model_rows.append({"module": "dual_burden_interaction_short_term", "outcome": outcome, "model": model_name, "status": "skipped_few_events", "n": n, "events": events, **counts})
                continue
            if "bladder_x_biomarker" not in interaction_terms:
                model_rows.append({"module": "dual_burden_interaction_short_term", "outcome": outcome, "model": model_name, "status": "skipped_no_interaction_variance", "n": n, "events": events, **counts})
                continue
            try:
                fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200
                )
            except Exception:
                try:
                    fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(maxiter=200)
                except Exception as exc:
                    model_rows.append({"module": "dual_burden_interaction_short_term", "outcome": outcome, "model": model_name, "status": "fit_failed", "error": str(exc), "n": n, "events": events, **counts})
                    continue
            risk_tables.append(predict_interaction_marginal_risks(
                fit, X, w, outcome, model_name, "dual_burden_interaction_short_term"
            ))
            for term in ["bladder_high_burden", "biomarker_high_burden", "bladder_x_biomarker"]:
                if term not in fit.params.index:
                    continue
                est = float(fit.params.get(term, np.nan))
                se = float(fit.bse.get(term, np.nan))
                model_rows.append({
                    "module": "dual_burden_interaction_short_term",
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "model": model_name,
                    "term": term,
                    "effect_name": "OR" if term != "bladder_x_biomarker" else "ratio_of_ORs",
                    "effect": math.exp(est),
                    "lcl": math.exp(est - 1.96 * se),
                    "ucl": math.exp(est + 1.96 * se),
                    "p": float(fit.pvalues.get(term, np.nan)),
                    "n": n,
                    "events": events,
                    **counts,
                    "status": "ok",
                    "error": "",
                })
    marginal = pd.concat(risk_tables, ignore_index=True, sort=False) if risk_tables else pd.DataFrame()
    return pd.DataFrame(model_rows), marginal


def make_interaction_diagnostics(df: pd.DataFrame, outcomes: Dict[str, str], module_name: str, cox: bool = False) -> pd.DataFrame:
    rows = []
    for outcome, outcome_type in outcomes.items():
        if outcome not in df.columns:
            rows.append({"module": module_name, "outcome": outcome, "status": "missing_outcome"})
            continue
        for model_name, covars in MODEL_SPECS.items():
            row = {"module": module_name, "outcome": outcome, "model": model_name, "status": "ok"}
            try:
                if cox:
                    X, terms, d = prepare_interaction_cox_matrix(df, outcome, covars)
                    row.update({
                        "n_model": int(len(d)),
                        "events": int(pd.to_numeric(d[outcome], errors="coerce").fillna(0).sum()),
                        "mean_followup_years": float(pd.to_numeric(d["followup_years"], errors="coerce").mean()) if len(d) else np.nan,
                        "max_followup_years": float(pd.to_numeric(d["followup_years"], errors="coerce").max()) if len(d) else np.nan,
                        "terms_available": ";".join(terms),
                        "n_design_columns": int(X.shape[1]),
                    })
                else:
                    X, y, w, terms, groups, d = prepare_interaction_weighted_matrix(df, outcome, outcome_type, covars)
                    row.update({
                        "n_model": int(len(d)),
                        "events": float(y.sum()) if outcome_type == "logistic" else np.nan,
                        "terms_available": ";".join(terms),
                        "n_design_columns": int(X.shape[1]),
                    })
                row.update(interaction_group_counts(d))
            except Exception as exc:
                row.update({"status": "prepare_failed", "error": str(exc)})
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_interaction_cox_matrix(df: pd.DataFrame, outcome: str, covars: Sequence[str]) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    needed = ["followup_years", outcome, "bladder_high_burden", "biomarker_high_burden", "survey_weight", "SDMVPSU"] + list(covars)
    existing = [c for c in needed if c in df.columns]
    d = df[existing].copy()
    d = d.dropna(subset=["followup_years", outcome, "bladder_high_burden", "biomarker_high_burden", "survey_weight"]).copy()
    d = d[(pd.to_numeric(d["followup_years"], errors="coerce") > 0) & (pd.to_numeric(d["survey_weight"], errors="coerce") > 0)].copy()
    if "SDMVPSU" not in d.columns:
        d["SDMVPSU"] = np.arange(len(d))
    covars_existing = [c for c in covars if c in d.columns]
    if covars_existing:
        d = d.dropna(subset=covars_existing).copy()
    X, interaction_terms = make_interaction_design_matrix(d, covars_existing, add_const=False)
    complete = X.notna().all(axis=1)
    d = d.loc[complete].copy()
    X = X.loc[complete].copy()
    variable_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[variable_cols]
    interaction_terms = [c for c in interaction_terms if c in X.columns and X[c].nunique(dropna=True) > 1]
    return X, interaction_terms, d


def run_interaction_cox_models(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        return pd.DataFrame([{"module": "dual_burden_interaction_mortality", "status": "skipped_lifelines_not_installed", "error": str(exc)}])
    rows = []
    for outcome in ["all_cause_death", "cvd_death", "cancer_death"]:
        if outcome not in df.columns:
            continue
        for model_name, covars in MODEL_SPECS.items():
            try:
                X, interaction_terms, d = prepare_interaction_cox_matrix(df, outcome, covars)
            except Exception as exc:
                rows.append({"module": "dual_burden_interaction_mortality", "outcome": outcome, "model": model_name, "status": "prepare_failed", "error": str(exc)})
                continue
            deaths = int(pd.to_numeric(d[outcome], errors="coerce").fillna(0).sum()) if outcome in d.columns else 0
            counts = interaction_group_counts(d)
            if len(d) < MIN_TOTAL_N_FOR_MODEL or deaths < MIN_DEATHS_FOR_COX:
                rows.append({"module": "dual_burden_interaction_mortality", "outcome": outcome, "model": model_name, "status": "skipped_small_events", "n": len(d), "events": deaths, **counts})
                continue
            if "bladder_x_biomarker" not in interaction_terms:
                rows.append({"module": "dual_burden_interaction_mortality", "outcome": outcome, "model": model_name, "status": "skipped_no_interaction_variance", "n": len(d), "events": deaths, **counts})
                continue
            model_df = pd.concat([d[["followup_years", outcome, "survey_weight", "SDMVPSU"]], X], axis=1).dropna().copy()
            deaths_model = int(pd.to_numeric(model_df[outcome], errors="coerce").fillna(0).sum())
            if model_df.shape[0] < MIN_TOTAL_N_FOR_MODEL or deaths_model < MIN_DEATHS_FOR_COX:
                rows.append({"module": "dual_burden_interaction_mortality", "outcome": outcome, "model": model_name, "status": "skipped_small_events_after_design", "n": int(model_df.shape[0]), "events": deaths_model, **counts})
                continue
            try:
                fit_status = "ok"
                fit_note = "unpenalized"
                try:
                    cph = CoxPHFitter()
                    cph.fit(model_df, duration_col="followup_years", event_col=outcome, weights_col="survey_weight", cluster_col="SDMVPSU", robust=True)
                except Exception as first_exc:
                    cph = CoxPHFitter(penalizer=0.01)
                    cph.fit(model_df, duration_col="followup_years", event_col=outcome, weights_col="survey_weight", cluster_col="SDMVPSU", robust=True)
                    fit_status = "ok_penalized_fallback"
                    fit_note = f"penalizer=0.01 after convergence failure: {str(first_exc)[:160]}"
                for term in ["bladder_high_burden", "biomarker_high_burden", "bladder_x_biomarker"]:
                    if term not in cph.summary.index:
                        continue
                    srow = cph.summary.loc[term]
                    rows.append({
                        "module": "dual_burden_interaction_mortality",
                        "outcome": outcome,
                        "model": model_name,
                        "term": term,
                        "effect_name": "HR" if term != "bladder_x_biomarker" else "ratio_of_HRs",
                        "effect": float(np.exp(srow["coef"])),
                        "lcl": float(np.exp(srow["coef lower 95%"])),
                        "ucl": float(np.exp(srow["coef upper 95%"])),
                        "p": float(srow["p"]),
                        "n": int(model_df.shape[0]),
                        "events": deaths_model,
                        **counts,
                        "status": fit_status,
                        "fit_note": fit_note,
                        "error": "",
                    })
            except Exception as exc:
                rows.append({"module": "dual_burden_interaction_mortality", "outcome": outcome, "model": model_name, "status": "fit_failed", "n": len(d), "events": deaths, "error": str(exc), **counts})
    return pd.DataFrame(rows)

def build_survey_weight(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_weight"] = np.nan
    out["base_weight_source"] = "missing"
    for c in WEIGHT_CANDIDATES:
        if c in out:
            w = pd.to_numeric(out[c], errors="coerce")
            use = out["base_weight"].isna() & w.notna() & (w > 0)
            out.loc[use, "base_weight"] = w[use]
            out.loc[use, "base_weight_source"] = c

    if "represented_years" not in out.columns:
        out["represented_years"] = np.where(out["base_weight_source"].eq("WTMECPRP"), 3.2, 2.0)
    out["represented_years"] = pd.to_numeric(out["represented_years"], errors="coerce")
    out.loc[out["represented_years"].isna(), "represented_years"] = np.where(out.loc[out["represented_years"].isna(), "base_weight_source"].eq("WTMECPRP"), 3.2, 2.0)

    # Multi-period weights: 2-year weights get 2 / total_years;
    # 2017-March 2020 pre-pandemic weights get 3.2 / total_years.
    period_years = out.loc[out["base_weight"].notna() & (out["base_weight"] > 0), ["cycle", "represented_years"]].drop_duplicates()
    total_years = float(period_years["represented_years"].sum()) if not period_years.empty else np.nan
    out["survey_weight"] = out["base_weight"]
    if total_years and not pd.isna(total_years) and total_years > 0:
        out["survey_weight"] = out["base_weight"] * out["represented_years"] / total_years
    out["total_represented_years_for_weight"] = total_years
    return out

# =========================================================
# 5) Tables and models
# =========================================================
def make_variable_coverage(df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "KIQ005", "KIQ010", "KIQ042", "KIQ430", "KIQ044", "KIQ450", "KIQ046", "KIQ470",
        "KIQ050", "KIQ052", "KIQ480", "LBXSCR", "URXUMA", "URXUCR", "LBXGH", "LBXCRP",
        "LBXHSCRP", "BMXBMI", "BMXWAIST", "HUQ010", "HSD010", "HUQ051", "DPQ010", "PFQ020",
        "SLQ050", "WTMEC2YR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"
    ]
    rows = []
    for cycle, g in df.groupby("cycle", dropna=False):
        for v in variables:
            rows.append({
                "cycle": cycle,
                "variable": v,
                "available": v in g.columns,
                "non_missing": int(g[v].notna().sum()) if v in g.columns else 0,
            })
    return pd.DataFrame(rows)


def make_cycle_audit(raw: pd.DataFrame, analytic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_cycles = sorted(set(raw.get("cycle", pd.Series(dtype=object)).dropna().astype(str)) | set(analytic.get("cycle", pd.Series(dtype=object)).dropna().astype(str)), key=cycle_start_year)
    for cycle in all_cycles:
        rg = raw[raw["cycle"].astype(str).eq(cycle)] if "cycle" in raw else pd.DataFrame()
        ag = analytic[analytic["cycle"].astype(str).eq(cycle)] if "cycle" in analytic else pd.DataFrame()
        rows.append({
            "cycle": cycle,
            "cycle_note": first_non_missing(rg["cycle_note"]) if "cycle_note" in rg else "",
            "represented_years": first_non_missing(rg["represented_years"]) if "represented_years" in rg else np.nan,
            "raw_rows": int(rg.shape[0]),
            "analytic_rows_age_weight_tier": int(ag.shape[0]),
            "weight_source": ";".join(sorted([str(x) for x in ag.get("base_weight_source", pd.Series(dtype=object)).dropna().unique()])),
            "core_ui_non_missing": int(ag["any_ui"].notna().sum()) if "any_ui" in ag else 0,
            "tier_non_missing": int(ag["nhanes_bladder_tier"].notna().sum()) if "nhanes_bladder_tier" in ag else 0,
        })
    return pd.DataFrame(rows)


def phenotype_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, g in df.groupby("cycle", dropna=False):
        for tier in ["Tier 0", "Tier 1", "Tier 2"]:
            mask = g["nhanes_bladder_tier"].astype(str) == tier
            rows.append({
                "cycle": cycle,
                "tier": tier,
                "n": int(mask.sum()),
                "weighted_percent": weighted_mean(mask.astype(float), g["survey_weight"]) * 100,
            })
        rows.append({
            "cycle": cycle,
            "tier": "Core UI positive",
            "n": int((g["any_ui"] == 1).sum()),
            "weighted_percent": weighted_mean((g["any_ui"] == 1).astype(float), g["survey_weight"]) * 100,
        })
        rows.append({
            "cycle": cycle,
            "tier": "Nocturia >=2/night",
            "n": int((g["nocturia_2plus"] == 1).sum()),
            "weighted_percent": weighted_mean((g["nocturia_2plus"] == 1).astype(float), g["survey_weight"]) * 100,
        })
    return pd.DataFrame(rows)


def weighted_by_tier(df: pd.DataFrame, outcomes: Dict[str, str], table_name: str) -> pd.DataFrame:
    rows = []
    for outcome, typ in outcomes.items():
        if outcome not in df.columns:
            continue
        for tier in ["Tier 0", "Tier 1", "Tier 2"]:
            g = df[df["nhanes_bladder_tier"].astype(str) == tier]
            mask = g[outcome].notna() & g["survey_weight"].notna() & (g["survey_weight"] > 0)
            if mask.sum() == 0:
                val = np.nan
            else:
                val = weighted_mean(g.loc[mask, outcome], g.loc[mask, "survey_weight"])
            rows.append({
                "table": table_name,
                "outcome": outcome,
                "outcome_type": typ,
                "tier": tier,
                "n_non_missing": int(mask.sum()),
                "weighted_mean_or_prevalence": val,
            })
    return pd.DataFrame(rows)


def make_design_matrix(d: pd.DataFrame, covars_existing: Sequence[str]) -> pd.DataFrame:
    """Build a fully aligned design matrix for tier terms and covariates.

    This helper is used by biomarker, short-term, and mortality models. It keeps
    each dummy matrix on the same index as the source DataFrame. This prevents
    silent row expansion or sample loss caused by pandas index alignment.
    """
    tier = pd.Categorical(d["nhanes_bladder_tier"], categories=["Tier 0", "Tier 1", "Tier 2"], ordered=True)
    tier_dummies = pd.get_dummies(tier, prefix="tier", drop_first=True, dtype=float)
    tier_dummies.index = d.index

    x_parts = [tier_dummies]
    for c in covars_existing:
        if c not in d.columns:
            continue
        if d[c].dtype == "object" or str(d[c].dtype).startswith("category"):
            tmp = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
            tmp.index = d.index
            x_parts.append(tmp)
        else:
            tmp = pd.to_numeric(d[c], errors="coerce").to_frame(c)
            tmp.index = d.index
            x_parts.append(tmp)

    X = pd.concat(x_parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()].copy()
    return X


def prepare_model_matrix(df: pd.DataFrame, outcome: str, outcome_type: str, covars: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], pd.Series, pd.DataFrame]:
    needed = [outcome, "nhanes_bladder_tier", "survey_weight", "SDMVPSU"] + list(covars)
    existing = [c for c in needed if c in df.columns]
    d0 = df[existing].copy()

    d = d0.dropna(subset=[outcome, "nhanes_bladder_tier", "survey_weight"]).copy()
    d = d[(pd.to_numeric(d["survey_weight"], errors="coerce") > 0)].copy()
    if "SDMVPSU" not in d.columns:
        d["SDMVPSU"] = np.arange(len(d))

    y = pd.to_numeric(d[outcome], errors="coerce")
    if outcome_type == "logistic":
        d = d[y.isin([0, 1])].copy()
    else:
        d = d[y.notna()].copy()
    y = pd.to_numeric(d[outcome], errors="coerce")

    covars_existing = [c for c in covars if c in d.columns]
    if covars_existing:
        d = d.dropna(subset=covars_existing).copy()
        y = pd.to_numeric(d[outcome], errors="coerce")

    X = make_design_matrix(d, covars_existing)
    w = pd.to_numeric(d["survey_weight"], errors="coerce")
    groups = d["SDMVPSU"].copy()

    complete = y.notna() & w.notna() & (w > 0) & X.notna().all(axis=1)
    d = d.loc[complete].copy()
    y = y.loc[complete]
    w = w.loc[complete]
    groups = groups.loc[complete]
    X = X.loc[complete].copy()

    # Remove no-variance columns after complete-case filtering.
    variable_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[variable_cols]
    tier_terms = [c for c in ["tier_Tier 1", "tier_Tier 2"] if c in X.columns and X[c].nunique(dropna=True) > 1]

    X = sm.add_constant(X, has_constant="add")
    return X, y, w, tier_terms, groups, d


def marginal_difference(model, X: pd.DataFrame, w: pd.Series, term: str, tier_terms: Sequence[str], outcome_type: str) -> float:
    if term not in X.columns:
        return np.nan
    X0 = X.copy()
    X1 = X.copy()
    for t in tier_terms:
        if t in X0.columns:
            X0[t] = 0
            X1[t] = 0
    X1[term] = 1
    p0 = model.predict(X0)
    p1 = model.predict(X1)
    return weighted_mean(pd.Series(p1 - p0, index=X.index), w)


def run_weighted_models(df: pd.DataFrame, outcomes: Dict[str, str], module_name: str) -> pd.DataFrame:
    rows = []
    for outcome, outcome_type in outcomes.items():
        if outcome not in df.columns:
            continue
        for model_name, covars in MODEL_SPECS.items():
            try:
                X, y, w, tier_terms, groups, d_model = prepare_model_matrix(df, outcome, outcome_type, covars)
            except Exception as exc:
                rows.append({"module": module_name, "outcome": outcome, "model": model_name, "status": "prepare_failed", "error": str(exc)})
                continue

            n = int(len(y))
            events = float(y.sum()) if outcome_type == "logistic" else np.nan
            if n < MIN_TOTAL_N_FOR_MODEL:
                rows.append({"module": module_name, "outcome": outcome, "model": model_name, "status": "skipped_small_n", "n": n, "events": events})
                continue
            if outcome_type == "logistic" and (events < MIN_EVENTS_FOR_LOGISTIC or (n - events) < MIN_EVENTS_FOR_LOGISTIC):
                rows.append({"module": module_name, "outcome": outcome, "model": model_name, "status": "skipped_few_events", "n": n, "events": events})
                continue
            if not tier_terms:
                rows.append({"module": module_name, "outcome": outcome, "model": model_name, "status": "skipped_no_tier_contrast", "n": n, "events": events})
                continue

            try:
                if outcome_type == "logistic":
                    fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(
                        cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200
                    )
                else:
                    fit = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": groups})
            except Exception:
                try:
                    if outcome_type == "logistic":
                        fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(maxiter=200)
                    else:
                        fit = sm.WLS(y, X, weights=w).fit()
                except Exception as exc:
                    rows.append({"module": module_name, "outcome": outcome, "model": model_name, "status": "fit_failed", "error": str(exc), "n": n, "events": events})
                    continue

            for term in tier_terms:
                est = float(fit.params.get(term, np.nan))
                se = float(fit.bse.get(term, np.nan))
                p = float(fit.pvalues.get(term, np.nan))
                rd = marginal_difference(fit, X, w, term, tier_terms, outcome_type)
                if outcome_type == "logistic":
                    effect = math.exp(est)
                    lcl = math.exp(est - 1.96 * se)
                    ucl = math.exp(est + 1.96 * se)
                    effect_name = "OR"
                    excess_per_1000 = rd * 1000 if pd.notna(rd) else np.nan
                else:
                    effect = est
                    lcl = est - 1.96 * se
                    ucl = est + 1.96 * se
                    effect_name = "beta"
                    excess_per_1000 = np.nan
                rows.append({
                    "module": module_name,
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "model": model_name,
                    "term": term.replace("tier_", ""),
                    "effect_name": effect_name,
                    "effect": effect,
                    "lcl": lcl,
                    "ucl": ucl,
                    "p": p,
                    "marginal_difference": rd,
                    "excess_per_1000": excess_per_1000,
                    "n": n,
                    "events": events,
                    "tier0_n": int((d_model["nhanes_bladder_tier"] == "Tier 0").sum()) if "nhanes_bladder_tier" in d_model.columns else np.nan,
                    "tier1_n": int((d_model["nhanes_bladder_tier"] == "Tier 1").sum()) if "nhanes_bladder_tier" in d_model.columns else np.nan,
                    "tier2_n": int((d_model["nhanes_bladder_tier"] == "Tier 2").sum()) if "nhanes_bladder_tier" in d_model.columns else np.nan,
                    "status": "ok",
                    "error": "",
                })
    return pd.DataFrame(rows)


def make_weighted_model_input_diagnostics(df: pd.DataFrame, outcomes: Dict[str, str], module_name: str) -> pd.DataFrame:
    rows = []
    for outcome, outcome_type in outcomes.items():
        if outcome not in df.columns:
            rows.append({"module": module_name, "outcome": outcome, "status": "missing_outcome"})
            continue
        for model_name, covars in MODEL_SPECS.items():
            row = {
                "module": module_name,
                "outcome": outcome,
                "outcome_type": outcome_type,
                "model": model_name,
                "n_analytic_total": int(len(df)),
                "n_outcome_nonmissing": int(pd.to_numeric(df[outcome], errors="coerce").notna().sum()),
                "n_tier_nonmissing": int(df["nhanes_bladder_tier"].notna().sum()) if "nhanes_bladder_tier" in df.columns else 0,
                "n_weight_positive": int((pd.to_numeric(df.get("survey_weight", pd.Series(np.nan, index=df.index)), errors="coerce") > 0).sum()),
                "covariates_requested": ";".join(covars),
                "covariates_available": ";".join([c for c in covars if c in df.columns]),
                "covariates_missing": ";".join([c for c in covars if c not in df.columns]),
                "status": "ok",
            }
            try:
                X, y, w, tier_terms, groups, d_model = prepare_model_matrix(df, outcome, outcome_type, covars)
                row.update({
                    "n_model": int(len(d_model)),
                    "events": float(y.sum()) if outcome_type == "logistic" else np.nan,
                    "non_events": float(len(y) - y.sum()) if outcome_type == "logistic" else np.nan,
                    "tier0_n": int((d_model["nhanes_bladder_tier"] == "Tier 0").sum()) if "nhanes_bladder_tier" in d_model.columns else np.nan,
                    "tier1_n": int((d_model["nhanes_bladder_tier"] == "Tier 1").sum()) if "nhanes_bladder_tier" in d_model.columns else np.nan,
                    "tier2_n": int((d_model["nhanes_bladder_tier"] == "Tier 2").sum()) if "nhanes_bladder_tier" in d_model.columns else np.nan,
                    "tier_terms_available": ";".join(tier_terms),
                    "n_design_columns": int(X.shape[1]),
                })
                for c in covars:
                    if c in df.columns:
                        row[f"missing_{c}"] = int(df[c].isna().sum())
            except Exception as exc:
                row.update({"status": "prepare_failed", "error": str(exc)})
            rows.append(row)
    return pd.DataFrame(rows)

# =========================================================
# 6) Mortality module
# =========================================================
def infer_mortality_cycle_from_name(path: Path) -> str:
    """Infer source NHANES cycle from public-use mortality file name."""
    name = path.name.upper()
    m = re.search(r"NHANES[_-]?(\d{4})[_-]?(\d{4})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(\d{4})[_-](\d{4})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return "unknown"


def find_mortality_files(root: Path) -> List[Path]:
    """Find all public-use NHANES linked mortality files.

    The user may store files under D:\\科研\\NHANES\\NHANES生存数据 as DAT files, or under
    a mortality subfolder. V8 loads all matching files rather than only the first file.
    """
    candidates: List[Path] = []
    seen = set()
    search_dirs = []
    for d in MORTALITY_SEARCH_DIRS:
        if d.exists() and d.is_dir():
            search_dirs.append(d)
    if root.exists() and root.is_dir() and root not in search_dirs:
        search_dirs.append(root)

    for d in search_dirs:
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in [".csv", ".txt", ".dat", ".xpt", ".sas7bdat"]:
                continue
            name = p.name.upper()
            # Avoid accidentally capturing ordinary NHANES files. Mortality DAT files usually
            # contain MORT, LMF, DEATH, or DTH in the file name.
            if not re.search(r"MORT|LMF|DEATH|DTH", name):
                continue
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(p)

    def sort_key(path: Path):
        cyc = infer_mortality_cycle_from_name(path)
        return (cycle_start_year(cyc), path.name.upper())

    return sorted(candidates, key=sort_key)



def _clean_numeric_mortality_series(s: pd.Series) -> pd.Series:
    """Convert public-use NHANES mortality fixed-width fields to numeric.

    The DAT files use blanks and periods for missing values. Reading as string first
    avoids silent truncation or misplaced decimal handling.
    """
    x = s.astype(str).str.strip()
    x = x.replace({"": np.nan, ".": np.nan, "..": np.nan, "...": np.nan})
    return pd.to_numeric(x, errors="coerce")


def read_public_mortality_dat(path: Path) -> pd.DataFrame:
    """Read NHANES public-use linked mortality DAT files.

    Raw DAT layout is verified against the public-use fixed-width format:
    SEQN 1-6, ELIGSTAT 15, MORTSTAT 16, UCOD_LEADING 17-19,
    DIABETES 20, HYPERTEN 21, PERMTH_INT 43-45, PERMTH_EXM 46-48.
    Python colspecs are 0-based and end-exclusive.
    """
    colspecs = [
        (0, 6),    # SEQN, 1-based 1-6
        (14, 15),  # ELIGSTAT, 1-based 15
        (15, 16),  # MORTSTAT, 1-based 16
        (16, 19),  # UCOD_LEADING, 1-based 17-19
        (19, 20),  # DIABETES, 1-based 20
        (20, 21),  # HYPERTEN, 1-based 21
        (42, 45),  # PERMTH_INT, 1-based 43-45
        (45, 48),  # PERMTH_EXM, 1-based 46-48
    ]
    names = [
        "SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
        "DIABETES_MORT", "HYPERTEN_MORT", "PERMTH_INT", "PERMTH_EXM"
    ]
    mort = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str, keep_default_na=False)
    for c in names:
        mort[c] = _clean_numeric_mortality_series(mort[c])
    return mort


def mortality_raw_file_diagnostics(path: Path, mort: pd.DataFrame) -> Dict[str, object]:
    """Generate file-level diagnostics for mortality DAT parser verification."""
    line_lengths = []
    first_nonempty_line = ""
    try:
        with open(path, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n\r")
                if line and not first_nonempty_line:
                    first_nonempty_line = line[:80]
                if i < 5000:
                    line_lengths.append(len(line))
    except Exception:
        pass

    elig = mort[mort.get("ELIGSTAT", pd.Series(index=mort.index, dtype=float)).eq(1)].copy()
    row = {
        "file": str(path),
        "file_name": path.name,
        "source_cycle": infer_mortality_cycle_from_name(path),
        "n_rows": int(mort.shape[0]),
        "n_eligible": int(elig.shape[0]),
        "seqn_min": float(mort["SEQN"].min()) if "SEQN" in mort and mort["SEQN"].notna().any() else np.nan,
        "seqn_max": float(mort["SEQN"].max()) if "SEQN" in mort and mort["SEQN"].notna().any() else np.nan,
        "line_length_min_first5000": int(min(line_lengths)) if line_lengths else np.nan,
        "line_length_max_first5000": int(max(line_lengths)) if line_lengths else np.nan,
        "first_nonempty_line_prefix": first_nonempty_line,
    }
    for c in ["PERMTH_INT", "PERMTH_EXM"]:
        if c in elig.columns:
            row[f"{c.lower()}_nonmissing"] = int(elig[c].notna().sum())
            row[f"{c.lower()}_mean_months"] = float(elig[c].mean()) if elig[c].notna().any() else np.nan
            row[f"{c.lower()}_median_months"] = float(elig[c].median()) if elig[c].notna().any() else np.nan
            row[f"{c.lower()}_max_months"] = float(elig[c].max()) if elig[c].notna().any() else np.nan
            row[f"{c.lower()}_max_years"] = float(elig[c].max() / 12.0) if elig[c].notna().any() else np.nan
    if "MORTSTAT" in elig.columns:
        row["all_cause_deaths_raw"] = int(elig["MORTSTAT"].fillna(0).sum())
    if "UCOD_LEADING" in elig.columns and "MORTSTAT" in elig.columns:
        row["cvd_deaths_raw"] = int(((elig["MORTSTAT"] == 1) & (elig["UCOD_LEADING"].isin([1, 5]))).sum())
        row["cancer_deaths_raw"] = int(((elig["MORTSTAT"] == 1) & (elig["UCOD_LEADING"] == 2)).sum())
    return row


def load_one_mortality(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        mort = pd.read_csv(path)
    elif suffix == ".xpt":
        mort = read_xpt(path)
    elif suffix == ".sas7bdat":
        mort = pd.read_sas(path, format="sas7bdat", encoding="latin1")
    else:
        mort = read_public_mortality_dat(path)
    mort.columns = [normalize_colname(c) for c in mort.columns]
    if "SEQN" in mort:
        mort["SEQN"] = pd.to_numeric(mort["SEQN"], errors="coerce")
    mort["mortality_source_file"] = path.name
    mort["mortality_source_cycle"] = infer_mortality_cycle_from_name(path)
    return mort


def load_mortality_files(paths: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests = []
    parser_validation = []
    frames = []
    for path in paths:
        try:
            if path.suffix.lower() == ".dat":
                raw_mort = read_public_mortality_dat(path)
                parser_validation.append(mortality_raw_file_diagnostics(path, raw_mort))
                df = raw_mort.copy()
                df["mortality_source_file"] = path.name
                df["mortality_source_cycle"] = infer_mortality_cycle_from_name(path)
            else:
                df = load_one_mortality(path)
                parser_validation.append({
                    "file": str(path),
                    "file_name": path.name,
                    "source_cycle": infer_mortality_cycle_from_name(path),
                    "n_rows": int(df.shape[0]),
                    "n_eligible": int(df.shape[0]),
                    "note": "non-DAT mortality file; fixed-width parser validation not applied",
                })
            df.columns = [normalize_colname(c) for c in df.columns]
            if "SEQN" in df:
                df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
            if "mortality_source_file" not in df.columns:
                df["mortality_source_file"] = path.name
            if "mortality_source_cycle" not in df.columns:
                df["mortality_source_cycle"] = infer_mortality_cycle_from_name(path)
            n_rows = int(df.shape[0])
            n_seqn = int(df["SEQN"].notna().sum()) if "SEQN" in df.columns else 0
            frames.append(df)
            manifests.append({
                "file": str(path),
                "file_name": path.name,
                "source_cycle": infer_mortality_cycle_from_name(path),
                "status": "ok",
                "n_rows": n_rows,
                "n_seqn_nonmissing": n_seqn,
                "error": "",
            })
        except Exception as exc:
            logger.warning("Mortality read failed | file=%s | error=%s", path.name, repr(exc))
            manifests.append({
                "file": str(path),
                "file_name": path.name,
                "source_cycle": infer_mortality_cycle_from_name(path),
                "status": "read_failed",
                "n_rows": np.nan,
                "n_seqn_nonmissing": np.nan,
                "error": str(exc),
            })
    if not frames:
        return pd.DataFrame(), pd.DataFrame(manifests), pd.DataFrame(parser_validation)
    mort = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    if "SEQN" in mort.columns:
        mort = mort[mort["SEQN"].notna()].copy()
        mort = mort.sort_values(["SEQN", "mortality_source_cycle", "mortality_source_file"]).drop_duplicates("SEQN", keep="first")
    return mort, pd.DataFrame(manifests), pd.DataFrame(parser_validation)


def standardize_mortality(mort: pd.DataFrame) -> pd.DataFrame:
    out = mort.copy()
    rename_map = {}
    for c in out.columns:
        if c.lower() == "seqn":
            rename_map[c] = "SEQN"
    out = out.rename(columns=rename_map)

    for c in ["ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_INT", "PERMTH_EXM"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "ELIGSTAT" in out.columns:
        out = out[(out["ELIGSTAT"].isna()) | (out["ELIGSTAT"] == 1)].copy()

    if "PERMTH_EXM" in out.columns:
        out["followup_months"] = out["PERMTH_EXM"]
        out["followup_time_source"] = "PERMTH_EXM"
    elif "PERMTH_INT" in out.columns:
        out["followup_months"] = out["PERMTH_INT"]
        out["followup_time_source"] = "PERMTH_INT"
    else:
        out["followup_months"] = np.nan
        out["followup_time_source"] = "missing"
    out["followup_years"] = pd.to_numeric(out["followup_months"], errors="coerce") / 12.0

    if "MORTSTAT" in out.columns:
        out["all_cause_death"] = np.where(out["MORTSTAT"] == 1, 1, np.where(out["MORTSTAT"].notna(), 0, np.nan))
    else:
        out["all_cause_death"] = np.nan

    if "UCOD_LEADING" in out.columns and "MORTSTAT" in out.columns:
        out["cvd_death"] = np.where((out["MORTSTAT"] == 1) & out["UCOD_LEADING"].isin([1, 5]), 1, np.where(out["MORTSTAT"].notna(), 0, np.nan))
        out["cancer_death"] = np.where((out["MORTSTAT"] == 1) & (out["UCOD_LEADING"] == 2), 1, np.where(out["MORTSTAT"].notna(), 0, np.nan))
    else:
        out["cvd_death"] = np.nan
        out["cancer_death"] = np.nan

    keep = [
        "SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_INT", "PERMTH_EXM",
        "followup_months", "followup_years", "followup_time_source",
        "all_cause_death", "cvd_death", "cancer_death", "mortality_source_file", "mortality_source_cycle"
    ]
    return out[[c for c in keep if c in out.columns]]


def _followup_summary(prefix: str, s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(x.shape[0]),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
    }


def make_mortality_linkage_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df is None or df.empty:
        return pd.DataFrame(rows)
    group_iter = list(df.groupby("cycle", dropna=False))
    group_iter.append(("ALL", df))
    for cycle, g in group_iter:
        linked = g["followup_years"].notna() if "followup_years" in g.columns else pd.Series(False, index=g.index)
        row = {
            "cycle": cycle,
            "analytic_n": int(g.shape[0]),
            "linked_mortality_n": int(linked.sum()),
            "linked_mortality_percent": float(linked.mean() * 100) if len(g) else np.nan,
        }
        if "followup_months" in g.columns:
            row.update(_followup_summary("followup_months_linked", g.loc[linked, "followup_months"]))
        if "followup_years" in g.columns:
            row.update(_followup_summary("followup_years_linked", g.loc[linked, "followup_years"]))
        for outcome in ["all_cause_death", "cvd_death", "cancer_death"]:
            if outcome in g.columns:
                row[f"{outcome}_events"] = int(pd.to_numeric(g.loc[linked, outcome], errors="coerce").fillna(0).sum())
            else:
                row[f"{outcome}_events"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def prepare_cox_matrix(df: pd.DataFrame, outcome: str, covars: Sequence[str]) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    needed = ["followup_years", outcome, "nhanes_bladder_tier", "survey_weight", "SDMVPSU"] + list(covars)
    existing = [c for c in needed if c in df.columns]
    d = df[existing].copy()
    d = d.dropna(subset=["followup_years", outcome, "nhanes_bladder_tier", "survey_weight"]).copy()
    d = d[(pd.to_numeric(d["followup_years"], errors="coerce") > 0) & (pd.to_numeric(d["survey_weight"], errors="coerce") > 0)].copy()
    if "SDMVPSU" not in d.columns:
        d["SDMVPSU"] = np.arange(len(d))
    covars_existing = [c for c in covars if c in d.columns]
    if covars_existing:
        d = d.dropna(subset=covars_existing).copy()

    X = make_design_matrix(d, covars_existing)
    complete = X.notna().all(axis=1)
    d = d.loc[complete].copy()
    X = X.loc[complete].copy()
    X = X.loc[:, [c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    tier_terms = [c for c in ["tier_Tier 1", "tier_Tier 2"] if c in X.columns and X[c].nunique(dropna=True) > 1]
    return X, tier_terms, d


def run_cox_models(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        return pd.DataFrame([{
            "module": "mortality", "status": "skipped_lifelines_not_installed", "error": str(exc)
        }])

    outcomes = ["all_cause_death", "cvd_death", "cancer_death"]
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for model_name, covars in MODEL_SPECS.items():
            try:
                X, tier_terms, d = prepare_cox_matrix(df, outcome, covars)
            except Exception as exc:
                rows.append({"module": "mortality", "outcome": outcome, "model": model_name, "status": "prepare_failed", "error": str(exc)})
                continue
            if d.empty:
                rows.append({"module": "mortality", "outcome": outcome, "model": model_name, "status": "skipped_empty_model_data", "n": 0, "events": 0})
                continue
            deaths = int(pd.to_numeric(d[outcome], errors="coerce").fillna(0).sum())
            if len(d) < MIN_TOTAL_N_FOR_MODEL or deaths < MIN_DEATHS_FOR_COX:
                rows.append({"module": "mortality", "outcome": outcome, "model": model_name, "status": "skipped_small_events", "n": len(d), "events": deaths})
                continue
            if not tier_terms:
                rows.append({"module": "mortality", "outcome": outcome, "model": model_name, "status": "skipped_no_tier_contrast", "n": len(d), "events": deaths})
                continue

            model_df = pd.concat([
                d[["followup_years", outcome, "survey_weight", "SDMVPSU"]],
                X,
            ], axis=1).dropna().copy()
            deaths_model = int(pd.to_numeric(model_df[outcome], errors="coerce").fillna(0).sum())
            if model_df.shape[0] < MIN_TOTAL_N_FOR_MODEL or deaths_model < MIN_DEATHS_FOR_COX:
                rows.append({"module": "mortality", "outcome": outcome, "model": model_name, "status": "skipped_small_events_after_design", "n": int(model_df.shape[0]), "events": deaths_model})
                continue
            try:
                cph = CoxPHFitter()
                cph.fit(
                    model_df,
                    duration_col="followup_years",
                    event_col=outcome,
                    weights_col="survey_weight",
                    cluster_col="SDMVPSU",
                    robust=True,
                )
                for term in tier_terms:
                    if term not in cph.summary.index:
                        continue
                    srow = cph.summary.loc[term]
                    rows.append({
                        "module": "mortality",
                        "outcome": outcome,
                        "model": model_name,
                        "term": term.replace("tier_", ""),
                        "effect_name": "HR",
                        "effect": float(np.exp(srow["coef"])),
                        "lcl": float(np.exp(srow["coef lower 95%"])),
                        "ucl": float(np.exp(srow["coef upper 95%"])),
                        "p": float(srow["p"]),
                        "n": int(model_df.shape[0]),
                        "events": deaths_model,
                        "tier0_n": int((d.loc[model_df.index, "nhanes_bladder_tier"] == "Tier 0").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "tier1_n": int((d.loc[model_df.index, "nhanes_bladder_tier"] == "Tier 1").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "tier2_n": int((d.loc[model_df.index, "nhanes_bladder_tier"] == "Tier 2").sum()) if set(model_df.index).issubset(set(d.index)) else np.nan,
                        "status": "ok",
                        "error": "",
                    })
            except Exception as exc:
                rows.append({"module": "mortality", "outcome": outcome, "model": model_name, "status": "fit_failed", "n": len(d), "events": deaths, "error": str(exc)})
    return pd.DataFrame(rows)


def make_mortality_input_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["all_cause_death", "cvd_death", "cancer_death"]
    rows = []
    linked = df["followup_years"].notna() if "followup_years" in df.columns else pd.Series(False, index=df.index)
    for outcome in outcomes:
        if outcome not in df.columns:
            rows.append({"module": "mortality", "outcome": outcome, "status": "missing_outcome"})
            continue
        for model_name, covars in MODEL_SPECS.items():
            row = {
                "module": "mortality",
                "outcome": outcome,
                "model": model_name,
                "n_analytic_total": int(len(df)),
                "n_linked_mortality": int(linked.sum()),
                "events_linked": int(pd.to_numeric(df.loc[linked, outcome], errors="coerce").fillna(0).sum()),
                "covariates_requested": ";".join(covars),
                "covariates_available": ";".join([c for c in covars if c in df.columns]),
                "covariates_missing": ";".join([c for c in covars if c not in df.columns]),
                "status": "ok",
            }
            try:
                X, tier_terms, d = prepare_cox_matrix(df, outcome, covars)
                row.update({
                    "n_model": int(len(d)),
                    "events_model": int(pd.to_numeric(d[outcome], errors="coerce").fillna(0).sum()) if outcome in d.columns else 0,
                    "mean_followup_years": float(pd.to_numeric(d["followup_years"], errors="coerce").mean()) if "followup_years" in d.columns and len(d) else np.nan,
                    "median_followup_years": float(pd.to_numeric(d["followup_years"], errors="coerce").median()) if "followup_years" in d.columns and len(d) else np.nan,
                    "min_followup_years": float(pd.to_numeric(d["followup_years"], errors="coerce").min()) if "followup_years" in d.columns and len(d) else np.nan,
                    "max_followup_years": float(pd.to_numeric(d["followup_years"], errors="coerce").max()) if "followup_years" in d.columns and len(d) else np.nan,
                    "mean_followup_months": float(pd.to_numeric(d.get("followup_months", pd.Series(dtype=float)), errors="coerce").mean()) if "followup_months" in d.columns and len(d) else np.nan,
                    "max_followup_months": float(pd.to_numeric(d.get("followup_months", pd.Series(dtype=float)), errors="coerce").max()) if "followup_months" in d.columns and len(d) else np.nan,
                    "tier0_n": int((d["nhanes_bladder_tier"] == "Tier 0").sum()) if "nhanes_bladder_tier" in d.columns else np.nan,
                    "tier1_n": int((d["nhanes_bladder_tier"] == "Tier 1").sum()) if "nhanes_bladder_tier" in d.columns else np.nan,
                    "tier2_n": int((d["nhanes_bladder_tier"] == "Tier 2").sum()) if "nhanes_bladder_tier" in d.columns else np.nan,
                    "tier_terms_available": ";".join(tier_terms),
                    "n_design_columns": int(X.shape[1]),
                })
                for c in covars:
                    if c in df.columns:
                        row[f"missing_{c}"] = int(df[c].isna().sum())
            except Exception as exc:
                row.update({"status": "prepare_failed", "error": str(exc)})
            rows.append(row)
    return pd.DataFrame(rows)

# =========================================================
# 7) Output helpers
# =========================================================
def save_outputs(tables: Dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(RUN_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    file_manifest = []
    for p in sorted(RUN_DIR.glob("*")):
        if p.is_file():
            file_manifest.append({"file": p.name, "size_bytes": p.stat().st_size})
    pd.DataFrame(file_manifest).to_csv(RUN_DIR / "file_manifest.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 8) Main
# =========================================================
def main() -> None:
    logger.info("Script version: %s", SCRIPT_VERSION)
    logger.info("NHANES root: %s", NHANES_ROOT)
    logger.info("Run directory: %s", RUN_DIR)

    raw, input_manifest, raw_inventory = load_all_nhanes()
    logger.info("Raw merged NHANES shape=%s", raw.shape)

    data = derive_covariates(raw)
    data = derive_bladder_phenotype(data)
    data = derive_biomarkers(data)
    data = derive_short_term_outcomes(data)
    data = build_survey_weight(data)

    # Late-life analytic set with valid bladder tier and survey weight.
    analytic = data.copy()
    analytic = analytic[(analytic["age"] >= AGE_MIN)]
    analytic = analytic[analytic["nhanes_bladder_tier"].notna()]
    analytic = analytic[analytic["survey_weight"].notna() & (analytic["survey_weight"] > 0)]
    analytic = derive_dual_burden_group(analytic)
    logger.info("Analytic NHANES shape=%s", analytic.shape)

    variable_coverage = make_variable_coverage(raw)
    cycle_audit = make_cycle_audit(raw, analytic)
    pheno_dist = phenotype_distribution(analytic)
    biomarker_by_tier = weighted_by_tier(analytic, BIOMARKER_OUTCOMES, "biomarker_by_tier")
    short_term_by_tier = weighted_by_tier(analytic, SHORT_TERM_OUTCOMES, "short_term_by_tier")
    dual_burden_dist = dual_burden_distribution(analytic)
    dual_burden_short_term_models = run_dual_burden_weighted_models(analytic, DUAL_BURDEN_SHORT_TERM_OUTCOMES)
    dual_burden_interaction_short_term_models, dual_burden_interaction_short_term_marginal_risks = run_interaction_weighted_models(analytic, DUAL_BURDEN_SHORT_TERM_OUTCOMES)
    dual_burden_interaction_short_term_diagnostics = make_interaction_diagnostics(analytic, DUAL_BURDEN_SHORT_TERM_OUTCOMES, "dual_burden_interaction_short_term", cox=False)

    biomarker_model_input_diagnostics = make_weighted_model_input_diagnostics(analytic, BIOMARKER_OUTCOMES, "biomarker")
    short_term_model_input_diagnostics = make_weighted_model_input_diagnostics(analytic, SHORT_TERM_OUTCOMES, "short_term")
    model_input_diagnostics = pd.concat([biomarker_model_input_diagnostics, short_term_model_input_diagnostics], ignore_index=True)

    biomarker_models = run_weighted_models(analytic, BIOMARKER_OUTCOMES, "biomarker")
    short_term_models = run_weighted_models(analytic, SHORT_TERM_OUTCOMES, "short_term")

    mortality_files = find_mortality_files(NHANES_ROOT)
    mortality_file_manifest = pd.DataFrame()
    mortality_linkage_audit = pd.DataFrame()
    mortality_models = pd.DataFrame()
    mortality_input_diagnostics = pd.DataFrame()
    mortality_parser_validation = pd.DataFrame()
    dual_burden_mortality_by_group_table = pd.DataFrame()
    dual_burden_mortality_models = pd.DataFrame()
    dual_burden_interaction_mortality_models = pd.DataFrame()
    dual_burden_interaction_mortality_diagnostics = pd.DataFrame()
    if mortality_files:
        logger.info("Loading %d linked mortality files.", len(mortality_files))
        mort_raw, mortality_file_manifest, mortality_parser_validation = load_mortality_files(mortality_files)
        mort = standardize_mortality(mort_raw)
        analytic = analytic.merge(mort, on="SEQN", how="left")
        mortality_linkage_audit = make_mortality_linkage_audit(analytic)
        mortality_input_diagnostics = make_mortality_input_diagnostics(analytic)
        mortality_models = run_cox_models(analytic)
        dual_burden_mortality_by_group_table = dual_burden_mortality_by_group(analytic)
        dual_burden_mortality_models = run_dual_burden_cox_models(analytic)
        dual_burden_interaction_mortality_models = run_interaction_cox_models(analytic)
        dual_burden_interaction_mortality_diagnostics = make_interaction_diagnostics(analytic, {"all_cause_death": "cox", "cvd_death": "cox", "cancer_death": "cox"}, "dual_burden_interaction_mortality", cox=True)
    else:
        logger.warning("No linked mortality files found. Mortality module skipped.")
        mortality_file_manifest = pd.DataFrame([{"status": "skipped_no_mortality_file"}])
        mortality_linkage_audit = pd.DataFrame([{"status": "skipped_no_mortality_file"}])
        mortality_models = pd.DataFrame([{"module": "mortality", "status": "skipped_no_mortality_file"}])
        mortality_input_diagnostics = pd.DataFrame([{"module": "mortality", "status": "skipped_no_mortality_file"}])
        mortality_parser_validation = pd.DataFrame([{"module": "mortality", "status": "skipped_no_mortality_file"}])
        dual_burden_mortality_by_group_table = pd.DataFrame([{"module": "dual_burden_mortality", "status": "skipped_no_mortality_file"}])
        dual_burden_mortality_models = pd.DataFrame([{"module": "dual_burden_mortality", "status": "skipped_no_mortality_file"}])
        dual_burden_interaction_mortality_models = pd.DataFrame([{"module": "dual_burden_interaction_mortality", "status": "skipped_no_mortality_file"}])
        dual_burden_interaction_mortality_diagnostics = pd.DataFrame([{"module": "dual_burden_interaction_mortality", "status": "skipped_no_mortality_file"}])

    mortality_status = pd.DataFrame([{
        "mortality_file_found": bool(len(mortality_files) > 0),
        "mortality_file_count": int(len(mortality_files)),
        "mortality_files": ";".join([str(p) for p in mortality_files]),
        "mortality_linked_n": int(analytic["followup_years"].notna().sum()) if "followup_years" in analytic.columns else 0,
        "all_cause_deaths": int(pd.to_numeric(analytic.get("all_cause_death", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "all_cause_death" in analytic.columns else 0,
        "cvd_deaths": int(pd.to_numeric(analytic.get("cvd_death", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "cvd_death" in analytic.columns else 0,
        "cancer_deaths": int(pd.to_numeric(analytic.get("cancer_death", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "cancer_death" in analytic.columns else 0,
        "mean_followup_years_linked": float(pd.to_numeric(analytic.loc[analytic.get("followup_years", pd.Series(index=analytic.index, dtype=float)).notna(), "followup_years"], errors="coerce").mean()) if "followup_years" in analytic.columns else np.nan,
        "median_followup_years_linked": float(pd.to_numeric(analytic.loc[analytic.get("followup_years", pd.Series(index=analytic.index, dtype=float)).notna(), "followup_years"], errors="coerce").median()) if "followup_years" in analytic.columns else np.nan,
        "max_followup_years_linked": float(pd.to_numeric(analytic.loc[analytic.get("followup_years", pd.Series(index=analytic.index, dtype=float)).notna(), "followup_years"], errors="coerce").max()) if "followup_years" in analytic.columns else np.nan,
    }])

    minimal_cols = [
        "SEQN", "cycle", "cycle_note", "represented_years", "age", "sex", "female",
        "race_ethnicity", "education", "poverty_ratio", "smoking",
        "SDMVPSU", "SDMVSTRA", "survey_weight", "base_weight_source",
        "total_represented_years_for_weight",
        "nhanes_bladder_tier", "bladder_high_burden", "biomarker_high_burden", "dual_burden_group",
        "any_ui", "frequent_ui", "stress_like_ui", "urgency_like_ui",
        "nocturia_2plus", "nocturia_3plus", "incontinence_severity_index",
        "egfr", "uacr", "uacr_log", "albuminuria", "ckd", "hba1c", "diabetes_biomarker",
        "bmi", "waist", "obesity", "crp_mg_l", "crp_log", "high_crp", "albumin", "hemoglobin",
        "biomarker_vulnerability_score", "biomarker_vulnerability_high",
        "poor_self_rated_health", "high_healthcare_use", "any_healthcare_use", "depressive_symptoms",
        "physical_limitation", "sleep_problem",
        "PERMTH_EXM", "PERMTH_INT", "followup_months", "followup_years", "followup_time_source",
        "all_cause_death", "cvd_death", "cancer_death", "mortality_source_cycle"
    ]
    minimal = analytic[[c for c in minimal_cols if c in analytic.columns]].copy()

    run_info = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(RUN_DIR),
        "nhanes_root": str(NHANES_ROOT),
        "age_min": AGE_MIN,
        "use_2017_march2020_prepandemic": USE_2017_MARCH2020_PREPANDEMIC,
        "main_scope": "NHANES biomarker contextualization, short-term burden, optional linked mortality",
        "no_figures": True,
        "no_cross_country_comparison": True,
        "weight_note": "Multi-period weights: standard 2-year WTMEC2YR scaled by 2/total_years; 2017-March 2020 WTMECPRP scaled by 3.2/total_years. The incomplete 2019-March 2020 sample is not analyzed alone.",
        "mortality_file_found": bool(len(mortality_files) > 0),
        "mortality_file_count": int(len(mortality_files)),
        "mortality_files": [str(p) for p in mortality_files],
        "model_note": "V11.2 keeps V10.1 raw-DAT-verified mortality parsing, V11 dual-burden grouping, and adds multiplicative interaction tests between Tier 2 bladder burden and high biomarker vulnerability.",
    }
    with open(RUN_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    tables = {
        "input_manifest": input_manifest,
        "raw_column_inventory": raw_inventory,
        "variable_coverage": variable_coverage,
        "cycle_audit": cycle_audit,
        "analytic_dataset_minimal": minimal,
        "phenotype_distribution": pheno_dist,
        "biomarker_by_tier": biomarker_by_tier,
        "short_term_by_tier": short_term_by_tier,
        "dual_burden_distribution": dual_burden_dist,
        "biomarker_models": biomarker_models,
        "short_term_models": short_term_models,
        "dual_burden_short_term_models": dual_burden_short_term_models,
        "dual_burden_interaction_short_term_models": dual_burden_interaction_short_term_models,
        "dual_burden_interaction_short_term_marginal_risks": dual_burden_interaction_short_term_marginal_risks,
        "dual_burden_interaction_short_term_diagnostics": dual_burden_interaction_short_term_diagnostics,
        "mortality_status": mortality_status,
        "mortality_file_manifest": mortality_file_manifest,
        "mortality_parser_validation": mortality_parser_validation,
        "mortality_linkage_audit": mortality_linkage_audit,
        "model_input_diagnostics": model_input_diagnostics,
        "mortality_input_diagnostics": mortality_input_diagnostics,
        "mortality_models": mortality_models,
        "dual_burden_mortality_by_group": dual_burden_mortality_by_group_table,
        "dual_burden_mortality_models": dual_burden_mortality_models,
        "dual_burden_interaction_mortality_models": dual_burden_interaction_mortality_models,
        "dual_burden_interaction_mortality_diagnostics": dual_burden_interaction_mortality_diagnostics,
    }
    save_outputs(tables)
    logger.info("Completed. Outputs written to %s", RUN_DIR)


if __name__ == "__main__":
    main()
