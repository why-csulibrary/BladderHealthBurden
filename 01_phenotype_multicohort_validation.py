# -*- coding: utf-8 -*-
"""
Formal manuscript-level analysis for bladder-health burden phenotypes.
Scope: HRS development cohort + cross-sectional external validation cohorts.
No figures. No longitudinal outcome analysis. No cross-country / cross-region burden comparison.

V4 clean keeps the real-variable fixed mapping from V3 and adds manuscript-clean outputs:
1) maximum-observed main analysis;
2) first-observed cross-sectional sensitivity;
3) a single toileting-related dependence outcome to avoid duplicated difficulty/assistance endpoints;
4) HRS within-cohort high resource/financial burden proxy indicators.
No figures. No longitudinal outcome analysis. No cross-country / cross-region burden comparison.
"""

from __future__ import annotations

import os
import json
import logging
import math
import re
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# =========================================================
# 0) Project paths
# =========================================================
# Place harmonised cohort CSV files under data/cohorts, or set BHB_COHORT_DATA.
# All result tables are written to outputs/phenotype unless BHB_OUTPUT_ROOT is set.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("BHB_COHORT_DATA", str(PROJECT_ROOT / "data" / "cohorts")))
OUT_ROOT = Path(os.environ.get("BHB_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))) / "phenotype"
RUN_DIR = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1) Global settings
# =========================================================
COHORTS = ["HRS", "KLoSA", "MHAS", "LASI", "ELSA", "SHARE"]
TIER_LEVELS = ["Tier 0", "Tier 1", "Tier 2"]

MODEL_SPECS = {
    "M1_age_sex": ["age", "sex"],
    "M2_age_sex_education": ["age", "sex", "education"],
    "M3_metabolic": ["age", "sex", "education", "diabetes", "hypertension"],
}

MIN_TOTAL_N_FOR_MODEL = 100
MIN_EVENTS_FOR_MODEL = 10
FREQUENT_UI_DAYS_CUTOFF = 15
SCRIPT_VERSION = "V4.2_clean_locked"

ANALYSIS_SETS = ["maximum_observed", "first_observed"]

FUNCTIONAL_OUTCOMES = ["mobility_limitation", "adl_limitation", "iadl_limitation", "toileting_dependency"]
CARE_OUTCOMES = ["homecare", "nursing_home"]
RESOURCE_OUTCOMES = ["hospitalization", "doctor_visit"]
PRIMARY_OUTCOMES = FUNCTIONAL_OUTCOMES + CARE_OUTCOMES + RESOURCE_OUTCOMES
HRS_NUMERIC_PROXY = ["oop_expense", "doctor_visits", "hospital_nights", "nursing_home_nights"]
HRS_HIGH_PROXY_OUTCOMES = [
    "high_oop_expense",
    "high_doctor_visits",
    "high_hospital_nights",
    "any_nursing_home_nights",
]

REGRESSION_COLUMNS = [
    "cohort", "tier_definition", "outcome", "model", "term", "n", "events",
    "status", "or", "ci_low", "ci_high", "p_value",
    "marginal_risk_tier0", "marginal_risk_tier", "marginal_rd",
    "marginal_excess_per_1000",
]

VALIDATION_COLUMNS = [
    "tier_definition", "outcome", "model", "term", "cohorts_with_model",
    "direction_positive", "direction_negative", "median_or", "min_or", "max_or",
    "median_marginal_excess_per_1000", "support_label",
]

MODEL_DIAGNOSTIC_COLUMNS = [
    "cohort", "tier_definition", "outcome", "n_non_missing", "events",
    "tier0_n", "tier1_n", "tier2_n", "ready_for_model",
]

# =========================================================
# 2) Fixed variable mapping based on the source data
# =========================================================
COHORT_CONFIG: Dict[str, Dict[str, object]] = {
    "HRS": {
        "file_candidates": ["HRS.csv", "hrs.csv"],
        "id": "hhidpn",
        "age": "ragey_e",
        "sex": "ragender",
        "education": "raeducl",
        "diabetes": "diabe",
        "hypertension": "hibpe",
        "core_ui": ["urinai"],
        "frequent_ui_days": "urinaf",
        "stress_like_ui": [],
        "urgency_like_ui": [],
        "pad_use": [],
        "mobility_limitation": ["walkra", "walksa", "walk1a"],
        "adl_limitation": ["adl6a"],
        "iadl_limitation": ["iadl5a"],
        "toileting_dependency": ["toilta"],
        "homecare": ["homcar"],
        "nursing_home": ["nrshom"],
        "doctor_visit": ["doctor"],
        "hospitalization": ["hosp"],
        "oop_expense": "oopmd",
        "doctor_visits": "doctim",
        "hospital_nights": "hspnit",
        "nursing_home_nights": "nrsnit",
    },
    "KLoSA": {
        "file_candidates": ["KLoSA.csv", "klosa.csv"],
        "id": "pid",
        "age": "agey",
        "sex": "ragender",
        "education": "raeducl",
        "diabetes": "diabe",
        "hypertension": "hibpe",
        "core_ui": ["urinai"],
        "frequent_ui_days": "urinaf",
        "stress_like_ui": [],
        "urgency_like_ui": [],
        "pad_use": ["urinpad"],
        "mobility_limitation": [],
        "adl_limitation": ["dressb", "bathb", "eatb", "toiltb", "bedb_k", "brushb", "groomb"],
        "iadl_limitation": ["mealsb", "shopb", "medsb", "moneyb", "phoneb", "transb", "gooutb", "laundryb", "housewkb"],
        "toileting_dependency": ["toiltb"],
        "homecare": ["homcar"],
        "nursing_home": [],
        "doctor_visit": ["doctor"],
        "hospitalization": ["hcfcl2y"],
        "oop_expense": "oopmd",
    },
    "MHAS": {
        "file_candidates": ["MHAS.csv", "mhas.csv"],
        "id": "rahhidnp",
        "age": "agey",
        "sex": "ragender",
        "education": "raeducl",
        "diabetes": "diabe",
        "hypertension": "hibpe",
        "core_ui": ["urina2y", "urinurg2y", "urincgh2y"],
        "frequent_ui_days": None,
        "stress_like_ui": ["urinurg2y"],
        "urgency_like_ui": ["urincgh2y"],
        "pad_use": [],
        "mobility_limitation": ["walkra", "walksa", "walk1a"],
        "adl_limitation": ["adltot6"],
        "iadl_limitation": ["iadlfour"],
        "toileting_dependency": ["toilta"],
        "homecare": [],
        "nursing_home": [],
        "doctor_visit": ["doctor1y"],
        "hospitalization": ["hosp1y"],
        "oop_expense": "oopmd1y",
    },
    "LASI": {
        "file_candidates": ["LASI.csv", "lasi.csv"],
        "id": "prim_key",
        "age": "r1agey",
        "sex": "ragender",
        "education": "raeducl",
        "diabetes": "r1diabe",
        "hypertension": "r1hibpe",
        "core_ui": ["r1urinae", "r1urincgh_l"],
        "frequent_ui_days": None,
        "stress_like_ui": ["r1urincgh_l"],
        "urgency_like_ui": [],
        "pad_use": [],
        "mobility_limitation": ["r1walkra", "r1walk100a"],
        "adl_limitation": ["r1adltot6"],
        "iadl_limitation": ["r1iadltot_l"],
        "toileting_dependency": ["r1toilta"],
        "homecare": [],
        "nursing_home": [],
        "doctor_visit": ["r1doctor1y"],
        "hospitalization": ["r1hosp1y"],
        "oop_expense": "r1oopmd1y_l",
    },
    "ELSA": {
        "file_candidates": ["ELSA.csv", "elsa.csv"],
        "id": "idauniqc",
        "age": "agey",
        "sex": "ragender",
        "education": "raeducl",
        "diabetes": "diabe",
        "hypertension": "hibpe",
        "core_ui": ["urinai"],
        "frequent_ui_days": None,
        "stress_like_ui": [],
        "urgency_like_ui": [],
        "pad_use": [],
        "mobility_limitation": ["walkra", "walk100a"],
        "adl_limitation": ["adltot6"],
        "iadl_limitation": ["iadltot2_e"],
        "toileting_dependency": ["toilta"],
        "homecare": [],
        "nursing_home": ["nhmliv"],
        "doctor_visit": [],
        "hospitalization": [],
        "oop_expense": None,
    },
    "SHARE": {
        "file_candidates": ["SHARE.csv", "share.csv"],
        "id": "mergeid",
        "age": "agey",
        "sex": "ragender",
        "education": "raeducl",
        "diabetes": "diabe",
        "hypertension": "hibpe",
        "core_ui": ["urinai"],
        "frequent_ui_days": None,
        "stress_like_ui": [],
        "urgency_like_ui": [],
        "pad_use": [],
        "mobility_limitation": ["walkra", "walk100a"],
        "adl_limitation": [],
        "iadl_limitation": [],
        "toileting_dependency": ["toilta"],
        "homecare": [],
        "nursing_home": [],
        "doctor_visit": ["doctor1y"],
        "hospitalization": ["hosp1y"],
        "oop_expense": "oopmd1y",
    },
}

# =========================================================
# 3) Logging
# =========================================================
# Use explicit UTF-8 logging so Chinese paths are readable on Windows and Linux.
logger = logging.getLogger("")
logger.setLevel(logging.INFO)
for h in list(logger.handlers):
    logger.removeHandler(h)
file_handler = logging.FileHandler(RUN_DIR / "run.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console)

# =========================================================
# 4) Helpers
# =========================================================
def base_name(col: str) -> str:
    return re.split(r"\s*\(", str(col).strip(), maxsplit=1)[0].strip()


def find_input_file(cohort: str, cfg: Dict[str, object]) -> Optional[Path]:
    for name in cfg["file_candidates"]:
        p = DATA_ROOT / str(name)
        if p.exists():
            return p
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.mask(x.isin([-1, -3, -7, -8, -9, -10, -11, -13, -17, -21, -23, -27, -99, 97, 98, 99, 997, 998, 999]))
    return x


def safe_exp(x: float) -> float:
    if pd.isna(x):
        return np.nan
    if x > 700:
        return np.inf
    if x < -700:
        return 0.0
    return float(math.exp(x))


def clean_string(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"nan": np.nan, "none": np.nan, "": np.nan})


def as_binary(s: pd.Series, allow_count: bool = True) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        x = safe_numeric(s)
        vals = set(pd.Series(x.dropna().unique()).astype(float).tolist())
        vals_int = set(int(v) for v in vals if float(v).is_integer())
        if vals_int.issubset({0, 1}):
            return x.map({0: 0.0, 1: 1.0})
        if vals_int.issubset({1, 2}):
            return x.map({1: 1.0, 2: 0.0})
        if vals_int.issubset({1, 5}):
            return x.map({1: 1.0, 5: 0.0})
        if allow_count:
            return (x > 0).astype(float).where(x.notna())
        return pd.Series(np.nan, index=s.index, dtype="float64")

    x = clean_string(s)
    yes_pattern = r"^(1|1\.yes|yes|y|true|t|positive|present|是|有)$"
    no_pattern = r"^(0|0\.no|2|5|no|n|false|f|negative|absent|否|无)$"
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[x.str.match(yes_pattern, na=False)] = 1.0
    out[x.str.match(no_pattern, na=False)] = 0.0
    return out


def as_female(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        x = safe_numeric(s)
        vals = set(pd.Series(x.dropna().unique()).astype(float).tolist())
        vals_int = set(int(v) for v in vals if float(v).is_integer())
        if vals_int.issubset({0, 1}):
            return x.map({0: 1.0, 1: 0.0})
        if vals_int.issubset({1, 2}):
            return x.map({1: 0.0, 2: 1.0})
        return pd.Series(np.nan, index=s.index, dtype="float64")

    x = clean_string(s)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[x.isin(["女性", "female", "woman", "f"])] = 1.0
    out[x.isin(["男性", "male", "man", "m"])] = 0.0
    return out


def as_frequency(s: pd.Series, cutoff: int = FREQUENT_UI_DAYS_CUTOFF) -> pd.Series:
    x = safe_numeric(s)
    return (x >= cutoff).astype(float).where(x.notna())


def combine_any(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    mat = df[cols]
    any_pos = (mat == 1).any(axis=1)
    any_obs = mat.notna().any(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out[any_obs] = any_pos[any_obs].astype(float)
    return out


def first_non_missing(s: pd.Series):
    x = s.dropna()
    return x.iloc[0] if len(x) else np.nan


def max_non_missing(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return x.max() if len(x) else np.nan


def read_cohort_raw(cohort: str, cfg: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    path = find_input_file(cohort, cfg)
    if path is None:
        raise FileNotFoundError(f"No input file found for {cohort} under {DATA_ROOT}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    base_to_full = {base_name(c): c for c in header}
    rows = []
    needed = set()

    for key, val in cfg.items():
        if key == "file_candidates":
            continue
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            if v is None:
                continue
            needed.add(str(v))

    for base in sorted(set(base_to_full) | needed):
        rows.append({
            "cohort": cohort,
            "base_name": base,
            "source_column": base_to_full.get(base),
            "selected": base in needed,
            "available": base in base_to_full,
        })

    usecols = [base_to_full[v] for v in needed if v in base_to_full]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    raw = raw.rename(columns={c: base_name(c) for c in raw.columns})
    return raw, pd.DataFrame(rows), path


def get_col(raw: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None or col not in raw.columns:
        return None
    return raw[col]


def get_binary(raw: pd.DataFrame, col: Optional[str], allow_count: bool = True) -> pd.Series:
    s = get_col(raw, col)
    if s is None:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return as_binary(s, allow_count=allow_count)


def build_any_from_cols(raw: pd.DataFrame, cols: List[str]) -> pd.Series:
    tmp = pd.DataFrame(index=raw.index)
    for col in cols:
        if col in raw.columns:
            tmp[col] = as_binary(raw[col], allow_count=True)
    return combine_any(tmp, list(tmp.columns))


def standardize_cohort(cohort: str, raw: pd.DataFrame, cfg: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame(index=raw.index)
    out["row_order"] = np.arange(len(raw))
    coverage_rows = []

    def record(name: str, cols: List[str]):
        for col in cols:
            coverage_rows.append({
                "cohort": cohort,
                "standard_name": name,
                "source_column": col,
                "available": col in raw.columns,
                "n_non_missing": int(raw[col].notna().sum()) if col in raw.columns else 0,
            })

    out["id"] = raw[cfg["id"]] if cfg["id"] in raw.columns else np.arange(len(raw))
    out["age"] = safe_numeric(raw[cfg["age"]]) if cfg["age"] in raw.columns else np.nan
    out["sex"] = as_female(raw[cfg["sex"]]) if cfg["sex"] in raw.columns else np.nan
    out["education"] = raw[cfg["education"]] if cfg["education"] in raw.columns else np.nan
    out["diabetes"] = get_binary(raw, cfg.get("diabetes"), allow_count=False)
    out["hypertension"] = get_binary(raw, cfg.get("hypertension"), allow_count=False)

    out["core_ui"] = build_any_from_cols(raw, cfg.get("core_ui", []))
    out["frequent_ui"] = as_frequency(raw[cfg["frequent_ui_days"]]) if cfg.get("frequent_ui_days") in raw.columns else np.nan
    out["stress_like_ui"] = build_any_from_cols(raw, cfg.get("stress_like_ui", []))
    out["urgency_like_ui"] = build_any_from_cols(raw, cfg.get("urgency_like_ui", []))
    out["pad_use"] = build_any_from_cols(raw, cfg.get("pad_use", []))

    for name in PRIMARY_OUTCOMES:
        out[name] = build_any_from_cols(raw, cfg.get(name, []))

    for name in HRS_NUMERIC_PROXY:
        source = cfg.get(name)
        out[name] = safe_numeric(raw[source]) if source in raw.columns else np.nan

    out["cohort"] = cohort

    for key, val in cfg.items():
        if key == "file_candidates":
            continue
        vals = val if isinstance(val, list) else [val]
        record(key, [str(v) for v in vals if v is not None])

    return out, pd.DataFrame(coverage_rows)


def collapse_person_level(df: pd.DataFrame, mode: str = "maximum_observed") -> pd.DataFrame:
    if mode == "first_observed":
        x = df.copy()
        if "row_order" not in x.columns:
            x["row_order"] = np.arange(len(x))
        x["_core_missing"] = x["core_ui"].isna().astype(int) if "core_ui" in x.columns else 1
        x = x.sort_values(["id", "_core_missing", "row_order"], kind="mergesort")
        first = x.drop_duplicates("id", keep="first").drop(columns=["_core_missing"], errors="ignore")
        return first.reset_index(drop=True)

    binary_cols = [
        "diabetes", "hypertension", "core_ui", "frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use",
    ] + PRIMARY_OUTCOMES
    numeric_cols = ["age"] + HRS_NUMERIC_PROXY
    first_cols = ["sex", "education", "cohort"]

    g = df.groupby("id", dropna=False, sort=False)
    parts = []

    max_cols = [c for c in binary_cols + numeric_cols if c in df.columns]
    if max_cols:
        parts.append(g[max_cols].max())

    existing_first_cols = [c for c in first_cols if c in df.columns]
    if existing_first_cols:
        parts.append(g[existing_first_cols].first())

    if not parts:
        return df[["id"]].drop_duplicates().reset_index(drop=True)

    collapsed = pd.concat(parts, axis=1).reset_index()
    return collapsed

def construct_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high_marker_cols = ["frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use"]
    symptom_only_high_cols = ["frequent_ui", "stress_like_ui", "urgency_like_ui"]

    out["high_burden_marker"] = combine_any(out, high_marker_cols)
    out["symptom_only_high_marker"] = combine_any(out, symptom_only_high_cols)

    tier = pd.Series(np.nan, index=out.index, dtype="object")
    tier[out["core_ui"] == 0] = "Tier 0"
    tier[(out["core_ui"] == 1) & (out["high_burden_marker"] != 1)] = "Tier 1"
    tier[out["high_burden_marker"] == 1] = "Tier 2"
    out["tier_primary"] = pd.Categorical(tier, categories=TIER_LEVELS, ordered=True)

    tier_symptom = pd.Series(np.nan, index=out.index, dtype="object")
    tier_symptom[out["core_ui"] == 0] = "Tier 0"
    tier_symptom[(out["core_ui"] == 1) & (out["symptom_only_high_marker"] != 1)] = "Tier 1"
    tier_symptom[out["symptom_only_high_marker"] == 1] = "Tier 2"
    out["tier_symptom_only"] = pd.Categorical(tier_symptom, categories=TIER_LEVELS, ordered=True)

    return out

# =========================================================
# 5) Summaries
# =========================================================
def summarize_phenotype_distribution(df: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("cohort"):
        denom = int(g[tier_col].notna().sum())
        for tier in TIER_LEVELS:
            n = int((g[tier_col] == tier).sum())
            rows.append({
                "cohort": cohort,
                "tier_definition": tier_col,
                "tier": tier,
                "n": n,
                "denominator": denom,
                "percent": 100 * n / denom if denom > 0 else np.nan,
            })
        rows.append({
            "cohort": cohort,
            "tier_definition": tier_col,
            "tier": "Core UI positive",
            "n": int((g["core_ui"] == 1).sum()),
            "denominator": int(g["core_ui"].notna().sum()),
            "percent": 100 * (g["core_ui"] == 1).sum() / g["core_ui"].notna().sum() if g["core_ui"].notna().sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_covariates_by_tier(df: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("cohort"):
        for tier in TIER_LEVELS:
            gt = g[g[tier_col] == tier]
            for var in ["age", "sex", "diabetes", "hypertension"]:
                x = pd.to_numeric(gt[var], errors="coerce") if var in gt.columns else pd.Series(dtype="float64")
                rows.append({
                    "cohort": cohort, "tier_definition": tier_col, "tier": tier,
                    "variable": var, "n_non_missing": int(x.notna().sum()),
                    "mean_or_prevalence": float(x.mean()) if x.notna().sum() else np.nan,
                    "sd": float(x.std()) if x.notna().sum() > 1 else np.nan,
                })
            if "education" in gt.columns:
                vc = gt["education"].value_counts(dropna=False)
                for level, n in vc.items():
                    rows.append({
                        "cohort": cohort, "tier_definition": tier_col, "tier": tier,
                        "variable": "education", "level": level,
                        "n": int(n), "percent": 100 * n / len(gt) if len(gt) else np.nan,
                    })
    return pd.DataFrame(rows)


def summarize_outcomes_by_tier(df: pd.DataFrame, tier_col: str, outcomes: List[str]) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("cohort"):
        for outcome in outcomes:
            if outcome not in g.columns:
                continue
            for tier in TIER_LEVELS:
                gt = g[g[tier_col] == tier]
                x = gt[outcome]
                rows.append({
                    "cohort": cohort,
                    "tier_definition": tier_col,
                    "outcome": outcome,
                    "tier": tier,
                    "n_non_missing": int(x.notna().sum()),
                    "events": int((x == 1).sum()),
                    "prevalence": float(x.mean()) if x.notna().sum() else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_unadjusted_gradient(prev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if prev.empty:
        return pd.DataFrame(rows)
    for (cohort, tier_def, outcome), g in prev.groupby(["cohort", "tier_definition", "outcome"]):
        ref = g[g["tier"] == "Tier 0"]
        if ref.empty or pd.isna(ref.iloc[0]["prevalence"]):
            continue
        p0 = ref.iloc[0]["prevalence"]
        for tier in ["Tier 1", "Tier 2"]:
            row = g[g["tier"] == tier]
            if row.empty or pd.isna(row.iloc[0]["prevalence"]):
                continue
            p1 = row.iloc[0]["prevalence"]
            rows.append({
                "cohort": cohort,
                "tier_definition": tier_def,
                "outcome": outcome,
                "contrast": f"{tier} vs Tier 0",
                "prevalence_tier": p1,
                "prevalence_tier0": p0,
                "prevalence_difference": p1 - p0,
                "excess_per_1000": 1000 * (p1 - p0),
                "prevalence_ratio": p1 / p0 if p0 > 0 else np.nan,
            })
    return pd.DataFrame(rows)

# =========================================================
# 6) Regression
# =========================================================
def prepare_design_matrix(data: pd.DataFrame, tier_col: str, covars: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    use_cols = ["outcome_y", tier_col] + [c for c in covars if c in data.columns]
    d = data[use_cols].copy()
    d = d.dropna(subset=["outcome_y", tier_col])
    d = d[d[tier_col].isin(TIER_LEVELS)]

    y = d["outcome_y"].astype(float)
    X = pd.DataFrame(index=d.index)
    tier_terms = []
    for col_name, tier_name in [("tier_Tier1", "Tier 1"), ("tier_Tier2", "Tier 2")]:
        z = (d[tier_col] == tier_name).astype(float)
        if z.nunique(dropna=True) > 1:
            X[col_name] = z
            tier_terms.append(col_name)

    for c in covars:
        if c not in d.columns:
            continue
        if c in ["age", "sex", "diabetes", "hypertension"]:
            X[c] = pd.to_numeric(d[c], errors="coerce")
        elif c == "education":
            dd = pd.get_dummies(d[c].astype("category"), prefix=c, drop_first=True, dummy_na=False)
            X = pd.concat([X, dd.astype(float)], axis=1)

    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep]
    y = y.loc[keep]

    # Recheck tier contrasts after covariate filtering.
    # A tier dummy can have variation before filtering but become all-zero after
    # dropping rows with missing covariates, which otherwise produces degenerate
    # GLM rows such as OR=1, CI=1, and non-informative p-values.
    valid_tier_terms = []
    for term in tier_terms:
        if term in X.columns and X[term].nunique(dropna=True) > 1:
            valid_tier_terms.append(term)
        elif term in X.columns:
            X = X.drop(columns=[term])
    tier_terms = valid_tier_terms

    X = sm.add_constant(X, has_constant="add")
    return X, y, tier_terms


def fit_logistic_model(data: pd.DataFrame, cohort: str, tier_col: str, outcome: str, model_name: str, covars: List[str]) -> List[dict]:
    d = data.copy()
    d["outcome_y"] = d[outcome]
    d = d[d["outcome_y"].notna() & d[tier_col].notna()]
    events = int((d["outcome_y"] == 1).sum())

    if len(d) < MIN_TOTAL_N_FOR_MODEL or events < MIN_EVENTS_FOR_MODEL or d["outcome_y"].nunique(dropna=True) < 2:
        return [{
            "cohort": cohort, "tier_definition": tier_col, "outcome": outcome, "model": model_name,
            "term": "model", "n": len(d), "events": events, "status": "skipped_small_or_no_variation",
        }]

    try:
        X, y, tier_terms = prepare_design_matrix(d, tier_col, covars)
        if len(y) < MIN_TOTAL_N_FOR_MODEL or int(y.sum()) < MIN_EVENTS_FOR_MODEL or y.nunique() < 2:
            return [{
                "cohort": cohort, "tier_definition": tier_col, "outcome": outcome, "model": model_name,
                "term": "model", "n": len(y), "events": int(y.sum()), "status": "skipped_after_covariate_filter",
            }]
        if not tier_terms:
            return [{
                "cohort": cohort, "tier_definition": tier_col, "outcome": outcome, "model": model_name,
                "term": "model", "n": len(y), "events": int(y.sum()), "status": "skipped_no_tier_contrast",
            }]

        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=50, disp=0)
        x0 = X.copy()
        for term0 in tier_terms:
            x0[term0] = 0.0
        p0 = float(fit.predict(x0).mean())

        rows = []
        for term, label in [("tier_Tier1", "Tier 1 vs Tier 0"), ("tier_Tier2", "Tier 2 vs Tier 0")]:
            if term not in tier_terms or term not in fit.params.index:
                continue
            beta = float(fit.params[term])
            se = float(fit.bse[term])
            xk = X.copy()
            for term0 in tier_terms:
                xk[term0] = 0.0
            xk[term] = 1.0
            pk = float(fit.predict(xk).mean())
            rows.append({
                "cohort": cohort,
                "tier_definition": tier_col,
                "outcome": outcome,
                "model": model_name,
                "term": label,
                "n": int(len(y)),
                "events": int(y.sum()),
                "status": "ok",
                "or": safe_exp(beta),
                "ci_low": safe_exp(beta - 1.96 * se),
                "ci_high": safe_exp(beta + 1.96 * se),
                "p_value": float(fit.pvalues[term]),
                "marginal_risk_tier0": p0,
                "marginal_risk_tier": pk,
                "marginal_rd": pk - p0,
                "marginal_excess_per_1000": 1000 * (pk - p0),
            })
        return rows
    except Exception as e:
        logging.warning("Model failed | %s | %s | %s | %s | %s", cohort, tier_col, outcome, model_name, e)
        return [{
            "cohort": cohort, "tier_definition": tier_col, "outcome": outcome, "model": model_name,
            "term": "model", "n": len(d), "events": events, "status": f"failed: {e}",
        }]


def model_input_diagnostics(df: pd.DataFrame, tier_col: str, outcomes: List[str]) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("cohort"):
        for outcome in outcomes:
            if outcome not in g.columns:
                rows.append({
                    "cohort": cohort,
                    "tier_definition": tier_col,
                    "outcome": outcome,
                    "n_non_missing": 0,
                    "events": 0,
                    "tier0_n": int((g[tier_col] == "Tier 0").sum()) if tier_col in g.columns else 0,
                    "tier1_n": int((g[tier_col] == "Tier 1").sum()) if tier_col in g.columns else 0,
                    "tier2_n": int((g[tier_col] == "Tier 2").sum()) if tier_col in g.columns else 0,
                    "ready_for_model": False,
                })
                continue
            d = g[g[outcome].notna() & g[tier_col].notna()]
            events = int((d[outcome] == 1).sum())
            rows.append({
                "cohort": cohort,
                "tier_definition": tier_col,
                "outcome": outcome,
                "n_non_missing": int(len(d)),
                "events": events,
                "tier0_n": int((d[tier_col] == "Tier 0").sum()),
                "tier1_n": int((d[tier_col] == "Tier 1").sum()),
                "tier2_n": int((d[tier_col] == "Tier 2").sum()),
                "ready_for_model": len(d) >= MIN_TOTAL_N_FOR_MODEL and events >= MIN_EVENTS_FOR_MODEL and d[outcome].nunique(dropna=True) >= 2,
            })
    return pd.DataFrame(rows, columns=MODEL_DIAGNOSTIC_COLUMNS)


def run_regression_models(df: pd.DataFrame, tier_col: str, outcomes: List[str]) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("cohort"):
        for outcome in outcomes:
            if outcome not in g.columns or g[outcome].notna().sum() == 0:
                continue
            for model_name, covars in MODEL_SPECS.items():
                available_covars = [c for c in covars if c in g.columns and g[c].notna().sum() > 0]
                rows.extend(fit_logistic_model(g, cohort, tier_col, outcome, model_name, available_covars))
    if not rows:
        logging.warning("No regression rows were generated for %s. Check variable_coverage and model_input_diagnostics.", tier_col)
        return pd.DataFrame(columns=REGRESSION_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=REGRESSION_COLUMNS)


def summarize_validation(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty or "status" not in reg.columns:
        logging.warning("Validation summary skipped because regression_models is empty.")
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    ok = reg[(reg["status"] == "ok") & reg["or"].notna() & reg["marginal_rd"].notna()].copy()
    if ok.empty:
        logging.warning("Validation summary skipped because no successful regression rows were available.")
        return pd.DataFrame(columns=VALIDATION_COLUMNS)

    rows = []
    for (tier_def, outcome, model, term), g in ok.groupby(["tier_definition", "outcome", "model", "term"]):
        direction_positive = int((g["marginal_rd"] > 0).sum())
        direction_negative = int((g["marginal_rd"] < 0).sum())
        n_cohort = int(g["cohort"].nunique())
        if direction_positive >= 4 and direction_negative == 0:
            support = "strong_directional_support"
        elif direction_positive >= 3 and direction_positive > direction_negative:
            support = "moderate_directional_support"
        elif direction_positive > direction_negative:
            support = "weak_directional_support"
        else:
            support = "inconsistent_or_no_support"
        rows.append({
            "tier_definition": tier_def,
            "outcome": outcome,
            "model": model,
            "term": term,
            "cohorts_with_model": n_cohort,
            "direction_positive": direction_positive,
            "direction_negative": direction_negative,
            "median_or": float(g["or"].median()),
            "min_or": float(g["or"].min()),
            "max_or": float(g["or"].max()),
            "median_marginal_excess_per_1000": float(g["marginal_excess_per_1000"].median()),
            "support_label": support,
        })
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def summarize_hrs_numeric_proxy(df: pd.DataFrame) -> pd.DataFrame:
    hrs = df[df["cohort"] == "HRS"].copy()
    rows = []
    for var in HRS_NUMERIC_PROXY:
        if var not in hrs.columns or hrs[var].notna().sum() == 0:
            continue
        for tier in TIER_LEVELS:
            x = hrs.loc[hrs["tier_primary"] == tier, var].dropna()
            rows.append({
                "cohort": "HRS",
                "tier_definition": "tier_primary",
                "variable": var,
                "tier": tier,
                "n_non_missing": int(x.shape[0]),
                "mean": float(x.mean()) if len(x) else np.nan,
                "sd": float(x.std()) if len(x) > 1 else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "q25": float(x.quantile(0.25)) if len(x) else np.nan,
                "q75": float(x.quantile(0.75)) if len(x) else np.nan,
            })
    return pd.DataFrame(rows)



def add_analysis_set(df: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "analysis_set", analysis_set)
    return out


def add_hrs_high_burden_proxies(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    threshold_rows = []
    for new_col in HRS_HIGH_PROXY_OUTCOMES:
        out[new_col] = np.nan

    for analysis_set in sorted(out["analysis_set"].dropna().unique()):
        hrs_mask = (out["analysis_set"] == analysis_set) & (out["cohort"] == "HRS")
        hrs = out.loc[hrs_mask]

        for source, new_col in [
            ("oop_expense", "high_oop_expense"),
            ("doctor_visits", "high_doctor_visits"),
            ("hospital_nights", "high_hospital_nights"),
        ]:
            x = pd.to_numeric(hrs[source], errors="coerce") if source in hrs.columns else pd.Series(dtype="float64")
            threshold = float(x.dropna().quantile(0.75)) if x.notna().sum() else np.nan
            valid = hrs_mask & out[source].notna() if source in out.columns else pd.Series(False, index=out.index)
            threshold_type = "within-HRS q75"
            if not pd.isna(threshold):
                if threshold <= 0:
                    out.loc[valid, new_col] = (pd.to_numeric(out.loc[valid, source], errors="coerce") > 0).astype(float)
                    threshold_type = "q75<=0; >0 used"
                else:
                    out.loc[valid, new_col] = (pd.to_numeric(out.loc[valid, source], errors="coerce") >= threshold).astype(float)
            threshold_rows.append({
                "analysis_set": analysis_set,
                "cohort": "HRS",
                "source_variable": source,
                "high_burden_variable": new_col,
                "threshold_type": threshold_type,
                "threshold_value": threshold,
                "n_non_missing": int(x.notna().sum()),
            })

        source = "nursing_home_nights"
        x = pd.to_numeric(hrs[source], errors="coerce") if source in hrs.columns else pd.Series(dtype="float64")
        valid = hrs_mask & out[source].notna() if source in out.columns else pd.Series(False, index=out.index)
        out.loc[valid, "any_nursing_home_nights"] = (pd.to_numeric(out.loc[valid, source], errors="coerce") > 0).astype(float)
        threshold_rows.append({
            "analysis_set": analysis_set,
            "cohort": "HRS",
            "source_variable": source,
            "high_burden_variable": "any_nursing_home_nights",
            "threshold_type": ">0 nights",
            "threshold_value": 0.0,
            "n_non_missing": int(x.notna().sum()),
        })

    return out, pd.DataFrame(threshold_rows)


def summarize_validation_scope(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    high_cols = ["frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use"]
    for (analysis_set, cohort), g in df.groupby(["analysis_set", "cohort"]):
        tier2_n = int((g["tier_primary"] == "Tier 2").sum())
        high_marker_observed = int(g[high_cols].notna().any(axis=1).sum()) if all(c in g.columns for c in high_cols) else 0
        scope = "full_tier_validation" if tier2_n > 0 else "core_only_validation"
        rows.append({
            "analysis_set": analysis_set,
            "cohort": cohort,
            "core_ui_non_missing": int(g["core_ui"].notna().sum()),
            "core_ui_positive": int((g["core_ui"] == 1).sum()),
            "tier0_n": int((g["tier_primary"] == "Tier 0").sum()),
            "tier1_n": int((g["tier_primary"] == "Tier 1").sum()),
            "tier2_n": tier2_n,
            "high_marker_observed_n": high_marker_observed,
            "validation_scope": scope,
        })
    return pd.DataFrame(rows)


def make_main_tables(
    phenotype_distribution: pd.DataFrame,
    validation_summary: pd.DataFrame,
    regression_models: pd.DataFrame,
    hrs_high_proxy_models: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_pheno = phenotype_distribution[
        (phenotype_distribution["analysis_set"] == "maximum_observed")
        & (phenotype_distribution["tier_definition"] == "tier_primary")
    ].copy()

    main_validation = validation_summary[
        (validation_summary["analysis_set"] == "maximum_observed")
        & (validation_summary["tier_definition"] == "tier_primary")
        & (validation_summary["model"] == "M3_metabolic")
    ].copy() if not validation_summary.empty else pd.DataFrame(columns=validation_summary.columns)

    main_hrs_proxy = hrs_high_proxy_models[
        (hrs_high_proxy_models["analysis_set"] == "maximum_observed")
        & (hrs_high_proxy_models["tier_definition"] == "tier_primary")
        & (hrs_high_proxy_models["model"] == "M3_metabolic")
        & (hrs_high_proxy_models["status"] == "ok")
    ].copy() if not hrs_high_proxy_models.empty else pd.DataFrame(columns=hrs_high_proxy_models.columns)

    return main_pheno, main_validation, main_hrs_proxy

def add_script_version(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "script_version" not in out.columns:
        out.insert(0, "script_version", SCRIPT_VERSION)
    return out


def make_file_manifest(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        rows.append({
            "script_version": SCRIPT_VERSION,
            "table_name": name,
            "csv_file": f"{name}.csv",
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "has_analysis_set": "analysis_set" in df.columns,
            "has_tier_definition": "tier_definition" in df.columns,
            "has_outcome": "outcome" in df.columns,
            "columns": ";".join(map(str, df.columns.tolist())),
        })
    return pd.DataFrame(rows)


def make_run_consistency_check(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    old_outcomes = {"toileting_difficulty", "toileting_assistance"}
    for name, df in tables.items():
        row = {
            "script_version": SCRIPT_VERSION,
            "table_name": name,
            "n_rows": int(df.shape[0]),
            "has_analysis_set": "analysis_set" in df.columns,
            "analysis_sets": ",".join(sorted(map(str, df["analysis_set"].dropna().unique()))) if "analysis_set" in df.columns else "",
            "has_old_toileting_outcomes": False,
            "has_toileting_dependency": False,
        }
        if "outcome" in df.columns:
            outcomes = set(map(str, df["outcome"].dropna().unique()))
            row["has_old_toileting_outcomes"] = bool(outcomes & old_outcomes)
            row["has_toileting_dependency"] = "toileting_dependency" in outcomes
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(tables: Dict[str, pd.DataFrame]) -> None:
    versioned = {name: add_script_version(df) for name, df in tables.items()}
    versioned["file_manifest"] = make_file_manifest(versioned)
    versioned["run_consistency_check"] = make_run_consistency_check(versioned)

    for name, df in versioned.items():
        df.to_csv(RUN_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    xlsx_path = RUN_DIR / "phenotype_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in versioned.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

# =========================================================
# 7) Main
# =========================================================
def main() -> None:
    logging.info("Run directory: %s", RUN_DIR)
    logging.info("Data root: %s", DATA_ROOT)

    cohort_frames = []
    coverage_frames = []
    raw_inventory_frames = []
    manifest_rows = []

    for cohort in COHORTS:
        cfg = COHORT_CONFIG[cohort]
        input_path = find_input_file(cohort, cfg)
        logging.info("Loading %s from %s", cohort, input_path)
        raw, raw_inventory, path = read_cohort_raw(cohort, cfg)
        logging.info("Loaded %s from %s | raw shape=%s", cohort, path, raw.shape)

        standardized, coverage = standardize_cohort(cohort, raw, cfg)
        for analysis_set in ANALYSIS_SETS:
            person = collapse_person_level(standardized, mode=analysis_set)
            person = construct_phenotypes(person)
            person.insert(0, "analysis_set", analysis_set)
            cohort_frames.append(person)
            logging.info("Standardized %s | %s shape=%s", cohort, analysis_set, person.shape)

        coverage_frames.append(coverage)
        raw_inventory_frames.append(raw_inventory)
        manifest_rows.append({
            "cohort": cohort,
            "path": str(path),
            "raw_rows": int(raw.shape[0]),
            "raw_cols_loaded": int(raw.shape[1]),
            "maximum_observed_rows": int(cohort_frames[-2].shape[0]),
            "first_observed_rows": int(cohort_frames[-1].shape[0]),
        })
        del raw, standardized
        gc.collect()

    analytic = pd.concat(cohort_frames, axis=0, ignore_index=True)
    analytic, hrs_high_proxy_thresholds = add_hrs_high_burden_proxies(analytic)
    variable_coverage = pd.concat(coverage_frames, axis=0, ignore_index=True)
    raw_column_inventory = pd.concat(raw_inventory_frames, axis=0, ignore_index=True)
    input_manifest = pd.DataFrame(manifest_rows)
    validation_scope = summarize_validation_scope(analytic)

    phenotype_parts = []
    covariate_parts = []
    outcome_prev_parts = []
    burden_parts = []
    diagnostic_parts = []
    regression_parts = []
    validation_parts = []
    hrs_numeric_parts = []
    hrs_high_prev_parts = []
    hrs_high_model_parts = []

    for analysis_set in ANALYSIS_SETS:
        a = analytic[analytic["analysis_set"] == analysis_set].copy()

        phenotype_primary = add_analysis_set(summarize_phenotype_distribution(a, "tier_primary"), analysis_set)
        phenotype_symptom = add_analysis_set(summarize_phenotype_distribution(a, "tier_symptom_only"), analysis_set)
        phenotype_distribution_set = pd.concat([phenotype_primary, phenotype_symptom], axis=0, ignore_index=True)
        phenotype_parts.append(phenotype_distribution_set)

        covariate_parts.append(add_analysis_set(summarize_covariates_by_tier(a, "tier_primary"), analysis_set))

        prev_primary = add_analysis_set(summarize_outcomes_by_tier(a, "tier_primary", PRIMARY_OUTCOMES), analysis_set)
        prev_symptom = add_analysis_set(summarize_outcomes_by_tier(a, "tier_symptom_only", PRIMARY_OUTCOMES), analysis_set)
        outcome_prev_set = pd.concat([prev_primary, prev_symptom], axis=0, ignore_index=True)
        outcome_prev_parts.append(outcome_prev_set)
        burden_parts.append(add_analysis_set(summarize_unadjusted_gradient(outcome_prev_set.drop(columns=["analysis_set"])), analysis_set))

        diag_primary = add_analysis_set(model_input_diagnostics(a, "tier_primary", PRIMARY_OUTCOMES), analysis_set)
        diag_symptom = add_analysis_set(model_input_diagnostics(a, "tier_symptom_only", PRIMARY_OUTCOMES), analysis_set)
        diagnostic_parts.append(pd.concat([diag_primary, diag_symptom], axis=0, ignore_index=True))

        logging.info("Running regression models for %s | tier_primary", analysis_set)
        reg_primary = run_regression_models(a, "tier_primary", PRIMARY_OUTCOMES)
        logging.info("Running regression models for %s | tier_symptom_only", analysis_set)
        reg_symptom = run_regression_models(a, "tier_symptom_only", PRIMARY_OUTCOMES)
        reg_set = pd.concat([reg_primary, reg_symptom], axis=0, ignore_index=True).reindex(columns=REGRESSION_COLUMNS)
        reg_set = add_analysis_set(reg_set, analysis_set)
        regression_parts.append(reg_set)
        val_set = summarize_validation(reg_set.drop(columns=["analysis_set"]))
        validation_parts.append(add_analysis_set(val_set, analysis_set))

        hrs_numeric_parts.append(add_analysis_set(summarize_hrs_numeric_proxy(a), analysis_set))

        hrs = a[a["cohort"] == "HRS"].copy()
        hrs_high_prev_parts.append(add_analysis_set(summarize_outcomes_by_tier(hrs, "tier_primary", HRS_HIGH_PROXY_OUTCOMES), analysis_set))
        hrs_high_reg = run_regression_models(hrs, "tier_primary", HRS_HIGH_PROXY_OUTCOMES)
        hrs_high_model_parts.append(add_analysis_set(hrs_high_reg, analysis_set))

    phenotype_distribution = pd.concat(phenotype_parts, axis=0, ignore_index=True)
    covariates_by_tier = pd.concat(covariate_parts, axis=0, ignore_index=True)
    outcome_prevalence = pd.concat(outcome_prev_parts, axis=0, ignore_index=True)
    burden_gradient = pd.concat(burden_parts, axis=0, ignore_index=True)
    model_diagnostics = pd.concat(diagnostic_parts, axis=0, ignore_index=True)
    regression_models = pd.concat(regression_parts, axis=0, ignore_index=True)
    validation_summary = pd.concat(validation_parts, axis=0, ignore_index=True)
    hrs_numeric_proxy_summary = pd.concat(hrs_numeric_parts, axis=0, ignore_index=True)
    hrs_high_proxy_prevalence = pd.concat(hrs_high_prev_parts, axis=0, ignore_index=True)
    hrs_high_proxy_models = pd.concat(hrs_high_model_parts, axis=0, ignore_index=True)

    main_table_phenotype_distribution, main_table_validation_results, main_table_hrs_resource_proxy = make_main_tables(
        phenotype_distribution, validation_summary, regression_models, hrs_high_proxy_models
    )

    sensitivity_table_phenotype_distribution_first_observed = phenotype_distribution[
        (phenotype_distribution["analysis_set"] == "first_observed")
        & (phenotype_distribution["tier_definition"] == "tier_primary")
    ].copy()
    sensitivity_table_validation_results_first_observed = validation_summary[
        (validation_summary["analysis_set"] == "first_observed")
        & (validation_summary["tier_definition"] == "tier_primary")
        & (validation_summary["model"] == "M3_metabolic")
    ].copy() if not validation_summary.empty else pd.DataFrame(columns=validation_summary.columns)
    sensitivity_table_hrs_resource_proxy_first_observed = hrs_high_proxy_models[
        (hrs_high_proxy_models["analysis_set"] == "first_observed")
        & (hrs_high_proxy_models["tier_definition"] == "tier_primary")
        & (hrs_high_proxy_models["model"] == "M3_metabolic")
        & (hrs_high_proxy_models["status"] == "ok")
    ].copy() if not hrs_high_proxy_models.empty else pd.DataFrame(columns=hrs_high_proxy_models.columns)

    minimal_cols = [
        "analysis_set", "cohort", "id", "age", "sex", "education", "diabetes", "hypertension",
        "core_ui", "frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use",
        "high_burden_marker", "symptom_only_high_marker", "tier_primary", "tier_symptom_only",
    ] + PRIMARY_OUTCOMES + HRS_NUMERIC_PROXY + HRS_HIGH_PROXY_OUTCOMES
    minimal_cols = [c for c in minimal_cols if c in analytic.columns]

    tables = {
        "input_manifest": input_manifest,
        "raw_column_inventory": raw_column_inventory,
        "variable_coverage": variable_coverage,
        "validation_scope": validation_scope,
        "analytic_dataset_minimal": analytic[minimal_cols],
        "phenotype_distribution": phenotype_distribution,
        "covariates_by_tier": covariates_by_tier,
        "outcome_prevalence_by_tier": outcome_prevalence,
        "burden_gradient_unadjusted": burden_gradient,
        "model_input_diagnostics": model_diagnostics,
        "regression_models": regression_models,
        "validation_summary": validation_summary,
        "hrs_numeric_proxy_summary": hrs_numeric_proxy_summary,
        "hrs_high_proxy_thresholds": hrs_high_proxy_thresholds,
        "hrs_high_proxy_prevalence": hrs_high_proxy_prevalence,
        "hrs_high_proxy_models": hrs_high_proxy_models,
        "main_table_phenotype_distribution": main_table_phenotype_distribution,
        "main_table_validation_results": main_table_validation_results,
        "main_table_hrs_resource_proxy": main_table_hrs_resource_proxy,
        "sensitivity_table_phenotype_distribution_first_observed": sensitivity_table_phenotype_distribution_first_observed,
        "sensitivity_table_validation_results_first_observed": sensitivity_table_validation_results_first_observed,
        "sensitivity_table_hrs_resource_proxy_first_observed": sensitivity_table_hrs_resource_proxy_first_observed,
    }
    write_outputs(tables)

    run_info = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(RUN_DIR),
        "data_root": str(DATA_ROOT),
        "cohorts": COHORTS,
        "analysis_sets": ANALYSIS_SETS,
        "tier_levels": TIER_LEVELS,
        "frequent_ui_days_cutoff": FREQUENT_UI_DAYS_CUTOFF,
        "model_specs": MODEL_SPECS,
        "min_total_n_for_model": MIN_TOTAL_N_FOR_MODEL,
        "min_events_for_model": MIN_EVENTS_FOR_MODEL,
        "note": "V4.2 clean locked. Maximum-observed main analysis plus first-observed sensitivity. Toileting outcomes are collapsed to toileting_dependency. Tier contrast terms are rechecked after covariate filtering. No figures. No longitudinal outcome analysis. No cross-country/regional burden comparison.",
    }
    with open(RUN_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    logging.info("Completed. Outputs written to %s", RUN_DIR)


if __name__ == "__main__":
    main()
