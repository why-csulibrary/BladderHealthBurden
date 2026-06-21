# -*- coding: utf-8 -*-
"""
HRS longitudinal enhancement for bladder-health burden phenotypes.

Scope:
1) HRS baseline-to-future care-dependency validation;
2) baseline bladder-health burden tier -> incident functional decline / care dependency;
3) pooled logistic discrete-time models;
4) compact clinical utility analysis for future high care-dependency risk.

No figures.
No cross-country comparison.
No modification to the main multi-cohort phenotype script.
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

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# =========================================================
# 0) Project paths
# =========================================================
# Place the HRS CSV file under data/cohorts, or set BHB_COHORT_DATA.
# Longitudinal outputs are written to outputs/hrs_longitudinal unless BHB_OUTPUT_ROOT is set.
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = Path(os.environ.get("BHB_COHORT_DATA", str(PROJECT_ROOT / "data" / "cohorts")))
OUT_ROOT = Path(os.environ.get("BHB_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))) / "hrs_longitudinal"
RUN_DIR = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1) Global settings
# =========================================================
SCRIPT_VERSION = "HRS_enhance_V1.0"

TIER_LEVELS = ["Tier 0", "Tier 1", "Tier 2"]
FREQUENT_UI_DAYS_CUTOFF = 15

MIN_TOTAL_N_FOR_MODEL = 100
MIN_EVENTS_FOR_MODEL = 10

MODEL_SPECS = {
    "M1_age_sex": ["age", "sex"],
    "M2_age_sex_education": ["age", "sex", "education"],
    "M3_metabolic": ["age", "sex", "education", "diabetes", "hypertension"],
}

HRS_CONFIG: Dict[str, object] = {
    "file_candidates": ["HRS.csv", "hrs.csv"],
    "id": "hhidpn",
    "wave_candidates": ["wave", "rwave", "year", "intyear", "interview_year"],
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
    "adl_limitation": ["adl6a"],
    "iadl_limitation": ["iadl5a"],
    "homecare": ["homcar"],
    "nursing_home": ["nrshom"],
    "hospitalization": ["hosp"],
}

INCIDENT_OUTCOMES = [
    "adl_limitation",
    "iadl_limitation",
    "homecare",
    "nursing_home",
    "adl_iadl_composite",
    "high_care_dependency_composite",
]

REGRESSION_COLUMNS = [
    "analysis", "outcome", "model", "term", "n_persons", "n_intervals", "events",
    "status", "or", "ci_low", "ci_high", "p_value",
    "marginal_risk_tier0", "marginal_risk_tier", "marginal_rd",
    "marginal_excess_per_1000",
]

UTILITY_COLUMNS = [
    "outcome", "model", "n", "events", "status",
    "auc", "calibration_intercept", "calibration_slope",
]

DCA_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


# =========================================================
# 2) Logging
# =========================================================
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
# 3) Helpers
# =========================================================
def base_name(col: str) -> str:
    return re.split(r"\s*\(", str(col).strip(), maxsplit=1)[0].strip()


def find_input_file(cfg: Dict[str, object]) -> Optional[Path]:
    for name in cfg["file_candidates"]:
        p = DATA_ROOT / str(name)
        if p.exists():
            return p
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.mask(x.isin([
        -1, -3, -7, -8, -9, -10, -11, -13, -17, -21, -23, -27,
        -99, 97, 98, 99, 997, 998, 999,
    ]))
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
    return s.astype(str).str.strip().str.lower().replace({
        "nan": np.nan,
        "none": np.nan,
        "": np.nan,
    })


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


# =========================================================
# 4) Read and standardize HRS
# =========================================================
def read_hrs_raw(cfg: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    path = find_input_file(cfg)
    if path is None:
        raise FileNotFoundError(f"No HRS input file found under {DATA_ROOT}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    base_to_full = {base_name(c): c for c in header}

    needed = set()
    for key, val in cfg.items():
        if key in ["file_candidates", "wave_candidates"]:
            continue
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            if v is not None:
                needed.add(str(v))

    for wave_col in cfg["wave_candidates"]:
        if wave_col in base_to_full:
            needed.add(str(wave_col))

    rows = []
    for base in sorted(set(base_to_full) | needed):
        rows.append({
            "base_name": base,
            "source_column": base_to_full.get(base),
            "selected": base in needed,
            "available": base in base_to_full,
        })

    usecols = [base_to_full[v] for v in needed if v in base_to_full]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    raw = raw.rename(columns={c: base_name(c) for c in raw.columns})
    return raw, pd.DataFrame(rows), path


def get_wave_column(raw: pd.DataFrame, cfg: Dict[str, object]) -> Optional[str]:
    for col in cfg["wave_candidates"]:
        if col in raw.columns:
            return str(col)
    return None


def standardize_hrs(raw: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["row_order"] = np.arange(len(raw))

    out["id"] = raw[cfg["id"]] if cfg["id"] in raw.columns else np.arange(len(raw))

    wave_col = get_wave_column(raw, cfg)
    if wave_col is not None:
        out["wave"] = safe_numeric(raw[wave_col])
    else:
        out["wave"] = np.nan

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

    for name in ["adl_limitation", "iadl_limitation", "homecare", "nursing_home", "hospitalization"]:
        out[name] = build_any_from_cols(raw, cfg.get(name, []))

    out["adl_iadl_composite"] = combine_any(out, ["adl_limitation", "iadl_limitation"])
    out["high_care_dependency_composite"] = combine_any(
        out,
        ["adl_limitation", "iadl_limitation", "homecare", "nursing_home", "hospitalization"],
    )

    out["cohort"] = "HRS"
    return out


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
# 5) Longitudinal construction
# =========================================================
def add_wave_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if out["wave"].notna().sum() > 0:
        out["_wave_sort"] = out["wave"]
    else:
        out["_wave_sort"] = out["row_order"]

    out = out.sort_values(["id", "_wave_sort", "row_order"], kind="mergesort")
    out["wave_order"] = out.groupby("id").cumcount()
    return out


def select_baseline(df: pd.DataFrame, tier_col: str = "tier_primary") -> pd.DataFrame:
    d = df[df[tier_col].notna()].copy()
    d = d.sort_values(["id", "wave_order"], kind="mergesort")
    baseline = d.drop_duplicates("id", keep="first").copy()
    baseline = baseline.rename(columns={
        "wave": "baseline_wave",
        "wave_order": "baseline_wave_order",
    })

    keep_cols = [
        "id", "baseline_wave", "baseline_wave_order",
        "age", "sex", "education", "diabetes", "hypertension",
        "core_ui", "frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use",
        "high_burden_marker", "symptom_only_high_marker",
        "tier_primary", "tier_symptom_only",
    ] + INCIDENT_OUTCOMES

    keep_cols = [c for c in keep_cols if c in baseline.columns]
    baseline = baseline[keep_cols].copy()

    rename_map = {}
    for outcome in INCIDENT_OUTCOMES:
        if outcome in baseline.columns:
            rename_map[outcome] = f"baseline_{outcome}"
    baseline = baseline.rename(columns=rename_map)

    return baseline.reset_index(drop=True)


def make_person_wave_longitudinal(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    outcome: str,
    tier_col: str = "tier_primary",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_outcome = f"baseline_{outcome}"
    if base_outcome not in baseline.columns or outcome not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    b = baseline[
        baseline[tier_col].notna()
        & baseline[base_outcome].notna()
        & (baseline[base_outcome] == 0)
    ].copy()

    base_cols = [
        "id", "baseline_wave", "baseline_wave_order",
        "age", "sex", "education", "diabetes", "hypertension",
        "tier_primary", "tier_symptom_only",
    ]
    base_cols = [c for c in base_cols if c in b.columns]

    future = df.merge(b[base_cols], on="id", how="inner", suffixes=("", "_baseline"))
    future = future[future["wave_order"] > future["baseline_wave_order"]].copy()
    future = future[future[outcome].notna()].copy()

    if future.empty:
        return pd.DataFrame(), pd.DataFrame()

    future["outcome_y"] = future[outcome].astype(float)
    future["followup_index"] = future["wave_order"] - future["baseline_wave_order"]
    future = future.sort_values(["id", "followup_index"], kind="mergesort")

    future["_event_cumsum"] = future.groupby("id")["outcome_y"].cumsum()
    future["_event_cumsum_lag"] = future.groupby("id")["_event_cumsum"].shift(1).fillna(0)
    future = future[future["_event_cumsum_lag"] == 0].copy()

    event_rows = []
    for pid, g in future.groupby("id", sort=False):
        event = int((g["outcome_y"] == 1).any())
        if event:
            t = int(g.loc[g["outcome_y"] == 1, "followup_index"].iloc[0])
        else:
            t = int(g["followup_index"].max())

        event_rows.append({
            "id": pid,
            "outcome": outcome,
            "event": event,
            "followup_intervals": t,
            "n_observed_intervals": int(g.shape[0]),
            "baseline_tier_primary": g["tier_primary"].iloc[0],
            "baseline_tier_symptom_only": g["tier_symptom_only"].iloc[0],
            "baseline_age": g["age"].iloc[0],
            "baseline_sex": g["sex"].iloc[0],
            "baseline_education": g["education"].iloc[0],
            "baseline_diabetes": g["diabetes"].iloc[0],
            "baseline_hypertension": g["hypertension"].iloc[0],
        })

    event_summary = pd.DataFrame(event_rows)

    future["analysis"] = "pooled_logistic"
    future["outcome"] = outcome

    return future.reset_index(drop=True), event_summary


# =========================================================
# 6) Summaries
# =========================================================
def summarize_baseline_distribution(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    denom = int(baseline["tier_primary"].notna().sum())

    for tier in TIER_LEVELS:
        n = int((baseline["tier_primary"] == tier).sum())
        rows.append({
            "tier_definition": "tier_primary",
            "tier": tier,
            "n": n,
            "denominator": denom,
            "percent": 100 * n / denom if denom > 0 else np.nan,
        })

    rows.append({
        "tier_definition": "tier_primary",
        "tier": "Core UI positive",
        "n": int((baseline["core_ui"] == 1).sum()),
        "denominator": int(baseline["core_ui"].notna().sum()),
        "percent": 100 * (baseline["core_ui"] == 1).sum() / baseline["core_ui"].notna().sum()
        if baseline["core_ui"].notna().sum() > 0 else np.nan,
    })

    return pd.DataFrame(rows)


def summarize_incidence(event_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if event_summary.empty:
        return pd.DataFrame(rows)

    for outcome, g in event_summary.groupby("outcome"):
        rows.append({
            "outcome": outcome,
            "tier": "Overall",
            "n_persons": int(g.shape[0]),
            "events": int(g["event"].sum()),
            "event_percent": 100 * g["event"].mean() if g.shape[0] else np.nan,
            "median_followup_intervals": float(g["followup_intervals"].median()) if g.shape[0] else np.nan,
        })

        for tier in TIER_LEVELS:
            gt = g[g["baseline_tier_primary"] == tier]
            rows.append({
                "outcome": outcome,
                "tier": tier,
                "n_persons": int(gt.shape[0]),
                "events": int(gt["event"].sum()),
                "event_percent": 100 * gt["event"].mean() if gt.shape[0] else np.nan,
                "median_followup_intervals": float(gt["followup_intervals"].median()) if gt.shape[0] else np.nan,
            })

    return pd.DataFrame(rows)


# =========================================================
# 7) Pooled logistic regression
# =========================================================
def prepare_design_matrix(
    data: pd.DataFrame,
    tier_col: str,
    covars: List[str],
    include_time: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    use_cols = ["outcome_y", tier_col, "followup_index"] + [c for c in covars if c in data.columns]
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

    if include_time:
        X["followup_index"] = pd.to_numeric(d["followup_index"], errors="coerce")

    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep]
    y = y.loc[keep]

    valid_tier_terms = []
    for term in tier_terms:
        if term in X.columns and X[term].nunique(dropna=True) > 1:
            valid_tier_terms.append(term)
        elif term in X.columns:
            X = X.drop(columns=[term])

    X = sm.add_constant(X, has_constant="add")
    return X, y, valid_tier_terms


def fit_pooled_logistic(
    data: pd.DataFrame,
    outcome: str,
    tier_col: str,
    model_name: str,
    covars: List[str],
) -> List[dict]:
    d = data.copy()
    d = d[d["outcome_y"].notna() & d[tier_col].notna()]

    n_persons = int(d["id"].nunique())
    n_intervals = int(d.shape[0])
    events = int(d.groupby("id")["outcome_y"].max().sum()) if n_intervals else 0

    if n_persons < MIN_TOTAL_N_FOR_MODEL or events < MIN_EVENTS_FOR_MODEL or d["outcome_y"].nunique(dropna=True) < 2:
        return [{
            "analysis": "pooled_logistic",
            "outcome": outcome,
            "model": model_name,
            "term": "model",
            "n_persons": n_persons,
            "n_intervals": n_intervals,
            "events": events,
            "status": "skipped_small_or_no_variation",
        }]

    try:
        X, y, tier_terms = prepare_design_matrix(d, tier_col, covars, include_time=True)

        n_persons_fit = int(d.loc[X.index, "id"].nunique())
        events_fit = int(d.loc[X.index].groupby("id")["outcome_y"].max().sum())

        if len(y) < MIN_TOTAL_N_FOR_MODEL or events_fit < MIN_EVENTS_FOR_MODEL or y.nunique() < 2:
            return [{
                "analysis": "pooled_logistic",
                "outcome": outcome,
                "model": model_name,
                "term": "model",
                "n_persons": n_persons_fit,
                "n_intervals": int(len(y)),
                "events": events_fit,
                "status": "skipped_after_covariate_filter",
            }]

        if not tier_terms:
            return [{
                "analysis": "pooled_logistic",
                "outcome": outcome,
                "model": model_name,
                "term": "model",
                "n_persons": n_persons_fit,
                "n_intervals": int(len(y)),
                "events": events_fit,
                "status": "skipped_no_tier_contrast",
            }]

        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=100, disp=0)

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
                "analysis": "pooled_logistic",
                "outcome": outcome,
                "model": model_name,
                "term": label,
                "n_persons": n_persons_fit,
                "n_intervals": int(len(y)),
                "events": events_fit,
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
        logging.warning("Pooled logistic failed | %s | %s | %s | %s", outcome, tier_col, model_name, e)
        return [{
            "analysis": "pooled_logistic",
            "outcome": outcome,
            "model": model_name,
            "term": "model",
            "n_persons": n_persons,
            "n_intervals": n_intervals,
            "events": events,
            "status": f"failed: {e}",
        }]


def run_pooled_logistic_models(longitudinal_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for outcome, d in longitudinal_frames.items():
        if d.empty:
            rows.append({
                "analysis": "pooled_logistic",
                "outcome": outcome,
                "model": "all",
                "term": "model",
                "n_persons": 0,
                "n_intervals": 0,
                "events": 0,
                "status": "skipped_no_followup_data",
            })
            continue

        for model_name, covars in MODEL_SPECS.items():
            available_covars = [c for c in covars if c in d.columns and d[c].notna().sum() > 0]
            rows.extend(fit_pooled_logistic(d, outcome, "tier_primary", model_name, available_covars))

    return pd.DataFrame(rows).reindex(columns=REGRESSION_COLUMNS)


# =========================================================
# 8) Clinical utility analysis
# =========================================================
def prepare_person_level_utility_data(event_summary: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = event_summary[event_summary["outcome"] == outcome].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.rename(columns={
        "event": "outcome_y",
        "baseline_tier_primary": "tier_primary",
        "baseline_age": "age",
        "baseline_sex": "sex",
        "baseline_education": "education",
        "baseline_diabetes": "diabetes",
        "baseline_hypertension": "hypertension",
    })

    return d


def prepare_utility_matrix(data: pd.DataFrame, covars: List[str], add_tier: bool) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    d = data.copy()
    use_cols = ["outcome_y"] + [c for c in covars if c in d.columns]
    if add_tier:
        use_cols.append("tier_primary")

    d = d[use_cols].dropna(subset=["outcome_y"])
    y = d["outcome_y"].astype(float)

    X = pd.DataFrame(index=d.index)

    for c in covars:
        if c not in d.columns:
            continue

        if c in ["age", "sex", "diabetes", "hypertension"]:
            X[c] = pd.to_numeric(d[c], errors="coerce")
        elif c == "education":
            dd = pd.get_dummies(d[c].astype("category"), prefix=c, drop_first=True, dummy_na=False)
            X = pd.concat([X, dd.astype(float)], axis=1)

    if add_tier:
        X["tier_Tier1"] = (d["tier_primary"] == "Tier 1").astype(float)
        X["tier_Tier2"] = (d["tier_primary"] == "Tier 2").astype(float)

    keep = X.notna().all(axis=1) & y.notna()
    original_index = X.loc[keep].index

    X = X.loc[keep].copy()
    y = y.loc[keep].copy()

    for col in ["tier_Tier1", "tier_Tier2"]:
        if col in X.columns and X[col].nunique(dropna=True) <= 1:
            X = X.drop(columns=[col])

    X = sm.add_constant(X, has_constant="add")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, original_index


def calibration_summary(y: pd.Series, pred: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred)
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    lp = np.log(pred / (1 - pred))

    Xcal = sm.add_constant(pd.DataFrame({"lp": lp}), has_constant="add")
    fit_cal = sm.GLM(y, Xcal, family=sm.families.Binomial()).fit(maxiter=100, disp=0)

    return float(fit_cal.params["const"]), float(fit_cal.params["lp"])

def fit_utility_model(data: pd.DataFrame, outcome: str, model_name: str, covars: List[str], add_tier: bool) -> Tuple[dict, pd.DataFrame]:
    if data.empty:
        return {
            "outcome": outcome,
            "model": model_name,
            "n": 0,
            "events": 0,
            "status": "skipped_no_data",
        }, pd.DataFrame()

    try:
        X, y, original_index = prepare_utility_matrix(data, covars, add_tier=add_tier)
        n = int(len(y))
        events = int(y.sum())

        if n < MIN_TOTAL_N_FOR_MODEL or events < MIN_EVENTS_FOR_MODEL or y.nunique() < 2:
            return {
                "outcome": outcome,
                "model": model_name,
                "n": n,
                "events": events,
                "status": "skipped_small_or_no_variation",
            }, pd.DataFrame()

        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
        pred = np.asarray(fit.predict(X))

        auc = np.nan
        if roc_auc_score is not None:
            auc = float(roc_auc_score(y, pred))

        cal_intercept, cal_slope = calibration_summary(y, pred)

        summary = {
            "outcome": outcome,
            "model": model_name,
            "n": n,
            "events": events,
            "status": "ok",
            "auc": auc,
            "calibration_intercept": cal_intercept,
            "calibration_slope": cal_slope,
        }

        pred_df = pd.DataFrame({
            "outcome": outcome,
            "model": model_name,
            "id": data.loc[original_index, "id"].values if "id" in data.columns else original_index,
            "observed": y.values,
            "predicted_risk": pred,
        })

        return summary, pred_df

    except Exception as e:
        logging.warning("Utility model failed | %s | %s | %s", outcome, model_name, e)
        return {
            "outcome": outcome,
            "model": model_name,
            "n": int(data.shape[0]),
            "events": int(data["outcome_y"].sum()) if "outcome_y" in data.columns else 0,
            "status": f"failed: {e}",
        }, pd.DataFrame()


def decision_curve(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred_df.empty:
        return pd.DataFrame(rows)

    for (outcome, model), g in pred_df.groupby(["outcome", "model"]):
        y = g["observed"].astype(float).values
        p = g["predicted_risk"].astype(float).values
        n = len(y)

        for threshold in DCA_THRESHOLDS:
            pred_pos = p >= threshold
            tp = np.sum((pred_pos == 1) & (y == 1))
            fp = np.sum((pred_pos == 1) & (y == 0))

            net_benefit = tp / n - fp / n * threshold / (1 - threshold)
            treat_all = np.mean(y) - (1 - np.mean(y)) * threshold / (1 - threshold)
            treat_none = 0.0

            rows.append({
                "outcome": outcome,
                "model": model,
                "threshold": threshold,
                "n": int(n),
                "event_rate": float(np.mean(y)),
                "net_benefit": float(net_benefit),
                "treat_all_net_benefit": float(treat_all),
                "treat_none_net_benefit": treat_none,
            })

    return pd.DataFrame(rows)

def run_utility_analysis(event_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    utility_outcomes = [
        "adl_iadl_composite",
        "high_care_dependency_composite",
    ]

    summaries = []
    predictions = []

    for outcome in utility_outcomes:
        d = prepare_person_level_utility_data(event_summary, outcome)
        covars = MODEL_SPECS["M3_metabolic"]

        s1, p1 = fit_utility_model(
            d,
            outcome=outcome,
            model_name="base_M3_metabolic",
            covars=covars,
            add_tier=False,
        )

        s2, p2 = fit_utility_model(
            d,
            outcome=outcome,
            model_name="base_M3_metabolic_plus_tier",
            covars=covars,
            add_tier=True,
        )

        summaries.extend([s1, s2])

        if not p1.empty:
            predictions.append(p1)
        if not p2.empty:
            predictions.append(p2)

    utility_summary = pd.DataFrame(summaries).reindex(columns=UTILITY_COLUMNS)

    if predictions:
        utility_prediction = pd.concat(predictions, axis=0, ignore_index=True)
    else:
        utility_prediction = pd.DataFrame(columns=[
            "outcome", "model", "id", "observed", "predicted_risk"
        ])

    decision_curve_table = decision_curve(utility_prediction)

    return utility_summary, utility_prediction, decision_curve_table
# =========================================================
# 9) Output helpers
# =========================================================
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
            "columns": ";".join(map(str, df.columns.tolist())),
        })
    return pd.DataFrame(rows)


def write_outputs(tables: Dict[str, pd.DataFrame]) -> None:
    versioned = {name: add_script_version(df) for name, df in tables.items()}
    versioned["file_manifest"] = make_file_manifest(versioned)

    for name, df in versioned.items():
        df.to_csv(RUN_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    xlsx_path = RUN_DIR / "hrs_enhance_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in versioned.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


# =========================================================
# 10) Main
# =========================================================
def main() -> None:
    logging.info("Run directory: %s", RUN_DIR)
    logging.info("Data root: %s", DATA_ROOT)

    raw, raw_column_inventory, path = read_hrs_raw(HRS_CONFIG)
    logging.info("Loaded HRS from %s | raw shape=%s", path, raw.shape)

    standardized = standardize_hrs(raw, HRS_CONFIG)
    standardized = construct_phenotypes(standardized)
    standardized = add_wave_order(standardized)

    baseline = select_baseline(standardized, tier_col="tier_primary")
    logging.info("Baseline analytic HRS persons: %s", baseline.shape[0])

    baseline_distribution = summarize_baseline_distribution(baseline)

    longitudinal_frames = {}
    event_summary_parts = []

    for outcome in INCIDENT_OUTCOMES:
        long_df, event_df = make_person_wave_longitudinal(
            standardized,
            baseline,
            outcome=outcome,
            tier_col="tier_primary",
        )

        longitudinal_frames[outcome] = long_df

        if not event_df.empty:
            event_summary_parts.append(event_df)

        logging.info(
            "Longitudinal data | outcome=%s | intervals=%s | persons=%s",
            outcome,
            long_df.shape[0],
            long_df["id"].nunique() if not long_df.empty else 0,
        )

    event_summary = pd.concat(event_summary_parts, axis=0, ignore_index=True) if event_summary_parts else pd.DataFrame()
    incidence_summary = summarize_incidence(event_summary)

    pooled_logistic_models = run_pooled_logistic_models(longitudinal_frames)

    pooled_longitudinal_dataset = pd.concat(
        [d for d in longitudinal_frames.values() if not d.empty],
        axis=0,
        ignore_index=True,
    ) if any(not d.empty for d in longitudinal_frames.values()) else pd.DataFrame()

    utility_summary, utility_prediction, decision_curve_table = run_utility_analysis(event_summary)

    input_manifest = pd.DataFrame([{
        "cohort": "HRS",
        "path": str(path),
        "raw_rows": int(raw.shape[0]),
        "raw_cols_loaded": int(raw.shape[1]),
        "standardized_rows": int(standardized.shape[0]),
        "baseline_persons": int(baseline.shape[0]),
        "unique_ids_raw": int(standardized["id"].nunique()),
        "wave_column_used": get_wave_column(raw, HRS_CONFIG) if get_wave_column(raw, HRS_CONFIG) is not None else "row_order_within_id",
    }])

    minimal_cols = [
        "id", "row_order", "wave", "wave_order",
        "age", "sex", "education", "diabetes", "hypertension",
        "core_ui", "frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use",
        "high_burden_marker", "symptom_only_high_marker",
        "tier_primary", "tier_symptom_only",
    ] + INCIDENT_OUTCOMES
    minimal_cols = [c for c in minimal_cols if c in standardized.columns]

    baseline_cols = [
        "id", "baseline_wave", "baseline_wave_order",
        "age", "sex", "education", "diabetes", "hypertension",
        "core_ui", "frequent_ui", "stress_like_ui", "urgency_like_ui", "pad_use",
        "high_burden_marker", "symptom_only_high_marker",
        "tier_primary", "tier_symptom_only",
    ] + [f"baseline_{x}" for x in INCIDENT_OUTCOMES]
    baseline_cols = [c for c in baseline_cols if c in baseline.columns]

    tables = {
        "input_manifest": input_manifest,
        "raw_column_inventory": raw_column_inventory,
        "hrs_standardized_minimal": standardized[minimal_cols],
        "baseline_dataset": baseline[baseline_cols],
        "baseline_phenotype_distribution": baseline_distribution,
        "person_level_event_summary": event_summary,
        "incident_outcome_summary": incidence_summary,
        "pooled_longitudinal_dataset": pooled_longitudinal_dataset,
        "pooled_logistic_models": pooled_logistic_models,
        "clinical_utility_summary": utility_summary,
        "clinical_utility_predictions": utility_prediction,
        "decision_curve_table": decision_curve_table,
    }

    write_outputs(tables)

    run_info = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(RUN_DIR),
        "data_root": str(DATA_ROOT),
        "input_file": str(path),
        "tier_levels": TIER_LEVELS,
        "frequent_ui_days_cutoff": FREQUENT_UI_DAYS_CUTOFF,
        "model_specs": MODEL_SPECS,
        "incident_outcomes": INCIDENT_OUTCOMES,
        "min_total_n_for_model": MIN_TOTAL_N_FOR_MODEL,
        "min_events_for_model": MIN_EVENTS_FOR_MODEL,
        "note": (
            "HRS-only longitudinal enhancement. Baseline is the first available wave with "
            "non-missing bladder-health burden tier. For each outcome, participants with "
            "baseline outcome=1 are excluded. Follow-up uses subsequent HRS rows/waves. "
            "Primary model is pooled logistic regression with follow-up interval adjustment. "
            "Clinical utility compares M3 metabolic base model against M3 plus bladder-health tier."
        ),
    }

    with open(RUN_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    del raw, standardized
    gc.collect()

    logging.info("Completed. Outputs written to %s", RUN_DIR)


if __name__ == "__main__":
    main()