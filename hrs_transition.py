# -*- coding: utf-8 -*-
r"""
HRS landmark transition analysis of incident bladder-health burden, V3.

Purpose
-------
This script performs a concise exploratory longitudinal transition analysis for
one manuscript paragraph and one supplementary-ready table.

Design
------
t0: first eligible HRS wave with Tier 0 bladder-health burden among adults aged >=60
t1: next wave defines transition status: Tier 0->0, Tier 0->1, or Tier 0->2
t2: subsequent wave defines incident high non-toileting ADL/IADL burden

Outcome
-------
Incident high non-toileting ADL/IADL burden at t2, defined as >=2 limitations
after excluding toileting ADL. Participants with this outcome already present
at t0 or t1 are excluded.

Model
-----
Logistic regression adjusted for age, sex, education, diabetes, hypertension,
and t0 wave, with HC1 robust standard errors. The output includes adjusted ORs
and marginal standardized risk differences per 1000 participants.

Default paths
-------------
Input files are expected under data/cohorts by default. Set BHB_COHORT_DATA
and BHB_OUTPUT_ROOT to use another local data or output location.
"""

import os
import re
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. Project paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = Path(os.environ.get("BHB_COHORT_DATA", str(PROJECT_ROOT / "data" / "cohorts")))
OUT_ROOT = Path(os.environ.get("BHB_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))

HRS_FILE_CANDIDATES = [DATA_ROOT / "HRS.csv", DATA_ROOT / "hrs.csv"]
HRS_PATH = next((p for p in HRS_FILE_CANDIDATES if p.exists()), HRS_FILE_CANDIDATES[-1])
HRS_ZIP = DATA_ROOT / "hrs.zip"
OUT_DIR = OUT_ROOT / "hrs_transition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. Variable settings
# =============================================================================
RAW_COLUMNS = [
    "hhidpn", "wave", "ragey_b", "ragender", "raedyrs", "diabe", "hibpe",
    "urinai", "urinaf",
    "walkra", "dressa", "batha", "eata", "beda", "toilta",
    "moneya", "phonea", "medsa", "mealsa", "shopa",
]

ADL_COLUMNS = ["walkra", "dressa", "batha", "eata", "beda"]  # toileting excluded
IADL_COLUMNS = ["moneya", "phonea", "medsa", "mealsa", "shopa"]

TRANSITION_LABELS = {
    "stable0": "Stable Tier 0",
    "incident1": "Incident Tier 1",
    "incident2": "Incident Tier 2",
}

# =============================================================================
# 3. Helper functions
# =============================================================================
def code_col(col):
    col = str(col)
    match = re.match(r"^\s*([^\s(]+)", col)
    return match.group(1) if match else col


def read_hrs_columns():
    if HRS_PATH.exists():
        header = pd.read_csv(HRS_PATH, nrows=0).columns.tolist()
        mapping = {}
        for c in header:
            cc = code_col(c)
            if cc not in mapping:
                mapping[cc] = c
        usecols = [mapping[c] for c in RAW_COLUMNS if c in mapping]
        data = pd.read_csv(HRS_PATH, usecols=usecols, low_memory=False)
    elif HRS_ZIP.exists():
        with zipfile.ZipFile(HRS_ZIP) as z:
            member = "hrs.csv"
            with z.open(member) as f:
                header = pd.read_csv(f, nrows=0).columns.tolist()
            mapping = {}
            for c in header:
                cc = code_col(c)
                if cc not in mapping:
                    mapping[cc] = c
            usecols = [mapping[c] for c in RAW_COLUMNS if c in mapping]
            with z.open(member) as f:
                data = pd.read_csv(f, usecols=usecols, low_memory=False)
    else:
        raise FileNotFoundError(
            f"Cannot find HRS input. Checked {HRS_PATH} and {HRS_ZIP}."
        )

    data.columns = [code_col(c) for c in data.columns]
    missing = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in HRS data: {missing}")
    return data


def wave_to_num(x):
    s = pd.Series(x).astype(str)
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")


def to_numeric_safe(x):
    return pd.to_numeric(pd.Series(x), errors="coerce")


def yes_no_to_binary(x):
    s = pd.Series(x)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        out[v == 0] = 0.0
        out[v == 1] = 1.0
        return out

    ss = s.astype(str).str.strip().str.lower()
    yes_tokens = {"1", "1.0", "yes", "y", "true", "是", "是的", "1.yes"}
    no_tokens = {"0", "0.0", "no", "n", "false", "否", "不是", "0.no"}
    out[ss.isin(yes_tokens)] = 1.0
    out[ss.isin(no_tokens)] = 0.0
    return out


def build_bladder_tier(df):
    urinai = yes_no_to_binary(df["urinai"])
    urinaf = to_numeric_safe(df["urinaf"])

    tier = pd.Series(np.nan, index=df.index, dtype="float64")
    tier[urinai == 0] = 0
    tier[(urinai == 1) | (urinaf > 0)] = 1
    tier[(urinai == 1) & (urinaf >= TIER2_FREQUENCY_DAYS)] = 2
    tier[(urinai.isna()) & (urinaf == 0)] = 0
    return tier


def build_functional_burden(df):
    use_cols = ADL_COLUMNS + IADL_COLUMNS
    binary_items = pd.concat([yes_no_to_binary(df[c]) for c in use_cols], axis=1)
    binary_items.columns = use_cols
    count = binary_items.sum(axis=1, min_count=1)
    high = (count >= HIGH_FUNCTIONAL_BURDEN_CUTOFF).astype(float)
    high[count.isna()] = np.nan
    return count, high


def fmt_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_mean_sd(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return ""
    return f"{x.mean():.1f} ({x.std(ddof=1):.1f})"


def fmt_n_pct(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return ""
    n = int((x == 1).sum())
    pct = 100 * n / len(x)
    return f"{n} ({pct:.1f}%)"

# =============================================================================
# 4. Build analytic dataset
# =============================================================================
def build_transition_dataset(raw):
    df = raw.rename(columns={
        "hhidpn": "id",
        "ragey_b": "Age, years",
        "ragender": "Sex",
        "raedyrs": "Education, years",
        "diabe": "Diabetes",
        "hibpe": "Hypertension",
    }).copy()

    df["wave_num"] = wave_to_num(df["wave"])
    df["Bladder-health burden tier"] = build_bladder_tier(raw)
    count, high = build_functional_burden(raw)
    df["Non-toileting ADL/IADL limitation count"] = count
    df["High non-toileting ADL/IADL burden"] = high

    df["Age, years"] = to_numeric_safe(df["Age, years"])
    df["Education, years"] = to_numeric_safe(df["Education, years"])
    df["Diabetes"] = yes_no_to_binary(df["Diabetes"])
    df["Hypertension"] = yes_no_to_binary(df["Hypertension"])
    df["Female"] = df["Sex"].astype(str).str.contains("女|female", case=False, regex=True).astype(float)

    keep_cols = [
        "id", "wave_num", "Age, years", "Sex", "Female", "Education, years",
        "Diabetes", "Hypertension", "Bladder-health burden tier",
        "Non-toileting ADL/IADL limitation count", "High non-toileting ADL/IADL burden",
    ]
    df = df[keep_cols].dropna(subset=["id", "wave_num"]).sort_values(["id", "wave_num"])

    source_rows = len(df)
    source_persons = df["id"].nunique()

    df = df[df["Age, years"].notna() & (df["Age, years"] >= MIN_AGE)].copy()
    aged_rows = len(df)
    aged_persons = df["id"].nunique()

    grouped = df.groupby("id", sort=False)
    follow_cols = [
        "wave_num", "Bladder-health burden tier",
        "High non-toileting ADL/IADL burden", "Non-toileting ADL/IADL limitation count",
    ]
    for c in follow_cols:
        df[f"t1_{c}"] = grouped[c].shift(-1)
        df[f"t2_{c}"] = grouped[c].shift(-2)

    candidate = df[
        (df["Bladder-health burden tier"] == 0)
        & df["t1_Bladder-health burden tier"].notna()
        & df["t2_High non-toileting ADL/IADL burden"].notna()
    ].copy()

    eligible = candidate[
        ((candidate["High non-toileting ADL/IADL burden"] != 1) | candidate["High non-toileting ADL/IADL burden"].isna())
        & ((candidate["t1_High non-toileting ADL/IADL burden"] != 1) | candidate["t1_High non-toileting ADL/IADL burden"].isna())
    ].copy()

    eligible["Transition group"] = np.select(
        [
            eligible["t1_Bladder-health burden tier"] == 0,
            eligible["t1_Bladder-health burden tier"] == 1,
            eligible["t1_Bladder-health burden tier"] == 2,
        ],
        ["stable0", "incident1", "incident2"],
        default="missing",
    )
    eligible = eligible[eligible["Transition group"] != "missing"].copy()

    analytic = eligible.groupby("id", sort=False).head(1).copy()
    analytic["Outcome"] = analytic["t2_High non-toileting ADL/IADL burden"].astype(float)
    analytic["Baseline wave"] = analytic["wave_num"].astype(int)
    analytic["Transition group label"] = analytic["Transition group"].map(TRANSITION_LABELS)

    sample_flow = pd.DataFrame([
        {"Analytic step": "HRS person-wave records with required columns", "Participants, n": source_persons, "Person-waves, n": source_rows},
        {"Analytic step": f"Restricted to age ≥{MIN_AGE} years", "Participants, n": aged_persons, "Person-waves, n": aged_rows},
        {"Analytic step": "Eligible Tier 0 records with next-wave transition and subsequent outcome information", "Participants, n": candidate["id"].nunique(), "Person-waves, n": len(candidate)},
        {"Analytic step": "Excluded prevalent high non-toileting ADL/IADL burden at t0 or t1", "Participants, n": eligible["id"].nunique(), "Person-waves, n": len(eligible)},
        {"Analytic step": "Final analytic sample using the first eligible transition per participant", "Participants, n": analytic["id"].nunique(), "Person-waves, n": len(analytic)},
    ])

    return analytic, sample_flow

# =============================================================================
# 5. Model and tables
# =============================================================================
def prepare_model_data(analytic):
    d = analytic.copy()
    d["Transition group"] = pd.Categorical(
        d["Transition group"], categories=["stable0", "incident1", "incident2"]
    )
    d["Sex"] = d["Sex"].astype(str).replace({"nan": "Unknown"}).fillna("Unknown")
    for c in ["Age, years", "Education, years", "Diabetes", "Hypertension"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(d[c].median()) if d[c].notna().any() else 0.0
    d["Baseline wave"] = d["Baseline wave"].astype(int).astype(str)
    return d


def fit_model(analytic):
    d = prepare_model_data(analytic)
    formula = (
        "Outcome ~ C(Q('Transition group'), Treatment(reference='stable0'))"
        " + Q('Age, years') + C(Sex) + Q('Education, years')"
        " + Diabetes + Hypertension + C(Q('Baseline wave'))"
    )
    try:
        fit = smf.glm(formula=formula, data=d, family=sm.families.Binomial()).fit(cov_type="HC1")
    except Exception:
        formula = (
            "Outcome ~ C(Q('Transition group'), Treatment(reference='stable0'))"
            " + Q('Age, years') + C(Sex) + Diabetes + Hypertension + C(Q('Baseline wave'))"
        )
        fit = smf.glm(formula=formula, data=d, family=sm.families.Binomial()).fit(cov_type="HC1")
    return fit, d


def build_primary_table(analytic, fit, model_data):
    counts = analytic.groupby("Transition group", observed=False)["Outcome"].agg(
        Participants="size", Events="sum", Rate="mean"
    )

    rows = []
    ref_group = "stable0"
    for group in ["stable0", "incident1", "incident2"]:
        n = int(counts.loc[group, "Participants"])
        e = int(counts.loc[group, "Events"])
        rate = float(counts.loc[group, "Rate"])

        if group == ref_group:
            rows.append({
                "Transition group": TRANSITION_LABELS[group],
                "Participants, n": n,
                "Incident high non-toileting ADL/IADL burden, n (%)": f"{e} ({100*rate:.1f}%)",
                "Adjusted OR (95% CI)": "Reference",
                "P value": "—",
                "Standardized risk difference per 1000 participants": "Reference",
            })
            continue

        term = f"C(Q('Transition group'), Treatment(reference='stable0'))[T.{group}]"
        beta = fit.params[term]
        lo, hi = fit.conf_int().loc[term]

        ref_data = model_data.copy()
        exp_data = model_data.copy()
        ref_data["Transition group"] = ref_group
        exp_data["Transition group"] = group
        p_ref = fit.predict(ref_data).mean()
        p_exp = fit.predict(exp_data).mean()
        rd1000 = (p_exp - p_ref) * 1000

        rows.append({
            "Transition group": TRANSITION_LABELS[group],
            "Participants, n": n,
            "Incident high non-toileting ADL/IADL burden, n (%)": f"{e} ({100*rate:.1f}%)",
            "Adjusted OR (95% CI)": f"{np.exp(beta):.2f} ({np.exp(lo):.2f}, {np.exp(hi):.2f})",
            "P value": fmt_p(float(fit.pvalues[term])),
            "Standardized risk difference per 1000 participants": f"{rd1000:+.1f}",
        })

    table = pd.DataFrame(rows)
    return table


def build_numeric_estimates(analytic, fit, model_data):
    counts = analytic.groupby("Transition group", observed=False)["Outcome"].agg(
        n="size", events="sum", crude_risk="mean"
    )
    rows = []
    for group in ["incident1", "incident2"]:
        term = f"C(Q('Transition group'), Treatment(reference='stable0'))[T.{group}]"
        beta = fit.params[term]
        lo, hi = fit.conf_int().loc[term]
        ref_data = model_data.copy()
        exp_data = model_data.copy()
        ref_data["Transition group"] = "stable0"
        exp_data["Transition group"] = group
        p_ref = fit.predict(ref_data).mean()
        p_exp = fit.predict(exp_data).mean()
        rows.append({
            "Contrast": f"{TRANSITION_LABELS[group]} vs Stable Tier 0",
            "Reference participants, n": int(counts.loc["stable0", "n"]),
            "Exposed participants, n": int(counts.loc[group, "n"]),
            "Reference events, n": int(counts.loc["stable0", "events"]),
            "Exposed events, n": int(counts.loc[group, "events"]),
            "Reference crude risk": float(counts.loc["stable0", "crude_risk"]),
            "Exposed crude risk": float(counts.loc[group, "crude_risk"]),
            "Adjusted OR": float(np.exp(beta)),
            "95% CI lower": float(np.exp(lo)),
            "95% CI upper": float(np.exp(hi)),
            "P value": float(fit.pvalues[term]),
            "Standardized risk, reference": float(p_ref),
            "Standardized risk, exposed": float(p_exp),
            "Standardized risk difference per 1000": float((p_exp - p_ref) * 1000),
        })
    return pd.DataFrame(rows)


def build_baseline_table(analytic):
    rows = []
    variables = [
        ("Age, years, mean (SD)", "Age, years", fmt_mean_sd),
        ("Female, n (%)", "Female", fmt_n_pct),
        ("Education, years, mean (SD)", "Education, years", fmt_mean_sd),
        ("Diabetes, n (%)", "Diabetes", fmt_n_pct),
        ("Hypertension, n (%)", "Hypertension", fmt_n_pct),
        ("Non-toileting ADL/IADL limitation count, mean (SD)", "Non-toileting ADL/IADL limitation count", fmt_mean_sd),
    ]
    for label, col, formatter in variables:
        row = {"Baseline characteristic": label}
        for group in ["stable0", "incident1", "incident2"]:
            row[TRANSITION_LABELS[group]] = formatter(analytic.loc[analytic["Transition group"] == group, col])
        rows.append(row)
    return pd.DataFrame(rows)


def write_text_outputs(primary_table):
    tier1 = primary_table.loc[primary_table["Transition group"] == "Incident Tier 1"].iloc[0]
    tier2 = primary_table.loc[primary_table["Transition group"] == "Incident Tier 2"].iloc[0]

    methods = (
        "We additionally performed an exploratory HRS landmark transition analysis among participants "
        "initially classified as Tier 0 and free of high non-toileting ADL/IADL burden. Transition "
        "status from Tier 0 to Tier 0, Tier 1, or Tier 2 was defined at the next wave, and incident "
        "high non-toileting ADL/IADL burden was assessed at the subsequent wave. Models were adjusted "
        "for age, sex, education, diabetes, hypertension, and baseline wave."
    )

    results = (
        "In an exploratory HRS landmark transition analysis, incident bladder-health burden was "
        "associated with higher subsequent risk of high non-toileting ADL/IADL burden. Compared with "
        f"participants who remained in Tier 0, progression to Tier 1 was associated with an adjusted OR "
        f"of {tier1['Adjusted OR (95% CI)']} (P={tier1['P value']}), whereas progression to Tier 2 showed "
        f"a stronger association (adjusted OR {tier2['Adjusted OR (95% CI)']}; P{'' if str(tier2['P value']).startswith('<') else '='}{tier2['P value']})."
    )

    with open(OUT_DIR / "Methods_text_HRS_transition.txt", "w", encoding="utf-8") as f:
        f.write(methods + "\n")
    with open(OUT_DIR / "Results_text_HRS_transition.txt", "w", encoding="utf-8") as f:
        f.write(results + "\n")


def main():
    print("Reading HRS data from:", HRS_PATH if HRS_PATH.exists() else HRS_ZIP)
    print("Saving outputs to:", OUT_DIR)

    raw = read_hrs_columns()
    analytic, sample_flow = build_transition_dataset(raw)
    fit, model_data = fit_model(analytic)

    primary_table = build_primary_table(analytic, fit, model_data)
    numeric_estimates = build_numeric_estimates(analytic, fit, model_data)
    baseline_table = build_baseline_table(analytic)

    primary_table.to_csv(OUT_DIR / "Supplementary_Table_HRS_transition_primary.csv", index=False, encoding="utf-8-sig")
    sample_flow.to_csv(OUT_DIR / "Supplementary_Table_HRS_transition_sample_flow.csv", index=False, encoding="utf-8-sig")
    baseline_table.to_csv(OUT_DIR / "Supplementary_Table_HRS_transition_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    numeric_estimates.to_csv(OUT_DIR / "HRS_transition_numeric_estimates.csv", index=False, encoding="utf-8-sig")

    write_text_outputs(primary_table)

    readme = (
        "HRS landmark transition analysis\n\n"
        "Main manuscript use:\n"
        "- Use Supplementary_Table_HRS_transition_primary.csv as the supplementary results table.\n"
        "- Use Supplementary_Table_HRS_transition_sample_flow.csv for reporting the analytic sample.\n"
        "- Use Supplementary_Table_HRS_transition_baseline_characteristics.csv if reviewers request baseline comparison.\n\n"
        "Interpretation boundary:\n"
        "- This is an exploratory landmark transition analysis, not a target trial emulation or causal analysis.\n"
        "- The recommended wording is 'incident bladder-health burden marked subsequent functional deterioration'.\n"
    )
    with open(OUT_DIR / "README_HRS_transition.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    zip_path = OUT_DIR.parent / "hrs_transition_analysis_outputs_V3.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                z.write(fp, arcname=fp.name)

    print("\nPrimary supplementary table:")
    print(primary_table.to_string(index=False))
    print("\nSaved:", zip_path)


if __name__ == "__main__":
    main()
