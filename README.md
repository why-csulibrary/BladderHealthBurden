# BladderHealthBurden
This repository contains workflow-oriented scripts for bladder-health burden phenotype construction, multicohort validation, HRS longitudinal analyses, and NHANES biomarker and mortality contextualisation. Original cohort-level data are not redistributed because access is governed by the corresponding cohort data-use policies.

## Main workflows

```bash
python 01_phenotype_multicohort_validation.py
python 02_hrs_analysis.py
python 03_nhanes_analysis.py
```

For HRS, the two longitudinal components can also be run separately:

```bash
python 02_hrs_analysis.py --longitudinal-only
python 02_hrs_analysis.py --transition-only
```

