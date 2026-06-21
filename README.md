# BladderHealthBurden
A streamlined public repository, BladderHealthBurden, containing workflow-oriented scripts for bladder-health burden phenotype construction and multicohort validation, HRS longitudinal analyses, and NHANES biomarker and mortality contextualisation will be made available upon publication. 

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

