# MXene WGS adsorption-energy surrogates: data, code and result tables

Data, code and result tables for the manuscript *Interpretable ensemble learning of adsorption energies on MXene
water–gas shift catalysts and diagnosis of transfer failure on doped MXenes* (revised version).

The DFT database is that of Nassar et al. (J. Phys. Chem. C 129 (2025) 2512–2524; Mendeley Data,
https://doi.org/10.17632/bct4m7r88s.1). The web application is in a separate repository:
https://github.com/17609858895/MXene-WGSR-Eads-Predictor.

## Folder structure

```text
data/raw/                     database workbook (sheets "WGS" and "dopants")
code/figures/
  generate_all_figures.py     pipeline (features, preprocessing, six regressors, CV-weighted TOPSIS ensemble) and Figs. 2–8
  revision_figures.py         revised Figs. 3, 4, 5, 7 and 8 of the revised manuscript
  revision_si_figures.py      Figs. S1–S3
code/tables/
  generate_paper_tables.py    Table 1 and Tables S1a, S2, S3b, S5a, S11a, S13
code/analysis/
  robustness_analyses.py      repeated splits, transfer by subset, window sensitivity (Tables S12a, S14)
  revision_analyses.py        baselines, group-wise validation, conformal intervals, pairwise ranking, conditional
                              importance, applicability-domain checks, few-shot analysis (Tables 2, 3, S3a, S3c, S4,
                              S5b, S6–S10, S11b, S12b–d, S15)
code/model_export/
  train_export_model.py       exports the ensemble used by the web application
results/tables/               result tables of the manuscript (CSV); results/tables/revision/ holds the additional analyses
requirements-paper.txt        package list; versions used are given below
```

`code/figures/generate_subfigures_fig2to8.ipynb`, `build_subfigure_ppts.py` and `extract_panels_from_combined_pdf.py` are
layout helpers from an earlier draft and are not needed to reproduce the reported results.

## Reproduce

Run from the repository root. Outputs are written to `outputs/` (not tracked).

```bash
python code/figures/generate_all_figures.py     # reference split, Figs. 2-8, intermediate CSV files
python code/tables/generate_paper_tables.py     # Table 1 and Supporting tables
python code/analysis/robustness_analyses.py     # repeated splits, transfer by subset, window sensitivity
python code/analysis/revision_analyses.py       # additional analyses (about 40 min on 8 cores)
python code/figures/revision_figures.py         # revised Figs. 3, 4, 5, 7, 8 (run after generate_all_figures.py)
python code/figures/revision_si_figures.py      # Figs. S1-S3 (run after revision_analyses.py)
```

`REV_QUICK=1 python code/analysis/revision_analyses.py` runs a short smoke test with few repetitions
(written to `outputs/tables/revision_quicktest/`).

Random seeds: 42 for the reference split and all model initialisations, 0–19 for the repeated splits,
1000 + draw for random few-shot draws and 2000 + draw for dopant-disjoint draws.

## Environment

Python 3.12.3, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, SciPy 1.13.1, statsmodels 0.14.2, matplotlib 3.9.2,
seaborn 0.13.2, networkx 3.3, joblib 1.4.2, openpyxl 3.1.5.

## Notes on the data

- Entries recorded as `-` are treated as missing; records without an adsorption energy are excluded (599 WGS and 94
  dopant records remain).
- The dopant column lists `Rd` for the four records named `Pd@W2CO2`; they are assigned to Pd.
- The WGS-domain model is not suitable for doped MXenes: the dopant is not represented in the training data. The dopant
  subset is used only for transfer diagnostics and for DFT-labelled trend summaries.

## Result files and tables of the revised manuscript

File names in `results/tables/` keep the numbering of the generating scripts; the corresponding tables are:

| File | Table |
| --- | --- |
| `Table1_WGS_Model_Performance.csv` | Table 1 |
| `TableS1_Label_Availability.csv` | Table S1a |
| `TableS2_Descriptors_and_Coverage.csv` | Table S2 |
| `TableS3_Missingness_Sensitivity.csv` | Table S3b |
| `TableS4a_CV_TOPSIS.csv` | Table S5a |
| `TableS4b_Wilcoxon_p.csv` | Table S5b (raw p values) |
| `TableS5_Repeated_Splits*.csv` | Table S6 (base models and ensemble) |
| `TableS6_Activation_Subset.csv` | Table S11a |
| `TableS7a_Transfer_by_Subset.csv` | Table S12a |
| `TableS7b_Dopant_Resolved.csv` | Table S13 |
| `TableS8_Window_Sensitivity.csv` | Table S14 |
| `revision/grouped_cv*.csv` | Table 2, Table S7 |
| `revision/splits_*.csv` | Tables S3c, S5a (mean weights), S6, S8, S9, S10 |
| `revision/ref_missingness_*.csv`, `revision/ref_wilcoxon_*.csv` | Tables S3a, S5b (Holm-adjusted) |
| `revision/S_*.csv` | Table S4 |
| `revision/activation_within_channel.csv` | Table S11b |
| `revision/dopant_*.csv` | Table S12b–d |
| `revision/fewshot_*.csv` | Table 3, Table S15 |
