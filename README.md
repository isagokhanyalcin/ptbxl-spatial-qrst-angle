# Frontal versus spatial QRS-T angle in PTB-XL+

Analysis code for the manuscript *What the frontal plane misses: spatial versus frontal QRS-T angle in 18,558 ECGs from PTB-XL+* (submitted to the Journal of Electrocardiology).

The spatial QRS-T angle is derived from the Glasgow (Uni-G) median beats in PTB-XL+. The 12-lead median beat is transformed into X, Y and Z leads with the Kors regression matrix. The QRS loop runs from QRS onset to QRS offset and the T loop from QRS offset to T-wave offset, using the Uni-G global fiducial points. The angle is computed between the mean QRS and T vectors (primary) and between the peak vectors. It is then compared with the frontal QRS-T angle, the difference between the frontal QRS and T axes, as printed on standard ECG reports.

## Data

Both datasets are open on PhysioNet under CC BY 4.0. They are not stored in this repository.

- PTB-XL v1.0.3: Wagner P, et al. Sci Data. 2020;7:154. https://doi.org/10.13026/kfzx-aw45
- PTB-XL+ v1.0.1: Strodthoff N, et al. Sci Data. 2023;10:279. https://doi.org/10.13026/g6h6-7g88

## Reproducing the analysis

```bash
pip install -r requirements.txt
bash scripts/00_download_tables.sh                                    # feature and label tables, about 200 MB, into data/
python scripts/00_download_median_beats.py data/ptbxl_database.csv mb  # about 43,600 small files, about 240 MB
python scripts/01_analysis_paper2.py data mb results
python scripts/02_make_manuscript_tables.py results tables.md supplementary_tables.md  # manuscript tables
```

The first run of the analysis reads every median beat and writes the derived vectors to `results/spatial_vectors.csv`. Later runs reuse that file.

## Output (`results/`)

| File | Content |
|---|---|
| `table_cohort.csv` | Cohort and healthy reference group characteristics |
| `table_distributions.csv` | Distribution of each angle by group |
| `table_reference_values.csv` | Age- and sex-specific percentiles in healthy adults |
| `table_concordance.csv` | Spatially abnormal ECGs classified as normal by the frontal angle |
| `table_missed_by_zshare.csv` | The same, by tertile of the out-of-plane (Z) share of the mean vectors |
| `table_missed_predictors.csv` | Logistic regression for a missed spatial abnormality |
| `table_roc.csv` | ROC analysis with DeLong comparison |
| `table_sensitivity.csv` | Sensitivity analyses |
| `fig1_frontal_vs_spatial.png`, `fig2_missed_by_zshare.png`, `fig3_roc.png` | Figures |
| `key_results.json` | Patient flow and summary statistics |

## Authors

İsa Gökhan Yalçın, Ahmet Yunus Hatip, Furkan Çağrı Oğuzlar. Department of Emergency Medicine, Süleyman Demirel University, Isparta, Türkiye.

Contact: isayalcin@sdu.edu.tr
