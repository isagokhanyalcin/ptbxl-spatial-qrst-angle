"""Builds the manuscript tables (markdown) directly from the result CSVs.
Usage: python3 -I 02_make_manuscript_tables.py <results_dir> <main_tables.md> <supplementary_tables.md>"""
import sys, re, pandas as pd

O, OUT, SUP = sys.argv[1], sys.argv[2], sys.argv[3]
nums = lambda s: [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", re.sub(r"np\.float64\(", "", str(s)))]
mn = lambda x, d=1: f"{x:.{d}f}".replace("-", "−")
pv = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
out, sup = [], []

# Table 1
t1 = pd.read_csv(f"{O}/table_cohort.csv", index_col=0)
rows = [("n", lambda r: f"{int(r.n):,}"),
        ("Age, years, median (IQR)*", lambda r: f"{r.age_median:.0f} ({r.age_q1:.0f}–{r.age_q3:.0f})"),
        ("Women, %", lambda r: f"{r.female_pct:.1f}"), ("Normal ECG only, %", lambda r: f"{r.norm_only_pct:.1f}"),
        ("Myocardial infarction (MI), %", lambda r: f"{r.MI_pct:.1f}"), ("ST/T change (STTC), %", lambda r: f"{r.STTC_pct:.1f}"),
        ("Hypertrophy (HYP), %", lambda r: f"{r.HYP_pct:.1f}"), ("Conduction disturbance (CD), %", lambda r: f"{r.CD_pct:.1f}"),
        ("Atrial fibrillation or flutter, %", lambda r: f"{r.afib_pct:.1f}"), ("Complete bundle branch block, %", lambda r: f"{r.bbb_pct:.1f}"),
        ("QRS duration, ms, median†", lambda r: f"{r.qrsd_median:.0f}")]
out += ["**Table 1.** Characteristics of the study cohort and the healthy reference group.", "",
        "| Characteristic | Study cohort | Healthy reference group |", "|---|---|---|"]
out += [f"| {lab} | {fn(t1['All'])} | {fn(t1['Healthy reference'])} |" for lab, fn in rows]
out += ["", "*Ages above 89 years are masked in PTB-XL and were excluded from age summaries. †12SL measurement. "
        "Superclasses are not mutually exclusive.", ""]

# Table 2
d = pd.read_csv(f"{O}/table_distributions.csv")
meas = {"sp_mean": "Spatial, Kors mean vectors", "sp_peak": "Spatial, Kors peak vectors", "fr_gl": "Frontal, Uni-G axes",
        "fr_ge": "Frontal, 12SL axes", "glasgow_vm": "Spatial, reported by Uni-G"}
grps = ["Healthy reference", "Normal only", "Any pathology"]
out += ["**Table 2.** QRS-T angles (°) by group: median (IQR).", "",
        "| Angle | Healthy reference (n = 5,470) | Normal ECG only (n = 8,282) | Any abnormality (n = 10,205) |", "|---|---|---|---|"]
for m, lab in meas.items():
    cells = []
    for g in grps:
        r = d[(d.group == g) & (d.measure == m)].iloc[0]
        cells.append(f"{r['median']:.0f} ({r.q1:.0f}–{r.q3:.0f})")
    out.append(f"| {lab} | " + " | ".join(cells) + " |")
out += ["", "The healthy reference group is a subset of the normal-ECG group.", ""]

# Table 3
t = pd.read_csv(f"{O}/table_reference_values.csv")
order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80", "all"]
out += ["**Table 3.** Spatial QRS-T angle (°) in healthy adults by age group and sex: median (2.5th–97.5th percentile).", "",
        "| Age, years | Sex | n | Mean vectors | Peak vectors |", "|---|---|---|---|---|"]
for a in order:
    for sx in ["M", "F"]:
        r1 = t[(t.age == a) & (t.sex == sx) & (t.measure == "sp_mean")].iloc[0]
        r2 = t[(t.age == a) & (t.sex == sx) & (t.measure == "sp_peak")].iloc[0]
        mark = "*" if bool(r1.provisional_n_lt_120) else ""
        al = {"all": "All ages", ">=80": "≥80"}.get(a, a.replace("-", "–"))
        out.append(f"| {al} | {'Men' if sx == 'M' else 'Women'} | {int(r1.n):,}{mark} | {r1['median']:.0f} ({r1['p2.5']:.0f}–{r1['p97.5']:.0f}) | "
                   f"{r2['median']:.0f} ({r2['p2.5']:.0f}–{r2['p97.5']:.0f}) |")
out += ["", "Healthy adults: age ≥ 18 years, ECG labelled only as normal, report validated by a second cardiologist, no atrial fibrillation or flutter "
        "and no signal quality flag. Nonparametric percentiles. *Below the CLSI EP28-A3c minimum of 120 per partition; provisional.", ""]

# Table 4
cc = pd.read_csv(f"{O}/table_concordance.csv").set_index("frontal_rule")
mz = pd.read_csv(f"{O}/table_missed_by_zshare.csv")
out += ["**Table 4.** Spatially abnormal ECGs (n = 4,838) classified as normal by the frontal angle.", "",
        "| | Frontal cut-off 90° | Frontal upper reference limit |", "|---|---|---|"]
a, b = cc.loc["fr_abn_90"], cc.loc["fr_abn_ref"]
out.append(f"| Missed, n (%) | {int(a.spatial_only):,} ({a.pct_spatial_abn_missed_by_frontal:.1f}) | {int(b.spatial_only):,} ({b.pct_spatial_abn_missed_by_frontal:.1f}) |")
out.append(f"| Abnormal by frontal angle only, n | {int(a.frontal_only):,} | {int(b.frontal_only):,} |")
out.append(f"| Kappa | {a.kappa:.2f} | {b.kappa:.2f} |")
for by, lab in [("zt_tert", "T vector"), ("zq_tert", "QRS vector")]:
    for _, r in mz[mz.by == by].iterrows():
        out.append(f"| Missed, %: {lab} out-of-plane share, {r.level} tertile ({r.z_share_range.replace('-', '–')}) | {r.missed_pct_90:.1f} | {r.missed_pct_ref:.1f} |")
out += ["", "Spatial abnormality: mean-vector angle above the sex-specific 97.5th percentile of the healthy group (106° in men, 97° in women). "
        "Frontal upper reference limit: 69° in men, 68° in women. Out-of-plane share: absolute Z component of the mean vector divided by its magnitude.", ""]

# Table 5
rc = pd.read_csv(f"{O}/table_roc.csv")
out += ["**Table 5.** Discrimination between ECGs labelled only as normal (n = 8,282) and abnormal ECGs: area under the ROC curve.", "",
        "| Comparison | Cases, n | Spatial, mean (95% CI) | Frontal, Uni-G (95% CI) | P* | Spatial, peak | Frontal, 12SL | Spatial, reported by Uni-G |",
        "|---|---|---|---|---|---|---|---|"]
for _, r in rc.iterrows():
    c1, c2 = nums(r.ci_sp_mean), nums(r.ci_fr_gl)
    out.append(f"| {r.comparison} | {int(r.n_cases):,} | {r.auc_sp_mean:.3f} ({c1[0]:.3f}–{c1[1]:.3f}) | {r.auc_fr_gl:.3f} ({c2[0]:.3f}–{c2[1]:.3f}) | "
               f"{pv(r.p_sp_mean_vs_fr_gl)} | {r.auc_sp_peak:.3f} | {r.auc_fr_ge:.3f} | {r.auc_glasgow_vm:.3f} |")
out += ["", "One abnormal ECG lacked the spatial angle reported by Uni-G and was not included in the ROC analysis. *DeLong test, spatial mean-vector angle versus frontal Uni-G angle.", ""]

# Supplementary
lr = pd.read_csv(f"{O}/table_missed_predictors.csv", index_col=0)
lab = {"zt10": "Out-of-plane share of T vector (per 0.1)", "zq10": "Out-of-plane share of QRS vector (per 0.1)", "age10": "Age (per 10 years)",
       "female": "Female sex", "qrsd10": "QRS duration (per 10 ms)", "afib": "Atrial fibrillation or flutter", "MI": "Myocardial infarction",
       "HYP": "Hypertrophy", "CD": "Conduction disturbance", "STTC": "ST/T change"}
sup += ["**Supplementary Table S1.** Predictors of a missed spatial abnormality (frontal angle within its upper reference limit) "
        "among spatially abnormal ECGs: multivariable logistic regression (1,266 missed among 4,682).", "", "| Variable | OR (95% CI) | P |", "|---|---|---|"]
sup += [f"| {v} | {lr.loc[k].OR:.2f} ({lr.loc[k].lo:.2f}–{lr.loc[k].hi:.2f}) | {pv(lr.loc[k].p)} |" for k, v in lab.items()]
s = pd.read_csv(f"{O}/table_sensitivity.csv")
sup += ["", "**Supplementary Table S2.** Sensitivity analyses: discrimination of any abnormality from normal ECGs.", "",
        "| Analysis | n | AUC, spatial mean | AUC, frontal Uni-G | Spearman rho, spatial vs frontal |", "|---|---|---|---|---|"]
sup += [f"| {r.analysis} | {int(r.n):,} | {r.auc_sp_mean:.3f} | {r.auc_fr_gl:.3f} | {r.spearman_sp_fr:.2f} |" for _, r in s.iterrows()]
sup.append("")
open(OUT, "w", encoding="utf-8").write("\n".join(out)); open(SUP, "w", encoding="utf-8").write("\n".join(sup))
print("\n".join(out)); print("\n".join(sup))
