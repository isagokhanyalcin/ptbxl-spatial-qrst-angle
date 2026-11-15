"""Paper 2: frontal versus spatial QRS-T angle in PTB-XL+.
Spatial angle computed from Uni-G median beats with the Kors regression matrix (mean and peak vectors).
Usage: python3 -I 01_analysis_paper2.py <data_dir> <median_beat_dir> <out_dir>
"""
import sys, ast, os, json, warnings
import numpy as np, pandas as pd, wfdb
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
D, MB, O = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(O, exist_ok=True)
R = {}

# ---------------- labels ----------------
db = pd.read_csv(f"{D}/ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(f"{D}/scp_statements.csv", index_col=0); diag = scp[scp.diagnostic == 1]
db["scp"] = db.scp_codes.apply(ast.literal_eval)
db["sup"] = db.scp.apply(lambda d: sorted({diag.loc[k, "diagnostic_class"] for k in d if k in diag.index}))
db["norm_only"] = db.sup.apply(lambda s: s == ["NORM"])
db["path"] = db.sup.apply(lambda s: len(s) > 0 and s != ["NORM"])
for c in ["MI", "STTC", "HYP", "CD"]:
    db[c] = db.sup.apply(lambda s, c=c: c in s)
db["afib"] = db.scp.apply(lambda d: "AFIB" in d or "AFLT" in d)
db["bbb"] = db.scp.apply(lambda d: "CLBBB" in d or "CRBBB" in d)
db["pace"] = db.pacemaker.notna() | db.scp.apply(lambda d: "PACE" in d)
db["noise"] = db[["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]].notna().any(axis=1)
db["age_ok"] = db.age < 300
db["female"] = db.sex == 1

u = pd.read_csv(f"{D}/unig_features.csv").set_index("ecg_id")
g = pd.read_csv(f"{D}/12sl_features.csv").set_index("ecg_id")
fold = lambda a: (a.abs()).where(a.abs() <= 180, 360 - a.abs())

# ---------------- spatial angle from median beats ----------------
# Kors regression matrix, leads I, II, V1-V6 -> X, Y, Z (Kors et al., Eur Heart J 1990)
K = np.array([[0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
              [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
              [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31]])
LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(a @ b / na / nb, -1, 1))))


def ang2(a, b):  # frontal-plane (X, Y) projection angle
    return ang(a[:2], b[:2])


CACHE = f"{O}/spatial_vectors.csv"
rows = []; n_missing_leads = 0
for i in ([] if os.path.exists(CACHE) else db.index):
    p = f"{MB}/{i:06d}_medians"
    if i not in u.index or not os.path.exists(p + ".dat"):
        continue
    f = u.loc[i]
    if f[["QRS_On_Global", "QRS_Off_Global", "T_Off_Global"]].isna().any():
        continue
    try:
        r = wfdb.rdrecord(p)
    except Exception:
        continue
    if not set(LEADS) <= set(r.sig_name):
        n_missing_leads += 1
        continue
    fs = r.fs; sig = pd.DataFrame(r.p_signal, columns=r.sig_name)
    v = K @ sig[LEADS].values.T
    qon, qoff, toff = (int(round(f[c] * fs / 1000)) for c in ["QRS_On_Global", "QRS_Off_Global", "T_Off_Global"])
    if not (0 < qon < qoff < toff <= v.shape[1]) or toff - qoff < 20:
        continue
    pre = max(qon - 10, 0)
    out = {"ecg_id": i}
    for bl_name, base in [("", v[:, [qon]]), ("_blpre", v[:, pre:qon].mean(1, keepdims=True) if qon > pre else v[:, [qon]])]:
        w = v - base
        q, t = w[:, qon:qoff], w[:, qoff:toff]
        qm, tm = q.mean(1), t.mean(1)
        qp = q[:, np.argmax(np.linalg.norm(q, axis=0))]; tp = t[:, np.argmax(np.linalg.norm(t, axis=0))]
        out["sp_mean" + bl_name] = ang(qm, tm)
        out["sp_peak" + bl_name] = ang(qp, tp)
        if bl_name == "":
            out["fr_kors_mean"] = ang2(qm, tm)
            out["zq"] = abs(qm[2]) / np.linalg.norm(qm)
            out["zt"] = abs(tm[2]) / np.linalg.norm(tm)
            out["qm_x"], out["qm_y"], out["qm_z"] = qm
            out["tm_x"], out["tm_y"], out["tm_z"] = tm
    rows.append(out)
if os.path.exists(CACHE):
    sp = pd.read_csv(CACHE, index_col="ecg_id"); n_missing_leads = int(open(CACHE + '.meta').read())
else:
    sp = pd.DataFrame(rows).set_index("ecg_id"); sp.to_csv(CACHE); open(CACHE + '.meta', 'w').write(str(n_missing_leads))

feat = pd.DataFrame({"fr_gl": fold(u.QRS_AxisFront_Global - u.T_AxisFront_Global),
                     "fr_ge": fold(g.R_AxisFrontal_Global - g.T_AxisFront_Global),
                     "glasgow_vm": u["VM_Ang_QRS-T"], "indet": u.QRS_AxisIndet_Global == 1,
                     "qrsd": g.QRS_Dur_Global, "hr": g.get("HR_Ventr_Global")})
x = db.join(feat, how="inner").join(sp, how="inner")

flow = {"all_ecgs": len(db), "median_beat_missing_leads": n_missing_leads, "with_median_beat_and_fiducials": len(sp)}
x = x[x.fr_gl.notna() & x.fr_ge.notna() & x.sp_mean.notna()]; flow["with_all_angles"] = len(x)
x = x[~x.pace]; flow["after_pacing_excluded"] = len(x)
x = x[~x.indet.fillna(False).astype(bool)]; flow["after_indeterminate_axis_excluded"] = len(x)
x = x.sort_values(["patient_id", "recording_date"]).groupby("patient_id", sort=False).head(1)
flow["primary_first_ecg_per_patient"] = len(x)
R["flow"] = flow

# ---------------- reference group & limits ----------------
ref = x[x.norm_only & x.validated_by_human & x.age_ok & (x.age >= 18) & ~x.afib & ~x.noise].copy()
ref["agegrp"] = pd.cut(ref.age, [17, 29, 39, 49, 59, 69, 79, 120], labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"])
R["ref_n"] = len(ref)
lim = {m: {sx: ref.loc[ref.female == sx, m].quantile(.975) for sx in [False, True]} for m in ["sp_mean", "sp_peak", "fr_gl"]}
R["upper_limits_sex"] = {m: {("F" if k else "M"): round(v, 1) for k, v in d.items()} for m, d in lim.items()}

rows = []
for (gr, sx), d in ref.groupby(["agegrp", "female"], observed=True):
    for m in ["sp_mean", "sp_peak", "fr_gl", "glasgow_vm"]:
        v = d[m].dropna()
        rows.append({"age": gr, "sex": "F" if sx else "M", "measure": m, "n": len(v), "median": v.median(),
                     "p2.5": v.quantile(.025), "p97.5": v.quantile(.975)})
for sx, d in ref.groupby("female"):
    for m in ["sp_mean", "sp_peak", "fr_gl", "glasgow_vm"]:
        v = d[m].dropna()
        rows.append({"age": "all", "sex": "F" if sx else "M", "measure": m, "n": len(v), "median": v.median(),
                     "mean": v.mean(), "sd": v.std(), "p2.5": v.quantile(.025), "p97.5": v.quantile(.975)})
T_ref = pd.DataFrame(rows); T_ref["provisional_n_lt_120"] = T_ref.n < 120
T_ref.to_csv(f"{O}/table_reference_values.csv", index=False)

# ---------------- distributions ----------------
desc = []
for lab, d in [("All", x), ("Normal only", x[x.norm_only]), ("Any pathology", x[x.path]), ("Healthy reference", ref)]:
    for m in ["sp_mean", "sp_peak", "fr_gl", "fr_ge", "fr_kors_mean", "glasgow_vm"]:
        v = d[m].dropna()
        desc.append({"group": lab, "measure": m, "n": len(v), "median": v.median(), "q1": v.quantile(.25), "q3": v.quantile(.75), "mean": v.mean(), "sd": v.std()})
pd.DataFrame(desc).to_csv(f"{O}/table_distributions.csv", index=False)
R["spearman"] = {k: round(stats.spearmanr(x[a], x[b], nan_policy="omit")[0], 3) for k, (a, b) in
                 {"sp_mean_vs_fr_gl": ("sp_mean", "fr_gl"), "sp_mean_vs_fr_ge": ("sp_mean", "fr_ge"),
                  "sp_mean_vs_fr_kors": ("sp_mean", "fr_kors_mean"), "fr_kors_vs_fr_gl": ("fr_kors_mean", "fr_gl"),
                  "sp_peak_vs_glasgow_vm": ("sp_peak", "glasgow_vm"), "sp_mean_vs_sp_peak": ("sp_mean", "sp_peak")}.items()}

# ---------------- classification concordance ----------------
x["sp_abn"] = np.where(x.female, x.sp_mean > lim["sp_mean"][True], x.sp_mean > lim["sp_mean"][False])
x["fr_abn_ref"] = np.where(x.female, x.fr_gl > lim["fr_gl"][True], x.fr_gl > lim["fr_gl"][False])
x["fr_abn_90"] = x.fr_gl > 90
cc = []
for fr in ["fr_abn_90", "fr_abn_ref"]:
    ct = pd.crosstab(x.sp_abn, x[fr])
    a = int(ct.loc[True, True]); b = int(ct.loc[True, False]); c = int(ct.loc[False, True]); d_ = int(ct.loc[False, False])
    cc.append({"frontal_rule": fr, "spatial_abn": a + b, "both_abn": a, "spatial_only": b, "frontal_only": c, "both_normal": d_,
               "pct_spatial_abn_missed_by_frontal": 100 * b / (a + b), "kappa": cohen_kappa_score(x.sp_abn, x[fr]),
               "pct_overall": 100 * (a + b) / len(x)})
T_cc = pd.DataFrame(cc); T_cc.to_csv(f"{O}/table_concordance.csv", index=False)

# missed by frontal, by T-vector out-of-plane share
ab = x[x.sp_abn].copy()
ab["zt_tert"] = pd.qcut(ab.zt, 3, labels=["low", "mid", "high"])
ab["zq_tert"] = pd.qcut(ab.zq, 3, labels=["low", "mid", "high"])
miss = []
for col in ["zt_tert", "zq_tert"]:
    for lev, d in ab.groupby(col, observed=True):
        miss.append({"by": col, "level": lev, "n": len(d), "z_share_range": f"{d[col.replace('_tert','')].min():.2f}-{d[col.replace('_tert','')].max():.2f}",
                     "missed_pct_90": 100 * (~d.fr_abn_90).mean(), "missed_pct_ref": 100 * (~d.fr_abn_ref).mean()})
T_miss = pd.DataFrame(miss); T_miss.to_csv(f"{O}/table_missed_by_zshare.csv", index=False)
m = ab[ab.age_ok].copy()
m["missed"] = (~m.fr_abn_ref).astype(int); m["age10"] = m.age / 10; m["qrsd10"] = m.qrsd / 10
m["zt10"] = m.zt * 10; m["zq10"] = m.zq * 10
preds = ["zt10", "zq10", "age10", "female", "qrsd10", "afib", "MI", "HYP", "CD", "STTC"]
mm = m[preds + ["missed"]].dropna().astype(float)
lr = sm.Logit(mm.missed, sm.add_constant(mm[preds])).fit(disp=0)
T_lr = pd.DataFrame({"OR": np.exp(lr.params), "lo": np.exp(lr.conf_int()[0]), "hi": np.exp(lr.conf_int()[1]), "p": lr.pvalues}).drop("const")
T_lr.to_csv(f"{O}/table_missed_predictors.csv"); R["missed_model_n"] = len(mm); R["missed_model_events"] = int(mm.missed.sum())

# sex-specific versus pooled spatial limit
ra = ref[ref.age_ok].assign(fem=lambda d: d.female.astype(int))
ols = sm.OLS(ra.sp_mean, sm.add_constant(ra[["fem", "age"]].astype(float))).fit()
R["sex_effect_age_adjusted"] = {"coef": round(ols.params["fem"], 2), "lo": round(ols.conf_int().loc["fem", 0], 2),
                                "hi": round(ols.conf_int().loc["fem", 1], 2), "p": float(ols.pvalues["fem"])}
pooled = ref.sp_mean.quantile(.975); R["pooled_upper_limit_sp_mean"] = round(pooled, 1)
x["sp_abn_pooled"] = x.sp_mean > pooled
sx = []
for rule, col in [("sex-specific", "sp_abn"), ("pooled", "sp_abn_pooled")]:
    for grp, d in [("all", x), ("men", x[~x.female]), ("women", x[x.female])]:
        a = d[d[col]]
        sx.append({"limit": rule, "group": grp, "spatial_abn": len(a), "missed_pct_ref": 100 * (~a.fr_abn_ref).mean(),
                   "missed_pct_90": 100 * (~a.fr_abn_90).mean()})
sx.append({"limit": "reclassified", "group": "men, abnormal only with pooled", "spatial_abn": int((~x.female & x.sp_abn_pooled & ~x.sp_abn).sum())})
sx.append({"limit": "reclassified", "group": "women, abnormal only with sex-specific", "spatial_abn": int((x.female & x.sp_abn & ~x.sp_abn_pooled).sum())})
pd.DataFrame(sx).to_csv(f"{O}/table_sex_specific_vs_pooled.csv", index=False)


# ---------------- diagnostic value ----------------
def delong(y, s1, s2):
    y = np.asarray(y, bool); pos, neg = np.where(y)[0], np.where(~y)[0]
    def comps(s):
        X, Y = s[pos], s[neg]; r = stats.rankdata(np.r_[X, Y]); rx = stats.rankdata(X); ry = stats.rankdata(Y)
        m_, n_ = len(X), len(Y); auc = (r[:m_].sum() - m_ * (m_ + 1) / 2) / (m_ * n_)
        return auc, (r[:m_] - rx) / n_, 1 - (r[m_:] - ry) / m_
    a1, v10a, v01a = comps(np.asarray(s1, float)); a2, v10b, v01b = comps(np.asarray(s2, float))
    S = np.cov(np.vstack([v10a, v10b])) / len(pos) + np.cov(np.vstack([v01a, v01b])) / len(neg)
    z = (a1 - a2) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return a1, (a1 - 1.96 * np.sqrt(S[0, 0]), a1 + 1.96 * np.sqrt(S[0, 0])), a2, (a2 - 1.96 * np.sqrt(S[1, 1]), a2 + 1.96 * np.sqrt(S[1, 1])), 2 * stats.norm.sf(abs(z))

roc_rows, curves = [], {}
for name, cases in [("Any pathology", x.path), ("MI", x.MI), ("STTC", x.STTC), ("HYP", x.HYP), ("CD", x.CD)]:
    sub = x[(x.norm_only | cases) & x.glasgow_vm.notna()]; y = cases.reindex(sub.index).values
    a1, c1, a2, c2, p = delong(y, sub.sp_mean.values, sub.fr_gl.values)
    row = {"comparison": name, "n_cases": int(y.sum()), "n_norm": int((~y).sum()),
           "auc_sp_mean": a1, "ci_sp_mean": c1, "auc_fr_gl": a2, "ci_fr_gl": c2, "p_sp_mean_vs_fr_gl": p}
    for mname in ["sp_peak", "fr_ge", "glasgow_vm"]:
        row[f"auc_{mname}"] = roc_auc_score(y, sub[mname])
    row["p_sp_mean_vs_sp_peak"] = delong(y, sub.sp_mean.values, sub.sp_peak.values)[4]
    roc_rows.append(row)
    curves[name] = {mname: roc_curve(y, sub[mname])[:2] for mname in ["sp_mean", "sp_peak", "fr_gl", "glasgow_vm"]}
T_roc = pd.DataFrame(roc_rows); T_roc.to_csv(f"{O}/table_roc.csv", index=False)

# sensitivity: alternative baseline, QRS<120, no AF
sens = []
for lab, d in [("Primary", x), ("Pre-QRS baseline", x.assign(sp_mean=x.sp_mean_blpre)), ("QRS < 120 ms", x[x.qrsd < 120]),
               ("No AF/flutter", x[~x.afib]), ("No quality flag", x[~x.noise]), ("Validated reports", x[x.validated_by_human])]:
    s2 = d[d.norm_only | d.path]
    sens.append({"analysis": lab, "n": len(s2), "auc_sp_mean": roc_auc_score(s2.path, s2.sp_mean), "auc_fr_gl": roc_auc_score(s2.path, s2.fr_gl),
                 "spearman_sp_fr": stats.spearmanr(d.sp_mean, d.fr_gl)[0]})
pd.DataFrame(sens).to_csv(f"{O}/table_sensitivity.csv", index=False)
R["baseline_agreement_median_absdiff"] = float((x.sp_mean - x.sp_mean_blpre).abs().median())

# Glasgow reported spatial angle vs Kors peak
gv = x[x.glasgow_vm.notna()]
dd = gv.glasgow_vm - gv.sp_peak
R["glasgow_vs_korspeak"] = {"n": len(gv), "bias": dd.mean(), "loa": [dd.mean() - 1.96 * dd.std(), dd.mean() + 1.96 * dd.std()],
                            "spearman": stats.spearmanr(gv.glasgow_vm, gv.sp_peak)[0],
                            "healthy_median_glasgow": ref.glasgow_vm.median(), "healthy_median_korspeak": ref.sp_peak.median()}

# ---------------- cohort table ----------------
def t1(d):
    gg = d[d.age_ok]
    return {"n": len(d), "age_median": gg.age.median(), "age_q1": gg.age.quantile(.25), "age_q3": gg.age.quantile(.75),
            "female_pct": 100 * d.female.mean(), "norm_only_pct": 100 * d.norm_only.mean(), "MI_pct": 100 * d.MI.mean(),
            "STTC_pct": 100 * d.STTC.mean(), "HYP_pct": 100 * d.HYP.mean(), "CD_pct": 100 * d.CD.mean(),
            "afib_pct": 100 * d.afib.mean(), "bbb_pct": 100 * d.bbb.mean(), "qrsd_median": d.qrsd.median()}
pd.DataFrame({"All": t1(x), "Healthy reference": t1(ref)}).to_csv(f"{O}/table_cohort.csv")

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
for ax, fem, lab in [(axs[0], False, "Men"), (axs[1], True, "Women")]:
    d = x[x.female == fem]; ul = lim["sp_mean"][fem]
    ax.hexbin(d.fr_gl, d.sp_mean, gridsize=50, bins="log", cmap="Greys", mincnt=1, extent=(0, 180, 0, 180))
    ax.axvline(90, color="#cc3333", ls=":", lw=1); ax.text(91, 4, "frontal 90°", color="#cc3333", fontsize=8)
    ax.axhline(ul, color="#1f5fbf", ls="--", lw=1); ax.text(178, ul + 2, f"spatial upper limit {ul:.0f}°", color="#1f5fbf", fontsize=8, ha="right")
    ax.plot([0, 180], [0, 180], color="#bbbbbb", lw=0.8)
    ax.set_title(f"{lab} (n = {len(d):,})", fontsize=10); ax.set_xlabel("Frontal QRS-T angle, Uni-G axes (°)")
    ax.set_xlim(0, 180); ax.set_ylim(0, 180)
axs[0].set_ylabel("Spatial QRS-T angle, Kors mean vectors (°)")
fig.tight_layout(); fig.savefig(f"{O}/fig1_frontal_vs_spatial.png", dpi=300); fig.savefig(f"{O}/fig1_frontal_vs_spatial.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.6))
sub = T_miss[T_miss.by == "zt_tert"]
ax.bar(np.arange(3) - 0.18, sub.missed_pct_90, 0.36, color="#bbbbbb", label="frontal cut-off 90°")
ax.bar(np.arange(3) + 0.18, sub.missed_pct_ref, 0.36, color="#1f5fbf", label="frontal upper reference limit")
for k, (a, b) in enumerate(zip(sub.missed_pct_90, sub.missed_pct_ref)):
    ax.text(k - 0.18, a + 1, f"{a:.0f}%", ha="center", fontsize=8); ax.text(k + 0.18, b + 1, f"{b:.0f}%", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{l} tertile\n({r})" for l, r in zip(sub.level, sub.z_share_range)])
ax.set_xlabel("Out-of-plane (Z) share of the mean T vector"); ax.set_ylabel("Spatially abnormal ECGs\nwith a normal frontal angle (%)")
ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{O}/fig2_missed_by_zshare.png", dpi=300); fig.savefig(f"{O}/fig2_missed_by_zshare.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
for axx, (name, c) in zip(axs, curves.items()):
    for mname, col, ls, lab in [("sp_mean", "#1f5fbf", "-", "Spatial, mean"), ("sp_peak", "#1f5fbf", "--", "Spatial, peak"),
                                ("fr_gl", "#888888", "-", "Frontal"), ("glasgow_vm", "#cc8833", ":", "Glasgow spatial")]:
        axx.plot(c[mname][0], c[mname][1], color=col, ls=ls, lw=1.2, label=lab)
    r = T_roc[T_roc.comparison == name].iloc[0]
    axx.plot([0, 1], [0, 1], color="#cccccc", lw=0.8); axx.set_title(f"{name}\nAUC {r.auc_sp_mean:.3f} vs {r.auc_fr_gl:.3f}")
    axx.set_xlabel("1 − specificity")
axs[0].set_ylabel("Sensitivity"); axs[0].legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout(); fig.savefig(f"{O}/fig3_roc.png", dpi=300); fig.savefig(f"{O}/fig3_roc.pdf"); plt.close(fig)

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 4)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(R), open(f"{O}/key_results.json", "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
print(json.dumps(clean(R), indent=1, ensure_ascii=False))
print(pd.read_csv(f"{O}/table_distributions.csv").to_string())
print(T_cc.to_string()); print(T_miss.to_string()); print(T_lr); print(T_roc.to_string())
print(pd.read_csv(f"{O}/table_sensitivity.csv").to_string())
print(T_ref[T_ref.age == "all"].to_string()); print(T_ref[T_ref.measure == "sp_mean"].to_string())
