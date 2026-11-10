"""Additional analyses for the revision of MTCOMM-D-26-07228 (Reviewers 1 and 2).

All models, preprocessing, cross-validation and entropy-weighted TOPSIS weighting
are imported from generate_all_figures.py, so the new analyses use exactly the
pipeline of the main-text results.

Parts
-----
1. Repeated stratified splits (seeds 0-19) with stored predictions:
   mean and adsorbate-mean baselines, adsorbate-resolved metrics, cross-conformal
   prediction intervals, pairwise ranking accuracy, ensemble variants,
   within-adsorbate (conditional) permutation importance and rank stability.
2. Grouped cross-validation: host, parent MXene, primary metal, termination,
   data source and adsorbate held out.
3. Reference split: Holm/BH-adjusted Wilcoxon tests, missingness-indicator
   concordance and ablation.
4. Activation-energy subset: within-channel rank correlations and a
   channel-stratified permutation test.
5. Dopant domain: descriptor shift, applicability-domain flags, interval coverage
   and descriptor back-filling.
6. Few-shot dopant analysis with identical training/test records for the ensemble
   and ExtraTrees, pooled and dopant-only training, a host-adsorbate group-mean
   baseline and three sampling designs (random, dopant-disjoint, leave one
   host-adsorbate combination out).
7. Software versions and final model configurations.

Outputs: outputs/tables/revision/ (CSV); the Supporting Information figures are drawn by
code/figures/revision_si_figures.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import generate_all_figures as g  # noqa: E402

assert g.ALIASES["Rd"] == "Pd"  # Pd@W2CO2 records (see generate_all_figures.ALIASES)

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "outputs" / "tables" / ("revision_quicktest" if os.environ.get("REV_QUICK") else "revision")
FIG = ROOT / "outputs" / "figures" / "FigS"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

QUICK = bool(os.environ.get("REV_QUICK"))  # smoke test with very few repetitions
N_SPLITS = 2 if QUICK else 20
N_DRAWS = 1 if QUICK else 20
N_JOBS = 6
ALPHA = 0.10  # 90% prediction intervals
MODEL_ORDER = ["RF", "ExtraTrees", "GBR", "HGBR", "SVR", "MLP"]
SUBSTRATE_NUM = ["a_ang", "Layer_dist_MX", "Layer_dist_MM", "Length_MX", "E_Form",
                 "Bader_M1", "Bader_X", "Band_gap_PBE_ev"]


def save(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8-sig")


# --------------------------------------------------------------------------- utils
def fit_suite(x_train: pd.DataFrame, y_train: pd.Series, x_ref: pd.DataFrame) -> dict:
    """Same protocol as generate_all_figures.prepare_training, keeping the
    out-of-fold predictions so that ensemble residuals can be calibrated."""
    models = g.make_models(g.build_preprocessor(x_ref))
    y_range = float(y_train.max() - y_train.min())
    n_folds = min(g.CV_FOLDS, len(y_train))
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=g.RANDOM_STATE)
    oof, rows = {}, []
    for name, pipe in models.items():
        oof[name] = cross_val_predict(clone(pipe), x_train, y_train, cv=cv)
        rows.append({"Model": name, **g.metrics(y_train, oof[name], y_range)})
    cvm = pd.DataFrame(rows).set_index("Model")
    top = g.entropy_topsis(cvm[g.METRIC_COLS])
    w = top["ensemble_weight"].reindex(cvm.index)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = pd.Series(1 / len(cvm), index=cvm.index)
    fitted = {n: clone(p).fit(x_train, y_train) for n, p in models.items()}
    oof_df = pd.DataFrame(oof, index=x_train.index)
    return {"fitted": fitted, "w": w, "cvm": cvm, "oof": oof_df, "top": top}


def topsis_two_criteria(cvm: pd.DataFrame) -> pd.Series:
    """Entropy-TOPSIS on the two non-redundant criteria (RMSE, MAE)."""
    benefit = pd.DataFrame({c: 1 / (cvm[c] + 1e-9) for c in ["RMSE", "MAE"]}, index=cvm.index)
    arr = benefit.to_numpy(float)
    arr = (arr - arr.min(0)) / (arr.max(0) - arr.min(0) + 1e-12)
    p = arr / (arr.sum(0, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(0) / np.log(len(cvm))
    wc = (1 - ent) / np.sum(1 - ent)
    wt = arr * wc
    dpos = np.sqrt(((wt - wt.max(0)) ** 2).sum(1))
    dneg = np.sqrt(((wt - wt.min(0)) ** 2).sum(1))
    clo = dneg / (dpos + dneg + 1e-12)
    return pd.Series(clo / clo.sum(), index=cvm.index)


def ens_predict(suite: dict, x: pd.DataFrame, w: pd.Series | None = None) -> tuple[dict, np.ndarray]:
    w = suite["w"] if w is None else w
    preds = {n: m.predict(x) for n, m in suite["fitted"].items()}
    ens = np.column_stack([preds[n] for n in w.index]) @ w.to_numpy()
    return preds, ens


def conformal_q(resid: np.ndarray, alpha: float = ALPHA) -> float:
    r = np.sort(np.abs(np.asarray(resid, float)))
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[min(k, n) - 1])


def mondrian_halfwidth(resid: pd.Series, groups_train: pd.Series, groups_test: pd.Series, alpha: float = ALPHA) -> np.ndarray:
    q_all = conformal_q(resid.to_numpy(), alpha)
    q = {gname: conformal_q(resid[groups_train == gname].to_numpy(), alpha) for gname in groups_train.unique()}
    return np.array([q.get(gname, q_all) if np.isfinite(q.get(gname, np.nan)) else q_all for gname in groups_test])


def group_mean_baseline(y_tr: pd.Series, key_tr: pd.Series, key_te: pd.Series, fallback: np.ndarray) -> np.ndarray:
    m = y_tr.groupby(key_tr.to_numpy()).mean()
    out = key_te.map(m).to_numpy(dtype=float)
    return np.where(np.isfinite(out), out, fallback)


def within_group_rho(y: np.ndarray, p: np.ndarray, groups: np.ndarray, min_n: int = 3) -> tuple[float, int]:
    vals, weights = [], []
    for gname in pd.unique(groups):
        m = groups == gname
        if m.sum() < min_n or np.ptp(y[m]) == 0:
            continue
        rho = 0.0 if np.ptp(p[m]) == 0 else spearmanr(y[m], p[m])[0]
        vals.append(rho)
        weights.append(m.sum())
    if not vals:
        return np.nan, 0
    return float(np.average(vals, weights=weights)), int(len(vals))


def skill(y: np.ndarray, p: np.ndarray, base: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2))


def groups_for(wgs: pd.DataFrame) -> dict[str, pd.Series]:
    nl = wgs["N_Layers"].copy()
    mode = wgs.groupby("Formula")["N_Layers"].agg(lambda s: s.mode().iloc[0])
    bad = nl > 20  # one Ti2NH2 record lists 55 layers in the source database
    nl[bad] = wgs.loc[bad, "Formula"].map(mode)
    nskel = (nl - 2 * wgs["T1_type"].notna()).astype(int)
    parent = (wgs["M1"].astype(str) + "|" + wgs["M2"].fillna("-").astype(str) + "|" + wgs["X"].astype(str) + "|" + nskel.astype(str))

    def norm_doi(s):
        if pd.isna(s):
            return "no literature DOI"
        s = str(s).strip().lower()
        for pre in ["https://doi.org/", "https://pubs.acs.org/doi/", "doi: ", "doi:"]:
            s = s.replace(pre, "")
        return s.strip()

    return {
        "host": wgs["Formula"].astype(str),
        "parent": parent,
        "metal": wgs["M1"].astype(str),
        "termination": wgs["T1_type"].fillna("bare").astype(str),
        "source": wgs["DOI_1(Energies)"].map(norm_doi),
        "adsorbate": wgs["Mol"].astype(str),
    }


def balanced_group_folds(groups: pd.Series, n_folds: int, seed: int = g.RANDOM_STATE) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    order = rng.permutation(sizes.index.to_numpy())
    order = sorted(order, key=lambda k: -sizes[k])  # largest first, random tie-break
    load = np.zeros(n_folds)
    assign = {}
    for k in order:
        f = int(np.argmin(load))
        assign[k] = f
        load[f] += sizes[k]
    fold = groups.map(assign).to_numpy()
    idx = np.arange(len(groups))
    return [(idx[fold != f], idx[fold == f]) for f in range(n_folds)]


def logo_folds(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(groups))
    gv = groups.to_numpy()
    return [(idx[gv != k], idx[gv == k]) for k in pd.unique(gv)]


# --------------------------------------------------------------------------- 1
DESCRIPTOR_FAMILIES = {
    "Termination chemistry": ["T1_type", "has_termination"] + [f"T1_type_{p}" for p in ["Z", "group", "period", "en", "radius"]],
    "Metal identity (M1, M2)": ["M1", "M2"] + [f"{m}_{p}" for m in ["M1", "M2"] for p in ["Z", "group", "period", "en", "radius"]],
    "X element (C/N)": ["X"] + [f"X_{p}" for p in ["Z", "group", "period", "en", "radius"]],
    "Charge distribution (Bader)": ["Bader_M1", "Bader_M2", "Bader_X", "Bader_T1"],
    "Slab geometry": ["a_ang", "N_Layers", "Layer_dist_MX", "Layer_dist_MM", "Length_MX", "Length_MT1", "Stacking"],
    "Formation energy and band gap": ["E_Form", "Band_gap_PBE_ev"],
    "Adsorbate descriptors": ["Mol"] + [f"Mol_{p}" for p in ["mw", "atoms", "valence", "radical_e"]],
}


def family_columns(x: pd.DataFrame, fam: list[str]) -> list[str]:
    cols = []
    for c in fam:
        if c in x.columns:
            cols.append(c)
        if f"{c}_missing" in x.columns:
            cols.append(f"{c}_missing")
    return cols


def conditional_family_importance(pipe, x_te: pd.DataFrame, y_te: np.ndarray, ads_te: np.ndarray,
                                  base_te: np.ndarray, seed: int, n_rep: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    p0 = pipe.predict(x_te)
    rmse0 = g.rmse(y_te, p0)
    sk0 = skill(y_te, p0, base_te)
    rows = []
    for fam, members in DESCRIPTOR_FAMILIES.items():
        cols = family_columns(x_te, members)
        d_rmse, d_skill, d_rmse_unc = [], [], []
        for _ in range(n_rep):
            xp = x_te.copy()
            if fam != "Adsorbate descriptors":
                for a in np.unique(ads_te):
                    m = np.where(ads_te == a)[0]
                    perm = rng.permutation(m)
                    for c in cols:  # column-wise assignment keeps each dtype
                        xp.iloc[m, xp.columns.get_loc(c)] = x_te[c].iloc[perm].to_numpy()
            xu = x_te.copy()
            permu = rng.permutation(len(x_te))
            for c in cols:
                xu[c] = x_te[c].iloc[permu].to_numpy()
            pp = pipe.predict(xp)
            pu = pipe.predict(xu)
            d_rmse.append(g.rmse(y_te, pp) - rmse0)
            d_skill.append(sk0 - skill(y_te, pp, base_te))
            d_rmse_unc.append(g.rmse(y_te, pu) - rmse0)
        rows.append({"Family": fam, "n_columns": len(cols),
                     "dRMSE_within_ads": float(np.mean(d_rmse)), "dSkill_within_ads": float(np.mean(d_skill)),
                     "dRMSE_unconditional": float(np.mean(d_rmse_unc))})
    return rows


def run_split(seed: int, x_wgs: pd.DataFrame, y: pd.Series, wgs: pd.DataFrame) -> dict:
    strat = wgs["Mol"].fillna("None").astype(str)
    xtr, xte, ytr, yte = train_test_split(x_wgs, y, test_size=0.2, random_state=seed, stratify=strat)
    suite = fit_suite(xtr, ytr, x_wgs)
    preds, ens = ens_predict(suite, xte)
    w2 = topsis_two_criteria(suite["cvm"])
    w_eq = pd.Series(1 / 6, index=suite["w"].index)
    _, ens2 = ens_predict(suite, xte, w2)
    _, ens_eq = ens_predict(suite, xte, w_eq)
    oof_ens = suite["oof"][suite["w"].index].to_numpy() @ suite["w"].to_numpy()
    resid = pd.Series(ytr.to_numpy() - oof_ens, index=xtr.index)
    ads_tr = wgs.loc[xtr.index, "Mol"].astype(str)
    ads_te = wgs.loc[xte.index, "Mol"].astype(str)
    term_tr = wgs.loc[xtr.index, "T1_type"].fillna("bare").astype(str)
    term_te = wgs.loc[xte.index, "T1_type"].fillna("bare").astype(str)
    gm = np.full(len(yte), ytr.mean())
    am = group_mean_baseline(ytr, ads_tr, ads_te, gm)
    atm = group_mean_baseline(ytr, ads_tr + "|" + term_tr, ads_te + "|" + term_te, am)
    hw_m = mondrian_halfwidth(resid, ads_tr, ads_te, ALPHA)
    hw_g = np.full(len(yte), conformal_q(resid.to_numpy(), ALPHA))
    hw_m68 = mondrian_halfwidth(resid, ads_tr, ads_te, 0.32)
    # rank stability: permutation importance of ExtraTrees on original columns
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(suite["fitted"]["ExtraTrees"], xte, yte, n_repeats=5, random_state=seed, scoring="r2", n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=xte.columns)
    cond = conditional_family_importance(suite["fitted"]["ExtraTrees"], xte, yte.to_numpy(), ads_te.to_numpy(), am, seed)
    return {
        "seed": seed, "idx": xte.index.to_numpy(), "y": yte.to_numpy(), "ads": ads_te.to_numpy(),
        "term": term_te.to_numpy(), "host": wgs.loc[xte.index, "Formula"].astype(str).to_numpy(),
        "preds": preds, "ens": ens, "ens2": ens2, "ens_eq": ens_eq, "w": suite["w"], "w2": w2,
        "cvm": suite["cvm"], "gm": gm, "am": am, "atm": atm,
        "hw_m": hw_m, "hw_g": hw_g, "hw_m68": hw_m68, "imp": imp, "cond": cond,
        "y_range": float(ytr.max() - ytr.min()),
    }


def summarise_splits(res: list[dict]) -> dict[str, pd.DataFrame]:
    out = {}
    # (a) baselines, models and ensemble variants
    rows = []
    for r in res:
        y = r["y"]
        cands = {"Global-mean baseline": r["gm"], "Adsorbate-mean baseline": r["am"],
                 "Adsorbate x termination mean baseline": r["atm"],
                 **{m: r["preds"][m] for m in MODEL_ORDER},
                 "TOPSIS ensemble (4 criteria)": r["ens"], "TOPSIS ensemble (RMSE, MAE)": r["ens2"],
                 "Equal-weight average": r["ens_eq"]}
        for name, p in cands.items():
            met = g.metrics(y, p, r["y_range"])
            rho_w, _ = within_group_rho(y, p, r["ads"], 5)
            rows.append({"seed": r["seed"], "Predictor": name, "R2": met["R2"], "RMSE": met["RMSE"], "MAE": met["MAE"],
                         "Skill_vs_adsorbate_mean": skill(y, p, r["am"]), "Within_adsorbate_rho": rho_w})
    perf = pd.DataFrame(rows)
    out["raw_split_performance"] = perf
    order = list(dict.fromkeys(perf["Predictor"]))
    agg = perf.groupby("Predictor").agg(
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"), R2_min=("R2", "min"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), RMSE_max=("RMSE", "max"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        Skill_mean=("Skill_vs_adsorbate_mean", "mean"), Skill_sd=("Skill_vs_adsorbate_mean", "std"),
        rho_w_mean=("Within_adsorbate_rho", "mean"), rho_w_sd=("Within_adsorbate_rho", "std"),
    ).loc[order]
    piv = perf.pivot(index="seed", columns="Predictor", values="RMSE")
    agg["Splits_RMSE_le_ExtraTrees"] = [(piv[c] <= piv["ExtraTrees"]).sum() for c in agg.index]
    piv_r2 = perf.pivot(index="seed", columns="Predictor", values="R2")
    agg["Wilcoxon_p_R2_vs_ExtraTrees"] = [np.nan if c == "ExtraTrees" else wilcoxon(piv_r2[c], piv_r2["ExtraTrees"]).pvalue for c in agg.index]
    out["split_performance_summary"] = agg.reset_index()

    # (b) adsorbate-resolved accuracy, pooled over the 20 hold-out sets
    pooled = pd.DataFrame({
        "seed": np.concatenate([[r["seed"]] * len(r["y"]) for r in res]),
        "ads": np.concatenate([r["ads"] for r in res]), "term": np.concatenate([r["term"] for r in res]),
        "y": np.concatenate([r["y"] for r in res]), "ens": np.concatenate([r["ens"] for r in res]),
        "et": np.concatenate([r["preds"]["ExtraTrees"] for r in res]), "am": np.concatenate([r["am"] for r in res]),
        "atm": np.concatenate([r["atm"] for r in res]), "hw_m": np.concatenate([r["hw_m"] for r in res]),
        "hw_g": np.concatenate([r["hw_g"] for r in res]), "hw_m68": np.concatenate([r["hw_m68"] for r in res]),
    })
    out["pooled_holdout_predictions"] = pooled
    rows = []
    for a, s in [("All", pooled)] + list(pooled.groupby("ads")):
        per_split_rho = []
        for _, ss in s.groupby("seed"):
            if len(ss) >= 5 and np.ptp(ss["y"]) > 0:
                per_split_rho.append(spearmanr(ss["y"], ss["ens"])[0])
        rows.append({
            "Adsorbate": a, "n_per_split": len(s) / N_SPLITS,
            "DFT_SD_within": float(s["y"].std()) if a != "All" else float(s.groupby("ads")["y"].std().mean()),
            "RMSE_ensemble": g.rmse(s["y"], s["ens"]), "MAE_ensemble": float(np.mean(np.abs(s["y"] - s["ens"]))),
            "RMSE_adsorbate_mean": g.rmse(s["y"], s["am"]), "RMSE_ads_x_term_mean": g.rmse(s["y"], s["atm"]),
            "Skill_vs_adsorbate_mean": skill(s["y"].to_numpy(), s["ens"].to_numpy(), s["am"].to_numpy()),
            "Skill_ET_vs_adsorbate_mean": skill(s["y"].to_numpy(), s["et"].to_numpy(), s["am"].to_numpy()),
            "Spearman_rho_pooled": float(spearmanr(s["y"], s["ens"])[0]),
            "Spearman_rho_per_split_mean": float(np.mean(per_split_rho)) if per_split_rho else np.nan,
            "Coverage90_mondrian": float(np.mean(np.abs(s["y"] - s["ens"]) <= s["hw_m"])),
            "Halfwidth90_mondrian": float(s["hw_m"].mean()),
            "Coverage90_global": float(np.mean(np.abs(s["y"] - s["ens"]) <= s["hw_g"])),
            "Halfwidth90_global": float(s["hw_g"].mean()),
            "Coverage68_mondrian": float(np.mean(np.abs(s["y"] - s["ens"]) <= s["hw_m68"])),
            "Halfwidth68_mondrian": float(s["hw_m68"].mean()),
        })
    out["adsorbate_resolved"] = pd.DataFrame(rows)

    # (c) pairwise ranking accuracy within adsorbate
    bins = [0, 0.1, 0.25, 0.5, 1.0, np.inf]
    labels = ["<=0.10", "0.10-0.25", "0.25-0.50", "0.50-1.00", ">1.00"]
    counts = {(m, b): [0, 0] for m in ["ens", "et"] for b in labels}
    for r in res:
        y, a = r["y"], r["ads"]
        for m, p in [("ens", r["ens"]), ("et", r["preds"]["ExtraTrees"])]:
            for ads in np.unique(a):
                ii = np.where(a == ads)[0]
                for i in range(len(ii)):
                    for j in range(i + 1, len(ii)):
                        dy = y[ii[i]] - y[ii[j]]
                        if dy == 0:
                            continue
                        b = labels[np.searchsorted(bins, abs(dy), side="left") - 1]
                        counts[(m, b)][1] += 1
                        counts[(m, b)][0] += int(np.sign(p[ii[i]] - p[ii[j]]) == np.sign(dy))
    rows = [{"DFT_gap_eV": b, "Pairs": counts[("ens", b)][1],
             "Ensemble_correct_order": counts[("ens", b)][0] / max(counts[("ens", b)][1], 1),
             "ExtraTrees_correct_order": counts[("et", b)][0] / max(counts[("et", b)][1], 1)} for b in labels]
    out["pairwise_ranking"] = pd.DataFrame(rows)

    # (d) conditional family importance
    cond = pd.DataFrame([dict(row, seed=r["seed"]) for r in res for row in r["cond"]])
    out["conditional_importance"] = cond.groupby("Family").agg(
        n_columns=("n_columns", "first"),
        dRMSE_within_ads_mean=("dRMSE_within_ads", "mean"), dRMSE_within_ads_sd=("dRMSE_within_ads", "std"),
        dSkill_within_ads_mean=("dSkill_within_ads", "mean"), dSkill_within_ads_sd=("dSkill_within_ads", "std"),
        dRMSE_unconditional_mean=("dRMSE_unconditional", "mean"),
    ).sort_values("dRMSE_within_ads_mean", ascending=False).reset_index()

    # (e) permutation-importance rank stability
    imp = pd.DataFrame([r["imp"] for r in res])
    ranks = imp.rank(axis=1, ascending=False)
    stab = pd.DataFrame({"mean_importance": imp.mean(), "sd_importance": imp.std(),
                         "median_rank": ranks.median(), "top3_frequency": (ranks <= 3).sum(),
                         "top10_frequency": (ranks <= 10).sum()}).sort_values("median_rank")
    out["importance_stability"] = stab.head(20).reset_index(names="Feature")

    # (f) ensemble weights
    wt = pd.DataFrame([r["w"] for r in res])
    wt2 = pd.DataFrame([r["w2"] for r in res])
    out["ensemble_weights"] = pd.DataFrame({"TOPSIS4_mean": wt.mean(), "TOPSIS4_sd": wt.std(),
                                            "TOPSIS2_mean": wt2.mean(), "TOPSIS2_sd": wt2.std()}).reset_index(names="Model")
    return out


# --------------------------------------------------------------------------- 2
def eval_fold(x: pd.DataFrame, y: pd.Series, wgs: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> dict:
    xtr, ytr, xte = x.iloc[tr], y.iloc[tr], x.iloc[te]
    suite = fit_suite(xtr, ytr, x)
    preds, ens = ens_predict(suite, xte)
    oof_ens = suite["oof"][suite["w"].index].to_numpy() @ suite["w"].to_numpy()
    resid = pd.Series(ytr.to_numpy() - oof_ens, index=xtr.index)
    ads_tr = wgs["Mol"].iloc[tr].astype(str)
    ads_te = wgs["Mol"].iloc[te].astype(str)
    term_tr = wgs["T1_type"].iloc[tr].fillna("bare").astype(str)
    term_te = wgs["T1_type"].iloc[te].fillna("bare").astype(str)
    gm = np.full(len(te), ytr.mean())
    am = group_mean_baseline(ytr, ads_tr, ads_te, gm)
    atm = group_mean_baseline(ytr, ads_tr + "|" + term_tr, ads_te + "|" + term_te, am)
    hw = mondrian_halfwidth(resid, ads_tr, ads_te, ALPHA)
    return {"te": te, "ens": ens, "et": preds["ExtraTrees"], "gm": gm, "am": am, "atm": atm, "hw": hw}


def grouped_cv(x_wgs: pd.DataFrame, y: pd.Series, wgs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grp = groups_for(wgs)
    idx = np.arange(len(y))
    rnd = list(KFold(5, shuffle=True, random_state=g.RANDOM_STATE).split(idx))
    schemes = {
        "Random 5-fold CV": rnd,
        "Leave-host-out (5 folds)": balanced_group_folds(grp["host"], 5),
        "Leave-parent-MXene-out (5 folds)": balanced_group_folds(grp["parent"], 5),
        "Leave-one-primary-metal-out": logo_folds(grp["metal"]),
        "Leave-one-termination-out": logo_folds(grp["termination"]),
        "Leave-one-data-source-out": logo_folds(grp["source"]),
        "Leave-one-adsorbate-out": logo_folds(grp["adsorbate"]),
    }
    n_groups = {"Random 5-fold CV": 5, "Leave-host-out (5 folds)": grp["host"].nunique(),
                "Leave-parent-MXene-out (5 folds)": grp["parent"].nunique(),
                "Leave-one-primary-metal-out": grp["metal"].nunique(),
                "Leave-one-termination-out": grp["termination"].nunique(),
                "Leave-one-data-source-out": grp["source"].nunique(),
                "Leave-one-adsorbate-out": grp["adsorbate"].nunique()}
    tasks = [(name, k, tr, te) for name, folds in schemes.items() for k, (tr, te) in enumerate(folds)]
    if QUICK:
        tasks = [t for i, t in enumerate(tasks) if t[1] == 0]
    outs = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(eval_fold)(x_wgs, y, wgs, tr, te) for _, _, tr, te in tasks)
    rows, per_group = [], []
    yv = y.to_numpy()
    ads = wgs["Mol"].astype(str).to_numpy()
    for name in schemes:
        res = [o for (n, _, _, _), o in zip(tasks, outs) if n == name]
        te = np.concatenate([o["te"] for o in res])
        cols = {k: np.concatenate([o[k] for o in res]) for k in ["ens", "et", "gm", "am", "atm", "hw"]}
        yy, aa = yv[te], ads[te]
        for pred_name, key in [("TOPSIS ensemble", "ens"), ("ExtraTrees", "et"), ("Adsorbate-mean baseline", "am"),
                               ("Adsorbate x termination mean baseline", "atm"), ("Global-mean baseline", "gm")]:
            p = cols[key]
            rho_w, _ = within_group_rho(yy, p, aa, 5)
            rows.append({"Scheme": name, "Groups": n_groups[name], "Folds": len(res), "Predictor": pred_name,
                         "R2": float(1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2)),
                         "RMSE": g.rmse(yy, p), "MAE": float(np.mean(np.abs(yy - p))),
                         "Skill_vs_adsorbate_mean": skill(yy, p, cols["am"]) if key != "am" else 0.0,
                         "Within_adsorbate_rho": rho_w,
                         "Coverage90": float(np.mean(np.abs(yy - cols["ens"]) <= cols["hw"])) if key == "ens" else np.nan})
        if name in ["Leave-one-primary-metal-out", "Leave-one-termination-out", "Leave-one-data-source-out", "Leave-one-adsorbate-out"]:
            key = {"Leave-one-primary-metal-out": "metal", "Leave-one-termination-out": "termination",
                   "Leave-one-data-source-out": "source", "Leave-one-adsorbate-out": "adsorbate"}[name]
            gl = grp[key].to_numpy()[te]
            for gname in pd.unique(gl):
                m = gl == gname
                per_group.append({"Scheme": name, "Held-out group": gname, "n": int(m.sum()),
                                  "RMSE_ensemble": g.rmse(yy[m], cols["ens"][m]), "RMSE_adsorbate_mean": g.rmse(yy[m], cols["am"][m]),
                                  "MAE_ensemble": float(np.mean(np.abs(yy[m] - cols["ens"][m]))),
                                  "Bias_ensemble": float(np.mean(cols["ens"][m] - yy[m])),
                                  "Coverage90": float(np.mean(np.abs(yy[m] - cols["ens"][m]) <= cols["hw"][m]))})
    return pd.DataFrame(rows), pd.DataFrame(per_group)


# --------------------------------------------------------------------------- 3
def reference_split_tests(ctx: dict, wgs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    models = MODEL_ORDER + ["TOPSIS_ensemble"]
    y = ctx["y_test"].to_numpy()
    pairs, pvals = [], []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs.append((a, b))
            pvals.append(wilcoxon(np.abs(y - ctx["preds_test"][a]), np.abs(y - ctx["preds_test"][b])).pvalue)
    holm = multipletests(pvals, method="holm")[1]
    bh = multipletests(pvals, method="fdr_bh")[1]
    out["wilcoxon_adjusted_long"] = pd.DataFrame([{"Model A": a, "Model B": b, "p_raw": p, "p_Holm": h, "p_BH": q}
                                                  for (a, b), p, h, q in zip(pairs, pvals, holm, bh)])
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for (a, b), h, p in zip(pairs, holm, pvals):
        mat.loc[a, b] = h  # upper triangle: Holm-adjusted
        mat.loc[b, a] = p  # lower triangle: raw
    out["wilcoxon_matrix_raw_lower_holm_upper"] = mat.reset_index(names="Model")

    # missingness indicators versus structural classes
    x = ctx["x_wgs"]
    single_metal = wgs["M2"].isna().astype(int).to_numpy()
    unterminated = wgs["T1_type"].isna().astype(int).to_numpy()
    rows = []
    for c in [c for c in x.columns if c.endswith("_missing") and x[c].nunique() > 1]:
        v = x[c].to_numpy()
        rows.append({"Indicator": c, "Missing records": int(v.sum()),
                     "Agreement with single-metal class (%)": 100 * float(np.mean(v == single_metal)),
                     "Agreement with unterminated class (%)": 100 * float(np.mean(v == unterminated))})
    out["missingness_concordance"] = pd.DataFrame(rows)

    # ablation: indicators replaced by explicit class flags
    def variant(xx):
        xx = xx.copy()
        xx["is_double_metal"] = (1 - single_metal)
        return xx
    ind_cols = [c for c in x.columns if c.endswith("_missing")]
    settings = {
        "All 80 features (reference)": x,
        "Indicators removed, explicit class flags added": variant(x).drop(columns=ind_cols),
        "Indicators and class flags removed": x.drop(columns=ind_cols + ["has_termination"]),
    }
    rows = []
    tr_idx, te_idx = ctx["x_train"].index, ctx["x_test"].index
    for name, xx in settings.items():
        suite = fit_suite(xx.loc[tr_idx], ctx["y_train"], xx)
        preds, ens = ens_predict(suite, xx.loc[te_idx])
        for m, p in [("ExtraTrees", preds["ExtraTrees"]), ("TOPSIS ensemble", ens)]:
            met = g.metrics(ctx["y_test"], p, ctx["y_range"])
            rows.append({"Feature set": name, "Features": xx.shape[1], "Model": m, **met})
    out["missingness_ablation"] = pd.DataFrame(rows)
    return out


def ablation_split(seed: int, settings: dict, y: pd.Series, wgs: pd.DataFrame) -> list[dict]:
    strat = wgs["Mol"].fillna("None").astype(str)
    rows = []
    for name, xx in settings.items():
        xtr, xte, ytr, yte = train_test_split(xx, y, test_size=0.2, random_state=seed, stratify=strat)
        suite = fit_suite(xtr, ytr, xx)
        preds, ens = ens_predict(suite, xte)
        for m, p in [("ExtraTrees", preds["ExtraTrees"]), ("TOPSIS ensemble", ens)]:
            rows.append({"seed": seed, "Feature set": name, "Features": xx.shape[1], "Model": m,
                         **g.metrics(yte, p, float(ytr.max() - ytr.min()))})
    return rows


def missingness_ablation_splits(x_wgs: pd.DataFrame, y: pd.Series, wgs: pd.DataFrame) -> pd.DataFrame:
    ind_cols = [c for c in x_wgs.columns if c.endswith("_missing")]
    flags = x_wgs.drop(columns=ind_cols).assign(is_double_metal=wgs["M2"].notna().astype(int).to_numpy())
    settings = {"All 80 features": x_wgs, "Indicators removed, explicit class flags added": flags,
                "Indicators and class flags removed": x_wgs.drop(columns=ind_cols + ["has_termination"])}
    out = Parallel(n_jobs=N_JOBS)(delayed(ablation_split)(s, settings, y, wgs) for s in range(N_SPLITS))
    df = pd.DataFrame([r for rows in out for r in rows])
    return df


# --------------------------------------------------------------------------- 4
def activation_subset(wgs: pd.DataFrame, n_perm: int = 20000) -> pd.DataFrame:
    sub = wgs[wgs["E_activation_ev"].notna()].copy()
    rng = np.random.default_rng(g.RANDOM_STATE)
    rows = []
    for ch, s in sub.groupby("Reaction"):
        rho, p = spearmanr(s["E_ads_ev"], s["E_activation_ev"])
        boots = []
        for _ in range(2000):
            ii = rng.integers(0, len(s), len(s))
            if np.ptp(s["E_ads_ev"].to_numpy()[ii]) > 0 and np.ptp(s["E_activation_ev"].to_numpy()[ii]) > 0:
                boots.append(spearmanr(s["E_ads_ev"].to_numpy()[ii], s["E_activation_ev"].to_numpy()[ii])[0])
        rows.append({"Group": ch, "n": len(s), "Spearman_rho": rho, "p": p,
                     "CI95_low": np.percentile(boots, 2.5), "CI95_high": np.percentile(boots, 97.5)})
    rho_all, p_all = spearmanr(sub["E_ads_ev"], sub["E_activation_ev"])
    # stratified Spearman: ranks within channel, pooled
    r1 = sub.groupby("Reaction")["E_ads_ev"].rank(pct=True)
    r2 = sub.groupby("Reaction")["E_activation_ev"].rank(pct=True)
    strat = float(np.corrcoef(r1 - r1.groupby(sub["Reaction"]).transform("mean"), r2 - r2.groupby(sub["Reaction"]).transform("mean"))[0, 1])
    # null: shuffle Ea within channel (keeps channel structure, removes within-channel association)
    ea = sub["E_activation_ev"].to_numpy()
    ch = sub["Reaction"].to_numpy()
    null = []
    for _ in range(n_perm):
        perm = ea.copy()
        for c in np.unique(ch):
            m = np.where(ch == c)[0]
            perm[m] = ea[rng.permutation(m)]
        null.append(spearmanr(sub["E_ads_ev"], perm)[0])
    null = np.asarray(null)
    rows.append({"Group": "All 62 records (pooled)", "n": len(sub), "Spearman_rho": rho_all, "p": p_all,
                 "CI95_low": np.nan, "CI95_high": np.nan})
    rows.append({"Group": "Within-channel (stratified) rank correlation", "n": len(sub), "Spearman_rho": strat, "p": np.nan,
                 "CI95_low": np.nan, "CI95_high": np.nan})
    rows.append({"Group": "Channel-preserving null of pooled rho (mean, 2.5-97.5%)", "n": n_perm,
                 "Spearman_rho": float(null.mean()), "p": float((np.sum(null >= rho_all) + 1) / (n_perm + 1)),
                 "CI95_low": float(np.percentile(null, 2.5)), "CI95_high": float(np.percentile(null, 97.5))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 5
def dopant_domain(ctx_suite: dict, ctx: dict, wgs: pd.DataFrame, dop: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    rows = []
    for c in g.NUMERIC_BASE:
        a, b = wgs[c].dropna(), dop[c].dropna()
        smd = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2) if len(b) > 1 else np.nan
        ks = ks_2samp(a, b) if len(b) >= 5 else None
        rows.append({"Descriptor": c, "WGS available": len(a), "Doped available": len(b),
                     "WGS median": a.median(), "WGS IQR low": a.quantile(.25), "WGS IQR high": a.quantile(.75),
                     "Doped median": b.median() if len(b) else np.nan,
                     "Doped IQR low": b.quantile(.25) if len(b) else np.nan, "Doped IQR high": b.quantile(.75) if len(b) else np.nan,
                     "SMD": smd, "KS_D": ks.statistic if ks else np.nan, "KS_p": ks.pvalue if ks else np.nan,
                     "Doped range inside WGS range (%)": 100 * float(np.mean((b >= a.min()) & (b <= a.max()))) if len(b) else np.nan})
    out["descriptor_shift_numeric"] = pd.DataFrame(rows)

    def tvd(s1, s2):
        p1, p2 = s1.value_counts(normalize=True), s2.value_counts(normalize=True)
        idx = p1.index.union(p2.index)
        return 0.5 * float(np.abs(p1.reindex(idx, fill_value=0) - p2.reindex(idx, fill_value=0)).sum())
    cat_rows = []
    for c, f in [("M1", lambda d: d["M1"].astype(str)), ("T1_type", lambda d: d["T1_type"].fillna("bare").astype(str)),
                 ("Mol", lambda d: d["Mol"].astype(str)), ("N_Layers", lambda d: d["N_Layers"].astype(int).astype(str)),
                 ("X", lambda d: d["X"].astype(str))]:
        s1, s2 = f(wgs), f(dop)
        top2 = s2.value_counts(normalize=True).head(3)
        cat_rows.append({"Descriptor": c, "TVD": tvd(s1, s2),
                         "Doped top categories": "; ".join(f"{k} {100*v:.0f}%" for k, v in top2.items()),
                         "WGS share of those categories": "; ".join(f"{k} {100*s1.eq(k).mean():.0f}%" for k in top2.index)})
    out["descriptor_shift_categorical"] = pd.DataFrame(cat_rows)

    # applicability-domain flags on the reference split
    prep = clone(ctx_suite["fitted"]["ExtraTrees"].named_steps["prep"]).fit(ctx["x_train"])
    ztr, zte, zdo = prep.transform(ctx["x_train"]), prep.transform(ctx["x_test"]), prep.transform(ctx["x_dop"])
    nn = NearestNeighbors(n_neighbors=6).fit(ztr)
    d_tr = nn.kneighbors(ztr)[0][:, 1:].mean(1)
    d_te = nn.kneighbors(zte, n_neighbors=5)[0].mean(1)
    d_do = nn.kneighbors(zdo, n_neighbors=5)[0].mean(1)
    thr = np.percentile(d_tr, 95)
    preds_te, ens_te = ens_predict(ctx_suite, ctx["x_test"])
    preds_do, ens_do = ens_predict(ctx_suite, ctx["x_dop"])
    sd_te = np.std(np.column_stack(list(preds_te.values())), axis=1)
    sd_do = np.std(np.column_stack(list(preds_do.values())), axis=1)
    oof_ens = ctx_suite["oof"][ctx_suite["w"].index].to_numpy() @ ctx_suite["w"].to_numpy()
    resid = pd.Series(ctx["y_train"].to_numpy() - oof_ens, index=ctx["x_train"].index)
    ads_tr = wgs.loc[ctx["x_train"].index, "Mol"].astype(str)
    hw_te = mondrian_halfwidth(resid, ads_tr, wgs.loc[ctx["x_test"].index, "Mol"].astype(str))
    hw_do = mondrian_halfwidth(resid, ads_tr, dop["Mol"].astype(str))
    desc_missing = dop["E_Form"].isna().to_numpy()
    y_te, y_do = ctx["y_test"].to_numpy(), dop["E_ads_ev"].to_numpy()
    lab = np.r_[np.zeros(len(d_te)), np.ones(len(d_do))]
    rows = []
    for name, m in [("WGS hold-out (n = 120)", None), ("Doped, all (n = 94)", np.ones(len(dop), bool)),
                    ("Doped, substrate descriptors available (n = 22)", ~desc_missing),
                    ("Doped, substrate descriptors missing (n = 72)", desc_missing)]:
        if m is None:
            d, sd, err, hw, flag_schema, flag_cov = d_te, sd_te, np.abs(y_te - ens_te), hw_te, np.zeros(len(d_te), bool), np.zeros(len(d_te), bool)
        else:
            d, sd, err, hw = d_do[m], sd_do[m], np.abs(y_do - ens_do)[m], hw_do[m]
            flag_schema = np.ones(m.sum(), bool)  # dopant field populated but absent from the trained representation
            flag_cov = desc_missing[m]
        rows.append({"Records": name, "Schema flag: unrepresented dopant (%)": 100 * flag_schema.mean(),
                     "Coverage flag: substrate descriptors missing (%)": 100 * flag_cov.mean(),
                     "kNN distance median": float(np.median(d)), "kNN flag > train P95 (%)": 100 * float(np.mean(d > thr)),
                     "Base-model SD median (eV)": float(np.median(sd)),
                     "90% interval coverage": float(np.mean(err <= hw)), "MAE (eV)": float(err.mean()),
                     "Any flag (%)": 100 * float(np.mean(flag_schema | flag_cov | (d > thr)))})
    out["ad_flags"] = pd.DataFrame(rows)
    out["ad_auc"] = pd.DataFrame([
        {"Score": "kNN distance (5 nearest training records)", "AUROC doped vs WGS hold-out": roc_auc_score(lab, np.r_[d_te, d_do]),
         "Spearman with |error| (doped)": spearmanr(d_do, np.abs(y_do - ens_do))[0]},
        {"Score": "Base-model disagreement (SD of six predictions)", "AUROC doped vs WGS hold-out": roc_auc_score(lab, np.r_[sd_te, sd_do]),
         "Spearman with |error| (doped)": spearmanr(sd_do, np.abs(y_do - ens_do))[0]},
        {"Score": "kNN distance, descriptor-complete doped records only", "AUROC doped vs WGS hold-out":
            roc_auc_score(np.r_[np.zeros(len(d_te)), np.ones((~desc_missing).sum())], np.r_[d_te, d_do[~desc_missing]]),
         "Spearman with |error| (doped)": spearmanr(d_do[~desc_missing], np.abs(y_do - ens_do)[~desc_missing])[0]},
    ])
    out["_knn_threshold"] = pd.DataFrame([{"train_P95": thr}])

    # descriptor back-filling from the undoped host and the undoped-host DFT value
    d = dop.copy()
    d["host"] = d["Formula"].replace({"FeMo2C": "Mo2C", "CuMo2C": "Mo2C"})
    host_desc = wgs.groupby("Formula")[SUBSTRATE_NUM + ["N_Layers", "Length_MT1", "Bader_M2", "Bader_T1"]].median()
    host_val = wgs.groupby(["Formula", "Mol"])["E_ads_ev"].mean()
    filled = d.copy()
    has_host = d["host"].isin(host_desc.index)
    for c in host_desc.columns:
        filled.loc[has_host, c] = d.loc[has_host, "host"].map(host_desc[c])
    x_fill = g.make_features(filled)
    for c in ctx["x_wgs"].columns.difference(x_fill.columns):
        x_fill[c] = np.nan
    x_fill = x_fill[ctx["x_wgs"].columns]
    _, ens_fill = ens_predict(ctx_suite, x_fill)
    hv = np.array([host_val.get((h, m), np.nan) for h, m in zip(d["host"], d["Mol"])])
    rows = []
    for name, p, m in [("Direct transfer (as in Section 3.8)", ens_do, np.ones(len(d), bool)),
                       ("Direct transfer, records with an undoped host value", ens_do, np.isfinite(hv)),
                       ("Substrate descriptors back-filled from undoped host", ens_fill, np.isfinite(hv) & has_host.to_numpy()),
                       ("Undoped-host DFT value as prediction", hv, np.isfinite(hv))]:
        yy, pp = y_do[m], p[m]
        rows.append({"Setting": name, "n": int(m.sum()), "R2": float(1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)),
                     "RMSE": g.rmse(yy, pp), "MAE": float(np.mean(np.abs(yy - pp))),
                     "Mean signed error": float(np.mean(pp - yy)), "Spearman": float(spearmanr(yy, pp)[0]),
                     "Distinct predictions": int(pd.Series(np.round(pp, 6)).nunique())})
    out["backfill"] = pd.DataFrame(rows)
    shift = pd.DataFrame({"host": d["host"], "Mol": d["Mol"], "doped": y_do, "undoped": hv}).dropna()
    shift["shift"] = shift["doped"] - shift["undoped"]
    out["doping_shift"] = shift.groupby(["host", "Mol"]).agg(n=("shift", "size"), undoped=("undoped", "first"),
                                                             doped_mean=("doped", "mean"), doped_min=("doped", "min"),
                                                             doped_max=("doped", "max"), shift_mean=("shift", "mean"),
                                                             abs_shift_max=("shift", lambda s: s.abs().max())).reset_index()
    return out


# --------------------------------------------------------------------------- 6
def few_shot_task(design: str, k: int, draw: int, tr_idx: np.ndarray, te_idx: np.ndarray,
                  x_wgs: pd.DataFrame, y_wgs: pd.Series, x_dop: pd.DataFrame, y_dop: pd.Series,
                  combo: np.ndarray, ads: np.ndarray, x_all: pd.DataFrame) -> list[dict]:
    xte, yte = x_dop.iloc[te_idx], y_dop.iloc[te_idx].to_numpy()
    combo_te = combo[te_idx]
    rows = []

    def record(setting, p):
        met = g.metrics(yte, p, float(np.ptp(y_dop)))
        rho = spearmanr(yte, p)[0] if np.ptp(p) > 0 else 0.0
        rho_w, n_g = within_group_rho(yte, np.asarray(p, float), combo_te, 3)
        rows.append({"Design": design, "k": k, "draw": draw, "Setting": setting, "n_test": len(te_idx),
                     "seen_combination_share": float(np.mean(np.isin(combo_te, combo[tr_idx]))) if len(tr_idx) else 0.0,
                     "R2": met["R2"], "RMSE": met["RMSE"], "MAE": met["MAE"], "Spearman_rho": rho,
                     "Within_combination_rho": rho_w, "Combinations_scored": n_g})

    xtr = pd.concat([x_wgs, x_dop.iloc[tr_idx]], ignore_index=True)
    ytr = pd.concat([y_wgs, y_dop.iloc[tr_idx]], ignore_index=True)
    suite = fit_suite(xtr, ytr, x_all)
    preds, ens = ens_predict(suite, xte)
    record("Ensemble, WGS + k dopant", ens)
    record("ExtraTrees, WGS + k dopant", preds["ExtraTrees"])
    if k >= 10:
        xs, ys = x_dop.iloc[tr_idx].reset_index(drop=True), y_dop.iloc[tr_idx].reset_index(drop=True)
        suite_d = fit_suite(xs, ys, x_all)
        preds_d, ens_d = ens_predict(suite_d, xte)
        record("Ensemble, k dopant only", ens_d)
        record("ExtraTrees, k dopant only", preds_d["ExtraTrees"])
    if k >= 5:
        ytr_d = y_dop.iloc[tr_idx]
        overall = np.full(len(te_idx), ytr_d.mean())
        am = group_mean_baseline(ytr_d, pd.Series(ads[tr_idx], index=ytr_d.index), pd.Series(ads[te_idx]), overall)
        gm = group_mean_baseline(ytr_d, pd.Series(combo[tr_idx], index=ytr_d.index), pd.Series(combo_te), am)
        record("Host-adsorbate group mean of k dopant records", gm)
    return rows


def few_shot(wgs: pd.DataFrame, dop: pd.DataFrame) -> pd.DataFrame:
    x_wgs = g.make_features(wgs)
    x_dop = g.make_features(dop)
    for col in x_wgs.columns.difference(x_dop.columns):
        x_dop[col] = np.nan
    for col in x_dop.columns.difference(x_wgs.columns):
        x_wgs[col] = np.nan
    x_dop = x_dop[x_wgs.columns].reset_index(drop=True)
    x_all = pd.concat([x_wgs, x_dop])
    y_wgs = wgs["E_ads_ev"].astype(float).reset_index(drop=True)
    x_wgs = x_wgs.reset_index(drop=True)
    y_dop = dop["E_ads_ev"].astype(float).reset_index(drop=True)
    combo = (dop["Formula"].astype(str) + "|" + dop["Mol"].astype(str)).to_numpy()
    ads = dop["Mol"].astype(str).to_numpy()
    dopant = dop["Dopant"].replace(g.ALIASES).astype(str).to_numpy()
    n = len(dop)
    tasks = [("Random", 0, 0, np.array([], int), np.arange(n))]
    for k in [5, 10, 20, 30, 50]:
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(1000 + draw)
            idx = rng.permutation(n)
            tasks.append(("Random", k, draw, idx[:k], idx[k:]))
    # dopant-disjoint: test dopant elements never appear in training
    elements = np.unique(dopant)
    for k in [10, 20, 30, 50]:
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(2000 + draw)
            test_el, cnt = [], 0
            for el in rng.permutation(elements):
                if cnt >= 30:
                    break
                test_el.append(el)
                cnt += int(np.sum(dopant == el))
            te = np.where(np.isin(dopant, test_el))[0]
            pool = np.where(~np.isin(dopant, test_el))[0]
            if len(pool) < k:
                continue
            tasks.append(("Dopant-disjoint", k, draw, rng.choice(pool, size=k, replace=False), te))
    # leave one dopant element out, all other dopant records available
    for j, el in enumerate(elements):
        te = np.where(dopant == el)[0]
        tasks.append(("Leave-one-dopant-out", int(n - len(te)), j, np.where(dopant != el)[0], te))
    # leave one host-adsorbate combination out
    for j, c in enumerate(pd.unique(combo)):
        te = np.where(combo == c)[0]
        tasks.append(("Leave-one-combination-out", int(n - len(te)), j, np.where(combo != c)[0], te))
    if QUICK:
        tasks = tasks[::25]
    outs = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(few_shot_task)(d, k, dr, tr, te, x_wgs, y_wgs, x_dop, y_dop, combo, ads, x_all) for d, k, dr, tr, te in tasks)
    df = pd.DataFrame([r for rows in outs for r in rows])
    # pooled leave-one-out predictions need record-level outputs; recompute pooled within-combination rho
    return df


def few_shot_pooled_lodo(wgs: pd.DataFrame, dop: pd.DataFrame) -> pd.DataFrame:
    """Record-level leave-one-dopant-out predictions, pooled to score dopant ordering within each
    host-adsorbate combination (each record predicted by a model that never saw its dopant element)."""
    x_wgs = g.make_features(wgs)
    x_dop = g.make_features(dop)
    for col in x_wgs.columns.difference(x_dop.columns):
        x_dop[col] = np.nan
    for col in x_dop.columns.difference(x_wgs.columns):
        x_wgs[col] = np.nan
    x_dop = x_dop[x_wgs.columns].reset_index(drop=True)
    x_all = pd.concat([x_wgs, x_dop])
    x_wgs = x_wgs.reset_index(drop=True)
    y_wgs = wgs["E_ads_ev"].astype(float).reset_index(drop=True)
    y_dop = dop["E_ads_ev"].astype(float).reset_index(drop=True)
    combo = (dop["Formula"].astype(str) + "|" + dop["Mol"].astype(str)).to_numpy()
    dopant = dop["Dopant"].replace(g.ALIASES).astype(str).to_numpy()

    def one(el):
        te = np.where(dopant == el)[0]
        tr = np.where(dopant != el)[0]
        xs, ys = x_dop.iloc[tr].reset_index(drop=True), y_dop.iloc[tr].reset_index(drop=True)
        et = g.make_models(g.build_preprocessor(x_all))["ExtraTrees"].fit(xs, ys)
        suite_p = fit_suite(pd.concat([x_wgs, xs], ignore_index=True), pd.concat([y_wgs, ys], ignore_index=True), x_all)
        preds_p, ens_p = ens_predict(suite_p, x_dop.iloc[te])
        gm = pd.Series(ys.to_numpy()).groupby(combo[tr]).mean()
        gmp = np.array([gm.get(c, ys.mean()) for c in combo[te]])
        return pd.DataFrame({"idx": te, "et_dop": et.predict(x_dop.iloc[te]), "ens_pool": ens_p,
                             "et_pool": preds_p["ExtraTrees"], "group_mean": gmp})
    els = np.unique(dopant)[:3] if QUICK else np.unique(dopant)
    parts = Parallel(n_jobs=N_JOBS)(delayed(one)(el) for el in els)
    p = pd.concat(parts).sort_values("idx").set_index("idx").reindex(range(len(dop)))
    keep = p.notna().all(axis=1).to_numpy()
    p, y_dop, combo = p[keep].reset_index(drop=True), y_dop[keep].reset_index(drop=True), combo[keep]
    rows = []
    for c in ["et_dop", "et_pool", "ens_pool", "group_mean"]:
        for cb in list(pd.unique(combo)) + ["All combinations (n-weighted)"]:
            if cb.startswith("All"):
                rho, ng = within_group_rho(y_dop.to_numpy(), p[c].to_numpy(), combo, 3)
                nn_ = len(combo)
            else:
                m = combo == cb
                if m.sum() < 3:
                    continue
                rho = spearmanr(y_dop[m], p.loc[m, c])[0] if np.ptp(p.loc[m, c]) > 0 else 0.0
                nn_ = int(m.sum())
            rows.append({"Predictor": c, "Combination": cb, "n": nn_, "Within_combination_rho": rho,
                         "RMSE": g.rmse(y_dop[combo == cb], p.loc[combo == cb, c]) if not cb.startswith("All") else g.rmse(y_dop, p[c])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 7
def software_and_configs() -> tuple[pd.DataFrame, pd.DataFrame]:
    import matplotlib, scipy, sklearn, seaborn, statsmodels, networkx, joblib, openpyxl
    vers = pd.DataFrame([{"Package": "Python", "Version": platform.python_version()}] + [
        {"Package": m.__name__, "Version": m.__version__} for m in
        [np, pd, sklearn, scipy, statsmodels, matplotlib, seaborn, networkx, joblib, openpyxl]])
    rows = []
    for name, pipe in g.make_models(g.build_preprocessor(g.make_features(g.load_data()[0]))).items():
        est = pipe.named_steps["model"]
        params = {k: v for k, v in est.get_params().items() if k in {
            "n_estimators", "max_depth", "max_features", "min_samples_leaf", "min_samples_split", "learning_rate",
            "subsample", "max_iter", "l2_regularization", "early_stopping", "validation_fraction", "C", "gamma",
            "epsilon", "kernel", "hidden_layer_sizes", "alpha", "learning_rate_init", "n_iter_no_change",
            "activation", "solver", "random_state", "bootstrap"}}
        rows.append({"Model": name, "Estimator": type(est).__name__, "Settings": json.dumps(params, default=str)})
    return vers, pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    wgs, dop = g.load_data()
    x_wgs = g.make_features(wgs)
    x_wgs = x_wgs.drop(columns=[c for c in x_wgs.columns if x_wgs[c].isna().all()])
    y = wgs["E_ads_ev"].astype(float)
    print("features", x_wgs.shape)

    vers, cfg = software_and_configs()
    save(vers, "S_software_versions")
    save(cfg, "S_model_configurations")

    ctx = g.prepare_training(wgs, dop)
    ctx_suite = fit_suite(ctx["x_train"], ctx["y_train"], ctx["x_wgs"])
    assert np.allclose(ctx_suite["w"].reindex(ctx["weights"].index).to_numpy(), ctx["weights"].to_numpy())

    for name, df in reference_split_tests(ctx, wgs).items():
        save(df, f"ref_{name}")
    print("reference tests", round(time.time() - t0))

    save(activation_subset(wgs), "activation_within_channel")

    for name, df in dopant_domain(ctx_suite, ctx, wgs, dop).items():
        save(df, f"dopant_{name.lstrip('_')}")
    print("dopant domain", round(time.time() - t0))

    abl = missingness_ablation_splits(x_wgs, y, wgs)
    save(abl, "splits_missingness_ablation_raw")
    save(abl.groupby(["Feature set", "Model"]).agg(Features=("Features", "first"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                                                   RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std")).reset_index(),
         "splits_missingness_ablation")
    print("ablation", round(time.time() - t0))

    res = Parallel(n_jobs=N_JOBS)(delayed(run_split)(s, x_wgs, y, wgs) for s in range(N_SPLITS))
    for name, df in summarise_splits(res).items():
        save(df, f"splits_{name}")
    print("splits", round(time.time() - t0))

    gcv, gcv_groups = grouped_cv(x_wgs, y, wgs)
    save(gcv, "grouped_cv")
    save(gcv_groups, "grouped_cv_by_group")
    print("grouped", round(time.time() - t0))

    fs = few_shot(wgs, dop)
    save(fs, "fewshot_raw")
    summ = fs.groupby(["Design", "Setting", "k"]).agg(
        draws=("draw", "count"), n_test=("n_test", "mean"), seen_combination_share=("seen_combination_share", "mean"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"), RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), rho_mean=("Spearman_rho", "mean"), rho_sd=("Spearman_rho", "std"),
        rho_within_mean=("Within_combination_rho", "mean"), rho_within_sd=("Within_combination_rho", "std"),
    ).reset_index()
    save(summ, "fewshot_summary")
    save(few_shot_pooled_lodo(wgs, dop), "fewshot_lodo_pooled_within_combination")
    print("few-shot", round(time.time() - t0))
    print("DONE")


if __name__ == "__main__":
    main()
