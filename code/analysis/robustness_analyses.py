"""Robustness and domain-shift analyses reported in the Supporting Information
(Tables S5, S7a, S8, S9 and Fig. S1).

1. Repeated stratified hold-out splits (split-to-split variability of all models
   and of the CV-weighted TOPSIS ensemble).
2. Subset-resolved direct-transfer metrics (descriptor coverage of both
   domains is reported in Table S2 by generate_paper_tables.py).
3. Few-shot dopant augmentation repeated over random draws, with a
   dopant-only ExtraTrees reference.
4. Sensitivity of the dopant ranking to the adsorption-window parameters.

All model settings, preprocessing and TOPSIS weighting are imported from
generate_all_figures.py so that the SI analyses use the same pipeline as the
main-text figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import generate_all_figures as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" / "FigS"
TAB.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

N_SPLITS = 20
N_DRAWS = 20
K_VALUES = [0, 5, 10, 20, 30, 50]


def aligned_features(wgs: pd.DataFrame, dop: pd.DataFrame, keep_dopant: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_wgs = g.make_features(wgs)
    x_dop = g.make_features(dop)
    for col in x_wgs.columns.difference(x_dop.columns):
        x_dop[col] = np.nan
    for col in x_dop.columns.difference(x_wgs.columns):
        x_wgs[col] = np.nan
    if not keep_dopant:
        empty = [c for c in x_wgs.columns if x_wgs[c].isna().all()]
        x_wgs = x_wgs.drop(columns=empty)
    x_dop = x_dop[x_wgs.columns]
    return x_wgs, x_dop


def fit_ensemble(x_train: pd.DataFrame, y_train: pd.Series, x_ref: pd.DataFrame):
    models = g.make_models(g.build_preprocessor(x_ref))
    cvm = g.cv_metric_table(models, x_train, y_train)
    top = g.entropy_topsis(cvm[g.METRIC_COLS])
    weights = top["ensemble_weight"].reindex(cvm.index)
    fitted = {n: clone(p).fit(x_train, y_train) for n, p in models.items()}
    return fitted, weights


def predict_ensemble(fitted: dict, weights: pd.Series, x: pd.DataFrame) -> tuple[dict, np.ndarray]:
    preds = {n: m.predict(x) for n, m in fitted.items()}
    ens = np.column_stack([preds[n] for n in weights.index]) @ weights.to_numpy()
    return preds, ens


# --------------------------------------------------------------------------- 1
def repeated_splits(wgs: pd.DataFrame, dop: pd.DataFrame) -> pd.DataFrame:
    x_wgs, _ = aligned_features(wgs, dop, keep_dopant=False)
    y = wgs["E_ads_ev"].astype(float)
    strat = wgs["Mol"].fillna("None").astype(str)
    rows = []
    for seed in range(N_SPLITS):
        xtr, xte, ytr, yte = train_test_split(x_wgs, y, test_size=0.2, random_state=seed, stratify=strat)
        fitted, w = fit_ensemble(xtr, ytr, x_wgs)
        preds, ens = predict_ensemble(fitted, w, xte)
        preds["TOPSIS_ensemble"] = ens
        y_range = float(ytr.max() - ytr.min())
        for name, p in preds.items():
            rows.append({"split_seed": seed, "Model": name, **g.metrics(yte, p, y_range)})
        for name, val in w.items():
            rows.append({"split_seed": seed, "Model": f"weight_{name}", "R2": val})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2
def coverage_and_transfer(wgs: pd.DataFrame, dop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = g.NUMERIC_BASE + g.CATEGORICAL_BASE + ["Dopant"]
    cov = pd.DataFrame(
        {
            "WGS_missing_pct": wgs.reindex(columns=cols).isna().mean() * 100,
            "Dopants_missing_pct": dop.reindex(columns=cols).isna().mean() * 100,
        }
    ).round(1)
    cov.index.name = "Descriptor"

    ctx = g.prepare_training(wgs, dop)
    d = dop.copy()
    d["pred"] = ctx["preds_dop"]["TOPSIS_ensemble"]
    d["substrate_descriptors"] = np.where(d["E_Form"].notna(), "available", "missing")
    rows = []
    for label, sub in [("All dopant records", d)] + [(f"Substrate descriptors {k}", s) for k, s in d.groupby("substrate_descriptors")]:
        m = g.metrics(sub["E_ads_ev"], sub["pred"], ctx["y_range"])
        rho = spearmanr(sub["E_ads_ev"], sub["pred"])[0] if len(sub) > 2 else np.nan
        rows.append(
            {
                "Subset": label,
                "n": len(sub),
                "Distinct predictions": int(sub["pred"].round(6).nunique()),
                "Distinct dopants": int(sub["Dopant"].nunique()),
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Spearman_rho": rho,
                "Mean signed error (pred - DFT)": float((sub["pred"] - sub["E_ads_ev"]).mean()),
            }
        )
    return cov, pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3
def few_shot(wgs: pd.DataFrame, dop: pd.DataFrame) -> pd.DataFrame:
    x_wgs, x_dop = aligned_features(wgs, dop, keep_dopant=True)
    x_all = pd.concat([x_wgs, x_dop])
    # Dopant elemental descriptors are undefined for WGS rows; keep them so that
    # the augmented models can use them once dopant samples enter training.
    y_wgs = wgs["E_ads_ev"].astype(float)
    y_dop = dop["E_ads_ev"].astype(float)
    x_dop = x_dop.reset_index(drop=True)
    y_dop = y_dop.reset_index(drop=True)
    rows = []
    for k in K_VALUES:
        draws = [0] if k == 0 else range(N_DRAWS)
        for draw in draws:
            rng = np.random.default_rng(1000 + draw)
            idx = rng.permutation(len(x_dop))
            tr_idx, te_idx = idx[:k], idx[k:]
            xtr = pd.concat([x_wgs, x_dop.iloc[tr_idx]], ignore_index=True)
            ytr = pd.concat([y_wgs, y_dop.iloc[tr_idx]], ignore_index=True)
            xte, yte = x_dop.iloc[te_idx], y_dop.iloc[te_idx]
            fitted, w = fit_ensemble(xtr, ytr, x_all)
            _, ens = predict_ensemble(fitted, w, xte)
            y_range = float(ytr.max() - ytr.min())
            rows.append({"k": k, "draw": draw, "Setting": "WGS + k dopant (ensemble)", "n_test": len(te_idx),
                         **g.metrics(yte, ens, y_range), "Spearman_rho": spearmanr(yte, ens)[0]})
            if k >= 10:
                et = g.make_models(g.build_preprocessor(x_all))["ExtraTrees"]
                et.fit(x_dop.iloc[tr_idx], y_dop.iloc[tr_idx])
                p = et.predict(xte)
                rows.append({"k": k, "draw": draw, "Setting": "k dopant only (ExtraTrees)", "n_test": len(te_idx),
                             **g.metrics(yte, p, y_range), "Spearman_rho": spearmanr(yte, p)[0]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4
def window_sensitivity(dop: pd.DataFrame) -> pd.DataFrame:
    d = dop.copy()
    d["Dopant_clean"] = d["Dopant"].replace(g.ALIASES)
    counts = d["Dopant_clean"].value_counts()
    keep = counts[counts >= 3].index

    def ranking(center: float, width: float) -> pd.Series:
        s = np.exp(-((d["E_ads_ev"] - center) / width) ** 2)
        return s.groupby(d["Dopant_clean"]).mean().loc[keep]

    ref = ranking(-0.80, 0.90)
    ref_top = set(ref.sort_values(ascending=False).head(3).index)
    rows = []
    for c in [-0.6, -0.8, -1.0]:
        for wd in [0.6, 0.9, 1.2]:
            r = ranking(c, wd)
            top = r.sort_values(ascending=False).head(3).index
            rows.append({"Window centre (eV)": c, "Window width (eV)": wd,
                         "Spearman rho vs reference": spearmanr(ref, r)[0],
                         "Top-3 dopants": ", ".join(top),
                         "Top-3 overlap with reference": len(ref_top & set(top))})
    return pd.DataFrame(rows)


def summarise_splits(df: pd.DataFrame) -> pd.DataFrame:
    perf = df[~df["Model"].str.startswith("weight_")]
    summ = perf.groupby("Model")[g.METRIC_COLS].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    piv = perf.pivot(index="split_seed", columns="Model", values="RMSE")
    summ["Rank-1 RMSE frequency"] = piv.idxmin(axis=1).value_counts().reindex(summ.index).fillna(0).astype(int)
    summ["Ensemble RMSE <= model RMSE (splits)"] = [(piv["TOPSIS_ensemble"] <= piv[m]).sum() for m in summ.index]
    order = ["RF", "ExtraTrees", "GBR", "HGBR", "SVR", "MLP", "TOPSIS_ensemble"]
    return summ.loc[order]


def summarise_few_shot(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby(["Setting", "k"]).agg(
        n_test=("n_test", "first"),
        draws=("draw", "count"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        rho_mean=("Spearman_rho", "mean"), rho_sd=("Spearman_rho", "std"),
    )
    return s.reset_index()


def plot_few_shot(s: pd.DataFrame) -> None:
    g.configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    styles = {"WGS + k dopant (ensemble)": (g.PALETTE["teal"], "o"), "k dopant only (ExtraTrees)": (g.PALETTE["orange"], "s")}
    for setting, sub in s.groupby("Setting"):
        color, marker = styles[setting]
        for ax, (m, lab) in zip(axes, [("R2", r"R$^2$"), ("RMSE", "RMSE (eV)"), ("rho", r"Spearman $\rho$")]):
            ax.errorbar(sub["k"], sub[f"{m}_mean"], yerr=sub[f"{m}_sd"].fillna(0), color=color, marker=marker,
                        capsize=3, lw=1.6, ms=5, label=setting)
            ax.set_xlabel("Dopant samples added to training, k")
            ax.set_ylabel(lab)
    axes[0].axhline(0, color=g.PALETTE["muted"], lw=0.8, ls="--")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    for i, ax in enumerate(axes):
        ax.text(-0.18, 1.04, f"({'abc'[i]})", transform=ax.transAxes, fontsize=14)
        g.style_axes(ax, tick_size=11, label_size=12)
    fig.tight_layout()
    fig.savefig(FIG / "FigS1_few_shot_dopant_augmentation.png", dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "FigS1_few_shot_dopant_augmentation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    wgs, dop = g.load_data()

    splits = repeated_splits(wgs, dop)
    splits.to_csv(TAB / "TableS5_Repeated_Splits_raw.csv", index=False, encoding="utf-8-sig")
    summarise_splits(splits).round(4).to_csv(TAB / "TableS5_Repeated_Splits.csv", encoding="utf-8-sig")

    _, transfer = coverage_and_transfer(wgs, dop)
    transfer.round(4).to_csv(TAB / "TableS7a_Transfer_by_Subset.csv", index=False, encoding="utf-8-sig")

    fs = few_shot(wgs, dop)
    fs.to_csv(TAB / "TableS9_Few_Shot_raw.csv", index=False, encoding="utf-8-sig")
    fs_sum = summarise_few_shot(fs)
    fs_sum.round(4).to_csv(TAB / "TableS9_Few_Shot.csv", index=False, encoding="utf-8-sig")
    plot_few_shot(fs_sum)

    window_sensitivity(dop).round(4).to_csv(TAB / "TableS8_Window_Sensitivity.csv", index=False, encoding="utf-8-sig")
    print("DONE")


if __name__ == "__main__":
    main()
