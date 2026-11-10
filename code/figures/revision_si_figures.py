"""Supporting Information figures for the MTCOMM-D-26-07228 revision.

Fig. S1  Validation beyond random splits: grouped cross-validation, adsorbate-resolved
         accuracy against group-mean baselines, pairwise ranking accuracy and
         90% prediction intervals.
Fig. S2  Doped-domain shift and applicability-domain signals.
Fig. S3  Few-shot dopant analysis with identical training/test records.

Reads the CSV files written by revision_analyses.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_all_figures as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "outputs" / "tables" / "revision"
FIG = ROOT / "outputs" / "figures" / "FigS"
FIG.mkdir(parents=True, exist_ok=True)
P = g.PALETTE
ADS_ORDER = ["H", "H2O", "H2", "CO", "CO2", "OH", "O", "O2"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_s1() -> None:
    gcv = pd.read_csv(TAB / "grouped_cv.csv")
    ads = pd.read_csv(TAB / "splits_adsorbate_resolved.csv").set_index("Adsorbate")
    pr = pd.read_csv(TAB / "splits_pairwise_ranking.csv")
    fig = plt.figure(figsize=(14.5, 10.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    schemes = list(dict.fromkeys(gcv["Scheme"]))
    short = {"Random 5-fold CV": "Random 5-fold", "Leave-host-out (5 folds)": "Leave host out",
             "Leave-parent-MXene-out (5 folds)": "Leave parent MXene out", "Leave-one-primary-metal-out": "Leave one M1 out",
             "Leave-one-termination-out": "Leave one termination out", "Leave-one-data-source-out": "Leave one source out",
             "Leave-one-adsorbate-out": "Leave one adsorbate out"}
    preds = [("TOPSIS ensemble", P["gold"], "Ensemble"), ("ExtraTrees", P["teal"], "ExtraTrees"),
             ("Adsorbate x termination mean baseline", P["purple"], "Adsorbate × termination mean"),
             ("Adsorbate-mean baseline", P["muted"], "Adsorbate mean")]
    y = np.arange(len(schemes))[::-1]
    h = 0.2
    for j, (pname, col, lab) in enumerate(preds):
        vals = [gcv[(gcv.Scheme == s) & (gcv.Predictor == pname)]["RMSE"].iloc[0] for s in schemes]
        ax[0].barh(y + (1.5 - j) * h, vals, height=h, color=col, label=lab, edgecolor="white", linewidth=0.4)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([short[s] for s in schemes])
    ax[0].set_xlabel("Pooled out-of-fold RMSE (eV)")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=9.5, frameon=False)
    g.style_axes(ax[0], tick_size=10.5, label_size=12)

    a_order = [a for a in ADS_ORDER if a in ads.index]
    x = np.arange(len(a_order))
    w = 0.27
    ax[1].bar(x - w, ads.loc[a_order, "RMSE_ensemble"], w, color=P["gold"], label="Ensemble")
    ax[1].bar(x, ads.loc[a_order, "RMSE_ads_x_term_mean"], w, color=P["purple"], label="Adsorbate × termination mean")
    ax[1].bar(x + w, ads.loc[a_order, "RMSE_adsorbate_mean"], w, color=P["muted"], label="Adsorbate mean")
    for xi, a in enumerate(a_order):
        ax[1].text(xi, max(ads.loc[a, ["RMSE_ensemble", "RMSE_ads_x_term_mean", "RMSE_adsorbate_mean"]]) + 0.12,
                   f"$\\rho$ = {ads.loc[a, 'Spearman_rho_per_split_mean']:.2f}", ha="center", fontsize=8.6, rotation=90, va="bottom")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([g.ADSORBATE_LABELS[a] for a in a_order])
    ax[1].set_ylabel("Hold-out RMSE (eV), 20 splits")
    ax[1].set_ylim(0, float(ads.loc[a_order, "RMSE_adsorbate_mean"].max()) * 1.45)
    ax[1].legend(loc="upper left", fontsize=9)
    g.style_axes(ax[1], tick_size=11, label_size=12)

    xb = np.arange(len(pr))
    ax[2].plot(xb, pr["Ensemble_correct_order"], "-o", color=P["gold"], lw=2, label="Ensemble")
    ax[2].plot(xb, pr["ExtraTrees_correct_order"], "-s", color=P["teal"], lw=1.6, label="ExtraTrees")
    ax[2].axhline(0.5, color=P["muted"], ls="--", lw=1)
    ax[2].text(len(pr) - 1, 0.515, "random ordering", ha="right", fontsize=9, color=P["muted"])
    for xi, n in zip(xb, pr["Pairs"]):
        ax[2].text(xi, 0.43, f"{n}", ha="center", fontsize=8.5, color=P["ink"])
    ax[2].set_xticks(xb)
    ax[2].set_xticklabels(pr["DFT_gap_eV"].str.replace("-", "–").str.replace("<=", "≤"))
    ax[2].set_xlabel(r"|$\Delta$E$_{ads}$| between two records of the same adsorbate (eV)")
    ax[2].set_ylabel("Fraction of pairs ordered correctly")
    ax[2].set_ylim(0.4, 1.0)
    ax[2].legend(loc="upper left", fontsize=9.5)
    g.style_axes(ax[2], tick_size=11, label_size=12)

    hw = ads.loc[a_order, "Halfwidth90_mondrian"]
    ax[3].bar(x, hw, 0.55, color=[g.ADSORBATE_COLORS[a] for a in a_order], edgecolor="white")
    ax[3].axhline(float(ads.loc["All", "Halfwidth90_global"]), color=P["ink"], ls="--", lw=1.1)
    ax[3].text(-0.4, float(ads.loc["All", "Halfwidth90_global"]) + 0.55, "single global width", ha="left", fontsize=9)
    for xi, a in enumerate(a_order):
        ax[3].text(xi, hw[a] + 0.12, f"{100 * ads.loc[a, 'Coverage90_mondrian']:.0f}%", ha="center", fontsize=9)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels([g.ADSORBATE_LABELS[a] for a in a_order])
    ax[3].set_ylabel("90% interval half-width (eV)")
    ax[3].set_ylim(0, float(hw.max()) * 1.15)
    g.style_axes(ax[3], tick_size=11, label_size=12)
    g.add_panel_labels(fig, ax, fontsize=18)
    save(fig, "FigS1_validation_baselines_intervals")


def fig_s2() -> None:
    wgs, dop = g.load_data()
    ctx = g.prepare_training(wgs, dop)
    prep = clone(ctx["fitted"]["ExtraTrees"].named_steps["prep"]).fit(ctx["x_train"])
    ztr, zte, zdo = (prep.transform(ctx[k]) for k in ["x_train", "x_test", "x_dop"])
    nn = NearestNeighbors(n_neighbors=6).fit(ztr)
    d_tr = nn.kneighbors(ztr)[0][:, 1:].mean(1)
    d_te = nn.kneighbors(zte, n_neighbors=5)[0].mean(1)
    d_do = nn.kneighbors(zdo, n_neighbors=5)[0].mean(1)
    thr = np.percentile(d_tr, 95)
    miss = dop["E_Form"].isna().to_numpy()
    flags = pd.read_csv(TAB / "dopant_ad_flags.csv")
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    bins = np.linspace(0.5, 3.5, 31)
    ax[0].hist(wgs["Bader_M1"], bins=bins, density=True, color=P["blue"], alpha=0.55, label=f"WGS (n = {len(wgs)})")
    ax[0].hist(dop["Bader_M1"].dropna(), bins=bins, density=True, color=P["red"], alpha=0.7, label=f"Doped, descriptors available (n = {int(dop['Bader_M1'].notna().sum())})")
    ax[0].set_xlabel("Bader charge of M1 (e)")
    ax[0].set_ylabel("Density")
    ax[0].legend(fontsize=9)
    g.style_axes(ax[0], tick_size=11, label_size=12)
    layers = [3, 4, 5, 6, 7, 9]
    wv = [100 * np.mean(wgs["N_Layers"] == k) for k in layers]
    dv = [100 * np.mean(dop["N_Layers"] == k) for k in layers]
    xx = np.arange(len(layers))
    ax[1].bar(xx - 0.2, wv, 0.4, color=P["blue"], alpha=0.8, label="WGS")
    ax[1].bar(xx + 0.2, dv, 0.4, color=P["red"], alpha=0.8, label="Doped")
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels([str(k) for k in layers])
    ax[1].set_xlabel("Number of atomic layers")
    ax[1].set_ylabel("Share of records (%)")
    ax[1].legend(fontsize=9.5)
    g.style_axes(ax[1], tick_size=11, label_size=12)
    b2 = np.linspace(0, max(d_do.max(), d_te.max()) * 1.05, 40)
    ax[2].hist(d_tr, bins=b2, density=True, color=P["muted"], alpha=0.5, label="Training (leave-self-out)")
    ax[2].hist(d_te, bins=b2, density=True, color=P["blue"], alpha=0.6, label="WGS hold-out")
    ax[2].hist(d_do[miss], bins=b2, density=True, color=P["orange"], alpha=0.6, label="Doped, descriptors missing")
    ax[2].hist(d_do[~miss], bins=b2, density=True, color=P["red"], alpha=0.7, label="Doped, descriptors available")
    ax[2].axvline(thr, color=P["ink"], ls="--", lw=1.2)
    ax[2].text(thr * 1.02, ax[2].get_ylim()[1] * 0.45, "training P95", fontsize=9)
    ax[2].set_xlabel("Mean distance to five nearest training records")
    ax[2].set_ylabel("Density")
    ax[2].legend(fontsize=8.6, loc="upper left")
    g.style_axes(ax[2], tick_size=11, label_size=12)
    lab = ["WGS\nhold-out", "Doped,\nall", "Doped, descr.\navailable", "Doped, descr.\nmissing"]
    cov = flags["90% interval coverage"].to_numpy()
    knn = flags["kNN flag > train P95 (%)"].to_numpy() / 100
    xx = np.arange(4)
    ax[3].bar(xx - 0.2, cov, 0.4, color=P["teal"], label="90% interval coverage")
    ax[3].bar(xx + 0.2, knn, 0.4, color=P["red"], alpha=0.8, label="Share flagged by distance")
    ax[3].axhline(0.9, color=P["ink"], ls="--", lw=1)
    ax[3].set_xticks(xx)
    ax[3].set_xticklabels(lab, fontsize=9.5)
    ax[3].set_ylabel("Fraction of records")
    ax[3].set_ylim(0, 1.08)
    ax[3].legend(fontsize=9, loc="upper left")
    g.style_axes(ax[3], tick_size=10, label_size=12)
    g.add_panel_labels(fig, ax, fontsize=18)
    save(fig, "FigS2_domain_shift_applicability")


def fig_s3() -> None:
    raw = pd.read_csv(TAB / "fewshot_raw.csv")
    styles = {"Ensemble, WGS + k dopant": (P["gold"], "o", "-"), "ExtraTrees, WGS + k dopant": (P["teal"], "s", "-"),
              "Ensemble, k dopant only": (P["red"], "o", "--"), "ExtraTrees, k dopant only": (P["blue"], "s", "--"),
              "Host-adsorbate group mean of k dopant records": (P["muted"], "^", ":")}
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.6))
    for row, design in enumerate(["Random", "Dopant-disjoint"]):
        sub = raw[raw["Design"] == design]
        for col, (m, lab) in enumerate([("R2", r"R$^2$ on held-out dopant records"),
                                        ("Within_combination_rho", r"Within host–adsorbate Spearman $\rho$")]):
            ax = axes[row, col]
            for setting, (c, mk, ls) in styles.items():
                q = sub[sub["Setting"] == setting].groupby("k")[m]
                if q.ngroups == 0:
                    continue
                med, lo, hi = q.median(), q.quantile(0.25), q.quantile(0.75)
                ax.errorbar(med.index, med, yerr=[med - lo, hi - med], color=c, marker=mk, ls=ls, capsize=3, lw=1.6, ms=5,
                            label=setting.replace("Host-adsorbate group mean of k dopant records", "Group mean (k dopant)"))
            ax.axhline(0, color=P["muted"], lw=0.8, ls="--")
            ax.set_xlabel("Dopant records in training, k")
            ax.set_ylabel(lab)
            ax.set_title("Random draws" if design == "Random" else "Test dopants absent from training", fontsize=12)
            if m == "R2":
                ax.set_ylim(-1.6, 1.0)
            g.style_axes(ax, tick_size=11, label_size=12)
    axes[0, 0].legend(fontsize=8.8, loc="lower right")
    fig.tight_layout()
    g.add_panel_labels(fig, axes.ravel(), fontsize=18)
    save(fig, "FigS3_few_shot_dopant_analysis")


if __name__ == "__main__":
    g.configure_style()
    which = sys.argv[1:] or ["s1", "s2", "s3"]
    for w in which:
        {"s1": fig_s1, "s2": fig_s2, "s3": fig_s3}[w]()
    print("DONE")
