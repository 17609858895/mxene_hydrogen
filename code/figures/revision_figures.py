"""Revised main-text figures for the MTCOMM-D-26-07228 revision.

Changes relative to generate_all_figures.py (data and models unchanged):
- Fig. 3: panel (c) replaced by a membership matrix (the consensus descriptors are shared by at least
  two of the three relevance measures; the earlier Venn sketch implied all three).
- Fig. 4: training/hold-out wording in legends and annotations; panel (g) shows
  hold-out errors in eV only, with R2 printed above each model, so that
  dimensionless and eV quantities no longer share one axis.
- Fig. 5: the Wilcoxon matrix shows Holm-adjusted p values (21 comparisons).
- Fig. 7: panel (c) shows channel-specific trends and within-channel Spearman
  coefficients instead of a single pooled smoother; panel labels added.
- Fig. 8: panel (a) states that the dopant records are an external domain never
  used in training; consistent fonts and minus signs; dopant label Rd -> Pd.

Run after generate_all_figures.py (it reuses the learning-curve file written there); writes into outputs/figures/Fig0X/.
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr, wilcoxon
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_all_figures as g  # noqa: E402

P = g.PALETTE
MINUS = "−"


def fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace("-", MINUS)


def figure_4(ctx: dict) -> None:
    models = [m for m in ctx["metrics"].index if m != "TOPSIS_ensemble"]
    fig = plt.figure(figsize=(16.4, 12.2))
    outer = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.62], hspace=0.32, wspace=0.18,
                             left=0.055, right=0.992, top=0.990, bottom=0.060)
    y_te, y_tr = ctx["y_test"].to_numpy(), ctx["y_train"].to_numpy()
    allv = [y_tr, y_te] + [ctx[k][m] for m in models for k in ("preds_train", "preds_test")]
    lo, hi = min(np.min(v) for v in allv) - 0.6, max(np.max(v) for v in allv) + 0.45
    xs = np.array([lo, hi])
    panel_axes = []
    for i, name in enumerate(models):
        inner = outer[i // 3, i % 3].subgridspec(1, 2, width_ratios=[4.15, 1.2], wspace=0.045)
        ax = fig.add_subplot(inner[0, 0])
        ax_res = fig.add_subplot(inner[0, 1], sharey=ax)
        panel_axes.append(ax)
        c = g.MODEL_COLORS.get(name, P["blue"])
        p_tr, p_te = ctx["preds_train"][name], ctx["preds_test"][name]
        ax.plot([lo, hi], [lo, hi], "--", color=P["ink"], lw=1.45, alpha=0.65, zorder=1)
        ax.fill_between(xs, xs - 0.5, xs + 0.5, color="#8A8F98", alpha=0.13, zorder=0)
        ax.scatter(y_tr, p_tr, s=44, facecolors="white", edgecolors=c, linewidths=1.25, alpha=0.78, zorder=3, label="Training (n = 479)")
        ax.scatter(y_te, p_te, s=68, facecolors=c, edgecolors=P["ink"], linewidths=0.9, marker="^", alpha=0.95, zorder=4, label="Hold-out (n = 120)")
        m = ctx["metrics"].loc[name]
        txt = (f"{g.label_model(name)}\n"
               f"Training R$^2$ = {m.Train_R2:.3f}\n"
               f"Hold-out R$^2$ = {m.R2:.3f}\n"
               f"Hold-out RMSE = {m.RMSE:.2f} eV")
        ax.text(0.045, 0.955, txt, transform=ax.transAxes, fontsize=11.5, fontweight="bold", va="top", color=P["ink"],
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c, lw=1.2, alpha=0.96))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(r"DFT E$_{ads}$ (eV)")
        ax.set_ylabel(r"Predicted E$_{ads}$ (eV)")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.75)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9.5, handletextpad=0.25, borderpad=0.2)
        g.style_axes(ax)
        r_tr, r_te = p_tr - y_tr, p_te - y_te
        ax_res.scatter(r_tr, p_tr, s=22, facecolors="white", edgecolors=c, linewidths=0.9, alpha=0.72, zorder=3)
        ax_res.scatter(r_te, p_te, s=34, facecolors=c, edgecolors=P["ink"], linewidths=0.75, marker="^", alpha=0.95, zorder=4)
        ax_res.axvline(0, color=P["muted"], lw=1.0, ls="--", alpha=0.68)
        rmax = max(np.percentile(np.abs(np.r_[r_tr, r_te]), 98) * 1.15, 0.75)
        ax_res.set_xlim(-rmax, rmax)
        ax_res.xaxis.set_major_locator(MaxNLocator(3))
        ax_res.set_xlabel("Residual (eV)", fontsize=11, fontweight="bold")
        ax_res.tick_params(axis="y", labelleft=False)
        ax_res.grid(alpha=0.25, linestyle="--", linewidth=0.65)
        g.style_axes(ax_res, tick_size=9, label_size=11)

    met = ctx["metrics"].loc[models + ["TOPSIS_ensemble"]]
    ax_m = fig.add_subplot(outer[2, :2])
    x = np.arange(len(met))
    width = 0.34
    ax_m.bar(x - width / 2, met["RMSE"], width=width, color="#E6A06A", edgecolor="white", label="RMSE")
    ax_m.bar(x + width / 2, met["MAE"], width=width, color="#8A7EB8", edgecolor="white", label="MAE")
    for xi, (r2, top) in enumerate(zip(met["R2"], met[["RMSE", "MAE"]].max(axis=1))):
        ax_m.text(xi, top + 0.03, f"R$^2$ = {r2:.3f}", ha="center", va="bottom", fontsize=10, color=P["ink"])
    ax_m.set_xticks(x)
    ax_m.set_xticklabels([g.label_model(i) for i in met.index], rotation=0)
    ax_m.set_ylabel("Hold-out error (eV)")
    ax_m.set_ylim(0, float(met["RMSE"].max()) * 1.25)
    ax_m.yaxis.set_major_locator(MaxNLocator(5))
    ax_m.legend(loc="upper right", ncol=2, fontsize=10)
    ax_m.grid(axis="y", alpha=0.26)
    g.style_axes(ax_m, tick_size=11, label_size=12)

    lc = pd.read_csv(g.OUT / "Fig04" / "extra_trees_learning_curve.csv")
    s = lc.groupby("Training samples").agg(tm=("Training", "mean"), ts=("Training", "std"), vm=("Validation", "mean"), vs=("Validation", "std"))
    ax_lc = fig.add_subplot(outer[2, 2])
    xl = s.index.to_numpy(float)
    for mc, sc, col, lab in [("tm", "ts", g.MODEL_COLORS["ExtraTrees"], "Training subsample"), ("vm", "vs", P["red"], "Hold-out set")]:
        ax_lc.plot(xl, s[mc], "-o", color=col, lw=1.8, ms=4.5, label=lab)
        ax_lc.fill_between(xl, s[mc] - s[sc].fillna(0), s[mc] + s[sc].fillna(0), color=col, alpha=0.13, linewidth=0)
    ax_lc.set_xlabel("Training records")
    ax_lc.set_ylabel(r"R$^2$")
    ax_lc.set_ylim(max(0.0, float(np.nanmin(s[["tm", "vm"]].values)) - 0.18), 1.04)
    ax_lc.xaxis.set_major_locator(MaxNLocator(5))
    ax_lc.yaxis.set_major_locator(MaxNLocator(5))
    ax_lc.legend(loc="lower right", fontsize=9)
    ax_lc.grid(alpha=0.26)
    g.style_axes(ax_lc, tick_size=11, label_size=12)
    g.add_panel_labels(fig, panel_axes + [ax_m, ax_lc], fontsize=24)
    g.savefig(fig, 4, "model_benchmark")


def figure_5(ctx: dict) -> None:
    models = [m for m in ctx["metrics"].index if m != "TOPSIS_ensemble"] + ["TOPSIS_ensemble"]
    y = ctx["y_test"].to_numpy()
    err = pd.DataFrame([{"Model": n, "AbsError": e} for n in models for e in np.abs(y - ctx["preds_test"][n])])
    err["Model_label"] = err["Model"].map(g.label_model)
    pairs, pv = [], []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs.append((a, b))
            pv.append(wilcoxon(np.abs(y - ctx["preds_test"][a]), np.abs(y - ctx["preds_test"][b])).pvalue)
    holm = multipletests(pv, method="holm")[1]
    pmat = pd.DataFrame(np.nan, index=models, columns=models)
    for (a, b), h in zip(pairs, holm):
        pmat.loc[a, b] = pmat.loc[b, a] = h
    fig = plt.figure(figsize=(15.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.28, 1.24], wspace=0.44)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    w = ctx["weights"].sort_values()
    ax0.barh([g.label_model(i) for i in w.index], w.values, color=[g.MODEL_COLORS.get(i, P["gold"]) for i in w.index], edgecolor="white")
    ax0.set_xlabel("Ensemble weight (training-set CV)")
    ax0.set_xlim(0, max(0.34, float(w.max()) + 0.06))
    for i, v in enumerate(w.values):
        ax0.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    g.style_axes(ax0)
    order = [g.label_model(i) for i in ctx["metrics"].sort_values("RMSE").index]
    pal = {g.label_model(k): v for k, v in g.MODEL_COLORS.items()}
    sns.boxplot(data=err, y="Model_label", x="AbsError", order=order, ax=ax1, palette=pal, fliersize=1.8, linewidth=0.8, orient="h")
    sns.stripplot(data=err.sample(min(len(err), 420), random_state=g.RANDOM_STATE), y="Model_label", x="AbsError", order=order,
                  ax=ax1, color=P["ink"], alpha=0.25, size=2, orient="h")
    ax1.set_xlabel(r"Hold-out |DFT $-$ predicted| E$_{ads}$ (eV)")
    ax1.set_ylabel("")
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    g.style_axes(ax1)
    logp = -np.log10(pmat.astype(float)).rename(index=g.label_model, columns=g.label_model)
    sns.heatmap(logp, cmap="rocket_r", vmin=0, vmax=np.nanpercentile(logp.values, 95), ax=ax2,
                cbar_kws={"label": r"$-\log_{10}(p_{Holm})$", "pad": 0.03}, linewidths=0.2, linecolor="white")
    ax2.tick_params(axis="x", rotation=35, labelsize=11)
    ax2.tick_params(axis="y", rotation=0, labelsize=11)
    for t in ax2.get_xticklabels():
        t.set_ha("right")
    g.style_axes(ax2, tick_size=12, show_ticks=False)
    pd.DataFrame(pmat).to_csv(g.OUT / "Fig05" / "wilcoxon_p_values_holm.csv", encoding="utf-8-sig")
    g.add_panel_labels(fig, [ax0, ax1, ax2])
    g.savefig(fig, 5, "topsis_wilcoxon")


def figure_7(ctx: dict, wgs: pd.DataFrame) -> None:
    import networkx as nx
    pipe = ctx["fitted"][ctx["best_model"]]
    r = permutation_importance(pipe, ctx["x_test"], ctx["y_test"], n_repeats=12, random_state=g.RANDOM_STATE, scoring="r2", n_jobs=1)
    imp = pd.DataFrame({"feature": ctx["x_test"].columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    numeric_top = [f for f in imp["feature"] if pd.api.types.is_numeric_dtype(ctx["x_test"][f]) and not f.endswith("_missing")
                   and pd.to_numeric(ctx["x_train"][f], errors="coerce").nunique(dropna=True) > 8]
    eff = pd.DataFrame([dict(zip(["Feature", "Effect", "CI_low", "CI_high"], (f, *g.counterfactual_effect(pipe, ctx["x_test"].copy(), f))))
                        for f in numeric_top[:8]]).sort_values("Effect")
    bar = wgs[wgs["E_activation_ev"].notna()].copy()
    fig = plt.figure(figsize=(14.6, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.02, 1.30], wspace=0.30)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    G = nx.DiGraph()
    edges = [("Composition", "Structure"), ("Composition", "Electronic state"), ("Surface termination", "Electronic state"),
             ("Structure", "E_ads"), ("Electronic state", "E_ads"), ("Adsorbate", "E_ads"),
             ("E_ads", "Activation barrier"), ("Reaction path", "Activation barrier")]
    G.add_edges_from(edges)
    pos = {"Composition": (0.14, 0.74), "Surface termination": (0.14, 0.34), "Structure": (0.42, 0.82), "Electronic state": (0.42, 0.46),
           "Adsorbate": (0.42, 0.12), "E_ads": (0.68, 0.49), "Reaction path": (0.68, 0.16), "Activation barrier": (0.92, 0.31)}
    labels = {"Composition": "Compo-\nsition", "Surface termination": "Surface\nterm.", "Structure": "Struc-\nture",
              "Electronic state": "Electr.\nstate", "Adsorbate": "Adsor-\nbate", "E_ads": r"E$_{ads}$",
              "Reaction path": "Reaction\npath", "Activation barrier": "Activ.\nbarrier"}
    col = {"Composition": P["blue"], "Surface termination": P["green"], "Structure": P["cyan"], "Electronic state": P["green"],
           "Adsorbate": P["red"], "E_ads": P["orange"], "Reaction path": P["gold"], "Activation barrier": P["purple"]}
    nx.draw_networkx_edges(G, pos, ax=ax0, arrows=True, arrowstyle="-|>", arrowsize=16, edge_color=P["muted"], width=1.4,
                           node_size=2300, min_source_margin=18, min_target_margin=22)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=2300, node_color=[col[n] for n in G.nodes()], edgecolors="white", linewidths=1.3)

    def lab_col(h):
        h = h.lstrip("#")
        rr, gg, bb = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "white" if 0.299 * rr + 0.587 * gg + 0.114 * bb < 150 else P["ink"]
    for n, (lx, ly) in pos.items():
        ax0.text(lx, ly, labels[n], ha="center", va="center", fontsize=8, fontweight="bold", color=lab_col(col[n]))
    ax0.set_axis_off()
    ax0.set_xlim(0, 1.05)
    ax0.set_ylim(0, 0.96)
    yy = np.arange(len(eff))
    ax1.errorbar(eff["Effect"], yy, xerr=[eff["Effect"] - eff["CI_low"], eff["CI_high"] - eff["Effect"]], fmt="o",
                 color=P["red"], ecolor=P["muted"], capsize=3)
    ax1.axvline(0, color=P["muted"], lw=0.9, ls="--")
    ax1.set_yticks(yy)
    ax1.set_yticklabels([g.label_feature(f) for f in eff["Feature"]])
    ax1.set_xlabel(r"Mean $\Delta$E$_{ads}$ for +0.5 SD (eV)")
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    g.style_axes(ax1)
    chan_col = {"H2O --> HO + H": P["blue"], "CO2*  --> CO* + O*": P["orange"], "H2 ---> 2H": P["purple"]}
    chan_lab = {"H2O --> HO + H": r"H$_2$O $\rightarrow$ OH + H", "CO2*  --> CO* + O*": r"CO$_2^*$ $\rightarrow$ CO$^*$ + O$^*$",
                "H2 ---> 2H": r"H$_2$ $\rightarrow$ 2H"}
    for ch, s in bar.groupby("Reaction"):
        rho, p = spearmanr(s["E_ads_ev"], s["E_activation_ev"])
        ptxt = "p < 0.001" if p < 0.001 else (f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")
        ax2.scatter(s["E_ads_ev"], s["E_activation_ev"], s=46, color=chan_col[ch], edgecolor="white", linewidth=0.4,
                    label=f"{chan_lab[ch]}\nn = {len(s)}, $\\rho$ = {rho:.2f} ({ptxt})", zorder=3)
        b1, b0 = np.polyfit(s["E_ads_ev"], s["E_activation_ev"], 1)
        xg = np.linspace(s["E_ads_ev"].min(), s["E_ads_ev"].max(), 20)
        ax2.plot(xg, b0 + b1 * xg, color=chan_col[ch], lw=1.6, alpha=0.9, zorder=2)
    rho_all, _ = spearmanr(bar["E_ads_ev"], bar["E_activation_ev"])
    ax2.text(0.03, 0.97, f"All 62 records: $\\rho$ = {rho_all:.2f}\nChannel-structure null: $\\rho$ $\\approx$ 0.51",
             transform=ax2.transAxes, va="top", fontsize=10.5, bbox=dict(fc="white", ec="none", alpha=0.85))
    ax2.set_xlabel(r"E$_{ads}$ (eV)")
    ax2.set_ylabel(r"E$_a$ (eV)")
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.87), fontsize=9.3, ncol=1, handletextpad=0.3, labelspacing=0.9)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    g.style_axes(ax2)
    eff.to_csv(g.OUT / "Fig07" / "counterfactual_sensitivity.csv", index=False, encoding="utf-8-sig")
    g.add_panel_labels(fig, [ax0, ax1, ax2])
    g.savefig(fig, 7, "causal_sensitivity_barrier")


def figure_8(ctx: dict) -> None:
    dop = ctx["dop"].copy()
    dop["pred_E_ads"] = ctx["preds_dop"]["TOPSIS_ensemble"]
    dop["residual"] = dop["pred_E_ads"] - dop["E_ads_ev"]
    dop["Dopant_clean"] = dop["Dopant"].replace(g.ALIASES)
    win = lambda e: np.exp(-((e + 0.80) / 0.90) ** 2)  # noqa: E731
    dop["dft_window_score"], dop["pred_screening_score"] = win(dop["E_ads_ev"]), win(dop["pred_E_ads"])
    mt = g.metrics(dop["E_ads_ev"], dop["pred_E_ads"])
    rho, _ = spearmanr(dop["E_ads_ev"], dop["pred_E_ads"])
    fig = plt.figure(figsize=(15.0, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.18], width_ratios=[1.10, 1.26, 1.12], hspace=0.58, wspace=0.34)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax3 = fig.add_subplot(gs[1, :])
    lo = min(dop["E_ads_ev"].min(), dop["pred_E_ads"].min()) - 0.25
    hi = max(dop["E_ads_ev"].max(), dop["pred_E_ads"].max()) + 0.25
    for mol, s in dop.groupby("Mol"):
        ax0.scatter(s["E_ads_ev"], s["pred_E_ads"], s=44, color=g.ADSORBATE_COLORS[mol], edgecolor="white", linewidth=0.3,
                    label=g.ADSORBATE_LABELS.get(mol, mol))
    ax0.plot([lo, hi], [lo, hi], ls="--", color=P["red"], lw=1.2)
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    ax0.yaxis.set_major_locator(MaxNLocator(5))
    ax0.set_xlabel(r"DFT E$_{ads}$, doped MXenes (eV)")
    ax0.set_ylabel(r"Predicted E$_{ads}$ (eV)")
    ax0.text(0.03, 0.66, f"External domain\n(not used in training)\nn = {len(dop)}; R$^2$ = {fmt(mt['R2'])}\n"
             f"RMSE = {mt['RMSE']:.2f} eV\n$\\rho$ = {rho:.2f}", transform=ax0.transAxes, ha="left", va="top",
             fontsize=9.6, bbox=dict(fc="white", ec=P["light"], boxstyle="round,pad=0.3", alpha=0.95))
    ax0.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, handletextpad=0.25, columnspacing=0.8)
    g.style_axes(ax0)
    counts = dop["Dopant_clean"].value_counts()
    keep = counts[counts >= 3].index
    box = dop[dop["Dopant_clean"].isin(keep)]
    order = box.groupby("Dopant_clean")["residual"].median().sort_values().index
    sns.boxplot(data=box, x="Dopant_clean", y="residual", order=order, palette=sns.color_palette("crest", n_colors=len(order)),
                fliersize=2, linewidth=0.8, ax=ax1)
    ax1.axhline(0, color=P["muted"], lw=0.8, ls="--")
    ax1.set_xlabel("Dopant (n $\\geq$ 3)")
    ax1.set_ylabel("Residual (eV)")
    ax1.tick_params(axis="x", rotation=35)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    g.style_axes(ax1, tick_size=11)
    rank = dop.groupby("Dopant_clean").agg(n=("E_ads_ev", "size"), mean_dft=("E_ads_ev", "mean"), mean_pred=("pred_E_ads", "mean"),
                                           transfer_mae=("residual", lambda s: np.mean(np.abs(s))), dft_score=("dft_window_score", "mean"),
                                           pred_score=("pred_screening_score", "mean")).sort_values("dft_score", ascending=False)
    top = rank[rank["n"] >= 3].head(12).iloc[::-1]
    ax2.barh(top.index, top["dft_score"], color=P["teal"])
    ax2.set_xlabel("DFT-window score (heuristic)")
    ax2.set_xlim(0, 1.08)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    for i, (_, row) in enumerate(top.iterrows()):
        ax2.text(row["dft_score"] + 0.012, i, f"n = {int(row['n'])}", va="center", fontsize=9, clip_on=False)
    g.style_axes(ax2, tick_size=11)
    from matplotlib import patches
    from matplotlib.colors import Normalize
    ax3.set_axis_off()
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(vmin=rank["dft_score"].min(), vmax=rank["dft_score"].max())
    for el, row in rank.iterrows():
        if el not in g.PERIODIC_POS:
            continue
        period, group = g.PERIODIC_POS[el]
        x, y = group, 7 - period
        rect = patches.Rectangle((x - 0.45, y - 0.40), 0.82, 0.72, facecolor=cmap(norm(row["dft_score"])), edgecolor="white", linewidth=1.0)
        ax3.add_patch(rect)
        tc = "white" if norm(row["dft_score"]) > 0.72 else P["ink"]
        if row["n"] < 3:
            rect.set_hatch("///")
            rect.set_edgecolor(P["muted"])
        ax3.text(x - 0.04, y + 0.12, el, ha="center", va="center", fontsize=12, weight="bold", color=tc)
        ax3.text(x - 0.04, y - 0.10, f"{row['dft_score']:.2f}", ha="center", va="center", fontsize=9, color=tc)
        ax3.text(x - 0.04, y - 0.28, f"n = {int(row['n'])}", ha="center", va="center", fontsize=8, color=tc)
    ax3.set_xlim(2.3, 12.85)
    ax3.set_ylim(0.35, 3.72)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax3, fraction=0.025, pad=0.015)
    cb.set_label("DFT-window score", fontweight="bold", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    dop.to_csv(g.OUT / "Fig08" / "dopants_predictions.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(g.OUT / "Fig08" / "dopant_ranking.csv", encoding="utf-8-sig")
    g.add_panel_labels(fig, [ax0, ax1, ax2, ax3])
    g.savefig(fig, 8, "dopant_validation_screening")



def figure_3(ctx: dict, wgs: pd.DataFrame) -> None:
    """Same panels (a) and (b) as generate_all_figures.figure_3; panel (c) now shows which of the three
    relevance measures place each consensus descriptor in their top ten (membership matrix), because the
    seven consensus descriptors are shared by at least two measures and only three by all three."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_selection import mutual_info_regression

    pipe = ctx["fitted"][ctx["best_model"]]
    r = permutation_importance(pipe, ctx["x_test"], ctx["y_test"], n_repeats=12, random_state=g.RANDOM_STATE, scoring="r2", n_jobs=1)
    imp = pd.DataFrame({"feature": ctx["x_test"].columns, "importance": r.importances_mean, "std": r.importances_std}).sort_values("importance", ascending=False)
    num = g.numeric_frame_for_analysis(wgs)
    candidate = [c for c in imp["feature"].head(18) if c in num.columns]
    if len(candidate) < 8:
        candidate = g.top_numeric_features(num, 14)
    corr = num[candidate].corr(method="spearman").fillna(0).clip(-1, 1)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist.values, 0)
    Z = linkage(squareform(dist.values, checks=False), method="average")
    y = num["E_ads_ev"]
    cands = [c for c in num.columns if c != "E_ads_ev"]
    tops = {
        "Pearson": set(num[cands].corrwith(y).abs().sort_values(ascending=False).head(10).index),
        "Spearman": set(num[cands].corrwith(y, method="spearman").abs().sort_values(ascending=False).head(10).index),
        "Mutual\ninformation": set(pd.Series(mutual_info_regression(num[cands], y, random_state=g.RANDOM_STATE), index=cands).sort_values(ascending=False).head(10).index),
    }
    member = pd.DataFrame({k: [c in v for c in cands] for k, v in tops.items()}, index=cands)
    member = member[member.sum(axis=1) >= 2]
    member = member.assign(n=member.sum(axis=1), lab=[g.label_feature(c) for c in member.index]).sort_values(["n", "lab"], ascending=[True, False])

    fig = plt.figure(figsize=(16.0, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.04, 1.16, 1.0], wspace=0.48)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    top_imp = imp.head(14).iloc[::-1]
    yb = np.arange(len(top_imp))
    ax0.barh(yb, top_imp["importance"], xerr=top_imp["std"], color=P["teal"], alpha=0.9)
    ax0.set_yticks(yb)
    ax0.set_yticklabels([g.label_feature(f) for f in top_imp["feature"]])
    ax0.set_xlabel(r"Mean R$^2$ decrease (hold-out)")
    ax0.grid(axis="x", alpha=0.22)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    g.style_axes(ax0, tick_size=10, label_size=12)
    dendrogram(Z, labels=[g.label_feature(c) for c in candidate], orientation="right", leaf_font_size=8.8,
               color_threshold=np.median(Z[:, 2]), above_threshold_color=P["muted"], ax=ax1)
    ax1.set_xlabel(r"1 $-$ |Spearman $\rho$|")
    g.style_axes(ax1, tick_size=9.5, label_size=12)
    cols = list(tops)
    colors = [P["blue"], P["orange"], P["purple"]]
    for yi, (feat, row) in enumerate(member.iterrows()):
        for xi, (cname, col) in enumerate(zip(cols, colors)):
            if row[cname]:
                ax2.scatter(xi, yi, s=260, color=col, edgecolor="white", linewidth=1.0, zorder=3)
            else:
                ax2.scatter(xi, yi, s=230, facecolor="white", edgecolor=P["muted"], linewidth=1.2, zorder=3)
        ax2.text(len(cols) - 0.42, yi, f"{int(row['n'])}/3", va="center", ha="left", fontsize=11, color=P["ink"])
    ax2.set_yticks(range(len(member)))
    ax2.set_yticklabels(member["lab"])
    ax2.set_xticks(range(len(cols)))
    ax2.set_xticklabels(cols)
    ax2.set_xlim(-0.5, len(cols) + 0.05)
    ax2.set_ylim(-0.6, len(member) - 0.4)
    ax2.set_xlabel("Top-ten membership")
    ax2.grid(False)
    g.style_axes(ax2, tick_size=11, label_size=12)
    member.drop(columns="lab").to_csv(g.OUT / "Fig03" / "relevance_measure_membership.csv", encoding="utf-8-sig")
    g.add_panel_labels(fig, [ax0, ax1, ax2])
    g.savefig(fig, 3, "feature_selection")


def main() -> None:
    g.configure_style()
    wgs, dop = g.load_data()
    ctx = g.prepare_training(wgs, dop)
    figure_3(ctx, wgs)
    figure_4(ctx)
    figure_5(ctx)
    figure_7(ctx, wgs)
    figure_8(ctx)
    print("DONE")


if __name__ == "__main__":
    main()
