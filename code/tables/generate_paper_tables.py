"""Generate Table 1 and Supporting Information Tables S1-S4, S6 and S7b.

Tables S5, S7a, S8 and S9 and Fig. S1 are produced by
code/analysis/robustness_analyses.py.
All models, preprocessing and TOPSIS weights come from generate_all_figures.py,
so the numbers here are identical to those in the main-text figures.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
FIG_SCRIPT = ROOT / "code" / "figures" / "generate_all_figures.py"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["RF", "ExtraTrees", "GBR", "HGBR", "SVR", "MLP", "TOPSIS_ensemble"]
WINDOW_CENTRE, WINDOW_WIDTH = -0.80, 0.90


def load_figure_module():
    spec = importlib.util.spec_from_file_location("figures", FIG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FEATURE_META = {
    "a_ang": ("Structural", "Lattice constant", "Å", "In-plane lattice constant of the MXene slab"),
    "Stacking": ("Structural", "Stacking sequence", "–", "Layer stacking mode (ABA or ABC)"),
    "N_Layers": ("Structural", "Number of atomic layers", "–", "Atomic layers in the slab"),
    "Layer_dist_MX": ("Structural", "M–X interlayer distance", "Å", "Distance between metal and C/N layers"),
    "Layer_dist_MM": ("Structural", "M–M interlayer distance", "Å", "Distance between adjacent metal layers"),
    "Length_MX": ("Structural", "M–X bond length", "Å", "Bond length between metal and C/N"),
    "Length_MT1": ("Structural", "M–T bond length", "Å", "Bond length between metal and surface termination"),
    "E_Form": ("Thermodynamic", "Formation energy", "eV", "DFT formation energy of the MXene"),
    "Bader_M1": ("Electronic", "Bader charge of M1", "e", "Bader charge on the primary metal"),
    "Bader_M2": ("Electronic", "Bader charge of M2", "e", "Bader charge on the secondary metal"),
    "Bader_X": ("Electronic", "Bader charge of X", "e", "Bader charge on C or N"),
    "Bader_T1": ("Electronic", "Bader charge of T", "e", "Bader charge on the surface termination"),
    "Band_gap_PBE_ev": ("Electronic", "PBE band gap", "eV", "Band gap from the PBE functional"),
    "M1": ("Composition", "Primary metal", "–", "Primary metal element"),
    "M2": ("Composition", "Secondary metal", "–", "Secondary metal element (ordered double-metal MXenes)"),
    "X": ("Composition", "X element", "–", "C or N"),
    "T1_type": ("Surface chemistry", "Termination type", "–", "Surface termination group"),
    "Mol": ("Adsorbate", "Adsorbate species", "–", "WGSR-related adsorbate"),
    "Dopant": ("Dopant", "Dopant element", "–", "Substitutional dopant (dopants sheet only)"),
    "E_ads_ev": ("Target", "Adsorption energy", "eV", "Primary prediction target, Eq. (1)"),
    "E_activation_ev": ("Secondary label", "Activation energy", "eV", "Used only for the consistency analysis"),
    "E_reaction_ev": ("Secondary label", "Reaction energy", "eV", "Used only for the consistency analysis"),
}


def table_s1_labels(wgs_raw: pd.DataFrame, dop_raw: pd.DataFrame) -> pd.DataFrame:
    role = {
        "E_ads_ev": "Primary regression target",
        "E_activation_ev": "Consistency analysis only",
        "E_reaction_ev": "Consistency analysis only",
        "Reaction": "Reaction-path grouping",
    }
    rows = []
    for sheet, df in [("WGS", wgs_raw), ("Dopants", dop_raw)]:
        for target, r in role.items():
            valid = int(df[target].notna().sum()) if target in df.columns else 0
            rows.append({"Sheet": sheet, "Records": len(df), "Label": target, "Role": r,
                         "Valid": valid, "Missing (%)": round((len(df) - valid) / len(df) * 100, 1)})
    return pd.DataFrame(rows)


def table_s2_descriptors(wgs: pd.DataFrame, dop: pd.DataFrame) -> pd.DataFrame:
    """Coverage is computed separately for the adsorption-energy-labelled WGS (n=599)
    and dopants (n=94) records; summary statistics refer to the WGS domain."""
    rows = []
    for feature, (category, display, unit, definition) in FEATURE_META.items():
        s_w = wgs[feature] if feature in wgs.columns else pd.Series(np.nan, index=wgs.index)
        s_d = dop[feature] if feature in dop.columns else pd.Series(np.nan, index=dop.index)
        num = pd.to_numeric(s_w, errors="coerce")
        is_num = unit != "–"
        row = {
            "Category": category, "Descriptor": feature, "Description": f"{display}: {definition}", "Unit": unit,
            "WGS valid (n = 599)": int(s_w.notna().sum()),
            "WGS missing (%)": round(s_w.isna().mean() * 100, 1),
            "Dopants valid (n = 94)": int(s_d.notna().sum()),
            "Dopants missing (%)": round(s_d.isna().mean() * 100, 1),
        }
        if is_num and num.notna().any():
            row.update({"WGS mean": round(num.mean(), 2), "WGS SD": round(num.std(), 2),
                        "WGS min": round(num.min(), 2), "WGS max": round(num.max(), 2)})
        else:
            row.update({"WGS mean": "", "WGS SD": "", "WGS min": "", "WGS max": ""})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_suite(fig, x_ref, x_train, x_test, y_train, y_test) -> pd.DataFrame:
    """Same protocol as prepare_training: CV-derived TOPSIS weights, single hold-out evaluation."""
    models = fig.make_models(fig.build_preprocessor(x_ref))
    y_range = float(y_train.max() - y_train.min())
    cvm = fig.cv_metric_table(models, x_train, y_train)
    weights = fig.entropy_topsis(cvm[fig.METRIC_COLS])["ensemble_weight"].reindex(cvm.index)
    preds, rows = {}, []
    for name, pipe in models.items():
        pipe.fit(x_train, y_train)
        preds[name] = pipe.predict(x_test)
        rows.append({"Model": name, **fig.metrics(y_test, preds[name], y_range)})
    ens = np.column_stack([preds[m] for m in weights.index]) @ weights.to_numpy()
    rows.append({"Model": "TOPSIS_ensemble", **fig.metrics(y_test, ens, y_range)})
    return pd.DataFrame(rows).set_index("Model")


def table_s3_missingness(ctx: dict, fig, wgs: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    sources = [c for c in fig.NUMERIC_BASE + fig.CATEGORICAL_BASE if c in wgs.columns and wgs[c].isna().mean() >= threshold]
    removed = [c for c in ctx["x_wgs"].columns if any(c == s or c.startswith(f"{s}_") for s in sources)]
    kept = [c for c in ctx["x_wgs"].columns if c not in removed]
    base = ctx["metrics"][fig.METRIC_COLS]
    red = evaluate_suite(fig, ctx["x_wgs"][kept], ctx["x_train"][kept], ctx["x_test"][kept], ctx["y_train"], ctx["y_test"])
    out = pd.DataFrame({
        "R2 (all, 80 features)": base["R2"], "RMSE (all)": base["RMSE"], "MAE (all)": base["MAE"],
        f"R2 (reduced, {len(kept)} features)": red["R2"], "RMSE (reduced)": red["RMSE"], "MAE (reduced)": red["MAE"],
    }).loc[MODEL_ORDER]
    out["ΔR2"] = out.iloc[:, 3] - out.iloc[:, 0]
    out["ΔRMSE (eV)"] = out["RMSE (reduced)"] - out["RMSE (all)"]
    out.attrs["sources"] = sources
    out.attrs["n_removed"] = len(removed)
    return out.round(4).reset_index(names="Model")


def table_s4_topsis(ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    top = ctx["topsis"].copy().reset_index(names="Model")
    top = top[["Model", "R2", "RMSE", "MAE", "nRMSE", "TOPSIS_score", "ensemble_weight", "w_R2", "w_RMSE", "w_MAE", "w_nRMSE"]]
    y = ctx["y_test"].to_numpy()
    p = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER)
    for a in MODEL_ORDER:
        for b in MODEL_ORDER:
            if a != b:
                p.loc[a, b] = wilcoxon(np.abs(y - ctx["preds_test"][a]), np.abs(y - ctx["preds_test"][b])).pvalue
    return top.round(4), p


def table_s6_activation(wgs: pd.DataFrame) -> pd.DataFrame:
    sub = wgs[wgs["E_activation_ev"].notna()]
    out = sub.groupby("Reaction").agg(
        n=("E_activation_ev", "size"),
        Eads_mean=("E_ads_ev", "mean"), Eads_SD=("E_ads_ev", "std"),
        Ea_mean=("E_activation_ev", "mean"), Ea_SD=("E_activation_ev", "std"),
        Erxn_mean=("E_reaction_ev", "mean"), Erxn_SD=("E_reaction_ev", "std"),
    ).reset_index()
    rho, p = spearmanr(sub["E_ads_ev"], sub["E_activation_ev"])
    out.attrs["rho"], out.attrs["p"] = rho, p
    return out.round(3)


def table_s7b_dopants(ctx: dict, fig) -> pd.DataFrame:
    d = ctx["dop"].copy()
    d["pred"] = ctx["preds_dop"]["TOPSIS_ensemble"]
    d["Dopant"] = d["Dopant"].replace(fig.ALIASES)
    win = lambda e: np.exp(-((e - WINDOW_CENTRE) / WINDOW_WIDTH) ** 2)  # noqa: E731
    d["dft_score"], d["pred_score"] = win(d["E_ads_ev"]), win(d["pred"])
    out = d.groupby("Dopant").agg(
        n=("E_ads_ev", "size"),
        DFT_Eads_mean=("E_ads_ev", "mean"),
        Pred_Eads_mean=("pred", "mean"),
        Transfer_MAE=("pred", lambda s: float(np.mean(np.abs(s - d.loc[s.index, "E_ads_ev"])))),
        DFT_window_score=("dft_score", "mean"),
        Pred_window_score=("pred_score", "mean"),
    ).sort_values("DFT_window_score", ascending=False)
    return out.round(3).reset_index()


def main() -> None:
    fig = load_figure_module()
    wgs, dop = fig.load_data()
    ctx = fig.prepare_training(wgs, dop)
    wgs_raw = pd.read_excel(fig.DATA_PATH, sheet_name="WGS").replace("-", np.nan)
    dop_raw = pd.read_excel(fig.DATA_PATH, sheet_name="dopants").replace("-", np.nan)

    t1 = ctx["metrics"].loc[MODEL_ORDER, ["R2", "RMSE", "MAE", "nRMSE", "Train_R2", "Train_RMSE", "Train_MAE"]].round(4)
    tables = {
        "Table1_WGS_Model_Performance": t1.reset_index(names="Model"),
        "TableS1_Label_Availability": table_s1_labels(wgs_raw, dop_raw),
        "TableS2_Descriptors_and_Coverage": table_s2_descriptors(wgs, dop),
        "TableS3_Missingness_Sensitivity": table_s3_missingness(ctx, fig, wgs),
        "TableS6_Activation_Subset": table_s6_activation(wgs),
        "TableS7b_Dopant_Resolved": table_s7b_dopants(ctx, fig),
    }
    top, pmat = table_s4_topsis(ctx)
    tables["TableS4a_CV_TOPSIS"] = top
    tables["TableS4b_Wilcoxon_p"] = pmat.reset_index(names="Model")

    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    # Workbook collects every table CSV in the folder, including those written by robustness_analyses.py.
    with pd.ExcelWriter(OUT / "Paper_Tables_MXene_Eads.xlsx", engine="openpyxl") as writer:
        for csv in sorted(OUT.glob("Table*.csv")):
            if not csv.stem.endswith("_raw"):
                pd.read_csv(csv).to_excel(writer, sheet_name=csv.stem[:31], index=False)
    s6 = tables["TableS6_Activation_Subset"]
    print("Eads-Ea Spearman", s6.attrs.get("rho"), s6.attrs.get("p"))
    print("Saved tables to", OUT)


if __name__ == "__main__":
    main()
