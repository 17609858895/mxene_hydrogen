from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


PAPER_ROOT = Path(__file__).resolve().parents[2]
FIG_SCRIPT = PAPER_ROOT / "code" / "figures" / "generate_all_figures.py"

sys.path.insert(0, str(FIG_SCRIPT.parent))
import generate_all_figures as paper_model  # noqa: E402


RAW_COLUMNS = [
    "Formula",
    "Stacking",
    "M1",
    "M2",
    "X",
    "T1_type",
    "Mol",
    "a_ang",
    "N_Layers",
    "Layer_dist_MX",
    "Layer_dist_MM",
    "Length_MX",
    "Length_MT1",
    "E_Form",
    "Bader_M1",
    "Bader_M2",
    "Bader_X",
    "Bader_T1",
    "Band_gap_PBE_ev",
]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main() -> None:
    wgs, dop = paper_model.load_data()
    context = paper_model.prepare_training(wgs, dop)

    base_model_names = list(context["weights"].index)
    weights = context["weights"].astype(float).to_dict()
    fitted = {name: context["fitted"][name] for name in base_model_names}

    y_test = context["y_test"]
    pred_ensemble = context["preds_test"]["TOPSIS_ensemble"]
    pred_best = context["preds_test"][context["best_model"]]

    bundle = {
        "model_type": "TOPSIS ensemble",
        "best_single_model": context["best_model"],
        "models": fitted,
        "weights": weights,
        "raw_columns": RAW_COLUMNS,
        "feature_columns": list(context["x_wgs"].columns),
        "numeric_base": paper_model.NUMERIC_BASE,
        "categorical_base": paper_model.CATEGORICAL_BASE,
        "training_domain": "WGS-domain MXene adsorption energy",
        "target": "E_ads_ev",
        "unit": "eV",
    }
    model_dir = PAPER_ROOT / "outputs" / "model_export" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model_bundle.joblib", compress=3)

    metrics = {
        "TOPSIS_ensemble": {
            "R2": float(r2_score(y_test, pred_ensemble)),
            "RMSE": rmse(y_test, pred_ensemble),
            "MAE": float(mean_absolute_error(y_test, pred_ensemble)),
        },
        context["best_model"]: {
            "R2": float(r2_score(y_test, pred_best)),
            "RMSE": rmse(y_test, pred_best),
            "MAE": float(mean_absolute_error(y_test, pred_best)),
        },
        "model_weights": weights,
        "n_wgs_training_rows": int(len(wgs)),
        "n_holdout_rows": int(len(y_test)),
        "notes": [
            "The deployed model is intended for WGS-domain MXene adsorption systems.",
            "Missing descriptors are imputed by the same preprocessing pipeline used during training.",
            "Dopant-domain transfer was weak in the paper diagnostics; do not use this app as a reliable dopant screening engine.",
        ],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    example_dir = PAPER_ROOT / "outputs" / "model_export" / "data"
    example_dir.mkdir(parents=True, exist_ok=True)

    example = wgs[RAW_COLUMNS].head(12).copy()
    example.to_csv(example_dir / "example_input.csv", index=False, encoding="utf-8-sig")

    template = wgs[RAW_COLUMNS].head(1).copy()
    for col in RAW_COLUMNS:
        if col not in template.columns:
            template[col] = np.nan
    template.to_csv(example_dir / "single_prediction_template.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
