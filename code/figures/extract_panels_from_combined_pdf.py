from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DPI = 600
ZOOM = DPI / 72


PANEL_BOXES: dict[int, tuple[str, list[tuple[str, tuple[float, float, float, float]]]]] = {
    2: (
        "Fig02_dataset_correlation.pdf",
        [
            ("Fig02_panel_a_label_availability", (0.025, 0.02, 0.325, 0.49)),
            ("Fig02_panel_b_eads_histogram", (0.345, 0.02, 0.655, 0.49)),
            ("Fig02_panel_c_eads_by_adsorbate", (0.675, 0.02, 0.990, 0.49)),
            ("Fig02_panel_d_pearson_correlation", (0.025, 0.515, 0.330, 0.995)),
            ("Fig02_panel_e_spearman_correlation", (0.350, 0.515, 0.655, 0.995)),
            ("Fig02_panel_f_mutual_information", (0.675, 0.515, 0.995, 0.995)),
        ],
    ),
    3: (
        "Fig03_feature_selection.pdf",
        [
            ("Fig03_panel_a_permutation_importance", (0.020, 0.025, 0.335, 0.985)),
            ("Fig03_panel_b_feature_dendrogram", (0.355, 0.025, 0.665, 0.985)),
            ("Fig03_panel_c_consensus_diagram", (0.685, 0.025, 0.985, 0.985)),
        ],
    ),
    4: (
        "Fig04_model_benchmark.pdf",
        [
            ("Fig04_panel_a_rf", (0.015, 0.010, 0.335, 0.315)),
            ("Fig04_panel_b_extratrees", (0.345, 0.010, 0.665, 0.315)),
            ("Fig04_panel_c_gbr", (0.675, 0.010, 0.995, 0.315)),
            ("Fig04_panel_d_hgbr", (0.015, 0.355, 0.335, 0.660)),
            ("Fig04_panel_e_svr", (0.345, 0.355, 0.665, 0.660)),
            ("Fig04_panel_f_mlp", (0.675, 0.355, 0.995, 0.660)),
            ("Fig04_panel_g_metrics_comparison", (0.020, 0.710, 0.665, 0.995)),
            ("Fig04_panel_h_learning_curve", (0.665, 0.740, 0.995, 0.995)),
        ],
    ),
    5: (
        "Fig05_topsis_wilcoxon.pdf",
        [
            ("Fig05_panel_a_topsis_weights", (0.020, 0.030, 0.325, 0.970)),
            ("Fig05_panel_b_error_distribution", (0.355, 0.030, 0.660, 0.970)),
            ("Fig05_panel_c_wilcoxon_heatmap", (0.690, 0.030, 0.995, 0.970)),
        ],
    ),
    6: (
        "Fig06_interpretability.pdf",
        [
            ("Fig06_panel_a_contribution_scatter", (0.015, 0.015, 0.350, 0.985)),
            ("Fig06_panel_b_permutation_importance", (0.370, 0.015, 0.665, 0.495)),
            ("Fig06_panel_c_pdp", (0.685, 0.015, 0.985, 0.495)),
            ("Fig06_panel_d_ale", (0.370, 0.535, 0.665, 0.985)),
            ("Fig06_panel_e_interaction_dependency", (0.685, 0.535, 0.985, 0.985)),
        ],
    ),
    7: (
        "Fig07_causal_sensitivity_barrier.pdf",
        [
            ("Fig07_panel_a_causal_dag", (0.015, 0.020, 0.335, 0.980)),
            ("Fig07_panel_b_counterfactual_effects", (0.355, 0.020, 0.665, 0.980)),
            ("Fig07_panel_c_ads_vs_barrier", (0.685, 0.020, 0.995, 0.980)),
        ],
    ),
    8: (
        "Fig08_dopant_validation_screening.pdf",
        [
            ("Fig08_panel_a_transfer_scatter", (0.000, 0.000, 0.315, 0.505)),
            ("Fig08_panel_b_residual_boxplot", (0.335, 0.000, 0.665, 0.430)),
            ("Fig08_panel_c_dft_score_barh", (0.690, 0.000, 0.995, 0.430)),
            ("Fig08_panel_d_periodic_heatmap", (0.030, 0.500, 0.995, 0.985)),
        ],
    ),
}


def render_pdf(pdf_path: Path) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_box(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = box
    return (
        max(0, round(x0 * width)),
        max(0, round(y0 * height)),
        min(width, round(x1 * width)),
        min(height, round(y1 * height)),
    )


def save_crop(crop: Image.Image, out_base: Path) -> None:
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    crop.save(png, dpi=(DPI, DPI), optimize=True)
    # Keep panel PDFs visually identical to the panel PNGs and the combined PDF.
    crop.save(pdf, "PDF", resolution=DPI)


def main() -> None:
    outputs: list[Path] = []
    for fig_no, (pdf_name, panels) in PANEL_BOXES.items():
        folder = ROOT / f"Fig{fig_no:02d}"
        pdf_path = folder / pdf_name
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        rendered = render_pdf(pdf_path)
        for panel_name, box in panels:
            crop = rendered.crop(crop_box(rendered.size, box))
            out_base = folder / panel_name
            save_crop(crop, out_base)
            outputs.append(out_base.with_suffix(".png"))
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
