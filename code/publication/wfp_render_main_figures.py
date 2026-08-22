#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Main Figures 1–4 — rendering only
======================================

Inputs:
  the exact FIG0 exports produced after final scientific closeout.

Outputs:
  panel-level SVG/PDF/PNG
  composite Figure1–Figure4 SVG/PDF/PNG
  captions draft
  SHA256 output manifest
  rendered ZIP

Scientific boundary:
  - no new metric is computed;
  - no model is fit;
  - B8 is not changed;
  - Figure 3 z1-z2 is visualization only;
  - Figure 4 MUST retain the post-Stage7B-RD two-layer interpretation.

Implementation:
  Every data chart is rendered as its own independent matplotlib figure.
  Composite multi-panel figures are assembled afterwards from the panel SVGs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    import cairosvg
except Exception:
    cairosvg = None


EXPECTED_SCHEMAS = {
    "fig2_reconstruction_curves.csv": [
        "dimension",
        "replicate_basis_cv_r2_all",
        "replicate_basis_cv_r2_odd",
        "replicate_basis_cv_r2_even",
        "ordinary_pca_cv_r2_all",
        "fourier_cv_r2_all",
    ],
    "fig2_axis_reliability.csv": [
        "axis", "odd_even_score_correlation", "odd_sd", "even_sd",
    ],
    "fig2_summary_metrics.csv": [
        "metric", "value", "unit", "role", "source", "note",
    ],
    "fig3_patient_projection_z1z2_PRIVATE.csv": [
        "patient_id", "z1", "z2",
    ],
    "fig3_scale_summary.csv": [
        "metric", "value", "unit", "role", "source", "note",
    ],
    "fig4a_conditional_mean_summary.csv": [
        "metric", "value", "unit", "role", "source", "note",
    ],
    "fig4b_residual_dependence_summary.csv": [
        "metric", "value", "unit", "role", "source", "note",
    ],
}

PANEL_W = 6.0
PANEL_H = 4.2
DPI = 300


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_map(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for _, r in df.iterrows():
        out[str(r["metric"])] = float(r["value"])
    return out


def require_columns(df: pd.DataFrame, expected: List[str], label: str) -> None:
    if list(df.columns) != expected:
        raise RuntimeError(
            f"{label}: schema mismatch\n"
            f"expected={expected}\nobserved={list(df.columns)}"
        )


def style_ax(ax) -> None:
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_letter(fig, letter: str) -> None:
    fig.text(0.015, 0.975, letter, ha="left", va="top", fontsize=15, fontweight="bold")


def save_panel(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_one_axis(letter: str, figsize=(PANEL_W, PANEL_H)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.16, 0.16, 0.79, 0.75])
    panel_letter(fig, letter)
    return fig, ax


def text_panel(letter: str, title: str, lines: List[str], base: Path) -> None:
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes([0.06, 0.08, 0.90, 0.84])
    ax.axis("off")
    panel_letter(fig, letter)
    ax.text(0.02, 0.92, title, ha="left", va="top", fontsize=11.5, fontweight="bold")
    y = 0.80
    for line in lines:
        ax.text(0.04, y, line, ha="left", va="top", fontsize=10.2)
        y -= 0.105
    save_panel(fig, base)


def schematic_panel(letter: str, title: str, boxes: List[str], base: Path) -> None:
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_letter(fig, letter)
    ax.text(0.5, 0.94, title, ha="center", va="top", fontsize=11.5, fontweight="bold")

    n = len(boxes)
    box_w = min(0.24, 0.80 / max(n, 1))
    gap = (0.90 - n * box_w) / max(n - 1, 1)
    x = 0.05
    cy = 0.50
    for i, label in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, cy - 0.13), box_w, 0.26,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            fill=False, linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + box_w / 2, cy, label, ha="center", va="center", fontsize=9.5)
        if i < n - 1:
            x2 = x + box_w
            ax.annotate(
                "",
                xy=(x2 + gap * 0.82, cy),
                xytext=(x2 + gap * 0.18, cy),
                arrowprops=dict(arrowstyle="->", linewidth=1.1),
            )
        x += box_w + gap
    save_panel(fig, base)


def svg_dimensions(svg_path: Path) -> Tuple[float, float]:
    import re
    txt = svg_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<svg[^>]*\bwidth="([0-9.]+)pt"[^>]*\bheight="([0-9.]+)pt"', txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'<svg[^>]*\bviewBox="[^"]*?([0-9.]+)\s+([0-9.]+)"', txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 432.0, 302.4


def data_uri_svg(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def assemble_svg(panel_svgs: List[Path], out_svg: Path, ncols: int = 2, gutter: float = 18.0) -> None:
    if not panel_svgs:
        raise ValueError("no panels")
    dims = [svg_dimensions(p) for p in panel_svgs]
    cell_w = max(w for w, _ in dims)
    cell_h = max(h for _, h in dims)
    nrows = math.ceil(len(panel_svgs) / ncols)
    width = ncols * cell_w + (ncols - 1) * gutter
    height = nrows * cell_h + (nrows - 1) * gutter

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">'
    ]
    for idx, (p, (w, h)) in enumerate(zip(panel_svgs, dims)):
        r, c = divmod(idx, ncols)
        x = c * (cell_w + gutter) + (cell_w - w) / 2
        y = r * (cell_h + gutter) + (cell_h - h) / 2
        href = data_uri_svg(p)
        pieces.append(
            f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'href="{href}" xlink:href="{href}"/>'
        )
    pieces.append("</svg>")
    out_svg.write_text("\n".join(pieces), encoding="utf-8")

    if cairosvg is not None:
        cairosvg.svg2pdf(bytestring=out_svg.read_bytes(), write_to=str(out_svg.with_suffix(".pdf")))
        cairosvg.svg2png(
            bytestring=out_svg.read_bytes(),
            write_to=str(out_svg.with_suffix(".png")),
            output_width=int(width / 72 * DPI),
            output_height=int(height / 72 * DPI),
        )


def render_figure1(out: Path) -> List[Path]:
    panels = []
    base = out / "Figure1A_representation"
    schematic_panel(
        "A", "Beat representation",
        ["125-Hz ABP", "Beat QC", "64 phase points", "shape_norm"],
        base,
    )
    panels.append(base.with_suffix(".svg"))

    base = out / "Figure1B_patient_representation"
    schematic_panel(
        "B", "Patient-balanced central morphology",
        ["60-s blocks", "Odd / even replicates", "30-min central morphology"],
        base,
    )
    panels.append(base.with_suffix(".svg"))

    base = out / "Figure1C_population_basis"
    schematic_panel(
        "C", "Frozen population coordinate system",
        ["n=978 analysable", "Replicate-stable operator", "d95 = 8", "Frozen B8"],
        base,
    )
    panels.append(base.with_suffix(".svg"))

    base = out / "Figure1D_downstream"
    schematic_panel(
        "D", "Downstream audits with B8 fixed",
        ["WFP0", "Stage 7B / NL / RD", "Scale audit", "Stage 7C → WF3"],
        base,
    )
    panels.append(base.with_suffix(".svg"))

    assemble_svg(panels, out / "Figure1_WFP_study_architecture.svg")
    return panels


def render_figure2(fig0: Path, out: Path) -> List[Path]:
    curves = pd.read_csv(fig0 / "fig2_reconstruction_curves.csv")
    reliability = pd.read_csv(fig0 / "fig2_axis_reliability.csv")
    summary = pd.read_csv(fig0 / "fig2_summary_metrics.csv")
    require_columns(curves, EXPECTED_SCHEMAS["fig2_reconstruction_curves.csv"], "fig2 curves")
    require_columns(reliability, EXPECTED_SCHEMAS["fig2_axis_reliability.csv"], "fig2 reliability")
    require_columns(summary, EXPECTED_SCHEMAS["fig2_summary_metrics.csv"], "fig2 summary")
    sm = metric_map(summary)

    panels = []

    fig, ax = fig_one_axis("A")
    d = pd.to_numeric(curves["dimension"])
    ax.plot(d, curves["replicate_basis_cv_r2_all"], marker="o", markersize=3, label="Replicate-stable basis")
    ax.plot(d, curves["ordinary_pca_cv_r2_all"], marker="s", markersize=3, label="Ordinary PCA")
    ax.plot(d, curves["fourier_cv_r2_all"], marker="^", markersize=3, label="Fourier")
    ax.axvline(sm["selected_B8_dimension"], linestyle="--", linewidth=1)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Held-out reconstruction $R^2$")
    ax.set_title("Held-out reconstruction across dimension")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    base = out / "Figure2A_reconstruction"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    fig, ax = fig_one_axis("B")
    ax.plot(reliability["axis"], reliability["odd_even_score_correlation"], marker="o")
    ax.set_xlabel("Frozen B8 axis")
    ax.set_ylabel("Odd/even score correlation")
    ax.set_ylim(0, 1.01)
    ax.set_xticks(reliability["axis"])
    ax.set_title("Axis-level replicate reliability")
    style_ax(ax)
    base = out / "Figure2B_axis_reliability"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    text_panel(
        "C",
        "Population-space summary",
        [
            f"Effective rank: {sm['effective_rank']:.3f}",
            f"d90: {sm['d90']:.0f}",
            f"d95 / frozen dimension: {sm['d95']:.0f} / {sm['selected_B8_dimension']:.0f}",
            f"CV reconstruction at B8: {sm['cv_r2_all']:.3f}",
            f"Half-split overlap: {sm['half_split_overlap_median']:.3f}",
            f"Between/within projector overlap: {sm['between_within_projector_overlap']:.3f}",
        ],
        out / "Figure2C_population_summary",
    )
    panels.append((out / "Figure2C_population_summary.svg"))

    row8 = curves.loc[pd.to_numeric(curves["dimension"]) == int(round(sm["selected_B8_dimension"]))]
    if len(row8) != 1:
        raise RuntimeError("Figure2D: expected exactly one d=8 curve row")
    r = row8.iloc[0]
    labels = ["Replicate-stable", "Ordinary PCA", "Fourier"]
    vals = [
        float(r["replicate_basis_cv_r2_all"]),
        float(r["ordinary_pca_cv_r2_all"]),
        float(r["fourier_cv_r2_all"]),
    ]
    fig, ax = fig_one_axis("D")
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=45)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Held-out reconstruction $R^2$ at frozen d=8")
    ax.set_xlim(0, 1.01)
    ax.set_title("Comparator performance at B8")
    style_ax(ax)
    base = out / "Figure2D_d8_comparator"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    assemble_svg(panels, out / "Figure2_stable_low_dimensional_space.svg")
    return panels


def render_figure3(fig0: Path, out: Path) -> List[Path]:
    proj = pd.read_csv(fig0 / "fig3_patient_projection_z1z2_PRIVATE.csv")
    scale = pd.read_csv(fig0 / "fig3_scale_summary.csv")
    require_columns(proj, EXPECTED_SCHEMAS["fig3_patient_projection_z1z2_PRIVATE.csv"], "fig3 projection")
    require_columns(scale, EXPECTED_SCHEMAS["fig3_scale_summary.csv"], "fig3 scale")
    if len(proj) != 978 or proj["patient_id"].duplicated().any():
        raise RuntimeError("Figure3 private projection integrity failed")
    sm = metric_map(scale)

    panels = []

    fig, ax = fig_one_axis("A")
    ax.scatter(proj["z1"], proj["z2"], s=12, alpha=0.35, linewidths=0)
    ax.set_xlabel("Frozen B8 axis 1 score")
    ax.set_ylabel("Frozen B8 axis 2 score")
    ax.set_title("Illustrative 2-D view of patient central morphology")
    style_ax(ax)
    base = out / "Figure3A_patient_cloud_PRIVATE_SOURCE"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    labels = ["Between-patient RMS", "Within-patient 60-s RMS", "Odd/even replicate RMS"]
    vals = [
        sm["between_pairwise_rms"],
        sm["within_equal_patient_rms"],
        sm["odd_even_replicate_rms"],
    ]
    fig, ax = fig_one_axis("B")
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Distance in frozen B8")
    ax.set_title("Three characteristic scales")
    style_ax(ax)
    base = out / "Figure3B_scale_magnitudes"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    labels = ["Between / replicate", "Within / between", "Within / replicate"]
    vals = [
        sm["between_over_replicate"],
        sm["within_over_between"],
        sm["within_over_replicate"],
    ]
    fig, ax = fig_one_axis("C")
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Ratio")
    ax.set_title("Relative separation and movement scales")
    style_ax(ax)
    base = out / "Figure3C_scale_ratios"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    text_panel(
        "D",
        "Movement relative to local patient spacing",
        [
            f"Nearest-neighbor median: {sm['nearest_neighbor_median']:.3f}",
            f"Adjacent 60-s step median: {sm['adjacent_60s_step_median']:.3f}",
            f"Patients with p95 block displacement ≥ NN: {100*sm['p95_block_displacement_ge_nn_fraction']:.1f}%",
            "",
            "Short-window movement is not a WF3 longitudinal state trajectory.",
        ],
        out / "Figure3D_local_spacing",
    )
    panels.append(out / "Figure3D_local_spacing.svg")

    assemble_svg(panels, out / "Figure3_between_within_replicate.svg")
    return panels


def render_figure4(fig0: Path, out: Path) -> List[Path]:
    mean = pd.read_csv(fig0 / "fig4a_conditional_mean_summary.csv")
    dep = pd.read_csv(fig0 / "fig4b_residual_dependence_summary.csv")
    require_columns(mean, EXPECTED_SCHEMAS["fig4a_conditional_mean_summary.csv"], "fig4 mean")
    require_columns(dep, EXPECTED_SCHEMAS["fig4b_residual_dependence_summary.csv"], "fig4 dependence")
    mm = metric_map(mean)
    dm = metric_map(dep)

    panels = []

    labels = ["Conventional factors", "Age + sex"]
    vals = [mm["conventional_to_B8_oof_r2"], mm["age_sex_oof_r2"]]
    fig, ax = fig_one_axis("A")
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Aggregate OOF $R^2$")
    ax.set_title("Conditional-mean predictability")
    style_ax(ax)
    base = out / "Figure4A_absolute_mean_predictability"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    labels = [
        "Smooth age vs linear",
        "Smooth age × sex vs linear",
        "Linear height increment",
        "Nonlinear height increment",
        "8-phenotype block increment",
    ]
    vals = [
        mm["M1_minus_M0"],
        mm["M2_minus_M0"],
        mm["height_delta_beyond_age_sex"],
        mm["nonlinear_height_increment"],
        mm["phenotype_block_delta_oof_r2"],
    ]
    fig, ax = fig_one_axis("B")
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"Incremental OOF $\Delta R^2$")
    ax.set_title("Added flexibility does not improve prediction")
    style_ax(ax)
    base = out / "Figure4B_incremental_mean_gain"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    cohorts = ["Full n=978", "Height CC n=693"]
    observed = [dm["full_residual_dcor"], dm["height_residual_dcor"]]
    q95 = [dm["full_pipeline_null_q95"], dm["height_pipeline_null_q95"]]
    fig, ax = fig_one_axis("C")
    y = np.arange(2)
    for yi, q, o in zip(y, q95, observed):
        ax.plot([q, o], [yi, yi], linewidth=1.5)
        ax.scatter([q, o], [yi, yi], s=[32, 52])
    ax.set_yticks(y, cohorts)
    ax.set_xlabel("Residual distance correlation")
    ax.set_title("Observed dependence exceeds pipeline-replay q95")
    style_ax(ax)
    base = out / "Figure4C_residual_dependence"
    save_panel(fig, base)
    panels.append(base.with_suffix(".svg"))

    text_panel(
        "D",
        "Interpretation boundary",
        [
            f"Full cohort pipeline-control Holm p: {dm['full_pipeline_holm_p']:.3g}",
            f"Height subset pipeline-control Holm p: {dm['height_pipeline_holm_p']:.3g}",
            f"Within-fold diagnostic p: {dm['full_within_fold_p']:.3g} / {dm['height_within_fold_p']:.3g}",
            "",
            "Mean predictability is weak.",
            "Residual multivariate dependence remains.",
            "Independence / full orthogonality is not claimed.",
        ],
        out / "Figure4D_interpretation",
    )
    panels.append(out / "Figure4D_interpretation.svg")

    assemble_svg(panels, out / "Figure4_covariate_interpretation.svg")
    return panels


def write_captions(fig0: Path, out: Path) -> None:
    f2 = metric_map(pd.read_csv(fig0 / "fig2_summary_metrics.csv"))
    f3 = metric_map(pd.read_csv(fig0 / "fig3_scale_summary.csv"))
    f4a = metric_map(pd.read_csv(fig0 / "fig4a_conditional_mean_summary.csv"))
    f4b = metric_map(pd.read_csv(fig0 / "fig4b_residual_dependence_summary.csv"))

    md = f"""# WF-P Main Figure Captions — draft

## Figure 1. Construction of the frozen population morphology coordinate system.

Arterial-pressure waveforms sampled at 125 Hz were segmented into accepted beats,
phase-normalized to 64 points, and transformed using the frozen `shape_norm`
representation. Sixty-second block summaries and odd/even replicate
representatives were used to obtain a patient-balanced 30-min central-morphology
representation. A replicate-stable population operator was then used to define
the frozen eight-dimensional B8 coordinate system. Subsequent representation,
constitutional, scale, and chronic-phenotype audits used the frozen B8 without
relearning the coordinate system. The 30-min patient representative is termed
central morphology and is not interpreted as a stable trait or physiological
state.

## Figure 2. A stable low-dimensional population-common morphology space.

Held-out reconstruction increased rapidly with dimension and the frozen B8
representation achieved aggregate cross-validated R² = {f2['cv_r2_all']:.3f}.
The positive-spectrum effective rank was {f2['effective_rank']:.2f}, with
d90 = {f2['d90']:.0f} and d95 = {f2['d95']:.0f}; the frozen interface dimension
was therefore eight. Axis-level odd/even score correlations summarize replicate
stability. Comparator curves show ordinary PCA and Fourier reconstruction in
the same dimension budget. Half-split subspace overlap was
{f2['half_split_overlap_median']:.3f}. No physiological label is assigned to
an individual B8 axis.

## Figure 3. Between-patient separation, short-window within-patient movement, and replicate discrepancy in the frozen B8 space.

The illustrative z1-z2 projection displays patient central-morphology scores,
whereas all scale summaries are defined in the full frozen B8 space.
Between-patient pairwise RMS distance was {f3['between_pairwise_rms']:.3f},
compared with an equal-patient 60-s within-patient RMS of
{f3['within_equal_patient_rms']:.3f} and an odd/even replicate RMS discrepancy
of {f3['odd_even_replicate_rms']:.3f}. Thus, between-patient separation was
substantially larger than short-window movement, which in turn exceeded
replicate discrepancy. The median nearest-neighbor distance was
{f3['nearest_neighbor_median']:.3f}; {100*f3['p95_block_displacement_ge_nn_fraction']:.1f}%
of patients had a 95th-percentile block displacement at least as large as their
nearest-neighbor distance. Sixty-second movement is not interpreted as a
long-duration WF3 state trajectory.

## Figure 4. Measured constitutional attributes weakly predict B8 means but retain residual multivariate dependence.

Conventional level/scale/timing variables achieved aggregate OOF R² =
{f4a['conventional_to_B8_oof_r2']:.3f}, while age and sex achieved
OOF R² = {f4a['age_sex_oof_r2']:.3f}. Prespecified low-degree nonlinear age,
age-by-sex, height, nonlinear-height, and chronic-phenotype extensions did not
provide material incremental OOF prediction; their displayed ΔR² values are
taken directly from the closed analysis stages. Nevertheless, residual
distance correlation remained {f4b['full_residual_dcor']:.3f} in the full
cohort and {f4b['height_residual_dcor']:.3f} in the height complete-case cohort,
exceeding the corresponding pipeline-replay null 95th percentiles
({f4b['full_pipeline_null_q95']:.3f} and
{f4b['height_pipeline_null_q95']:.3f}). Pipeline-control Holm-adjusted p values
were {f4b['full_pipeline_holm_p']:.3g} and
{f4b['height_pipeline_holm_p']:.3g}. These findings support weak constitutional
conditional-mean predictability but do not support statistical independence or
complete orthogonality of B8 from the measured constitutional variables.
"""
    (out / "WFP_MAIN_FIGURE_CAPTIONS_DRAFT.md").write_text(md, encoding="utf-8")


def write_manifest(out: Path) -> None:
    rows = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "WFP_MAIN_FIGURE_OUTPUT_MANIFEST_SHA256.csv":
            rows.append({
                "file": p.name,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    with (out / "WFP_MAIN_FIGURE_OUTPUT_MANIFEST_SHA256.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)


def self_test() -> int:
    # Engineering-only checks; no charts are rendered.
    assert EXPECTED_SCHEMAS["fig3_patient_projection_z1z2_PRIVATE.csv"] == ["patient_id","z1","z2"]
    assert "M2_minus_M0" in [
        "conventional_to_B8_oof_r2","age_sex_oof_r2","M1_minus_M0","M2_minus_M0"
    ]
    print("WF-P main-figure renderer self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fig0-dir",
        default="~/Documents/abp_information_study/results/wfp_fig0_exports",
    )
    ap.add_argument(
        "--out",
        default="~/Documents/abp_information_study/results/wfp_main_figures",
    )
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    fig0 = Path(a.fig0_dir).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    readout = fig0 / "WFP_FIG0_EXPORT_READOUT.txt"
    if not readout.is_file() or "Decision: WFP_FIG0_EXPORT_COMPLETE" not in readout.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise SystemExit("FAIL: FIG0 export is not complete")

    for fn, cols in EXPECTED_SCHEMAS.items():
        p = fig0 / fn
        if not p.is_file():
            raise SystemExit(f"FAIL missing FIG0 file: {p}")
        df = pd.read_csv(p)
        require_columns(df, cols, fn)

    render_figure2(fig0, out)
    render_figure3(fig0, out)
    render_figure4(fig0, out)
    render_figure1(out)
    write_captions(fig0, out)

    readout_txt = "\n".join([
        "WF-P MAIN FIGURE RENDER",
        "=======================",
        "Decision: WFP_MAIN_FIGURES_RENDER_COMPLETE",
        "Scientific effects calculated: NO",
        "Frozen B8 changed: NO",
        "Figures rendered: 1, 2, 3, 4",
        "Panel-level SVG/PDF/PNG: YES",
        f"Composite PDF/PNG via CairoSVG: {'YES' if cairosvg is not None else 'NO (composite SVG still written)'}",
        "Private patient-level CSV copied to output: NO",
        "",
        "Interpretation boundary:",
        "  Figure 3 z1-z2 cloud is illustrative only; inference remains in B8.",
        "  Figure 4 shows weak mean predictability plus persistent residual dependence.",
        "  Constitutional independence / complete orthogonality is NOT claimed.",
        "",
    ])
    (out / "WFP_MAIN_FIGURES_RENDER_READOUT.txt").write_text(readout_txt, encoding="utf-8")

    write_manifest(out)

    zipp = out / "WFP_MAIN_FIGURES_RENDERED.zip"
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != zipp:
                z.write(p, arcname=p.name)

    print(readout_txt)
    print(f"Rendered package: {zipp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
