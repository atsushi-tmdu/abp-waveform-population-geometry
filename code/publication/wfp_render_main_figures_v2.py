#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Main Figures 1–4 — visual-QC revision v2
=============================================

Rendering-only revision based on visual review of the first rendered package.

No scientific analysis is performed. No scientific metric is added or changed.

Presentation changes from v1:
- Figure 1 becomes one integrated workflow schematic to eliminate box/arrow
  collisions and excessive blank space.
- Figure 2 replaces the text-only summary panel with a stability-metric dot plot
  and uses a deliberately zoomed reliability axis, explicitly labeled.
- Figure 3 uses a logarithmic scale for heterogeneous scale ratios and converts
  the text-heavy local-spacing panel into a graphical summary.
- Figure 4 uses lollipop-style mean-predictability displays, separates observed
  residual dCor from pipeline-null q95 with distinct marker shapes, and replaces
  the text-only final panel with an interpretation flow diagram.
- Composite PNG and PDF are always generated from high-resolution panel PNGs
  using Pillow; composite SVG is also written. CairoSVG remains an optional
  vector-PDF path when available.

Scientific boundaries:
- B8 remains frozen.
- Figure 3 z1-z2 projection is illustrative only.
- Figure 4 retains: weak conditional-mean predictability + persistent residual
  multivariate dependence; no independence/full-orthogonality claim.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

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

try:
    from PIL import Image
except Exception:
    Image = None


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
PANEL_H = 4.15
DPI = 300


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_map(df: pd.DataFrame) -> Dict[str, float]:
    return {str(r["metric"]): float(r["value"]) for _, r in df.iterrows()}


def require_columns(df: pd.DataFrame, expected: List[str], label: str) -> None:
    if list(df.columns) != expected:
        raise RuntimeError(
            f"{label}: schema mismatch\nexpected={expected}\nobserved={list(df.columns)}"
        )


def style_ax(ax) -> None:
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_letter(fig, letter: str) -> None:
    fig.text(0.015, 0.975, letter, ha="left", va="top",
             fontsize=15, fontweight="bold")


def fig_one_axis(letter: str, left=0.18, bottom=0.16, right=0.95, top=0.91):
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes([left, bottom, right-left, top-bottom])
    panel_letter(fig, letter)
    return fig, ax


def save_panel(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax, xs, ys, fmt="{:.3f}", dx=4) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(x), (x, y), xytext=(dx, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5)


def svg_dimensions(svg_path: Path) -> Tuple[float, float]:
    import re
    txt = svg_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'<svg[^>]*\bwidth="([0-9.]+)pt"[^>]*\bheight="([0-9.]+)pt"', txt
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    return 432.0, 302.4


def svg_data_uri(path: Path) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def assemble_svg(panel_svgs: List[Path], out_svg: Path, ncols=2, gutter=18.0) -> None:
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
        href = svg_data_uri(p)
        pieces.append(
            f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'href="{href}" xlink:href="{href}"/>'
        )
    pieces.append("</svg>")
    out_svg.write_text("\n".join(pieces), encoding="utf-8")

    if cairosvg is not None:
        cairosvg.svg2pdf(bytestring=out_svg.read_bytes(),
                         write_to=str(out_svg.with_suffix(".vector.pdf")))


def assemble_raster(panel_pngs: List[Path], out_png: Path, ncols=2, gutter=40) -> None:
    if Image is None:
        return
    ims = [Image.open(p).convert("RGB") for p in panel_pngs]
    cell_w = max(im.width for im in ims)
    cell_h = max(im.height for im in ims)
    nrows = math.ceil(len(ims) / ncols)
    canvas = Image.new(
        "RGB",
        (ncols * cell_w + (ncols - 1) * gutter,
         nrows * cell_h + (nrows - 1) * gutter),
        255,
    )
    for i, im in enumerate(ims):
        r, c = divmod(i, ncols)
        x = c * (cell_w + gutter) + (cell_w - im.width) // 2
        y = r * (cell_h + gutter) + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    canvas.save(out_png, dpi=(DPI, DPI))
    canvas.save(out_png.with_suffix(".pdf"), "PDF", resolution=DPI)


def assemble(panel_bases: List[Path], composite_base: Path, ncols=2) -> None:
    svgs = [p.with_suffix(".svg") for p in panel_bases]
    pngs = [p.with_suffix(".png") for p in panel_bases]
    assemble_svg(svgs, composite_base.with_suffix(".svg"), ncols=ncols)
    assemble_raster(pngs, composite_base.with_suffix(".png"), ncols=ncols)


def draw_box(ax, xy, wh, label, fontsize=9.2, linewidth=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        fill=False, linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, textwrap.fill(label, 20),
            ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, p1, p2):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->", linewidth=1.1))


def render_figure1(out: Path) -> None:
    fig = plt.figure(figsize=(12.0, 5.4))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.90])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.96, "A", fontsize=15, fontweight="bold", va="top")
    ax.text(0.08, 0.94, "Waveform-to-patient representation",
            fontsize=11.5, fontweight="bold", va="top")

    top_labels = [
        "125-Hz ABP",
        "Beat QC",
        "64-point phase-normalized beat",
        "shape_norm",
        "60-s block summaries",
        "Odd/even replicate representatives",
        "30-min patient central morphology",
    ]
    xs = np.linspace(0.03, 0.84, len(top_labels))
    bw, bh, y = 0.12, 0.16, 0.66
    for i, (x, lab) in enumerate(zip(xs, top_labels)):
        draw_box(ax, (x, y), (bw, bh), lab, fontsize=8.4)
        if i < len(top_labels)-1:
            arrow(ax, (x+bw+0.005, y+bh/2),
                  (xs[i+1]-0.005, y+bh/2))

    ax.text(0.02, 0.53, "B", fontsize=15, fontweight="bold", va="top")
    ax.text(0.08, 0.51, "Population coordinate construction",
            fontsize=11.5, fontweight="bold", va="top")

    mid_labels = [
        "n=978 analysable patients",
        "Replicate-stable population operator",
        "d90 = 6; d95 = 8",
        "Frozen B8 coordinate system",
    ]
    xs2 = [0.12, 0.34, 0.56, 0.78]
    bw2, bh2, y2 = 0.16, 0.15, 0.32
    for i, (x, lab) in enumerate(zip(xs2, mid_labels)):
        draw_box(ax, (x, y2), (bw2, bh2), lab, fontsize=9)
        if i < len(mid_labels)-1:
            arrow(ax, (x+bw2+0.008, y2+bh2/2),
                  (xs2[i+1]-0.008, y2+bh2/2))

    ax.text(0.02, 0.22, "C", fontsize=15, fontweight="bold", va="top")
    ax.text(0.08, 0.20, "Downstream audits with B8 fixed",
            fontsize=11.5, fontweight="bold", va="top")

    branch_labels = [
        "WFP0\nrepresentation identifiability",
        "Stage 7B / 7B-NL / 7B-RD\nconstitutional mapping",
        "Between–within\nscale audit",
        "Stage 7C\nchronic phenotypes",
        "WF3\nlongitudinal trajectories",
    ]
    xs3 = [0.06, 0.25, 0.48, 0.66, 0.84]
    widths = [0.15, 0.20, 0.15, 0.15, 0.12]
    y3, bh3 = 0.03, 0.12
    for x, w, lab in zip(xs3, widths, branch_labels):
        draw_box(ax, (x, y3), (w, bh3), lab, fontsize=8.5)
    arrow(ax, (0.81, y2), (0.36, y3+bh3))
    arrow(ax, (0.81, y2), (0.555, y3+bh3))
    arrow(ax, (0.81, y2), (0.735, y3+bh3))
    arrow(ax, (0.81, y2), (0.90, y3+bh3))

    base = out / "Figure1_WFP_study_architecture"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def render_figure2(fig0: Path, out: Path) -> None:
    curves = pd.read_csv(fig0 / "fig2_reconstruction_curves.csv")
    reliability = pd.read_csv(fig0 / "fig2_axis_reliability.csv")
    summary = pd.read_csv(fig0 / "fig2_summary_metrics.csv")
    require_columns(curves, EXPECTED_SCHEMAS["fig2_reconstruction_curves.csv"], "fig2 curves")
    require_columns(reliability, EXPECTED_SCHEMAS["fig2_axis_reliability.csv"], "fig2 reliability")
    require_columns(summary, EXPECTED_SCHEMAS["fig2_summary_metrics.csv"], "fig2 summary")
    sm = metric_map(summary)
    panel_bases = []

    fig, ax = fig_one_axis("A")
    d = pd.to_numeric(curves["dimension"])
    ax.plot(d, curves["replicate_basis_cv_r2_all"], marker="o", markersize=3,
            label="Replicate-stable basis")
    ax.plot(d, curves["ordinary_pca_cv_r2_all"], marker="s", markersize=3,
            linestyle="--", label="Ordinary PCA")
    ax.plot(d, curves["fourier_cv_r2_all"], marker="^", markersize=3,
            linestyle=":", label="Fourier")
    ax.axvline(sm["selected_B8_dimension"], linestyle="--", linewidth=1)
    ax.text(sm["selected_B8_dimension"]+0.25, 0.36, "Frozen B8", rotation=90,
            va="bottom", fontsize=8.5)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Held-out reconstruction $R^2$")
    ax.set_title("Held-out reconstruction across dimension")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_ax(ax)
    base = out / "Figure2A_reconstruction"
    save_panel(fig, base); panel_bases.append(base)

    fig, ax = fig_one_axis("B")
    ax.plot(reliability["axis"], reliability["odd_even_score_correlation"], marker="o")
    ax.set_xlabel("Frozen B8 axis")
    ax.set_ylabel("Odd/even score correlation")
    ax.set_ylim(0.98, 1.001)
    ax.set_xticks(reliability["axis"])
    ax.set_title("Axis-level replicate reliability")
    ax.text(0.02, 0.04, "Truncated y-axis", transform=ax.transAxes, fontsize=8)
    style_ax(ax)
    base = out / "Figure2B_axis_reliability"
    save_panel(fig, base); panel_bases.append(base)

    labels = [
        "Within-window variance\ncaptured by B8",
        "Between/within\nprojector overlap",
        "Half-split\nsubspace overlap",
    ]
    vals = [
        sm["within_variance_captured_by_between_basis"],
        sm["between_within_projector_overlap"],
        sm["half_split_overlap_median"],
    ]
    fig, ax = fig_one_axis("C", left=0.28)
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    add_value_labels(ax, vals, y)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.88, 1.005)
    ax.set_xlabel("Overlap / captured fraction")
    ax.set_title("Population-space stability")
    ax.text(0.02, 0.04, "Truncated x-axis", transform=ax.transAxes, fontsize=8)
    style_ax(ax)
    base = out / "Figure2C_population_stability"
    save_panel(fig, base); panel_bases.append(base)

    row8 = curves.loc[pd.to_numeric(curves["dimension"]) == int(round(sm["selected_B8_dimension"]))]
    if len(row8) != 1:
        raise RuntimeError("expected exactly one d=8 row")
    r = row8.iloc[0]
    labels = ["Fourier", "Ordinary PCA", "Replicate-stable"]
    vals = [
        float(r["fourier_cv_r2_all"]),
        float(r["ordinary_pca_cv_r2_all"]),
        float(r["replicate_basis_cv_r2_all"]),
    ]
    fig, ax = fig_one_axis("D", left=0.25)
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    add_value_labels(ax, vals, y)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.84, 1.005)
    ax.set_xlabel("Held-out reconstruction $R^2$ at d=8")
    ax.set_title("Comparator performance at frozen dimension")
    ax.text(0.02, 0.04, "Truncated x-axis", transform=ax.transAxes, fontsize=8)
    style_ax(ax)
    base = out / "Figure2D_d8_comparator"
    save_panel(fig, base); panel_bases.append(base)

    assemble(panel_bases, out / "Figure2_stable_low_dimensional_space", ncols=2)


def render_figure3(fig0: Path, out: Path) -> None:
    proj = pd.read_csv(fig0 / "fig3_patient_projection_z1z2_PRIVATE.csv")
    scale = pd.read_csv(fig0 / "fig3_scale_summary.csv")
    require_columns(proj, EXPECTED_SCHEMAS["fig3_patient_projection_z1z2_PRIVATE.csv"], "fig3 projection")
    require_columns(scale, EXPECTED_SCHEMAS["fig3_scale_summary.csv"], "fig3 scale")
    if len(proj) != 978 or proj["patient_id"].duplicated().any():
        raise RuntimeError("Figure3 projection integrity failed")
    sm = metric_map(scale)
    panel_bases = []

    fig, ax = fig_one_axis("A")
    ax.scatter(proj["z1"], proj["z2"], s=11, alpha=0.30, linewidths=0)
    ax.set_xlabel("Frozen B8 axis 1 score")
    ax.set_ylabel("Frozen B8 axis 2 score")
    ax.set_title("Illustrative 2-D view of patient central morphology")
    style_ax(ax)
    base = out / "Figure3A_patient_cloud_PRIVATE_SOURCE"
    save_panel(fig, base); panel_bases.append(base)

    labels = ["Odd/even replicate RMS", "Within-patient 60-s RMS", "Between-patient RMS"]
    vals = [sm["odd_even_replicate_rms"], sm["within_equal_patient_rms"], sm["between_pairwise_rms"]]
    fig, ax = fig_one_axis("B", left=0.30)
    y = np.arange(len(labels))
    for yi, v in zip(y, vals):
        ax.plot([0, v], [yi, yi], linewidth=1)
    ax.scatter(vals, y, s=48)
    add_value_labels(ax, vals, y)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Distance in frozen B8")
    ax.set_title("Three characteristic scales")
    style_ax(ax)
    base = out / "Figure3B_scale_magnitudes"
    save_panel(fig, base); panel_bases.append(base)

    labels = ["Within / between", "Within / replicate", "Between / replicate"]
    vals = [sm["within_over_between"], sm["within_over_replicate"], sm["between_over_replicate"]]
    fig, ax = fig_one_axis("C", left=0.30)
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=48)
    add_value_labels(ax, vals, y)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 30)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Ratio (log scale)")
    ax.set_title("Relative separation and movement scales")
    style_ax(ax)
    base = out / "Figure3C_scale_ratios"
    save_panel(fig, base); panel_bases.append(base)

    labels = ["Adjacent 60-s step median", "Nearest-neighbor median"]
    vals = [sm["adjacent_60s_step_median"], sm["nearest_neighbor_median"]]
    fig, ax = fig_one_axis("D", left=0.31)
    y = np.array([0.35, 0.75])
    for yi, v in zip(y, vals):
        ax.plot([0, v], [yi, yi], linewidth=1)
    ax.scatter(vals, y, s=48)
    add_value_labels(ax, vals, y)
    ax.set_yticks(y, labels)
    ax.set_ylim(0.05, 1.15)
    ax.set_xlabel("Distance in frozen B8")
    ax.set_title("Movement relative to local patient spacing")
    ax.text(
        0.98, 0.15,
        f"{100*sm['p95_block_displacement_ge_nn_fraction']:.1f}%\n"
        "of patients had p95 block\n"
        "displacement ≥ nearest-neighbor distance",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.2,
    )
    style_ax(ax)
    base = out / "Figure3D_local_spacing"
    save_panel(fig, base); panel_bases.append(base)

    assemble(panel_bases, out / "Figure3_between_within_replicate", ncols=2)


def interpretation_diagram(base: Path, dm: Dict[str, float]) -> None:
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.86])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_letter(fig, "D")
    ax.text(0.5, 0.94, "Interpretation boundary", ha="center", va="top",
            fontsize=11.5, fontweight="bold")

    draw_box(
        ax, (0.05, 0.57), (0.39, 0.24),
        "Conditional mean\nLinear / spline / limited interactions\n→ weak OOF predictability",
        fontsize=9.3
    )
    draw_box(
        ax, (0.56, 0.57), (0.39, 0.24),
        "Residual dependence\nObserved dCor exceeds artifact-control null\n→ dependence remains",
        fontsize=9.3
    )
    draw_box(
        ax, (0.25, 0.15), (0.50, 0.21),
        "Final interpretation\nWeak mean predictability ≠ independence\n"
        "Do not claim full orthogonality",
        fontsize=9.6, linewidth=1.4
    )
    arrow(ax, (0.245, 0.56), (0.40, 0.37))
    arrow(ax, (0.755, 0.56), (0.60, 0.37))
    ax.text(
        0.5, 0.06,
        f"Pipeline-control Holm p={dm['full_pipeline_holm_p']:.3g} (full) and "
        f"{dm['height_pipeline_holm_p']:.3g} (height complete-case)",
        ha="center", va="center", fontsize=8.3,
    )
    save_panel(fig, base)


def render_figure4(fig0: Path, out: Path) -> None:
    mean = pd.read_csv(fig0 / "fig4a_conditional_mean_summary.csv")
    dep = pd.read_csv(fig0 / "fig4b_residual_dependence_summary.csv")
    require_columns(mean, EXPECTED_SCHEMAS["fig4a_conditional_mean_summary.csv"], "fig4 mean")
    require_columns(dep, EXPECTED_SCHEMAS["fig4b_residual_dependence_summary.csv"], "fig4 dependence")
    mm = metric_map(mean); dm = metric_map(dep)
    panel_bases = []

    labels = ["Age + sex", "Conventional factors"]
    vals = [mm["age_sex_oof_r2"], mm["conventional_to_B8_oof_r2"]]
    fig, ax = fig_one_axis("A", left=0.28)
    y = np.arange(len(labels))
    for yi, v in zip(y, vals):
        ax.plot([0, v], [yi, yi], linewidth=1)
    ax.scatter(vals, y, s=48)
    add_value_labels(ax, vals, y)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.15)
    ax.set_xlabel("Aggregate OOF $R^2$")
    ax.set_title("Conditional-mean predictability")
    style_ax(ax)
    base = out / "Figure4A_absolute_mean_predictability"
    save_panel(fig, base); panel_bases.append(base)

    labels = [
        "Smooth age × sex vs linear",
        "8-phenotype block",
        "Smooth age vs linear",
        "Nonlinear height",
        "Linear height",
    ]
    vals = [
        mm["M2_minus_M0"],
        mm["phenotype_block_delta_oof_r2"],
        mm["M1_minus_M0"],
        mm["nonlinear_height_increment"],
        mm["height_delta_beyond_age_sex"],
    ]
    fig, ax = fig_one_axis("B", left=0.34)
    y = np.arange(len(labels))
    for yi, v in zip(y, vals):
        ax.plot([v, 0], [yi, yi], linewidth=1)
    ax.scatter(vals, y, s=48)
    ax.axvline(0, linestyle="--", linewidth=1)
    add_value_labels(ax, vals, y, fmt="{:.4f}", dx=5)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.012, 0.001)
    ax.set_xlabel(r"Incremental OOF $\Delta R^2$")
    ax.set_title("Added flexibility does not improve prediction")
    style_ax(ax)
    base = out / "Figure4B_incremental_mean_gain"
    save_panel(fig, base); panel_bases.append(base)

    cohorts = ["Full n=978", "Height CC n=693"]
    obs = [dm["full_residual_dcor"], dm["height_residual_dcor"]]
    q95 = [dm["full_pipeline_null_q95"], dm["height_pipeline_null_q95"]]
    fig, ax = fig_one_axis("C", left=0.27)
    y = np.arange(len(cohorts))
    for yi, q, o in zip(y, q95, obs):
        ax.plot([q, o], [yi, yi], linestyle=":", linewidth=1)
    ax.scatter(q95, y, marker="x", s=58, label="Pipeline-null 95th percentile")
    ax.scatter(obs, y, marker="o", s=48, label="Observed residual dCor")
    ax.set_yticks(y, cohorts)
    ax.set_xlim(0.08, 0.19)
    ax.set_xlabel("Residual distance correlation")
    ax.set_title("Residual dependence persists after pipeline control")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_ax(ax)
    base = out / "Figure4C_residual_dependence"
    save_panel(fig, base); panel_bases.append(base)

    base = out / "Figure4D_interpretation"
    interpretation_diagram(base, dm); panel_bases.append(base)

    assemble(panel_bases, out / "Figure4_covariate_interpretation", ncols=2)


def write_captions(fig0: Path, out: Path) -> None:
    f2 = metric_map(pd.read_csv(fig0 / "fig2_summary_metrics.csv"))
    f3 = metric_map(pd.read_csv(fig0 / "fig3_scale_summary.csv"))
    f4a = metric_map(pd.read_csv(fig0 / "fig4a_conditional_mean_summary.csv"))
    f4b = metric_map(pd.read_csv(fig0 / "fig4b_residual_dependence_summary.csv"))

    md = f"""# WF-P Main Figure Captions — visual-QC revision v2

## Figure 1. Construction of the frozen population morphology coordinate system.

Arterial-pressure waveforms sampled at 125 Hz were segmented into accepted
beats, phase-normalized to 64 points, and transformed using the frozen
`shape_norm` representation. Sixty-second block summaries and odd/even
replicate representatives were used to obtain a patient-balanced 30-min
central-morphology representation. A replicate-stable population operator was
then used to define the frozen eight-dimensional B8 coordinate system.
Subsequent representation, constitutional, scale, and chronic-phenotype audits
used the frozen B8 without relearning the coordinate system. The 30-min patient
representative is termed central morphology and is not interpreted as a stable
trait or physiological state.

## Figure 2. A stable low-dimensional population-common morphology space.

Held-out reconstruction increased rapidly with dimension and the frozen B8
representation achieved aggregate cross-validated R² = {f2['cv_r2_all']:.3f}.
The positive-spectrum effective rank was {f2['effective_rank']:.2f}, with
d90 = {f2['d90']:.0f} and d95 = {f2['d95']:.0f}. Axis-level odd/even score
correlations summarize replicate stability; the reliability panel uses a
truncated y-axis to show differences near unity. Population-space stability is
summarized by within-window variance captured by the between-person basis,
between/within projector overlap, and half-split subspace overlap. Comparator
curves and the frozen-dimension panel show ordinary PCA and Fourier
reconstruction under the same dimension budget. No physiological label is
assigned to an individual B8 axis.

## Figure 3. Between-patient separation, short-window within-patient movement, and replicate discrepancy in the frozen B8 space.

The illustrative z1-z2 projection displays patient central-morphology scores,
whereas all scale summaries are defined in the full frozen B8 space.
Between-patient pairwise RMS distance was {f3['between_pairwise_rms']:.3f},
compared with an equal-patient 60-s within-patient RMS of
{f3['within_equal_patient_rms']:.3f} and an odd/even replicate RMS discrepancy
of {f3['odd_even_replicate_rms']:.3f}. The ratio panel uses a logarithmic
horizontal scale because the three prespecified ratios differ by more than two
orders of magnitude. The median nearest-neighbor distance was
{f3['nearest_neighbor_median']:.3f}; the median adjacent 60-s step was
{f3['adjacent_60s_step_median']:.3f}; and
{100*f3['p95_block_displacement_ge_nn_fraction']:.1f}% of patients had a
95th-percentile block displacement at least as large as their nearest-neighbor
distance. Sixty-second movement is not interpreted as a long-duration WF3
state trajectory.

## Figure 4. Measured constitutional attributes weakly predict B8 means but retain residual multivariate dependence.

Conventional level/scale/timing variables achieved aggregate OOF R² =
{f4a['conventional_to_B8_oof_r2']:.3f}, while age and sex achieved OOF R² =
{f4a['age_sex_oof_r2']:.3f}. Prespecified low-degree nonlinear age,
age-by-sex, height, nonlinear-height, and chronic-phenotype extensions did not
provide material incremental OOF prediction. Residual distance correlation
remained {f4b['full_residual_dcor']:.3f} in the full cohort and
{f4b['height_residual_dcor']:.3f} in the height complete-case cohort, exceeding
the corresponding pipeline-replay null 95th percentiles
({f4b['full_pipeline_null_q95']:.3f} and
{f4b['height_pipeline_null_q95']:.3f}). Pipeline-control Holm-adjusted p values
were {f4b['full_pipeline_holm_p']:.3g} and
{f4b['height_pipeline_holm_p']:.3g}. These findings support weak constitutional
conditional-mean predictability but do not support statistical independence or
complete orthogonality of B8 from the measured constitutional variables.
"""
    (out / "WFP_MAIN_FIGURE_CAPTIONS_DRAFT_v2.md").write_text(md, encoding="utf-8")


def write_manifest(out: Path) -> None:
    rows = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "WFP_MAIN_FIGURE_OUTPUT_MANIFEST_SHA256.csv":
            rows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    with (out / "WFP_MAIN_FIGURE_OUTPUT_MANIFEST_SHA256.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes", "sha256"])
        w.writeheader(); w.writerows(rows)


def self_test() -> int:
    assert EXPECTED_SCHEMAS["fig3_patient_projection_z1z2_PRIVATE.csv"] == ["patient_id","z1","z2"]
    assert Image is None or hasattr(Image, "new")
    print("WF-P main-figure v2 renderer self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fig0-dir",
        default="~/Documents/abp_information_study/results/wfp_fig0_exports",
    )
    ap.add_argument(
        "--out",
        default="~/Documents/abp_information_study/results/wfp_main_figures_v2",
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
        require_columns(pd.read_csv(p), cols, fn)

    render_figure1(out)
    render_figure2(fig0, out)
    render_figure3(fig0, out)
    render_figure4(fig0, out)
    write_captions(fig0, out)

    readout_txt = "\n".join([
        "WF-P MAIN FIGURE RENDER v2",
        "==========================",
        "Decision: WFP_MAIN_FIGURES_V2_RENDER_COMPLETE",
        "Scientific effects calculated: NO",
        "Frozen B8 changed: NO",
        "Figures rendered: 1, 2, 3, 4",
        "Composite SVG: YES",
        f"Composite PNG/PDF via Pillow: {'YES' if Image is not None else 'NO'}",
        f"Optional composite vector PDF via CairoSVG: {'YES' if cairosvg is not None else 'NO'}",
        "Private patient-level CSV copied to output: NO",
        "",
        "Visual-QC revisions:",
        "  Figure 1 integrated workflow; collisions removed.",
        "  Figure 2 stability metrics graphical; reliability axis explicitly truncated.",
        "  Figure 3 ratio panel logarithmic; local-spacing panel graphical.",
        "  Figure 4 observed dCor separated from null q95; interpretation diagram added.",
        "",
        "Interpretation boundary unchanged:",
        "  Figure 3 z1-z2 cloud is illustrative only.",
        "  Figure 4 = weak mean predictability + persistent residual dependence.",
        "  Constitutional independence / complete orthogonality is NOT claimed.",
        "",
    ])
    (out / "WFP_MAIN_FIGURES_V2_RENDER_READOUT.txt").write_text(readout_txt, encoding="utf-8")

    write_manifest(out)

    zipp = out / "WFP_MAIN_FIGURES_V2_RENDERED.zip"
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != zipp:
                z.write(p, arcname=p.name)

    print(readout_txt)
    print(f"Rendered package: {zipp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
