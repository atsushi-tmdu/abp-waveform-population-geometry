#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P FIG0 — Figure-ready data export
====================================

Purpose
-------
Create figure-ready tables from the *closed* WF-P result set.

This stage is NOT a new scientific analysis.
It only:
- verifies WF-P final closeout;
- copies / reshapes already-frozen derived outputs;
- subsets the authoritative summary into figure-specific tables.

No new regression / PCA / hypothesis test / eigendecomposition is run here.
No raw waveform arrays are opened.
No new scientific metric is created.

Primary deliverables
--------------------
Main Figure 1:
- architecture panel text / source manifest only (schematic figure)

Main Figure 2:
- reconstruction curves
- Fourier comparator curve
- axis reliability table
- selected summary metrics

Main Figure 3:
- illustrative patient z1-z2 projection
- scale summary metrics

Main Figure 4:
- conditional-mean predictability summary
- residual dependence summary
- phenotype block summary

Boundary
--------
- B8 is not relearned or relabeled.
- z1/z2 cloud is an illustrative projection only.
- No claim of constitutional independence is supported.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, re, shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_summary_map(path: Path) -> Dict[str, dict]:
    df = pd.read_csv(path)
    req = {"section", "metric", "value", "unit", "role", "source", "note"}
    if set(df.columns) != req:
        raise RuntimeError(f"Unexpected closeout summary columns: {df.columns.tolist()}")
    out = {}
    for _, r in df.iterrows():
        out[str(r["metric"])] = dict(r)
    return out


def fetch(sm: Dict[str, dict], metric: str) -> dict:
    if metric not in sm:
        raise RuntimeError(f"Metric absent from authoritative summary: {metric}")
    return sm[metric]


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def self_test() -> int:
    # Minimal shape checks only.
    tmp = pd.DataFrame({
        "section": ["a"], "metric": ["m"], "value": [1.0], "unit": ["u"],
        "role": ["r"], "source": ["s"], "note": [""],
    })
    p = Path(".__wfp_fig0_selftest.csv")
    tmp.to_csv(p, index=False)
    sm = read_summary_map(p)
    assert fetch(sm, "m")["value"] == 1.0
    p.unlink()
    print("WF-P FIG0 export self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--spec")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_fig0_exports")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if a.spec is None:
        raise SystemExit("Missing required --spec")

    root = Path(a.project_root).expanduser().resolve()
    specp = Path(a.spec).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(specp.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_FIGURE_READY_EXPORT":
        raise SystemExit("FAIL invalid FIG0 frozen spec status")

    scriptp = Path(__file__).resolve()
    if spec.get("export_script_sha256") != sha256_file(scriptp):
        raise SystemExit("FAIL export script hash mismatch")

    closeout_summary = root / "results" / "wfp_final_closeout" / "WFP_AUTHORITATIVE_RESULTS_SUMMARY.csv"
    fig_manifest = root / "results" / "wfp_final_closeout" / "WFP_FIGURE_MANIFEST_v0_1.md"
    closeout_readout = root / "results" / "wfp_final_closeout" / "WFP_FINAL_CLOSEOUT_READOUT.txt"

    discovery_curve = root / "results" / "wfp_discovery_validation1000" / "wfp_cv_reconstruction_curve.csv"
    fourier_curve = root / "results" / "wfp_discovery_validation1000" / "wfp_fourier_comparator_curve.csv"
    axis_reliability = root / "results" / "wfp_discovery_validation1000" / "wfp_axis_reliability.csv"
    patient_scores = root / "results" / "wfp_discovery_validation1000" / "wfp_patient_scores_DISCOVERY_PRIVATE.csv"

    required = {
        "closeout_summary_sha256": closeout_summary,
        "figure_manifest_sha256": fig_manifest,
        "closeout_readout_sha256": closeout_readout,
        "discovery_curve_sha256": discovery_curve,
        "fourier_curve_sha256": fourier_curve,
        "axis_reliability_sha256": axis_reliability,
        "patient_scores_sha256": patient_scores,
    }
    for key, path in required.items():
        if not path.is_file():
            raise SystemExit(f"FAIL missing source file: {path}")
        if spec.get(key) != sha256_file(path):
            raise SystemExit(f"FAIL frozen source hash mismatch: {key}")

    if "Decision: WFP_FINAL_CLOSEOUT_COMPLETE" not in read_text(closeout_readout):
        raise SystemExit("FAIL closeout decision marker absent")

    sm = read_summary_map(closeout_summary)

    # ---------- Figure 1 (schematic text only) ----------
    fig1_text = [
        "# WF-P Main Figure 1 source notes",
        "",
        "Panel A: 125-Hz ABP -> accepted beats -> 64-point shape_norm.",
        "Panel B: 60-s blocks -> odd/even replicate representatives -> 30-min patient central morphology.",
        "Panel C: replicate-stable population operator -> frozen B8.",
        "Panel D: downstream audits = WFP0, Stage7B/7B-NL/7B-RD, scale audit, Stage7C.",
        "",
        "Caption boundary:",
        "- one 30-min representative is central morphology, not trait/state;",
        "- B8 is not relearned in downstream audits.",
    ]
    (out / "fig1_schematic_panel_text.md").write_text("\n".join(fig1_text) + "\n", encoding="utf-8")

    # ---------- Figure 2 ----------
    rcurve = pd.read_csv(discovery_curve)
    fcurve = pd.read_csv(fourier_curve)
    ar = pd.read_csv(axis_reliability)

    merged = rcurve.merge(fcurve, on="dimension", how="left", validate="one_to_one")
    merged.to_csv(out / "fig2_reconstruction_curves.csv", index=False)
    ar.to_csv(out / "fig2_axis_reliability.csv", index=False)

    fig2_rows = []
    for key in [
        "effective_rank", "d90", "d95", "selected_B8_dimension", "cv_r2_all",
        "half_split_overlap_median", "within_variance_captured_by_between_basis",
        "between_within_projector_overlap",
    ]:
        rec = fetch(sm, key)
        fig2_rows.append({
            "metric": key,
            "value": rec["value"],
            "unit": rec["unit"],
            "role": rec["role"],
            "source": rec["source"],
            "note": rec["note"],
        })
    pd.DataFrame(fig2_rows).to_csv(out / "fig2_summary_metrics.csv", index=False)

    # ---------- Figure 3 ----------
    ps = pd.read_csv(patient_scores, dtype={"patient_id": str})
    need = ["patient_id"] + [f"z{i}" for i in range(1, 9)]
    if not set(need).issubset(ps.columns):
        raise RuntimeError("patient score file missing required z1..z8 columns")
    ps[["patient_id", "z1", "z2"]].to_csv(out / "fig3_patient_projection_z1z2_PRIVATE.csv", index=False)

    fig3_rows = []
    for key in [
        "between_pairwise_rms", "between_pairwise_median", "nearest_neighbor_median",
        "within_equal_patient_rms", "adjacent_60s_step_median", "odd_even_replicate_rms",
        "between_over_replicate", "within_over_between", "within_over_replicate",
        "p95_block_displacement_ge_nn_fraction",
    ]:
        rec = fetch(sm, key)
        fig3_rows.append({
            "metric": key,
            "value": rec["value"],
            "unit": rec["unit"],
            "role": rec["role"],
            "source": rec["source"],
            "note": rec["note"],
        })
    pd.DataFrame(fig3_rows).to_csv(out / "fig3_scale_summary.csv", index=False)

    # ---------- Figure 4 ----------
    fig4a_keys = [
        "conventional_to_B8_oof_r2",
        "age_sex_oof_r2",
        "M1_minus_M0",
        "M2_minus_M0",
        "height_delta_beyond_age_sex",
        "nonlinear_height_increment",
        "phenotype_block_delta_oof_r2",
    ]
    fig4a_rows = []
    for key in fig4a_keys:
        rec = fetch(sm, key)
        fig4a_rows.append({
            "metric": key,
            "value": rec["value"],
            "unit": rec["unit"],
            "role": rec["role"],
            "source": rec["source"],
            "note": rec["note"],
        })
    pd.DataFrame(fig4a_rows).to_csv(out / "fig4a_conditional_mean_summary.csv", index=False)

    fig4b_keys = [
        "full_residual_dcor",
        "full_pipeline_null_q95",
        "full_pipeline_holm_p",
        "full_within_fold_p",
        "height_residual_dcor",
        "height_pipeline_null_q95",
        "height_pipeline_holm_p",
        "height_within_fold_p",
    ]
    fig4b_rows = []
    for key in fig4b_keys:
        rec = fetch(sm, key)
        fig4b_rows.append({
            "metric": key,
            "value": rec["value"],
            "unit": rec["unit"],
            "role": rec["role"],
            "source": rec["source"],
            "note": rec["note"],
        })
    pd.DataFrame(fig4b_rows).to_csv(out / "fig4b_residual_dependence_summary.csv", index=False)

    fig4_caption = [
        "# Figure 4 interpretation boundary",
        "",
        "Preferred title:",
        "\"Measured constitutional attributes weakly predict B8 means but retain residual multivariate dependence.\"",
        "",
        "Do not write:",
        "- B8 is independent of constitutional factors",
        "- B8 is unrelated to age/sex/height",
        "- B8 is fully orthogonal to constitutional attributes",
        "",
        "Panel A message:",
        "low-df nonlinear flexibility does not rescue conditional-mean prediction.",
        "",
        "Panel B message:",
        "residual constitutional dependence persists after pipeline/fold artifact controls.",
    ]
    (out / "fig4_interpretation_boundary.md").write_text("\n".join(fig4_caption) + "\n", encoding="utf-8")

    # ---------- Source manifest for this export ----------
    manifest_rows = []
    for label, path in sorted(required.items()):
        manifest_rows.append({
            "label": label,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_csv(
        out / "WFP_FIG0_SOURCE_MANIFEST_SHA256.csv",
        manifest_rows,
        ["label", "path", "bytes", "sha256"],
    )

    # ---------- Export readout ----------
    readout = "\n".join([
        "WF-P FIG0 — FIGURE-READY DATA EXPORT",
        "====================================",
        "Decision: WFP_FIG0_EXPORT_COMPLETE",
        "New scientific effects calculated: NO",
        "Frozen B8 changed: NO",
        "Raw waveform arrays opened: NO",
        "",
        "Exports:",
        "  fig1_schematic_panel_text.md",
        "  fig2_reconstruction_curves.csv",
        "  fig2_axis_reliability.csv",
        "  fig2_summary_metrics.csv",
        "  fig3_patient_projection_z1z2_PRIVATE.csv",
        "  fig3_scale_summary.csv",
        "  fig4a_conditional_mean_summary.csv",
        "  fig4b_residual_dependence_summary.csv",
        "  fig4_interpretation_boundary.md",
        "  WFP_FIG0_SOURCE_MANIFEST_SHA256.csv",
        "",
        "Boundary:",
        "  Figure 4 must use the post-Stage7B-RD two-layer interpretation.",
        "  No constitutional independence / full orthogonality claim is authorized.",
        "",
    ])
    (out / "WFP_FIG0_EXPORT_READOUT.txt").write_text(readout, encoding="utf-8")
    print(readout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
