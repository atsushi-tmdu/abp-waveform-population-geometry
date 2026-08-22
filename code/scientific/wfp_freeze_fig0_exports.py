#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P FIG0 figure-ready export sources before export."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_EXPORT_SHA256 = "435a2d3e2fb0cde3abccb4a9f0becc303364d6d8ca2d031f7880a9fc982c2e5a"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--export-script", required=True)
    ap.add_argument("--out", default="~/Documents/abp_information_study/freeze/wfp_fig0")
    a = ap.parse_args()

    root = Path(a.project_root).expanduser().resolve()
    script = Path(a.export_script).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not script.is_file():
        raise SystemExit(f"FAIL missing export script: {script}")
    if sha256_file(script) != EXPECTED_EXPORT_SHA256:
        raise SystemExit("FAIL export script hash mismatch")

    closeout_summary = root / "results" / "wfp_final_closeout" / "WFP_AUTHORITATIVE_RESULTS_SUMMARY.csv"
    fig_manifest = root / "results" / "wfp_final_closeout" / "WFP_FIGURE_MANIFEST_v0_1.md"
    closeout_readout = root / "results" / "wfp_final_closeout" / "WFP_FINAL_CLOSEOUT_READOUT.txt"

    discovery_curve = root / "results" / "wfp_discovery_validation1000" / "wfp_cv_reconstruction_curve.csv"
    fourier_curve = root / "results" / "wfp_discovery_validation1000" / "wfp_fourier_comparator_curve.csv"
    axis_reliability = root / "results" / "wfp_discovery_validation1000" / "wfp_axis_reliability.csv"
    patient_scores = root / "results" / "wfp_discovery_validation1000" / "wfp_patient_scores_DISCOVERY_PRIVATE.csv"

    req = {
        "closeout_summary_sha256": closeout_summary,
        "figure_manifest_sha256": fig_manifest,
        "closeout_readout_sha256": closeout_readout,
        "discovery_curve_sha256": discovery_curve,
        "fourier_curve_sha256": fourier_curve,
        "axis_reliability_sha256": axis_reliability,
        "patient_scores_sha256": patient_scores,
    }

    for label, path in req.items():
        if not path.is_file():
            raise SystemExit(f"FAIL missing source {label}: {path}")

    txt = closeout_readout.read_text(encoding="utf-8", errors="replace")
    if "Decision: WFP_FINAL_CLOSEOUT_COMPLETE" not in txt:
        raise SystemExit("FAIL closeout decision marker absent")

    spec = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "FIG0",
        "status": "FROZEN_BEFORE_FIGURE_READY_EXPORT",
        "scientific_role": "figure_ready_export_only",
        "new_scientific_effects_authorized": False,
        "frozen_B8_change_authorized": False,
        "plotting_rule": "No scientific metric may be recomputed differently during plotting.",
        "figure_scope": {
            "main_figures": [1, 2, 3, 4],
            "supplementary_scope_authorized": False
        },
        "export_script_sha256": sha256_file(script),
    }
    for label, path in req.items():
        spec[label] = sha256_file(path)

    specp = out / "WFP_FIG0_FROZEN_SPEC.json"
    specp.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    txtout = (
        "WF-P FIG0 SPEC FREEZE\n"
        "=====================\n"
        "Decision: WFP_FIG0_SPEC_FREEZE_PASS\n"
        "Scientific role: FIGURE-READY EXPORT ONLY\n"
        "Main figure scope: 1-4\n"
        "New scientific effects authorized: NO\n"
        "Frozen B8 changed authorized: NO\n"
        "Supplementary export authorized: NO\n"
        f"Export script SHA256: {sha256_file(script)}\n"
        f"Frozen spec SHA256: {sha256_file(specp)}\n"
    )
    (out / "WFP_FIG0_FROZEN_SPEC.txt").write_text(txtout, encoding="utf-8")
    print(txtout, end="")

if __name__ == "__main__":
    main()
