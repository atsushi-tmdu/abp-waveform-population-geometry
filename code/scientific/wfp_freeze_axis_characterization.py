#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage 5 axis characterization before execution."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS_SHA256 = "d51164268d1064e30ef44f56b5c24732e36eb93f4721fb007d0777f49f2de182"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--discovery-results", required=True)
    ap.add_argument("--geometry-audit", required=True)
    ap.add_argument("--discovery-script", required=True)
    ap.add_argument("--analysis-script", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    results=Path(args.discovery_results).expanduser().resolve()
    geometry=Path(args.geometry_audit).expanduser().resolve()
    disc_script=Path(args.discovery_script).expanduser().resolve()
    analysis=Path(args.analysis_script).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if sha256_file(analysis) != EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL: Stage 5 analysis script hash mismatch")

    g=json.loads(geometry.read_text())
    if g.get("decision")!="WFP_GEOMETRY_COMPARATOR_AUDIT_COMPLETE":
        raise SystemExit("FAIL: Stage 4 geometry audit not complete")
    if int(g.get("dimension_fixed",-1))!=8:
        raise SystemExit("FAIL: Stage 4 dimension is not 8")
    if g.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: clinical-label boundary violated")

    coord=results/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores=results/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    reliability=results/"wfp_axis_reliability.csv"
    for p in (coord,scores,reliability,disc_script):
        if not p.is_file():
            raise SystemExit(f"FAIL missing required input: {p}")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":5,
      "status":"FROZEN_BEFORE_AXIS_CHARACTERIZATION",
      "scientific_role":"descriptive_characterization_of_frozen_discovery_basis",
      "dimension":8,
      "questions":[
        "What waveform deformation is represented by each already-frozen WF-P axis?",
        "How much of each frozen axis lies in the same fixed Fourier d=8 comparator subspace?",
        "What non-Fourier residual remains for each frozen axis?"
      ],
      "display_rule":"population mean +/- 1 empirical population score SD along each frozen axis; no post-displacement renormalization",
      "fourier_rule":"project each frozen axis onto the exact fixed Fourier d=8 subspace used in Stage 3",
      "outputs":[
        "axis-wise score scale and reliability table",
        "axis-wise Fourier d=8 energy fraction",
        "axis morphology curves",
        "Fourier component and non-Fourier residual vectors",
        "descriptive figures"
      ],
      "clinical_labels_authorized":False,
      "waveform_reprocessing_authorized":False,
      "dimension_reselection_authorized":False,
      "basis_change_authorized":False,
      "physiological_axis_naming_authorized":False,
      "analysis_script_sha256":sha256_file(analysis),
      "geometry_audit_sha256":sha256_file(geometry),
      "discovery_script_sha256":sha256_file(disc_script),
      "coordinates_sha256":sha256_file(coord),
      "patient_scores_sha256":sha256_file(scores),
      "axis_reliability_sha256":sha256_file(reliability)
    }
    p=out/"WFP_AXIS_CHARACTERIZATION_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    txt=(
      "WF-P AXIS CHARACTERIZATION SPEC FREEZE\n"
      "======================================\n"
      "Decision: WFP_AXIS_CHARACTERIZATION_SPEC_FREEZE_PASS\n"
      "Dimension: 8\n"
      "Waveform reprocessing authorized: NO\n"
      "Clinical labels authorized: NO\n"
      "Dimension reselection authorized: NO\n"
      "Basis change authorized: NO\n"
      "Physiological axis naming authorized: NO\n"
      f"Analysis script SHA256: {sha256_file(analysis)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_AXIS_CHARACTERIZATION_FROZEN_SPEC.txt").write_text(txt)
    print(txt,end="")

if __name__=="__main__":
    main()
