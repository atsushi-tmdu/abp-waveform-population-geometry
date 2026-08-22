#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage 6A residual localization before execution."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS_SHA256 = "858a1b35b244f0aefb6f99e32474a70f487f135621003b30585e90f12c71bdd8"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--discovery-results", required=True)
    ap.add_argument("--stage5", required=True)
    ap.add_argument("--analysis-script", required=True)
    ap.add_argument("--run1-script", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    project=Path(args.project_root).expanduser().resolve()
    results=Path(args.discovery_results).expanduser().resolve()
    stage5=Path(args.stage5).expanduser().resolve()
    analysis=Path(args.analysis_script).expanduser().resolve()
    run1=Path(args.run1_script).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if sha256_file(analysis)!=EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL: Stage 6A analysis script hash mismatch")

    s5=json.loads(stage5.read_text())
    if s5.get("decision")!="WFP_AXIS_CHARACTERIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5 axis characterization not complete")
    if int(s5.get("dimension",-1))!=8:
        raise SystemExit("FAIL: Stage 5 dimension is not 8")
    if s5.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: Stage 5 clinical-label boundary violated")

    coord=results/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores=results/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    for p in (coord,scores,run1):
        if not p.is_file():
            raise SystemExit(f"FAIL: required input missing: {p}")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"6A",
      "status":"FROZEN_BEFORE_RESIDUAL_LOCALIZATION",
      "scientific_role":"post_discovery_descriptive_residual_localization_only",
      "frozen_dimension":8,
      "projection":"Q8 = I - B8 B8^T",
      "source":"Validation1000 discovery cohort; frozen eligibility reproduced exactly",
      "block_rule":{
        "block_sec":60,
        "min_accepted_beats_per_block":32,
        "min_total_blocks":6,
        "min_odd_blocks":3,
        "min_even_blocks":3
      },
      "questions":[
        "fraction of patient-central morphology variance outside frozen B8",
        "odd/even reproducibility of residual morphology outside frozen B8",
        "phase localization of positive reproducible residual energy",
        "fraction and phase localization of short-window within-person residual morphology outside B8"
      ],
      "primary_phase_profile":"normalized diagonal of positive part of odd/even residual cross-covariance",
      "secondary_phase_profiles":[
        "patient-central residual variance diagonal",
        "raw odd/even residual auto-variance diagonal",
        "equal-patient within-block residual covariance diagonal"
      ],
      "localization_metrics":{
        "effective_phase_support":"exp entropy of normalized 64-point phase-energy profile",
        "max_phase":True,
        "fixed_non_circular_window_widths_points":[4,8,16]
      },
      "notch_region_prespecified":False,
      "notch_specific_testing_authorized":False,
      "clinical_labels_authorized":False,
      "dimension_reselection_authorized":False,
      "basis_change_authorized":False,
      "analysis_script_sha256":sha256_file(analysis),
      "run1_script_sha256":sha256_file(run1),
      "stage5_result_sha256":sha256_file(stage5),
      "coordinates_sha256":sha256_file(coord),
      "patient_scores_sha256":sha256_file(scores)
    }

    p=out/"WFP_RESIDUAL_LOCALIZATION_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    txt=(
      "WF-P RESIDUAL LOCALIZATION SPEC FREEZE\n"
      "======================================\n"
      "Decision: WFP_RESIDUAL_LOCALIZATION_SPEC_FREEZE_PASS\n"
      "Frozen dimension: 8\n"
      "Notch region prespecified: NO\n"
      "Notch-specific testing authorized: NO\n"
      "Clinical labels authorized: NO\n"
      "Dimension reselection authorized: NO\n"
      "Basis change authorized: NO\n"
      f"Analysis script SHA256: {sha256_file(analysis)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_RESIDUAL_LOCALIZATION_FROZEN_SPEC.txt").write_text(txt,encoding="utf-8")
    print(txt,end="")

if __name__=="__main__":
    main()
