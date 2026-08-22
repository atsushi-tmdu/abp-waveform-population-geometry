#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage 5B within-B8 localization before execution."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS_SHA256 = "53b631bc682a3018caafc6086cdfa63dac6759ac7a6a007fb3026adf1dd3e0dd"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--discovery-results", required=True)
    ap.add_argument("--stage5", required=True)
    ap.add_argument("--analysis-script", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    results=Path(args.discovery_results).expanduser().resolve()
    stage5=Path(args.stage5).expanduser().resolve()
    analysis=Path(args.analysis_script).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if sha256_file(analysis)!=EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL: Stage 5B analysis script hash mismatch")

    s5=json.loads(stage5.read_text())
    if s5.get("decision")!="WFP_AXIS_CHARACTERIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5 not complete")
    if int(s5.get("dimension",-1))!=8:
        raise SystemExit("FAIL: Stage 5 dimension is not 8")
    if s5.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: clinical-label boundary violated")

    coord=results/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores=results/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    for p in (coord,scores):
        if not p.is_file():
            raise SystemExit(f"FAIL missing required input: {p}")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"5B",
      "status":"FROZEN_BEFORE_WITHIN_B8_LOCALIZATION",
      "scientific_role":"post_discovery_descriptive_geometry_only",
      "frozen_dimension":8,
      "primary_analysis":{
        "scope":"all directions in frozen B8",
        "metrics":["shape","slope","curvature"],
        "window_widths_points":[4,8,16],
        "window_scan":"all non-circular contiguous windows on the relevant operator grid",
        "optimization":"maximum fraction of total operator energy within each window"
      },
      "targeted_axis5_6_check":{
        "status":"EXPLORATORY_POST_STAGE5_VISUAL_HYPOTHESIS",
        "pair":[5,6],
        "rule":"evaluate Axis5-6 span at the SAME windows selected by full-B8 analysis and rank it against all 28 axis pairs"
      },
      "notch_region_prespecified":False,
      "notch_specific_testing_authorized":False,
      "waveform_reprocessing_authorized":False,
      "clinical_labels_authorized":False,
      "dimension_reselection_authorized":False,
      "basis_change_authorized":False,
      "analysis_script_sha256":sha256_file(analysis),
      "stage5_result_sha256":sha256_file(stage5),
      "coordinates_sha256":sha256_file(coord),
      "patient_scores_sha256":sha256_file(scores)
    }

    p=out/"WFP_WITHIN_B8_LOCALIZATION_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    txt=(
      "WF-P WITHIN-B8 LOCALIZATION SPEC FREEZE\n"
      "=======================================\n"
      "Decision: WFP_WITHIN_B8_LOCALIZATION_SPEC_FREEZE_PASS\n"
      "Frozen dimension: 8\n"
      "Primary scope: ALL B8 DIRECTIONS\n"
      "Metrics: shape, slope, curvature\n"
      "Window widths: 4, 8, 16 points\n"
      "Axis 5-6 check: EXPLORATORY / POST-STAGE5 VISUAL HYPOTHESIS\n"
      "Notch region prespecified: NO\n"
      "Waveform reprocessing authorized: NO\n"
      "Clinical labels authorized: NO\n"
      "Basis change authorized: NO\n"
      f"Analysis script SHA256: {sha256_file(analysis)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_WITHIN_B8_LOCALIZATION_FROZEN_SPEC.txt").write_text(txt,encoding="utf-8")
    print(txt,end="")

if __name__=="__main__":
    main()
