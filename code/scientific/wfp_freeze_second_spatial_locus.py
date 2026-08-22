#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage 5D second spatial locus before execution."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS_SHA256 = "21ebef6b1776f9ccb493aeec585ff103fb3a7994263de0c5f75e9622b7c22453"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--discovery-results",required=True)
    ap.add_argument("--stage5b-results",required=True)
    ap.add_argument("--stage5c-result",required=True)
    ap.add_argument("--analysis-script",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    discovery=Path(args.discovery_results).expanduser().resolve()
    stage5b=Path(args.stage5b_results).expanduser().resolve()
    stage5c=Path(args.stage5c_result).expanduser().resolve()
    analysis=Path(args.analysis_script).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if sha256_file(analysis)!=EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL: Stage 5D analysis script hash mismatch")

    s5b=stage5b/"WFP_WITHIN_B8_LOCALIZATION.json"
    dirs=stage5b/"wfp_within_b8_localized_directions.csv"
    wins=stage5b/"wfp_within_b8_localization_best_windows.csv"
    coord=discovery/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores=discovery/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"

    for p in (s5b,dirs,wins,stage5c,coord,scores):
        if not p.is_file():
            raise SystemExit(f"FAIL: required input missing: {p}")

    j5b=json.loads(s5b.read_text())
    j5c=json.loads(stage5c.read_text())

    if j5b.get("decision")!="WFP_WITHIN_B8_LOCALIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5B not complete")
    if j5c.get("decision")!="WFP_SECOND_LOCALIZED_DIRECTION_COMPLETE":
        raise SystemExit("FAIL: Stage 5C not complete")
    if j5b.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: Stage 5B clinical-label boundary violated")
    if j5c.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: Stage 5C clinical-label boundary violated")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"5D",
      "status":"FROZEN_BEFORE_SECOND_SPATIAL_LOCUS",
      "scientific_role":"explicitly_exploratory_post_stage5c_spatial_characterization",
      "rationale":"Stage5C second independent direction remained near first spatial locus; now answer the distinct question of the next spatially separated localization site",
      "frozen_dimension":8,
      "metrics":["shape","slope","curvature"],
      "window_widths_points":[4,8,16],
      "first_locus_source":"Stage5B full-B8 optimum separately for each metric/window width",
      "spatial_exclusion_rule":"exclude first window expanded by ceil(width/2) operator-grid points on each side",
      "second_locus_search":"all same-width windows not intersecting excluded zone; optimize over all frozen B8 directions",
      "third_spatial_locus_search_authorized":False,
      "axis5_6_targeted_check":{
        "status":"EXPLORATORY_POST_STAGE5_VISUAL_HYPOTHESIS",
        "rule":"evaluate Axis5-6 at the frozen W2 selected by full B8 and rank against all 28 axis pairs"
      },
      "notch_region_prespecified":False,
      "notch_specific_testing_authorized":False,
      "waveform_reprocessing_authorized":False,
      "clinical_labels_authorized":False,
      "dimension_reselection_authorized":False,
      "basis_change_authorized":False,
      "analysis_script_sha256":sha256_file(analysis),
      "coordinates_sha256":sha256_file(coord),
      "patient_scores_sha256":sha256_file(scores),
      "stage5b_result_sha256":sha256_file(s5b),
      "stage5b_first_directions_sha256":sha256_file(dirs),
      "stage5b_best_windows_sha256":sha256_file(wins),
      "stage5c_result_sha256":sha256_file(stage5c)
    }

    p=out/"WFP_SECOND_SPATIAL_LOCUS_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    txt=(
      "WF-P SECOND SPATIAL LOCUS SPEC FREEZE\n"
      "=====================================\n"
      "Decision: WFP_SECOND_SPATIAL_LOCUS_SPEC_FREEZE_PASS\n"
      "Scientific role: EXPLORATORY POST-STAGE5C\n"
      "Frozen dimension: 8\n"
      "Search authorized: SECOND SPATIAL LOCUS ONLY\n"
      "Third spatial locus authorized: NO\n"
      "Spatial exclusion: first window + half-window guard on each side\n"
      "Notch region prespecified: NO\n"
      "Clinical labels authorized: NO\n"
      "Waveform reprocessing authorized: NO\n"
      f"Analysis script SHA256: {sha256_file(analysis)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_SECOND_SPATIAL_LOCUS_FROZEN_SPEC.txt").write_text(txt,encoding="utf-8")
    print(txt,end="")

if __name__=="__main__":
    main()
