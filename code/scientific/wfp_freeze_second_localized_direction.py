#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage 5C before second-direction effect access."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS_SHA256 = "aec8408e4e9a5c1c076ed601690f7fb0ce1028cce5fb94b9b83462be509aca98"

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
    ap.add_argument("--analysis-script",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    discovery=Path(args.discovery_results).expanduser().resolve()
    stage5b=Path(args.stage5b_results).expanduser().resolve()
    analysis=Path(args.analysis_script).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if sha256_file(analysis)!=EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL: Stage 5C analysis script hash mismatch")

    s5b_path=stage5b/"WFP_WITHIN_B8_LOCALIZATION.json"
    first_dirs=stage5b/"wfp_within_b8_localized_directions.csv"
    best_windows=stage5b/"wfp_within_b8_localization_best_windows.csv"
    coord=discovery/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores=discovery/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"

    for p in (s5b_path,first_dirs,best_windows,coord,scores):
        if not p.is_file():
            raise SystemExit(f"FAIL: required input missing: {p}")

    s5b=json.loads(s5b_path.read_text())
    if s5b.get("decision")!="WFP_WITHIN_B8_LOCALIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5B not complete")
    if int(s5b.get("frozen_dimension",-1))!=8:
        raise SystemExit("FAIL: frozen dimension not 8")
    if s5b.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: clinical-label boundary violated")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"5C",
      "status":"FROZEN_BEFORE_SECOND_LOCALIZED_DIRECTION",
      "scientific_role":"explicitly_exploratory_post_stage5b_characterization",
      "rationale":"Stage 5B first localized direction concentrated near upstroke/peak; inspect exactly one independent second direction before outside-B8 residual analysis",
      "frozen_dimension":8,
      "metrics":["shape","slope","curvature"],
      "window_widths_points":[4,8,16],
      "first_direction_source":"fixed Stage5B full-B8 optimum separately for each metric/window width",
      "second_direction_constraint":"c^T c1 = 0 in frozen B8 coefficient space; because B8 is orthonormal this is shape-L2 orthogonality",
      "window_scan":"all non-circular contiguous windows; no forced spatial separation from first window",
      "third_or_higher_direction_search_authorized":False,
      "notch_region_prespecified":False,
      "notch_specific_testing_authorized":False,
      "waveform_reprocessing_authorized":False,
      "clinical_labels_authorized":False,
      "dimension_reselection_authorized":False,
      "basis_change_authorized":False,
      "analysis_script_sha256":sha256_file(analysis),
      "coordinates_sha256":sha256_file(coord),
      "patient_scores_sha256":sha256_file(scores),
      "stage5b_result_sha256":sha256_file(s5b_path),
      "stage5b_first_directions_sha256":sha256_file(first_dirs),
      "stage5b_best_windows_sha256":sha256_file(best_windows)
    }

    p=out/"WFP_SECOND_LOCALIZED_DIRECTION_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    txt=(
      "WF-P SECOND LOCALIZED DIRECTION SPEC FREEZE\n"
      "===========================================\n"
      "Decision: WFP_SECOND_LOCALIZED_DIRECTION_SPEC_FREEZE_PASS\n"
      "Scientific role: EXPLORATORY POST-STAGE5B\n"
      "Frozen dimension: 8\n"
      "Search authorized: SECOND DIRECTION ONLY\n"
      "Third/higher directions authorized: NO\n"
      "Forced phase separation from first direction: NO\n"
      "Notch region prespecified: NO\n"
      "Clinical labels authorized: NO\n"
      "Waveform reprocessing authorized: NO\n"
      f"Analysis script SHA256: {sha256_file(analysis)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_SECOND_LOCALIZED_DIRECTION_FROZEN_SPEC.txt").write_text(txt,encoding="utf-8")
    print(txt,end="")

if __name__=="__main__":
    main()
