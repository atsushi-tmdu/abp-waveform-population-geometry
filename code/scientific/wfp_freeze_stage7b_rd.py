#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage7B-RD artifact adjudication before its null-control outputs."""

from pathlib import Path
import argparse, hashlib, json, re

EXPECTED_ANALYSIS_SHA256 = "a0d851c7b76a47cf645859a5c64ba72f8edb8afa769857587a853521cf614610"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--score-file",required=True)
    ap.add_argument("--temporal-linkage",required=True)
    ap.add_argument("--height-preflight",required=True)
    ap.add_argument("--stage7b-nl-results-json",required=True)
    ap.add_argument("--stage7b-nl-readout",required=True)
    ap.add_argument("--stage7b-nl-spec",required=True)
    ap.add_argument("--analysis-script",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    score=Path(a.score_file).expanduser().resolve()
    temporal=Path(a.temporal_linkage).expanduser().resolve()
    height=Path(a.height_preflight).expanduser().resolve()
    nlres=Path(a.stage7b_nl_results_json).expanduser().resolve()
    nlread=Path(a.stage7b_nl_readout).expanduser().resolve()
    nlspec=Path(a.stage7b_nl_spec).expanduser().resolve()
    analysis=Path(a.analysis_script).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    for p in [score,temporal,height,nlres,nlread,nlspec,analysis]:
        if not p.is_file():
            raise SystemExit(f"FAIL missing {p}")

    if sha(analysis)!=EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL Stage7B-RD analysis script hash mismatch")

    txt=nlread.read_text(encoding="utf-8",errors="replace")
    required=[
        "Decision: NO_MATERIAL_OOF_MEAN_GAIN_BUT_RESIDUAL_MODEL_FREE_DEPENDENCE_FLAG",
        "Full age+sex vs OOF residual B8: dCor=0.167447",
        "Height age+sex+height vs OOF residual B8: dCor=0.180459",
    ]
    absent=[x for x in required if x not in txt]
    if absent:
        raise SystemExit(f"FAIL Stage7B-NL trigger markers absent: {absent}")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"7B-RD",
      "status":"FROZEN_BEFORE_STAGE7B_RD_ADJUDICATION",
      "scientific_role":"post_stage7b_nl_residual_dependence_artifact_adjudication",
      "trigger_observed_before_RD_freeze":True,
      "trigger":"Stage7B-NL residual distance-correlation signal",
      "primary_question":"Could OOF residualization / finite-sample fitting generate the observed residual dCor?",
      "primary_control":{
        "name":"pipeline-replay residual-permutation null",
        "steps":[
          "fit full-sample linear constitutional mean",
          "permute complete 8-D residual vectors across patients",
          "add permuted residuals to fitted linear mean",
          "rerun complete frozen 5-fold OOF linear prediction",
          "recompute OOF residuals",
          "recompute residual dCor"
        ],
        "full_model":"age+sex",
        "height_model":"age+sex+height_median_cm",
        "permutations":999,
        "plus_one_p":True,
        "seed_namespace":20260825,
        "multiplicity":"Holm across full978 and height693 primary pipeline controls",
        "null_assumption":"residual-vector exchangeability under the artifact-control null"
      },
      "secondary_control":{
        "name":"within-frozen-fold residual-vector permutation",
        "purpose":"preserve fold-level residual distributions and X imbalance",
        "permutations":999,
        "plus_one_p":True,
        "formal_primary_gate":False
      },
      "observed_dcor_to_replay":{
        "full978":0.167447,
        "height693":0.180459
      },
      "forbidden":[
        "modifying Stage7B",
        "modifying Stage7B-NL",
        "changing B8",
        "new nonlinear mean model",
        "variance regression",
        "covariance regression",
        "distributional model search",
        "subgroup exploration",
        "disease/treatment/outcome access",
        "Ztrait/Zstate labeling"
      ],
      "analysis_script_sha256":sha(analysis),
      "score_file_sha256":sha(score),
      "temporal_linkage_sha256":sha(temporal),
      "height_preflight_sha256":sha(height),
      "stage7b_nl_results_sha256":sha(nlres),
      "stage7b_nl_readout_sha256":sha(nlread),
      "stage7b_nl_spec_sha256":sha(nlspec)
    }

    p=out/"WFP_STAGE7B_RD_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    txtout=(
      "WF-P STAGE 7B-RD SPEC FREEZE\n"
      "============================\n"
      "Decision: WFP_STAGE7B_RD_SPEC_FREEZE_PASS\n"
      "Scientific role: POST-RESULT ARTIFACT ADJUDICATION\n"
      "Stage7B-NL dCor trigger observed before this freeze: YES\n"
      "Primary control: full pipeline-replay residual permutation\n"
      "Primary permutations: 999 per cohort\n"
      "Primary multiplicity: Holm across full978 and height693\n"
      "Secondary diagnostic: within-frozen-fold residual permutation\n"
      "Secondary permutations: 999 per cohort\n"
      "Variance/covariance model authorized: NO\n"
      "Stage7B / Stage7B-NL modification authorized: NO\n"
      "Ztrait/Zstate labels authorized: NO\n"
      f"Analysis SHA256: {sha(analysis)}\n"
      f"Frozen spec SHA256: {sha(p)}\n"
    )
    (out/"WFP_STAGE7B_RD_FROZEN_SPEC.txt").write_text(txtout,encoding="utf-8")
    print(txtout,end="")

if __name__=="__main__":
    main()
