#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage7C phenotype association specification."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS="d3cd5397f456e864ea380bf6e0b4e85040f5fe7fba12aac70971d437957eb636"

PHENOTYPES=[
    "congestive_heart_failure",
    "cardiac_arrhythmias",
    "valvular_disease",
    "peripheral_vascular_disease",
    "hypertension",
    "diabetes",
    "renal_failure",
    "chronic_pulmonary_disease",
]

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
    ap.add_argument("--wfp0-factors",required=True)
    ap.add_argument("--phenotype-private",required=True)
    ap.add_argument("--phenotype-preflight-json",required=True)
    ap.add_argument("--scale-results-json",required=True)
    ap.add_argument("--stage7b-results-json",required=True)
    ap.add_argument("--analysis-script",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    score=Path(a.score_file).expanduser().resolve()
    temporal=Path(a.temporal_linkage).expanduser().resolve()
    factors=Path(a.wfp0_factors).expanduser().resolve()
    pheno=Path(a.phenotype_private).expanduser().resolve()
    preflight=Path(a.phenotype_preflight_json).expanduser().resolve()
    scale=Path(a.scale_results_json).expanduser().resolve()
    stage7b=Path(a.stage7b_results_json).expanduser().resolve()
    analysis=Path(a.analysis_script).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    for p in (score,temporal,factors,pheno,preflight,scale,stage7b,analysis):
        if not p.is_file():
            raise SystemExit(f"FAIL missing {p}")

    if sha(analysis)!=EXPECTED_ANALYSIS:
        raise SystemExit("FAIL Stage7C analysis-script hash")

    pj=json.loads(preflight.read_text())
    sj=json.loads(scale.read_text())
    bj=json.loads(stage7b.read_text())

    if pj.get("decision")!="WFP_STAGE7C_PHENOTYPE_PREFLIGHT_COMPLETE":
        raise SystemExit("FAIL phenotype preflight incomplete")
    if pj.get("main_stage7c_candidates")!=PHENOTYPES:
        raise SystemExit("FAIL phenotype candidate family mismatch")
    if int(pj.get("exact_current_admission_n",-1))!=887:
        raise SystemExit("FAIL exact-admission n != 887")

    if sj.get("decision")!="WFP_BETWEEN_WITHIN_SCALE_AUDIT_COMPLETE":
        raise SystemExit("FAIL scale audit incomplete")
    if bj.get("decision")!="WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE":
        raise SystemExit("FAIL Stage7B incomplete")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"7C",
      "status":"FROZEN_BEFORE_STAGE7C_ASSOCIATION",
      "scientific_role":"post_discovery_cross_sectional_phenotype_mapping",
      "analysis_n":887,
      "phenotype_family":PHENOTYPES,
      "phenotypes_entered_jointly":True,
      "primary_baseline":[
        "age_years_capped90",
        "sex",
        "level_mmhg",
        "log_scale_sd",
        "log_duration_sec"
      ],
      "age_form":"linear",
      "height_primary_adjustment":False,
      "primary_effect":"joint 8-phenotype model aggregate OOF R2 minus baseline aggregate OOF R2",
      "fold_rule":"sha256('20260820:<patient_id>') mod 5",
      "phenotype_specific_effects":[
        "marginal delta OOF R2 vs baseline",
        "unique delta OOF R2 within joint phenotype block",
        "joint-model adjusted B8 coefficient-vector norm",
        "coefficient norm normalized by pre-frozen between/within/NN scales"
      ],
      "global_test":{
        "family_size":8,
        "statistic":"partial multivariate SSE improvement",
        "permutation":"reduced-model residual permutation",
        "permutations":2000,
        "seed_namespace":20260823,
        "plus_one_p":True,
        "multiplicity":"Benjamini-Hochberg FDR",
        "FDR_level":0.05
      },
      "axis_specific_coefficients":"DESCRIPTIVE_ONLY_NO_64_TEST_FAMILY",
      "prior_history_analysis":"DEFERRED_NOT_PRIMARY",
      "forbidden":[
        "relearning or rotating B8",
        "disease-specific subspace learning",
        "post-result phenotype family expansion",
        "axis-specific multiplicity fishing",
        "causal interpretation",
        "pre-waveform-known interpretation of current-admission codes",
        "Ztrait/Zstate labeling"
      ],
      "analysis_script_sha256":sha(analysis),
      "score_file_sha256":sha(score),
      "temporal_linkage_sha256":sha(temporal),
      "wfp0_factor_sha256":sha(factors),
      "phenotype_private_sha256":sha(pheno),
      "phenotype_preflight_sha256":sha(preflight),
      "scale_results_sha256":sha(scale),
      "stage7b_results_sha256":sha(stage7b)
    }

    p=out/"WFP_STAGE7C_ASSOCIATION_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    txt=(
      "WF-P STAGE 7C ASSOCIATION SPEC FREEZE\n"
      "=====================================\n"
      "Decision: WFP_STAGE7C_ASSOCIATION_SPEC_FREEZE_PASS\n"
      "Analysis n: 887 exact-current-admission patients\n"
      "Phenotype family: 8 prespecified Quan-Elixhauser-style groups\n"
      "Primary baseline: age + sex + level + log(scale) + log(duration)\n"
      "Primary effect: joint phenotype-block incremental OOF R2\n"
      "Global phenotype tests: 8 only; 2000 residual permutations; BH-FDR 0.05\n"
      "Axis-specific coefficients: DESCRIPTIVE ONLY\n"
      "Frozen B8 changed: NO\n"
      "Disease-specific subspace learning: NO\n"
      "Prior-history analysis: DEFERRED\n"
      f"Analysis SHA256: {sha(analysis)}\n"
      f"Frozen spec SHA256: {sha(p)}\n"
    )
    (out/"WFP_STAGE7C_ASSOCIATION_FROZEN_SPEC.txt").write_text(txt,encoding="utf-8")
    print(txt,end="")

if __name__=="__main__":
    main()
