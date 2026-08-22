#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, hashlib, json

EXPECTED_ANALYSIS_SHA256="e45780356e1b1e5b86e6d144d9ba16515c1093e08ca225c0d1d481b3b077da75"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--discovery-results",required=True)
    ap.add_argument("--temporal-linkage",required=True)
    ap.add_argument("--height-results",required=True)
    ap.add_argument("--wfp0-results",required=True)
    ap.add_argument("--analysis-script",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    discovery=Path(a.discovery_results).expanduser().resolve()
    temporal=Path(a.temporal_linkage).expanduser().resolve()
    heightdir=Path(a.height_results).expanduser().resolve()
    wfp0=Path(a.wfp0_results).expanduser().resolve()
    analysis=Path(a.analysis_script).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    score=discovery/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    coord=discovery/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    tpriv=temporal/"WFP_TEMPORAL_LINKAGE_AUDIT_PRIVATE.csv"
    tjson=temporal/"WFP_TEMPORAL_LINKAGE_AUDIT.json"
    hpriv=heightdir/"WFP_HEIGHT_PREFLIGHT_PRIVATE.csv"
    hjson=heightdir/"WFP_HEIGHT_PREFLIGHT.json"
    p0fac=wfp0/"wfp0_patient_conventional_factors_PRIVATE.csv"
    p0json=wfp0/"WFP0_MINIMAL_IDENTIFIABILITY_RESULTS.json"

    for p in (score,coord,tpriv,tjson,hpriv,hjson,p0fac,p0json,analysis):
        if not p.is_file():
            raise SystemExit(f"FAIL missing required input {p}")

    if sha256_file(analysis)!=EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL analysis script hash mismatch")

    tj=json.loads(tjson.read_text())
    hj=json.loads(hjson.read_text())
    pj=json.loads(p0json.read_text())

    if tj.get("decision")!="WFP_TEMPORAL_LINKAGE_AUDIT_COMPLETE":
        raise SystemExit("FAIL temporal linkage audit incomplete")
    if hj.get("decision")!="WFP_HEIGHT_PREFLIGHT_COMPLETE":
        raise SystemExit("FAIL height preflight incomplete")
    hres=hj.get("results",{})
    if hres.get("authorized_role_for_stage7b")!="SECONDARY_COMPLETE_CASE_SENSITIVITY_ONLY":
        raise SystemExit("FAIL frozen height role mismatch")
    if int(hres.get("height_nonmissing_n",-1))!=693:
        raise SystemExit("FAIL frozen height n != 693")
    if pj.get("decision")!="GO_CONVENTIONAL_FACTORS_DO_NOT_DOMINATE_FROZEN_B8":
        raise SystemExit("FAIL WF-P0 did not authorize Stage7B")

    spec={
      "schema_version":1,
      "work_package":"WF-P2",
      "stage":"7B",
      "status":"FROZEN_BEFORE_STAGE7B_CONSTITUTIONAL_ASSOCIATION",
      "expected_full_n":978,
      "primary_model":"age + sex",
      "primary_metric":"aggregate 5-fold patient OOF R2 across frozen z1..z8",
      "fold_rule":"sha256('20260820:<patient_id>') mod 5",
      "Q5":"OOF residual geometry after age+sex",
      "nonlinear_age_sensitivity":"age + age^2 + sex",
      "conventional_incremental_check":"level + log(scale) + log(duration), then add age + sex",
      "height_role":"SECONDARY_COMPLETE_CASE_SENSITIVITY_ONLY",
      "height_expected_n":693,
      "height_imputation":False,
      "chronic_phenotypes":"DEFERRED_TO_STAGE7C",
      "frozen_B8_relearning_authorized":False,
      "Ztrait_Zstate_labels_authorized":False,
      "analysis_script_sha256":sha256_file(analysis),
      "score_file_sha256":sha256_file(score),
      "coordinate_file_sha256":sha256_file(coord),
      "temporal_linkage_sha256":sha256_file(tpriv),
      "height_private_sha256":sha256_file(hpriv),
      "wfp0_factor_sha256":sha256_file(p0fac),
      "wfp0_result_sha256":sha256_file(p0json),
      "temporal_audit_sha256":sha256_file(tjson),
      "height_preflight_sha256":sha256_file(hjson)
    }

    p=out/"WFP_STAGE7B_CONSTITUTIONAL_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    txt=(
      "WF-P STAGE 7B CONSTITUTIONAL Q4/Q5 SPEC FREEZE\n"
      "================================================\n"
      "Decision: WFP_STAGE7B_CONSTITUTIONAL_SPEC_FREEZE_PASS\n"
      "Full cohort: 978\n"
      "Primary model: age + sex\n"
      "Primary metric: aggregate 5-fold patient OOF R2 for frozen B8\n"
      "Q5: OOF age+sex residual geometry\n"
      "Height: SECONDARY COMPLETE-CASE ONLY (n=693), NO IMPUTATION\n"
      "Conventional incremental check: YES\n"
      "Chronic phenotype mapping: DEFERRED TO STAGE 7C\n"
      "Frozen B8 changed: NO\n"
      "Ztrait/Zstate labels authorized: NO\n"
      f"Analysis SHA256: {sha256_file(analysis)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_STAGE7B_CONSTITUTIONAL_FROZEN_SPEC.txt").write_text(txt)
    print(txt,end="")

if __name__=="__main__":
    main()
