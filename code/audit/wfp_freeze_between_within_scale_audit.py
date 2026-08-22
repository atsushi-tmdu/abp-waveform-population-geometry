#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json
EXPECTED_RUN1="811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01"
EXPECTED_ANALYSIS="cbcd5f5601c11c3887dcae33ba825f9a0c3f47b6e6fb15acb08bb8a81924d3e7"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run1-script",required=True); ap.add_argument("--discovery-results",required=True)
    ap.add_argument("--analysis-script",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    run1=Path(a.run1_script).expanduser().resolve(); disc=Path(a.discovery_results).expanduser().resolve()
    ana=Path(a.analysis_script).expanduser().resolve(); out=Path(a.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    score=disc/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"; coord=disc/"WFP_DISCOVERY_COMMON_COORDINATES.npz"; result=disc/"WFP_DISCOVERY_RESULTS.json"
    for p in (run1,score,coord,result,ana):
        if not p.is_file(): raise SystemExit(f"FAIL missing {p}")
    if sha(run1)!=EXPECTED_RUN1: raise SystemExit("FAIL Run1 hash")
    if sha(ana)!=EXPECTED_ANALYSIS: raise SystemExit("FAIL analysis hash")
    r=json.loads(result.read_text())
    if int(r.get("analysable_n",-1))!=978 or r.get("clinical_labels_accessed") is not False: raise SystemExit("FAIL discovery status")
    spec={"status":"FROZEN_BEFORE_BETWEEN_WITHIN_SCALE_AUDIT","expected_n":978,"dimension":8,
          "primary_distance":"raw Euclidean distance in frozen B8","clinical_labels_authorized":False,
          "B8_relearning_authorized":False,"automatic_gate":"NONE_DESCRIPTIVE_SCALE_AUDIT",
          "authoritative_run1_sha256":sha(run1),"analysis_script_sha256":sha(ana),
          "score_file_sha256":sha(score),"coordinate_file_sha256":sha(coord),"discovery_result_sha256":sha(result)}
    p=out/"WFP_BETWEEN_WITHIN_SCALE_FROZEN_SPEC.json"; p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    txt=("WF-P BETWEEN–WITHIN SCALE AUDIT SPEC FREEZE\n===========================================\n"
         "Decision: WFP_BETWEEN_WITHIN_SCALE_SPEC_FREEZE_PASS\nExpected patients: 978\nFrozen B8 dimension: 8\n"
         "Primary distance: raw Euclidean distance in frozen B8\nClinical labels authorized: NO\nB8 relearning/reselection authorized: NO\n"
         "Automatic scientific gate: NONE\nZtrait/Zstate labels authorized: NO\n"
         f"Analysis SHA256: {sha(ana)}\nFrozen spec SHA256: {sha(p)}\n")
    (out/"WFP_BETWEEN_WITHIN_SCALE_FROZEN_SPEC.txt").write_text(txt); print(txt,end="")
if __name__=="__main__": main()
