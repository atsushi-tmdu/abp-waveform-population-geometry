#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze Stage 4B comparator-audit code and questions before execution."""

from pathlib import Path
import argparse, hashlib, json

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--geometry-script", required=True)
    ap.add_argument("--output-lock", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    script=Path(args.geometry_script).expanduser().resolve()
    lock=Path(args.output_lock).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    l=json.loads(lock.read_text())
    if l.get("decision")!="WFP_DISCOVERY_OUTPUT_LOCK_PASS":
        raise SystemExit("FAIL: discovery output lock is not PASS")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "stage":"4B",
      "status":"FROZEN_BEFORE_GEOMETRY_COMPARATOR_AUDIT",
      "dimension":8,
      "questions":[
        "replicate-corrected d=8 versus ordinary patient-mean PCA d=8 subspace geometry",
        "replicate-corrected d=8 versus fixed Fourier d=8 subspace geometry",
        "report already-prespecified held-out CV R2 differences at d=8"
      ],
      "metrics":[
        "projector overlap",
        "principal angles",
        "full-sample variance capture for replicate/PCA",
        "held-out CV R2 differences"
      ],
      "clinical_labels_authorized":False,
      "dimension_reselection_authorized":False,
      "basis_replacement_authorized":False,
      "geometry_script_sha256":sha256_file(script),
      "discovery_output_lock_sha256":sha256_file(lock)
    }
    p=out/"WFP_GEOMETRY_AUDIT_FROZEN_SPEC.json"
    p.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    txt=(
      "WF-P GEOMETRY AUDIT SPEC FREEZE\n"
      "================================\n"
      "Decision: WFP_GEOMETRY_AUDIT_SPEC_FREEZE_PASS\n"
      "Dimension: 8\n"
      "Clinical labels authorized: NO\n"
      "Dimension reselection authorized: NO\n"
      "Basis replacement authorized: NO\n"
      f"Geometry script SHA256: {sha256_file(script)}\n"
      f"Frozen spec SHA256: {sha256_file(p)}\n"
    )
    (out/"WFP_GEOMETRY_AUDIT_FROZEN_SPEC.txt").write_text(txt)
    print(txt,end="")

if __name__=="__main__":
    main()
