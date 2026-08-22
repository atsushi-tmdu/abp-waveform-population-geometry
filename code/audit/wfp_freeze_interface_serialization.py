#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

EXPECTED_SOURCE_N=1000
EXPECTED_ANALYSABLE_N=978

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def sha_lines(lines):
    h=hashlib.sha256()
    for x in lines:
        h.update(str(x).encode()); h.update(b"\n")
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default="~/Documents/abp_information_study")
    ap.add_argument("--input",default="~/Documents/abp_information_study/data/abp125_validation1000")
    ap.add_argument("--serializer-script",required=True)
    ap.add_argument("--out",default="~/Documents/abp_information_study/freeze/wfp_interface_serialization")
    a=ap.parse_args()

    root=Path(a.project_root).expanduser().resolve()
    inp=Path(a.input).expanduser().resolve()
    serializer=Path(a.serializer_script).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    ddir=root/"results"/"wfp_discovery_validation1000"
    paths={
      "discovery_script":root/"code"/"wfp_validation1000_discovery.py",
      "run1_script":root/"code"/"wp2_run1_development50.py",
      "discovery_frozen_spec":root/"freeze"/"WFP_DISCOVERY_FROZEN_SPEC.json",
      "discovery_results":ddir/"WFP_DISCOVERY_RESULTS.json",
      "discovery_readout":ddir/"WFP_DISCOVERY_READOUT.txt",
      "common_coordinates":ddir/"WFP_DISCOVERY_COMMON_COORDINATES.npz",
      "population_eigenspectra":ddir/"wfp_population_eigenspectra.csv",
      "cv_reconstruction_curve":ddir/"wfp_cv_reconstruction_curve.csv",
      "axis_reliability":ddir/"wfp_axis_reliability.csv",
      "serializer_script":serializer,
    }
    for k,p in paths.items():
        if not p.is_file(): raise SystemExit(f"FAIL missing {k}: {p}")

    res=json.loads(paths["discovery_results"].read_text())
    if res.get("decision")!="WFP_DISCOVERY_COMMON_BASIS_IDENTIFIED":
        raise SystemExit("FAIL discovery decision")
    if int(res.get("source_n",-1))!=EXPECTED_SOURCE_N or int(res.get("analysable_n",-1))!=EXPECTED_ANALYSABLE_N:
        raise SystemExit("FAIL discovery cohort counts")
    if res.get("analysis_script_sha256")!=sha(paths["discovery_script"]): raise SystemExit("FAIL discovery script hash")
    if res.get("authoritative_run1_sha256")!=sha(paths["run1_script"]): raise SystemExit("FAIL Run1 hash")
    if res.get("frozen_spec_sha256")!=sha(paths["discovery_frozen_spec"]): raise SystemExit("FAIL discovery spec hash")

    if "Selected common-basis dimension: 8" not in paths["discovery_readout"].read_text(errors="replace"):
        raise SystemExit("FAIL frozen d=8 marker")

    cases=sorted((inp/"cases").glob("*.npz"))
    names=[p.name for p in cases]
    if len(names)!=EXPECTED_SOURCE_N or len(names)!=len(set(names)):
        raise SystemExit("FAIL source case filename set")

    spec={
      "schema_version":1,
      "stage":"WF-P_INTERFACE_SERIALIZATION",
      "status":"FROZEN_BEFORE_RELEASE_SERIALIZATION_REPLAY",
      "scientific_effects_already_closed":True,
      "new_scientific_analysis_authorized":False,
      "B8_change_authorized":False,
      "source_n":EXPECTED_SOURCE_N,
      "expected_analysable_n":EXPECTED_ANALYSABLE_N,
      "case_filename_manifest_sha256":sha_lines(names),
      "case_filenames":names,
      "source_hashes":{k:sha(p) for k,p in paths.items()},
      "serialization_targets":{
        "Sigma_B_ordinary":"ordinary covariance of 30-min central morphology",
        "S_rep":"replicate-corrected symmetric odd/even operator used by final discovery",
        "S_rep_positive":"positive-spectrum PSD form of S_rep",
      },
      "hard_rule":"stored B8 basis is authoritative and must never be replaced by replay eigenvectors",
    }
    sp=out/"WFP_INTERFACE_SERIALIZATION_FROZEN_SPEC.json"
    sp.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    (out/"WFP_INTERFACE_SERIALIZATION_CASE_FILENAMES_PRIVATE.txt").write_text("\n".join(names)+"\n")
    txt="\n".join([
      "WF-P INTERFACE SERIALIZATION FREEZE",
      "==================================",
      "Decision: WFP_INTERFACE_SERIALIZATION_FREEZE_PASS",
      "Scientific role: ENGINEERING RELEASE SERIALIZATION ONLY",
      "New scientific analysis authorized: NO",
      "Frozen B8 change authorized: NO",
      f"Source case filenames pinned: {len(names)}",
      f"Expected analysable n: {EXPECTED_ANALYSABLE_N}",
      "",
      "Operators to serialize:",
      "  Sigma_B_ordinary = ordinary covariance of 30-min central morphology",
      "  S_rep = final replicate-corrected primary operator",
      "  S_rep_positive = positive-spectrum PSD form of S_rep",
      "",
      f"Serializer SHA256: {sha(serializer)}",
      f"Frozen spec SHA256: {sha(sp)}",
      "",
    ])
    (out/"WFP_INTERFACE_SERIALIZATION_FROZEN_SPEC.txt").write_text(txt)
    print(txt)

if __name__=="__main__": main()
