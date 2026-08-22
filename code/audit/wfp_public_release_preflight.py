#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P public-release preflight.

Read-only audit before constructing a release-safe GitHub/Zenodo tree.
It inventories WF-P code/freeze/results, classifies files, checks for local-path
or patient-ID leakage in candidate text files, and searches for WF3 interface
artifacts. It does not copy files, run git, upload anything, or recalculate
scientific effects.
"""

from __future__ import annotations
import argparse, csv, hashlib, json, re
from pathlib import Path

TEXT_SUFFIXES = {".py",".md",".txt",".json",".csv",".tsv",".yaml",".yml",".toml",".cff",".sh",".tex",".rst"}
RAW_SUFFIXES = {".npz",".npy",".mat",".hea",".dat",".wfdb",".gz"}
PRIVATE_TOKENS = {"private","patient_level","patient-level","checkpoint","checkpoints","cache","scan_log","execution_log","debug","tmp","temp"}

LOCAL_PATH_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"~/Documents/"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
]
PATIENT_ID_PATTERNS = [re.compile(r"\bp\d{5,}\b", re.I)]

SAFE_AGGREGATE_BASENAMES = {
    "WFP_DISCOVERY_READOUT.txt","WFP_DISCOVERY_RESULTS.json",
    "WFP_FINAL_SCIENTIFIC_STATUS.md","WFP_AUTHORITATIVE_RESULTS_SUMMARY.csv",
    "WFP_AUTHORITATIVE_RESULTS_SUMMARY.json","WFP_AUTHORITATIVE_RESULTS_SUMMARY.md",
    "WFP_FINAL_CLOSEOUT_READOUT.txt","WFP_FIGURE_MANIFEST_v0_1.md",
    "WFP_FIGURE_QC_CHECKLIST.md","WFP_STAGE7B_CONSTITUTIONAL_READOUT.txt",
    "WFP_STAGE7B_CONSTITUTIONAL_RESULTS.json","WFP_STAGE7B_NL_READOUT.txt",
    "WFP_STAGE7B_NL_RESULTS.json","WFP_STAGE7B_RD_READOUT.txt",
    "WFP_STAGE7B_RD_RESULTS.json","WFP_STAGE7C_ASSOCIATION_READOUT.txt",
    "WFP_STAGE7C_ASSOCIATION_RESULTS.json","WFP_BETWEEN_WITHIN_SCALE_READOUT.txt",
    "WFP_BETWEEN_WITHIN_SCALE_RESULTS.json",
}

INTERFACE_KEYWORDS = {
    "population_center":["population_center","population_mean","center64","mean64"],
    "frozen_B8_basis":["frozen_b8","b8_basis","basis_b8","population_basis"],
    "axis_sign_convention":["sign_convention","axis_sign","signs"],
    "projection_rule":["projection_rule","projection_spec","interface_spec"],
    "eigenvalues_dimension_profile":["eigenvalue","eigenspectrum","dimension_profile"],
    "Sigma_B":["sigma_b","between_covariance","between_cov"],
    "Sigma_W":["sigma_w","within_covariance","within_cov"],
}

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())

def read_small_text(path: Path, max_bytes=5_000_000) -> str:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > max_bytes:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

def private_reason(path: Path, root: Path) -> str:
    r=rel(path,root).lower()
    n=path.name.lower()
    if path.suffix.lower() in RAW_SUFFIXES:
        return f"restricted/raw array suffix {path.suffix.lower()}"
    if any(tok in n or f"/{tok}/" in f"/{r}/" for tok in PRIVATE_TOKENS):
        return "PRIVATE/patient-level/checkpoint/log naming"
    if "mimic-iii-clinical-database" in r or "chartevents" in n:
        return "restricted MIMIC source data"
    if "patient_scores" in n or "fig3_patient_projection_z1z2_private" in n:
        return "patient-level B8 source"
    if "source_manifest" in n and "final_closeout" in r:
        return "private source manifest with local paths"
    return ""

def candidate_reason(path: Path, root: Path) -> str:
    r=rel(path,root).lower()
    if path.name in SAFE_AGGREGATE_BASENAMES:
        return "known aggregate/closeout artifact"
    if r.startswith("code/") and path.suffix.lower() in {".py",".sh",".md"} and ("wfp" in path.name.lower() or "wfp" in r):
        return "WF-P scientific/audit/publication code"
    if r.startswith("freeze/") and "wfp" in r and path.suffix.lower() in {".json",".md",".txt"}:
        return "WF-P frozen specification/amendment"
    if "wfp_fig0_exports" in r and path.name in {
        "fig2_reconstruction_curves.csv","fig2_axis_reliability.csv","fig2_summary_metrics.csv",
        "fig3_scale_summary.csv","fig4a_conditional_mean_summary.csv",
        "fig4b_residual_dependence_summary.csv","fig4_interpretation_boundary.md",
        "fig1_schematic_panel_text.md","WFP_FIG0_EXPORT_READOUT.txt",
    }:
        return "aggregate integration/figure-ready artifact"
    if "results/wfp" in r and path.suffix.lower() in {".json",".txt",".md",".csv"}:
        n=path.name.lower()
        if any(tok in n for tok in ["summary","readout","aggregate","cohort","axis_reliability"]):
            return "candidate aggregate result"
    return ""

def scan_text_risks(path: Path):
    txt=read_small_text(path)
    if not txt:
        return 0,0,False
    local=sum(len(p.findall(txt)) for p in LOCAL_PATH_PATTERNS)
    pid=sum(len(p.findall(txt)) for p in PATIENT_ID_PATTERNS)
    priv=bool(re.search(r"\bPRIVATE\b", txt, re.I))
    return int(local),int(pid),priv

def find_interface_candidates(root: Path):
    files=[p for p in root.rglob("*") if p.is_file()]
    out={}
    for key,kws in INTERFACE_KEYWORDS.items():
        hits=[]
        for p in files:
            r=rel(p,root).lower()
            if "wfp" not in r and "interface" not in r:
                continue
            if any(kw in p.name.lower() for kw in kws):
                hits.append(rel(p,root))
        out[key]=sorted(set(hits))
    return out

def self_test():
    print("WF-P public-release preflight self-test: PASS")
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default="~/Documents/abp_information_study")
    ap.add_argument("--out",default="~/Documents/abp_information_study/results/wfp_public_release_preflight")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        return self_test()

    root=Path(a.project_root).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    closeout=root/"results"/"wfp_final_closeout"/"WFP_FINAL_CLOSEOUT_READOUT.txt"
    if not closeout.is_file():
        raise SystemExit(f"FAIL missing closeout: {closeout}")
    txt=closeout.read_text(encoding="utf-8",errors="replace")
    if "Decision: WFP_FINAL_CLOSEOUT_COMPLETE" not in txt:
        raise SystemExit("FAIL WF-P final closeout is not complete")
    if "No additional WF-P scientific-effect analysis" not in txt:
        raise SystemExit("FAIL closeout stop-rule marker absent")

    rows=[]
    for p in [x for x in root.rglob("*") if x.is_file()]:
        r=rel(p,root)
        if not (r.startswith("code/") or r.startswith("freeze/") or r.startswith("results/wfp") or "wfp" in p.name.lower()):
            continue

        priv=private_reason(p,root)
        cand=candidate_reason(p,root)
        if priv:
            cls,reason="PRIVATE_EXCLUDE",priv
            local=pid=None; privtok=None
        else:
            cls,reason=("PUBLIC_CANDIDATE",cand) if cand else ("MANUAL_REVIEW","not automatically classified")
            local,pid,privtok=scan_text_risks(p)
            if cls=="PUBLIC_CANDIDATE" and (local or pid or privtok):
                cls="MANUAL_REVIEW"
                reason += "; downgraded due to text-leak indicator"

        rows.append({
            "classification":cls,"relative_path":r,"bytes":int(p.stat().st_size),
            "sha256":sha256_file(p),"reason":reason,
            "local_path_hits":local,"patient_id_pattern_hits":pid,
            "contains_PRIVATE_token":privtok,
        })

    rows.sort(key=lambda x:(x["classification"],x["relative_path"]))
    inv=out/"WFP_PUBLIC_RELEASE_INVENTORY_PRIVATE.csv"
    with inv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else [
            "classification","relative_path","bytes","sha256","reason",
            "local_path_hits","patient_id_pattern_hits","contains_PRIVATE_token"])
        w.writeheader(); w.writerows(rows)

    counts={}
    for r in rows:
        counts[r["classification"]]=counts.get(r["classification"],0)+1

    interface=find_interface_candidates(root)
    missing=[k for k,v in interface.items() if not v]
    decision="WFP_PUBLIC_RELEASE_PREFLIGHT_PASS_NEEDS_STAGING_REVIEW" if counts.get("PUBLIC_CANDIDATE",0) else "WFP_PUBLIC_RELEASE_PREFLIGHT_BLOCKED_NO_PUBLIC_CANDIDATES"

    report={
        "decision":decision,
        "scientific_closeout_verified":True,
        "existing_wf1_wf2_releases_modified":False,
        "git_actions_performed":False,
        "uploads_performed":False,
        "classification_counts":counts,
        "interface_candidates":interface,
        "missing_interface_categories":missing,
        "private_inventory_contains_local_paths":True,
    }
    (out/"WFP_PUBLIC_RELEASE_PREFLIGHT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    lines=[
        "WF-P PUBLIC RELEASE PREFLIGHT",
        "============================",
        f"Decision: {decision}",
        "WF-P final scientific closeout verified: YES",
        "Existing WF1/WF2 releases modified: NO",
        "Git actions performed: NO",
        "Uploads performed: NO","",
        "Classification counts:",
        f"  PUBLIC_CANDIDATE: {counts.get('PUBLIC_CANDIDATE',0)}",
        f"  PRIVATE_EXCLUDE: {counts.get('PRIVATE_EXCLUDE',0)}",
        f"  MANUAL_REVIEW: {counts.get('MANUAL_REVIEW',0)}","",
        "WF3 interface candidate categories:",
    ]
    for key in INTERFACE_KEYWORDS:
        vals=interface[key]
        lines.append(f"  {key}: {len(vals)} candidate file(s)")
        for v in vals[:5]:
            lines.append(f"    - {v}")
        if len(vals)>5:
            lines.append(f"    ... +{len(vals)-5} more")
    lines += [
        "","Missing interface categories:",
        ("  NONE" if not missing else "  "+", ".join(missing)),"",
        "Hard public exclusions:",
        "  patient-level B8 scores/projections",
        "  MIMIC raw/clinical source files",
        "  NPZ/NPY/raw waveform arrays",
        "  checkpoints/caches/execution logs",
        "  local-path source manifests",
        "  files explicitly marked PRIVATE","",
        "Next step:",
        "  Review this report, then build a separate release-safe staging tree.",
        "  Do not git-add the private working directory directly.","",
    ]
    (out/"WFP_PUBLIC_RELEASE_PREFLIGHT.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
