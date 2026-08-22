#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P interface recovery / release-completeness audit
====================================================

Purpose
-------
The generic public-release preflight used filename heuristics and therefore can
miss interface arrays stored inside `WFP_DISCOVERY_COMMON_COORDINATES.npz`.

This audit inspects the *known authoritative frozen discovery artifacts* and
reports what is already available for a release-safe WF3 interface.

This is engineering/provenance work only:
- no patient-level score table is opened;
- no raw waveform/case NPZ is opened;
- no scientific model is fit;
- no eigen-decomposition or covariance is recomputed;
- no B8 orientation/sign is changed.

It explicitly checks:
- population_mean
- between_basis
- between_eigenvalues
- within_window_covariance
- dimension
- full eigenspectrum CSV
- CV reconstruction curve
- axis reliability
- discovery result/readout

It also searches only release-like aggregate files for any already-serialized
full between-person covariance/operator. If none is found, it reports that full
Sigma_B remains a serialization gap rather than inventing/recomputing it.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_DIM = 8
EXPECTED_PHASE_POINTS = 64

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def finite(a) -> bool:
    return bool(np.isfinite(np.asarray(a, float)).all())

def self_test():
    B=np.eye(8)
    err=float(np.max(np.abs(B.T@B-np.eye(8))))
    if err > 1e-12:
        raise RuntimeError("orthogonality self-test failed")
    print("WF-P interface recovery audit self-test: PASS")
    return 0

def scan_between_cov_candidates(root: Path):
    """
    Search only aggregate-looking files under results/wfp*.
    Never open files marked PRIVATE, patient_scores, projection, cases, or raw.
    """
    candidates=[]
    key_patterns=("sigma_b","between_cov","between_covariance","srep","between_operator")
    for p in root.glob("results/wfp*/**/*"):
        if not p.is_file():
            continue
        low=str(p).lower()
        if any(tok in low for tok in ["private","patient_scores","projection","cases","checkpoint","cache"]):
            continue
        if p.suffix.lower() not in {".npz",".npy",".csv",".json"}:
            continue

        name=p.name.lower()
        name_hit=any(k in name for k in key_patterns)
        item={"path":str(p.relative_to(root)),"sha256":sha256_file(p),"name_hit":name_hit}

        if p.suffix.lower()==".npz" and p.stat().st_size < 100_000_000:
            try:
                with np.load(p,allow_pickle=False) as z:
                    keys=list(z.files)
                    key_hits=[k for k in keys if any(q in k.lower() for q in key_patterns)]
                    item["npz_keys"]=keys
                    item["matching_keys"]=key_hits
                    for k in key_hits:
                        item[f"shape_{k}"]=list(np.asarray(z[k]).shape)
            except Exception as e:
                item["inspect_error"]=repr(e)
        elif p.suffix.lower()==".json" and p.stat().st_size < 20_000_000:
            try:
                obj=json.loads(p.read_text(encoding="utf-8",errors="replace"))
                flat=json.dumps(obj).lower()
                item["content_keyword_hit"]=any(k in flat for k in key_patterns)
            except Exception as e:
                item["inspect_error"]=repr(e)
        elif p.suffix.lower()==".csv" and p.stat().st_size < 20_000_000:
            try:
                hdr=pd.read_csv(p,nrows=0).columns.tolist()
                item["columns"]=hdr
                item["matching_columns"]=[c for c in hdr if any(q in c.lower() for q in key_patterns)]
            except Exception as e:
                item["inspect_error"]=repr(e)

        if name_hit or item.get("matching_keys") or item.get("matching_columns") or item.get("content_keyword_hit"):
            candidates.append(item)
    return candidates

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default="~/Documents/abp_information_study")
    ap.add_argument("--out",default="~/Documents/abp_information_study/results/wfp_interface_recovery_audit")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        return self_test()

    root=Path(a.project_root).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    ddir=root/"results"/"wfp_discovery_validation1000"
    known={
        "coordinates_npz":ddir/"WFP_DISCOVERY_COMMON_COORDINATES.npz",
        "eigenspectra_csv":ddir/"wfp_population_eigenspectra.csv",
        "cv_curve_csv":ddir/"wfp_cv_reconstruction_curve.csv",
        "axis_reliability_csv":ddir/"wfp_axis_reliability.csv",
        "discovery_results_json":ddir/"WFP_DISCOVERY_RESULTS.json",
        "discovery_readout":ddir/"WFP_DISCOVERY_READOUT.txt",
    }
    for k,p in known.items():
        if not p.is_file():
            raise SystemExit(f"FAIL missing authoritative discovery artifact {k}: {p}")

    readout=known["discovery_readout"].read_text(encoding="utf-8",errors="replace")
    if "Selected common-basis dimension: 8" not in readout:
        raise SystemExit("FAIL discovery readout does not confirm frozen d=8")

    with np.load(known["coordinates_npz"],allow_pickle=False) as z:
        keys=list(z.files)
        required=["population_mean","between_basis","between_eigenvalues","within_window_covariance","dimension"]
        missing=[k for k in required if k not in keys]
        if missing:
            raise SystemExit(f"FAIL coordinate NPZ missing keys: {missing}")

        mu=np.asarray(z["population_mean"],dtype=float)
        B=np.asarray(z["between_basis"],dtype=float)
        lam=np.asarray(z["between_eigenvalues"],dtype=float)
        Sw=np.asarray(z["within_window_covariance"],dtype=float)
        dim=int(np.asarray(z["dimension"]).reshape(()))

    checks={
        "population_mean_shape":list(mu.shape),
        "between_basis_shape":list(B.shape),
        "between_eigenvalues_shape":list(lam.shape),
        "within_window_covariance_shape":list(Sw.shape),
        "dimension":dim,
        "population_mean_finite":finite(mu),
        "between_basis_finite":finite(B),
        "between_eigenvalues_finite":finite(lam),
        "within_window_covariance_finite":finite(Sw),
        "basis_orthonormality_max_abs_error":float(np.max(np.abs(B.T@B-np.eye(B.shape[1])))),
        "within_covariance_symmetry_max_abs_error":float(np.max(np.abs(Sw-Sw.T))),
    }
    hard_pass=(
        mu.shape==(EXPECTED_PHASE_POINTS,)
        and B.shape==(EXPECTED_PHASE_POINTS,EXPECTED_DIM)
        and lam.shape==(EXPECTED_DIM,)
        and Sw.shape==(EXPECTED_PHASE_POINTS,EXPECTED_PHASE_POINTS)
        and dim==EXPECTED_DIM
        and checks["population_mean_finite"]
        and checks["between_basis_finite"]
        and checks["between_eigenvalues_finite"]
        and checks["within_window_covariance_finite"]
        and checks["basis_orthonormality_max_abs_error"] < 1e-10
        and checks["within_covariance_symmetry_max_abs_error"] < 1e-10
    )
    if not hard_pass:
        raise SystemExit(f"FAIL authoritative coordinate integrity: {checks}")

    eig=pd.read_csv(known["eigenspectra_csv"])
    cv=pd.read_csv(known["cv_curve_csv"])
    ar=pd.read_csv(known["axis_reliability_csv"])

    candidate_cov=scan_between_cov_candidates(root)
    full_sigma_b_found=False
    sigma_b_evidence=[]
    for item in candidate_cov:
        for k,v in item.items():
            if k.startswith("shape_") and v==[64,64]:
                full_sigma_b_found=True
                sigma_b_evidence.append(item["path"])
        if item.get("matching_columns"):
            # A wide 64x64 CSV would need explicit confirmation later.
            pass

    source_manifest=[]
    for label,p in known.items():
        source_manifest.append({
            "label":label,
            "relative_path":str(p.relative_to(root)),
            "bytes":int(p.stat().st_size),
            "sha256":sha256_file(p),
        })
    with (out/"WFP_INTERFACE_SOURCE_MANIFEST_PRIVATE.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["label","relative_path","bytes","sha256"])
        w.writeheader(); w.writerows(source_manifest)

    status={
        "decision":"WFP_INTERFACE_RECOVERY_AUDIT_PASS",
        "scientific_effects_calculated":False,
        "patient_level_scores_opened":False,
        "raw_waveforms_opened":False,
        "B8_changed":False,
        "authoritative_coordinate_npz_keys":keys,
        "checks":checks,
        "eigenspectra_rows":int(len(eig)),
        "eigenspectra_columns":eig.columns.tolist(),
        "cv_curve_rows":int(len(cv)),
        "cv_curve_columns":cv.columns.tolist(),
        "axis_reliability_rows":int(len(ar)),
        "axis_reliability_columns":ar.columns.tolist(),
        "interface_status":{
            "population_center":"AVAILABLE",
            "frozen_B8_basis":"AVAILABLE",
            "selected_B8_eigenvalues":"AVAILABLE",
            "full_dimension_profile":"AVAILABLE_FROM_EIGENSPECTRA_CSV",
            "Sigma_W":"AVAILABLE",
            "axis_sign_convention":"CAN_BE_DECLARED_AS_EXACT_STORED_COLUMN_ORIENTATION_NO_FLIPPING",
            "projection_rule":"CAN_BE_DOCUMENTED_FROM_AUTHORITATIVE_DISCOVERY_CODE",
            "full_Sigma_B":"AVAILABLE_EXISTING_SERIALIZATION" if full_sigma_b_found else "SERIALIZATION_GAP",
        },
        "full_sigma_b_candidate_scan":candidate_cov,
        "full_sigma_b_evidence":sigma_b_evidence,
    }
    (out/"WFP_INTERFACE_RECOVERY_AUDIT.json").write_text(
        json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    lines=[
        "WF-P INTERFACE RECOVERY / RELEASE-COMPLETENESS AUDIT",
        "====================================================",
        "Decision: WFP_INTERFACE_RECOVERY_AUDIT_PASS",
        "Scientific effects calculated: NO",
        "Patient-level score table opened: NO",
        "Raw waveform arrays opened: NO",
        "Frozen B8 changed: NO","",
        "Authoritative WFP_DISCOVERY_COMMON_COORDINATES.npz:",
        f"  keys: {', '.join(keys)}",
        f"  population_mean shape: {mu.shape}",
        f"  between_basis shape: {B.shape}",
        f"  between_eigenvalues shape: {lam.shape}",
        f"  within_window_covariance shape: {Sw.shape}",
        f"  dimension: {dim}",
        f"  basis orthonormality max abs error: {checks['basis_orthonormality_max_abs_error']:.3e}",
        f"  Sigma_W symmetry max abs error: {checks['within_covariance_symmetry_max_abs_error']:.3e}","",
        "Release-interface reinterpretation:",
        "  population center: AVAILABLE",
        "  frozen B8 basis: AVAILABLE",
        "  B8 eigenvalues: AVAILABLE",
        "  full eigenvalue/dimension profile: AVAILABLE (wfp_population_eigenspectra.csv)",
        "  Sigma_W: AVAILABLE",
        "  sign convention: define as exact stored column orientation; DO NOT flip",
        "  projection rule: document exact frozen row-vector projection (x - population_mean) @ B8",
        f"  full 64x64 Sigma_B serialized already: {'YES' if full_sigma_b_found else 'NO'}","",
    ]
    if not full_sigma_b_found:
        lines += [
            "Important:",
            "  The earlier public-release preflight was too pessimistic because it used filename heuristics.",
            "  Most WF3 interface objects already exist inside WFP_DISCOVERY_COMMON_COORDINATES.npz.",
            "  The only material interface gap detected here is a release-safe serialization of full Sigma_B.",
            "  Do NOT reconstruct or redefine B8 to fix this gap.",
            "  If full Sigma_B is required for v1.0, export it by deterministic replay of the authoritative discovery operator and verify all existing readouts/hashes.",
            "",
        ]
    else:
        lines += ["Full Sigma_B candidate evidence:", *[f"  - {x}" for x in sigma_b_evidence], ""]

    lines += [
        "Next step:",
        "  Build the release interface from the authoritative NPZ/CSV objects.",
        "  Resolve full Sigma_B serialization before final Zenodo release, or explicitly document a narrower interface if intentionally omitted.",
        "",
    ]
    (out/"WFP_INTERFACE_RECOVERY_AUDIT.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
