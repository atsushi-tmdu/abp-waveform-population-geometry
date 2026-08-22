#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 4B — frozen comparator geometry audit.

Scientific role: descriptive follow-up within the already-open Validation1000
discovery cohort. No clinical labels.

Questions fixed before execution:
1) Is the selected d=8 replicate-corrected subspace geometrically the same as
   the d=8 ordinary patient-mean PCA subspace?
2) How different is the d=8 replicate subspace from the fixed Fourier d=8 subspace?
3) What are the already-prespecified held-out reconstruction differences at d=8?

No dimension reselection and no new basis is promoted to primary.
"""

from __future__ import annotations

from pathlib import Path
import argparse, hashlib, importlib.util, json
import numpy as np
import pandas as pd

D = 8
EXPECTED_DISCOVERY_SCRIPT_SHA256 = "a928ae3c3a81ebf9ba662cbde819d4384c7c8b13d96565ce29f32e4315d1c4ca"
EXPECTED_RUN1_SHA256 = "811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def deterministic_sign(U):
    U=np.asarray(U,float).copy()
    for j in range(U.shape[1]):
        k=int(np.argmax(np.abs(U[:,j])))
        if U[k,j] < 0:
            U[:,j] *= -1
    return U

def pca_basis(M):
    X=M-np.mean(M,axis=0)
    _,_,vt=np.linalg.svd(X,full_matrices=False)
    return deterministic_sign(vt.T)

def projector_overlap(A,B):
    d=min(A.shape[1],B.shape[1])
    return float(np.sum((A[:,:d].T @ B[:,:d])**2)/d)

def principal_angles_deg(A,B):
    s=np.linalg.svd(A.T@B,compute_uv=False)
    return np.degrees(np.arccos(np.clip(s,-1,1)))

def subspace_capture(M,U):
    mu=np.mean(M,axis=0)
    X=M-mu
    total=float(np.sum(X*X))
    proj=X@U@U.T
    return float(np.sum(proj*proj)/total) if total>0 else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--discovery-results", required=True)
    ap.add_argument("--discovery-script", required=True)
    ap.add_argument("--run1-script", required=True)
    ap.add_argument("--output-lock", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    project=Path(args.project_root).expanduser().resolve()
    input_dir=Path(args.input).expanduser().resolve()
    results=Path(args.discovery_results).expanduser().resolve()
    discovery_script=Path(args.discovery_script).expanduser().resolve()
    run1_path=Path(args.run1_script).expanduser().resolve()
    lock_path=Path(args.output_lock).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if sha256_file(discovery_script) != EXPECTED_DISCOVERY_SCRIPT_SHA256:
        raise SystemExit("FAIL: discovery script hash mismatch")
    if sha256_file(run1_path) != EXPECTED_RUN1_SHA256:
        raise SystemExit("FAIL: Run-1 hash mismatch")

    lock=json.loads(lock_path.read_text())
    if lock.get("decision") != "WFP_DISCOVERY_OUTPUT_LOCK_PASS":
        raise SystemExit("FAIL: discovery output lock not PASS")

    disc=load_module(discovery_script,"wfp_discovery_frozen")
    run1=load_module(run1_path,"wfp_run1_frozen")

    coord=np.load(results/"WFP_DISCOVERY_COMMON_COORDINATES.npz",allow_pickle=False)
    Urep=np.asarray(coord["between_basis"],float)
    if Urep.shape != (64,D):
        raise SystemExit(f"FAIL: expected replicate basis 64x8, got {Urep.shape}")

    cases=sorted((input_dir/"cases").glob("*.npz"))
    if len(cases) != 1000:
        raise SystemExit(f"FAIL: expected 1000 cases, found {len(cases)}")

    patients=[]
    failures=[]
    for p in cases:
        try:
            patients.append(disc.construct_patient(p,run1))
        except Exception as e:
            failures.append({"file":p.name,"error":repr(e)})

    if len(patients) != 978 or len(failures) != 22:
        raise SystemExit(
            f"FAIL: frozen-rule cohort identity changed: analysable={len(patients)}, failures={len(failures)}"
        )

    M=np.vstack([x["all_rep"] for x in patients])
    Upca=pca_basis(M)[:,:D]
    Uf=disc.fourier_basis()[:,:D]

    ang_pca=principal_angles_deg(Urep,Upca)
    ang_fourier=principal_angles_deg(Urep,Uf)

    cv=pd.read_csv(results/"wfp_cv_reconstruction_curve.csv")
    fourier=pd.read_csv(results/"wfp_fourier_comparator_curve.csv")
    r=cv.loc[cv["dimension"]==D].iloc[0]
    f=fourier.loc[fourier["dimension"]==D].iloc[0]

    summary={
        "schema_version":1,
        "work_package":"WF-P",
        "stage":"4B",
        "decision":"WFP_GEOMETRY_COMPARATOR_AUDIT_COMPLETE",
        "scientific_role":"discovery_descriptive_followup_only",
        "source_n":1000,
        "analysable_n":978,
        "dimension_fixed":D,
        "clinical_labels_accessed":False,
        "dimension_reselected":False,
        "primary_basis_changed":False,
        "replicate_vs_ordinary_pca":{
            "projector_overlap":projector_overlap(Urep,Upca),
            "principal_angles_degrees":[float(x) for x in ang_pca],
            "max_principal_angle_degrees":float(np.max(ang_pca)),
            "median_principal_angle_degrees":float(np.median(ang_pca)),
            "replicate_fullsample_variance_capture":subspace_capture(M,Urep),
            "ordinary_pca_fullsample_variance_capture":subspace_capture(M,Upca),
            "cv_r2_replicate_d8":float(r["replicate_basis_cv_r2_all"]),
            "cv_r2_ordinary_pca_d8":float(r["ordinary_pca_cv_r2_all"]),
            "cv_r2_difference_replicate_minus_pca":
                float(r["replicate_basis_cv_r2_all"]-r["ordinary_pca_cv_r2_all"]),
        },
        "replicate_vs_fourier":{
            "projector_overlap":projector_overlap(Urep,Uf),
            "principal_angles_degrees":[float(x) for x in ang_fourier],
            "max_principal_angle_degrees":float(np.max(ang_fourier)),
            "median_principal_angle_degrees":float(np.median(ang_fourier)),
            "cv_r2_replicate_d8":float(r["replicate_basis_cv_r2_all"]),
            "cv_r2_fourier_d8":float(f["fourier_cv_r2_all"]),
            "cv_r2_difference_replicate_minus_fourier":
                float(r["replicate_basis_cv_r2_all"]-f["fourier_cv_r2_all"]),
        },
        "boundary":[
            "No age/sex or clinical labels were accessed.",
            "No new dimension was selected.",
            "The replicate-corrected d=8 basis remains the frozen WF-P discovery coordinate system.",
            "This remains discovery evidence."
        ],
        "input_hashes":{
            "discovery_output_lock_sha256":sha256_file(lock_path),
            "discovery_script_sha256":sha256_file(discovery_script),
            "run1_script_sha256":sha256_file(run1_path),
            "coordinates_sha256":sha256_file(results/"WFP_DISCOVERY_COMMON_COORDINATES.npz"),
        }
    }

    out_json=out/"WFP_GEOMETRY_COMPARATOR_AUDIT.json"
    out_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

    a=summary["replicate_vs_ordinary_pca"]
    b=summary["replicate_vs_fourier"]
    lines=[
        "WF-P DISCOVERY GEOMETRY COMPARATOR AUDIT",
        "========================================",
        "Decision: WFP_GEOMETRY_COMPARATOR_AUDIT_COMPLETE",
        "Scientific role: DISCOVERY DESCRIPTIVE FOLLOW-UP ONLY",
        "Dimension fixed: 8",
        "Clinical labels accessed: NO",
        "Dimension reselected: NO",
        "Primary basis changed: NO",
        "",
        "Replicate basis vs ordinary PCA (d=8):",
        f"  projector overlap: {a['projector_overlap']}",
        f"  principal angles (deg): {a['principal_angles_degrees']}",
        f"  max principal angle (deg): {a['max_principal_angle_degrees']}",
        f"  median principal angle (deg): {a['median_principal_angle_degrees']}",
        f"  full-sample variance capture, replicate: {a['replicate_fullsample_variance_capture']}",
        f"  full-sample variance capture, PCA: {a['ordinary_pca_fullsample_variance_capture']}",
        f"  CV R2 difference replicate - PCA: {a['cv_r2_difference_replicate_minus_pca']}",
        "",
        "Replicate basis vs Fourier (d=8):",
        f"  projector overlap: {b['projector_overlap']}",
        f"  principal angles (deg): {b['principal_angles_degrees']}",
        f"  max principal angle (deg): {b['max_principal_angle_degrees']}",
        f"  median principal angle (deg): {b['median_principal_angle_degrees']}",
        f"  CV R2 replicate: {b['cv_r2_replicate_d8']}",
        f"  CV R2 Fourier: {b['cv_r2_fourier_d8']}",
        f"  CV R2 difference replicate - Fourier: {b['cv_r2_difference_replicate_minus_fourier']}",
        "",
        "Boundary:",
        "  Replicate-corrected d=8 remains the frozen WF-P discovery basis.",
        "  No clinical interpretation of axes is authorized at this stage.",
    ]
    (out/"WFP_GEOMETRY_COMPARATOR_AUDIT.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
