#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

N_SOURCE=1000
N_ANALYSABLE=978
P=64
D=8
TOL_ARRAY=5e-12
TOL_EIG=5e-10
TOL_BASIS=5e-9

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

def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,str(path))
    if s is None or s.loader is None: raise RuntimeError(f"cannot load {path}")
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def write_vec(path,x,name):
    pd.DataFrame({"index":np.arange(1,len(x)+1),name:np.asarray(x,float)}).to_csv(path,index=False)

def write_mat(path,A,prefix):
    A=np.asarray(A,float)
    df=pd.DataFrame(A,columns=[f"{prefix}{j+1:02d}" for j in range(A.shape[1])])
    df.insert(0,"row",np.arange(1,A.shape[0]+1))
    df.to_csv(path,index=False)

def sign_align(U,B):
    corr=np.sum(U*B,axis=0)
    s=np.where(corr>=0,1.0,-1.0)
    return U*s[None,:],corr,s

def self_test():
    rng=np.random.default_rng(1)
    Q,_=np.linalg.qr(rng.normal(size=(64,8)))
    U=Q.copy(); U[:,[1,4]]*=-1
    Ua,_,_=sign_align(U,Q)
    if np.max(np.abs(Ua-Q))>1e-12: raise RuntimeError("sign-align self-test")
    print("WF-P interface serializer self-test: PASS")
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default="~/Documents/abp_information_study")
    ap.add_argument("--input",default="~/Documents/abp_information_study/data/abp125_validation1000")
    ap.add_argument("--spec")
    ap.add_argument("--out",default="~/Documents/abp_information_study/results/wfp_release_interface_v1")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test: return self_test()
    if not a.spec: raise SystemExit("Missing required --spec")

    root=Path(a.project_root).expanduser().resolve()
    inp=Path(a.input).expanduser().resolve()
    specp=Path(a.spec).expanduser().resolve()
    out=Path(a.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    frozen=json.loads(specp.read_text())
    if frozen.get("status")!="FROZEN_BEFORE_RELEASE_SERIALIZATION_REPLAY":
        raise SystemExit("FAIL serialization frozen spec status")

    ddir=root/"results"/"wfp_discovery_validation1000"
    current={
      "discovery_script":root/"code"/"wfp_validation1000_discovery.py",
      "run1_script":root/"code"/"wp2_run1_development50.py",
      "discovery_frozen_spec":root/"freeze"/"WFP_DISCOVERY_FROZEN_SPEC.json",
      "discovery_results":ddir/"WFP_DISCOVERY_RESULTS.json",
      "discovery_readout":ddir/"WFP_DISCOVERY_READOUT.txt",
      "common_coordinates":ddir/"WFP_DISCOVERY_COMMON_COORDINATES.npz",
      "population_eigenspectra":ddir/"wfp_population_eigenspectra.csv",
      "cv_reconstruction_curve":ddir/"wfp_cv_reconstruction_curve.csv",
      "axis_reliability":ddir/"wfp_axis_reliability.csv",
      "serializer_script":Path(__file__).resolve(),
    }
    for k,p in current.items():
        if not p.is_file(): raise SystemExit(f"FAIL missing {k}: {p}")
        if frozen["source_hashes"].get(k)!=sha(p): raise SystemExit(f"FAIL hash mismatch {k}")

    cases=sorted((inp/"cases").glob("*.npz")); names=[p.name for p in cases]
    if len(cases)!=N_SOURCE or sha_lines(names)!=frozen["case_filename_manifest_sha256"]:
        raise SystemExit("FAIL frozen case filename set changed")

    discovery=load_module(current["discovery_script"],"wfp_discovery_release_replay")
    run1=load_module(current["run1_script"],"wfp_run1_release_replay")

    patients=[]; failures=[]
    for i,p in enumerate(cases,1):
        try: patients.append(discovery.construct_patient(p,run1))
        except Exception as e: failures.append({"file":p.name,"error":repr(e)})
        if i%50==0 or i==len(cases): print(f"[interface replay] {i}/{len(cases)}",flush=True)

    if len(patients)!=N_ANALYSABLE: raise SystemExit(f"FAIL replay analysable n={len(patients)}")
    auth=json.loads(current["discovery_results"].read_text())
    af=sorted(str(x.get("file")) for x in auth.get("failures",[]))
    rf=sorted(str(x.get("file")) for x in failures)
    if af!=rf: raise SystemExit("FAIL replay failure identities differ")

    M=np.vstack([x["all_rep"] for x in patients])
    O=np.vstack([x["odd_rep"] for x in patients])
    E=np.vstack([x["even_rep"] for x in patients])
    Sw=np.mean(np.stack([x["within_cov"] for x in patients]),axis=0)
    Srep,_,_=discovery.sym_cross_operator(O,E)
    vals_rep,Urep=discovery.eig_sorted(Srep)
    mu=np.mean(M,axis=0)
    X=M-mu
    SigmaB=X.T@X/(len(M)-1)
    vals_B,_=discovery.eig_sorted(SigmaB)
    Srep_pos=Urep@np.diag(np.clip(vals_rep,0,None))@Urep.T

    with np.load(current["common_coordinates"],allow_pickle=False) as z:
        mu0=np.asarray(z["population_mean"],float)
        B0=np.asarray(z["between_basis"],float)
        lam0=np.asarray(z["between_eigenvalues"],float)
        Sw0=np.asarray(z["within_window_covariance"],float)
        d0=int(np.asarray(z["dimension"]).reshape(()))
    if d0!=D or mu0.shape!=(P,) or B0.shape!=(P,D) or Sw0.shape!=(P,P):
        raise SystemExit("FAIL stored coordinate shapes")

    eig=pd.read_csv(current["population_eigenspectra"])
    rep_csv=eig["replicate_eigenvalue"].to_numpy(float)
    ord_csv=eig["ordinary_patient_mean_eigenvalue"].to_numpy(float)
    Ua,corr,signs=sign_align(Urep[:,:D],B0)

    chk={
      "population_mean_max_abs_error":float(np.max(np.abs(mu-mu0))),
      "Sigma_W_max_abs_error":float(np.max(np.abs(Sw-Sw0))),
      "selected_replicate_eigenvalue_max_abs_error":float(np.max(np.abs(vals_rep[:D]-lam0))),
      "full_replicate_eigenspectrum_max_abs_error":float(np.max(np.abs(vals_rep-rep_csv))),
      "full_ordinary_eigenspectrum_max_abs_error":float(np.max(np.abs(vals_B-ord_csv))),
      "B8_abs_axis_correlations":[float(abs(x)) for x in corr],
      "B8_sign_aligned_max_abs_error":float(np.max(np.abs(Ua-B0))),
      "B8_projector_max_abs_error":float(np.max(np.abs(Urep[:,:D]@Urep[:,:D].T-B0@B0.T))),
      "Sigma_B_symmetry_max_abs_error":float(np.max(np.abs(SigmaB-SigmaB.T))),
      "S_rep_symmetry_max_abs_error":float(np.max(np.abs(Srep-Srep.T))),
      "S_rep_positive_min_eigenvalue":float(np.min(np.linalg.eigvalsh(Srep_pos))),
    }
    if chk["population_mean_max_abs_error"]>TOL_ARRAY: raise SystemExit(f"FAIL mean replay {chk}")
    if chk["Sigma_W_max_abs_error"]>TOL_ARRAY: raise SystemExit(f"FAIL SigmaW replay {chk}")
    if chk["selected_replicate_eigenvalue_max_abs_error"]>TOL_EIG: raise SystemExit(f"FAIL eigen replay {chk}")
    if chk["full_replicate_eigenspectrum_max_abs_error"]>TOL_EIG: raise SystemExit(f"FAIL Srep spectrum {chk}")
    if chk["full_ordinary_eigenspectrum_max_abs_error"]>TOL_EIG: raise SystemExit(f"FAIL SigmaB spectrum {chk}")
    if min(chk["B8_abs_axis_correlations"])<1-1e-8: raise SystemExit(f"FAIL B8 axis replay {chk}")
    if chk["B8_sign_aligned_max_abs_error"]>TOL_BASIS or chk["B8_projector_max_abs_error"]>TOL_BASIS:
        raise SystemExit(f"FAIL B8 replay {chk}")

    # Aggregate release-safe outputs only.
    write_vec(out/"population_center_64.csv",mu0,"population_center")
    write_mat(out/"frozen_B8_basis_64x8.csv",B0,"z")
    write_vec(out/"selected_replicate_eigenvalues_8.csv",lam0,"eigenvalue")
    write_mat(out/"Sigma_W_short_window_64x64.csv",Sw0,"p")
    write_mat(out/"Sigma_B_ordinary_64x64.csv",SigmaB,"p")
    write_mat(out/"S_rep_replicate_corrected_64x64.csv",Srep,"p")
    write_mat(out/"S_rep_positive_64x64.csv",Srep_pos,"p")
    (out/"population_eigenspectra.csv").write_bytes(current["population_eigenspectra"].read_bytes())
    (out/"cv_reconstruction_curve.csv").write_bytes(current["cv_reconstruction_curve"].read_bytes())
    (out/"axis_reliability.csv").write_bytes(current["axis_reliability"].read_bytes())

    (out/"axis_sign_convention.json").write_text(json.dumps({
      "version":"wfp-interface-v1.0.0",
      "rule":"Use exact stored column orientation from WFP_DISCOVERY_COMMON_COORDINATES.npz; no post-hoc sign flip.",
      "axis_order":[f"z{i}" for i in range(1,9)],
      "replay_absolute_axis_correlations":chk["B8_abs_axis_correlations"],
      "replay_signs_for_integrity_only":[int(x) for x in signs],
    },indent=2,sort_keys=True)+"\n")

    (out/"projection_spec.json").write_text(json.dumps({
      "version":"wfp-interface-v1.0.0",
      "phase_points":64,
      "phase_endpoint":False,
      "input_role":"64-vector block/patient central morphology in inherited shape-normalized phase representation",
      "central_vector_normalization":"subtract phase mean, divide by phase population SD (numpy std, ddof=0); block/patient central vectors are renormalized after averaging",
      "projection_row_vector_formula":"z = (x64 - population_center) @ frozen_B8_basis",
      "equivalent_column_formula":"z = B8.T @ (x64 - population_center)",
      "B8_relearning_allowed":False,
      "patient_specific_rotation_allowed":False,
    },indent=2,sort_keys=True)+"\n")

    readme=f"""# WF-P frozen B8 interface v1.0.0

This directory contains release-safe aggregate objects for applying the frozen
WF-P population morphology coordinate system in later work, including WF3.

## Key files

- `population_center_64.csv`: frozen 64-vector population center.
- `frozen_B8_basis_64x8.csv`: authoritative frozen 64x8 B8 basis.
- `selected_replicate_eigenvalues_8.csv`: eigenvalues corresponding to B8.
- `population_eigenspectra.csv`: full replicate-corrected and ordinary spectra.
- `Sigma_W_short_window_64x64.csv`: short-window within-person covariance.
- `Sigma_B_ordinary_64x64.csv`: ordinary covariance of 30-min patient central morphology.
- `S_rep_replicate_corrected_64x64.csv`: replicate-corrected symmetric between-person operator used as the final discovery primary operator.
- `S_rep_positive_64x64.csv`: positive-spectrum PSD form of `S_rep`.

Both ordinary `Sigma_B` and replicate-corrected `S_rep` are included because
they are distinct objects with distinct scientific roles.

## Projection

For an already normalized 64-vector central morphology `x64`:

`z = (x64 - population_center) @ frozen_B8_basis`

Do not project raw mmHg waveform samples directly.

## Orientation

Axis order is z1...z8. The exported basis preserves the exact stored
orientation from the frozen discovery artifact. No post-hoc sign flip, rotation,
or relearning is allowed.

## Replay integrity

- analysable n: {len(patients)}
- frozen-rule exclusions: {len(failures)}
- population-center max error: {chk['population_mean_max_abs_error']:.3e}
- Sigma_W max error: {chk['Sigma_W_max_abs_error']:.3e}
- selected-eigenvalue max error: {chk['selected_replicate_eigenvalue_max_abs_error']:.3e}
- full S_rep spectrum max error: {chk['full_replicate_eigenspectrum_max_abs_error']:.3e}
- full ordinary spectrum max error: {chk['full_ordinary_eigenspectrum_max_abs_error']:.3e}
- B8 projector max error: {chk['B8_projector_max_abs_error']:.3e}

## Interpretation boundary

This interface does not label any B8 axis as physiological state or stable
trait, and does not imply constitutional independence, disease-specific meaning,
or treatment-response meaning.
"""
    (out/"README.md").write_text(readme)

    (out/"WFP_B8_INTERFACE_v1.0.0.json").write_text(json.dumps({
      "interface_version":"wfp-interface-v1.0.0",
      "scientific_effects_calculated_by_serialization":False,
      "B8_changed":False,
      "patient_level_outputs_written":False,
      "source_cohort":"Validation1000",
      "source_n":N_SOURCE,
      "analysable_n":N_ANALYSABLE,
      "authoritative_discovery_decision":auth.get("decision"),
      "authoritative_primary_operator":auth.get("primary_operator"),
      "serialization_frozen_spec_sha256":sha(specp),
      "replay_checks":chk,
    },indent=2,sort_keys=True)+"\n")

    rows=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!="INTERFACE_SHA256.csv":
            rows.append({"file":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})
    with (out/"INTERFACE_SHA256.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["file","bytes","sha256"]); w.writeheader(); w.writerows(rows)

    txt="\n".join([
      "WF-P RELEASE-SAFE INTERFACE SERIALIZATION",
      "=========================================",
      "Decision: WFP_RELEASE_INTERFACE_SERIALIZATION_PASS",
      "Scientific effects calculated: NO",
      "Frozen B8 changed: NO",
      "Patient-level outputs written: NO",
      f"Analysable replay n: {len(patients)}",
      f"Frozen-rule exclusions replayed: {len(failures)}",
      "",
      "Aggregate objects written:",
      "  population_center_64.csv",
      "  frozen_B8_basis_64x8.csv",
      "  selected_replicate_eigenvalues_8.csv",
      "  population_eigenspectra.csv",
      "  Sigma_W_short_window_64x64.csv",
      "  Sigma_B_ordinary_64x64.csv",
      "  S_rep_replicate_corrected_64x64.csv",
      "  S_rep_positive_64x64.csv",
      "  projection_spec.json",
      "  axis_sign_convention.json",
      "  WFP_B8_INTERFACE_v1.0.0.json",
      "  README.md",
      "  INTERFACE_SHA256.csv",
      "",
      "Replay integrity:",
      f"  population center max error: {chk['population_mean_max_abs_error']:.3e}",
      f"  Sigma_W max error: {chk['Sigma_W_max_abs_error']:.3e}",
      f"  selected eigenvalues max error: {chk['selected_replicate_eigenvalue_max_abs_error']:.3e}",
      f"  full S_rep eigenspectrum max error: {chk['full_replicate_eigenspectrum_max_abs_error']:.3e}",
      f"  full ordinary eigenspectrum max error: {chk['full_ordinary_eigenspectrum_max_abs_error']:.3e}",
      f"  B8 projector max error: {chk['B8_projector_max_abs_error']:.3e}",
      "",
      "Next step:",
      "  Public-safety audit of this interface directory, then GitHub staging.",
      "",
    ])
    (out/"WFP_RELEASE_INTERFACE_SERIALIZATION_READOUT.txt").write_text(txt)
    print(txt)
    return 0

if __name__=="__main__": raise SystemExit(main())
