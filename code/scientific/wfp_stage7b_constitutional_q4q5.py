#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7B / WF-P2 constitutional Q4/Q5 analysis.

Primary:
- age + sex -> frozen B8 (full n=978)
- Q5 residual geometry after cross-fitted age+sex conditioning

Secondary:
- age + age^2 + sex sensitivity
- conventional factors + age + sex incremental analysis
- age + sex + height complete-case sensitivity (n=693; no imputation)

Chronic phenotype mapping is deferred to Stage 7C.
Frozen B8 is never relearned or rotated for the primary analysis.
"""

from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_N = 978
EXPECTED_HEIGHT_N = 693
D = 8
CV_FOLDS = 5
CV_SEED = 20260820

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_fold(pid: str) -> int:
    key = f"{CV_SEED}:{pid}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % CV_FOLDS

def make_design(df: pd.DataFrame, kind: str) -> np.ndarray:
    age = df["age_years_capped90"].to_numpy(float)
    female = df["female"].to_numpy(float)
    conv = df[["level_mmhg","log_scale_sd","log_duration_sec"]].to_numpy(float)

    if kind == "age_sex":
        return np.column_stack([age, female])
    if kind == "age2_sex":
        return np.column_stack([age, age**2, female])
    if kind == "conventional":
        return conv
    if kind == "conventional_plus_age_sex":
        return np.column_stack([conv, age, female])
    if kind == "age_sex_height":
        return np.column_stack([age, female, df["height_median_cm"].to_numpy(float)])
    if kind == "conventional_plus_age_sex_height":
        return np.column_stack([conv, age, female, df["height_median_cm"].to_numpy(float)])
    raise ValueError(kind)

def standardize_train_test(Xtr, Xte):
    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (Xtr-mu)/sd, (Xte-mu)/sd

def oof_multivariate(df, Z, kind):
    folds = np.asarray([stable_fold(pid) for pid in df["patient_id"].astype(str)])
    X = make_design(df, kind)
    pred = np.full_like(Z, np.nan, dtype=float)
    for f in range(CV_FOLDS):
        te = folds == f
        tr = ~te
        Xtr, Xte = standardize_train_test(X[tr], X[te])
        Atr = np.column_stack([np.ones(np.sum(tr)), Xtr])
        Ate = np.column_stack([np.ones(np.sum(te)), Xte])
        beta = np.linalg.lstsq(Atr, Z[tr], rcond=None)[0]
        pred[te] = Ate @ beta
    if not np.all(np.isfinite(pred)):
        raise RuntimeError(f"nonfinite OOF predictions for {kind}")
    return pred

def aggregate_r2(Z, pred):
    sse = float(np.sum((Z-pred)**2))
    zc = Z - np.mean(Z, axis=0, keepdims=True)
    sst = float(np.sum(zc**2))
    return float(1-sse/sst)

def axis_r2(Z, pred):
    out=[]
    for j in range(Z.shape[1]):
        y=Z[:,j]; p=pred[:,j]
        sse=float(np.sum((y-p)**2))
        sst=float(np.sum((y-np.mean(y))**2))
        out.append(float(1-sse/sst))
    return out

def covariance_metrics(X):
    C=np.cov(np.asarray(X,float), rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(0.5*(C+C.T))
    order=np.argsort(vals)[::-1]
    vals=np.maximum(vals[order],0.0); vecs=vecs[:,order]
    total=float(np.sum(vals))
    p=vals[vals>0]/total
    er=float(np.exp(-np.sum(p*np.log(p))))
    c=np.cumsum(vals)/total
    return {
        "trace": total,
        "effective_rank": er,
        "d90": int(np.searchsorted(c,0.90)+1),
        "d95": int(np.searchsorted(c,0.95)+1),
        "eigenvalues": [float(v) for v in vals],
        "eigenvectors": vecs,
    }

def self_test():
    rng=np.random.default_rng(20260821)
    n=300
    df=pd.DataFrame({
        "patient_id":[f"p{i:06d}" for i in range(n)],
        "age_years_capped90":rng.normal(65,12,n),
        "female":rng.integers(0,2,n),
        "level_mmhg":rng.normal(85,10,n),
        "log_scale_sd":rng.normal(2.5,0.2,n),
        "log_duration_sec":rng.normal(-0.1,0.15,n),
        "height_median_cm":rng.normal(170,9,n),
    })
    age=df["age_years_capped90"].to_numpy()
    fem=df["female"].to_numpy()
    Z=np.column_stack([
        0.7*(age-age.mean())/age.std()+0.4*fem+rng.normal(scale=0.3,size=n)
        for _ in range(D)
    ])
    p=oof_multivariate(df,Z,"age_sex")
    if aggregate_r2(Z,p) < 0.5:
        raise RuntimeError("self-test failed")
    print("WF-P Stage7B constitutional Q4/Q5 self-test: PASS")
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--discovery-results", required=False, default="~/Documents/abp_information_study/results/wfp_discovery_validation1000")
    ap.add_argument("--temporal-linkage", required=False, default="~/Documents/abp_information_study/results/wfp_temporal_linkage_audit/WFP_TEMPORAL_LINKAGE_AUDIT_PRIVATE.csv")
    ap.add_argument("--height-preflight", required=False, default="~/Documents/abp_information_study/results/wfp_height_preflight/WFP_HEIGHT_PREFLIGHT_PRIVATE.csv")
    ap.add_argument("--wfp0-results", required=False, default="~/Documents/abp_information_study/results/wfp0_minimal_identifiability")
    ap.add_argument("--spec", required=False, default="~/Documents/abp_information_study/freeze/wfp_stage7b_constitutional/WFP_STAGE7B_CONSTITUTIONAL_FROZEN_SPEC.json")
    ap.add_argument("--out", required=False, default="~/Documents/abp_information_study/results/wfp_stage7b_constitutional")
    ap.add_argument("--self-test", action="store_true")
    args=ap.parse_args()
    if args.self_test:
        return self_test()

    script_path=Path(__file__).resolve()
    discovery=Path(args.discovery_results).expanduser().resolve()
    temporal_path=Path(args.temporal_linkage).expanduser().resolve()
    height_path=Path(args.height_preflight).expanduser().resolve()
    wfp0=Path(args.wfp0_results).expanduser().resolve()
    spec_path=Path(args.spec).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    score_path=discovery/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    coord_path=discovery/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    p0fac=wfp0/"wfp0_patient_conventional_factors_PRIVATE.csv"
    p0json=wfp0/"WFP0_MINIMAL_IDENTIFIABILITY_RESULTS.json"

    spec=json.loads(spec_path.read_text())
    if spec.get("status")!="FROZEN_BEFORE_STAGE7B_CONSTITUTIONAL_ASSOCIATION":
        raise SystemExit("FAIL: Stage7B frozen spec status invalid")
    if spec.get("analysis_script_sha256")!=sha256_file(script_path):
        raise SystemExit("FAIL: Stage7B analysis script hash mismatch")

    inputs={
        "score_file_sha256":score_path,
        "coordinate_file_sha256":coord_path,
        "temporal_linkage_sha256":temporal_path,
        "height_private_sha256":height_path,
        "wfp0_factor_sha256":p0fac,
        "wfp0_result_sha256":p0json,
    }
    for key,p in inputs.items():
        if not p.is_file():
            raise SystemExit(f"FAIL missing {p}")
        if spec.get(key)!=sha256_file(p):
            raise SystemExit(f"FAIL input hash mismatch {key}")

    pj=json.loads(p0json.read_text())
    if pj.get("decision")!="GO_CONVENTIONAL_FACTORS_DO_NOT_DOMINATE_FROZEN_B8":
        raise SystemExit("FAIL: WF-P0 did not authorize Stage7B")

    scores=pd.read_csv(score_path,dtype={"patient_id":str})
    temporal=pd.read_csv(temporal_path,dtype={"patient_id":str})
    height=pd.read_csv(height_path,dtype={"patient_id":str})
    factors=pd.read_csv(p0fac,dtype={"patient_id":str})

    if len(scores)!=EXPECTED_N:
        raise SystemExit(f"FAIL expected {EXPECTED_N} score rows")

    df=(scores
        .merge(temporal[["patient_id","age_years_capped90","gender"]],on="patient_id",how="left",validate="one_to_one")
        .merge(height[["patient_id","height_median_cm"]],on="patient_id",how="left",validate="one_to_one")
        .merge(factors[["patient_id","level_mmhg","log_scale_sd","log_duration_sec"]],on="patient_id",how="left",validate="one_to_one")
    )
    if len(df)!=EXPECTED_N:
        raise SystemExit("FAIL merged cohort size")
    if df[["age_years_capped90","gender","level_mmhg","log_scale_sd","log_duration_sec"]].isna().any().any():
        raise SystemExit("FAIL primary variables incomplete")

    sex=df["gender"].astype(str).str.upper().str.strip()
    if (~sex.isin(["M","F"])).any():
        raise SystemExit("FAIL unexpected gender coding")
    df["female"]=(sex=="F").astype(float)

    Z=df[[f"z{j}" for j in range(1,D+1)]].to_numpy(float)

    with np.load(coord_path,allow_pickle=False) as z:
        B8=np.asarray(z["between_basis"],float)
        mu=np.asarray(z["population_mean"],float)
        dim=int(np.asarray(z["dimension"]).reshape(()))
    if B8.shape!=(64,D) or mu.shape!=(64,) or dim!=D:
        raise SystemExit("FAIL frozen coordinate dimensions")

    raw=covariance_metrics(Z)
    rows=[]
    preds={}
    for kind in ["age_sex","age2_sex","conventional","conventional_plus_age_sex"]:
        pred=oof_multivariate(df,Z,kind); preds[kind]=pred
        resid=Z-pred; rm=covariance_metrics(resid)
        row={
            "analysis_subset":"full_978",
            "model":kind,
            "n":len(df),
            "aggregate_oof_r2":aggregate_r2(Z,pred),
            "residual_trace_fraction_vs_raw":rm["trace"]/raw["trace"],
            "residual_effective_rank":rm["effective_rank"],
            "residual_d90":rm["d90"],
            "residual_d95":rm["d95"],
        }
        for j,v in enumerate(axis_r2(Z,pred),1):
            row[f"z{j}_oof_r2"]=v
        rows.append(row)

    hdf=df[df["height_median_cm"].notna()].copy()
    if len(hdf)!=EXPECTED_HEIGHT_N:
        raise SystemExit(f"FAIL expected height n={EXPECTED_HEIGHT_N}, found {len(hdf)}")
    Zh=hdf[[f"z{j}" for j in range(1,D+1)]].to_numpy(float)
    rawh=covariance_metrics(Zh)
    for kind in ["age_sex","age_sex_height","conventional","conventional_plus_age_sex_height"]:
        pred=oof_multivariate(hdf,Zh,kind)
        resid=Zh-pred; rm=covariance_metrics(resid)
        row={
            "analysis_subset":"height_complete_case_693",
            "model":kind,
            "n":len(hdf),
            "aggregate_oof_r2":aggregate_r2(Zh,pred),
            "residual_trace_fraction_vs_raw":rm["trace"]/rawh["trace"],
            "residual_effective_rank":rm["effective_rank"],
            "residual_d90":rm["d90"],
            "residual_d95":rm["d95"],
        }
        for j,v in enumerate(axis_r2(Zh,pred),1):
            row[f"z{j}_oof_r2"]=v
        rows.append(row)

    mdf=pd.DataFrame(rows)
    mdf.to_csv(out/"wfp_stage7b_model_summary.csv",index=False)

    age_row=mdf[(mdf.analysis_subset=="full_978")&(mdf.model=="age_sex")].iloc[0]
    age2_row=mdf[(mdf.analysis_subset=="full_978")&(mdf.model=="age2_sex")].iloc[0]
    conv_row=mdf[(mdf.analysis_subset=="full_978")&(mdf.model=="conventional")].iloc[0]
    comb_row=mdf[(mdf.analysis_subset=="full_978")&(mdf.model=="conventional_plus_age_sex")].iloc[0]

    hb=mdf[(mdf.analysis_subset=="height_complete_case_693")&(mdf.model=="age_sex")].iloc[0]
    hh=mdf[(mdf.analysis_subset=="height_complete_case_693")&(mdf.model=="age_sex_height")].iloc[0]
    hc=mdf[(mdf.analysis_subset=="height_complete_case_693")&(mdf.model=="conventional")].iloc[0]
    hcomb=mdf[(mdf.analysis_subset=="height_complete_case_693")&(mdf.model=="conventional_plus_age_sex_height")].iloc[0]

    primary_resid=Z-preds["age_sex"]
    rdf=pd.DataFrame(primary_resid,columns=[f"z{j}_resid_age_sex" for j in range(1,D+1)])
    rdf.insert(0,"patient_id",df["patient_id"].astype(str).to_numpy())
    rdf.to_csv(out/"wfp_stage7b_age_sex_residual_scores_PRIVATE.csv",index=False)

    rmet=covariance_metrics(primary_resid)
    Ures=np.asarray(rmet["eigenvectors"],float)
    np.savez_compressed(
        out/"WFP_STAGE7B_CONSTITUTIONAL_RESIDUAL_COORDINATES.npz",
        raw_population_mean=mu.astype(np.float64),
        raw_frozen_B8=B8.astype(np.float64),
        age_sex_residual_score_eigenvectors=Ures.astype(np.float64),
        age_sex_residual_phase_basis=(B8@Ures).astype(np.float64),
        age_sex_residual_score_eigenvalues=np.asarray(rmet["eigenvalues"],float),
    )

    result={
        "schema_version":1,
        "work_package":"WF-P2",
        "stage":"7B",
        "decision":"WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE",
        "source_n":EXPECTED_N,
        "height_complete_case_n":EXPECTED_HEIGHT_N,
        "frozen_B8_changed":False,
        "primary":{
            "model":"age + sex",
            "aggregate_oof_r2":float(age_row["aggregate_oof_r2"]),
            "residual_trace_fraction":float(age_row["residual_trace_fraction_vs_raw"]),
            "residual_effective_rank":float(age_row["residual_effective_rank"]),
            "residual_d90":int(age_row["residual_d90"]),
            "residual_d95":int(age_row["residual_d95"]),
        },
        "nonlinear_age_sensitivity":{
            "aggregate_oof_r2":float(age2_row["aggregate_oof_r2"]),
            "delta_r2_vs_primary":float(age2_row["aggregate_oof_r2"]-age_row["aggregate_oof_r2"]),
        },
        "incremental_beyond_conventional":{
            "conventional_oof_r2":float(conv_row["aggregate_oof_r2"]),
            "combined_conventional_age_sex_oof_r2":float(comb_row["aggregate_oof_r2"]),
            "incremental_age_sex_delta_r2":float(comb_row["aggregate_oof_r2"]-conv_row["aggregate_oof_r2"]),
        },
        "height_secondary":{
            "n":EXPECTED_HEIGHT_N,
            "age_sex_oof_r2_same_subset":float(hb["aggregate_oof_r2"]),
            "age_sex_height_oof_r2":float(hh["aggregate_oof_r2"]),
            "height_delta_r2_beyond_age_sex":float(hh["aggregate_oof_r2"]-hb["aggregate_oof_r2"]),
            "conventional_age_sex_height_oof_r2":float(hcomb["aggregate_oof_r2"]),
            "constitutional_delta_r2_beyond_conventional":float(hcomb["aggregate_oof_r2"]-hc["aggregate_oof_r2"]),
        },
        "boundary":[
            "Discovery/derivation evidence only.",
            "No causal interpretation.",
            "Height secondary complete-case only; no imputation.",
            "Chronic phenotype mapping deferred to Stage7C.",
            "Raw frozen B8 remains primary.",
            "No Ztrait/Zstate labels authorized."
        ],
        "hashes":{
            "frozen_spec_sha256":sha256_file(spec_path),
            "analysis_script_sha256":sha256_file(script_path),
        },
        "environment":{
            "python":sys.version,
            "platform":platform.platform(),
            "numpy":np.__version__,
            "pandas":pd.__version__,
        }
    }
    (out/"WFP_STAGE7B_CONSTITUTIONAL_RESULTS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "WF-P STAGE 7B — CONSTITUTIONAL Q4/Q5",
        "====================================",
        "Decision: WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE",
        f"Full cohort n: {EXPECTED_N}",
        f"Height complete-case n: {EXPECTED_HEIGHT_N}",
        "Frozen B8 changed: NO",
        "",
        "Q4 primary — age + sex -> frozen B8:",
        f"  aggregate OOF R2: {float(age_row['aggregate_oof_r2']):.6f}",
    ]
    for j in range(1,D+1):
        lines.append(f"  z{j} OOF R2: {float(age_row[f'z{j}_oof_r2']):.6f}")
    lines += [
        "",
        "Q5 — residual population geometry after age + sex:",
        f"  residual trace fraction: {float(age_row['residual_trace_fraction_vs_raw']):.6f}",
        f"  residual effective rank: {float(age_row['residual_effective_rank']):.6f}",
        f"  residual d90/d95: {int(age_row['residual_d90'])}/{int(age_row['residual_d95'])}",
        "",
        "Nonlinear age sensitivity:",
        f"  age + age^2 + sex OOF R2: {float(age2_row['aggregate_oof_r2']):.6f}",
        f"  delta R2 vs primary: {float(age2_row['aggregate_oof_r2']-age_row['aggregate_oof_r2']):.6f}",
        "",
        "Incremental constitutional information beyond WF-P0 conventional factors:",
        f"  conventional-only OOF R2: {float(conv_row['aggregate_oof_r2']):.6f}",
        f"  conventional + age + sex OOF R2: {float(comb_row['aggregate_oof_r2']):.6f}",
        f"  incremental age+sex delta R2: {float(comb_row['aggregate_oof_r2']-conv_row['aggregate_oof_r2']):.6f}",
        "",
        "Height secondary complete-case sensitivity:",
        f"  age + sex OOF R2 in same subset: {float(hb['aggregate_oof_r2']):.6f}",
        f"  age + sex + height OOF R2: {float(hh['aggregate_oof_r2']):.6f}",
        f"  height delta R2 beyond age+sex: {float(hh['aggregate_oof_r2']-hb['aggregate_oof_r2']):.6f}",
        f"  conventional + age + sex + height OOF R2: {float(hcomb['aggregate_oof_r2']):.6f}",
        "",
        "Boundary:",
        "  Do not call any axis Ztrait or Zstate.",
        "  Chronic phenotype mapping requires separate Stage7C freeze.",
        "  Raw frozen B8 remains primary."
    ]
    (out/"WFP_STAGE7B_CONSTITUTIONAL_READOUT.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
