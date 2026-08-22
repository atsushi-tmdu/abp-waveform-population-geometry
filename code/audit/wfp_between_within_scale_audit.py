#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_FS=125.0
FULL_WINDOW_SEC=1800
BLOCK_SEC=60
MIN_BEATS_PER_BLOCK=32
MIN_TOTAL_BLOCKS=6
MIN_ODD_BLOCKS=3
MIN_EVEN_BLOCKS=3
EXPECTED_N=978
P=64
D=8
REPLAY_TOL=5e-10
EXPECTED_RUN1_SHA256="811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError("module load failed")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def normalize_shape(v):
    v=np.asarray(v,float); c=v-np.mean(v); sd=float(np.std(c))
    if v.shape!=(P,) or not np.all(np.isfinite(v)) or sd<=1e-12: raise RuntimeError("normalize failed")
    return c/sd

def qsummary(x):
    x=np.asarray(x,float)
    return {k:float(v) for k,v in zip(
        ["min","q05","q25","median","q75","q95","max","mean"],
        [np.min(x),np.percentile(x,5),np.percentile(x,25),np.percentile(x,50),
         np.percentile(x,75),np.percentile(x,95),np.max(x),np.mean(x)]
    )}

def pairwise_distances(Z):
    Z=np.asarray(Z,float); g=np.sum(Z*Z,axis=1)
    d2=np.maximum(g[:,None]+g[None,:]-2*(Z@Z.T),0.0)
    iu=np.triu_indices(len(Z),k=1)
    return np.sqrt(d2[iu])

def nearest_neighbor_distances(Z):
    Z=np.asarray(Z,float); g=np.sum(Z*Z,axis=1)
    d2=np.maximum(g[:,None]+g[None,:]-2*(Z@Z.T),0.0)
    np.fill_diagonal(d2,np.inf)
    return np.sqrt(np.min(d2,axis=1))

def construct_patient(path,run1,mu,B8,stored):
    pid,fs0,x125=run1.load_case(path); pid=str(pid)
    if pid!=str(stored["patient_id"]): raise RuntimeError(f"{pid}: pid mismatch")
    if abs(float(fs0)-SOURCE_FS)>1e-6: raise RuntimeError(f"{pid}: fs mismatch")
    total_sec=len(x125)/SOURCE_FS
    peaks=run1.detect_systolic_peaks_125(x125)
    built=run1.build_125_catalog(x125,peaks)
    if built is None: raise RuntimeError(f"{pid}: catalog failed")
    catalog,rep125=built
    start=total_sec-FULL_WINDOW_SEC
    qc=run1.locked_qc_for_window(catalog,rep125,start,total_sec)
    if qc is None: raise RuntimeError(f"{pid}: QC failed")
    acc=np.asarray(qc["accepted_idx"],int)
    shape=np.asarray(rep125["shape_norm"][acc],float)
    centers=catalog["center_sec"].to_numpy(float)[acc]
    q_list=[]; block_ids=[]; kappas=[]
    for b in range(FULL_WINDOW_SEC//BLOCK_SEC):
        a=start+b*BLOCK_SEC; z=a+BLOCK_SEC
        sel=(centers>=a)&((centers<z) if b<(FULL_WINDOW_SEC//BLOCK_SEC)-1 else (centers<=z))
        Xb=shape[sel]
        if len(Xb)<MIN_BEATS_PER_BLOCK: continue
        mean_shape=np.mean(Xb,axis=0); k=float(np.std(mean_shape-np.mean(mean_shape)))
        q=normalize_shape(mean_shape)
        if not np.isfinite(k): continue
        q_list.append(q); block_ids.append(b); kappas.append(k)
    odd=[q for q,b in zip(q_list,block_ids) if b%2==1]
    even=[q for q,b in zip(q_list,block_ids) if b%2==0]
    if len(q_list)<MIN_TOTAL_BLOCKS or len(odd)<MIN_ODD_BLOCKS or len(even)<MIN_EVEN_BLOCKS:
        raise RuntimeError(f"{pid}: block eligibility failed")
    all_rep=normalize_shape(np.mean(np.vstack(q_list),axis=0))
    odd_rep=normalize_shape(np.mean(np.vstack(odd),axis=0))
    even_rep=normalize_shape(np.mean(np.vstack(even),axis=0))
    z_all=(all_rep-mu)@B8; z_odd=(odd_rep-mu)@B8; z_even=(even_rep-mu)@B8
    z_blocks=(np.vstack(q_list)-mu[None,:])@B8
    stored_z=np.asarray([stored[f"z{j}"] for j in range(1,D+1)],float)
    replay_err=float(np.max(np.abs(z_all-stored_z)))
    if int(stored["eligible_blocks"])!=len(q_list): raise RuntimeError(f"{pid}: block replay mismatch")
    kappa_err=abs(float(np.median(kappas))-float(stored["median_block_coherence"]))
    zb_mean=np.mean(z_blocks,axis=0)
    centered=z_blocks-zb_mean[None,:]
    within_mse=float(np.mean(np.sum(centered**2,axis=1)))
    within_axis_var=np.mean(centered**2,axis=0)
    block_disp=np.linalg.norm(centered,axis=1)
    block_to_central=np.linalg.norm(z_blocks-z_all[None,:],axis=1)
    steps=[float(np.linalg.norm(z_blocks[k]-z_blocks[k-1])) for k in range(1,len(block_ids)) if block_ids[k]==block_ids[k-1]+1]
    return dict(patient_id=pid,z_all=z_all,z_odd=z_odd,z_even=z_even,z_blocks=z_blocks,
                within_mse=within_mse,within_axis_var=within_axis_var,block_disp=block_disp,
                block_to_central=block_to_central,adjacent_steps=steps,
                replicate_distance=float(np.linalg.norm(z_odd-z_even)),
                replay_score_error=replay_err,replay_kappa_error=float(kappa_err))

def self_test():
    rng=np.random.default_rng(20260822); Z=rng.normal(size=(100,D))
    p=pairwise_distances(Z); nn=nearest_neighbor_distances(Z)
    if len(p)!=4950 or len(nn)!=100: raise RuntimeError("distance self-test failed")
    print("WF-P Between–Within Scale Audit self-test: PASS"); return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="~/Documents/abp_information_study/data/abp125_validation1000")
    ap.add_argument("--run1-script",default="~/Documents/abp_information_study/code/wp2_run1_development50.py")
    ap.add_argument("--discovery-results",default="~/Documents/abp_information_study/results/wfp_discovery_validation1000")
    ap.add_argument("--spec",default="~/Documents/abp_information_study/freeze/wfp_between_within_scale/WFP_BETWEEN_WITHIN_SCALE_FROZEN_SPEC.json")
    ap.add_argument("--out",default="~/Documents/abp_information_study/results/wfp_between_within_scale")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test: return self_test()

    script=Path(__file__).resolve()
    inp=Path(a.input).expanduser().resolve(); run1p=Path(a.run1_script).expanduser().resolve()
    disc=Path(a.discovery_results).expanduser().resolve(); specp=Path(a.spec).expanduser().resolve()
    out=Path(a.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    scorep=disc/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"; coordp=disc/"WFP_DISCOVERY_COMMON_COORDINATES.npz"; resultp=disc/"WFP_DISCOVERY_RESULTS.json"
    spec=json.loads(specp.read_text())
    if spec.get("status")!="FROZEN_BEFORE_BETWEEN_WITHIN_SCALE_AUDIT": raise SystemExit("FAIL spec status")
    if spec.get("analysis_script_sha256")!=sha256_file(script): raise SystemExit("FAIL script hash")
    if spec.get("authoritative_run1_sha256")!=sha256_file(run1p): raise SystemExit("FAIL run1 hash")
    for key,p in {"score_file_sha256":scorep,"coordinate_file_sha256":coordp,"discovery_result_sha256":resultp}.items():
        if spec.get(key)!=sha256_file(p): raise SystemExit(f"FAIL input hash {key}")
    if sha256_file(run1p)!=EXPECTED_RUN1_SHA256: raise SystemExit("FAIL authoritative Run1 hash unexpected")

    scores=pd.read_csv(scorep,dtype={"patient_id":str})
    if len(scores)!=EXPECTED_N or scores.patient_id.duplicated().any(): raise SystemExit("FAIL score cohort")
    with np.load(coordp,allow_pickle=False) as z:
        mu=np.asarray(z["population_mean"],float); B8=np.asarray(z["between_basis"],float); dim=int(np.asarray(z["dimension"]).reshape(()))
    if mu.shape!=(P,) or B8.shape!=(P,D) or dim!=D: raise SystemExit("FAIL coordinate dims")
    run1=load_module(run1p,"wfp_scale_run1")
    case_map={p.name.split("__")[0]:p for p in sorted((inp/"cases").glob("*.npz"))}
    patients=[]
    for i,pid in enumerate(scores.patient_id.astype(str),1):
        sr=scores.loc[scores.patient_id.astype(str)==pid].iloc[0]
        patients.append(construct_patient(case_map[pid],run1,mu,B8,sr))
        if i%50==0 or i==EXPECTED_N: print(f"[progress] {i}/{EXPECTED_N}",flush=True)

    max_score=max(p["replay_score_error"] for p in patients); max_k=max(p["replay_kappa_error"] for p in patients)
    ok=max_score<=REPLAY_TOL and max_k<=REPLAY_TOL
    itxt="\n".join([
        "WF-P BETWEEN–WITHIN SCALE AUDIT INTEGRITY","==========================================",
        f"Decision: {'PASS' if ok else 'FAIL_STOP_BEFORE_SCIENTIFIC_READOUT'}",
        f"Patients replayed: {len(patients)}",
        f"Frozen B8 score replay max abs error: {max_score:.3e}",
        f"Block coherence replay max abs error: {max_k:.3e}",
        "Frozen B8 changed: NO","Clinical labels accessed: NO",""
    ])
    (out/"WFP_BETWEEN_WITHIN_SCALE_INTEGRITY.txt").write_text(itxt)
    if not ok: print(itxt); raise SystemExit(2)

    Z=np.vstack([p["z_all"] for p in patients]); Zc=Z-np.mean(Z,axis=0,keepdims=True)
    radius=np.linalg.norm(Zc,axis=1); pair=pairwise_distances(Z); nn=nearest_neighbor_distances(Z)
    between_pairwise_rms=float(np.sqrt(np.mean(pair**2))); between_radius_rms=float(np.sqrt(np.mean(radius**2)))
    between_axis_var=np.var(Z,axis=0,ddof=1)
    rep=np.asarray([p["replicate_distance"] for p in patients]); replicate_rms=float(np.sqrt(np.mean(rep**2)))
    within_mse=np.asarray([p["within_mse"] for p in patients]); within_rms=float(np.sqrt(np.mean(within_mse)))
    patient_within_rms=np.sqrt(within_mse)
    within_axis_var=np.mean(np.vstack([p["within_axis_var"] for p in patients]),axis=0)
    block_disp=np.concatenate([p["block_disp"] for p in patients]); block_to_central=np.concatenate([p["block_to_central"] for p in patients])
    steps=np.asarray([x for p in patients for x in p["adjacent_steps"]],float)
    med_disp=np.asarray([np.median(p["block_disp"]) for p in patients]); p95_disp=np.asarray([np.percentile(p["block_disp"],95) for p in patients]); max_disp=np.asarray([np.max(p["block_disp"]) for p in patients])
    rmed=med_disp/nn; rp95=p95_disp/nn; rmax=max_disp/nn
    frac95=float(np.mean(p95_disp>=nn)); fracmax=float(np.mean(max_disp>=nn))

    axis_rows=[]
    for j in range(D):
        od=np.asarray([p["z_odd"][j]-p["z_even"][j] for p in patients])
        repvar=float(np.mean(od**2)/2.0)
        bvar=float(between_axis_var[j]); wvar=float(within_axis_var[j])
        axis_rows.append({"axis":j+1,"between_variance":bvar,"within_equal_patient_variance":wvar,
                          "within_to_between_variance_ratio":wvar/bvar,
                          "odd_even_discrepancy_variance_proxy":repvar,
                          "odd_even_proxy_to_between_ratio":repvar/bvar})
    adf=pd.DataFrame(axis_rows); adf.to_csv(out/"wfp_between_within_axis_scales.csv",index=False)

    pd.DataFrame([{
        "patient_id":p["patient_id"],"population_radius":float(radius[i]),"nearest_neighbor_distance":float(nn[i]),
        "odd_even_replicate_distance":float(rep[i]),"within_rms":float(patient_within_rms[i]),
        "median_block_displacement":float(med_disp[i]),"p95_block_displacement":float(p95_disp[i]),"max_block_displacement":float(max_disp[i])
    } for i,p in enumerate(patients)]).to_csv(out/"wfp_between_within_patient_scales_PRIVATE.csv",index=False)

    ratios={
        "between_pairwise_rms_over_replicate_rms":between_pairwise_rms/replicate_rms,
        "within_rms_over_between_pairwise_rms":within_rms/between_pairwise_rms,
        "within_rms_over_replicate_rms":within_rms/replicate_rms,
        "replicate_rms_over_between_pairwise_rms":replicate_rms/between_pairwise_rms,
    }
    result={
        "decision":"WFP_BETWEEN_WITHIN_SCALE_AUDIT_COMPLETE","patients_n":EXPECTED_N,"frozen_B8_changed":False,
        "clinical_labels_accessed":False,
        "between":{"pairwise_distance":qsummary(pair),"pairwise_rms":between_pairwise_rms,"population_radius":qsummary(radius),
                   "population_radius_rms":between_radius_rms,"nearest_neighbor_distance":qsummary(nn)},
        "within":{"equal_patient_within_rms":within_rms,"patient_within_rms":qsummary(patient_within_rms),
                  "block_displacement":qsummary(block_disp),"block_to_central":qsummary(block_to_central),
                  "adjacent_block_step":qsummary(steps) if len(steps) else None},
        "replicate":{"odd_even_distance":qsummary(rep),"odd_even_rms":replicate_rms},
        "scale_ratios":{k:float(v) for k,v in ratios.items()},
        "nearest_neighbor_crossing":{"median_block_displacement_over_nn":qsummary(rmed),"p95_block_displacement_over_nn":qsummary(rp95),
                                     "max_block_displacement_over_nn":qsummary(rmax),
                                     "fraction_patients_p95_block_displacement_ge_nn":frac95,
                                     "fraction_patients_max_block_displacement_ge_nn":fracmax},
        "boundary":["Scale audit only","Odd/even discrepancy is not pure noise","60-s movement is not WF3 long-duration trajectory","No Ztrait/Zstate labels"],
        "hashes":{"frozen_spec_sha256":sha256_file(specp),"analysis_script_sha256":sha256_file(script),"run1_sha256":sha256_file(run1p)}
    }
    (out/"WFP_BETWEEN_WITHIN_SCALE_RESULTS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "WF-P BETWEEN–WITHIN SCALE AUDIT","===============================",
        "Decision: WFP_BETWEEN_WITHIN_SCALE_AUDIT_COMPLETE",f"Patients: {EXPECTED_N}","Frozen B8 changed: NO","Clinical labels accessed: NO","",
        "Between-patient geometry:",f"  pairwise RMS distance: {between_pairwise_rms:.6f}",f"  pairwise distance median: {np.median(pair):.6f}",
        f"  population-center radius RMS: {between_radius_rms:.6f}",f"  nearest-neighbor distance median: {np.median(nn):.6f}",
        f"  nearest-neighbor distance q05/q95: {np.percentile(nn,5):.6f} / {np.percentile(nn,95):.6f}","",
        "Within-patient 60-s movement:",f"  equal-patient within RMS: {within_rms:.6f}",f"  patient within-RMS median: {np.median(patient_within_rms):.6f}",
        f"  block displacement median: {np.median(block_disp):.6f}",f"  block displacement q95: {np.percentile(block_disp,95):.6f}",
        f"  adjacent 60-s step median: {np.median(steps):.6f}" if len(steps) else "  adjacent 60-s step median: NA","",
        "Odd/even replicate discrepancy:",f"  RMS distance: {replicate_rms:.6f}",f"  median distance: {np.median(rep):.6f}","",
        "Scale ratios:",f"  between pairwise RMS / replicate RMS: {ratios['between_pairwise_rms_over_replicate_rms']:.6f}",
        f"  within RMS / between pairwise RMS: {ratios['within_rms_over_between_pairwise_rms']:.6f}",
        f"  within RMS / replicate RMS: {ratios['within_rms_over_replicate_rms']:.6f}","",
        "Movement relative to nearest other patient:",f"  median(patient median-block-displacement / NN): {np.median(rmed):.6f}",
        f"  median(patient p95-block-displacement / NN): {np.median(rp95):.6f}",
        f"  patients with p95 block displacement >= NN: {frac95:.6f}",f"  patients with max block displacement >= NN: {fracmax:.6f}","",
        "Per-axis within/between variance ratios:"
    ]
    for _,r in adf.iterrows():
        lines.append(f"  z{int(r.axis)}: within/between={r.within_to_between_variance_ratio:.6f}; odd-even-proxy/between={r.odd_even_proxy_to_between_ratio:.6f}")
    lines += ["","Boundary:","  Odd/even discrepancy is NOT pure noise.","  60-s movement is NOT WF3 long-duration trajectory.","  No Ztrait/Zstate labels are authorized."]
    (out/"WFP_BETWEEN_WITHIN_SCALE_READOUT.txt").write_text("\n".join(lines)+"\n")
    print(itxt); print("\n".join(lines)); return 0

if __name__=="__main__": raise SystemExit(main())
