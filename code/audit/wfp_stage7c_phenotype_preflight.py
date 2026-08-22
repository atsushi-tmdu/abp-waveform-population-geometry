#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_N=978
EXPECTED_EXACT_ADMISSION_N=887
PHENOTYPES=[
"congestive_heart_failure","cardiac_arrhythmias","valvular_disease",
"peripheral_vascular_disease","hypertension","diabetes","renal_failure",
"chronic_pulmonary_disease"]
MAIN_MIN_EXPOSED=50
MAIN_MIN_UNEXPOSED=200
EXPLORATORY_MIN_EXPOSED=30

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def subject_from_patient_id(pid):
    digits=re.sub(r"\D","",str(pid))
    if not digits: raise ValueError(f"Cannot derive SUBJECT_ID from {pid!r}")
    return int(digits)

def detect_waveform_start_column(cols):
    preferred=["waveform_window_start","waveform_start","waveform_start_datetime","window_start","window_start_datetime"]
    lower={str(c).lower():str(c) for c in cols}
    for c in preferred:
        if c in lower: return lower[c]
    for c in cols:
        lc=str(c).lower()
        if ("waveform" in lc and "start" in lc) or ("window" in lc and "start" in lc):
            return str(c)
    raise RuntimeError(f"Could not identify waveform-start column. Columns={cols}")

def clean_icd9(x):
    if pd.isna(x): return ""
    return str(x).strip().upper().replace(".","")

def phenotype_flags(code):
    c=clean_icd9(code); p3=c[:3]; p4=c[:4]
    return {
      "congestive_heart_failure": int(
        c in {"39891","40201","40211","40291","40401","40403","40411","40413","40491","40493"}
        or p4 in {"4254","4255","4257","4258","4259"} or p3=="428"),
      "cardiac_arrhythmias": int(
        c in {"42613","42610","42612","99601","99604"}
        or p4 in {"4260","4267","4269","4270","4271","4272","4273","4274","4276","4278","4279","7850","V450","V533"}),
      "valvular_disease": int(
        p4 in {"0932","7463","7464","7465","7466","V422","V433"}
        or p3 in {"394","395","396","397","424"}),
      "peripheral_vascular_disease": int(
        p4 in {"0930","4373","4431","4432","4438","4439","4471","5571","5579","V434"}
        or p3 in {"440","441"}),
      "hypertension": int(p3 in {"401","402","403","404","405"}),
      "diabetes": int(p4 in {"2500","2501","2502","2503","2504","2505","2506","2507","2508","2509"}),
      "renal_failure": int(
        c in {"40301","40311","40391","40402","40403","40412","40413","40492","40493"}
        or p4 in {"5880","V420","V451"} or p3 in {"585","586","V56"}),
      "chronic_pulmonary_disease": int(
        p4 in {"4168","4169","5064","5081","5088"}
        or p3 in {"490","491","492","493","494","495","496","500","501","502","503","504","505"}),
    }

def role_from_counts(exposed,unexposed):
    if exposed>=MAIN_MIN_EXPOSED and unexposed>=MAIN_MIN_UNEXPOSED: return "MAIN_STAGE7C_CANDIDATE"
    if exposed>=EXPLORATORY_MIN_EXPOSED and unexposed>=MAIN_MIN_UNEXPOSED: return "EXPLORATORY_SUPPLEMENT_ONLY"
    return "DESCRIPTIVE_PREVALENCE_ONLY"

def self_test():
    tests={"4280":"congestive_heart_failure","42731":"cardiac_arrhythmias","4241":"valvular_disease",
           "44020":"peripheral_vascular_disease","4019":"hypertension","25000":"diabetes",
           "5856":"renal_failure","496":"chronic_pulmonary_disease"}
    for code,exp in tests.items():
        if phenotype_flags(code)[exp]!=1: raise RuntimeError(f"mapping failed {code}->{exp}")
    if any(phenotype_flags("7806").values()): raise RuntimeError("neutral mapping failed")
    print("WF-P Stage7C phenotype preflight self-test: PASS")
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--score-file",default="~/Documents/abp_information_study/results/wfp_discovery_validation1000/wfp_patient_scores_DISCOVERY_PRIVATE.csv")
    ap.add_argument("--temporal-linkage",default="~/Documents/abp_information_study/results/wfp_temporal_linkage_audit/WFP_TEMPORAL_LINKAGE_AUDIT_PRIVATE.csv")
    ap.add_argument("--admissions")
    ap.add_argument("--diagnoses")
    ap.add_argument("--out",default="~/Documents/abp_information_study/results/wfp_stage7c_phenotype_preflight")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test: return self_test()
    if not a.admissions or not a.diagnoses: raise SystemExit("--admissions and --diagnoses are required")

    scorep=Path(a.score_file).expanduser().resolve()
    tempp=Path(a.temporal_linkage).expanduser().resolve()
    admp=Path(a.admissions).expanduser().resolve()
    diagp=Path(a.diagnoses).expanduser().resolve()
    out=Path(a.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    scores=pd.read_csv(scorep,usecols=["patient_id"],dtype={"patient_id":str})
    if len(scores)!=EXPECTED_N or scores.patient_id.duplicated().any(): raise SystemExit("FAIL frozen cohort")
    scores["subject_id"]=scores.patient_id.map(subject_from_patient_id)

    head=pd.read_csv(tempp,nrows=0)
    start_col=detect_waveform_start_column(list(head.columns))
    if "patient_id" not in head.columns: raise SystemExit(f"FAIL no patient_id in temporal file; columns={list(head.columns)}")
    temporal=pd.read_csv(tempp,usecols=["patient_id",start_col],dtype={"patient_id":str})
    temporal["waveform_start"]=pd.to_datetime(temporal[start_col],errors="coerce")
    temporal=temporal[["patient_id","waveform_start"]]
    if temporal.patient_id.duplicated().any(): raise SystemExit("FAIL duplicate temporal patient_id")
    cohort=scores.merge(temporal,on="patient_id",how="left",validate="one_to_one")
    if cohort.waveform_start.isna().any(): raise SystemExit(f"FAIL missing waveform_start n={cohort.waveform_start.isna().sum()}")

    subjects=set(cohort.subject_id.astype(int))
    adm=pd.read_csv(admp,usecols=["SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME"],low_memory=False)
    adm=adm[adm.SUBJECT_ID.isin(subjects)].copy()
    adm["ADMITTIME"]=pd.to_datetime(adm.ADMITTIME,errors="coerce")
    adm["DISCHTIME"]=pd.to_datetime(adm.DISCHTIME,errors="coerce")
    adm=adm.dropna(subset=["SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME"])
    adm["SUBJECT_ID"]=adm.SUBJECT_ID.astype(int); adm["HADM_ID"]=adm.HADM_ID.astype(int)
    groups={int(s):g.copy() for s,g in adm.groupby("SUBJECT_ID")}

    exact={}; prior={}; ambiguous=[]
    for _,r in cohort.iterrows():
        sid=int(r.subject_id); t=pd.Timestamp(r.waveform_start); g=groups.get(sid)
        if g is None:
            exact[sid]=None; prior[sid]=[]; continue
        ex=g[(g.ADMITTIME<=t)&(t<=g.DISCHTIME)]
        if len(ex)>1: ambiguous.append((sid,t.isoformat(),ex.HADM_ID.astype(int).tolist()))
        exact[sid]=int(ex.iloc[0].HADM_ID) if len(ex)==1 else None
        prior[sid]=g[g.DISCHTIME<t].HADM_ID.astype(int).tolist()
    if ambiguous: raise SystemExit(f"FAIL ambiguous exact admissions {ambiguous[:10]}")
    exact_n=sum(v is not None for v in exact.values())
    if exact_n!=EXPECTED_EXACT_ADMISSION_N:
        raise SystemExit(f"FAIL exact admission replay expected {EXPECTED_EXACT_ADMISSION_N}, found {exact_n}")

    diag=pd.read_csv(diagp,usecols=["SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"],dtype={"ICD9_CODE":str},low_memory=False)
    diag=diag[diag.SUBJECT_ID.isin(subjects)].copy()
    diag["SEQ_NUM"]=pd.to_numeric(diag.SEQ_NUM,errors="coerce")
    diag=diag[diag.SEQ_NUM.notna()&(diag.SEQ_NUM!=1)].copy()
    diag["HADM_ID"]=pd.to_numeric(diag.HADM_ID,errors="coerce")
    diag=diag.dropna(subset=["HADM_ID"]); diag["HADM_ID"]=diag.HADM_ID.astype(int)
    diag["ICD9_CODE"]=diag.ICD9_CODE.map(clean_icd9)

    hadm_flags={}
    for hadm,g in diag.groupby("HADM_ID"):
        agg={p:0 for p in PHENOTYPES}
        for code in g.ICD9_CODE:
            fl=phenotype_flags(code)
            for p in PHENOTYPES: agg[p]=max(agg[p],fl[p])
        hadm_flags[int(hadm)]=agg

    rows=[]
    for _,r in cohort.iterrows():
        pid=str(r.patient_id); sid=int(r.subject_id); curr=exact[sid]; ph=prior[sid]
        pf={p:0 for p in PHENOTYPES}
        for h in ph:
            hf=hadm_flags.get(int(h),{})
            for p in PHENOTYPES: pf[p]=max(pf[p],int(hf.get(p,0)))
        cf=None if curr is None else {p:int(hadm_flags.get(int(curr),{}).get(p,0)) for p in PHENOTYPES}
        row={"patient_id":pid,"subject_id":sid,"waveform_start":pd.Timestamp(r.waveform_start).isoformat(),
             "exact_current_hadm_id":curr if curr is not None else np.nan,"n_prior_completed_admissions":len(ph)}
        for p in PHENOTYPES:
            row[f"{p}__prior_completed_history"]=pf[p]
            if cf is None:
                row[f"{p}__current_exact_admission"]=np.nan
                row[f"{p}__combined_current_or_prior"]=1.0 if pf[p]==1 else np.nan
            else:
                row[f"{p}__current_exact_admission"]=cf[p]
                row[f"{p}__combined_current_or_prior"]=max(pf[p],cf[p])
        rows.append(row)
    pdf=pd.DataFrame(rows)
    pdf.to_csv(out/"WFP_STAGE7C_PHENOTYPE_PREFLIGHT_PRIVATE.csv",index=False)

    summaries=[]
    for p in PHENOTYPES:
        c=pdf[f"{p}__current_exact_admission"]; known=c.notna()
        exposed=int((c[known]==1).sum()); unexposed=int((c[known]==0).sum())
        priorv=pdf[f"{p}__prior_completed_history"]; comb=pdf[f"{p}__combined_current_or_prior"]
        summaries.append({"phenotype":p,"current_exact_known_n":int(known.sum()),"current_exact_exposed_n":exposed,
                          "current_exact_unexposed_n":unexposed,"current_exact_prevalence":exposed/int(known.sum()),
                          "prior_history_exposed_n_full978":int((priorv==1).sum()),
                          "prior_history_prevalence_full978":float(np.mean(priorv==1)),
                          "combined_known_n":int(comb.notna().sum()),"combined_exposed_n":int((comb==1).sum()),
                          "stage7c_role_from_frozen_yield_rule":role_from_counts(exposed,unexposed)})
    sdf=pd.DataFrame(summaries); sdf.to_csv(out/"wfp_stage7c_phenotype_prevalence_summary.csv",index=False)
    mainc=sdf.loc[sdf.stage7c_role_from_frozen_yield_rule=="MAIN_STAGE7C_CANDIDATE","phenotype"].tolist()
    expl=sdf.loc[sdf.stage7c_role_from_frozen_yield_rule=="EXPLORATORY_SUPPLEMENT_ONLY","phenotype"].tolist()
    desc=sdf.loc[sdf.stage7c_role_from_frozen_yield_rule=="DESCRIPTIVE_PREVALENCE_ONLY","phenotype"].tolist()

    result={"decision":"WFP_STAGE7C_PHENOTYPE_PREFLIGHT_COMPLETE","scientific_role":"EFFECT_BLIND_PHENOTYPE_AVAILABILITY_PREVALENCE_ONLY",
            "frozen_cohort_n":EXPECTED_N,"morphology_score_values_read":False,"morphology_phenotype_associations_calculated":False,
            "frozen_B8_changed":False,"waveform_start_column_detected":start_col,"exact_current_admission_n":int(exact_n),
            "phenotypes_prespecified":PHENOTYPES,"main_stage7c_candidates":mainc,
            "exploratory_supplement_only":expl,"descriptive_prevalence_only":desc,
            "boundary":["Current exact-admission phenotype is discharge-coded cross-sectional phenotype, not known-at-waveform-time information.",
                        "Prior completed history uses only admissions with DISCHTIME strictly before waveform start.",
                        "No z1..z8 values are read.","No phenotype-morphology association is authorized until a separate freeze."],
            "hashes":{"script_sha256":sha256_file(Path(__file__).resolve()),"score_file_sha256":sha256_file(scorep),
                      "temporal_linkage_sha256":sha256_file(tempp),"admissions_sha256":sha256_file(admp),"diagnoses_sha256":sha256_file(diagp)}}
    (out/"WFP_STAGE7C_PHENOTYPE_PREFLIGHT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["WF-P STAGE 7C PHENOTYPE PREFLIGHT","================================",
           "Decision: WFP_STAGE7C_PHENOTYPE_PREFLIGHT_COMPLETE",
           "Scientific role: EFFECT-BLIND PHENOTYPE AVAILABILITY / PREVALENCE ONLY",
           f"Frozen cohort n: {EXPECTED_N}","Morphology score values read: NO","Morphology-phenotype associations calculated: NO",
           "Frozen B8 changed: NO","",f"Exact admission containing waveform start: {exact_n}/{EXPECTED_N}","",
           "Prespecified phenotype prevalence and frozen role:"]
    for _,r in sdf.iterrows():
        lines.append(f"  {r.phenotype}: current exact {int(r.current_exact_exposed_n)}/{int(r.current_exact_known_n)} "
                     f"({float(r.current_exact_prevalence):.4f}); prior-history positive "
                     f"{int(r.prior_history_exposed_n_full978)}/{EXPECTED_N}; role={r.stage7c_role_from_frozen_yield_rule}")
    lines+=["","Main Stage7C candidates from frozen yield rule:","  "+(", ".join(mainc) if mainc else "NONE"),
            "","Exploratory supplement only:","  "+(", ".join(expl) if expl else "NONE"),
            "","Descriptive prevalence only:","  "+(", ".join(desc) if desc else "NONE"),
            "","Boundary:","  Do NOT merge phenotype columns with z1..z8 until Stage7C association analysis is separately frozen."]
    (out/"WFP_STAGE7C_PHENOTYPE_PREFLIGHT.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines)); return 0

if __name__=="__main__": raise SystemExit(main())
