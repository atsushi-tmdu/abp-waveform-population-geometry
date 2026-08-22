#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7A.1 — effect-blind temporal linkage audit (engineering fix v1.1).

Purpose
-------
Resolve why some frozen WF-P waveform windows do not fall exactly inside
MIMIC-III ICUSTAYS / ADMISSIONS intervals, WITHOUT reading morphology scores.

Key correction from Stage 7A:
- age is calculated directly at the waveform-window timestamp from PATIENTS.DOB
- age does NOT require an exact ADMISSION link
- ages >89 are top-coded to 90 for MIMIC-III deidentification

This script reads only patient_id from the frozen morphology score CSV.
It does not read z1..z8 or calculate any morphology-covariate association.
"""

from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

EXPECTED_N = 978
GAP_THRESHOLDS_H = [1, 6, 12, 24, 48, 72, 168]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def find_table(root: Path, stem: str) -> Path:
    for name in (
        f"{stem}.csv", f"{stem}.CSV", f"{stem}.csv.gz", f"{stem}.CSV.gz",
        f"{stem.lower()}.csv", f"{stem.lower()}.csv.gz"
    ):
        p = root / name
        if p.is_file():
            return p
    raise FileNotFoundError(stem)

def read_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d.columns = [str(c).lower().strip() for c in d.columns]
    return d

def subject_id_from_pid(pid: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(pid).strip())
    return int(m.group(1)) if m else None

def record_start(record_name: str, record_path: str):
    s = str(record_name).strip()
    if not s:
        s = Path(str(record_path)).name
    m = re.search(
        r"p?\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})(?:-(\d{2}))?",
        s
    )
    if not m:
        return pd.NaT
    vals = list(map(int, m.groups()[:5]))
    sec = int(m.group(6)) if m.group(6) else 0
    try:
        return pd.Timestamp(
            year=vals[0], month=vals[1], day=vals[2],
            hour=vals[3], minute=vals[4], second=sec
        )
    except Exception:
        return pd.NaT

def interval_distance_hours(t, start, end) -> Tuple[float, str]:
    if pd.isna(t) or pd.isna(start) or pd.isna(end):
        return np.nan, "missing"
    if start <= t <= end:
        return 0.0, "inside"
    if t < start:
        return (start - t).total_seconds()/3600.0, "before_start"
    return (t - end).total_seconds()/3600.0, "after_end"

def nearest_interval(group: pd.DataFrame, t, start_col, end_col):
    if group is None or len(group) == 0 or pd.isna(t):
        return None, np.nan, "missing", 0
    vals = []
    for idx, r in group.iterrows():
        gap, direction = interval_distance_hours(t, r[start_col], r[end_col])
        if np.isfinite(gap):
            vals.append((gap, idx, direction))
    if not vals:
        return None, np.nan, "missing", 0
    vals.sort(key=lambda x: x[0])
    best_gap = vals[0][0]
    tied = [v for v in vals if abs(v[0]-best_gap) < 1e-12]
    best = group.loc[tied[0][1]]
    return best, float(best_gap), tied[0][2], len(tied)

def qstr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "NA"
    q = np.quantile(x, [0, .25, .5, .75, .9, .95, .99, 1])
    return (
        "min={:.3f}, q25={:.3f}, median={:.3f}, q75={:.3f}, "
        "q90={:.3f}, q95={:.3f}, q99={:.3f}, max={:.3f}"
    ).format(*q)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--score-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clinical = Path(a.clinical_root).expanduser().resolve()
    manifest_path = Path(a.manifest).expanduser().resolve()
    score_path = Path(a.score_file).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    patients_path = find_table(clinical, "PATIENTS")
    admissions_path = find_table(clinical, "ADMISSIONS")
    icustays_path = find_table(clinical, "ICUSTAYS")

    # patient_id only: morphology values remain sealed
    ids = pd.read_csv(score_path, usecols=["patient_id"], dtype={"patient_id": str})
    if len(ids) != EXPECTED_N or ids["patient_id"].duplicated().any():
        raise SystemExit("FAIL: frozen score patient-id frame is not the expected 978 unique patients")

    man = pd.read_csv(manifest_path, dtype={"patient_id": str}, low_memory=False)
    need = ["patient_id","record_path","record_name","fs","start_sample"]
    missing = [x for x in need if x not in man.columns]
    if missing:
        raise SystemExit(f"FAIL: manifest missing {missing}")

    c = ids.merge(man[need], on="patient_id", how="left", validate="one_to_one")
    if c["record_name"].isna().any():
        raise SystemExit("FAIL: frozen patient missing from manifest")

    c["subject_id"] = c["patient_id"].map(subject_id_from_pid)
    c["record_start"] = [
        record_start(rn, rp) for rn, rp in zip(c["record_name"], c["record_path"])
    ]
    fs = pd.to_numeric(c["fs"], errors="coerce")
    ss = pd.to_numeric(c["start_sample"], errors="coerce")
    c["window_start"] = c["record_start"] + pd.to_timedelta(ss/fs, unit="s")

    pat = read_table(patients_path)
    adm = read_table(admissions_path)
    icu = read_table(icustays_path)

    pat["subject_id"] = pd.to_numeric(pat["subject_id"], errors="coerce").astype("Int64")
    adm["subject_id"] = pd.to_numeric(adm["subject_id"], errors="coerce").astype("Int64")
    icu["subject_id"] = pd.to_numeric(icu["subject_id"], errors="coerce").astype("Int64")
    adm["hadm_id"] = pd.to_numeric(adm["hadm_id"], errors="coerce").astype("Int64")
    icu["hadm_id"] = pd.to_numeric(icu["hadm_id"], errors="coerce").astype("Int64")
    icu["icustay_id"] = pd.to_numeric(icu["icustay_id"], errors="coerce").astype("Int64")

    pat["dob"] = pd.to_datetime(pat["dob"], errors="coerce")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    pat_idx = pat.set_index("subject_id", drop=False)
    ag = {int(k): g.copy() for k,g in adm.dropna(subset=["subject_id"]).groupby("subject_id")}
    ig = {int(k): g.copy() for k,g in icu.dropna(subset=["subject_id"]).groupby("subject_id")}

    rows = []
    for _, r in c.iterrows():
        sid = int(r["subject_id"]) if pd.notna(r["subject_id"]) else None
        t = r["window_start"]

        pr = None
        if sid is not None and sid in pat_idx.index:
            pr = pat_idx.loc[sid]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]

        age_raw = np.nan
        age_top = np.nan
        age90 = pd.NA
        if pr is not None and pd.notna(pr.get("dob", pd.NaT)) and pd.notna(t):
            # Engineering fix v1.1:
            # MIMIC-III shifts DOB by ~300 years for patients aged >89.
            # pandas Timedelta in nanosecond resolution can overflow for such
            # intervals, so perform the subtraction using Python datetime.
            t_py = pd.Timestamp(t).to_pydatetime()
            dob_py = pd.Timestamp(pr["dob"]).to_pydatetime()
            age_raw = (t_py - dob_py).total_seconds()/(365.2425*86400)
            if age_raw >= 0:
                if age_raw > 89:
                    age_top = 90.0
                    age90 = True
                else:
                    age_top = float(age_raw)
                    age90 = False

        aa = ag.get(sid, pd.DataFrame()) if sid is not None else pd.DataFrame()
        ii = ig.get(sid, pd.DataFrame()) if sid is not None else pd.DataFrame()

        exact_a = aa[(aa["admittime"] <= t) & (t <= aa["dischtime"])] if len(aa) and pd.notna(t) else pd.DataFrame()
        exact_i = ii[(ii["intime"] <= t) & (t <= ii["outtime"])] if len(ii) and pd.notna(t) else pd.DataFrame()

        na, gap_a, dir_a, tie_a = nearest_interval(aa, t, "admittime", "dischtime")
        ni, gap_i, dir_i, tie_i = nearest_interval(ii, t, "intime", "outtime")

        rows.append({
            "patient_id": r["patient_id"],
            "subject_id": sid,
            "record_start": r["record_start"],
            "window_start": t,
            "age_raw_waveform": age_raw,
            "age_years_capped90": age_top,
            "age_90plus": age90,
            "gender": pr.get("gender", pd.NA) if pr is not None else pd.NA,
            "n_admissions_for_subject": len(aa),
            "n_icustays_for_subject": len(ii),
            "n_exact_admission": len(exact_a),
            "n_exact_icustay": len(exact_i),
            "nearest_admission_gap_h": gap_a,
            "nearest_admission_direction": dir_a,
            "nearest_admission_ties": tie_a,
            "nearest_admission_hadm_id": na.get("hadm_id", pd.NA) if na is not None else pd.NA,
            "nearest_icustay_gap_h": gap_i,
            "nearest_icustay_direction": dir_i,
            "nearest_icustay_ties": tie_i,
            "nearest_icustay_id": ni.get("icustay_id", pd.NA) if ni is not None else pd.NA,
            "nearest_icustay_hadm_id": ni.get("hadm_id", pd.NA) if ni is not None else pd.NA,
        })

    d = pd.DataFrame(rows)
    d.to_csv(out/"WFP_TEMPORAL_LINKAGE_AUDIT_PRIVATE.csv", index=False)

    exact_adm = d["n_exact_admission"].eq(1)
    exact_icu = d["n_exact_icustay"].eq(1)
    neither = ~(exact_adm | exact_icu)
    no_adm_exact = ~exact_adm
    no_icu_exact = ~exact_icu

    counts_by_threshold = {}
    for h in GAP_THRESHOLDS_H:
        counts_by_threshold[str(h)] = {
            "admission_within_h_among_nonexact":
                int((no_adm_exact & d["nearest_admission_gap_h"].le(h)).sum()),
            "icustay_within_h_among_nonexact":
                int((no_icu_exact & d["nearest_icustay_gap_h"].le(h)).sum()),
            "admission_total_exact_or_within_h":
                int((exact_adm | d["nearest_admission_gap_h"].le(h)).sum()),
            "icustay_total_exact_or_within_h":
                int((exact_icu | d["nearest_icustay_gap_h"].le(h)).sum()),
        }

    cross = {
        "exact_both": int((exact_adm & exact_icu).sum()),
        "exact_admission_only": int((exact_adm & ~exact_icu).sum()),
        "exact_icustay_only": int((~exact_adm & exact_icu).sum()),
        "neither_exact": int((~exact_adm & ~exact_icu).sum()),
    }

    result: Dict[str,Any] = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "7A.1",
        "decision": "WFP_TEMPORAL_LINKAGE_AUDIT_COMPLETE",
        "scientific_role": "EFFECT_BLIND_LINKAGE_DIAGNOSTIC_ONLY",
        "morphology_score_values_read": False,
        "morphology_covariate_associations_calculated": False,
        "cohort_n": EXPECTED_N,
        "timestamp": {
            "record_start_parse_n": int(c["record_start"].notna().sum()),
            "window_start_n": int(c["window_start"].notna().sum()),
        },
        "constitutional": {
            "patients_match_n": int(d["gender"].notna().sum()),
            "age_at_waveform_topcoded_nonmissing_n": int(d["age_years_capped90"].notna().sum()),
            "sex_gender_nonmissing_n": int(d["gender"].notna().sum()),
            "age_90plus_n": int(d["age_90plus"].eq(True).sum()),
        },
        "exact_linkage": cross,
        "nearest_gap_threshold_counts": counts_by_threshold,
        "nonexact_admission_gap_hours_summary": qstr(d.loc[no_adm_exact, "nearest_admission_gap_h"]),
        "nonexact_icustay_gap_hours_summary": qstr(d.loc[no_icu_exact, "nearest_icustay_gap_h"]),
        "nonexact_admission_direction_counts":
            d.loc[no_adm_exact, "nearest_admission_direction"].value_counts(dropna=False).to_dict(),
        "nonexact_icustay_direction_counts":
            d.loc[no_icu_exact, "nearest_icustay_direction"].value_counts(dropna=False).to_dict(),
        "boundary": [
            "No z1..z8 values were read.",
            "No morphology association was calculated.",
            "This audit does not choose a temporal tolerance.",
            "Any tolerance or complete-case rule must be frozen in Stage 7B after reviewing this effect-blind audit.",
            "Age is computed at waveform-window time and does not depend on admission/ICU linkage."
        ],
        "hashes": {
            "manifest": sha256_file(manifest_path),
            "score_file": sha256_file(score_path),
            "patients": sha256_file(patients_path),
            "admissions": sha256_file(admissions_path),
            "icustays": sha256_file(icustays_path),
        }
    }
    (out/"WFP_TEMPORAL_LINKAGE_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)+"\n",
        encoding="utf-8"
    )

    lines = [
        "WF-P TEMPORAL LINKAGE AUDIT",
        "===========================",
        "Decision: WFP_TEMPORAL_LINKAGE_AUDIT_COMPLETE",
        "Scientific role: EFFECT-BLIND LINKAGE DIAGNOSTIC ONLY",
        f"Frozen cohort n: {EXPECTED_N}",
        "Morphology score values read: NO",
        "Morphology-covariate associations calculated: NO",
        "",
        "Timestamp reconstruction:",
        f"  record start parsed: {int(c['record_start'].notna().sum())}/{EXPECTED_N}",
        f"  waveform window start available: {int(c['window_start'].notna().sum())}/{EXPECTED_N}",
        "",
        "Constitutional variables:",
        f"  age at waveform time (90+ top-coded): {int(d['age_years_capped90'].notna().sum())}/{EXPECTED_N}",
        f"  age 90+ top-coded patients: {int(d['age_90plus'].eq(True).sum())}",
        f"  sex/gender: {int(d['gender'].notna().sum())}/{EXPECTED_N}",
        "",
        "Exact temporal linkage cross-tab:",
        f"  exact both ADMISSION + ICUSTAY: {cross['exact_both']}",
        f"  exact ADMISSION only: {cross['exact_admission_only']}",
        f"  exact ICUSTAY only: {cross['exact_icustay_only']}",
        f"  neither exact: {cross['neither_exact']}",
        "",
        "Nearest interval gaps among non-exact cases:",
        f"  ADMISSION gap hours: {result['nonexact_admission_gap_hours_summary']}",
        f"  ICUSTAY gap hours: {result['nonexact_icustay_gap_hours_summary']}",
        f"  ADMISSION direction: {result['nonexact_admission_direction_counts']}",
        f"  ICUSTAY direction: {result['nonexact_icustay_direction_counts']}",
        "",
        "Coverage if a tolerance were later chosen (DIAGNOSTIC ONLY; NOT YET AUTHORIZED):"
    ]
    for h in GAP_THRESHOLDS_H:
        z = counts_by_threshold[str(h)]
        lines += [
            f"  <= {h:3d} h: ADMISSION exact-or-near "
            f"{z['admission_total_exact_or_within_h']}/{EXPECTED_N}; "
            f"ICUSTAY exact-or-near {z['icustay_total_exact_or_within_h']}/{EXPECTED_N}"
        ]
    lines += [
        "",
        "Boundary:",
        "  No tolerance selected here.",
        "  No clinical variables merged with z1..z8.",
        "  Return this TXT before Stage 7B freeze."
    ]
    (out/"WFP_TEMPORAL_LINKAGE_AUDIT.txt").write_text(
        "\n".join(lines)+"\n", encoding="utf-8"
    )
    print("\n".join(lines))

if __name__ == "__main__":
    main()
