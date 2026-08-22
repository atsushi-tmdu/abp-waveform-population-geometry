#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7A — effect-blind baseline phenotype / clinical linkage preflight.

Links the frozen WF-P discovery cohort to MIMIC-III clinical tables WITHOUT
reading morphology score values or calculating morphology-covariate associations.

Reads:
- Validation1000 waveform manifest (record locator only)
- frozen WF-P patient-score CSV: patient_id column ONLY
- MIMIC-III PATIENTS / ADMISSIONS / ICUSTAYS
- DIAGNOSES_ICD only for diagnosis-availability counts
- D_ITEMS / CHARTEVENTS presence only (CHARTEVENTS is NOT scanned)

Outputs:
- aggregate linkage/availability report
- PRIVATE deidentified patient-level linkage table for later frozen analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

EXPECTED_ANALYSABLE_N = 978

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def find_table(root: Path, stem: str, required: bool = True) -> Optional[Path]:
    candidates = [
        root / f"{stem}.csv", root / f"{stem}.CSV",
        root / f"{stem}.csv.gz", root / f"{stem}.CSV.gz",
        root / f"{stem.lower()}.csv", root / f"{stem.lower()}.csv.gz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    if required:
        raise FileNotFoundError(
            f"Could not find {stem}.csv or {stem}.csv.gz directly under {root}"
        )
    return None

def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def parse_subject_id(patient_id: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(patient_id).strip())
    return int(m.group(1)) if m else None

def parse_record_datetime(record_name: str, record_path: str):
    text = str(record_name).strip()
    if not text:
        text = Path(str(record_path)).name
    m = re.search(
        r"p?\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})(?:-(\d{2}))?",
        text
    )
    if not m:
        return pd.NaT
    year, month, day, hour, minute = map(int, m.groups()[:5])
    second = int(m.group(6)) if m.group(6) is not None else 0
    try:
        return pd.Timestamp(
            year=year, month=month, day=day,
            hour=hour, minute=minute, second=second
        )
    except Exception:
        return pd.NaT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--score-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clinical = Path(args.clinical_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    score_path = Path(args.score_file).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not clinical.is_dir():
        raise SystemExit(f"FAIL: clinical root not found: {clinical}")

    patients_path = find_table(clinical, "PATIENTS")
    admissions_path = find_table(clinical, "ADMISSIONS")
    icustays_path = find_table(clinical, "ICUSTAYS")
    diagnoses_path = find_table(clinical, "DIAGNOSES_ICD", required=False)
    d_items_path = find_table(clinical, "D_ITEMS", required=False)
    chartevents_path = find_table(clinical, "CHARTEVENTS", required=False)

    frozen_ids = pd.read_csv(
        score_path, usecols=["patient_id"], dtype={"patient_id": str}
    )
    if len(frozen_ids) != EXPECTED_ANALYSABLE_N:
        raise SystemExit(
            f"FAIL: expected {EXPECTED_ANALYSABLE_N} frozen patients, "
            f"found {len(frozen_ids)}"
        )
    if frozen_ids["patient_id"].duplicated().any():
        raise SystemExit("FAIL: duplicate patient_id in frozen score file")

    manifest = pd.read_csv(
        manifest_path, dtype={"patient_id": str}, low_memory=False
    )
    needed = [
        "patient_id", "record_path", "record_name",
        "fs", "start_sample", "abp_source_name"
    ]
    missing = [c for c in needed if c not in manifest.columns]
    if missing:
        raise SystemExit(f"FAIL: manifest missing fields: {missing}")

    cohort = frozen_ids.merge(
        manifest[needed], on="patient_id", how="left", validate="one_to_one"
    )
    if cohort["record_name"].isna().any():
        raise SystemExit("FAIL: frozen WF-P patient missing from manifest")

    cohort["subject_id"] = cohort["patient_id"].map(parse_subject_id)
    cohort["record_start"] = [
        parse_record_datetime(a, b)
        for a, b in zip(cohort["record_name"], cohort["record_path"])
    ]
    cohort["fs_num"] = pd.to_numeric(cohort["fs"], errors="coerce")
    cohort["start_sample_num"] = pd.to_numeric(
        cohort["start_sample"], errors="coerce"
    )
    cohort["window_start"] = cohort["record_start"] + pd.to_timedelta(
        cohort["start_sample_num"] / cohort["fs_num"], unit="s"
    )

    pat = read_table(patients_path)
    adm = read_table(admissions_path)
    icu = read_table(icustays_path)

    for req, df, label in [
        ({"subject_id", "gender", "dob"}, pat, "PATIENTS"),
        ({"subject_id", "hadm_id", "admittime", "dischtime"}, adm, "ADMISSIONS"),
        ({"subject_id", "hadm_id", "icustay_id", "intime", "outtime"}, icu, "ICUSTAYS"),
    ]:
        if not req.issubset(df.columns):
            raise SystemExit(f"FAIL: {label} missing {sorted(req-set(df.columns))}")

    pat["subject_id"] = pd.to_numeric(
        pat["subject_id"], errors="coerce"
    ).astype("Int64")
    adm["subject_id"] = pd.to_numeric(
        adm["subject_id"], errors="coerce"
    ).astype("Int64")
    adm["hadm_id"] = pd.to_numeric(
        adm["hadm_id"], errors="coerce"
    ).astype("Int64")
    icu["subject_id"] = pd.to_numeric(
        icu["subject_id"], errors="coerce"
    ).astype("Int64")
    icu["hadm_id"] = pd.to_numeric(
        icu["hadm_id"], errors="coerce"
    ).astype("Int64")
    icu["icustay_id"] = pd.to_numeric(
        icu["icustay_id"], errors="coerce"
    ).astype("Int64")

    pat["dob"] = pd.to_datetime(pat["dob"], errors="coerce")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    pat_idx = pat.set_index("subject_id", drop=False)
    adm_groups = {
        int(k): g.copy()
        for k, g in adm.dropna(subset=["subject_id"]).groupby("subject_id")
    }
    icu_groups = {
        int(k): g.copy()
        for k, g in icu.dropna(subset=["subject_id"]).groupby("subject_id")
    }

    rows = []
    for _, r in cohort.iterrows():
        sid = int(r["subject_id"]) if pd.notna(r["subject_id"]) else None
        t = r["window_start"]

        x: Dict[str, Any] = {
            "patient_id": r["patient_id"],
            "subject_id": sid,
            "window_start": "" if pd.isna(t) else str(t),
            "abp_source_name": r["abp_source_name"],
            "subject_match_in_patients": False,
            "n_exact_icustay_matches": 0,
            "n_exact_admission_matches": 0,
            "icustay_id": pd.NA,
            "hadm_id": pd.NA,
            "gender": pd.NA,
            "age_years_capped90": np.nan,
            "age_90plus": pd.NA,
            "ethnicity": pd.NA,
            "marital_status": pd.NA,
            "insurance": pd.NA,
            "language": pd.NA,
            "religion": pd.NA,
            "admission_type": pd.NA,
            "dbsource": pd.NA,
            "first_careunit": pd.NA,
            "last_careunit": pd.NA,
        }

        pr = None
        if sid is not None and sid in pat_idx.index:
            pr = pat_idx.loc[sid]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            x["subject_match_in_patients"] = True
            x["gender"] = pr.get("gender", pd.NA)

        if sid is not None and pd.notna(t):
            ig = icu_groups.get(sid, pd.DataFrame())
            ag = adm_groups.get(sid, pd.DataFrame())

            exact_icu = (
                ig[(ig["intime"] <= t) & (t <= ig["outtime"])]
                if len(ig) else pd.DataFrame()
            )
            exact_adm = (
                ag[(ag["admittime"] <= t) & (t <= ag["dischtime"])]
                if len(ag) else pd.DataFrame()
            )

            x["n_exact_icustay_matches"] = len(exact_icu)
            x["n_exact_admission_matches"] = len(exact_adm)

            chosen_adm = None
            if len(exact_icu) == 1:
                ir = exact_icu.iloc[0]
                x["icustay_id"] = ir.get("icustay_id", pd.NA)
                x["hadm_id"] = ir.get("hadm_id", pd.NA)
                x["dbsource"] = ir.get("dbsource", pd.NA)
                x["first_careunit"] = ir.get("first_careunit", pd.NA)
                x["last_careunit"] = ir.get("last_careunit", pd.NA)

                z = ag[ag["hadm_id"] == ir.get("hadm_id")] if len(ag) else pd.DataFrame()
                if len(z) == 1:
                    chosen_adm = z.iloc[0]

            if chosen_adm is None and len(exact_adm) == 1:
                chosen_adm = exact_adm.iloc[0]
                x["hadm_id"] = chosen_adm.get("hadm_id", x["hadm_id"])

            if chosen_adm is not None:
                for col in [
                    "ethnicity", "marital_status", "insurance",
                    "language", "religion", "admission_type"
                ]:
                    if col in chosen_adm.index:
                        x[col] = chosen_adm.get(col, pd.NA)

                if pr is not None:
                    dob = pr.get("dob", pd.NaT)
                    at = chosen_adm.get("admittime", pd.NaT)
                    if pd.notna(dob) and pd.notna(at):
                        age = (
                            (at - dob).total_seconds()
                            / (365.2425 * 86400.0)
                        )
                        if age > 89:
                            x["age_years_capped90"] = 90.0
                            x["age_90plus"] = True
                        elif age >= 0:
                            x["age_years_capped90"] = float(age)
                            x["age_90plus"] = False
        rows.append(x)

    link = pd.DataFrame(rows)

    diag_summary = {
        "table_present": diagnoses_path is not None,
        "linked_patients_with_any_current_admission_icd9": None,
        "linked_fraction_with_any_current_admission_icd9": None,
        "unique_icd9_codes_in_linked_admissions": None,
        "phenotype_mapping_performed": False,
    }
    if diagnoses_path is not None:
        diag = read_table(diagnoses_path)
        if {"hadm_id", "icd9_code"}.issubset(diag.columns):
            diag["hadm_id"] = pd.to_numeric(
                diag["hadm_id"], errors="coerce"
            ).astype("Int64")
            linked_hadm = set(
                int(v) for v in link["hadm_id"].dropna().astype(int)
            )
            dz = diag[diag["hadm_id"].isin(linked_hadm)]
            hadms_dx = set(
                int(v) for v in dz["hadm_id"].dropna().astype(int)
            )
            n_dx = int(link["hadm_id"].map(
                lambda v: pd.notna(v) and int(v) in hadms_dx
            ).sum())
            diag_summary.update({
                "linked_patients_with_any_current_admission_icd9": n_dx,
                "linked_fraction_with_any_current_admission_icd9":
                    n_dx / len(link),
                "unique_icd9_codes_in_linked_admissions":
                    int(dz["icd9_code"].dropna().astype(str).nunique()),
            })

    def nn(col):
        return int(link[col].notna().sum())

    def nlevels(col):
        return int(link[col].dropna().astype(str).nunique())

    pat_match = int(link["subject_match_in_patients"].sum())
    exact_icu = int((link["n_exact_icustay_matches"] == 1).sum())
    exact_adm = int((link["n_exact_admission_matches"] == 1).sum())

    private_path = out / "WFP_BASELINE_LINKAGE_PRIVATE.csv"
    link.to_csv(private_path, index=False)

    decision = (
        "WFP_BASELINE_LINKAGE_PREFLIGHT_PASS"
        if pat_match == EXPECTED_ANALYSABLE_N and exact_icu >= 900
        else "WFP_BASELINE_LINKAGE_PREFLIGHT_REVIEW_REQUIRED"
    )

    result = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "7A",
        "decision": decision,
        "scientific_role": "effect_blind_clinical_linkage_and_variable_inventory",
        "frozen_morphology_score_values_read": False,
        "morphology_covariate_associations_calculated": False,
        "clinical_values_opened": True,
        "cohort_n": EXPECTED_ANALYSABLE_N,
        "linkage": {
            "patient_subject_match_n": pat_match,
            "exact_one_icustay_match_n": exact_icu,
            "exact_one_admission_match_n": exact_adm,
            "nonunique_or_missing_icustay_match_n":
                EXPECTED_ANALYSABLE_N - exact_icu,
            "record_datetime_parse_nonmissing_n":
                int(cohort["record_start"].notna().sum()),
            "window_start_nonmissing_n":
                int(cohort["window_start"].notna().sum()),
        },
        "variable_inventory": {
            "tier_A_constitutional_core": {
                "age": {"nonmissing_n": nn("age_years_capped90"),
                        "role": "PRIMARY_CANDIDATE"},
                "sex": {"nonmissing_n": nn("gender"),
                        "levels_n": nlevels("gender"),
                        "role": "PRIMARY_CANDIDATE"},
            },
            "tier_A2_anthropometric": {
                "height": {
                    "status": "NOT_EXTRACTED",
                    "chartevents_present": chartevents_path is not None,
                    "d_items_present": d_items_path is not None,
                    "role": "PRIMARY_IF_RELIABLE_COVERAGE"
                },
                "weight_bmi": {
                    "status": "NOT_EXTRACTED",
                    "role": "SECONDARY_SLOW_PROXY_ONLY"
                },
            },
            "tier_B_demographic_social_context": {
                "ethnicity": {
                    "nonmissing_n": nn("ethnicity"),
                    "levels_n": nlevels("ethnicity"),
                    "role": "SECONDARY_NOT_BIOLOGICAL_TRAIT"
                },
                "marital_status": {"nonmissing_n": nn("marital_status")},
                "insurance": {"nonmissing_n": nn("insurance")},
                "language": {"nonmissing_n": nn("language")},
            },
            "tier_C_chronic_baseline_phenotype": {
                "diagnosis_source": diag_summary,
                "planned_examples": [
                    "hypertension", "diabetes", "chronic kidney disease",
                    "coronary artery disease/prior myocardial infarction",
                    "chronic heart failure", "atrial fibrillation",
                    "valvular disease", "peripheral vascular disease",
                    "chronic pulmonary disease", "Charlson/Elixhauser burden"
                ],
                "role":
                    "ASSOCIATION_MAPPING_NOT_PRIMARY_CONSTITUTIONAL_RESIDUALIZATION"
            },
            "tier_T_technical_context": {
                "dbsource": {
                    "nonmissing_n": nn("dbsource"),
                    "levels_n": nlevels("dbsource")
                },
                "first_careunit": {
                    "nonmissing_n": nn("first_careunit"),
                    "levels_n": nlevels("first_careunit")
                },
                "last_careunit": {
                    "nonmissing_n": nn("last_careunit"),
                    "levels_n": nlevels("last_careunit")
                },
                "abp_source_name": {
                    "nonmissing_n": nn("abp_source_name"),
                    "levels_n": nlevels("abp_source_name")
                },
                "admission_type": {
                    "nonmissing_n": nn("admission_type"),
                    "levels_n": nlevels("admission_type"),
                    "role": "ACUTE_CONTEXT_NOT_FIXED_TRAIT"
                },
            },
        },
        "source_tables": {
            "patients": str(patients_path),
            "admissions": str(admissions_path),
            "icustays": str(icustays_path),
            "diagnoses_icd": str(diagnoses_path) if diagnoses_path else None,
            "d_items": str(d_items_path) if d_items_path else None,
            "chartevents": {
                "path": str(chartevents_path) if chartevents_path else None,
                "opened": False,
                "size_bytes":
                    chartevents_path.stat().st_size if chartevents_path else None,
            },
        },
        "input_hashes": {
            "validation_manifest_sha256": sha256_file(manifest_path),
            "frozen_score_file_sha256": sha256_file(score_path),
            "patients_sha256": sha256_file(patients_path),
            "admissions_sha256": sha256_file(admissions_path),
            "icustays_sha256": sha256_file(icustays_path),
            "private_linkage_sha256": sha256_file(private_path),
        },
        "boundary": [
            "No z1..z8 values were read.",
            "No morphology-covariate association was calculated.",
            "CHARTEVENTS was not scanned.",
            "Chronic ICD-9 phenotype definitions were not applied.",
            "Stage 7B must freeze covariate tiers and missingness rules before clinical variables are merged with frozen morphology scores."
        ],
    }

    (out / "WFP_BASELINE_LINKAGE_PREFLIGHT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )

    lines = [
        "WF-P BASELINE PHENOTYPE LINKAGE PREFLIGHT",
        "========================================",
        f"Decision: {decision}",
        "Scientific role: EFFECT-BLIND CLINICAL LINKAGE / VARIABLE INVENTORY",
        f"Frozen WF-P cohort n: {EXPECTED_ANALYSABLE_N}",
        "Morphology score values read: NO",
        "Morphology-covariate associations calculated: NO",
        "Clinical values opened: YES",
        "",
        "Linkage:",
        f"  patient -> PATIENTS match: {pat_match}/{EXPECTED_ANALYSABLE_N}",
        f"  exact ICUSTAY containing waveform window start: {exact_icu}/{EXPECTED_ANALYSABLE_N}",
        f"  exact ADMISSION containing waveform window start: {exact_adm}/{EXPECTED_ANALYSABLE_N}",
        "",
        "Core constitutional availability:",
        f"  age (90+ top-coded): {nn('age_years_capped90')}/{EXPECTED_ANALYSABLE_N}",
        f"  sex/gender: {nn('gender')}/{EXPECTED_ANALYSABLE_N}",
        "",
        "Additional inventory:",
        f"  ethnicity: {nn('ethnicity')}/{EXPECTED_ANALYSABLE_N}",
        f"  dbsource: {nn('dbsource')}/{EXPECTED_ANALYSABLE_N}",
        f"  first careunit: {nn('first_careunit')}/{EXPECTED_ANALYSABLE_N}",
        f"  DIAGNOSES_ICD present: {'YES' if diagnoses_path else 'NO'}",
        f"  D_ITEMS present: {'YES' if d_items_path else 'NO'}",
        f"  CHARTEVENTS present: {'YES' if chartevents_path else 'NO'} (NOT OPENED)",
        "",
        "Planned roles:",
        "  Core residualization: age + sex; add height only if reliable coverage is demonstrated.",
        "  Weight/BMI: secondary slow/body-size proxy only.",
        "  Ethnicity: secondary demographic/context association; not a biological-trait interpretation.",
        "  Chronic diseases/comorbidity burden: association mapping; do not remove them in primary constitutional residualization.",
        "  dbsource/careunit/ABP-vs-ART: technical/context sensitivity.",
        "",
        "Boundary:",
        "  Do not merge clinical covariates with z1..z8 until Stage 7B is frozen.",
    ]
    (out / "WFP_BASELINE_LINKAGE_PREFLIGHT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))

if __name__ == "__main__":
    main()
