#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7A.2 — effect-blind height availability / stability preflight.

Purpose
-------
Assess whether height can be included as an extended constitutional covariate
before opening morphology-covariate associations.

Scientific boundary
-------------------
- Reads frozen WF-P score file: patient_id ONLY.
- Reads CHARTEVENTS only for prespecified height ITEMIDs and the 978 frozen subjects.
- Does NOT read z1..z8.
- Does NOT calculate morphology associations.
- Does NOT change the frozen B8 basis or dimension.

Height ITEMIDs follow MIT-LCP MIMIC-III height_first_day.sql:
  226730, 920, 1394, 4187, 3486, 3485, 4188
The following are converted from inches to cm:
  920, 1394, 4187, 3486
Other listed height ITEMIDs are treated as cm.

Patient-level height = median of all valid converted measurements.
Validity range is prospectively fixed at 100 < height_cm <= 250.
Rows marked ERROR != 0 are excluded.

Yield rule frozen before seeing height:
- coverage >= 0.80: authorize extended constitutional model age+sex+height
- 0.60 <= coverage < 0.80: height secondary complete-case sensitivity only
- coverage < 0.60: descriptive availability only; do not use in Q5 residual geometry

No imputation is authorized in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

EXPECTED_N = 978
HEIGHT_ITEMIDS = {226730, 920, 1394, 4187, 3486, 3485, 4188}
INCH_ITEMIDS = {920, 1394, 4187, 3486}
CHUNKSIZE = 1_000_000
MIN_HEIGHT_CM = 100.0
MAX_HEIGHT_CM = 250.0

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def subject_id_from_pid(pid: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(pid).strip())
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chartevents", required=True)
    ap.add_argument("--score-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    a = ap.parse_args()

    ce_path = Path(a.chartevents).expanduser().resolve()
    score_path = Path(a.score_file).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not ce_path.is_file():
        raise SystemExit(f"FAIL: CHARTEVENTS not found: {ce_path}")
    if not score_path.is_file():
        raise SystemExit(f"FAIL: frozen score file not found: {score_path}")

    ids = pd.read_csv(
        score_path, usecols=["patient_id"], dtype={"patient_id": str}
    )
    if len(ids) != EXPECTED_N or ids["patient_id"].duplicated().any():
        raise SystemExit(
            "FAIL: frozen score patient-id frame is not 978 unique patients"
        )
    ids["subject_id"] = ids["patient_id"].map(subject_id_from_pid)
    if ids["subject_id"].isna().any():
        raise SystemExit("FAIL: could not parse subject_id")
    subject_set = set(ids["subject_id"].astype(int))

    # Store only the tiny selected height rows.
    selected = []
    total_rows = 0
    candidate_item_rows = 0
    subject_candidate_rows = 0
    valid_rows = 0
    t0 = time.time()

    usecols = ["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM", "ERROR"]
    reader = pd.read_csv(
        ce_path,
        usecols=usecols,
        chunksize=a.chunksize,
        low_memory=False,
        compression="infer"
    )

    for k, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        chunk.columns = [str(c).lower() for c in chunk.columns]

        item = pd.to_numeric(chunk["itemid"], errors="coerce")
        m_item = item.isin(HEIGHT_ITEMIDS)
        candidate_item_rows += int(m_item.sum())
        if not m_item.any():
            if k % 20 == 0:
                elapsed = (time.time()-t0)/60
                print(
                    f"[progress] chunks={k} rows={total_rows:,} "
                    f"selected_valid={valid_rows:,} elapsed_min={elapsed:.1f}",
                    flush=True
                )
            continue

        z = chunk.loc[m_item].copy()
        z["subject_id"] = pd.to_numeric(
            z["subject_id"], errors="coerce"
        ).astype("Int64")
        z = z[z["subject_id"].isin(subject_set)]
        subject_candidate_rows += len(z)
        if not len(z):
            continue

        z["itemid"] = pd.to_numeric(z["itemid"], errors="coerce").astype("Int64")
        z["valuenum"] = pd.to_numeric(z["valuenum"], errors="coerce")
        z["error"] = pd.to_numeric(z["error"], errors="coerce")

        # MIMIC convention: keep error NULL or 0.
        z = z[z["error"].isna() | z["error"].eq(0)]
        z = z[z["valuenum"].notna() & z["itemid"].notna()]

        z["height_cm"] = z["valuenum"].astype(float)
        m_in = z["itemid"].astype(int).isin(INCH_ITEMIDS)
        z.loc[m_in, "height_cm"] = z.loc[m_in, "height_cm"] * 2.54

        z = z[
            (z["height_cm"] > MIN_HEIGHT_CM) &
            (z["height_cm"] <= MAX_HEIGHT_CM)
        ]

        valid_rows += len(z)
        if len(z):
            selected.append(
                z[["subject_id","itemid","charttime","height_cm"]].copy()
            )

        if k % 20 == 0:
            elapsed = (time.time()-t0)/60
            print(
                f"[progress] chunks={k} rows={total_rows:,} "
                f"selected_valid={valid_rows:,} elapsed_min={elapsed:.1f}",
                flush=True
            )

    if selected:
        h = pd.concat(selected, ignore_index=True)
    else:
        h = pd.DataFrame(
            columns=["subject_id","itemid","charttime","height_cm"]
        )

    # subject-level robust summaries
    if len(h):
        g = h.groupby("subject_id")["height_cm"]
        summary = g.agg(
            n_height="size",
            height_median_cm="median",
            height_mean_cm="mean",
            height_min_cm="min",
            height_max_cm="max",
            height_sd_cm="std",
        ).reset_index()

        q25 = g.quantile(0.25).rename("height_q25_cm")
        q75 = g.quantile(0.75).rename("height_q75_cm")
        summary = summary.merge(q25, on="subject_id").merge(q75, on="subject_id")
        summary["height_iqr_cm"] = (
            summary["height_q75_cm"] - summary["height_q25_cm"]
        )
    else:
        summary = pd.DataFrame(columns=[
            "subject_id","n_height","height_median_cm","height_mean_cm",
            "height_min_cm","height_max_cm","height_sd_cm",
            "height_q25_cm","height_q75_cm","height_iqr_cm"
        ])

    private = ids.merge(summary, on="subject_id", how="left", validate="one_to_one")
    private.to_csv(out/"WFP_HEIGHT_PREFLIGHT_PRIVATE.csv", index=False)

    n_height = int(private["height_median_cm"].notna().sum())
    coverage = n_height / EXPECTED_N
    n_ge2 = int(private["n_height"].fillna(0).ge(2).sum())
    n_iqr5 = int(
        (
            private["height_median_cm"].notna() &
            (
                private["height_iqr_cm"].isna() |
                private["height_iqr_cm"].le(5.0)
            )
        ).sum()
    )

    if coverage >= 0.80:
        role = "AUTHORIZE_EXTENDED_CONSTITUTIONAL_MODEL"
    elif coverage >= 0.60:
        role = "SECONDARY_COMPLETE_CASE_SENSITIVITY_ONLY"
    else:
        role = "DESCRIPTIVE_AVAILABILITY_ONLY"

    vals = private["height_median_cm"].dropna().to_numpy(float)
    if len(vals):
        qq = np.quantile(vals, [0,.25,.5,.75,1])
        height_summary = {
            "min": float(qq[0]), "q25": float(qq[1]),
            "median": float(qq[2]), "q75": float(qq[3]),
            "max": float(qq[4])
        }
    else:
        height_summary = {}

    result: Dict[str,Any] = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "7A.2",
        "decision": "WFP_HEIGHT_PREFLIGHT_COMPLETE",
        "scientific_role": "EFFECT_BLIND_HEIGHT_AVAILABILITY_ONLY",
        "morphology_score_values_read": False,
        "morphology_covariate_associations_calculated": False,
        "cohort_n": EXPECTED_N,
        "height_definition": {
            "itemids": sorted(HEIGHT_ITEMIDS),
            "inch_to_cm_itemids": sorted(INCH_ITEMIDS),
            "patient_summary": "median_of_all_valid_measurements",
            "valid_range_cm": [MIN_HEIGHT_CM, MAX_HEIGHT_CM],
            "error_rows_excluded": True
        },
        "yield_rule": {
            "coverage_ge_0_80": "extended constitutional model age+sex+height authorized",
            "coverage_0_60_to_0_80": "secondary complete-case sensitivity only",
            "coverage_lt_0_60": "descriptive only"
        },
        "results": {
            "height_nonmissing_n": n_height,
            "coverage": coverage,
            "n_with_at_least_2_measurements": n_ge2,
            "n_with_height_and_iqr_le_5cm_or_single_measurement": n_iqr5,
            "patient_height_distribution_cm": height_summary,
            "authorized_role_for_stage7b": role
        },
        "scan": {
            "chartevents_rows_scanned": total_rows,
            "height_item_rows_all_subjects": candidate_item_rows,
            "height_item_rows_frozen_subjects_before_qc": subject_candidate_rows,
            "valid_height_rows_frozen_subjects": valid_rows
        },
        "boundary": [
            "No z1..z8 values were read.",
            "No morphology association was calculated.",
            "No height imputation is authorized.",
            "This stage only determines the role of height in the later frozen Q4/Q5 specification."
        ],
        "hashes": {
            "chartevents_sha256": sha256_file(ce_path),
            "score_file_sha256": sha256_file(score_path)
        }
    }

    (out/"WFP_HEIGHT_PREFLIGHT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8"
    )

    lines = [
        "WF-P HEIGHT AVAILABILITY PREFLIGHT",
        "================================",
        "Decision: WFP_HEIGHT_PREFLIGHT_COMPLETE",
        "Scientific role: EFFECT-BLIND HEIGHT AVAILABILITY ONLY",
        f"Frozen cohort n: {EXPECTED_N}",
        "Morphology score values read: NO",
        "Morphology-covariate associations calculated: NO",
        "",
        f"Height available: {n_height}/{EXPECTED_N} ({coverage:.4f})",
        f"At least 2 valid height measurements: {n_ge2}/{EXPECTED_N}",
        f"Height with IQR<=5 cm or single measurement: {n_iqr5}/{EXPECTED_N}",
        f"Patient-level height distribution (cm): {height_summary}",
        "",
        f"Stage 7B height role: {role}",
        "",
        "Frozen yield rule:",
        "  coverage >=0.80 -> extended constitutional model age+sex+height",
        "  0.60-<0.80 -> secondary complete-case sensitivity only",
        "  <0.60 -> descriptive availability only",
        "",
        "Boundary:",
        "  No imputation.",
        "  Do not merge height with z1..z8 until Stage 7B is frozen."
    ]
    (out/"WFP_HEIGHT_PREFLIGHT.txt").write_text(
        "\n".join(lines)+"\n", encoding="utf-8"
    )
    print("\n".join(lines))

if __name__ == "__main__":
    main()
