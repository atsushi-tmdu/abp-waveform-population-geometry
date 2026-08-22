#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7B-NL — effect-blind preflight.

This is a NEW sensitivity stage and does not modify the prospectively frozen
Stage 7B analysis.

The script:
- reads patient_id only from the frozen B8 score file;
- checks age/sex and height availability;
- verifies the completed Stage 7B readout;
- does NOT read z1..z8 values;
- does NOT calculate nonlinear effects.
"""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_N = 978
EXPECTED_HEIGHT_N = 693

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

PRIMARY_HEIGHT_COLUMN = "height_median_cm"


def detect_height_column(df: pd.DataFrame) -> str:
    """Use the exact height summary already used in frozen Stage 7B.

    The height preflight intentionally stores several patient-level summaries
    (median, mean, min, max, quartiles). They share the same non-missing count,
    so yield-based autodetection is not identifiable. Stage 7B used the
    patient-level median height; this sensitivity stage must replay that choice
    rather than select among summaries after seeing the schema.
    """
    if PRIMARY_HEIGHT_COLUMN not in df.columns:
        raise RuntimeError(
            f"Required frozen height column missing: {PRIMARY_HEIGHT_COLUMN}; "
            f"columns={list(df.columns)}"
        )
    x = pd.to_numeric(df[PRIMARY_HEIGHT_COLUMN], errors="coerce")
    valid = x.notna() & (x > 100) & (x <= 250)
    if int(valid.sum()) != EXPECTED_HEIGHT_N:
        raise RuntimeError(
            f"Frozen height column {PRIMARY_HEIGHT_COLUMN} has "
            f"{int(valid.sum())} valid patients; expected {EXPECTED_HEIGHT_N}"
        )
    return PRIMARY_HEIGHT_COLUMN

def self_test() -> int:
    # Multiple summaries may legitimately have identical coverage. The frozen
    # median-height column must still be selected deterministically.
    n = EXPECTED_HEIGHT_N
    df = pd.DataFrame({
        "patient_id": [f"p{i:04d}" for i in range(n)],
        "height_median_cm": np.full(n, 170.0),
        "height_mean_cm": np.full(n, 171.0),
        "height_min_cm": np.full(n, 168.0),
        "height_max_cm": np.full(n, 173.0),
    })
    if detect_height_column(df) != "height_median_cm":
        raise RuntimeError("frozen median-height selection self-test failed")
    print("WF-P Stage7B-NL preflight self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file")
    ap.add_argument("--temporal-linkage")
    ap.add_argument("--height-preflight")
    ap.add_argument("--stage7b-spec")
    ap.add_argument("--stage7b-results-json")
    ap.add_argument("--stage7b-readout")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    needed = [
        "score_file","temporal_linkage","height_preflight",
        "stage7b_spec","stage7b_results_json","stage7b_readout","out"
    ]
    missing = [x for x in needed if getattr(a, x) is None]
    if missing:
        raise SystemExit(f"Missing required args: {missing}")

    scorep = Path(a.score_file).expanduser().resolve()
    temporalp = Path(a.temporal_linkage).expanduser().resolve()
    heightp = Path(a.height_preflight).expanduser().resolve()
    specp = Path(a.stage7b_spec).expanduser().resolve()
    resultp = Path(a.stage7b_results_json).expanduser().resolve()
    readoutp = Path(a.stage7b_readout).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for p in [scorep, temporalp, heightp, specp, resultp, readoutp]:
        if not p.is_file():
            raise SystemExit(f"FAIL missing file: {p}")

    # Critical boundary: score values are not opened here.
    scores = pd.read_csv(scorep, usecols=["patient_id"], dtype={"patient_id": str})
    if len(scores) != EXPECTED_N or scores["patient_id"].duplicated().any():
        raise SystemExit("FAIL: frozen score cohort is not 978 unique patients")

    header = pd.read_csv(temporalp, nrows=0)
    required_temporal = ["patient_id","age_years_capped90","gender"]
    absent = [c for c in required_temporal if c not in header.columns]
    if absent:
        raise SystemExit(
            f"FAIL required temporal-linkage columns absent: {absent}; "
            f"columns={list(header.columns)}"
        )

    temporal = pd.read_csv(
        temporalp,
        usecols=required_temporal,
        dtype={"patient_id": str},
    )
    if temporal["patient_id"].duplicated().any():
        raise SystemExit("FAIL duplicate patient_id in temporal linkage")

    height = pd.read_csv(heightp, dtype={"patient_id": str})
    if "patient_id" not in height.columns:
        raise SystemExit("FAIL height-preflight file has no patient_id")
    if height["patient_id"].duplicated().any():
        raise SystemExit("FAIL duplicate patient_id in height-preflight file")
    height_col = detect_height_column(height)

    cohort = (
        scores.merge(temporal, on="patient_id", how="left", validate="one_to_one")
        .merge(
            height[["patient_id", height_col]],
            on="patient_id", how="left", validate="one_to_one"
        )
    )

    age = pd.to_numeric(cohort["age_years_capped90"], errors="coerce")
    sex = cohort["gender"].astype(str).str.upper().str.strip()
    hcm = pd.to_numeric(cohort[height_col], errors="coerce")
    hvalid = hcm.notna() & (hcm > 100) & (hcm <= 250)

    if int(age.notna().sum()) != EXPECTED_N:
        raise SystemExit(f"FAIL age availability={int(age.notna().sum())}/{EXPECTED_N}")
    if int(sex.isin(["M","F"]).sum()) != EXPECTED_N:
        raise SystemExit("FAIL sex availability/coding")
    if int(hvalid.sum()) != EXPECTED_HEIGHT_N:
        raise SystemExit(
            f"FAIL height complete-case count={int(hvalid.sum())}; "
            f"expected={EXPECTED_HEIGHT_N}"
        )

    readout = readoutp.read_text(encoding="utf-8", errors="replace")
    if "Decision: WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE" not in readout:
        raise SystemExit("FAIL completed Stage7B decision not found")

    markers = [
        "aggregate OOF R2: 0.029482",
        "age + age^2 + sex OOF R2: 0.026249",
        "Height complete-case n: 693",
        "age + sex + height OOF R2: 0.027070",
    ]
    missing_markers = [m for m in markers if m not in readout]
    if missing_markers:
        raise SystemExit(f"FAIL expected Stage7B markers absent: {missing_markers}")

    result = {
        "decision": "WFP_STAGE7B_NL_PREFLIGHT_PASS",
        "scientific_role": "EFFECT_BLIND_PRE_NONLINEAR_SENSITIVITY",
        "full_cohort_n": EXPECTED_N,
        "height_complete_case_n": EXPECTED_HEIGHT_N,
        "height_column_detected": height_col,
        "age_complete_n": int(age.notna().sum()),
        "sex_complete_n": int(sex.isin(["M","F"]).sum()),
        "age_topcoded90_n": int((age == 90).sum()),
        "morphology_score_values_read": False,
        "nonlinear_effects_calculated": False,
        "existing_stage7b_modified": False,
        "hashes": {
            "score_file_sha256": sha256_file(scorep),
            "temporal_linkage_sha256": sha256_file(temporalp),
            "height_preflight_sha256": sha256_file(heightp),
            "stage7b_spec_sha256": sha256_file(specp),
            "stage7b_results_sha256": sha256_file(resultp),
            "stage7b_readout_sha256": sha256_file(readoutp),
            "preflight_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "boundary": [
            "Existing Stage7B remains unchanged.",
            "No z1..z8 score values are read in this preflight.",
            "No nonlinear constitutional effect is calculated.",
            "Age above 89 is top-coded at 90 and cannot be reconstructed.",
            "No disease, treatment, outcome, or trajectory variable is accessed.",
        ],
    }

    (out / "WFP_STAGE7B_NL_PREFLIGHT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    txt = "\n".join([
        "WF-P STAGE 7B-NL PREFLIGHT",
        "===========================",
        "Decision: WFP_STAGE7B_NL_PREFLIGHT_PASS",
        "Scientific role: EFFECT-BLIND PRE-NONLINEAR SENSITIVITY",
        f"Frozen Stage7B cohort n: {EXPECTED_N}",
        f"Height complete-case n: {EXPECTED_HEIGHT_N}",
        f"Detected height column: {height_col}",
        f"Age available: {int(age.notna().sum())}/{EXPECTED_N}",
        f"Sex available: {int(sex.isin(['M','F']).sum())}/{EXPECTED_N}",
        f"Age top-coded at 90: {int((age == 90).sum())}",
        "Morphology score values read: NO",
        "Nonlinear effects calculated: NO",
        "Existing Stage7B modified: NO",
        "",
        "Next boundary:",
        "  Freeze the Stage7B-NL model family before opening nonlinear B8 effects.",
        "",
    ])
    (out / "WFP_STAGE7B_NL_PREFLIGHT.txt").write_text(txt, encoding="utf-8")
    print(txt, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
