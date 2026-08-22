#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 2B — effect-blind block-eligibility yield audit (Development50 only)

No Validation1000 access.
No morphology values, PCA, covariance, eigenvalues, effective rank, correlations,
or scientific-effect quantities are output.

Purpose:
Choose a minimum accepted-beats-per-60-s-block threshold using DEVELOPMENT
engineering yield only, before Validation1000 discovery access.

Candidate thresholds are frozen: 8, 16, 32 accepted beats per 60-s block.

Patient replicate-geometry eligibility is frozen as:
- >= 6 eligible 60-s blocks total
- >= 3 odd-indexed eligible blocks
- >= 3 even-indexed eligible blocks

Automatic threshold rule:
Choose the LARGEST candidate threshold for which >= 45/50 (90%) Development50
patients satisfy the patient eligibility rule.
If none qualifies, decision = STOP_AND_MODIFY.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

SOURCE_FS = 125.0
FULL_WINDOW_SEC = 1800
BLOCK_SEC = 60
EXPECTED_N = 50
EXPECTED_RUN1_SHA256 = "811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01"
CANDIDATE_MIN_BEATS = (8, 16, 32)
MIN_TOTAL_BLOCKS = 6
MIN_ODD_BLOCKS = 3
MIN_EVEN_BLOCKS = 3
MIN_DEV_ANALYSABLE = 45

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("wfp_run1_authoritative", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import authoritative Run-1")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def patient_block_counts(path: Path, run1) -> list[int]:
    pid, fs0, x125 = run1.load_case(path)
    if abs(float(fs0) - SOURCE_FS) > 1e-6:
        raise RuntimeError(f"{pid}: fs mismatch")
    total_sec = len(x125) / SOURCE_FS
    if total_sec + 1e-9 < FULL_WINDOW_SEC:
        raise RuntimeError(f"{pid}: signal shorter than 30 min")

    peaks = run1.detect_systolic_peaks_125(x125)
    built = run1.build_125_catalog(x125, peaks)
    if built is None:
        raise RuntimeError(f"{pid}: catalog failed")
    catalog, rep125 = built

    start = total_sec - FULL_WINDOW_SEC
    qc = run1.locked_qc_for_window(catalog, rep125, start, total_sec)
    if qc is None:
        raise RuntimeError(f"{pid}: locked QC failed")

    acc = np.asarray(qc["accepted_idx"], dtype=int)
    centers = catalog["center_sec"].to_numpy(float)[acc]

    counts = []
    for b in range(FULL_WINDOW_SEC // BLOCK_SEC):
        a = start + b * BLOCK_SEC
        z = a + BLOCK_SEC
        if b < (FULL_WINDOW_SEC // BLOCK_SEC) - 1:
            n = int(np.sum((centers >= a) & (centers < z)))
        else:
            n = int(np.sum((centers >= a) & (centers <= z)))
        counts.append(n)
    return counts

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_block_yield")
    args = ap.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    run1_path = root / "code" / "wp2_run1_development50.py"
    cases_dir = root / "data" / "abp125_pilot50" / "cases"

    if not run1_path.is_file():
        raise SystemExit("FAIL: authoritative Run-1 missing")
    observed_sha = sha256_file(run1_path)
    if observed_sha != EXPECTED_RUN1_SHA256:
        raise SystemExit(
            "FAIL: authoritative Run-1 SHA256 mismatch\n"
            f"observed={observed_sha}\nexpected={EXPECTED_RUN1_SHA256}"
        )
    cases = sorted(cases_dir.glob("*.npz"))
    if len(cases) != EXPECTED_N:
        raise SystemExit(f"FAIL: expected 50 Development50 cases, found {len(cases)}")

    run1 = load_module(run1_path)
    all_counts = []
    failures = []
    for p in cases:
        try:
            all_counts.append(patient_block_counts(p, run1))
        except Exception as e:
            failures.append({"file": p.name, "error": repr(e)})

    rows = []
    if not failures:
        A = np.asarray(all_counts, dtype=int)  # patient x 30 blocks
        for thr in CANDIDATE_MIN_BEATS:
            eligible = A >= thr
            total = eligible.sum(axis=1)
            even = eligible[:, ::2].sum(axis=1)
            odd = eligible[:, 1::2].sum(axis=1)
            usable = (
                (total >= MIN_TOTAL_BLOCKS)
                & (odd >= MIN_ODD_BLOCKS)
                & (even >= MIN_EVEN_BLOCKS)
            )
            rows.append({
                "min_beats_per_block": thr,
                "analysable_patients": int(usable.sum()),
                "analysable_fraction": float(usable.mean()),
                "eligible_blocks_min": int(total.min()),
                "eligible_blocks_median": float(np.median(total)),
                "eligible_blocks_max": int(total.max()),
                "odd_blocks_min": int(odd.min()),
                "even_blocks_min": int(even.min()),
            })

    qualifying = [
        r["min_beats_per_block"]
        for r in rows
        if r["analysable_patients"] >= MIN_DEV_ANALYSABLE
    ]
    selected = max(qualifying) if qualifying else None

    decision = (
        "WFP_BLOCK_YIELD_PASS_FREEZE_THRESHOLD"
        if selected is not None and not failures
        else "WFP_BLOCK_YIELD_STOP_AND_MODIFY"
    )

    result: Dict[str, Any] = {
        "schema_version": 1,
        "decision": decision,
        "scientific_role": "effect_blind_engineering_yield_only",
        "source_cohort": "Development50",
        "validation1000_accessed": False,
        "scientific_effects_calculated": False,
        "population_pca_performed": False,
        "population_covariance_performed": False,
        "candidate_min_beats_per_block": list(CANDIDATE_MIN_BEATS),
        "patient_eligibility_rule": {
            "min_total_eligible_blocks": MIN_TOTAL_BLOCKS,
            "min_odd_eligible_blocks": MIN_ODD_BLOCKS,
            "min_even_eligible_blocks": MIN_EVEN_BLOCKS,
        },
        "threshold_selection_rule": (
            "largest candidate min_beats_per_block with >=45/50 Development50 "
            "patients satisfying the frozen patient eligibility rule"
        ),
        "selected_min_beats_per_block": selected,
        "yield_table": rows,
        "failures": failures,
        "authoritative_run1_sha256": observed_sha,
    }

    (out / "WFP_BLOCK_YIELD.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )

    lines = [
        "WF-P BLOCK-ELIGIBILITY EFFECT-BLIND YIELD AUDIT",
        "================================================",
        f"Decision: {decision}",
        "Source: Development50 only",
        "Validation1000 accessed: NO",
        "Scientific effects calculated: NO",
        "Population PCA performed: NO",
        "Population covariance performed: NO",
        "",
        "Frozen patient eligibility:",
        f"  total eligible blocks >= {MIN_TOTAL_BLOCKS}",
        f"  odd eligible blocks >= {MIN_ODD_BLOCKS}",
        f"  even eligible blocks >= {MIN_EVEN_BLOCKS}",
        "",
        "Candidate thresholds:",
    ]
    for r in rows:
        lines.append(
            f"  min beats/block={r['min_beats_per_block']}: "
            f"analysable={r['analysable_patients']}/50; "
            f"block count min/median/max="
            f"{r['eligible_blocks_min']}/{r['eligible_blocks_median']:.1f}/{r['eligible_blocks_max']}"
        )
    lines += [
        "",
        "Selection rule:",
        "  largest candidate threshold with >=45/50 analysable Development50 patients",
        f"Selected min beats/block: {selected if selected is not None else 'NONE'}",
        "",
        "Boundary: PASS authorizes freezing the Validation1000 discovery specification.",
    ]
    (out / "WFP_BLOCK_YIELD.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if decision.startswith("WFP_BLOCK_YIELD_PASS") else 2

if __name__ == "__main__":
    raise SystemExit(main())
