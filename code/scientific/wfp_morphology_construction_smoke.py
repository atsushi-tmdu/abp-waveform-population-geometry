#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 2 — morphology-construction engineering smoke (Development50 only)

Purpose
-------
Verify implementation invariants for the WF-P morphology representation without
opening Validation1000 and without calculating population scientific effects.

This script:
- uses Development50 only;
- reuses the authoritative WF1/WF2 beat construction and locked 30-min QC;
- constructs 60-s block central morphology;
- separates central shape direction from block coherence;
- constructs patient-level all/odd/even representatives;
- checks deterministic invariants and reports counts/ranges only;
- does NOT perform population PCA, eigendecomposition, effective rank,
  between-person covariance analysis, age/sex analysis, or scientific hypothesis tests.

Expected project layout
-----------------------
~/Documents/abp_information_study/
  code/wp2_run1_development50.py
  data/abp125_pilot50/cases/*.npz
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

SOURCE_FS = 125.0
P = 64
BLOCK_SEC = 60
FULL_WINDOW_SEC = 1800
EXPECTED_DEV_N = 50

FORBIDDEN_SCIENCE_TERMS = [
    "population_pca",
    "population_eigenvalue",
    "effective_rank",
    "d90",
    "d95",
    "between_covariance",
    "principal_angle",
    "age_association",
    "sex_association",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def load_run1(path: Path):
    spec = importlib.util.spec_from_file_location("wfp_run1_authoritative", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import authoritative Run-1 script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def normalize_shape(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or len(v) != P or not np.all(np.isfinite(v)):
        return None
    c = v - np.mean(v)
    sd = float(np.std(c))
    if sd <= 1e-12:
        return None
    return c / sd

def block_central_shape(X: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    X: accepted shape_norm beats, rows=beats, cols=phase.
    Returns:
      normalized central morphology direction q (mean0, sd1)
      coherence kappa = phase-SD of the unrenormalized mean shape
    Since each beat has phase-SD 1, kappa is a concentration/coherence measure.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != P or X.shape[0] == 0:
        return None, np.nan
    m = np.mean(X, axis=0)
    kappa = float(np.std(m - np.mean(m)))
    q = normalize_shape(m)
    return q, kappa

def patient_rep(block_shapes: List[np.ndarray]) -> np.ndarray | None:
    if not block_shapes:
        return None
    M = np.vstack(block_shapes)
    return normalize_shape(np.mean(M, axis=0))

def process_case(path: Path, run1) -> Dict[str, Any]:
    pid, fs0, x125 = run1.load_case(path)
    if abs(float(fs0) - SOURCE_FS) > 1e-6:
        raise RuntimeError(f"{pid}: expected 125 Hz, got {fs0}")

    total_sec = len(x125) / SOURCE_FS
    if total_sec + 1e-9 < FULL_WINDOW_SEC:
        raise RuntimeError(f"{pid}: signal shorter than 30 min")

    peaks = run1.detect_systolic_peaks_125(x125)
    built = run1.build_125_catalog(x125, peaks)
    if built is None:
        raise RuntimeError(f"{pid}: could not build native beat catalog")
    catalog, rep125 = built

    # Match authoritative WP2 Run-1: use the final retained 30-min window.
    start = total_sec - FULL_WINDOW_SEC
    end = total_sec
    qc = run1.locked_qc_for_window(catalog, rep125, start, end)
    if qc is None:
        raise RuntimeError(f"{pid}: authoritative 30-min locked QC failed")

    accepted_idx = np.asarray(qc["accepted_idx"], dtype=int)
    shape = np.asarray(rep125["shape_norm"][accepted_idx], dtype=float)
    centers = catalog["center_sec"].to_numpy(float)[accepted_idx]

    block_shapes = []
    block_kappa = []
    block_ids = []
    block_beats = []

    for b in range(FULL_WINDOW_SEC // BLOCK_SEC):
        a = start + b * BLOCK_SEC
        z = a + BLOCK_SEC
        # Half-open blocks except final block; prevents duplicated boundary beats.
        if b < (FULL_WINDOW_SEC // BLOCK_SEC) - 1:
            sel = (centers >= a) & (centers < z)
        else:
            sel = (centers >= a) & (centers <= z)

        Xb = shape[sel]
        if len(Xb) == 0:
            continue

        q, kappa = block_central_shape(Xb)
        if q is None or not np.isfinite(kappa):
            continue

        block_shapes.append(q)
        block_kappa.append(kappa)
        block_ids.append(b)
        block_beats.append(int(len(Xb)))

    all_rep = patient_rep(block_shapes)
    odd_shapes = [q for q, b in zip(block_shapes, block_ids) if b % 2 == 1]
    even_shapes = [q for q, b in zip(block_shapes, block_ids) if b % 2 == 0]
    odd_rep = patient_rep(odd_shapes)
    even_rep = patient_rep(even_shapes)

    if all_rep is None or odd_rep is None or even_rep is None:
        raise RuntimeError(f"{pid}: failed to construct all/odd/even patient representatives")

    # Engineering-only replicate correlation: patient-level value is not written.
    # Only finite/nonfinite status contributes to the summary gate.
    den = np.linalg.norm(odd_rep - odd_rep.mean()) * np.linalg.norm(even_rep - even_rep.mean())
    replicate_corr = float(np.dot(odd_rep - odd_rep.mean(), even_rep - even_rep.mean()) / den) if den > 0 else np.nan

    # Invariants.
    max_abs_mean = max(
        abs(float(np.mean(all_rep))),
        abs(float(np.mean(odd_rep))),
        abs(float(np.mean(even_rep))),
        max(abs(float(np.mean(q))) for q in block_shapes),
    )
    max_abs_sd_err = max(
        abs(float(np.std(all_rep)) - 1.0),
        abs(float(np.std(odd_rep)) - 1.0),
        abs(float(np.std(even_rep)) - 1.0),
        max(abs(float(np.std(q)) - 1.0) for q in block_shapes),
    )

    return {
        "patient_id": pid,
        "accepted_beats": int(len(accepted_idx)),
        "eligible_blocks": int(len(block_shapes)),
        "odd_blocks": int(len(odd_shapes)),
        "even_blocks": int(len(even_shapes)),
        "min_beats_per_block": int(min(block_beats)),
        "max_beats_per_block": int(max(block_beats)),
        "median_beats_per_block": float(np.median(block_beats)),
        "min_block_coherence": float(np.min(block_kappa)),
        "max_block_coherence": float(np.max(block_kappa)),
        "median_block_coherence": float(np.median(block_kappa)),
        "replicate_corr_finite": bool(np.isfinite(replicate_corr)),
        "max_abs_shape_mean": max_abs_mean,
        "max_abs_shape_sd_error": max_abs_sd_err,
    }

def self_test() -> None:
    rng = np.random.default_rng(20260820)
    base = normalize_shape(np.sin(np.linspace(0, 2*np.pi, P, endpoint=False)))
    assert base is not None
    X = []
    for _ in range(100):
        noise = rng.normal(scale=0.05, size=P)
        x = normalize_shape(base + noise)
        assert x is not None
        X.append(x)
    X = np.vstack(X)
    q, k = block_central_shape(X)
    assert q is not None and np.isfinite(k)
    assert abs(np.mean(q)) < 1e-12
    assert abs(np.std(q) - 1.0) < 1e-12
    assert 0 < k <= 1.01

    reps = [q, normalize_shape(q + rng.normal(scale=.02, size=P))]
    assert reps[1] is not None
    p = patient_rep(reps)
    assert p is not None
    assert abs(np.mean(p)) < 1e-12
    assert abs(np.std(p) - 1.0) < 1e-12
    print("WFP morphology smoke self-test: PASS")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_morphology_smoke")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = Path(args.project_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    run1_path = root / "code" / "wp2_run1_development50.py"
    source_dir = root / "data" / "abp125_pilot50" / "cases"
    val_dir = root / "data" / "abp125_validation1000"

    if not run1_path.is_file():
        raise SystemExit(f"FAIL: authoritative Run-1 script missing: {run1_path}")
    if not source_dir.is_dir():
        raise SystemExit(f"FAIL: Development50 cases directory missing: {source_dir}")

    # Guard: this script never reads Validation1000.
    validation_accessed = False

    run1 = load_run1(run1_path)
    cases = sorted(source_dir.glob("*.npz"))
    if len(cases) != EXPECTED_DEV_N:
        raise SystemExit(f"FAIL: expected {EXPECTED_DEV_N} Development50 NPZs, found {len(cases)}")

    rows = []
    failures = []
    for path in cases:
        try:
            rows.append(process_case(path, run1))
        except Exception as e:
            failures.append({"file": path.name, "error": repr(e)})

    # No patient-level scientific readout is written.
    df = pd.DataFrame(rows)

    gate_checks = {
        "development_case_count_50": len(cases) == 50,
        "successful_patients_50": len(rows) == 50,
        "patient_failures_zero": len(failures) == 0,
        "validation1000_accessed_false": validation_accessed is False,
        "all_have_odd_even_blocks": bool(len(df) == 50 and (df["odd_blocks"] > 0).all() and (df["even_blocks"] > 0).all()),
        "all_replicate_corr_finite": bool(len(df) == 50 and df["replicate_corr_finite"].all()),
        "normalization_mean_invariant": bool(len(df) == 50 and df["max_abs_shape_mean"].max() <= 1e-10),
        "normalization_sd_invariant": bool(len(df) == 50 and df["max_abs_shape_sd_error"].max() <= 1e-10),
        "all_coherence_finite_positive": bool(
            len(df) == 50
            and np.isfinite(df["min_block_coherence"]).all()
            and (df["min_block_coherence"] > 0).all()
        ),
    }

    passed = all(gate_checks.values())

    summary = {
        "schema_version": 1,
        "decision": "WFP_MORPHOLOGY_SMOKE_PASS" if passed else "WFP_MORPHOLOGY_SMOKE_FAIL",
        "scientific_role": "engineering_only",
        "source_cohort": "Development50",
        "validation1000_accessed": False,
        "population_pca_performed": False,
        "population_covariance_performed": False,
        "scientific_effect_analysis_performed": False,
        "authoritative_run1_sha256": sha256_file(run1_path),
        "counts": {
            "input_cases": len(cases),
            "successful_patients": len(rows),
            "failed_patients": len(failures),
        },
        "implementation_ranges_only": {
            "eligible_blocks_min": int(df["eligible_blocks"].min()) if len(df) else None,
            "eligible_blocks_max": int(df["eligible_blocks"].max()) if len(df) else None,
            "accepted_beats_min": int(df["accepted_beats"].min()) if len(df) else None,
            "accepted_beats_max": int(df["accepted_beats"].max()) if len(df) else None,
            "coherence_min_over_patients": float(df["min_block_coherence"].min()) if len(df) else None,
            "coherence_max_over_patients": float(df["max_block_coherence"].max()) if len(df) else None,
            "normalization_max_abs_mean": float(df["max_abs_shape_mean"].max()) if len(df) else None,
            "normalization_max_abs_sd_error": float(df["max_abs_shape_sd_error"].max()) if len(df) else None,
        },
        "gate_checks": gate_checks,
        "failures": failures,
        "forbidden_scientific_outputs": FORBIDDEN_SCIENCE_TERMS,
    }

    atomic_json(out / "WFP_MORPHOLOGY_SMOKE.json", summary)

    lines = [
        "WF-P MORPHOLOGY-CONSTRUCTION ENGINEERING SMOKE",
        "=============================================",
        f"Decision: {summary['decision']}",
        "Scientific role: ENGINEERING_ONLY",
        f"Development50 input cases: {len(cases)}",
        f"Successful patients: {len(rows)}",
        f"Failed patients: {len(failures)}",
        "Validation1000 accessed: NO",
        "Population PCA performed: NO",
        "Population covariance performed: NO",
        "Scientific effect analysis performed: NO",
        "",
        "Implementation ranges only:",
    ]
    for k, v in summary["implementation_ranges_only"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Gate checks:"]
    for k, v in gate_checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
    lines += [
        "",
        "Authorization boundary:",
        "PASS authorizes preparation/freezing of the Validation1000 WF-P discovery analysis specification.",
        "It does NOT authorize independent confirmatory validation.",
        "",
    ]
    atomic_text(out / "WFP_MORPHOLOGY_SMOKE.txt", "\n".join(lines))

    print("\n".join(lines))
    return 0 if passed else 2

if __name__ == "__main__":
    raise SystemExit(main())
