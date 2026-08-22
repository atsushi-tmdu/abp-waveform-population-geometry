#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 6A — residual morphology localization beyond the frozen d=8 space.

Scientific role:
Post-discovery descriptive characterization only.

This stage DOES NOT:
- change the frozen d=8 basis;
- select a new dimension;
- name a notch region in advance;
- access age/sex/diagnosis/treatment/outcome;
- define Zstate or Ztrait.

It asks:
1) How much patient-central morphology variance remains outside frozen B8?
2) How much of that residual is reproducible across odd/even temporal replicates?
3) Where in normalized beat phase does reproducible residual energy lie?
4) How much short-window within-person block variation remains outside B8, and where?

The primary localization object is the diagonal of the positive part of the
odd/even replicate residual cross-covariance after orthogonal projection away
from B8. No local physiological label is assigned.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SOURCE_FS = 125.0
P = 64
D = 8
BLOCK_SEC = 60
FULL_WINDOW_SEC = 1800
MIN_BEATS_PER_BLOCK = 32
MIN_TOTAL_BLOCKS = 6
MIN_ODD_BLOCKS = 3
MIN_EVEN_BLOCKS = 3
EXPECTED_SOURCE_N = 1000
EXPECTED_ANALYSABLE_N = 978
EXPECTED_EXCLUSIONS_N = 22
POS_TOL_REL = 1e-10
WINDOW_WIDTHS = (4, 8, 16)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
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

def construct_patient_blocks(path: Path, run1) -> Dict[str, Any]:
    pid, fs0, x125 = run1.load_case(path)
    if abs(float(fs0) - SOURCE_FS) > 1e-6:
        raise RuntimeError(f"{pid}: expected 125 Hz")
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
    shape = np.asarray(rep125["shape_norm"][acc], dtype=float)
    centers = catalog["center_sec"].to_numpy(float)[acc]

    q_list: List[np.ndarray] = []
    block_ids: List[int] = []

    for b in range(FULL_WINDOW_SEC // BLOCK_SEC):
        a = start + b * BLOCK_SEC
        z = a + BLOCK_SEC
        if b < (FULL_WINDOW_SEC // BLOCK_SEC) - 1:
            sel = (centers >= a) & (centers < z)
        else:
            sel = (centers >= a) & (centers <= z)
        Xb = shape[sel]
        if len(Xb) < MIN_BEATS_PER_BLOCK:
            continue
        q = normalize_shape(np.mean(Xb, axis=0))
        if q is None:
            continue
        q_list.append(q)
        block_ids.append(b)

    odd = [q for q,b in zip(q_list,block_ids) if b % 2 == 1]
    even = [q for q,b in zip(q_list,block_ids) if b % 2 == 0]

    if len(q_list) < MIN_TOTAL_BLOCKS or len(odd) < MIN_ODD_BLOCKS or len(even) < MIN_EVEN_BLOCKS:
        raise RuntimeError(
            f"{pid}: frozen eligibility failed "
            f"(total={len(q_list)}, odd={len(odd)}, even={len(even)})"
        )

    def rep(xs: List[np.ndarray]) -> np.ndarray:
        z = normalize_shape(np.mean(np.vstack(xs), axis=0))
        if z is None:
            raise RuntimeError(f"{pid}: representative normalization failed")
        return z

    return {
        "patient_id": pid,
        "blocks": np.vstack(q_list),
        "all_rep": rep(q_list),
        "odd_rep": rep(odd),
        "even_rep": rep(even),
        "eligible_blocks": len(q_list),
    }

def covariance(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    A = X - np.mean(X, axis=0, keepdims=True)
    return A.T @ A / (len(X)-1)

def sym_cross_cov(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    A = X - np.mean(X, axis=0, keepdims=True)
    B = Y - np.mean(Y, axis=0, keepdims=True)
    C = A.T @ B / (len(X)-1)
    return 0.5 * (C + C.T)

def positive_part(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5*(S+S.T))
    scale = max(1.0, float(np.max(np.abs(vals))))
    tol = POS_TOL_REL * scale
    pos = np.where(vals > tol, vals, 0.0)
    return vecs @ np.diag(pos) @ vecs.T, vals

def normalize_profile(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v = np.clip(v, 0.0, None)
    s = float(np.sum(v))
    if s <= 0:
        return np.full_like(v, np.nan)
    return v / s

def effective_support(profile: np.ndarray) -> float:
    p = np.asarray(profile, float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan
    return float(np.exp(-np.sum(p*np.log(p))))

def max_contiguous_window(profile: np.ndarray, width: int) -> Dict[str, Any]:
    p = np.asarray(profile, float)
    if len(p) != P or width < 1 or width > P:
        raise ValueError("invalid profile/window")
    best_sum = -np.inf
    best_start = None
    # Non-circular windows: beat phase endpoints are retained as boundaries.
    for s in range(0, P-width+1):
        val = float(np.sum(p[s:s+width]))
        if val > best_sum:
            best_sum = val
            best_start = s
    return {
        "width_points": width,
        "fraction": best_sum,
        "start_index": int(best_start),
        "end_index": int(best_start + width - 1),
        "start_phase": float(best_start / P),
        "end_phase": float((best_start + width - 1) / P),
    }

def self_test() -> int:
    rng = np.random.default_rng(20260820)
    A = rng.normal(size=(P,D))
    A -= np.mean(A,axis=0,keepdims=True)
    B,_ = np.linalg.qr(A)
    P8 = B@B.T
    Q = np.eye(P)-P8

    x = rng.normal(size=P)
    r = Q@x
    if np.max(np.abs(B.T@r)) > 1e-10:
        print("SELF-TEST FAIL: residual not orthogonal to B8", file=sys.stderr)
        return 1

    prof = normalize_profile(r*r)
    if not np.isclose(np.sum(prof),1.0):
        print("SELF-TEST FAIL: phase profile does not sum to 1", file=sys.stderr)
        return 1

    print("WFP residual localization self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=False, default="~/Documents/abp_information_study")
    ap.add_argument("--input", required=False, default="~/Documents/abp_information_study/data/abp125_validation1000")
    ap.add_argument("--discovery-results", required=False, default="~/Documents/abp_information_study/results/wfp_discovery_validation1000")
    ap.add_argument("--run1-script", required=False, default="~/Documents/abp_information_study/code/wp2_run1_development50.py")
    ap.add_argument("--spec", required=False, default="~/Documents/abp_information_study/freeze/wfp_residual_localization/WFP_RESIDUAL_LOCALIZATION_FROZEN_SPEC.json")
    ap.add_argument("--out", required=False, default="~/Documents/abp_information_study/results/wfp_residual_localization")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    project = Path(args.project_root).expanduser().resolve()
    input_dir = Path(args.input).expanduser().resolve()
    discovery_results = Path(args.discovery_results).expanduser().resolve()
    run1_path = Path(args.run1_script).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_RESIDUAL_LOCALIZATION":
        raise SystemExit("FAIL: residual localization spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("FAIL: residual localization script hash mismatch")
    if spec.get("run1_script_sha256") != sha256_file(run1_path):
        raise SystemExit("FAIL: Run-1 hash mismatch")

    coord_path = discovery_results / "WFP_DISCOVERY_COMMON_COORDINATES.npz"
    score_path = discovery_results / "wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    if spec.get("coordinates_sha256") != sha256_file(coord_path):
        raise SystemExit("FAIL: coordinate hash mismatch")
    if spec.get("patient_scores_sha256") != sha256_file(score_path):
        raise SystemExit("FAIL: patient-score hash mismatch")

    with np.load(coord_path, allow_pickle=False) as z:
        mu = np.asarray(z["population_mean"], float)
        B = np.asarray(z["between_basis"], float)
    if mu.shape != (P,) or B.shape != (P,D):
        raise SystemExit(f"FAIL: frozen coordinate shape mismatch: mu={mu.shape}, B={B.shape}")

    P8 = B @ B.T
    Q8 = np.eye(P) - P8

    frozen_scores = pd.read_csv(score_path)
    if len(frozen_scores) != EXPECTED_ANALYSABLE_N:
        raise SystemExit("FAIL: frozen score row count not 978")
    frozen_score_map = frozen_scores.set_index("patient_id")[[f"z{j}" for j in range(1,D+1)]]

    run1 = load_module(run1_path, "wfp_run1_residual")
    cases = sorted((input_dir / "cases").glob("*.npz"))
    if len(cases) != EXPECTED_SOURCE_N:
        raise SystemExit(f"FAIL: expected 1000 cases, found {len(cases)}")

    patients = []
    failures = []
    t0 = time.perf_counter()
    for k,p in enumerate(cases, start=1):
        try:
            patients.append(construct_patient_blocks(p, run1))
        except Exception as e:
            failures.append({"file": p.name, "error": repr(e)})
        if k % 50 == 0 or k == len(cases):
            elapsed = time.perf_counter() - t0
            rate = k / elapsed if elapsed > 0 else np.nan
            eta = (len(cases)-k)/rate if rate > 0 else np.nan
            print(
                f"[progress] {k}/{len(cases)} processed; "
                f"analysable_so_far={len(patients)}; failures_so_far={len(failures)}; "
                f"elapsed_min={elapsed/60:.1f}; eta_min={eta/60:.1f}",
                flush=True
            )

    if len(patients) != EXPECTED_ANALYSABLE_N or len(failures) != EXPECTED_EXCLUSIONS_N:
        raise SystemExit(
            "FAIL: frozen cohort identity changed: "
            f"analysable={len(patients)}, failures={len(failures)}"
        )

    pids = [x["patient_id"] for x in patients]
    if len(set(pids)) != len(pids):
        raise SystemExit("FAIL: duplicate patient IDs")

    M = np.vstack([x["all_rep"] for x in patients])
    O = np.vstack([x["odd_rep"] for x in patients])
    E = np.vstack([x["even_rep"] for x in patients])

    # Exact identity check against frozen Stage-3 scores.
    current_scores = (M - mu) @ B
    diffs = []
    for i,pid in enumerate(pids):
        if pid not in frozen_score_map.index:
            raise SystemExit(f"FAIL: patient {pid} absent from frozen score table")
        old = frozen_score_map.loc[pid].to_numpy(float)
        diffs.append(float(np.max(np.abs(current_scores[i]-old))))
    max_score_diff = float(np.max(diffs))
    if max_score_diff > 1e-10:
        raise SystemExit(f"FAIL: reconstructed scores differ from frozen scores; max={max_score_diff}")

    # Central morphology residual outside B8.
    X = M - mu
    R = X @ Q8
    central_total = float(np.sum(X*X))
    central_resid = float(np.sum(R*R))
    central_resid_fraction = central_resid/central_total if central_total>0 else np.nan
    central_cov_resid = covariance(R)
    central_profile = normalize_profile(np.diag(central_cov_resid))

    # Odd/even replicate residual geometry outside B8.
    RO = (O - np.mean(O,axis=0,keepdims=True)) @ Q8
    RE = (E - np.mean(E,axis=0,keepdims=True)) @ Q8
    Soo = covariance(RO)
    See = covariance(RE)
    Sauto = 0.5*(Soo+See)
    Scross = sym_cross_cov(RO,RE)
    Spos, cross_eigs = positive_part(Scross)

    raw_resid_trace = float(np.trace(Sauto))
    signed_repro_trace = float(np.trace(Scross))
    positive_repro_trace = float(np.trace(Spos))
    signed_reliability_ratio = signed_repro_trace/raw_resid_trace if raw_resid_trace>0 else np.nan
    positive_reliability_ratio = positive_repro_trace/raw_resid_trace if raw_resid_trace>0 else np.nan

    repro_profile = normalize_profile(np.diag(Spos))
    raw_replicate_profile = normalize_profile(np.diag(Sauto))

    # Equal-patient short-window within covariance, then residual outside B8.
    within_covs = []
    within_resid_covs = []
    for x in patients:
        Qb = np.asarray(x["blocks"], float)
        A = Qb - np.mean(Qb, axis=0, keepdims=True)
        if len(A) < 2:
            raise SystemExit("FAIL: patient has <2 blocks after frozen eligibility")
        C = A.T @ A / (len(A)-1)
        Ar = A @ Q8
        Cr = Ar.T @ Ar / (len(Ar)-1)
        within_covs.append(C)
        within_resid_covs.append(Cr)

    Sw = np.mean(np.stack(within_covs),axis=0)
    Swr = np.mean(np.stack(within_resid_covs),axis=0)
    within_resid_fraction = float(np.trace(Swr)/np.trace(Sw)) if np.trace(Sw)>0 else np.nan
    within_profile = normalize_profile(np.diag(Swr))

    phase = np.arange(P,dtype=float)/P
    phase_df = pd.DataFrame({
        "phase_index": np.arange(P,dtype=int),
        "phase": phase,
        "central_residual_energy_fraction": central_profile,
        "replicate_raw_residual_energy_fraction": raw_replicate_profile,
        "replicate_positive_reproducible_residual_energy_fraction": repro_profile,
        "within_block_residual_energy_fraction": within_profile,
    })
    phase_df.to_csv(out/"wfp_residual_phase_profile.csv",index=False)

    windows = {
        "central_residual": [max_contiguous_window(central_profile,w) for w in WINDOW_WIDTHS],
        "reproducible_residual": [max_contiguous_window(repro_profile,w) for w in WINDOW_WIDTHS],
        "within_block_residual": [max_contiguous_window(within_profile,w) for w in WINDOW_WIDTHS],
    }

    summary: Dict[str,Any] = {
        "schema_version":1,
        "work_package":"WF-P",
        "stage":"6A",
        "decision":"WFP_RESIDUAL_LOCALIZATION_COMPLETE",
        "scientific_role":"post_discovery_descriptive_residual_localization_only",
        "source_n":EXPECTED_SOURCE_N,
        "analysable_n":len(patients),
        "frozen_rule_exclusions_n":len(failures),
        "waveform_arrays_opened":True,
        "clinical_labels_accessed":False,
        "dimension_reselected":False,
        "basis_changed":False,
        "notch_region_prespecified":False,
        "central_morphology":{
            "residual_variance_fraction_outside_B8":central_resid_fraction,
            "phase_effective_support_points":effective_support(central_profile),
            "max_phase_index":int(np.nanargmax(central_profile)),
            "max_phase":float(phase[int(np.nanargmax(central_profile))]),
        },
        "odd_even_residual_reproducibility":{
            "raw_residual_variance_trace":raw_resid_trace,
            "signed_cross_trace":signed_repro_trace,
            "positive_cross_trace":positive_repro_trace,
            "signed_reproducibility_ratio":signed_reliability_ratio,
            "positive_reproducibility_ratio":positive_reliability_ratio,
            "positive_repro_phase_effective_support_points":effective_support(repro_profile),
            "max_phase_index":int(np.nanargmax(repro_profile)),
            "max_phase":float(phase[int(np.nanargmax(repro_profile))]),
            "negative_eigenvalue_mass_fraction":
                float(np.sum(np.abs(cross_eigs[cross_eigs<0])) / np.sum(np.abs(cross_eigs)))
                if np.sum(np.abs(cross_eigs))>0 else np.nan,
        },
        "within_block":{
            "residual_variance_fraction_outside_B8":within_resid_fraction,
            "phase_effective_support_points":effective_support(within_profile),
            "max_phase_index":int(np.nanargmax(within_profile)),
            "max_phase":float(phase[int(np.nanargmax(within_profile))]),
        },
        "max_contiguous_window_energy":windows,
        "identity_checks":{
            "max_abs_score_difference_vs_frozen_stage3":max_score_diff,
        },
        "boundary":[
            "No phase interval is named as dicrotic notch in this stage.",
            "Localization is descriptive and data-derived within the already-open discovery cohort.",
            "No clinical labels were accessed.",
            "The frozen d=8 basis is unchanged.",
            "A notch-specific module is justified only if a reproducible localized residual pattern is scientifically convincing after this audit."
        ],
        "input_hashes":{
            "frozen_spec_sha256":sha256_file(spec_path),
            "run1_script_sha256":sha256_file(run1_path),
            "coordinates_sha256":sha256_file(coord_path),
            "patient_scores_sha256":sha256_file(score_path),
        },
        "failures":failures,
    }

    (out/"WFP_RESIDUAL_LOCALIZATION.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",
        encoding="utf-8"
    )

    eig_df = pd.DataFrame({
        "eigen_index":np.arange(1,P+1),
        "residual_cross_eigenvalue_desc":np.sort(cross_eigs)[::-1],
    })
    eig_df.to_csv(out/"wfp_residual_replicate_spectrum.csv",index=False)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(phase, central_profile, label="Central morphology residual")
    ax.plot(phase, repro_profile, label="Reproducible odd/even residual")
    ax.plot(phase, within_profile, label="Within-block residual")
    ax.set_xlabel("Normalized beat phase")
    ax.set_ylabel("Normalized residual energy")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out/"WFP_FIGURE_RESIDUAL_PHASE_LOCALIZATION.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"WFP_FIGURE_RESIDUAL_PHASE_LOCALIZATION.pdf",bbox_inches="tight")
    plt.close(fig)

    lines=[
        "WF-P RESIDUAL MORPHOLOGY LOCALIZATION",
        "=====================================",
        "Decision: WFP_RESIDUAL_LOCALIZATION_COMPLETE",
        "Scientific role: POST-DISCOVERY DESCRIPTIVE RESIDUAL LOCALIZATION ONLY",
        f"Source n: {EXPECTED_SOURCE_N}",
        f"Analysable n: {len(patients)}",
        f"Frozen-rule exclusions: {len(failures)}",
        "Clinical labels accessed: NO",
        "Dimension reselected: NO",
        "Primary basis changed: NO",
        "Notch region prespecified: NO",
        "",
        "Central morphology residual outside frozen B8:",
        f"  residual variance fraction: {central_resid_fraction}",
        f"  phase effective support (of 64 points): {effective_support(central_profile)}",
        f"  maximum residual-energy phase: {summary['central_morphology']['max_phase']}",
        "",
        "Odd/even reproducible residual outside frozen B8:",
        f"  signed reproducibility ratio: {signed_reliability_ratio}",
        f"  positive-part reproducibility ratio: {positive_reliability_ratio}",
        f"  positive-repro phase effective support (of 64 points): {effective_support(repro_profile)}",
        f"  maximum reproducible residual-energy phase: {summary['odd_even_residual_reproducibility']['max_phase']}",
        f"  negative eigenvalue mass fraction: {summary['odd_even_residual_reproducibility']['negative_eigenvalue_mass_fraction']}",
        "",
        "Within-block morphology residual outside frozen B8:",
        f"  residual variance fraction: {within_resid_fraction}",
        f"  phase effective support (of 64 points): {effective_support(within_profile)}",
        f"  maximum residual-energy phase: {summary['within_block']['max_phase']}",
        "",
        "Max contiguous-window energy fractions:",
    ]
    for name, arr in windows.items():
        lines.append(f"  {name}:")
        for w in arr:
            lines.append(
                f"    width={w['width_points']} points: fraction={w['fraction']}; "
                f"phase={w['start_phase']}..{w['end_phase']}"
            )
    lines += [
        "",
        f"Identity check max |score difference| vs frozen Stage 3: {max_score_diff}",
        "",
        "Boundary:",
        "  Do NOT call any phase interval 'dicrotic notch' yet.",
        "  This stage only identifies whether reproducible residual morphology is localized.",
        "  A notch-specific analysis, if warranted, requires a separate frozen specification.",
    ]
    (out/"WFP_RESIDUAL_LOCALIZATION.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
