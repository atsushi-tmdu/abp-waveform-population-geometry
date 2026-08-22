#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WP2 Run 1 — development50 feasibility analysis
===============================================

Purpose
-------
Minimal development-cohort feasibility run for the ABP Waveform Research
Program, Work Package 2 (cross-phase covariance dependence).

This script deliberately reuses the scientific definitions from Paper 1:
  * native 125-Hz peak detection / midpoint beat boundaries
  * beat-level QC at 125 Hz only
  * accepted beat identities locked
  * phase-normalization to 64 points with endpoint=False
  * three representations: absolute, pulse_centered, shape_norm

Run-1 scope is intentionally narrow:
  p = 64, full retained 30-min window, four contiguous equal-width blocks,
  all accepted beats, and two surrogate controls.

Inputs
------
An extracted Paper-1 development cohort directory containing:
    cases/*.npz
Each case NPZ must contain at least: ABP, patient_id, fs.

Outputs
-------
  wp2_run1_patient_results.csv
  wp2_run1_cohort_summary.csv
  WP2_RUN1_INTEGRITY.json
  WP2_RUN1_INTEGRITY.txt
  WP2_RUN1_GATE.txt

The script does not read validation1000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

SOURCE_FS = 125
BEAT_POINTS = 64
FULL_WINDOW_SEC = 1800
REFERENCE_BLOCKS = 4
REPRESENTATIONS = ("shape_norm", "pulse_centered", "absolute")
DEFAULT_SEED = 20260816
DEFAULT_SURROGATES = 200
DEFAULT_BOOTSTRAP = 2000

# Numerical tolerances are for implementation checks, not scientific thresholds.
PSD_ABS_TOL = 1e-10
ORDER_TOL = 5e-10
ERANK_EQ_TOL = 5e-10


def interp_finite(x: np.ndarray) -> np.ndarray | None:
    """Paper-1-compatible finite-value interpolation."""
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if not np.any(good):
        return None
    if not np.all(good):
        idx = np.arange(len(x))
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else np.nan


def med_mad(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)


def detect_systolic_peaks_125(x: np.ndarray) -> np.ndarray:
    """Peak detection exactly on native 125-Hz ABP, matching Paper 1."""
    fs = SOURCE_FS
    x = interp_finite(x)
    if x is None:
        return np.array([], dtype=int)

    z = x - np.mean(x)
    if np.std(z) <= 1e-10:
        return np.array([], dtype=int)
    z = z / np.std(z)

    nyq = fs / 2.0
    lo = 0.5 / nyq
    hi_hz = min(5.0, 0.90 * nyq)
    hi = hi_hz / nyq
    if not (0 < lo < hi < 1):
        return np.array([], dtype=int)

    b, a = butter(2, [lo, hi], btype="band")
    try:
        z = filtfilt(b, a, z)
    except Exception:
        return np.array([], dtype=int)

    if np.std(z) <= 1e-10:
        return np.array([], dtype=int)
    z = z / np.std(z)

    peaks, _ = find_peaks(
        z,
        distance=max(1, int(0.30 * fs)),
        prominence=0.30,
    )
    return peaks


def sample_interval_to_64(
    x: np.ndarray, fs: float, left_sec: float, right_sec: float
) -> np.ndarray | None:
    """Paper-1 phase normalization: 64 equally spaced points, endpoint=False."""
    if not np.isfinite(left_sec) or not np.isfinite(right_sec):
        return None
    if right_sec <= left_sec:
        return None

    tq = np.linspace(left_sec, right_sec, BEAT_POINTS, endpoint=False)
    last_t = (len(x) - 1) / float(fs)
    if tq[0] < 0 or tq[-1] > last_t:
        return None

    pos = tq * float(fs)
    idx = np.arange(len(x), dtype=float)
    return np.interp(pos, idx, np.asarray(x, dtype=float))


def make_representations(B: np.ndarray) -> Dict[str, np.ndarray] | None:
    """Paper-1 absolute, pulse-centered, and shape-normalized beat matrices."""
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] == 0:
        return None
    if not np.all(np.isfinite(B)):
        return None

    meanp = np.mean(B, axis=1)
    centered = B - meanp[:, None]
    sd = np.std(centered, axis=1)
    if np.any(sd <= 1e-10):
        return None

    sbp = np.max(B, axis=1)
    dbp = np.min(B, axis=1)
    pp = sbp - dbp
    if np.any(pp <= 1e-8):
        return None

    return {
        "absolute": B,
        "pulse_centered": centered,
        "shape_norm": centered / sd[:, None],
        "pp": pp,
        "map_proxy": meanp,
        "sbp": sbp,
        "dbp": dbp,
    }


def build_125_catalog(x125: np.ndarray, peaks: np.ndarray):
    """Define midpoint-to-midpoint beat intervals once at native 125 Hz."""
    if len(peaks) < 4:
        return None

    bounds = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
    rows: List[dict] = []
    beats: List[np.ndarray] = []

    for k in range(1, len(peaks) - 1):
        left = int(bounds[k - 1])
        right = int(bounds[k])
        if right <= left:
            continue

        dur = (right - left) / float(SOURCE_FS)
        if dur < 0.30 or dur > 2.00:
            continue

        left_sec = left / float(SOURCE_FS)
        right_sec = right / float(SOURCE_FS)
        b = sample_interval_to_64(x125, SOURCE_FS, left_sec, right_sec)
        if b is None or not np.all(np.isfinite(b)):
            continue

        pp = float(np.max(b) - np.min(b))
        sd = float(np.std(b - np.mean(b)))
        if pp <= 1e-8 or sd <= 1e-10:
            continue

        beat_id = len(rows)
        rows.append(
            {
                "beat_id": beat_id,
                "left_sec": left_sec,
                "right_sec": right_sec,
                "center_sec": float(peaks[k]) / float(SOURCE_FS),
                "duration_sec": dur,
            }
        )
        beats.append(b)

    if len(rows) < 8:
        return None

    B = np.vstack(beats)
    rep = make_representations(B)
    if rep is None:
        return None

    return pd.DataFrame(rows), rep


def locked_qc_for_window(
    catalog: pd.DataFrame,
    rep125: Dict[str, np.ndarray],
    start: float,
    end: float,
):
    """Paper-1 QC: native 125-Hz shape template + duration/log-PP MAD rules."""
    sel = (
        (catalog["center_sec"].values >= start)
        & (catalog["center_sec"].values <= end)
    )
    idx = np.flatnonzero(sel)
    if len(idx) < 8:
        return None

    shape = rep125["shape_norm"][idx]
    dur = catalog["duration_sec"].values[idx]
    pp = rep125["pp"][idx]

    template0 = np.median(shape, axis=0)
    corrs = np.asarray([safe_corr(b, template0) for b in shape])
    keep = np.isfinite(corrs) & (corrs >= 0.75)

    dmed, dmad = med_mad(dur)
    if dmad > 1e-12:
        keep &= np.abs(dur - dmed) <= 5.0 * 1.4826 * dmad

    logpp = np.log(pp)
    pmed, pmad = med_mad(logpp)
    if pmad > 1e-12:
        keep &= np.abs(logpp - pmed) <= 5.0 * 1.4826 * pmad

    accepted_idx = idx[keep]
    if len(accepted_idx) < 6:
        return None

    return {
        "selected_idx": idx,
        "accepted_idx": accepted_idx,
        "n_before": int(len(idx)),
        "n_after": int(len(accepted_idx)),
    }


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("Need at least two beats in a 2-D matrix")
    Xc = X - np.mean(X, axis=0, keepdims=True)
    C = (Xc.T @ Xc) / float(Xc.shape[0] - 1)
    C = 0.5 * (C + C.T)
    return C


def normalized_psd(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    eig = np.linalg.eigvalsh(C)
    scale = max(1.0, float(np.max(np.abs(eig))))
    min_eig = float(np.min(eig))
    if min_eig < -PSD_ABS_TOL * scale:
        raise ValueError(f"Substantively negative covariance eigenvalue: {min_eig:.3e}")
    eig = np.where(eig < 0.0, 0.0, eig)
    tr = float(np.sum(eig))
    if not np.isfinite(tr) or tr <= 0.0:
        raise ValueError("Covariance trace must be positive")
    rho = C / tr
    lam = eig / tr
    return rho, lam, min_eig


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def matrix_entropy(C: np.ndarray) -> Tuple[float, float]:
    _, lam, min_eig = normalized_psd(C)
    return entropy_from_probs(lam), min_eig


def effective_rank_cov(C: np.ndarray) -> float:
    S, _ = matrix_entropy(C)
    return float(np.exp(S))


def effective_rank_svd(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = s**2
    if np.sum(eig) <= 0:
        raise ValueError("SVD variance is zero")
    p = eig / np.sum(eig)
    return float(np.exp(entropy_from_probs(p)))


def block_slices(p: int, k: int) -> List[slice]:
    if p % k != 0:
        raise ValueError(f"p={p} must be divisible by k={k}")
    w = p // k
    return [slice(j * w, (j + 1) * w) for j in range(k)]


def pinch_covariance(C: np.ndarray, k: int) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    p = C.shape[0]
    out = np.zeros_like(C)
    for sl in block_slices(p, k):
        out[sl, sl] = C[sl, sl]
    return out


def dephase_covariance(C: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(np.asarray(C, dtype=np.float64)))


def delta_metrics(X: np.ndarray) -> dict:
    C = covariance_matrix(X)
    S_full, min_eig = matrix_entropy(C)
    erank_cov = float(np.exp(S_full))
    erank_svd = effective_rank_svd(X)
    erank_abs_diff = abs(erank_cov - erank_svd)
    if erank_abs_diff > ERANK_EQ_TOL * max(1.0, erank_svd):
        raise RuntimeError(
            f"Covariance/SVD effective-rank mismatch: {erank_cov} vs {erank_svd}"
        )

    deltas = {}
    for k in (2, 4, 8):
        S_pinched, _ = matrix_entropy(pinch_covariance(C, k))
        deltas[f"delta{k}"] = float(S_pinched - S_full)

    S_diag, _ = matrix_entropy(dephase_covariance(C))
    deltas["delta_total"] = float(S_diag - S_full)

    vals = [0.0, deltas["delta2"], deltas["delta4"], deltas["delta8"], deltas["delta_total"]]
    if any(vals[j + 1] + ORDER_TOL < vals[j] for j in range(len(vals) - 1)):
        raise RuntimeError(f"Pinching entropy ordering failed: {vals}")
    if vals[1] < -ORDER_TOL:
        raise RuntimeError(f"Negative delta beyond tolerance: {vals}")

    return {
        "effective_rank": erank_cov,
        "effective_rank_svd": erank_svd,
        "effective_rank_abs_diff": erank_abs_diff,
        "min_cov_eigenvalue": min_eig,
        **deltas,
    }


def patient_seed(global_seed: int, patient_id: str, stream: str) -> int:
    text = f"{global_seed}|{patient_id}|{stream}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def surrogate_phase_independent(B_abs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate A: independently permute beat labels within each phase coordinate."""
    B = np.asarray(B_abs, dtype=np.float64)
    n, p = B.shape
    out = np.empty_like(B)
    for j in range(p):
        out[:, j] = B[rng.permutation(n), j]
    return out


def surrogate_block_reassigned(
    B_abs: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Surrogate B: preserve each contiguous block intact, reassign blocks across beats."""
    B = np.asarray(B_abs, dtype=np.float64)
    n, p = B.shape
    out = np.empty_like(B)
    for sl in block_slices(p, k):
        out[:, sl] = B[rng.permutation(n), sl]
    return out


def surrogate_distributions(
    B_abs: np.ndarray,
    patient_id: str,
    n_surrogates: int,
    global_seed: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], dict]:
    """Generate both Run-1 surrogate distributions from accepted absolute beats."""
    out_a = {rep: [] for rep in REPRESENTATIONS}
    out_b = {rep: [] for rep in REPRESENTATIONS}

    rng_a = np.random.default_rng(patient_seed(global_seed, patient_id, "surrogate_A"))
    rng_b = np.random.default_rng(patient_seed(global_seed, patient_id, "surrogate_B4"))

    preservation_checks = {
        "surrogate_A_marginals_max_abs_error": 0.0,
        "surrogate_B_within_block_cov_max_abs_error": 0.0,
    }

    C_orig = covariance_matrix(B_abs)
    blocks = block_slices(B_abs.shape[1], REFERENCE_BLOCKS)

    for b in range(n_surrogates):
        A = surrogate_phase_independent(B_abs, rng_a)
        B = surrogate_block_reassigned(B_abs, REFERENCE_BLOCKS, rng_b)

        if b == 0:
            # Exact empirical-marginal preservation for Surrogate A.
            err_a = 0.0
            for j in range(B_abs.shape[1]):
                err_a = max(
                    err_a,
                    float(np.max(np.abs(np.sort(A[:, j]) - np.sort(B_abs[:, j])))),
                )
            preservation_checks["surrogate_A_marginals_max_abs_error"] = err_a

            # Exact within-block covariance preservation for Surrogate B BEFORE
            # representation-specific row transforms.
            Cb = covariance_matrix(B)
            err_b = 0.0
            for sl in blocks:
                err_b = max(
                    err_b,
                    float(np.max(np.abs(Cb[sl, sl] - C_orig[sl, sl]))),
                )
            preservation_checks["surrogate_B_within_block_cov_max_abs_error"] = err_b

        rep_a = make_representations(A)
        rep_b = make_representations(B)
        if rep_a is None or rep_b is None:
            raise RuntimeError(f"{patient_id}: surrogate representation construction failed")

        for rep in REPRESENTATIONS:
            out_a[rep].append(delta_metrics(rep_a[rep])["delta4"])
            out_b[rep].append(delta_metrics(rep_b[rep])["delta4"])

    return out_a, out_b, preservation_checks


def quantiles(a: Iterable[float]) -> Tuple[float, float, float]:
    x = np.asarray(list(a), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    q = np.percentile(x, [2.5, 50.0, 97.5])
    return float(q[0]), float(q[1]), float(q[2])


def bootstrap_median_ci(
    x: np.ndarray, n_boot: int, seed: int
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=float)
    n = len(x)
    for b in range(n_boot):
        meds[b] = np.median(x[rng.integers(0, n, size=n)])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def load_case(path: Path):
    d = np.load(str(path), allow_pickle=False)
    required = {"ABP", "patient_id", "fs"}
    missing = required.difference(d.files)
    if missing:
        raise RuntimeError(f"{path.name}: missing NPZ keys {sorted(missing)}")
    pid_raw = d["patient_id"]
    try:
        pid = str(pid_raw.item())
    except Exception:
        pid = str(pid_raw)
    fs0 = float(np.asarray(d["fs"]).item())
    x = interp_finite(np.asarray(d["ABP"], dtype=float))
    if x is None:
        raise RuntimeError(f"{path.name}: ABP has no finite values")
    return pid, fs0, x


def process_patient(
    path: Path,
    n_surrogates: int,
    global_seed: int,
) -> Tuple[List[dict], dict]:
    pid, fs0, x125 = load_case(path)
    if abs(fs0 - SOURCE_FS) > 1e-6:
        raise RuntimeError(f"{pid}: expected 125 Hz, got {fs0}")

    total_sec = len(x125) / float(SOURCE_FS)
    if total_sec + 1e-9 < FULL_WINDOW_SEC:
        raise RuntimeError(f"{pid}: signal shorter than 30 min ({total_sec:.3f} s)")

    peaks = detect_systolic_peaks_125(x125)
    built = build_125_catalog(x125, peaks)
    if built is None:
        raise RuntimeError(f"{pid}: could not build native beat catalog")
    catalog, rep125 = built

    qc = locked_qc_for_window(
        catalog,
        rep125,
        total_sec - FULL_WINDOW_SEC,
        total_sec,
    )
    if qc is None:
        raise RuntimeError(f"{pid}: 30-min locked QC failed")

    accepted_idx = qc["accepted_idx"]
    B_abs = np.asarray(rep125["absolute"][accepted_idx], dtype=float)
    reps_obs = make_representations(B_abs)
    if reps_obs is None:
        raise RuntimeError(f"{pid}: observed representation construction failed")

    sur_a, sur_b, preservation = surrogate_distributions(
        B_abs,
        pid,
        n_surrogates=n_surrogates,
        global_seed=global_seed,
    )

    rows: List[dict] = []
    max_erank_diff = 0.0
    min_eig = np.inf

    for rep in REPRESENTATIONS:
        obs = delta_metrics(reps_obs[rep])
        max_erank_diff = max(max_erank_diff, obs["effective_rank_abs_diff"])
        min_eig = min(min_eig, obs["min_cov_eigenvalue"])

        a_lo, a_med, a_hi = quantiles(sur_a[rep])
        b_lo, b_med, b_hi = quantiles(sur_b[rep])
        e4_a = float(obs["delta4"] - a_med)
        e4_b = float(obs["delta4"] - b_med)

        rows.append(
            {
                "patient_id": pid,
                "representation": rep,
                "n_beats_before_qc": qc["n_before"],
                "n_beats_after_qc": qc["n_after"],
                "effective_rank": obs["effective_rank"],
                "delta2_obs": obs["delta2"],
                "delta4_obs": obs["delta4"],
                "delta8_obs": obs["delta8"],
                "delta_total_obs": obs["delta_total"],
                "surrogate_A_delta4_q025": a_lo,
                "surrogate_A_delta4_median": a_med,
                "surrogate_A_delta4_q975": a_hi,
                "surrogate_B_delta4_q025": b_lo,
                "surrogate_B_delta4_median": b_med,
                "surrogate_B_delta4_q975": b_hi,
                "E4_preprocessing_excess": e4_a,
                "E4_local_surrogate_excess": e4_b,
            }
        )

    audit = {
        "patient_id": pid,
        "n_peaks": int(len(peaks)),
        "n_catalog_beats": int(len(catalog)),
        "n_beats_before_qc": int(qc["n_before"]),
        "n_beats_after_qc": int(qc["n_after"]),
        "max_effective_rank_cov_svd_abs_diff": float(max_erank_diff),
        "min_observed_cov_eigenvalue": float(min_eig),
        **preservation,
    }
    return rows, audit


def make_cohort_summary(
    df: pd.DataFrame,
    n_bootstrap: int,
    global_seed: int,
) -> pd.DataFrame:
    rows = []
    for rep in REPRESENTATIONS:
        s = df[df["representation"] == rep].copy()
        e = s["E4_local_surrogate_excess"].to_numpy(float)
        ea = s["E4_preprocessing_excess"].to_numpy(float)
        obs = s["delta4_obs"].to_numpy(float)
        lo, hi = bootstrap_median_ci(
            e,
            n_boot=n_bootstrap,
            seed=patient_seed(global_seed, f"cohort_{rep}", "bootstrap_E4"),
        )
        rows.append(
            {
                "representation": rep,
                "n_patients": int(len(s)),
                "median_delta4_obs": float(np.median(obs)),
                "median_surrogate_A_delta4": float(np.median(s["surrogate_A_delta4_median"])),
                "median_surrogate_B_delta4": float(np.median(s["surrogate_B_delta4_median"])),
                "median_E4_preprocessing_excess": float(np.median(ea)),
                "median_E4_local_surrogate_excess": float(np.median(e)),
                "bootstrap95_low_median_E4": lo,
                "bootstrap95_high_median_E4": hi,
                "fraction_patients_E4_gt0": float(np.mean(e > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_run1_gate(summary: pd.DataFrame) -> Tuple[str, List[str]]:
    """Run-1 gate only; this is NOT the final WP2 GO decision."""
    reasons: List[str] = []
    s = summary[summary["representation"] == "shape_norm"]
    if len(s) != 1:
        return "FAIL_IMPLEMENTATION", ["shape_norm cohort summary missing or duplicated"]
    r = s.iloc[0]

    c1 = bool(r["median_E4_local_surrogate_excess"] > 0.0)
    c2 = bool(r["bootstrap95_low_median_E4"] > 0.0)
    c3 = bool(r["fraction_patients_E4_gt0"] >= 0.70)
    c4 = bool(r["median_E4_preprocessing_excess"] > 0.0)

    reasons.append(f"median E4(local surrogate) > 0: {c1}")
    reasons.append(f"bootstrap 95% lower bound > 0: {c2}")
    reasons.append(f"fraction E4>0 >= 0.70: {c3}")
    reasons.append(f"median excess beyond preprocessing null > 0: {c4}")

    if c1 and c2 and c3 and c4:
        return "RUN1_PASS_PROCEED_TO_ROBUSTNESS_LADDER", reasons
    return "RUN1_STOP_OR_REASSESS_BEFORE_ANY_VALIDATION", reasons


def self_test() -> None:
    """Synthetic implementation checks only; not scientific evidence."""
    rng = np.random.default_rng(12345)

    # 1) Representation invariants.
    B = rng.normal(size=(200, 64)) + np.linspace(50.0, 120.0, 64)[None, :]
    reps = make_representations(B)
    assert reps is not None
    assert np.max(np.abs(np.mean(reps["pulse_centered"], axis=1))) < 1e-12
    assert np.max(np.abs(np.mean(reps["shape_norm"], axis=1))) < 1e-12
    assert np.max(np.abs(np.std(reps["shape_norm"], axis=1) - 1.0)) < 1e-12

    # 2) Covariance effective rank equals Paper-1 SVD effective rank.
    for rep in REPRESENTATIONS:
        m = delta_metrics(reps[rep])
        assert m["effective_rank_abs_diff"] <= ERANK_EQ_TOL * max(1.0, m["effective_rank"])

    # 3) Pinching ordering on a random PSD covariance.
    Z = rng.normal(size=(400, 64))
    m = delta_metrics(Z)
    assert -ORDER_TOL <= m["delta2"] <= m["delta4"] + ORDER_TOL
    assert m["delta4"] <= m["delta8"] + ORDER_TOL
    assert m["delta8"] <= m["delta_total"] + ORDER_TOL

    # 4) Surrogate preservation invariants.
    A = surrogate_phase_independent(B, rng)
    for j in range(B.shape[1]):
        assert np.array_equal(np.sort(A[:, j]), np.sort(B[:, j]))

    Bs = surrogate_block_reassigned(B, 4, rng)
    C0 = covariance_matrix(B)
    C1 = covariance_matrix(Bs)
    for sl in block_slices(64, 4):
        assert np.allclose(C0[sl, sl], C1[sl, sl], rtol=0, atol=1e-12)

    # 5) A strong shared latent factor across all phase blocks should create
    #    nonlocal block dependence that is reduced by independent block reassignment.
    n = 1000
    latent = rng.normal(size=(n, 1))
    loading = np.r_[
        np.linspace(0.5, 1.5, 16),
        np.linspace(-1.0, -0.4, 16),
        np.linspace(0.8, 1.2, 16),
        np.linspace(-1.4, -0.6, 16),
    ][None, :]
    X = 80.0 + 12.0 * latent @ loading + rng.normal(scale=1.0, size=(n, 64))
    obs = delta_metrics(X)["delta4"]
    nulls = []
    for _ in range(40):
        Xb = surrogate_block_reassigned(X, 4, rng)
        nulls.append(delta_metrics(Xb)["delta4"])
    assert obs > np.median(nulls)

    print("SELF-TEST: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Paper-1 development directory containing cases/*.npz")
    ap.add_argument("--out", help="Output directory for WP2 Run 1")
    ap.add_argument("--surrogates", type=int, default=DEFAULT_SURROGATES)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--allow-non50", action="store_true", help="Diagnostic only; do not use for frozen development50 run")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.input or not args.out:
        ap.error("--input and --out are required unless --self-test is used")

    if args.surrogates < 1 or args.bootstrap < 1:
        ap.error("--surrogates and --bootstrap must be positive")

    indir = Path(os.path.expanduser(args.input)).resolve()
    outdir = Path(os.path.expanduser(args.out)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    case_files = sorted((indir / "cases").glob("*.npz"))
    if not case_files:
        raise RuntimeError(f"No case files under {indir / 'cases'}")
    if not args.allow_non50 and len(case_files) != 50:
        raise RuntimeError(
            f"Development freeze expects exactly 50 case NPZ files; found {len(case_files)}. "
            "Use --allow-non50 only for diagnostics."
        )

    all_rows: List[dict] = []
    audits: List[dict] = []
    failures: List[str] = []
    seen_pids = set()

    for i, path in enumerate(case_files, start=1):
        try:
            rows, audit = process_patient(
                path,
                n_surrogates=args.surrogates,
                global_seed=args.seed,
            )
            pid = audit["patient_id"]
            if pid in seen_pids:
                raise RuntimeError(f"duplicate patient_id: {pid}")
            seen_pids.add(pid)
            all_rows.extend(rows)
            audits.append(audit)
            print(f"[{i}/{len(case_files)}] {pid}: PASS, accepted beats={audit['n_beats_after_qc']}")
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            print(f"[{i}/{len(case_files)}] {path.name}: FAIL: {exc}")

    df = pd.DataFrame(all_rows)
    patient_csv = outdir / "wp2_run1_patient_results.csv"
    df.to_csv(patient_csv, index=False)

    if len(df):
        summary = make_cohort_summary(df, args.bootstrap, args.seed)
    else:
        summary = pd.DataFrame()
    summary_csv = outdir / "wp2_run1_cohort_summary.csv"
    summary.to_csv(summary_csv, index=False)

    audit_obj = {
        "script": Path(__file__).name,
        "input": str(indir),
        "n_case_files": int(len(case_files)),
        "n_unique_patients_success": int(len(seen_pids)),
        "n_failed_cases": int(len(failures)),
        "n_surrogates_per_patient": int(args.surrogates),
        "n_bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "reference_grid_points": BEAT_POINTS,
        "reference_blocks": REFERENCE_BLOCKS,
        "validation1000_accessed": False,
        "failures": failures,
        "patient_audits": audits,
    }

    if audits:
        audit_obj["max_effective_rank_cov_svd_abs_diff"] = float(
            max(a["max_effective_rank_cov_svd_abs_diff"] for a in audits)
        )
        audit_obj["max_surrogate_A_marginals_abs_error"] = float(
            max(a["surrogate_A_marginals_max_abs_error"] for a in audits)
        )
        audit_obj["max_surrogate_B_within_block_cov_abs_error"] = float(
            max(a["surrogate_B_within_block_cov_max_abs_error"] for a in audits)
        )
        audit_obj["min_observed_cov_eigenvalue"] = float(
            min(a["min_observed_cov_eigenvalue"] for a in audits)
        )

    integrity_pass = (
        len(failures) == 0
        and len(seen_pids) == len(case_files)
        and (args.allow_non50 or len(case_files) == 50)
    )
    audit_obj["integrity_pass"] = bool(integrity_pass)

    with open(outdir / "WP2_RUN1_INTEGRITY.json", "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, indent=2, ensure_ascii=False)

    lines = [
        "WP2 Run 1 integrity audit",
        "=========================",
        "",
        f"Input case files              : {len(case_files)}",
        f"Unique successful patients    : {len(seen_pids)}",
        f"Failed cases                  : {len(failures)}",
        f"Surrogates per patient/type   : {args.surrogates}",
        f"Bootstrap replicates          : {args.bootstrap}",
        f"Reference phase grid          : {BEAT_POINTS}",
        f"Reference block partition     : {REFERENCE_BLOCKS}",
        "Validation1000 accessed        : False",
        "",
    ]
    if audits:
        lines += [
            f"Max cov/SVD erank difference  : {audit_obj['max_effective_rank_cov_svd_abs_diff']:.3e}",
            f"Max Surrogate-A marginal error: {audit_obj['max_surrogate_A_marginals_abs_error']:.3e}",
            f"Max Surrogate-B block-cov err : {audit_obj['max_surrogate_B_within_block_cov_abs_error']:.3e}",
            f"Min observed cov eigenvalue   : {audit_obj['min_observed_cov_eigenvalue']:.3e}",
            "",
        ]
    if failures:
        lines += ["FAILURES", "--------", *failures, ""]
    lines.append("PASS" if integrity_pass else "FAIL")
    (outdir / "WP2_RUN1_INTEGRITY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if integrity_pass and len(summary):
        gate, reasons = evaluate_run1_gate(summary)
    else:
        gate, reasons = "FAIL_IMPLEMENTATION", ["Integrity audit did not pass"]

    gate_lines = [
        "WP2 Run 1 feasibility gate",
        "==========================",
        "",
        f"Decision: {gate}",
        "",
        "This is the minimal Run-1 gate only. It is NOT the final WP2 GO decision.",
        "Validation1000 must remain unopened until the robustness ladder is completed and frozen.",
        "",
        "Reference development representation: shape_norm",
        *[f"- {r}" for r in reasons],
    ]
    (outdir / "WP2_RUN1_GATE.txt").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    print("")
    print("Integrity:", "PASS" if integrity_pass else "FAIL")
    print("Run-1 gate:", gate)
    print("Wrote:", patient_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", outdir / "WP2_RUN1_INTEGRITY.txt")
    print("Wrote:", outdir / "WP2_RUN1_GATE.txt")

    if not integrity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
