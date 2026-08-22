#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 3 — Validation1000 discovery analysis

Scientific role
---------------
Discovery / derivation ONLY. This is not independent confirmatory validation.

Primary question
----------------
Estimate the reproducible between-person geometry of 30-min central ABP morphology
using odd/even 60-s block replicates, then determine whether a patient-common basis
generalizes to held-out patients and can serve as a common coordinate system for
future WF3 longitudinal trajectories.

Primary operator
----------------
Let O_i and E_i be odd/even block-based patient representatives.
The replicate-corrected symmetric between-person operator is

    S_rep = 1/2 { Cov(O,E) + Cov(E,O) }.

This targets patient-level morphology variation reproducible across disjoint temporal
block replicates. Finite-sample S_rep need not be PSD; positive eigenvalues define
the reproducible spectrum. Negative spectral mass is reported as a diagnostic.

No age/sex or clinical labels are read in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SOURCE_FS = 125.0
P = 64
BLOCK_SEC = 60
FULL_WINDOW_SEC = 1800
MIN_BEATS_PER_BLOCK = 32
MIN_TOTAL_BLOCKS = 6
MIN_ODD_BLOCKS = 3
MIN_EVEN_BLOCKS = 3
EXPECTED_SOURCE_N = 1000
EXPECTED_RUN1_SHA256 = "811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01"
CV_FOLDS = 5
CV_SEED = 20260820
MAX_DIM = 24
R2_ABSOLUTE_TARGET = 0.90
R2_CEILING_FRACTION = 0.95
R2_REPLICATE_TARGET = 0.85
RANDOM_SUBSPACES = 200
RANDOM_SEED = 20260821
STABILITY_SPLITS = 100
STABILITY_SEED = 20260822
POS_TOL_REL = 1e-10

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

def deterministic_sign(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, float).copy()
    for j in range(U.shape[1]):
        k = int(np.argmax(np.abs(U[:, j])))
        if U[k, j] < 0:
            U[:, j] *= -1.0
    return U

def construct_patient(path: Path, run1) -> Dict[str, Any]:
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
    shape = np.asarray(rep125["shape_norm"][acc], float)
    centers = catalog["center_sec"].to_numpy(float)[acc]

    q_list: List[np.ndarray] = []
    kappa_list: List[float] = []
    block_ids: List[int] = []
    beat_counts: List[int] = []

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
        mean_shape = np.mean(Xb, axis=0)
        kappa = float(np.std(mean_shape - np.mean(mean_shape)))
        q = normalize_shape(mean_shape)
        if q is None or not np.isfinite(kappa):
            continue
        q_list.append(q)
        kappa_list.append(kappa)
        block_ids.append(b)
        beat_counts.append(int(len(Xb)))

    odd = [q for q, b in zip(q_list, block_ids) if b % 2 == 1]
    even = [q for q, b in zip(q_list, block_ids) if b % 2 == 0]

    if len(q_list) < MIN_TOTAL_BLOCKS or len(odd) < MIN_ODD_BLOCKS or len(even) < MIN_EVEN_BLOCKS:
        raise RuntimeError(
            f"{pid}: frozen block eligibility failed "
            f"(total={len(q_list)}, odd={len(odd)}, even={len(even)})"
        )

    def rep(xs: List[np.ndarray]) -> np.ndarray:
        z = normalize_shape(np.mean(np.vstack(xs), axis=0))
        if z is None:
            raise RuntimeError(f"{pid}: representative normalization failed")
        return z

    all_rep = rep(q_list)
    odd_rep = rep(odd)
    even_rep = rep(even)

    Q = np.vstack(q_list)
    qbar = np.mean(Q, axis=0)
    centered = Q - qbar
    within_cov = centered.T @ centered / max(1, len(Q) - 1)

    return {
        "patient_id": pid,
        "all_rep": all_rep,
        "odd_rep": odd_rep,
        "even_rep": even_rep,
        "within_cov": within_cov,
        "eligible_blocks": len(q_list),
        "odd_blocks": len(odd),
        "even_blocks": len(even),
        "median_block_coherence": float(np.median(kappa_list)),
        "median_beats_per_block": float(np.median(beat_counts)),
    }

def sym_cross_operator(O: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mo = np.mean(O, axis=0)
    me = np.mean(E, axis=0)
    Ao = O - mo
    Ae = E - me
    C = Ao.T @ Ae / (len(O) - 1)
    S = 0.5 * (C + C.T)
    return S, mo, me

def eig_sorted(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = deterministic_sign(vecs[:, order])
    return vals, vecs

def positive_rank(vals: np.ndarray) -> int:
    scale = max(1.0, float(np.max(np.abs(vals))))
    tol = POS_TOL_REL * scale
    return int(np.sum(vals > tol))

def positive_spectrum_metrics(vals: np.ndarray) -> Dict[str, Any]:
    vals = np.asarray(vals, float)
    pos = vals[vals > POS_TOL_REL * max(1.0, float(np.max(np.abs(vals))))]
    neg = vals[vals < -POS_TOL_REL * max(1.0, float(np.max(np.abs(vals))))]
    total_abs = float(np.sum(np.abs(vals)))
    neg_mass = float(np.sum(np.abs(neg)))
    if len(pos) == 0 or np.sum(pos) <= 0:
        return {
            "positive_rank": 0, "effective_rank_positive": np.nan,
            "d90_positive": None, "d95_positive": None,
            "negative_mass_fraction": neg_mass / total_abs if total_abs > 0 else np.nan,
        }
    p = pos / np.sum(pos)
    erank = float(np.exp(-np.sum(p * np.log(p))))
    c = np.cumsum(pos) / np.sum(pos)
    d90 = int(np.searchsorted(c, 0.90) + 1)
    d95 = int(np.searchsorted(c, 0.95) + 1)
    return {
        "positive_rank": int(len(pos)),
        "effective_rank_positive": erank,
        "d90_positive": d90,
        "d95_positive": d95,
        "negative_mass_fraction": neg_mass / total_abs if total_abs > 0 else 0.0,
    }

def stable_fold(pid: str) -> int:
    key = f"{CV_SEED}:{pid}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % CV_FOLDS

def aggregate_r2(y: np.ndarray, mean: np.ndarray, U: np.ndarray) -> Tuple[float, float]:
    D = y - mean
    pred = D @ U @ U.T
    err = D - pred
    return float(np.sum(err * err)), float(np.sum(D * D))

def pca_basis(M: np.ndarray) -> np.ndarray:
    X = M - np.mean(M, axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return deterministic_sign(vt.T)

def fourier_basis() -> np.ndarray:
    phi = np.arange(P, dtype=float) / P
    cols = []
    for k in range(1, P // 2):
        cols.append(np.cos(2*np.pi*k*phi))
        cols.append(np.sin(2*np.pi*k*phi))
    A = np.column_stack(cols)
    A = A - np.mean(A, axis=0, keepdims=True)
    Q, _ = np.linalg.qr(A)
    return deterministic_sign(Q)

def cv_curves(M, O, E, pids) -> Tuple[pd.DataFrame, Dict[int, int]]:
    fold_ids = np.array([stable_fold(x) for x in pids], dtype=int)
    accum: Dict[Tuple[str, int], List[float]] = {}
    fold_pos_rank = {}

    for f in range(CV_FOLDS):
        tr = fold_ids != f
        te = fold_ids == f
        if tr.sum() < 20 or te.sum() < 1:
            raise RuntimeError(f"CV fold {f}: invalid sizes")

        Srep, _, _ = sym_cross_operator(O[tr], E[tr])
        vals, Urep = eig_sorted(Srep)
        pr = positive_rank(vals)
        fold_pos_rank[f] = pr

        Upca = pca_basis(M[tr])
        mu = np.mean(M[tr], axis=0)

        maxd = min(MAX_DIM, pr, Upca.shape[1])
        for d in range(1, maxd + 1):
            for label, Y in (("all", M[te]), ("odd", O[te]), ("even", E[te])):
                sse, sst = aggregate_r2(Y, mu, Urep[:, :d])
                accum.setdefault((f"rep_{label}_sse", d), []).append(sse)
                accum.setdefault((f"rep_{label}_sst", d), []).append(sst)

            sse, sst = aggregate_r2(M[te], mu, Upca[:, :d])
            accum.setdefault(("pca_all_sse", d), []).append(sse)
            accum.setdefault(("pca_all_sst", d), []).append(sst)

    dims = sorted({
        d for (name, d), vals in accum.items()
        if name == "rep_all_sse" and len(vals) == CV_FOLDS
    })
    rows = []
    for d in dims:
        def r2(prefix):
            sse = sum(accum[(prefix+"_sse", d)])
            sst = sum(accum[(prefix+"_sst", d)])
            return 1.0 - sse/sst if sst > 0 else np.nan
        rows.append({
            "dimension": d,
            "replicate_basis_cv_r2_all": r2("rep_all"),
            "replicate_basis_cv_r2_odd": r2("rep_odd"),
            "replicate_basis_cv_r2_even": r2("rep_even"),
            "ordinary_pca_cv_r2_all": r2("pca_all"),
        })
    return pd.DataFrame(rows), fold_pos_rank

def choose_dimension(curve: pd.DataFrame) -> Tuple[int | None, Dict[str, Any]]:
    if curve.empty:
        return None, {"reason": "empty_cv_curve"}
    maxd = int(curve["dimension"].max())
    ceiling = float(curve.loc[curve["dimension"] == maxd, "replicate_basis_cv_r2_all"].iloc[0])
    threshold = max(R2_ABSOLUTE_TARGET, R2_CEILING_FRACTION * ceiling)
    ok = curve[
        (curve["replicate_basis_cv_r2_all"] >= threshold)
        & (curve["replicate_basis_cv_r2_odd"] >= R2_REPLICATE_TARGET)
        & (curve["replicate_basis_cv_r2_even"] >= R2_REPLICATE_TARGET)
    ]
    if ok.empty:
        return None, {
            "reason": "no_dimension_met_frozen_reconstruction_rule",
            "ceiling_dimension": maxd,
            "ceiling_r2": ceiling,
            "required_all_r2": threshold,
            "required_odd_even_r2": R2_REPLICATE_TARGET,
        }
    d = int(ok["dimension"].min())
    return d, {
        "reason": "frozen_rule_met",
        "ceiling_dimension": maxd,
        "ceiling_r2": ceiling,
        "required_all_r2": threshold,
        "required_odd_even_r2": R2_REPLICATE_TARGET,
    }

def random_mean_zero_basis(rng: np.random.Generator, d: int) -> np.ndarray:
    A = rng.normal(size=(P, d))
    one = np.ones(P, float)
    one /= np.linalg.norm(one)
    A = A - one[:, None] * (one @ A)[None, :]
    Q, _ = np.linalg.qr(A)
    return Q[:, :d]

def random_baseline_cv(M, pids, d) -> np.ndarray:
    fold_ids = np.array([stable_fold(x) for x in pids], dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    vals = []
    for _ in range(RANDOM_SUBSPACES):
        U = random_mean_zero_basis(rng, d)
        sse_total = 0.0
        sst_total = 0.0
        for f in range(CV_FOLDS):
            tr = fold_ids != f
            te = fold_ids == f
            mu = np.mean(M[tr], axis=0)
            sse, sst = aggregate_r2(M[te], mu, U)
            sse_total += sse
            sst_total += sst
        vals.append(1.0 - sse_total/sst_total)
    return np.asarray(vals, float)

def halfsplit_stability(O, E, d) -> np.ndarray:
    rng = np.random.default_rng(STABILITY_SEED)
    n = len(O)
    out = []
    for _ in range(STABILITY_SPLITS):
        perm = rng.permutation(n)
        a = perm[:n//2]
        b = perm[n//2:]
        Sa, _, _ = sym_cross_operator(O[a], E[a])
        Sb, _, _ = sym_cross_operator(O[b], E[b])
        va, Ua = eig_sorted(Sa)
        vb, Ub = eig_sorted(Sb)
        if positive_rank(va) < d or positive_rank(vb) < d:
            continue
        A = Ua[:, :d]
        B = Ub[:, :d]
        overlap = float(np.sum((A.T @ B)**2) / d)
        out.append(overlap)
    return np.asarray(out, float)

def axis_reliability(O, E, U, d) -> pd.DataFrame:
    mo = np.mean(O, axis=0)
    me = np.mean(E, axis=0)
    Zo = (O - mo) @ U[:, :d]
    Ze = (E - me) @ U[:, :d]
    rows = []
    for j in range(d):
        a, b = Zo[:, j], Ze[:, j]
        den = np.std(a, ddof=1) * np.std(b, ddof=1)
        corr = float(np.cov(a, b, ddof=1)[0,1] / den) if den > 0 else np.nan
        rows.append({
            "axis": j+1,
            "odd_even_score_correlation": corr,
            "odd_sd": float(np.std(a, ddof=1)),
            "even_sd": float(np.std(b, ddof=1)),
        })
    return pd.DataFrame(rows)

def validate_spec(spec_path: Path, script_path: Path, run1_path: Path) -> Dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_VALIDATION1000_WFP_DISCOVERY_ACCESS":
        raise RuntimeError("Frozen spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(script_path):
        raise RuntimeError("Discovery script hash differs from frozen spec")
    if spec.get("authoritative_run1_sha256") != sha256_file(run1_path):
        raise RuntimeError("Run-1 hash differs from frozen spec")
    frozen_min_beats = (
        spec.get("block_definition", {}).get("min_accepted_beats_per_block")
    )
    if frozen_min_beats != MIN_BEATS_PER_BLOCK:
        raise RuntimeError(
            "Frozen min-beats rule differs from code: "
            f"spec={frozen_min_beats}, code={MIN_BEATS_PER_BLOCK}"
        )
    return spec

def self_test() -> int:
    rng = np.random.default_rng(123)
    n = 200
    # Mean-zero ambient basis.
    A = rng.normal(size=(P, 4))
    A -= np.mean(A, axis=0, keepdims=True)
    U, _ = np.linalg.qr(A)
    z = rng.normal(size=(n, 4))
    signal = z @ U[:, :4].T
    O = signal + rng.normal(scale=.2, size=(n,P))
    E = signal + rng.normal(scale=.2, size=(n,P))
    O -= np.mean(O, axis=1, keepdims=True)
    E -= np.mean(E, axis=1, keepdims=True)
    S, _, _ = sym_cross_operator(O, E)
    vals, V = eig_sorted(S)
    if positive_rank(vals) < 4:
        print("SELF-TEST FAIL: positive rank", file=sys.stderr)
        return 1
    overlap = float(np.sum((U[:, :4].T @ V[:, :4])**2)/4)
    if overlap < 0.8:
        print("SELF-TEST FAIL: recovered subspace", overlap, file=sys.stderr)
        return 1
    print("WFP discovery analysis self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--input", default="~/Documents/abp_information_study/data/abp125_validation1000")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_discovery_validation1000")
    ap.add_argument("--spec", default="~/Documents/abp_information_study/freeze/WFP_DISCOVERY_FROZEN_SPEC.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    project = Path(args.project_root).expanduser().resolve()
    input_dir = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    run1_path = project / "code" / "wp2_run1_development50.py"
    script_path = Path(__file__).resolve()
    spec = validate_spec(spec_path, script_path, run1_path)
    run1 = load_module(run1_path, "wfp_run1")

    cases = sorted((input_dir / "cases").glob("*.npz"))
    if len(cases) != EXPECTED_SOURCE_N:
        raise SystemExit(f"FAIL: expected 1000 Validation1000 cases, found {len(cases)}")

    patients = []
    failures = []
    for p in cases:
        try:
            patients.append(construct_patient(p, run1))
        except Exception as e:
            failures.append({"file": p.name, "error": repr(e)})

    if len(patients) < 100:
        raise SystemExit("FAIL: fewer than 100 analysable patients; discovery not interpretable")

    pids = [x["patient_id"] for x in patients]
    if len(set(pids)) != len(pids):
        raise SystemExit("FAIL: duplicate analysable patient IDs")

    M = np.vstack([x["all_rep"] for x in patients])
    O = np.vstack([x["odd_rep"] for x in patients])
    E = np.vstack([x["even_rep"] for x in patients])
    Swithin = np.mean(np.stack([x["within_cov"] for x in patients], axis=0), axis=0)

    Srep, mo, me = sym_cross_operator(O, E)
    vals, Urep = eig_sorted(Srep)
    specm = positive_spectrum_metrics(vals)
    pr = specm["positive_rank"]

    # Ordinary all-representative covariance benchmark.
    mu_all = np.mean(M, axis=0)
    Xall = M - mu_all
    Sall = Xall.T @ Xall / (len(M)-1)
    vals_all, Uall = eig_sorted(Sall)

    cv, fold_pr = cv_curves(M, O, E, pids)
    dstar, dmeta = choose_dimension(cv)

    # Fixed Fourier comparator curve at even dimensions only.
    Uf = fourier_basis()
    fold_ids = np.array([stable_fold(x) for x in pids], dtype=int)
    fourier_rows = []
    for d in range(2, min(MAX_DIM, Uf.shape[1]) + 1, 2):
        sse_total = sst_total = 0.0
        for f in range(CV_FOLDS):
            tr = fold_ids != f
            te = fold_ids == f
            mu = np.mean(M[tr], axis=0)
            sse, sst = aggregate_r2(M[te], mu, Uf[:, :d])
            sse_total += sse
            sst_total += sst
        fourier_rows.append({
            "dimension": d,
            "fourier_cv_r2_all": 1.0 - sse_total/sst_total
        })
    fourier_df = pd.DataFrame(fourier_rows)

    decision = "WFP_DISCOVERY_COMMON_BASIS_NOT_ESTABLISHED"
    selected = {}
    if dstar is not None and dstar <= pr:
        decision = "WFP_DISCOVERY_COMMON_BASIS_IDENTIFIED"
        Ub = Urep[:, :dstar]

        rand = random_baseline_cv(M, pids, dstar)
        cvrow = cv[cv["dimension"] == dstar].iloc[0]

        stab = halfsplit_stability(O, E, dstar)

        # Within-window block geometry.
        wvals, Uw = eig_sorted(Swithin)
        Uw_d = Uw[:, :min(dstar, Uw.shape[1])]
        capture_w_by_b = float(np.trace(Ub.T @ Swithin @ Ub) / np.trace(Swithin)) if np.trace(Swithin) > 0 else np.nan

        posvals = np.clip(vals, 0, None)
        Srep_pos = Urep @ np.diag(posvals) @ Urep.T
        capture_b_by_w = (
            float(np.trace(Uw_d.T @ Srep_pos @ Uw_d) / np.trace(Srep_pos))
            if np.trace(Srep_pos) > 0 else np.nan
        )
        overlap = float(np.sum((Ub.T @ Uw_d)**2) / min(dstar, Uw_d.shape[1]))
        svals = np.linalg.svd(Ub.T @ Uw_d, compute_uv=False)
        angles = np.degrees(np.arccos(np.clip(svals, -1, 1)))

        selected = {
            "dimension": dstar,
            "cv_r2_all": float(cvrow["replicate_basis_cv_r2_all"]),
            "cv_r2_odd": float(cvrow["replicate_basis_cv_r2_odd"]),
            "cv_r2_even": float(cvrow["replicate_basis_cv_r2_even"]),
            "ordinary_pca_cv_r2_all_same_d": float(cvrow["ordinary_pca_cv_r2_all"]),
            "random_subspace_r2_median": float(np.median(rand)),
            "random_subspace_r2_95th": float(np.percentile(rand, 95)),
            "random_subspace_percentile_of_primary": float(np.mean(rand <= float(cvrow["replicate_basis_cv_r2_all"]))),
            "halfsplit_subspace_overlap_n": int(len(stab)),
            "halfsplit_subspace_overlap_median": float(np.median(stab)) if len(stab) else np.nan,
            "halfsplit_subspace_overlap_q05": float(np.percentile(stab, 5)) if len(stab) else np.nan,
            "within_window_variance_captured_by_between_basis": capture_w_by_b,
            "positive_between_variance_captured_by_within_basis": capture_b_by_w,
            "between_within_projector_overlap": overlap,
            "principal_angles_degrees": [float(x) for x in angles],
        }

        axis_df = axis_reliability(O, E, Urep, dstar)
        axis_df.to_csv(out / "wfp_axis_reliability.csv", index=False)

        # Freeze-ready coordinate objects for later independent validation / WF3.
        np.savez_compressed(
            out / "WFP_DISCOVERY_COMMON_COORDINATES.npz",
            population_mean=mu_all.astype(np.float64),
            between_basis=Ub.astype(np.float64),
            between_eigenvalues=vals[:dstar].astype(np.float64),
            within_window_covariance=Swithin.astype(np.float64),
            dimension=np.asarray(dstar, dtype=np.int64),
        )

        # Patient-level scores: local scientific output; not for public release without review.
        scores = (M - mu_all) @ Ub
        sdf = pd.DataFrame(scores, columns=[f"z{j+1}" for j in range(dstar)])
        sdf.insert(0, "patient_id", pids)
        sdf["eligible_blocks"] = [x["eligible_blocks"] for x in patients]
        sdf["median_block_coherence"] = [x["median_block_coherence"] for x in patients]
        sdf.to_csv(out / "wfp_patient_scores_DISCOVERY_PRIVATE.csv", index=False)

    cv.to_csv(out / "wfp_cv_reconstruction_curve.csv", index=False)
    fourier_df.to_csv(out / "wfp_fourier_comparator_curve.csv", index=False)
    pd.DataFrame({
        "eigen_index": np.arange(1, len(vals)+1),
        "replicate_eigenvalue": vals,
        "ordinary_patient_mean_eigenvalue": vals_all,
    }).to_csv(out / "wfp_population_eigenspectra.csv", index=False)

    result = {
        "schema_version": 1,
        "decision": decision,
        "scientific_role": "discovery_derivation_only",
        "source": "Validation1000",
        "source_n": len(cases),
        "analysable_n": len(patients),
        "frozen_rule_exclusions_n": len(failures),
        "independent_confirmatory_validation": False,
        "age_sex_analysis_performed": False,
        "clinical_labels_accessed": False,
        "primary_operator": "symmetric odd/even replicate cross-covariance",
        "positive_spectrum": specm,
        "dimension_selection": dmeta,
        "selected_basis": selected,
        "cv_positive_rank_by_fold": {str(k): int(v) for k,v in fold_pr.items()},
        "frozen_spec_sha256": sha256_file(spec_path),
        "analysis_script_sha256": sha256_file(script_path),
        "authoritative_run1_sha256": sha256_file(run1_path),
        "failures": failures,
        "interpretation_boundary": [
            "30-min patient representative is central morphology, not trait or state.",
            "Within-window block covariance is not WF3 long-duration temporal covariance.",
            "No Zstate or Ztrait claim is authorized.",
            "Discovery findings require independent validation before confirmatory claims.",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
    }
    atomic_json(out / "WFP_DISCOVERY_RESULTS.json", result)

    lines = [
        "WF-P VALIDATION1000 DISCOVERY READOUT",
        "====================================",
        f"Decision: {decision}",
        "Scientific role: DISCOVERY / DERIVATION ONLY",
        f"Source n: {len(cases)}",
        f"Analysable n: {len(patients)}",
        f"Frozen-rule exclusions: {len(failures)}",
        "Independent confirmatory validation: NO",
        "Clinical labels accessed: NO",
        "Age/sex analysis performed: NO",
        "",
        "Primary replicate-corrected positive spectrum:",
        f"  positive rank: {specm['positive_rank']}",
        f"  effective rank: {specm['effective_rank_positive']}",
        f"  d90: {specm['d90_positive']}",
        f"  d95: {specm['d95_positive']}",
        f"  negative spectral mass fraction: {specm['negative_mass_fraction']}",
        "",
        f"Selected common-basis dimension: {dstar if dstar is not None else 'NONE'}",
    ]
    if selected:
        lines += [
            f"  CV R2 all: {selected['cv_r2_all']}",
            f"  CV R2 odd: {selected['cv_r2_odd']}",
            f"  CV R2 even: {selected['cv_r2_even']}",
            f"  Ordinary PCA CV R2 same d: {selected['ordinary_pca_cv_r2_all_same_d']}",
            f"  Random-subspace 95th percentile R2: {selected['random_subspace_r2_95th']}",
            f"  Half-split subspace overlap median: {selected['halfsplit_subspace_overlap_median']}",
            f"  Within-window variance captured by between basis: {selected['within_window_variance_captured_by_between_basis']}",
            f"  Between/within projector overlap: {selected['between_within_projector_overlap']}",
        ]
    lines += [
        "",
        "Boundary:",
        "  This is discovery evidence only.",
        "  Do not label axes as Zstate or Ztrait.",
        "  Do not perform age/sex analysis until a separate clinical-linkage preflight/spec is frozen.",
    ]
    atomic_text(out / "WFP_DISCOVERY_READOUT.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
