#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7B-RD — Residual Dependence Adjudication
===================================================

Scientific role
---------------
Post-result artifact-control audit triggered by the Stage7B-NL residual
distance-correlation signal.

This stage asks a deliberately narrow question:

    Could the observed dependence between constitutional variables X and the
    OOF-linear residual B8 coordinates be produced by the OOF residualization
    pipeline / fold structure itself?

It does NOT test or model physiological variance structure directly.

Primary artifact control
------------------------
Pipeline-replay residual-permutation null:

1. Fit the corresponding linear constitutional model to all patients.
2. Keep the fitted linear mean fixed.
3. Permute complete 8-D residual vectors across patients.
4. Form pseudo-B8 outcomes under residual exchangeability.
5. Re-run the entire frozen 5-fold OOF linear prediction pipeline.
6. Recompute OOF residuals.
7. Recompute distance correlation between X and those OOF residuals.

This propagates finite-sample OOF fitting/residualization through the null.

Secondary diagnostic
--------------------
Fold-stratified permutation of the observed OOF residual vectors. Residual
vectors are permuted only within the same frozen CV fold. This preserves any
fold-level residual distribution and fold-level X imbalance.

Important
---------
- The pipeline-replay residual-permutation null assumes residual-vector
  exchangeability under the null. It is an artifact-control sensitivity, not
  an exact distribution-free conditional-independence test.
- A persistent signal is NOT automatically a nonlinear-mean finding. It can
  reflect conditional variance/covariance or other higher-order dependence.
- No variance/covariance model is fit here.
- Existing Stage7B and Stage7B-NL results are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

EXPECTED_N = 978
EXPECTED_HEIGHT_N = 693
D = 8
CV_FOLDS = 5
CV_SEED = 20260820
PIPELINE_PERMUTATIONS = 999
BLOCKED_PERMUTATIONS = 999
PERM_SEED = 20260825
REPLAY_DCOR_TOL = 5e-6

EXPECTED_FULL_DCOR = 0.167447
EXPECTED_HEIGHT_DCOR = 0.180459


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_fold(pid: str) -> int:
    h = hashlib.sha256(f"{CV_SEED}:{pid}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % CV_FOLDS


def standardize_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X - mu) / sd


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    g = np.sum(X * X, axis=1)
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def double_center(Dm: np.ndarray) -> np.ndarray:
    return (
        Dm
        - Dm.mean(axis=1, keepdims=True)
        - Dm.mean(axis=0, keepdims=True)
        + Dm.mean()
    )


def centered_distance_matrix(X: np.ndarray) -> np.ndarray:
    return double_center(pairwise_euclidean(standardize_matrix(X)))


def dcor_from_centered(A: np.ndarray, B: np.ndarray) -> float:
    dcov2 = float(np.mean(A * B))
    dvarx2 = float(np.mean(A * A))
    dvary2 = float(np.mean(B * B))
    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0
    dcor2 = max(dcov2, 0.0) / math.sqrt(dvarx2 * dvary2)
    return math.sqrt(max(dcor2, 0.0))


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    return dcor_from_centered(
        centered_distance_matrix(X),
        centered_distance_matrix(Y),
    )


def female_from_gender(s: pd.Series) -> np.ndarray:
    g = s.astype(str).str.upper().str.strip()
    if (~g.isin(["M", "F"])).any():
        raise RuntimeError("unexpected gender coding")
    return (g == "F").astype(float).to_numpy()


def prepare_fold_design(
    df: pd.DataFrame,
    continuous_cols: list[str],
) -> Tuple[np.ndarray, list[dict]]:
    folds = np.asarray([stable_fold(pid) for pid in df["patient_id"].astype(str)])
    plans = []
    for f in range(CV_FOLDS):
        tr = folds != f
        te = folds == f
        if np.sum(te) == 0:
            raise RuntimeError(f"empty CV fold {f}")

        Xtr_c = df.loc[tr, continuous_cols].to_numpy(float)
        Xte_c = df.loc[te, continuous_cols].to_numpy(float)
        mu = np.mean(Xtr_c, axis=0)
        sd = np.std(Xtr_c, axis=0, ddof=0)
        sd = np.where(sd <= 1e-12, 1.0, sd)
        Xtr_c = (Xtr_c - mu) / sd
        Xte_c = (Xte_c - mu) / sd

        sex_tr = df.loc[tr, "female"].to_numpy(float)[:, None]
        sex_te = df.loc[te, "female"].to_numpy(float)[:, None]

        Atr = np.column_stack([np.ones(np.sum(tr)), Xtr_c, sex_tr])
        Ate = np.column_stack([np.ones(np.sum(te)), Xte_c, sex_te])

        # Fixed-design pseudoinverse can be reused across all null datasets.
        pinv = np.linalg.pinv(Atr)
        plans.append({
            "fold": f,
            "train_mask": tr,
            "test_mask": te,
            "train_pinv": pinv,
            "test_design": Ate,
        })
    return folds, plans


def oof_predict_from_plans(
    Z: np.ndarray,
    plans: list[dict],
) -> np.ndarray:
    pred = np.full_like(Z, np.nan, dtype=float)
    for p in plans:
        tr = p["train_mask"]
        te = p["test_mask"]
        beta = p["train_pinv"] @ Z[tr]
        pred[te] = p["test_design"] @ beta
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("nonfinite OOF predictions")
    return pred


def full_linear_fit(
    df: pd.DataFrame,
    Z: np.ndarray,
    continuous_cols: list[str],
) -> Tuple[np.ndarray, np.ndarray]:
    Xc = df[continuous_cols].to_numpy(float)
    Xc = standardize_matrix(Xc)
    sex = df["female"].to_numpy(float)[:, None]
    A = np.column_stack([np.ones(len(df)), Xc, sex])
    beta = np.linalg.lstsq(A, Z, rcond=None)[0]
    fitted = A @ beta
    resid = Z - fitted
    return fitted, resid


def pipeline_replay_null(
    df: pd.DataFrame,
    Z: np.ndarray,
    X_dcor: np.ndarray,
    continuous_cols: list[str],
    n_perm: int,
    seed: int,
    label: str,
) -> Dict[str, float]:
    folds, plans = prepare_fold_design(df, continuous_cols)
    pred_obs = oof_predict_from_plans(Z, plans)
    resid_obs = Z - pred_obs

    A_x = centered_distance_matrix(X_dcor)
    B_obs = centered_distance_matrix(resid_obs)
    obs = dcor_from_centered(A_x, B_obs)

    fitted_full, resid_full = full_linear_fit(df, Z, continuous_cols)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, float)
    for b in range(n_perm):
        perm = rng.permutation(len(df))
        Zb = fitted_full + resid_full[perm]
        pred_b = oof_predict_from_plans(Zb, plans)
        resid_b = Zb - pred_b
        B_b = centered_distance_matrix(resid_b)
        null[b] = dcor_from_centered(A_x, B_b)
        if (b + 1) % 50 == 0 or (b + 1) == n_perm:
            print(f"[pipeline-replay {label}] {b+1}/{n_perm}", flush=True)

    p = (1.0 + float(np.sum(null >= obs - 1e-15))) / (n_perm + 1.0)
    return {
        "observed_dcor": float(obs),
        "pipeline_null_p_plus_one": float(p),
        "pipeline_null_median": float(np.median(null)),
        "pipeline_null_q95": float(np.percentile(null, 95)),
        "observed_minus_pipeline_q95": float(obs - np.percentile(null, 95)),
        "pipeline_permutations": int(n_perm),
    }


def blocked_fold_null(
    df: pd.DataFrame,
    Z: np.ndarray,
    X_dcor: np.ndarray,
    continuous_cols: list[str],
    n_perm: int,
    seed: int,
    label: str,
) -> Dict[str, float]:
    folds, plans = prepare_fold_design(df, continuous_cols)
    pred = oof_predict_from_plans(Z, plans)
    resid = Z - pred

    A = centered_distance_matrix(X_dcor)
    B = centered_distance_matrix(resid)
    obs = dcor_from_centered(A, B)

    dvarx2 = float(np.mean(A * A))
    dvary2 = float(np.mean(B * B))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, float)

    fold_indices = [np.flatnonzero(folds == f) for f in range(CV_FOLDS)]
    base = np.arange(len(df))
    for b in range(n_perm):
        perm = base.copy()
        for idx in fold_indices:
            perm[idx] = rng.permutation(idx)

        Bp = B[perm][:, perm]
        dcov2 = float(np.mean(A * Bp))
        dcor2 = max(dcov2, 0.0) / math.sqrt(dvarx2 * dvary2)
        null[b] = math.sqrt(max(dcor2, 0.0))

        if (b + 1) % 100 == 0 or (b + 1) == n_perm:
            print(f"[within-fold diagnostic {label}] {b+1}/{n_perm}", flush=True)

    p = (1.0 + float(np.sum(null >= obs - 1e-15))) / (n_perm + 1.0)
    return {
        "observed_dcor": float(obs),
        "within_fold_null_p_plus_one": float(p),
        "within_fold_null_median": float(np.median(null)),
        "within_fold_null_q95": float(np.percentile(null, 95)),
        "observed_minus_within_fold_q95": float(obs - np.percentile(null, 95)),
        "within_fold_permutations": int(n_perm),
    }


def holm_two(p1: float, p2: float) -> Tuple[float, float]:
    p = np.asarray([p1, p2], float)
    order = np.argsort(p)
    adj = np.empty(2, float)
    first = min(1.0, 2.0 * p[order[0]])
    second = min(1.0, max(first, p[order[1]]))
    adj[order[0]] = first
    adj[order[1]] = second
    return float(adj[0]), float(adj[1])


def parse_expected_readout(readout: str) -> Tuple[float, float]:
    m1 = re.search(
        r"Full age\+sex vs OOF residual B8:\s*dCor=([0-9.]+)",
        readout,
    )
    m2 = re.search(
        r"Height age\+sex\+height vs OOF residual B8:\s*dCor=([0-9.]+)",
        readout,
    )
    if not m1 or not m2:
        raise RuntimeError("could not parse Stage7B-NL dCor values from readout")
    return float(m1.group(1)), float(m2.group(1))


def self_test() -> int:
    rng = np.random.default_rng(20260825)
    n = 90
    df = pd.DataFrame({
        "patient_id": [f"p{i:04d}" for i in range(n)],
        "age": rng.uniform(20, 90, n),
        "female": rng.integers(0, 2, n).astype(float),
        "height": rng.normal(170, 8, n),
    })
    X = np.column_stack([df["age"], df["female"]])
    Z = np.column_stack([
        0.03 * df["age"].to_numpy() + rng.normal(size=n),
        rng.normal(size=n),
        rng.normal(size=n),
    ])
    folds, plans = prepare_fold_design(df, ["age"])
    pred = oof_predict_from_plans(Z, plans)
    if pred.shape != Z.shape or not np.all(np.isfinite(pred)):
        raise RuntimeError("OOF replay self-test failed")
    d = distance_correlation(X, Z - pred)
    if not np.isfinite(d):
        raise RuntimeError("dCor self-test failed")

    # Exercise both null engines with tiny budgets; only integrity is tested.
    a = pipeline_replay_null(df, Z, X, ["age"], 9, 1, "selftest")
    b = blocked_fold_null(df, Z, X, ["age"], 9, 2, "selftest")
    if not (0 < a["pipeline_null_p_plus_one"] <= 1):
        raise RuntimeError("pipeline-null p self-test failed")
    if not (0 < b["within_fold_null_p_plus_one"] <= 1):
        raise RuntimeError("blocked-null p self-test failed")

    print("WF-P Stage7B-RD analysis self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file")
    ap.add_argument("--temporal-linkage")
    ap.add_argument("--height-preflight")
    ap.add_argument("--stage7b-nl-results-json")
    ap.add_argument("--stage7b-nl-readout")
    ap.add_argument("--stage7b-nl-spec")
    ap.add_argument("--spec")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    needed = [
        "score_file","temporal_linkage","height_preflight",
        "stage7b_nl_results_json","stage7b_nl_readout",
        "stage7b_nl_spec","spec","out",
    ]
    missing = [x for x in needed if getattr(a, x) is None]
    if missing:
        raise SystemExit(f"Missing required args: {missing}")

    scriptp = Path(__file__).resolve()
    scorep = Path(a.score_file).expanduser().resolve()
    temporalp = Path(a.temporal_linkage).expanduser().resolve()
    heightp = Path(a.height_preflight).expanduser().resolve()
    nlrp = Path(a.stage7b_nl_results_json).expanduser().resolve()
    nlreadp = Path(a.stage7b_nl_readout).expanduser().resolve()
    nlspecp = Path(a.stage7b_nl_spec).expanduser().resolve()
    specp = Path(a.spec).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(specp.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_STAGE7B_RD_ADJUDICATION":
        raise SystemExit("FAIL invalid Stage7B-RD frozen spec status")
    if spec.get("analysis_script_sha256") != sha256_file(scriptp):
        raise SystemExit("FAIL Stage7B-RD analysis-script hash mismatch")

    filemap = {
        "score_file_sha256": scorep,
        "temporal_linkage_sha256": temporalp,
        "height_preflight_sha256": heightp,
        "stage7b_nl_results_sha256": nlrp,
        "stage7b_nl_readout_sha256": nlreadp,
        "stage7b_nl_spec_sha256": nlspecp,
    }
    for key, p in filemap.items():
        if spec.get(key) != sha256_file(p):
            raise SystemExit(f"FAIL frozen input hash mismatch: {key}")

    nlread = nlreadp.read_text(encoding="utf-8", errors="replace")
    if "NO_MATERIAL_OOF_MEAN_GAIN_BUT_RESIDUAL_MODEL_FREE_DEPENDENCE_FLAG" not in nlread:
        raise SystemExit("FAIL Stage7B-NL trigger decision not found")

    parsed_full, parsed_height = parse_expected_readout(nlread)
    if abs(parsed_full - EXPECTED_FULL_DCOR) > 5e-6:
        raise SystemExit(f"FAIL unexpected Stage7B-NL full dCor {parsed_full}")
    if abs(parsed_height - EXPECTED_HEIGHT_DCOR) > 5e-6:
        raise SystemExit(f"FAIL unexpected Stage7B-NL height dCor {parsed_height}")

    scores = pd.read_csv(scorep, dtype={"patient_id": str})
    temporal = pd.read_csv(
        temporalp,
        usecols=["patient_id","age_years_capped90","gender"],
        dtype={"patient_id": str},
    )
    height = pd.read_csv(
        heightp,
        usecols=["patient_id","height_median_cm"],
        dtype={"patient_id": str},
    )

    zcols = [f"z{j}" for j in range(1, D+1)]
    df = (
        scores[["patient_id"] + zcols]
        .merge(temporal, on="patient_id", how="left", validate="one_to_one")
        .merge(height, on="patient_id", how="left", validate="one_to_one")
    )
    if len(df) != EXPECTED_N:
        raise SystemExit(f"FAIL merged n={len(df)} expected={EXPECTED_N}")

    df["age"] = pd.to_numeric(df["age_years_capped90"], errors="coerce")
    df["female"] = female_from_gender(df["gender"])
    df["height_cm"] = pd.to_numeric(df["height_median_cm"], errors="coerce")
    if df["age"].isna().any():
        raise SystemExit("FAIL age missing")

    Z = df[zcols].to_numpy(float)
    if not np.all(np.isfinite(Z)):
        raise SystemExit("FAIL nonfinite B8 scores")

    X_full = np.column_stack([
        df["age"].to_numpy(float),
        df["female"].to_numpy(float),
    ])

    # Replay observed Stage7B-NL statistic before any artifact-control readout.
    folds_full, plans_full = prepare_fold_design(df, ["age"])
    pred_full = oof_predict_from_plans(Z, plans_full)
    replay_full = distance_correlation(X_full, Z - pred_full)
    if abs(replay_full - EXPECTED_FULL_DCOR) > REPLAY_DCOR_TOL:
        raise SystemExit(
            f"FAIL full dCor replay mismatch: {replay_full:.9f} "
            f"vs {EXPECTED_FULL_DCOR:.9f}"
        )

    hdf = df[
        df["height_cm"].notna()
        & (df["height_cm"] > 100)
        & (df["height_cm"] <= 250)
    ].copy()
    if len(hdf) != EXPECTED_HEIGHT_N:
        raise SystemExit(f"FAIL height subset n={len(hdf)} expected={EXPECTED_HEIGHT_N}")
    Zh = hdf[zcols].to_numpy(float)
    X_height = np.column_stack([
        hdf["age"].to_numpy(float),
        hdf["female"].to_numpy(float),
        hdf["height_cm"].to_numpy(float),
    ])

    folds_h, plans_h = prepare_fold_design(hdf, ["age","height_cm"])
    pred_h = oof_predict_from_plans(Zh, plans_h)
    replay_h = distance_correlation(X_height, Zh - pred_h)
    if abs(replay_h - EXPECTED_HEIGHT_DCOR) > REPLAY_DCOR_TOL:
        raise SystemExit(
            f"FAIL height dCor replay mismatch: {replay_h:.9f} "
            f"vs {EXPECTED_HEIGHT_DCOR:.9f}"
        )

    print("Observed dCor replay integrity: PASS", flush=True)

    full_pipeline = pipeline_replay_null(
        df, Z, X_full, ["age"],
        PIPELINE_PERMUTATIONS, PERM_SEED, "full978",
    )
    height_pipeline = pipeline_replay_null(
        hdf, Zh, X_height, ["age","height_cm"],
        PIPELINE_PERMUTATIONS, PERM_SEED + 1, "height693",
    )

    full_blocked = blocked_fold_null(
        df, Z, X_full, ["age"],
        BLOCKED_PERMUTATIONS, PERM_SEED + 2, "full978",
    )
    height_blocked = blocked_fold_null(
        hdf, Zh, X_height, ["age","height_cm"],
        BLOCKED_PERMUTATIONS, PERM_SEED + 3, "height693",
    )

    holm_full, holm_height = holm_two(
        full_pipeline["pipeline_null_p_plus_one"],
        height_pipeline["pipeline_null_p_plus_one"],
    )
    full_pipeline["holm_p_across_two_primary_pipeline_controls"] = holm_full
    height_pipeline["holm_p_across_two_primary_pipeline_controls"] = holm_height

    primary_flags = {
        "full978": bool(holm_full < 0.05),
        "height693": bool(holm_height < 0.05),
    }
    blocked_flags = {
        "full978": bool(full_blocked["within_fold_null_p_plus_one"] < 0.05),
        "height693": bool(height_blocked["within_fold_null_p_plus_one"] < 0.05),
    }

    if not any(primary_flags.values()):
        decision = "RESIDUAL_DEPENDENCE_COMPATIBLE_WITH_PIPELINE_RESIDUALIZATION_NULL"
    elif (
        all(primary_flags.values())
        and all(blocked_flags.values())
    ):
        decision = "RESIDUAL_DEPENDENCE_PERSISTS_AFTER_PIPELINE_AND_FOLD_CONTROLS"
    else:
        decision = "RESIDUAL_DEPENDENCE_PERSISTS_IN_PIPELINE_CONTROL_BUT_IS_NOT_UNIFORMLY_ROBUST"

    summary_rows = [
        {
            "cohort": "full978",
            "observed_dcor": replay_full,
            **full_pipeline,
            **full_blocked,
            "pipeline_holm_flag_0.05": primary_flags["full978"],
            "within_fold_diagnostic_flag_0.05": blocked_flags["full978"],
        },
        {
            "cohort": "height693",
            "observed_dcor": replay_h,
            **height_pipeline,
            **height_blocked,
            "pipeline_holm_flag_0.05": primary_flags["height693"],
            "within_fold_diagnostic_flag_0.05": blocked_flags["height693"],
        },
    ]
    pd.DataFrame(summary_rows).to_csv(
        out / "wfp_stage7b_rd_adjudication_summary.csv",
        index=False,
    )

    result = {
        "decision": decision,
        "scientific_role": "POST_STAGE7B_NL_RESIDUAL_DEPENDENCE_ARTIFACT_ADJUDICATION",
        "trigger_observed_before_RD_freeze": True,
        "trigger": "Stage7B-NL residual distance correlation flag",
        "existing_stage7b_modified": False,
        "existing_stage7b_nl_modified": False,
        "frozen_B8_changed": False,
        "full978": {
            "pipeline_replay": full_pipeline,
            "within_fold_diagnostic": full_blocked,
        },
        "height693": {
            "pipeline_replay": height_pipeline,
            "within_fold_diagnostic": height_blocked,
        },
        "interpretation_boundary": [
            "Pipeline-replay residual permutation is an artifact-control sensitivity under residual-vector exchangeability, not an exact distribution-free test.",
            "A persistent signal is not automatically nonlinear conditional-mean dependence.",
            "Persistent residual dependence may reflect conditional variance/covariance or other higher-order dependence.",
            "No variance/covariance/distributional model is fit or automatically authorized.",
            "No Ztrait/Zstate interpretation is authorized.",
            "Stage7B and Stage7B-NL remain unchanged.",
        ],
        "hashes": {
            "frozen_spec_sha256": sha256_file(specp),
            "analysis_script_sha256": sha256_file(scriptp),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "WFP_STAGE7B_RD_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "WF-P STAGE 7B-RD — RESIDUAL DEPENDENCE ADJUDICATION",
        "====================================================",
        f"Decision: {decision}",
        "Scientific role: POST-STAGE7B-NL ARTIFACT-CONTROL AUDIT",
        "Trigger observed before RD freeze: YES",
        "Existing Stage7B modified: NO",
        "Existing Stage7B-NL modified: NO",
        "Frozen B8 changed: NO",
        "",
        "Full cohort n=978:",
        f"  observed residual dCor replay: {replay_full:.6f}",
        f"  pipeline-replay null median/q95: "
        f"{full_pipeline['pipeline_null_median']:.6f} / "
        f"{full_pipeline['pipeline_null_q95']:.6f}",
        f"  pipeline-replay plus-one p: "
        f"{full_pipeline['pipeline_null_p_plus_one']:.6g}",
        f"  pipeline-replay Holm p: {holm_full:.6g}",
        f"  within-fold null median/q95: "
        f"{full_blocked['within_fold_null_median']:.6f} / "
        f"{full_blocked['within_fold_null_q95']:.6f}",
        f"  within-fold diagnostic p: "
        f"{full_blocked['within_fold_null_p_plus_one']:.6g}",
        "",
        "Height complete-case n=693:",
        f"  observed residual dCor replay: {replay_h:.6f}",
        f"  pipeline-replay null median/q95: "
        f"{height_pipeline['pipeline_null_median']:.6f} / "
        f"{height_pipeline['pipeline_null_q95']:.6f}",
        f"  pipeline-replay plus-one p: "
        f"{height_pipeline['pipeline_null_p_plus_one']:.6g}",
        f"  pipeline-replay Holm p: {holm_height:.6g}",
        f"  within-fold null median/q95: "
        f"{height_blocked['within_fold_null_median']:.6f} / "
        f"{height_blocked['within_fold_null_q95']:.6f}",
        f"  within-fold diagnostic p: "
        f"{height_blocked['within_fold_null_p_plus_one']:.6g}",
        "",
        "Boundary:",
        "  Pipeline replay is an artifact-control sensitivity, not an exact distribution-free test.",
        "  Persistent residual dCor is not automatically a nonlinear-mean effect.",
        "  No variance/distributional model is fit here.",
        "  Do not proceed automatically to variance/covariance modeling.",
        "  No Ztrait/Zstate interpretation is authorized.",
        "",
    ]
    (out / "WFP_STAGE7B_RD_READOUT.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
