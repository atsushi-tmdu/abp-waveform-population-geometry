#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7B-NL — Nonlinear Constitutional Sensitivity Audit
==============================================================

Separate post-Stage7B sensitivity analysis. The prospectively frozen Stage7B
analysis remains unchanged.

Full cohort (n=978)
-------------------
M0:  age + sex                                  [linear replay]
M1:  RCS4(age) + sex                            [smooth age]
M2:  RCS4(age) * sex                            [limited interaction]

Height complete-case cohort (n=693)
-----------------------------------
H0:   age + sex                                 [linear replay]
H0h:  age + sex + height                        [linear height replay]
H1:   RCS4(age) + sex
H1h:  RCS4(age) + sex + height
H2:   RCS4(age) + sex + RCS4(height)
H3:   H2 + sex * linear(height)                 [1-df interaction]

RCS4 uses four training-fold knots at 5%, 35%, 65%, and 95%, yielding
three spline basis columns (one linear plus two nonlinear). Test-fold data
never select knots.

Primary interpretive quantity
-----------------------------
Aggregate multivariate patient-level 5-fold OOF R^2 and prespecified delta R^2.

Model-free residual-dependence sensitivity
------------------------------------------
Biased sample distance correlation between constitutional predictors and
OOF residual B8 coordinates after the matched linear model:
- full cohort: X=(age, sex), residuals after M0
- height subset: X=(age, sex, height), residuals after H0h

999 patient-level permutations with plus-one p-values; Holm adjustment across
the two sensitivity tests. A positive result is not specific to nonlinear mean
dependence and may reflect variance or higher-order dependence.

No random forest, neural network, boosting, kernel-ridge model search,
hyperparameter search, additional interactions, disease/treatment/outcome
variables, or distributional modeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_N = 978
EXPECTED_HEIGHT_N = 693
D = 8

CV_FOLDS = 5
CV_SEED = 20260820

KNOT_PROBS = np.array([0.05, 0.35, 0.65, 0.95], dtype=float)

DCOR_PERMUTATIONS = 999
DCOR_SEED = 20260824

REPLAY_TOL = 1e-9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_fold(pid: str) -> int:
    key = f"{CV_SEED}:{pid}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % CV_FOLDS


def rcs_parameters(x_train: np.ndarray) -> Dict[str, object]:
    x = np.asarray(x_train, float)
    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite values in spline training data")
    knots = np.quantile(x, KNOT_PROBS)
    if np.any(np.diff(knots) <= 1e-10):
        raise ValueError(f"non-distinct RCS knots: {knots.tolist()}")
    span = float(knots[-1] - knots[0])
    if span <= 1e-12:
        raise ValueError("degenerate RCS knot span")
    return {"knots": knots, "span": span}


def _tp3(x: np.ndarray, knot: float) -> np.ndarray:
    return np.maximum(np.asarray(x, float) - float(knot), 0.0) ** 3


def rcs_basis(x: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    """
    Four-knot restricted cubic spline basis with total 3 df:
      [x, nonlinear_1, nonlinear_2].

    Scaling does not change the column span and is applied only for numerical
    stability; all final design columns are then standardized within fold.
    """
    x = np.asarray(x, float)
    k = np.asarray(params["knots"], float)
    span = float(params["span"])

    k_penultimate = float(k[-2])
    k_last = float(k[-1])
    denominator = k_last - k_penultimate

    cols = [x]
    for j in range(len(k) - 2):
        kj = float(k[j])
        nonlinear = (
            _tp3(x, kj)
            - _tp3(x, k_penultimate) * (k_last - kj) / denominator
            + _tp3(x, k_last) * (k_penultimate - kj) / denominator
        ) / (span ** 3)
        cols.append(nonlinear)

    return np.column_stack(cols)


def standardize_design(
    X_train_raw: np.ndarray, X_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X_train_raw, axis=0)
    sd = np.std(X_train_raw, axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X_train_raw - mu) / sd, (X_test_raw - mu) / sd


def raw_design_for_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    age_tr = train["age"].to_numpy(float)
    age_te = test["age"].to_numpy(float)
    sex_tr = train["female"].to_numpy(float)
    sex_te = test["female"].to_numpy(float)

    meta: Dict[str, object] = {"model": model}

    if model in {"M0", "H0"}:
        return (
            np.column_stack([age_tr, sex_tr]),
            np.column_stack([age_te, sex_te]),
            meta,
        )

    if model == "H0h":
        htr = train["height_cm"].to_numpy(float)
        hte = test["height_cm"].to_numpy(float)
        return (
            np.column_stack([age_tr, sex_tr, htr]),
            np.column_stack([age_te, sex_te, hte]),
            meta,
        )

    age_params = rcs_parameters(age_tr)
    age_basis_tr = rcs_basis(age_tr, age_params)
    age_basis_te = rcs_basis(age_te, age_params)
    meta["age_knots"] = np.asarray(age_params["knots"], float).tolist()

    if model in {"M1", "H1"}:
        return (
            np.column_stack([age_basis_tr, sex_tr]),
            np.column_stack([age_basis_te, sex_te]),
            meta,
        )

    if model == "M2":
        # RCS(age) * sex: age smooth, sex main effect, and all three
        # age-basis-by-sex interaction columns.
        return (
            np.column_stack([
                age_basis_tr,
                sex_tr,
                age_basis_tr * sex_tr[:, None],
            ]),
            np.column_stack([
                age_basis_te,
                sex_te,
                age_basis_te * sex_te[:, None],
            ]),
            meta,
        )

    htr = train["height_cm"].to_numpy(float)
    hte = test["height_cm"].to_numpy(float)

    if model == "H1h":
        return (
            np.column_stack([age_basis_tr, sex_tr, htr]),
            np.column_stack([age_basis_te, sex_te, hte]),
            meta,
        )

    height_params = rcs_parameters(htr)
    height_basis_tr = rcs_basis(htr, height_params)
    height_basis_te = rcs_basis(hte, height_params)
    meta["height_knots"] = np.asarray(height_params["knots"], float).tolist()

    if model == "H2":
        return (
            np.column_stack([age_basis_tr, sex_tr, height_basis_tr]),
            np.column_stack([age_basis_te, sex_te, height_basis_te]),
            meta,
        )

    if model == "H3":
        # Exactly one secondary interaction df: sex x linear height.
        return (
            np.column_stack([
                age_basis_tr,
                sex_tr,
                height_basis_tr,
                sex_tr * htr,
            ]),
            np.column_stack([
                age_basis_te,
                sex_te,
                height_basis_te,
                sex_te * hte,
            ]),
            meta,
        )

    raise ValueError(f"unknown model: {model}")


def oof_predict(
    df: pd.DataFrame,
    Z: np.ndarray,
    model: str,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    folds = np.asarray(
        [stable_fold(pid) for pid in df["patient_id"].astype(str)],
        dtype=int,
    )
    pred = np.full_like(Z, np.nan, dtype=float)
    metadata: List[Dict[str, object]] = []

    for fold in range(CV_FOLDS):
        test_mask = folds == fold
        train_mask = ~test_mask
        if int(np.sum(test_mask)) == 0 or int(np.sum(train_mask)) == 0:
            raise RuntimeError(f"empty CV fold: {fold}")

        train = df.loc[train_mask]
        test = df.loc[test_mask]
        Xtr_raw, Xte_raw, meta = raw_design_for_fold(train, test, model)
        Xtr, Xte = standardize_design(Xtr_raw, Xte_raw)

        Atr = np.column_stack([np.ones(len(Xtr)), Xtr])
        Ate = np.column_stack([np.ones(len(Xte)), Xte])

        beta = np.linalg.lstsq(Atr, Z[train_mask], rcond=None)[0]
        pred[test_mask] = Ate @ beta

        metadata.append({
            "model": model,
            "fold": int(fold),
            "train_n": int(np.sum(train_mask)),
            "test_n": int(np.sum(test_mask)),
            **meta,
        })

    if not np.all(np.isfinite(pred)):
        raise RuntimeError(f"nonfinite OOF predictions for model {model}")
    return pred, metadata


def aggregate_r2(Z: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((Z - pred) ** 2))
    centered = Z - np.mean(Z, axis=0, keepdims=True)
    sst = float(np.sum(centered ** 2))
    return float(1.0 - sse / sst)


def axis_r2(Z: np.ndarray, pred: np.ndarray) -> np.ndarray:
    values = []
    for j in range(Z.shape[1]):
        y = Z[:, j]
        denominator = float(np.sum((y - np.mean(y)) ** 2))
        values.append(
            1.0 - float(np.sum((y - pred[:, j]) ** 2)) / denominator
        )
    return np.asarray(values, float)


def standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X - mu) / sd


def pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    sq = np.sum(X * X, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def double_center(Dm: np.ndarray) -> np.ndarray:
    return (
        Dm
        - Dm.mean(axis=1, keepdims=True)
        - Dm.mean(axis=0, keepdims=True)
        + Dm.mean()
    )


def distance_correlation_from_centered(
    A: np.ndarray, B: np.ndarray
) -> float:
    dcov2 = float(np.mean(A * B))
    dvarx2 = float(np.mean(A * A))
    dvary2 = float(np.mean(B * B))
    if dvarx2 <= 0.0 or dvary2 <= 0.0:
        return 0.0

    # Biased sample distance correlation. Tiny negative dcov2 values can arise
    # from floating-point arithmetic and are truncated at zero.
    dcor2 = max(dcov2, 0.0) / math.sqrt(dvarx2 * dvary2)
    return float(math.sqrt(max(dcor2, 0.0)))


def distance_correlation_permutation(
    X: np.ndarray,
    Y: np.ndarray,
    permutations: int,
    seed: int,
    label: str,
) -> Dict[str, float]:
    Xs = standardize_columns(X)
    Ys = standardize_columns(Y)

    A = double_center(pairwise_distance_matrix(Xs))
    B = double_center(pairwise_distance_matrix(Ys))

    observed = distance_correlation_from_centered(A, B)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)

    n = len(Xs)
    for b in range(permutations):
        perm = rng.permutation(n)
        Bp = B[perm][:, perm]
        null[b] = distance_correlation_from_centered(A, Bp)

        if permutations >= 100 and (
            (b + 1) % 100 == 0 or b + 1 == permutations
        ):
            print(
                f"[distance-correlation {label}] "
                f"{b+1}/{permutations}",
                flush=True,
            )

    pvalue = (
        1.0 + float(np.sum(null >= observed - 1e-15))
    ) / (permutations + 1.0)

    return {
        "dcor": float(observed),
        "permutation_p_plus_one": float(pvalue),
        "null_median": float(np.median(null)),
        "null_q95": float(np.percentile(null, 95)),
        "observed_minus_null_q95": float(
            observed - np.percentile(null, 95)
        ),
        "permutations": int(permutations),
    }


def holm_adjust_two(p1: float, p2: float) -> Tuple[float, float]:
    p = np.asarray([p1, p2], float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(2, float)

    first = min(1.0, 2.0 * p[order[0]])
    second = min(1.0, max(first, p[order[1]]))
    adjusted_sorted[0] = first
    adjusted_sorted[1] = second

    adjusted = np.empty(2, float)
    adjusted[order[0]] = adjusted_sorted[0]
    adjusted[order[1]] = adjusted_sorted[1]
    return float(adjusted[0]), float(adjusted[1])


def classify_delta(delta: float) -> str:
    if delta >= 0.03:
        return "MATERIAL_GE_0.03"
    if delta >= 0.01:
        return "MODEST_0.01_TO_LT_0.03"
    return "NO_MATERIAL_GAIN_LT_0.01"


def flatten_fold_metadata(
    rows: List[Dict[str, object]]
) -> pd.DataFrame:
    flat = []
    for row in rows:
        item = dict(row)
        for key in ["age_knots", "height_knots"]:
            if key in item:
                values = item.pop(key)
                for idx, value in enumerate(values, start=1):
                    item[f"{key}_k{idx}"] = float(value)
        flat.append(item)
    return pd.DataFrame(flat)


def self_test() -> int:
    rng = np.random.default_rng(20260824)
    n = 360

    age = rng.uniform(20.0, 90.0, n)
    female = rng.integers(0, 2, n).astype(float)
    height = rng.normal(170.0, 9.0, n)

    df = pd.DataFrame({
        "patient_id": [f"p{i:05d}" for i in range(n)],
        "age": age,
        "female": female,
        "height_cm": height,
    })

    # Strong smooth nonlinear signal to test the spline/CV machinery.
    nonlinear = ((age - 58.0) / 20.0) ** 2
    Z = rng.normal(scale=0.45, size=(n, D))
    Z[:, 0] += 1.1 * nonlinear
    Z[:, 1] -= 0.7 * nonlinear

    pred0, _ = oof_predict(df, Z, "M0")
    pred1, _ = oof_predict(df, Z, "M1")
    if aggregate_r2(Z, pred1) <= aggregate_r2(Z, pred0) + 0.05:
        raise RuntimeError(
            "RCS OOF self-test failed: "
            f"delta={aggregate_r2(Z,pred1)-aggregate_r2(Z,pred0):.6f}"
        )

    # Clearly dependent toy pair for distance-correlation machinery.
    X = rng.normal(size=(120, 2))
    Y = np.column_stack([X[:, 0] ** 2, X[:, 1]]) + rng.normal(
        scale=0.04, size=(120, 2)
    )
    dcor = distance_correlation_permutation(
        X, Y, permutations=39, seed=1234, label="self-test"
    )
    if dcor["dcor"] <= dcor["null_q95"]:
        raise RuntimeError("distance-correlation self-test failed")

    print("WF-P Stage7B-NL analysis self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file")
    ap.add_argument("--temporal-linkage")
    ap.add_argument("--height-preflight")
    ap.add_argument("--stage7b-spec")
    ap.add_argument("--stage7b-results-json")
    ap.add_argument("--stage7b-readout")
    ap.add_argument("--preflight-json")
    ap.add_argument("--spec")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    required = [
        "score_file",
        "temporal_linkage",
        "height_preflight",
        "stage7b_spec",
        "stage7b_results_json",
        "stage7b_readout",
        "preflight_json",
        "spec",
        "out",
    ]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {missing}")

    script_path = Path(__file__).resolve()
    score_path = Path(args.score_file).expanduser().resolve()
    temporal_path = Path(args.temporal_linkage).expanduser().resolve()
    height_path = Path(args.height_preflight).expanduser().resolve()
    stage7b_spec_path = Path(args.stage7b_spec).expanduser().resolve()
    stage7b_result_path = Path(
        args.stage7b_results_json
    ).expanduser().resolve()
    stage7b_readout_path = Path(args.stage7b_readout).expanduser().resolve()
    preflight_path = Path(args.preflight_json).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(spec_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_STAGE7B_NL_EFFECT_ACCESS":
        raise SystemExit("FAIL: invalid Stage7B-NL frozen status")
    if frozen.get("analysis_script_sha256") != sha256_file(script_path):
        raise SystemExit("FAIL: Stage7B-NL analysis-script hash mismatch")

    input_files = {
        "score_file_sha256": score_path,
        "temporal_linkage_sha256": temporal_path,
        "height_preflight_sha256": height_path,
        "stage7b_spec_sha256": stage7b_spec_path,
        "stage7b_results_sha256": stage7b_result_path,
        "stage7b_readout_sha256": stage7b_readout_path,
        "preflight_json_sha256": preflight_path,
    }
    for key, path in input_files.items():
        if frozen.get(key) != sha256_file(path):
            raise SystemExit(f"FAIL: frozen input hash mismatch: {key}")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("decision") != "WFP_STAGE7B_NL_PREFLIGHT_PASS":
        raise SystemExit("FAIL: Stage7B-NL preflight did not PASS")
    height_column = str(preflight.get("height_column_detected"))
    if height_column != "height_median_cm":
        raise SystemExit(
            f"FAIL: expected frozen height_median_cm, found {height_column}"
        )

    stage7b = json.loads(
        stage7b_result_path.read_text(encoding="utf-8")
    )
    if stage7b.get("decision") != "WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE":
        raise SystemExit("FAIL: Stage7B result is not complete")

    reference = frozen["stage7b_reference_oof_r2"]

    scores = pd.read_csv(score_path, dtype={"patient_id": str})
    temporal = pd.read_csv(
        temporal_path,
        usecols=["patient_id", "age_years_capped90", "gender"],
        dtype={"patient_id": str},
    )
    height = pd.read_csv(height_path, dtype={"patient_id": str})

    zcols = [f"z{j}" for j in range(1, D + 1)]
    absent = [
        c for c in ["patient_id"] + zcols if c not in scores.columns
    ]
    if absent:
        raise SystemExit(f"FAIL: score columns absent: {absent}")
    if height_column not in height.columns:
        raise SystemExit(
            f"FAIL: height column absent from height preflight: {height_column}"
        )

    df = (
        scores[["patient_id"] + zcols]
        .merge(
            temporal,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            height[["patient_id", height_column]],
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
    )

    if len(df) != EXPECTED_N:
        raise SystemExit(
            f"FAIL: merged full cohort n={len(df)}, expected={EXPECTED_N}"
        )

    df["age"] = pd.to_numeric(
        df["age_years_capped90"], errors="coerce"
    )
    gender = df["gender"].astype(str).str.upper().str.strip()
    if (~gender.isin(["M", "F"])).any():
        raise SystemExit("FAIL: unexpected gender coding")
    df["female"] = (gender == "F").astype(float)
    df["height_cm"] = pd.to_numeric(
        df[height_column], errors="coerce"
    )

    if df["age"].isna().any():
        raise SystemExit("FAIL: age missing in full cohort")

    Z = df[zcols].to_numpy(float)
    if not np.all(np.isfinite(Z)):
        raise SystemExit("FAIL: nonfinite B8 scores")

    model_rows = []
    fold_metadata: List[Dict[str, object]] = []
    full_predictions: Dict[str, np.ndarray] = {}

    for model in ["M0", "M1", "M2"]:
        pred, meta = oof_predict(df, Z, model)
        full_predictions[model] = pred
        fold_metadata.extend(meta)
        model_rows.append({
            "cohort": "full978",
            "model": model,
            "n": len(df),
            "oof_r2": aggregate_r2(Z, pred),
        })

    hdf = df[df["height_cm"].notna()].copy()
    if len(hdf) != EXPECTED_HEIGHT_N:
        raise SystemExit(
            f"FAIL: height complete-case n={len(hdf)}, "
            f"expected={EXPECTED_HEIGHT_N}"
        )
    Zh = hdf[zcols].to_numpy(float)

    height_predictions: Dict[str, np.ndarray] = {}
    for model in ["H0", "H0h", "H1", "H1h", "H2", "H3"]:
        pred, meta = oof_predict(hdf, Zh, model)
        height_predictions[model] = pred
        fold_metadata.extend(meta)
        model_rows.append({
            "cohort": "height693",
            "model": model,
            "n": len(hdf),
            "oof_r2": aggregate_r2(Zh, pred),
        })

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(
        out / "wfp_stage7b_nl_model_summary.csv", index=False
    )

    r2 = {
        (row["cohort"], row["model"]): float(row["oof_r2"])
        for row in model_rows
    }

    replay_checks = {
        "full_M0": (
            r2[("full978", "M0")],
            float(reference["full_linear_age_sex"]),
        ),
        "height_H0": (
            r2[("height693", "H0")],
            float(reference["height_subset_linear_age_sex"]),
        ),
        "height_H0h": (
            r2[("height693", "H0h")],
            float(reference["height_subset_linear_age_sex_height"]),
        ),
    }
    for name, (observed, expected) in replay_checks.items():
        if abs(observed - expected) > REPLAY_TOL:
            raise SystemExit(
                f"FAIL: Stage7B replay mismatch {name}: "
                f"observed={observed:.12f}; expected={expected:.12f}; "
                f"abs_diff={abs(observed-expected):.3e}"
            )

    # Axis-specific results are frozen as descriptive only.
    axis_rows = []
    for model, pred in full_predictions.items():
        for axis, value in enumerate(axis_r2(Z, pred), start=1):
            axis_rows.append({
                "cohort": "full978",
                "model": model,
                "axis": axis,
                "oof_r2_DESCRIPTIVE_ONLY": float(value),
            })
    for model, pred in height_predictions.items():
        for axis, value in enumerate(axis_r2(Zh, pred), start=1):
            axis_rows.append({
                "cohort": "height693",
                "model": model,
                "axis": axis,
                "oof_r2_DESCRIPTIVE_ONLY": float(value),
            })
    pd.DataFrame(axis_rows).to_csv(
        out / "wfp_stage7b_nl_axis_r2_DESCRIPTIVE.csv",
        index=False,
    )

    flatten_fold_metadata(fold_metadata).to_csv(
        out / "wfp_stage7b_nl_fold_spline_parameters.csv",
        index=False,
    )

    # Model-free dependence remaining after the corresponding linear OOF model.
    X_full = np.column_stack([
        df["age"].to_numpy(float),
        df["female"].to_numpy(float),
    ])
    residual_full = Z - full_predictions["M0"]
    dcor_full = distance_correlation_permutation(
        X_full,
        residual_full,
        permutations=DCOR_PERMUTATIONS,
        seed=DCOR_SEED,
        label="full978",
    )

    X_height = np.column_stack([
        hdf["age"].to_numpy(float),
        hdf["female"].to_numpy(float),
        hdf["height_cm"].to_numpy(float),
    ])
    residual_height = Zh - height_predictions["H0h"]
    dcor_height = distance_correlation_permutation(
        X_height,
        residual_height,
        permutations=DCOR_PERMUTATIONS,
        seed=DCOR_SEED + 1,
        label="height693",
    )

    holm_full, holm_height = holm_adjust_two(
        dcor_full["permutation_p_plus_one"],
        dcor_height["permutation_p_plus_one"],
    )
    dcor_full["holm_p_across_two_sensitivities"] = holm_full
    dcor_height["holm_p_across_two_sensitivities"] = holm_height

    pd.DataFrame([
        {
            "cohort": "full978",
            "X": "age+sex",
            "Y": "OOF residual B8 after M0",
            **dcor_full,
        },
        {
            "cohort": "height693",
            "X": "age+sex+height",
            "Y": "OOF residual B8 after H0h",
            **dcor_height,
        },
    ]).to_csv(
        out / "wfp_stage7b_nl_distance_correlation_sensitivity.csv",
        index=False,
    )

    M0 = r2[("full978", "M0")]
    M1 = r2[("full978", "M1")]
    M2 = r2[("full978", "M2")]

    H0 = r2[("height693", "H0")]
    H0h = r2[("height693", "H0h")]
    H1 = r2[("height693", "H1")]
    H1h = r2[("height693", "H1h")]
    H2 = r2[("height693", "H2")]
    H3 = r2[("height693", "H3")]

    deltas = {
        "M1_minus_M0_age_smooth": float(M1 - M0),
        "M2_minus_M1_age_by_sex_interaction": float(M2 - M1),
        "M2_minus_M0_total_age_smooth_plus_interaction": float(M2 - M0),
        "H0h_minus_H0_linear_height_replay": float(H0h - H0),
        "H1_minus_H0_age_smooth_height_subset": float(H1 - H0),
        "H1h_minus_H1_linear_height_after_smooth_age": float(H1h - H1),
        "H2_minus_H1h_nonlinear_height_increment": float(H2 - H1h),
        "H3_minus_H2_height_by_sex_interaction": float(H3 - H2),
    }

    key_deltas = {
        "age_smooth": deltas["M1_minus_M0_age_smooth"],
        "age_smooth_plus_interaction":
            deltas["M2_minus_M0_total_age_smooth_plus_interaction"],
        "nonlinear_height":
            deltas["H2_minus_H1h_nonlinear_height_increment"],
        "height_by_sex":
            deltas["H3_minus_H2_height_by_sex_interaction"],
    }
    interpretation_bands = {
        key: classify_delta(value) for key, value in key_deltas.items()
    }
    max_key_delta = max(key_deltas.values())

    residual_dependence_flag = (
        dcor_full["holm_p_across_two_sensitivities"] < 0.05
        or dcor_height["holm_p_across_two_sensitivities"] < 0.05
    )

    if max_key_delta >= 0.03:
        decision = (
            "MATERIAL_NONLINEAR_CONSTITUTIONAL_MEAN_SIGNAL_"
            "REINTERPRET_LINEAR_STAGE7B"
        )
    elif max_key_delta >= 0.01:
        decision = (
            "MODEST_NONLINEAR_CONSTITUTIONAL_INFORMATION_"
            "NO_STRONG_REVERSAL"
        )
    elif residual_dependence_flag:
        decision = (
            "NO_MATERIAL_OOF_MEAN_GAIN_BUT_"
            "RESIDUAL_MODEL_FREE_DEPENDENCE_FLAG"
        )
    else:
        decision = (
            "NONLINEAR_AUDIT_SUPPORTS_"
            "WEAK_CONSTITUTIONAL_PREDICTABILITY"
        )

    result = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "7B-NL",
        "decision": decision,
        "scientific_role":
            "POST_STAGE7B_NONLINEAR_CONSTITUTIONAL_SENSITIVITY",
        "full_cohort_n": EXPECTED_N,
        "height_complete_case_n": EXPECTED_HEIGHT_N,
        "existing_stage7b_modified": False,
        "frozen_B8_changed": False,
        "primary_interpretive_quantity":
            "aggregate patient-level 5-fold OOF R2",
        "models": {
            "M0": M0, "M1": M1, "M2": M2,
            "H0": H0, "H0h": H0h, "H1": H1,
            "H1h": H1h, "H2": H2, "H3": H3,
        },
        "deltas": deltas,
        "key_delta_interpretation_bands": interpretation_bands,
        "maximum_prespecified_key_delta_oof_r2":
            float(max_key_delta),
        "distance_correlation": {
            "full978": dcor_full,
            "height693": dcor_height,
            "interpretation":
                "Residual dCor is not specific to nonlinear mean dependence; "
                "it may reflect variance or higher-order dependence.",
        },
        "boundary": [
            "Existing Stage7B remains the prospectively frozen primary analysis.",
            "Stage7B-NL is a separate post-Stage7B sensitivity audit.",
            "No model or knot choice is selected after effect access.",
            "Axis-specific results are descriptive only.",
            "Residual distance correlation is not proof of nonlinear mean association.",
            "No variance/distributional extension is automatically authorized.",
            "No random forest, neural network, boosting, kernel-ridge, or hyperparameter search is used.",
            "Age above 89 is top-coded at 90.",
            "No Ztrait/Zstate claim is authorized.",
        ],
        "hashes": {
            "frozen_spec_sha256": sha256_file(spec_path),
            "analysis_script_sha256": sha256_file(script_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    (out / "WFP_STAGE7B_NL_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "WF-P STAGE 7B-NL — NONLINEAR CONSTITUTIONAL SENSITIVITY",
        "========================================================",
        f"Decision: {decision}",
        f"Full cohort n: {EXPECTED_N}",
        f"Height complete-case n: {EXPECTED_HEIGHT_N}",
        "Existing Stage7B modified: NO",
        "Frozen B8 changed: NO",
        "",
        "Full cohort — age/sex:",
        f"  M0 linear age + sex OOF R2: {M0:.6f}",
        f"  M1 RCS4(age) + sex OOF R2: {M1:.6f}",
        f"  delta M1-M0: {M1-M0:.6f} "
        f"[{classify_delta(M1-M0)}]",
        f"  M2 RCS4(age) * sex OOF R2: {M2:.6f}",
        f"  delta M2-M1 interaction: {M2-M1:.6f}",
        f"  delta M2-M0 total nonlinear+interaction: {M2-M0:.6f} "
        f"[{classify_delta(M2-M0)}]",
        "",
        "Height complete-case cohort:",
        f"  H0 linear age + sex OOF R2: {H0:.6f}",
        f"  H0h linear age + sex + height OOF R2: {H0h:.6f}",
        f"  linear height replay delta H0h-H0: {H0h-H0:.6f}",
        f"  H1 RCS4(age) + sex OOF R2: {H1:.6f}",
        f"  age smooth delta H1-H0: {H1-H0:.6f}",
        f"  H1h H1 + linear height OOF R2: {H1h:.6f}",
        f"  linear height after smooth age H1h-H1: {H1h-H1:.6f}",
        f"  H2 RCS4(age) + sex + RCS4(height) OOF R2: {H2:.6f}",
        f"  nonlinear height increment H2-H1h: {H2-H1h:.6f} "
        f"[{classify_delta(H2-H1h)}]",
        f"  H3 H2 + sex*linear(height) OOF R2: {H3:.6f}",
        f"  height-by-sex increment H3-H2: {H3-H2:.6f} "
        f"[{classify_delta(H3-H2)}]",
        "",
        "Model-free residual-dependence sensitivity:",
        f"  Full age+sex vs OOF residual B8: "
        f"dCor={dcor_full['dcor']:.6f}; "
        f"perm p={dcor_full['permutation_p_plus_one']:.6g}; "
        f"Holm p={dcor_full['holm_p_across_two_sensitivities']:.6g}; "
        f"null q95={dcor_full['null_q95']:.6f}",
        f"  Height age+sex+height vs OOF residual B8: "
        f"dCor={dcor_height['dcor']:.6f}; "
        f"perm p={dcor_height['permutation_p_plus_one']:.6g}; "
        f"Holm p={dcor_height['holm_p_across_two_sensitivities']:.6g}; "
        f"null q95={dcor_height['null_q95']:.6f}",
        "",
        "Pre-frozen effect-size interpretation bands:",
        "  delta OOF R2 <0.01: no material predictive improvement",
        "  0.01 to <0.03: modest nonlinear information",
        "  >=0.03: material nonlinear information requiring reinterpretation",
        "",
        "Boundary:",
        "  Stage7B primary results remain unchanged.",
        "  Do not state independence from age/sex/height.",
        "  Residual dCor is not specific to nonlinear conditional-mean effects.",
        "  Do not proceed automatically to variance/distributional modeling.",
        "  No Ztrait/Zstate interpretation is authorized.",
        "",
    ]

    (out / "WFP_STAGE7B_NL_READOUT.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
