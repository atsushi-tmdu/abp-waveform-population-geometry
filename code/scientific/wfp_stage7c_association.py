#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 7C — chronic phenotype association mapping
=====================================================

Primary scientific target:
Does the prespecified block of eight admission-coded chronic phenotypes add
out-of-fold information about the frozen B8 morphology coordinates beyond
age, sex, and WF-P0 conventional level/scale/timing factors?

Primary cohort:
887 frozen WF-P patients with an exact admission containing waveform start.

Primary baseline:
age (linear) + sex + level + log(scale) + log(duration)

Primary phenotype block:
- congestive heart failure
- cardiac arrhythmias
- valvular disease
- peripheral vascular disease
- hypertension
- diabetes
- renal failure
- chronic pulmonary disease

All eight phenotypes are entered jointly.

Primary effect:
Delta aggregate 5-fold patient OOF R2:
  full joint phenotype model - baseline model

Phenotype-specific key effects:
1) marginal Delta OOF R2: baseline+phenotype - baseline
2) unique Delta OOF R2: full joint model - joint model without phenotype
3) adjusted frozen-B8 coefficient-vector norm from the full joint model
4) Freedman-Lane-style reduced-model residual permutation global test using
   partial multivariate SSE improvement, with BH-FDR across exactly 8 tests.

Axis-specific coefficients are DESCRIPTIVE ONLY and are not a 64-test
confirmatory family.

Interpretation boundaries:
- admission diagnoses are discharge-coded cross-sectional phenotypes;
- association is not causal and not necessarily known at waveform time;
- frozen B8 is never relearned, rotated, or reselected;
- no Ztrait/Zstate claim;
- no disease-specific subspace is learned;
- prior-history mapping is deferred unless later explicitly justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_N = 887
D = 8
CV_FOLDS = 5
CV_SEED = 20260820
PERMUTATIONS = 2000
PERM_SEED = 20260823

PHENOTYPES = [
    "congestive_heart_failure",
    "cardiac_arrhythmias",
    "valvular_disease",
    "peripheral_vascular_disease",
    "hypertension",
    "diabetes",
    "renal_failure",
    "chronic_pulmonary_disease",
]

CONTINUOUS_BASELINE = [
    "age_years_capped90",
    "level_mmhg",
    "log_scale_sd",
    "log_duration_sec",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_fold(pid: str) -> int:
    key = f"{CV_SEED}:{pid}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % CV_FOLDS


def build_design(
    df: pd.DataFrame,
    phenotypes: List[str],
    *,
    continuous_mean: np.ndarray | None = None,
    continuous_sd: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Design excludes intercept. Continuous baseline is standardized; binary
    sex and phenotype indicators remain 0/1 so phenotype coefficients retain
    an adjusted shift interpretation.
    """
    Xc = df[CONTINUOUS_BASELINE].to_numpy(float)
    if continuous_mean is None:
        continuous_mean = np.mean(Xc, axis=0)
    if continuous_sd is None:
        continuous_sd = np.std(Xc, axis=0, ddof=0)
    continuous_sd = np.where(continuous_sd <= 1e-12, 1.0, continuous_sd)
    Xcs = (Xc - continuous_mean) / continuous_sd

    cols = [Xcs, df[["female"]].to_numpy(float)]
    if phenotypes:
        cols.append(df[phenotypes].to_numpy(float))
    X = np.column_stack(cols)
    return X, continuous_mean, continuous_sd


def fit_ols(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    pred = A @ beta
    resid = Y - pred
    sse = float(np.sum(resid * resid))
    return beta, pred, sse


def oof_predict(df: pd.DataFrame, Z: np.ndarray, phenotypes: List[str]) -> np.ndarray:
    folds = np.asarray([stable_fold(pid) for pid in df["patient_id"].astype(str)])
    pred = np.full_like(Z, np.nan, dtype=float)

    for f in range(CV_FOLDS):
        te = folds == f
        tr = ~te
        if int(np.sum(te)) == 0 or int(np.sum(tr)) == 0:
            raise RuntimeError(f"empty CV fold {f}")

        Xtr_raw = df.loc[tr, CONTINUOUS_BASELINE].to_numpy(float)
        mu = np.mean(Xtr_raw, axis=0)
        sd = np.std(Xtr_raw, axis=0, ddof=0)
        sd = np.where(sd <= 1e-12, 1.0, sd)

        Xtr, _, _ = build_design(
            df.loc[tr], phenotypes, continuous_mean=mu, continuous_sd=sd
        )
        Xte, _, _ = build_design(
            df.loc[te], phenotypes, continuous_mean=mu, continuous_sd=sd
        )

        Atr = np.column_stack([np.ones(np.sum(tr)), Xtr])
        Ate = np.column_stack([np.ones(np.sum(te)), Xte])
        beta = np.linalg.lstsq(Atr, Z[tr], rcond=None)[0]
        pred[te] = Ate @ beta

    if not np.all(np.isfinite(pred)):
        raise RuntimeError("nonfinite OOF predictions")
    return pred


def aggregate_r2(Z: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((Z - pred) ** 2))
    zc = Z - np.mean(Z, axis=0, keepdims=True)
    sst = float(np.sum(zc ** 2))
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(m, float)
    out[order] = q
    return out


def permutation_global_test(
    df: pd.DataFrame,
    Z: np.ndarray,
    phenotype: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Reduced-model residual permutation. Reduced model contains baseline and all
    other seven phenotypes. Full model adds the target phenotype.
    Test statistic is multivariate partial SSE improvement.
    """
    others = [p for p in PHENOTYPES if p != phenotype]

    Xred, mu, sd = build_design(df, others)
    Xfull, _, _ = build_design(
        df, PHENOTYPES, continuous_mean=mu, continuous_sd=sd
    )

    bred, fitted_red, sse_red = fit_ols(Xred, Z)
    bfull, fitted_full, sse_full = fit_ols(Xfull, Z)
    tobs = max(0.0, sse_red - sse_full)

    residual_red = Z - fitted_red
    exceed = 0

    Ared = np.column_stack([np.ones(len(Xred)), Xred])
    Afull = np.column_stack([np.ones(len(Xfull)), Xfull])

    for _ in range(PERMUTATIONS):
        perm = rng.permutation(len(df))
        Zb = fitted_red + residual_red[perm]

        bred_b = np.linalg.lstsq(Ared, Zb, rcond=None)[0]
        bfull_b = np.linalg.lstsq(Afull, Zb, rcond=None)[0]

        rred = Zb - Ared @ bred_b
        rfull = Zb - Afull @ bfull_b

        tb = float(np.sum(rred * rred) - np.sum(rfull * rfull))
        if tb >= tobs - 1e-12:
            exceed += 1

    p = (exceed + 1.0) / (PERMUTATIONS + 1.0)
    return {
        "partial_sse_improvement": float(tobs),
        "permutation_p": float(p),
        "permutations": PERMUTATIONS,
    }


def self_test() -> int:
    rng = np.random.default_rng(20260823)
    n = 400
    df = pd.DataFrame({
        "patient_id": [f"p{i:06d}" for i in range(n)],
        "age_years_capped90": rng.normal(65, 12, n),
        "level_mmhg": rng.normal(80, 12, n),
        "log_scale_sd": rng.normal(2.5, 0.2, n),
        "log_duration_sec": rng.normal(-0.1, 0.15, n),
        "female": rng.integers(0, 2, n),
    })
    for p in PHENOTYPES:
        df[p] = rng.binomial(1, 0.25, n)

    # Strong deterministic synthetic multivariate signal in phenotype 1.
    # The other seven phenotypes remain deliberately null.
    Z = rng.normal(scale=0.4, size=(n, D))
    signal = df[PHENOTYPES[0]].to_numpy(float)
    direction = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0])
    Z += signal[:, None] * direction[None, :]

    pb = oof_predict(df, Z, [])
    ps = oof_predict(df, Z, [PHENOTYPES[0]])
    pj = oof_predict(df, Z, PHENOTYPES)

    r2b = aggregate_r2(Z, pb)
    r2s = aggregate_r2(Z, ps)
    r2j = aggregate_r2(Z, pj)

    # Unit tests should be deterministic and comfortably separated from zero.
    # With this frozen synthetic construction, the signal phenotype alone and
    # the joint block must both materially improve OOF R2 over baseline.
    if (r2s - r2b) <= 0.15:
        raise RuntimeError(
            f"OOF signal-phenotype self-test failed: delta={r2s-r2b:.6f}"
        )
    if (r2j - r2b) <= 0.10:
        raise RuntimeError(
            f"OOF joint-block self-test failed: delta={r2j-r2b:.6f}"
        )

    q = bh_fdr(np.array([0.001, 0.02, 0.5, 0.8]))
    if np.any(q < np.array([0.001, 0.02, 0.5, 0.8]) - 1e-12):
        raise RuntimeError("BH self-test failed")

    print("WF-P Stage7C association self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file", required=False)
    ap.add_argument("--temporal-linkage", required=False)
    ap.add_argument("--wfp0-factors", required=False)
    ap.add_argument("--phenotype-private", required=False)
    ap.add_argument("--phenotype-preflight-json", required=False)
    ap.add_argument("--scale-results-json", required=False)
    ap.add_argument("--stage7b-results-json", required=False)
    ap.add_argument("--spec", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    required = [
        "score_file", "temporal_linkage", "wfp0_factors",
        "phenotype_private", "phenotype_preflight_json",
        "scale_results_json", "stage7b_results_json", "spec", "out",
    ]
    missing = [x for x in required if getattr(args, x) is None]
    if missing:
        raise SystemExit(f"Missing required args: {missing}")

    script_path = Path(__file__).resolve()
    scorep = Path(args.score_file).expanduser().resolve()
    temporalp = Path(args.temporal_linkage).expanduser().resolve()
    factorp = Path(args.wfp0_factors).expanduser().resolve()
    phenop = Path(args.phenotype_private).expanduser().resolve()
    preflightp = Path(args.phenotype_preflight_json).expanduser().resolve()
    scalep = Path(args.scale_results_json).expanduser().resolve()
    stage7bp = Path(args.stage7b_results_json).expanduser().resolve()
    specp = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(specp.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_STAGE7C_ASSOCIATION":
        raise SystemExit("FAIL: Stage7C frozen spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(script_path):
        raise SystemExit("FAIL: Stage7C analysis-script hash mismatch")

    file_map = {
        "score_file_sha256": scorep,
        "temporal_linkage_sha256": temporalp,
        "wfp0_factor_sha256": factorp,
        "phenotype_private_sha256": phenop,
        "phenotype_preflight_sha256": preflightp,
        "scale_results_sha256": scalep,
        "stage7b_results_sha256": stage7bp,
    }
    for key, path in file_map.items():
        if spec.get(key) != sha256_file(path):
            raise SystemExit(f"FAIL: frozen input mismatch: {key}")

    pre = json.loads(preflightp.read_text(encoding="utf-8"))
    if pre.get("main_stage7c_candidates") != PHENOTYPES:
        raise SystemExit(
            "FAIL: phenotype family differs from frozen preflight candidates"
        )

    scores = pd.read_csv(scorep, dtype={"patient_id": str})
    temporal = pd.read_csv(temporalp, dtype={"patient_id": str})
    factors = pd.read_csv(factorp, dtype={"patient_id": str})
    pheno = pd.read_csv(phenop, dtype={"patient_id": str})

    current_cols = [f"{p}__current_exact_admission" for p in PHENOTYPES]

    df = (
        scores[["patient_id"] + [f"z{j}" for j in range(1, D + 1)]]
        .merge(
            temporal[["patient_id", "age_years_capped90", "gender"]],
            on="patient_id", how="left", validate="one_to_one"
        )
        .merge(
            factors[
                ["patient_id", "level_mmhg", "log_scale_sd", "log_duration_sec"]
            ],
            on="patient_id", how="left", validate="one_to_one"
        )
        .merge(
            pheno[["patient_id", "exact_current_hadm_id"] + current_cols],
            on="patient_id", how="left", validate="one_to_one"
        )
    )

    df = df[df["exact_current_hadm_id"].notna()].copy()
    if len(df) != EXPECTED_N:
        raise SystemExit(f"FAIL: expected exact-admission n={EXPECTED_N}, found {len(df)}")

    for p, c in zip(PHENOTYPES, current_cols):
        if df[c].isna().any():
            raise SystemExit(f"FAIL: current exact phenotype missing: {p}")
        df[p] = df[c].astype(float)

    if df[
        ["age_years_capped90", "gender"] + CONTINUOUS_BASELINE[1:]
    ].isna().any().any():
        raise SystemExit("FAIL: baseline covariates incomplete in exact cohort")

    sex = df["gender"].astype(str).str.upper().str.strip()
    if (~sex.isin(["M", "F"])).any():
        raise SystemExit("FAIL: unexpected gender coding")
    df["female"] = (sex == "F").astype(float)

    Z = df[[f"z{j}" for j in range(1, D + 1)]].to_numpy(float)

    # OOF predictive effect sizes.
    pred_baseline = oof_predict(df, Z, [])
    pred_joint = oof_predict(df, Z, PHENOTYPES)

    r2_baseline = aggregate_r2(Z, pred_baseline)
    r2_joint = aggregate_r2(Z, pred_joint)
    block_delta = float(r2_joint - r2_baseline)

    phenotype_rows = []

    # Full-sample joint-model coefficient vectors.
    Xfull, _, _ = build_design(df, PHENOTYPES)
    beta_full, pred_full, sse_full = fit_ols(Xfull, Z)

    # beta rows: intercept; 4 continuous; female; then 8 phenotypes.
    phenotype_beta_start = 1 + len(CONTINUOUS_BASELINE) + 1

    scalej = json.loads(scalep.read_text(encoding="utf-8"))
    between_rms = float(scalej["between"]["pairwise_rms"])
    within_rms = float(scalej["within"]["equal_patient_within_rms"])
    nn_median = float(scalej["between"]["nearest_neighbor_distance"]["median"])

    rng_master = np.random.default_rng(PERM_SEED)

    for idx, p in enumerate(PHENOTYPES):
        # Marginal added predictive value.
        pred_marginal = oof_predict(df, Z, [p])
        r2_marginal = aggregate_r2(Z, pred_marginal)
        marginal_delta = float(r2_marginal - r2_baseline)

        # Unique predictive value within the full phenotype block.
        others = [q for q in PHENOTYPES if q != p]
        pred_without = oof_predict(df, Z, others)
        r2_without = aggregate_r2(Z, pred_without)
        unique_delta = float(r2_joint - r2_without)

        beta_vec = np.asarray(beta_full[phenotype_beta_start + idx], float)
        beta_norm = float(np.linalg.norm(beta_vec))

        # Independent deterministic RNG stream per phenotype.
        seed_bytes = hashlib.sha256(
            f"{PERM_SEED}:{p}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(seed_bytes[:8], "little", signed=False)
        test = permutation_global_test(
            df, Z, p, np.random.default_rng(seed)
        )

        row = {
            "phenotype": p,
            "exposed_n": int(df[p].sum()),
            "unexposed_n": int(len(df) - df[p].sum()),
            "marginal_delta_oof_r2_vs_baseline": marginal_delta,
            "unique_delta_oof_r2_within_joint_block": unique_delta,
            "joint_adjusted_B8_shift_norm": beta_norm,
            "shift_norm_over_between_pairwise_rms":
                beta_norm / between_rms,
            "shift_norm_over_within_rms":
                beta_norm / within_rms,
            "shift_norm_over_median_nearest_neighbor":
                beta_norm / nn_median,
            "partial_sse_improvement": test["partial_sse_improvement"],
            "global_permutation_p": test["permutation_p"],
        }
        for j in range(D):
            row[f"z{j+1}_joint_adjusted_coefficient_DESCRIPTIVE"] = float(
                beta_vec[j]
            )
        phenotype_rows.append(row)

    pdf = pd.DataFrame(phenotype_rows)
    pdf["global_BH_q"] = bh_fdr(pdf["global_permutation_p"].to_numpy(float))
    pdf["global_BH_FDR05"] = pdf["global_BH_q"] <= 0.05

    # Stable output ordering is the frozen phenotype order.
    pdf["_ord"] = pdf["phenotype"].map({p: i for i, p in enumerate(PHENOTYPES)})
    pdf = pdf.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    pdf.to_csv(out / "wfp_stage7c_phenotype_association_summary.csv", index=False)

    # Private patient table with the exact analysis frame and raw phenotype flags,
    # but no new scientific coordinate system.
    priv_cols = (
        ["patient_id", "age_years_capped90", "female"]
        + CONTINUOUS_BASELINE[1:]
        + PHENOTYPES
        + [f"z{j}" for j in range(1, D + 1)]
    )
    df[priv_cols].to_csv(
        out / "wfp_stage7c_analysis_frame_PRIVATE.csv",
        index=False,
    )

    significant = pdf.loc[pdf["global_BH_FDR05"], "phenotype"].astype(str).tolist()

    result = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "7C",
        "decision": "WFP_STAGE7C_ASSOCIATION_COMPLETE",
        "scientific_role": "POST_DISCOVERY_CROSS_SECTIONAL_PHENOTYPE_MAPPING",
        "analysis_n": EXPECTED_N,
        "frozen_B8_changed": False,
        "primary_baseline": "age + sex + level + log(scale) + log(duration)",
        "phenotype_family": PHENOTYPES,
        "all_phenotypes_joint": True,
        "primary_block_effect": {
            "baseline_oof_r2": r2_baseline,
            "joint_phenotype_model_oof_r2": r2_joint,
            "phenotype_block_delta_oof_r2": block_delta,
        },
        "phenotype_global_testing": {
            "test": "reduced-model residual permutation partial multivariate SSE improvement",
            "permutations": PERMUTATIONS,
            "plus_one_p": True,
            "multiple_testing": "Benjamini-Hochberg across exactly 8 prespecified phenotype global tests",
            "FDR_level": 0.05,
            "FDR05_phenotypes": significant,
        },
        "scale_reference": {
            "between_pairwise_rms": between_rms,
            "within_60s_equal_patient_rms": within_rms,
            "median_nearest_neighbor_distance": nn_median,
        },
        "boundary": [
            "Current-admission phenotypes are discharge-coded cross-sectional associations.",
            "No causal or pre-waveform-known interpretation is authorized.",
            "Axis-specific coefficients are descriptive only.",
            "No disease-specific basis/subspace is learned.",
            "No Ztrait/Zstate claim is authorized.",
            "Prior completed-history phenotype mapping is not promoted into this primary analysis.",
        ],
        "hashes": {
            "frozen_spec_sha256": sha256_file(specp),
            "analysis_script_sha256": sha256_file(script_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    (out / "WFP_STAGE7C_ASSOCIATION_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "WF-P STAGE 7C — CHRONIC PHENOTYPE MAPPING",
        "==========================================",
        "Decision: WFP_STAGE7C_ASSOCIATION_COMPLETE",
        "Scientific role: POST-DISCOVERY CROSS-SECTIONAL PHENOTYPE MAPPING",
        f"Analysis n: {EXPECTED_N}",
        "Frozen B8 changed: NO",
        "",
        "Primary phenotype-block effect beyond age/sex + conventional factors:",
        f"  baseline OOF R2: {r2_baseline:.6f}",
        f"  joint 8-phenotype OOF R2: {r2_joint:.6f}",
        f"  phenotype-block incremental OOF R2: {block_delta:.6f}",
        "",
        "Phenotype-specific results:",
    ]

    for _, r in pdf.iterrows():
        lines.append(
            f"  {r['phenotype']}: n+={int(r['exposed_n'])}; "
            f"marginal deltaR2={r['marginal_delta_oof_r2_vs_baseline']:.6f}; "
            f"unique deltaR2={r['unique_delta_oof_r2_within_joint_block']:.6f}; "
            f"shift_norm={r['joint_adjusted_B8_shift_norm']:.6f}; "
            f"shift/between={r['shift_norm_over_between_pairwise_rms']:.4f}; "
            f"shift/within={r['shift_norm_over_within_rms']:.4f}; "
            f"p={r['global_permutation_p']:.6g}; "
            f"BH-q={r['global_BH_q']:.6g}; "
            f"FDR05={bool(r['global_BH_FDR05'])}"
        )

    lines += [
        "",
        "BH-FDR 0.05 global phenotype associations:",
        "  " + (", ".join(significant) if significant else "NONE"),
        "",
        "Boundary:",
        "  Current admission phenotypes are discharge-coded cross-sectional associations.",
        "  Axis-specific coefficients are descriptive only.",
        "  Do not infer causality, pre-waveform phenotype knowledge, Ztrait, or Zstate.",
        "  Do not learn disease-specific subspaces from this result.",
    ]

    (out / "WFP_STAGE7C_ASSOCIATION_READOUT.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
