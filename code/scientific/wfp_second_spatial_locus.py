#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 5D — second SPATIAL localization locus within frozen B8.

Explicitly exploratory post-Stage5C characterization.

Goal
----
Stage 5C found a second independent morphology direction, but it could still localize
to the same upstroke/peak region. Stage 5D asks a different question:

    After excluding the first Stage-5B localization locus in PHASE,
    where is the strongest remaining localization locus?

Primary rule
------------
For each metric and each already-frozen window width:
1. Take the first Stage-5B full-B8 optimal window W1.
2. Expand W1 by a guard band of ceil(width/2) operator-grid points on each side.
3. Search all same-width windows that DO NOT intersect this excluded zone.
4. Within each allowed window, optimize over ALL directions in frozen B8.
5. Keep the single best remaining spatial locus W2.
6. STOP. No third spatial locus is searched.

Secondary targeted check
------------------------
At W2, evaluate span(Axis5, Axis6) and rank it against all 28 axis pairs.

No waveform reprocessing. No clinical labels. No dicrotic-notch phase is prespecified.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P = 64
D = 8
METRICS = ("shape", "slope", "curvature")
WINDOW_WIDTHS = (4, 8, 16)
TARGET_PAIR = (5, 6)
TOL = 1e-12

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def difference_operator(order: int) -> np.ndarray:
    if order == 0:
        return np.eye(P, dtype=float)
    if order == 1:
        A = np.zeros((P - 1, P), dtype=float)
        for i in range(P - 1):
            A[i, i] = -1.0
            A[i, i + 1] = 1.0
        return A
    if order == 2:
        A = np.zeros((P - 2, P), dtype=float)
        for i in range(P - 2):
            A[i, i] = 1.0
            A[i, i + 1] = -2.0
            A[i, i + 2] = 1.0
        return A
    raise ValueError(order)

def metric_operator(metric: str) -> np.ndarray:
    return {
        "shape": difference_operator(0),
        "slope": difference_operator(1),
        "curvature": difference_operator(2),
    }[metric]

def operator_phase(metric: str) -> np.ndarray:
    if metric == "shape":
        return np.arange(P, dtype=float) / P
    if metric == "slope":
        return (np.arange(P - 1, dtype=float) + 0.5) / P
    if metric == "curvature":
        return (np.arange(P - 2, dtype=float) + 1.0) / P
    raise ValueError(metric)

def generalized_top_direction(
    basis: np.ndarray,
    operator: np.ndarray,
    start: int,
    width: int,
) -> Tuple[float, np.ndarray]:
    """
    Maximize ||M_W A B c||^2 / ||A B c||^2.
    Returns max localization fraction and unit coefficient vector.
    """
    Y = operator @ basis
    if start < 0 or start + width > Y.shape[0]:
        raise ValueError("window outside operator grid")

    G = Y.T @ Y
    Yw = Y[start:start + width, :]
    L = Yw.T @ Yw

    g, V = np.linalg.eigh(0.5 * (G + G.T))
    scale = max(1.0, float(np.max(np.abs(g))))
    keep = g > 1e-12 * scale
    if not np.any(keep):
        return np.nan, np.full(basis.shape[1], np.nan)

    W = V[:, keep] @ np.diag(1.0 / np.sqrt(g[keep]))
    H = W.T @ L @ W
    h, U = np.linalg.eigh(0.5 * (H + H.T))
    c = W @ U[:, int(np.argmax(h))]

    nc = float(np.linalg.norm(c))
    if nc <= TOL:
        return np.nan, np.full(basis.shape[1], np.nan)
    c /= nc

    y = Y @ c
    denom = float(np.dot(y, y))
    numer = float(np.dot(y[start:start + width], y[start:start + width]))
    frac = numer / denom if denom > 0 else np.nan

    k = int(np.argmax(np.abs(c)))
    if c[k] < 0:
        c = -c
    return float(frac), c

def intervals_intersect(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)

def guard_points(width: int) -> int:
    return int(math.ceil(width / 2.0))

def allowed_second_windows(
    metric: str,
    width: int,
    first_start: int,
    first_end: int,
):
    nloc = metric_operator(metric).shape[0]
    g = guard_points(width)
    ex0 = max(0, first_start - g)
    ex1 = min(nloc - 1, first_end + g)

    starts = []
    for s in range(0, nloc - width + 1):
        e = s + width - 1
        if not intervals_intersect(s, e, ex0, ex1):
            starts.append(s)
    return starts, ex0, ex1

def score_sd(scores: pd.DataFrame, c: np.ndarray) -> float:
    Z = scores[[f"z{j}" for j in range(1, D + 1)]].to_numpy(float)
    return float(np.std(Z @ c, ddof=1))

def pair_basis(B: np.ndarray, pair: Tuple[int, int]) -> np.ndarray:
    return B[:, [pair[0] - 1, pair[1] - 1]]

def pair_localization(
    B: np.ndarray,
    pair: Tuple[int, int],
    metric: str,
    start: int,
    width: int,
):
    frac, cp = generalized_top_direction(
        pair_basis(B, pair), metric_operator(metric), start, width
    )
    c8 = np.zeros(D, dtype=float)
    c8[pair[0] - 1] = cp[0]
    c8[pair[1] - 1] = cp[1]
    return frac, c8

def operator_energy_profile(metric: str, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    op = metric_operator(metric)
    y = op @ direction
    e = y * y
    s = float(np.sum(e))
    if s > 0:
        e = e / s
    return operator_phase(metric), e

def self_test() -> int:
    # First window at middle; verify allowed windows are spatially excluded.
    metric = "shape"
    width = 8
    starts, ex0, ex1 = allowed_second_windows(metric, width, 24, 31)
    for s in starts:
        e = s + width - 1
        if intervals_intersect(s, e, ex0, ex1):
            print("SELF-TEST FAIL: allowed window intersects excluded zone", file=sys.stderr)
            return 1
    if not starts:
        print("SELF-TEST FAIL: no allowed windows", file=sys.stderr)
        return 1

    rng = np.random.default_rng(20260820)
    A = rng.normal(size=(P, D))
    A -= np.mean(A, axis=0, keepdims=True)
    B, _ = np.linalg.qr(A)
    frac, c = generalized_top_direction(B, metric_operator("shape"), starts[0], width)
    if not np.isfinite(frac) or abs(np.linalg.norm(c) - 1.0) > 1e-10:
        print("SELF-TEST FAIL: optimizer", file=sys.stderr)
        return 1

    print("WFP second spatial locus self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--discovery-results",
        default="~/Documents/abp_information_study/results/wfp_discovery_validation1000",
    )
    ap.add_argument(
        "--stage5b-results",
        default="~/Documents/abp_information_study/results/wfp_within_b8_localization",
    )
    ap.add_argument(
        "--stage5c-result",
        default="~/Documents/abp_information_study/results/wfp_second_localized_direction/WFP_SECOND_LOCALIZED_DIRECTION.json",
    )
    ap.add_argument(
        "--spec",
        default="~/Documents/abp_information_study/freeze/wfp_second_spatial_locus/WFP_SECOND_SPATIAL_LOCUS_FROZEN_SPEC.json",
    )
    ap.add_argument(
        "--out",
        default="~/Documents/abp_information_study/results/wfp_second_spatial_locus",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    discovery = Path(args.discovery_results).expanduser().resolve()
    stage5b = Path(args.stage5b_results).expanduser().resolve()
    stage5c_path = Path(args.stage5c_result).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SECOND_SPATIAL_LOCUS":
        raise SystemExit("FAIL: Stage 5D frozen spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("FAIL: Stage 5D analysis script hash mismatch")

    coord_path = discovery / "WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores_path = discovery / "wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    stage5b_json = stage5b / "WFP_WITHIN_B8_LOCALIZATION.json"
    stage5b_dirs = stage5b / "wfp_within_b8_localized_directions.csv"
    stage5b_windows = stage5b / "wfp_within_b8_localization_best_windows.csv"

    expected = {
        "coordinates_sha256": coord_path,
        "patient_scores_sha256": scores_path,
        "stage5b_result_sha256": stage5b_json,
        "stage5b_first_directions_sha256": stage5b_dirs,
        "stage5b_best_windows_sha256": stage5b_windows,
        "stage5c_result_sha256": stage5c_path,
    }
    for key, p in expected.items():
        if not p.is_file():
            raise SystemExit(f"FAIL: required input missing: {p}")
        if spec.get(key) != sha256_file(p):
            raise SystemExit(f"FAIL: input hash mismatch: {key}")

    s5b = json.loads(stage5b_json.read_text(encoding="utf-8"))
    s5c = json.loads(stage5c_path.read_text(encoding="utf-8"))
    if s5b.get("decision") != "WFP_WITHIN_B8_LOCALIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5B not complete")
    if s5c.get("decision") != "WFP_SECOND_LOCALIZED_DIRECTION_COMPLETE":
        raise SystemExit("FAIL: Stage 5C not complete")
    if s5b.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: Stage 5B clinical-label boundary violated")
    if s5c.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: Stage 5C clinical-label boundary violated")

    with np.load(coord_path, allow_pickle=False) as z:
        mu = np.asarray(z["population_mean"], float)
        B = np.asarray(z["between_basis"], float)
    scores = pd.read_csv(scores_path)
    first_dirs = pd.read_csv(stage5b_dirs)
    first_windows = pd.read_csv(stage5b_windows)

    if mu.shape != (P,) or B.shape != (P, D):
        raise SystemExit("FAIL: frozen coordinate dimensions invalid")
    if len(scores) != 978:
        raise SystemExit(f"FAIL: expected 978 score rows, found {len(scores)}")
    if np.max(np.abs(B.T @ B - np.eye(D))) > 1e-10:
        raise SystemExit("FAIL: B8 not orthonormal")

    scan_rows = []
    summary_rows = []
    pair_rows = []
    curve_df = pd.DataFrame({
        "phase_index": np.arange(P, dtype=int),
        "phase": np.arange(P, dtype=float) / P,
        "population_mean": mu,
    })

    for metric in METRICS:
        for width in WINDOW_WIDTHS:
            fr = first_dirs[
                (first_dirs["metric"] == metric) &
                (first_dirs["width_points"] == width)
            ]
            fw = first_windows[
                (first_windows["metric"] == metric) &
                (first_windows["width_points"] == width)
            ]
            if len(fr) != 1 or len(fw) != 1:
                raise SystemExit(
                    f"FAIL: Stage5B first result not unique for {metric}, width={width}"
                )
            fr = fr.iloc[0]
            fw = fw.iloc[0]

            c1 = np.array([fr[f"c_axis{j}"] for j in range(1, D + 1)], float)
            c1 /= np.linalg.norm(c1)
            v1 = B @ c1
            first_sd = float(fr["score_sd"])
            first_fraction = float(fr["full_b8_localization_fraction"])
            first_start = int(fw["start_index"])
            first_end = int(fw["end_index"])

            starts, ex0, ex1 = allowed_second_windows(
                metric, width, first_start, first_end
            )
            if not starts:
                raise SystemExit(
                    f"FAIL: no allowed second windows for {metric}, width={width}"
                )

            op = metric_operator(metric)
            ph = operator_phase(metric)
            candidates = []
            for s in starts:
                frac, c2 = generalized_top_direction(B, op, s, width)
                candidates.append((frac, s, c2))
                scan_rows.append({
                    "metric": metric,
                    "width_points": width,
                    "first_start_index": first_start,
                    "first_end_index": first_end,
                    "guard_points_each_side": guard_points(width),
                    "excluded_start_index": ex0,
                    "excluded_end_index": ex1,
                    "candidate_start_index": s,
                    "candidate_end_index": s + width - 1,
                    "candidate_start_phase": float(ph[s]),
                    "candidate_end_phase": float(ph[s + width - 1]),
                    "candidate_center_phase": float(
                        0.5 * (ph[s] + ph[s + width - 1])
                    ),
                    "localization_fraction": frac,
                    **{f"c_axis{j+1}": float(c2[j]) for j in range(D)},
                })

            candidates.sort(key=lambda x: x[0], reverse=True)
            second_fraction, second_start, c2 = candidates[0]
            second_end = second_start + width - 1
            v2 = B @ c2
            second_sd = score_sd(scores, c2)

            # Axis5-6 and all-pair context at the SAME W2.
            pair_stats = []
            for pair in itertools.combinations(range(1, D + 1), 2):
                frac, cp = pair_localization(B, pair, metric, second_start, width)
                pair_stats.append((pair, frac, cp))
            pair_stats.sort(key=lambda x: x[1], reverse=True)

            target = None
            for rank, (pair, frac, cp) in enumerate(pair_stats, start=1):
                pair_rows.append({
                    "metric": metric,
                    "width_points": width,
                    "second_start_phase": float(ph[second_start]),
                    "second_end_phase": float(ph[second_end]),
                    "pair_axis_a": pair[0],
                    "pair_axis_b": pair[1],
                    "pair_rank_at_second_locus": rank,
                    "pair_localization_fraction": frac,
                    "fraction_of_full_b8_second_locus_optimum":
                        frac / second_fraction if second_fraction > 0 else np.nan,
                    **{f"c_axis{j+1}": float(cp[j]) for j in range(D)},
                })
                if pair == TARGET_PAIR:
                    target = (rank, frac, cp)

            if target is None:
                raise RuntimeError("Axis5-6 pair missing")
            target_rank, target_fraction, target_c = target

            summary_rows.append({
                "metric": metric,
                "width_points": width,
                "guard_points_each_side": guard_points(width),
                "first_localization_fraction": first_fraction,
                "first_start_phase": float(ph[first_start]),
                "first_end_phase": float(ph[first_end]),
                "excluded_start_phase": float(ph[ex0]),
                "excluded_end_phase": float(ph[ex1]),
                "second_localization_fraction": float(second_fraction),
                "second_to_first_localization_fraction":
                    float(second_fraction / first_fraction)
                    if first_fraction > 0 else np.nan,
                "second_start_phase": float(ph[second_start]),
                "second_end_phase": float(ph[second_end]),
                "second_center_phase": float(
                    0.5 * (ph[second_start] + ph[second_end])
                ),
                "second_score_sd": second_sd,
                "second_axis5_6_squared_weight":
                    float(c2[4] ** 2 + c2[5] ** 2),
                "axis5_6_localization_fraction_at_second_locus":
                    float(target_fraction),
                "axis5_6_fraction_of_full_second_locus_optimum":
                    float(target_fraction / second_fraction)
                    if second_fraction > 0 else np.nan,
                "axis5_6_pair_rank_of_28_at_second_locus":
                    int(target_rank),
                **{f"second_c_axis{j+1}": float(c2[j]) for j in range(D)},
            })

            tag = f"{metric}_w{width}"
            curve_df[f"{tag}_first_direction"] = v1
            curve_df[f"{tag}_second_spatial_direction"] = v2
            curve_df[f"{tag}_first_minus1sd"] = mu - first_sd * v1
            curve_df[f"{tag}_first_plus1sd"] = mu + first_sd * v1
            curve_df[f"{tag}_second_minus1sd"] = mu - second_sd * v2
            curve_df[f"{tag}_second_plus1sd"] = mu + second_sd * v2

    scan_df = pd.DataFrame(scan_rows)
    summary_df = pd.DataFrame(summary_rows)
    pairs_df = pd.DataFrame(pair_rows)

    scan_df.to_csv(out / "wfp_second_spatial_locus_scan.csv", index=False)
    summary_df.to_csv(out / "wfp_second_spatial_locus_summary.csv", index=False)
    pairs_df.to_csv(out / "wfp_axis_pair_localization_at_second_locus.csv", index=False)
    curve_df.to_csv(out / "wfp_second_spatial_locus_curves.csv", index=False)

    # MAIN FIGURE 1: actual optimized operator-energy profiles, width=8.
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row, metric in enumerate(METRICS):
        sr = summary_df[
            (summary_df["metric"] == metric) &
            (summary_df["width_points"] == 8)
        ].iloc[0]
        fr = first_dirs[
            (first_dirs["metric"] == metric) &
            (first_dirs["width_points"] == 8)
        ].iloc[0]

        c1 = np.array([fr[f"c_axis{j}"] for j in range(1, D + 1)], float)
        c1 /= np.linalg.norm(c1)
        c2 = np.array([sr[f"second_c_axis{j}"] for j in range(1, D + 1)], float)
        c2 /= np.linalg.norm(c2)
        v1 = B @ c1
        v2 = B @ c2

        ph1, e1 = operator_energy_profile(metric, v1)
        ph2, e2 = operator_energy_profile(metric, v2)

        ax = axes[row, 0]
        ax.plot(ph1, e1)
        ax.axvspan(
            float(sr["first_start_phase"]),
            float(sr["first_end_phase"]),
            alpha=0.15,
        )
        ax.axvspan(
            float(sr["excluded_start_phase"]),
            float(sr["excluded_end_phase"]),
            alpha=0.07,
        )
        ax.set_title(
            f"{metric}: 1st locus operator energy | loc="
            f"{float(sr['first_localization_fraction']):.3f}"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("Operator-energy fraction")
        ax.grid(False)

        ax = axes[row, 1]
        ax.plot(ph2, e2)
        ax.axvspan(
            float(sr["second_start_phase"]),
            float(sr["second_end_phase"]),
            alpha=0.15,
        )
        ax.set_title(
            f"{metric}: 2nd spatial locus | loc="
            f"{float(sr['second_localization_fraction']):.3f} | "
            f"Axis5-6 pair rank="
            f"{int(sr['axis5_6_pair_rank_of_28_at_second_locus'])}/28"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("Operator-energy fraction")
        ax.grid(False)

    fig.tight_layout()
    fig.savefig(
        out / "WFP_FIGURE_FIRST_SECOND_SPATIAL_LOCUS_OPERATOR_ENERGY_WIDTH8.png",
        dpi=220, bbox_inches="tight"
    )
    fig.savefig(
        out / "WFP_FIGURE_FIRST_SECOND_SPATIAL_LOCUS_OPERATOR_ENERGY_WIDTH8.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

    # MAIN FIGURE 2: morphology perturbations corresponding to those directions.
    phase = np.arange(P, dtype=float) / P
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row, metric in enumerate(METRICS):
        sr = summary_df[
            (summary_df["metric"] == metric) &
            (summary_df["width_points"] == 8)
        ].iloc[0]
        fr = first_dirs[
            (first_dirs["metric"] == metric) &
            (first_dirs["width_points"] == 8)
        ].iloc[0]
        c1 = np.array([fr[f"c_axis{j}"] for j in range(1, D + 1)], float)
        c1 /= np.linalg.norm(c1)
        c2 = np.array([sr[f"second_c_axis{j}"] for j in range(1, D + 1)], float)
        c2 /= np.linalg.norm(c2)
        v1 = B @ c1
        v2 = B @ c2
        sd1 = float(fr["score_sd"])
        sd2 = float(sr["second_score_sd"])

        ax = axes[row, 0]
        ax.plot(phase, mu, label="Mean")
        ax.plot(phase, mu + sd1 * v1, linestyle="--", label="+1 SD")
        ax.plot(phase, mu - sd1 * v1, linestyle=":", label="-1 SD")
        ax.axvspan(
            float(sr["first_start_phase"]),
            float(sr["first_end_phase"]),
            alpha=0.15,
        )
        ax.set_title(f"{metric}: morphology at 1st locus")
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("shape_norm")
        ax.legend()

        ax = axes[row, 1]
        ax.plot(phase, mu, label="Mean")
        ax.plot(phase, mu + sd2 * v2, linestyle="--", label="+1 SD")
        ax.plot(phase, mu - sd2 * v2, linestyle=":", label="-1 SD")
        ax.axvspan(
            float(sr["second_start_phase"]),
            float(sr["second_end_phase"]),
            alpha=0.15,
        )
        ax.set_title(
            f"{metric}: morphology at 2nd spatial locus | "
            f"phase={float(sr['second_start_phase']):.2f}–"
            f"{float(sr['second_end_phase']):.2f}"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("shape_norm")
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        out / "WFP_FIGURE_FIRST_SECOND_SPATIAL_LOCUS_MORPHOLOGY_WIDTH8.png",
        dpi=220, bbox_inches="tight"
    )
    fig.savefig(
        out / "WFP_FIGURE_FIRST_SECOND_SPATIAL_LOCUS_MORPHOLOGY_WIDTH8.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "5D",
        "decision": "WFP_SECOND_SPATIAL_LOCUS_COMPLETE",
        "scientific_role": "explicitly_exploratory_post_stage5c_spatial_characterization",
        "waveform_arrays_opened": False,
        "clinical_labels_accessed": False,
        "frozen_dimension": 8,
        "basis_changed": False,
        "third_spatial_locus_searched": False,
        "first_locus_source": "Stage 5B full-B8 optimum",
        "spatial_exclusion_rule":
            "exclude first window expanded by ceil(width/2) operator-grid points on each side",
        "second_locus_optimization":
            "optimize over all frozen B8 directions within each allowed same-width window",
        "notch_region_prespecified": False,
        "targeted_axis5_6_status": "EXPLORATORY_POST_STAGE5_VISUAL_HYPOTHESIS",
        "summary": summary_df.to_dict(orient="records"),
        "boundary": [
            "This is exploratory characterization after viewing Stages 5B/5C.",
            "Only one second spatial locus is searched; no third locus is searched.",
            "No phase interval is automatically labeled dicrotic notch.",
            "Axis 5-6 is evaluated only as a targeted exploratory check and is ranked against all 28 pairs.",
            "No waveform reprocessing or clinical-label access occurs.",
            "The frozen B8 basis remains unchanged."
        ],
        "input_hashes": {
            "frozen_spec_sha256": sha256_file(spec_path),
            "coordinates_sha256": sha256_file(coord_path),
            "patient_scores_sha256": sha256_file(scores_path),
            "stage5b_result_sha256": sha256_file(stage5b_json),
            "stage5b_first_directions_sha256": sha256_file(stage5b_dirs),
            "stage5b_best_windows_sha256": sha256_file(stage5b_windows),
            "stage5c_result_sha256": sha256_file(stage5c_path),
        }
    }
    (out / "WFP_SECOND_SPATIAL_LOCUS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )

    lines = [
        "WF-P SECOND SPATIAL LOCALIZATION LOCUS",
        "=====================================",
        "Decision: WFP_SECOND_SPATIAL_LOCUS_COMPLETE",
        "Scientific role: EXPLICITLY EXPLORATORY POST-STAGE5C SPATIAL CHARACTERIZATION",
        "Waveform arrays opened: NO",
        "Clinical labels accessed: NO",
        "Frozen dimension: 8",
        "Primary basis changed: NO",
        "Third spatial locus searched: NO",
        "Notch region prespecified: NO",
        "",
        "Spatial exclusion rule:",
        "  exclude Stage5B first window expanded by ceil(width/2) points on each side",
        "",
        "First versus second spatial loci:",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"  {r['metric']} width={int(r['width_points'])}: "
            f"first loc={r['first_localization_fraction']:.6f}; "
            f"first phase={r['first_start_phase']:.6f}..{r['first_end_phase']:.6f}; "
            f"excluded phase={r['excluded_start_phase']:.6f}..{r['excluded_end_phase']:.6f}; "
            f"second loc={r['second_localization_fraction']:.6f}; "
            f"second/first={r['second_to_first_localization_fraction']:.6f}; "
            f"second phase={r['second_start_phase']:.6f}..{r['second_end_phase']:.6f}; "
            f"second Axis5-6 squared weight={r['second_axis5_6_squared_weight']:.6f}; "
            f"Axis5-6 at W2={r['axis5_6_localization_fraction_at_second_locus']:.6f}; "
            f"Axis5-6/full W2={r['axis5_6_fraction_of_full_second_locus_optimum']:.6f}; "
            f"Axis5-6 pair rank={int(r['axis5_6_pair_rank_of_28_at_second_locus'])}/28"
        )
    lines += [
        "",
        "Boundary:",
        "  Do NOT label W2 as dicrotic notch from phase alone.",
        "  Inspect the width=8 operator-energy figure and morphology figure once.",
        "  Do not search a third spatial locus.",
        "  After this, proceed to Stage 6A outside-B8 residual localization.",
    ]
    (out / "WFP_SECOND_SPATIAL_LOCUS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
