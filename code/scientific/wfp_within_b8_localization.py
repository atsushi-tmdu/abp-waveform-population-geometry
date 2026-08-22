#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 5B — localized morphology directions WITHIN the frozen B8 space.

Purpose
-------
Determine whether local waveform features can be represented as linear combinations
of the already-frozen 8-D WF-P population morphology basis.

This stage uses NO waveform reprocessing and NO clinical labels.

Primary, non-targeted analysis:
For every prespecified contiguous phase window and each prespecified operator
(shape amplitude, first difference, second difference), find the direction
v = B8 c that maximizes the fraction of operator energy inside the window.

Secondary, explicitly exploratory targeted analysis:
Evaluate whether span(Axis 5, Axis 6) can reproduce the full-B8 optimal localized
direction at the SAME selected window, and rank all 28 axis pairs at that window.

No dicrotic-notch region is prespecified or named in the calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P = 64
D = 8
WINDOW_WIDTHS = (4, 8, 16)
METRICS = ("shape", "slope", "curvature")
PAIR_TARGET = (5, 6)  # 1-indexed, exploratory targeted check prompted after Stage 5 inspection.
TOL = 1e-12

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def difference_operator(order: int) -> np.ndarray:
    if order == 0:
        return np.eye(P, dtype=float)
    if order == 1:
        Dm = np.zeros((P-1, P), dtype=float)
        for i in range(P-1):
            Dm[i, i] = -1.0
            Dm[i, i+1] = 1.0
        return Dm
    if order == 2:
        Dm = np.zeros((P-2, P), dtype=float)
        for i in range(P-2):
            Dm[i, i] = 1.0
            Dm[i, i+1] = -2.0
            Dm[i, i+2] = 1.0
        return Dm
    raise ValueError("order must be 0,1,2")

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
        return (np.arange(P-1, dtype=float) + 0.5) / P
    if metric == "curvature":
        return (np.arange(P-2, dtype=float) + 1.0) / P
    raise ValueError(metric)

def generalized_top_direction(
    basis: np.ndarray,
    operator: np.ndarray,
    start: int,
    width: int,
) -> Tuple[float, np.ndarray]:
    """
    Maximize ||M_W A B c||^2 / ||A B c||^2.
    Returns max concentration and Euclidean-unit coefficient vector c.
    """
    Y = operator @ basis
    nloc = Y.shape[0]
    if start < 0 or start + width > nloc:
        raise ValueError("window outside operator grid")

    G = Y.T @ Y
    Yw = Y[start:start+width, :]
    A = Yw.T @ Yw

    # Whiten G on its numerical range.
    g, V = np.linalg.eigh(0.5*(G+G.T))
    gscale = max(1.0, float(np.max(np.abs(g))))
    keep = g > 1e-12 * gscale
    if not np.any(keep):
        return np.nan, np.full(basis.shape[1], np.nan)

    W = V[:, keep] @ np.diag(1.0/np.sqrt(g[keep]))
    H = W.T @ A @ W
    h, U = np.linalg.eigh(0.5*(H+H.T))
    u = U[:, int(np.argmax(h))]
    c = W @ u
    nc = float(np.linalg.norm(c))
    if nc <= TOL:
        return np.nan, np.full(basis.shape[1], np.nan)
    c = c / nc

    y = Y @ c
    denom = float(np.dot(y, y))
    numer = float(np.dot(y[start:start+width], y[start:start+width]))
    frac = numer/denom if denom > 0 else np.nan

    # Deterministic sign: largest absolute coefficient positive.
    k = int(np.argmax(np.abs(c)))
    if c[k] < 0:
        c = -c
    return float(frac), c

def scan_windows(basis: np.ndarray, metric: str, width: int) -> pd.DataFrame:
    op = metric_operator(metric)
    phase = operator_phase(metric)
    rows = []
    for start in range(0, op.shape[0]-width+1):
        frac, c = generalized_top_direction(basis, op, start, width)
        rows.append({
            "metric": metric,
            "width_points": width,
            "start_index": start,
            "end_index": start + width - 1,
            "start_phase": float(phase[start]),
            "end_phase": float(phase[start+width-1]),
            "center_phase": float(0.5*(phase[start]+phase[start+width-1])),
            "max_localization_fraction": frac,
            **{f"c_axis{j+1}": float(c[j]) for j in range(basis.shape[1])},
        })
    return pd.DataFrame(rows)

def best_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["max_localization_fraction"].idxmax()]

def score_sd_for_direction(scores: pd.DataFrame, c: np.ndarray) -> float:
    Z = scores[[f"z{j}" for j in range(1,D+1)]].to_numpy(float)
    return float(np.std(Z @ c, ddof=1))

def pair_basis(B: np.ndarray, pair: Tuple[int,int]) -> np.ndarray:
    return B[:, [pair[0]-1, pair[1]-1]]

def pair_concentration_at_window(
    B: np.ndarray, pair: Tuple[int,int], metric: str, start: int, width: int
) -> Tuple[float, np.ndarray]:
    frac, cp = generalized_top_direction(pair_basis(B,pair), metric_operator(metric), start, width)
    c8 = np.zeros(D, dtype=float)
    c8[pair[0]-1] = cp[0]
    c8[pair[1]-1] = cp[1]
    return frac, c8

def self_test() -> int:
    # Construct a frozen 8-D basis containing one deliberately localized direction.
    rng = np.random.default_rng(20260820)
    local = np.zeros(P)
    local[40:46] = np.array([0.2, -0.5, 1.0, -1.0, 0.5, -0.2])
    local -= np.mean(local)
    local /= np.linalg.norm(local)

    A = rng.normal(size=(P, D-1))
    A -= np.mean(A, axis=0, keepdims=True)
    # Remove local component and QR.
    A -= local[:,None] @ (local[None,:] @ A)
    Q,_ = np.linalg.qr(A)
    B = np.column_stack([local, Q[:,:D-1]])
    B,_ = np.linalg.qr(B)

    df = scan_windows(B, "shape", 8)
    b = best_row(df)
    if float(b["max_localization_fraction"]) < 0.5:
        print("SELF-TEST FAIL: localized direction not recovered", file=sys.stderr)
        return 1
    print("WFP within-B8 localization self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-results", default="~/Documents/abp_information_study/results/wfp_discovery_validation1000")
    ap.add_argument("--stage5", default="~/Documents/abp_information_study/results/wfp_axis_characterization/WFP_AXIS_CHARACTERIZATION.json")
    ap.add_argument("--spec", default="~/Documents/abp_information_study/freeze/wfp_within_b8_localization/WFP_WITHIN_B8_LOCALIZATION_FROZEN_SPEC.json")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_within_b8_localization")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    results = Path(args.discovery_results).expanduser().resolve()
    stage5_path = Path(args.stage5).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_WITHIN_B8_LOCALIZATION":
        raise SystemExit("FAIL: frozen Stage 5B spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("FAIL: Stage 5B analysis script hash mismatch")
    if spec.get("stage5_result_sha256") != sha256_file(stage5_path):
        raise SystemExit("FAIL: Stage 5 result hash mismatch")

    stage5 = json.loads(stage5_path.read_text(encoding="utf-8"))
    if stage5.get("decision") != "WFP_AXIS_CHARACTERIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5 not complete")
    if int(stage5.get("dimension",-1)) != D:
        raise SystemExit("FAIL: frozen dimension not 8")
    if stage5.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: clinical-label boundary violated")

    coord_path = results / "WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores_path = results / "wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    if spec.get("coordinates_sha256") != sha256_file(coord_path):
        raise SystemExit("FAIL: coordinate hash mismatch")
    if spec.get("patient_scores_sha256") != sha256_file(scores_path):
        raise SystemExit("FAIL: score hash mismatch")

    with np.load(coord_path, allow_pickle=False) as z:
        mu = np.asarray(z["population_mean"], float)
        B = np.asarray(z["between_basis"], float)
    scores = pd.read_csv(scores_path)

    if mu.shape != (P,) or B.shape != (P,D):
        raise SystemExit(f"FAIL: frozen coordinate dimensions invalid: mu={mu.shape}, B={B.shape}")
    if len(scores) != 978:
        raise SystemExit(f"FAIL: expected 978 score rows, found {len(scores)}")

    # Numerical orthonormality check.
    ortho_err = float(np.max(np.abs(B.T@B - np.eye(D))))
    if ortho_err > 1e-10:
        raise SystemExit(f"FAIL: B8 not orthonormal at expected precision: {ortho_err}")

    scan_frames = []
    best_rows = []
    pair_rows = []
    direction_rows = []
    curve_df = pd.DataFrame({"phase_index":np.arange(P), "phase":np.arange(P)/P, "population_mean":mu})

    for metric in METRICS:
        for width in WINDOW_WIDTHS:
            df = scan_windows(B, metric, width)
            scan_frames.append(df)
            br = best_row(df)
            best_rows.append(br.to_dict())

            c_full = np.array([br[f"c_axis{j}"] for j in range(1,D+1)], float)
            v_full = B @ c_full
            sd_full = score_sd_for_direction(scores, c_full)

            tag = f"{metric}_w{width}"
            curve_df[f"{tag}_direction"] = v_full
            curve_df[f"{tag}_minus1sd"] = mu - sd_full*v_full
            curve_df[f"{tag}_plus1sd"] = mu + sd_full*v_full

            direction_rows.append({
                "metric":metric,
                "width_points":width,
                "full_b8_localization_fraction":float(br["max_localization_fraction"]),
                "start_phase":float(br["start_phase"]),
                "end_phase":float(br["end_phase"]),
                "center_phase":float(br["center_phase"]),
                "score_sd":sd_full,
                "axis5_6_energy_fraction_in_full_optimum":
                    float(c_full[4]**2 + c_full[5]**2),
                **{f"c_axis{j+1}":float(c_full[j]) for j in range(D)},
            })

            # Fair pair comparison at the SAME full-B8-selected window.
            start = int(br["start_index"])
            pair_stats = []
            for pair in itertools.combinations(range(1,D+1),2):
                frac, c8 = pair_concentration_at_window(B,pair,metric,start,width)
                pair_stats.append((pair,frac,c8))
            pair_stats.sort(key=lambda x: x[1], reverse=True)

            target_rank = None
            for rank,(pair,frac,c8) in enumerate(pair_stats,start=1):
                if pair == PAIR_TARGET:
                    target_rank = rank
                    target_frac = frac
                    target_c8 = c8
                    break

            # Save all pair ranks, not only 5-6, so the targeted check is contextualized.
            for rank,(pair,frac,c8) in enumerate(pair_stats,start=1):
                pair_rows.append({
                    "metric":metric,
                    "width_points":width,
                    "full_b8_selected_start_phase":float(br["start_phase"]),
                    "full_b8_selected_end_phase":float(br["end_phase"]),
                    "pair_axis_a":pair[0],
                    "pair_axis_b":pair[1],
                    "pair_rank_at_fixed_window":rank,
                    "pair_localization_fraction_at_fixed_window":frac,
                    "fraction_of_full_b8_optimum":
                        frac/float(br["max_localization_fraction"])
                        if float(br["max_localization_fraction"])>0 else np.nan,
                    **{f"c_axis{j+1}":float(c8[j]) for j in range(D)},
                })

            sd56 = score_sd_for_direction(scores, target_c8)
            v56 = B @ target_c8
            curve_df[f"{tag}_axis5_6_direction"] = v56
            curve_df[f"{tag}_axis5_6_minus1sd"] = mu - sd56*v56
            curve_df[f"{tag}_axis5_6_plus1sd"] = mu + sd56*v56

    scan_df = pd.concat(scan_frames, ignore_index=True)
    best_df = pd.DataFrame(best_rows)
    pairs_df = pd.DataFrame(pair_rows)
    directions_df = pd.DataFrame(direction_rows)

    scan_df.to_csv(out/"wfp_within_b8_localization_scan.csv",index=False)
    best_df.to_csv(out/"wfp_within_b8_localization_best_windows.csv",index=False)
    pairs_df.to_csv(out/"wfp_axis_pair_localization_at_full_b8_windows.csv",index=False)
    directions_df.to_csv(out/"wfp_within_b8_localized_directions.csv",index=False)
    curve_df.to_csv(out/"wfp_within_b8_localized_direction_curves.csv",index=False)

    # Plot localization capacity over phase for all frozen widths.
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9,5))
        for width in WINDOW_WIDTHS:
            d = scan_df[(scan_df["metric"]==metric)&(scan_df["width_points"]==width)]
            ax.plot(d["center_phase"], d["max_localization_fraction"], label=f"width={width}")
        ax.set_xlabel("Normalized beat phase (window center)")
        ax.set_ylabel(f"Max fraction of {metric} energy localizable within window")
        ax.set_ylim(0,1.02)
        ax.legend()
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(out/f"WFP_FIGURE_{metric.upper()}_LOCALIZATION_CAPACITY.png",dpi=220,bbox_inches="tight")
        fig.savefig(out/f"WFP_FIGURE_{metric.upper()}_LOCALIZATION_CAPACITY.pdf",bbox_inches="tight")
        plt.close(fig)

    # Main figure: width=8 full-B8 optimum versus Axis5-6 constrained optimum.
    fig, axes = plt.subplots(3,2,figsize=(11,11))
    for row,metric in enumerate(METRICS):
        width=8
        br = directions_df[(directions_df["metric"]==metric)&(directions_df["width_points"]==width)].iloc[0]
        cfull=np.array([br[f"c_axis{j}"] for j in range(1,D+1)],float)
        vfull=B@cfull
        sdfull=float(br["score_sd"])

        pr = pairs_df[
            (pairs_df["metric"]==metric)&
            (pairs_df["width_points"]==width)&
            (pairs_df["pair_axis_a"]==5)&
            (pairs_df["pair_axis_b"]==6)
        ].iloc[0]
        c56=np.array([pr[f"c_axis{j}"] for j in range(1,D+1)],float)
        v56=B@c56
        sd56=score_sd_for_direction(scores,c56)

        phase=np.arange(P)/P
        ax=axes[row,0]
        ax.plot(phase,mu,label="Mean")
        ax.plot(phase,mu+sdfull*vfull,linestyle="--",label="+1 SD")
        ax.plot(phase,mu-sdfull*vfull,linestyle=":",label="-1 SD")
        ax.axvspan(float(br["start_phase"]),float(br["end_phase"]),alpha=0.15)
        ax.set_title(
            f"{metric}: full B8 optimum | loc={float(br['full_b8_localization_fraction']):.3f}"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("shape_norm")
        ax.legend()

        ax=axes[row,1]
        ax.plot(phase,mu,label="Mean")
        ax.plot(phase,mu+sd56*v56,linestyle="--",label="+1 SD")
        ax.plot(phase,mu-sd56*v56,linestyle=":",label="-1 SD")
        ax.axvspan(float(br["start_phase"]),float(br["end_phase"]),alpha=0.15)
        ax.set_title(
            f"{metric}: Axis 5-6 constrained | loc={float(pr['pair_localization_fraction_at_fixed_window']):.3f} "
            f"| pair rank={int(pr['pair_rank_at_fixed_window'])}/28"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("shape_norm")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/"WFP_FIGURE_LOCALIZED_DIRECTIONS_WIDTH8.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"WFP_FIGURE_LOCALIZED_DIRECTIONS_WIDTH8.pdf",bbox_inches="tight")
    plt.close(fig)

    # Coefficient composition for width=8 full-B8 optima.
    fig, axes = plt.subplots(3,1,figsize=(8,9),sharex=True)
    for ax,metric in zip(axes,METRICS):
        r=directions_df[(directions_df["metric"]==metric)&(directions_df["width_points"]==8)].iloc[0]
        coeff=np.array([r[f"c_axis{j}"] for j in range(1,D+1)],float)
        ax.bar(np.arange(1,D+1),coeff)
        ax.set_ylabel("Coefficient")
        ax.set_title(
            f"{metric} width=8 optimum | Axis5-6 squared weight="
            f"{float(r['axis5_6_energy_fraction_in_full_optimum']):.3f}"
        )
        ax.grid(False)
    axes[-1].set_xlabel("Frozen WF-P axis")
    fig.tight_layout()
    fig.savefig(out/"WFP_FIGURE_LOCALIZED_DIRECTION_COEFFICIENTS_WIDTH8.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"WFP_FIGURE_LOCALIZED_DIRECTION_COEFFICIENTS_WIDTH8.pdf",bbox_inches="tight")
    plt.close(fig)

    # Summarize pair 5-6 at fixed full-B8 windows.
    targeted=[]
    for metric in METRICS:
        for width in WINDOW_WIDTHS:
            full=directions_df[(directions_df["metric"]==metric)&(directions_df["width_points"]==width)].iloc[0]
            pr=pairs_df[
                (pairs_df["metric"]==metric)&
                (pairs_df["width_points"]==width)&
                (pairs_df["pair_axis_a"]==5)&
                (pairs_df["pair_axis_b"]==6)
            ].iloc[0]
            targeted.append({
                "metric":metric,
                "width_points":width,
                "full_b8_start_phase":float(full["start_phase"]),
                "full_b8_end_phase":float(full["end_phase"]),
                "full_b8_localization_fraction":float(full["full_b8_localization_fraction"]),
                "axis5_6_localization_fraction":float(pr["pair_localization_fraction_at_fixed_window"]),
                "axis5_6_fraction_of_full_optimum":float(pr["fraction_of_full_b8_optimum"]),
                "axis5_6_pair_rank_of_28":int(pr["pair_rank_at_fixed_window"]),
                "full_optimum_axis5_6_squared_weight":
                    float(full["axis5_6_energy_fraction_in_full_optimum"]),
            })

    result: Dict[str,Any] = {
        "schema_version":1,
        "work_package":"WF-P",
        "stage":"5B",
        "decision":"WFP_WITHIN_B8_LOCALIZATION_COMPLETE",
        "scientific_role":"post_discovery_descriptive_geometry_only",
        "waveform_arrays_opened":False,
        "clinical_labels_accessed":False,
        "frozen_dimension":8,
        "basis_changed":False,
        "notch_region_prespecified":False,
        "primary_analysis":{
            "scope":"all frozen B8 directions",
            "metrics":list(METRICS),
            "window_widths_points":list(WINDOW_WIDTHS),
            "method":"maximize operator-energy concentration in each contiguous phase window"
        },
        "targeted_axis5_6_analysis":{
            "status":"EXPLORATORY_POST_STAGE5_VISUAL_HYPOTHESIS",
            "pair":[5,6],
            "fair_context":"evaluated at the same full-B8-selected windows and ranked against all 28 axis pairs",
            "results":targeted
        },
        "full_b8_best_directions":directions_df.to_dict(orient="records"),
        "input_hashes":{
            "frozen_spec_sha256":sha256_file(spec_path),
            "stage5_result_sha256":sha256_file(stage5_path),
            "coordinates_sha256":sha256_file(coord_path),
            "patient_scores_sha256":sha256_file(scores_path)
        },
        "boundary":[
            "No localized direction is called dicrotic notch in this stage.",
            "Axis 5-6 targeted results are explicitly exploratory and were prompted after viewing Stage 5 axis plots.",
            "The all-B8 localization analysis is the primary unbiased geometric characterization.",
            "No clinical labels or new waveform data are accessed.",
            "The frozen B8 basis and d=8 are unchanged."
        ]
    }
    (out/"WFP_WITHIN_B8_LOCALIZATION.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    lines=[
        "WF-P LOCALIZED MORPHOLOGY WITHIN FROZEN B8",
        "==========================================",
        "Decision: WFP_WITHIN_B8_LOCALIZATION_COMPLETE",
        "Scientific role: POST-DISCOVERY DESCRIPTIVE GEOMETRY ONLY",
        "Waveform arrays opened: NO",
        "Clinical labels accessed: NO",
        "Frozen dimension: 8",
        "Primary basis changed: NO",
        "Notch region prespecified: NO",
        "",
        "Full-B8 best localized directions:",
    ]
    for _,r in directions_df.iterrows():
        lines.append(
            f"  {r['metric']} width={int(r['width_points'])}: "
            f"fraction={r['full_b8_localization_fraction']:.6f}; "
            f"phase={r['start_phase']:.6f}..{r['end_phase']:.6f}; "
            f"Axis5-6 squared weight={r['axis5_6_energy_fraction_in_full_optimum']:.6f}"
        )
    lines += [
        "",
        "Exploratory Axis 5-6 targeted check at the SAME full-B8-selected windows:",
    ]
    for r in targeted:
        lines.append(
            f"  {r['metric']} width={r['width_points']}: "
            f"Axis5-6 fraction={r['axis5_6_localization_fraction']:.6f}; "
            f"fraction of full-B8 optimum={r['axis5_6_fraction_of_full_optimum']:.6f}; "
            f"pair rank={r['axis5_6_pair_rank_of_28']}/28"
        )
    lines += [
        "",
        "Boundary:",
        "  Do NOT label any direction as dicrotic notch yet.",
        "  Axis 5-6 check is explicitly exploratory/post hoc.",
        "  If a localized descending-limb direction is compelling, define a separate notch/local-feature validation specification next.",
        "  Only after Stage 5B should Stage 6A outside-B8 residual localization be run.",
    ]
    (out/"WFP_WITHIN_B8_LOCALIZATION.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
