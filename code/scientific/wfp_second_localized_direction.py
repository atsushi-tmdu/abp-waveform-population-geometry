#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 5C — second independent localized direction within frozen B8.

This is an explicitly exploratory follow-up defined after Stage 5B.

For each metric (shape, slope, curvature) and each already-frozen window width
(4, 8, 16 points):

1. Take the FIRST localized full-B8 direction c1 from Stage 5B as fixed.
2. Restrict the frozen B8 coefficient space to c^T c1 = 0.
   Because B8 is orthonormal, this is also orthogonality of waveform directions.
3. Re-scan every non-circular contiguous phase window.
4. Select the window/direction with the largest localization fraction.
5. STOP at the second direction. No third-or-higher direction is searched.

The second direction is allowed to localize at the same phase as the first:
independence is morphological-direction independence, not forced spatial separation.

No waveform reprocessing and no clinical labels are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        A = np.zeros((P-1, P), dtype=float)
        for i in range(P-1):
            A[i, i] = -1.0
            A[i, i+1] = 1.0
        return A
    if order == 2:
        A = np.zeros((P-2, P), dtype=float)
        for i in range(P-2):
            A[i, i] = 1.0
            A[i, i+1] = -2.0
            A[i, i+2] = 1.0
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
        return (np.arange(P-1, dtype=float) + 0.5) / P
    if metric == "curvature":
        return (np.arange(P-2, dtype=float) + 1.0) / P
    raise ValueError(metric)

def orthogonal_complement(c1: np.ndarray) -> np.ndarray:
    c1 = np.asarray(c1, float)
    c1 = c1 / np.linalg.norm(c1)
    _, _, vt = np.linalg.svd(c1.reshape(1, -1), full_matrices=True)
    N = vt[1:, :].T  # D x (D-1)
    if N.shape != (D, D-1):
        raise RuntimeError("orthogonal complement dimension mismatch")
    err = max(
        float(np.max(np.abs(N.T @ N - np.eye(D-1)))),
        float(np.max(np.abs(c1 @ N))),
    )
    if err > 1e-10:
        raise RuntimeError(f"orthogonal complement numerical failure: {err}")
    return N

def generalized_top_direction(
    basis: np.ndarray,
    operator: np.ndarray,
    start: int,
    width: int,
) -> Tuple[float, np.ndarray]:
    """
    Maximize ||M_W A basis a||^2 / ||A basis a||^2.
    Returns localization fraction and Euclidean-unit coefficients a.
    """
    Y = operator @ basis
    if start < 0 or start + width > Y.shape[0]:
        raise ValueError("window outside operator grid")

    G = Y.T @ Y
    Yw = Y[start:start+width, :]
    L = Yw.T @ Yw

    g, V = np.linalg.eigh(0.5*(G+G.T))
    scale = max(1.0, float(np.max(np.abs(g))))
    keep = g > 1e-12 * scale
    if not np.any(keep):
        return np.nan, np.full(basis.shape[1], np.nan)

    W = V[:, keep] @ np.diag(1.0/np.sqrt(g[keep]))
    H = W.T @ L @ W
    h, U = np.linalg.eigh(0.5*(H+H.T))
    a = W @ U[:, int(np.argmax(h))]
    na = float(np.linalg.norm(a))
    if na <= TOL:
        return np.nan, np.full(basis.shape[1], np.nan)
    a /= na

    y = Y @ a
    denom = float(np.dot(y, y))
    numer = float(np.dot(y[start:start+width], y[start:start+width]))
    frac = numer/denom if denom > 0 else np.nan

    k = int(np.argmax(np.abs(a)))
    if a[k] < 0:
        a = -a
    return float(frac), a

def score_sd(scores: pd.DataFrame, c: np.ndarray) -> float:
    Z = scores[[f"z{j}" for j in range(1,D+1)]].to_numpy(float)
    return float(np.std(Z @ c, ddof=1))

def window_overlap_fraction(
    first_start: int, second_start: int, width: int
) -> float:
    a = set(range(first_start, first_start+width))
    b = set(range(second_start, second_start+width))
    return len(a & b) / float(width)

def scan_second(
    B: np.ndarray,
    c1: np.ndarray,
    metric: str,
    width: int,
) -> pd.DataFrame:
    N = orthogonal_complement(c1)
    B2 = B @ N
    op = metric_operator(metric)
    phase = operator_phase(metric)

    rows = []
    for start in range(0, op.shape[0]-width+1):
        frac, a = generalized_top_direction(B2, op, start, width)
        c2 = N @ a
        c2 /= np.linalg.norm(c2)

        # Deterministic sign in original frozen-axis coordinates.
        k = int(np.argmax(np.abs(c2)))
        if c2[k] < 0:
            c2 = -c2

        rows.append({
            "metric": metric,
            "width_points": width,
            "start_index": start,
            "end_index": start+width-1,
            "start_phase": float(phase[start]),
            "end_phase": float(phase[start+width-1]),
            "center_phase": float(0.5*(phase[start]+phase[start+width-1])),
            "second_localization_fraction": frac,
            "abs_c1_dot_c2": float(abs(np.dot(c1, c2))),
            "axis5_6_squared_weight": float(c2[4]**2+c2[5]**2),
            **{f"c_axis{j+1}": float(c2[j]) for j in range(D)},
        })
    return pd.DataFrame(rows)

def self_test() -> int:
    rng = np.random.default_rng(20260820)
    A = rng.normal(size=(P,D))
    A -= np.mean(A,axis=0,keepdims=True)
    B,_ = np.linalg.qr(A)

    c1 = rng.normal(size=D)
    c1 /= np.linalg.norm(c1)
    df = scan_second(B,c1,"shape",8)
    row = df.loc[df["second_localization_fraction"].idxmax()]
    c2 = np.array([row[f"c_axis{j}"] for j in range(1,D+1)],float)
    if abs(np.dot(c1,c2)) > 1e-10:
        print("SELF-TEST FAIL: c2 not orthogonal to c1", file=sys.stderr)
        return 1
    print("WFP second localized direction self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-results", default="~/Documents/abp_information_study/results/wfp_discovery_validation1000")
    ap.add_argument("--stage5b-results", default="~/Documents/abp_information_study/results/wfp_within_b8_localization")
    ap.add_argument("--spec", default="~/Documents/abp_information_study/freeze/wfp_second_localized_direction/WFP_SECOND_LOCALIZED_DIRECTION_FROZEN_SPEC.json")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_second_localized_direction")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    discovery = Path(args.discovery_results).expanduser().resolve()
    stage5b = Path(args.stage5b_results).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SECOND_LOCALIZED_DIRECTION":
        raise SystemExit("FAIL: Stage 5C frozen spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("FAIL: Stage 5C analysis script hash mismatch")

    coord_path = discovery/"WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores_path = discovery/"wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    stage5b_json = stage5b/"WFP_WITHIN_B8_LOCALIZATION.json"
    first_dir_path = stage5b/"wfp_within_b8_localized_directions.csv"
    first_win_path = stage5b/"wfp_within_b8_localization_best_windows.csv"

    expected = {
        "coordinates_sha256": coord_path,
        "patient_scores_sha256": scores_path,
        "stage5b_result_sha256": stage5b_json,
        "stage5b_first_directions_sha256": first_dir_path,
        "stage5b_best_windows_sha256": first_win_path,
    }
    for key,p in expected.items():
        if not p.is_file():
            raise SystemExit(f"FAIL: required input missing: {p}")
        if spec.get(key) != sha256_file(p):
            raise SystemExit(f"FAIL: input hash mismatch: {key}")

    s5b = json.loads(stage5b_json.read_text(encoding="utf-8"))
    if s5b.get("decision") != "WFP_WITHIN_B8_LOCALIZATION_COMPLETE":
        raise SystemExit("FAIL: Stage 5B not complete")
    if int(s5b.get("frozen_dimension",-1)) != D:
        raise SystemExit("FAIL: frozen dimension is not 8")
    if s5b.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: clinical-label boundary violated")

    with np.load(coord_path,allow_pickle=False) as z:
        mu=np.asarray(z["population_mean"],float)
        B=np.asarray(z["between_basis"],float)
    scores=pd.read_csv(scores_path)
    first_dirs=pd.read_csv(first_dir_path)
    first_wins=pd.read_csv(first_win_path)

    if B.shape != (P,D) or mu.shape != (P,):
        raise SystemExit("FAIL: frozen coordinate shape invalid")
    if len(scores) != 978:
        raise SystemExit(f"FAIL: expected 978 score rows, found {len(scores)}")
    if np.max(np.abs(B.T@B-np.eye(D))) > 1e-10:
        raise SystemExit("FAIL: frozen B8 not orthonormal")

    all_scans=[]
    summary_rows=[]
    curve_df=pd.DataFrame({
        "phase_index":np.arange(P,dtype=int),
        "phase":np.arange(P,dtype=float)/P,
        "population_mean":mu
    })

    for metric in METRICS:
        for width in WINDOW_WIDTHS:
            frow=first_dirs[
                (first_dirs["metric"]==metric)&
                (first_dirs["width_points"]==width)
            ]
            wrow=first_wins[
                (first_wins["metric"]==metric)&
                (first_wins["width_points"]==width)
            ]
            if len(frow)!=1 or len(wrow)!=1:
                raise SystemExit(
                    f"FAIL: Stage5B first result not unique for {metric}, width={width}"
                )
            frow=frow.iloc[0]
            wrow=wrow.iloc[0]

            c1=np.array([frow[f"c_axis{j}"] for j in range(1,D+1)],float)
            c1/=np.linalg.norm(c1)
            first_fraction=float(frow["full_b8_localization_fraction"])
            first_start=int(wrow["start_index"])
            first_end=int(wrow["end_index"])
            first_sd=float(frow["score_sd"])

            scan=scan_second(B,c1,metric,width)
            all_scans.append(scan)
            brow=scan.loc[scan["second_localization_fraction"].idxmax()]
            c2=np.array([brow[f"c_axis{j}"] for j in range(1,D+1)],float)
            c2/=np.linalg.norm(c2)
            second_fraction=float(brow["second_localization_fraction"])
            second_sd=score_sd(scores,c2)
            second_start=int(brow["start_index"])
            second_end=int(brow["end_index"])

            if abs(np.dot(c1,c2)) > 1e-9:
                raise SystemExit(f"FAIL: c1/c2 orthogonality violated for {metric} w={width}")
            if second_fraction > first_fraction + 1e-8:
                raise SystemExit(
                    f"FAIL: constrained second localization exceeds first unexpectedly "
                    f"for {metric} w={width}"
                )

            phase_grid=operator_phase(metric)
            overlap=window_overlap_fraction(first_start,second_start,width)
            center_shift=float(
                0.5*(phase_grid[second_start]+phase_grid[second_end])
                - 0.5*(phase_grid[first_start]+phase_grid[first_end])
            )

            summary_rows.append({
                "metric":metric,
                "width_points":width,
                "first_localization_fraction":first_fraction,
                "second_localization_fraction":second_fraction,
                "second_to_first_fraction":second_fraction/first_fraction if first_fraction>0 else np.nan,
                "first_start_phase":float(phase_grid[first_start]),
                "first_end_phase":float(phase_grid[first_end]),
                "second_start_phase":float(brow["start_phase"]),
                "second_end_phase":float(brow["end_phase"]),
                "first_second_window_overlap_fraction":overlap,
                "second_minus_first_center_phase":center_shift,
                "abs_c1_dot_c2":float(abs(np.dot(c1,c2))),
                "second_score_sd":second_sd,
                "second_axis5_6_squared_weight":float(c2[4]**2+c2[5]**2),
                **{f"second_c_axis{j+1}":float(c2[j]) for j in range(D)},
            })

            tag=f"{metric}_w{width}"
            v1=B@c1
            v2=B@c2
            curve_df[f"{tag}_first_direction"]=v1
            curve_df[f"{tag}_second_direction"]=v2
            curve_df[f"{tag}_first_minus1sd"]=mu-first_sd*v1
            curve_df[f"{tag}_first_plus1sd"]=mu+first_sd*v1
            curve_df[f"{tag}_second_minus1sd"]=mu-second_sd*v2
            curve_df[f"{tag}_second_plus1sd"]=mu+second_sd*v2

    scan_df=pd.concat(all_scans,ignore_index=True)
    summary_df=pd.DataFrame(summary_rows)
    scan_df.to_csv(out/"wfp_second_localization_scan.csv",index=False)
    summary_df.to_csv(out/"wfp_second_localized_directions.csv",index=False)
    curve_df.to_csv(out/"wfp_second_localized_direction_curves.csv",index=False)

    # Width=8 first-vs-second morphology figure.
    fig,axes=plt.subplots(3,2,figsize=(11,11))
    phase=np.arange(P,dtype=float)/P
    for row,metric in enumerate(METRICS):
        width=8
        sr=summary_df[
            (summary_df["metric"]==metric)&
            (summary_df["width_points"]==width)
        ].iloc[0]
        fr=first_dirs[
            (first_dirs["metric"]==metric)&
            (first_dirs["width_points"]==width)
        ].iloc[0]

        c1=np.array([fr[f"c_axis{j}"] for j in range(1,D+1)],float)
        c1/=np.linalg.norm(c1)
        c2=np.array([sr[f"second_c_axis{j}"] for j in range(1,D+1)],float)
        c2/=np.linalg.norm(c2)
        sd1=float(fr["score_sd"])
        sd2=float(sr["second_score_sd"])
        v1=B@c1
        v2=B@c2

        ax=axes[row,0]
        ax.plot(phase,mu,label="Mean")
        ax.plot(phase,mu+sd1*v1,linestyle="--",label="+1 SD")
        ax.plot(phase,mu-sd1*v1,linestyle=":",label="-1 SD")
        ax.axvspan(
            float(sr["first_start_phase"]),
            float(sr["first_end_phase"]),
            alpha=0.15
        )
        ax.set_title(
            f"{metric}: 1st localized direction | loc={float(sr['first_localization_fraction']):.3f}"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("shape_norm")
        ax.legend()

        ax=axes[row,1]
        ax.plot(phase,mu,label="Mean")
        ax.plot(phase,mu+sd2*v2,linestyle="--",label="+1 SD")
        ax.plot(phase,mu-sd2*v2,linestyle=":",label="-1 SD")
        ax.axvspan(
            float(sr["second_start_phase"]),
            float(sr["second_end_phase"]),
            alpha=0.15
        )
        ax.set_title(
            f"{metric}: 2nd independent direction | loc={float(sr['second_localization_fraction']):.3f} "
            f"| overlap={float(sr['first_second_window_overlap_fraction']):.2f}"
        )
        ax.set_xlabel("Normalized beat phase")
        ax.set_ylabel("shape_norm")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out/"WFP_FIGURE_FIRST_VS_SECOND_LOCALIZED_WIDTH8.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"WFP_FIGURE_FIRST_VS_SECOND_LOCALIZED_WIDTH8.pdf",bbox_inches="tight")
    plt.close(fig)

    # Operator-energy profiles of the SECOND width=8 directions.
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=False)
    for ax,metric in zip(axes,METRICS):
        sr=summary_df[
            (summary_df["metric"]==metric)&
            (summary_df["width_points"]==8)
        ].iloc[0]
        c2=np.array([sr[f"second_c_axis{j}"] for j in range(1,D+1)],float)
        v2=B@c2
        op=metric_operator(metric)
        y=op@v2
        energy=y*y
        if np.sum(energy)>0:
            energy=energy/np.sum(energy)
        ph=operator_phase(metric)
        ax.plot(ph,energy)
        ax.axvspan(float(sr["second_start_phase"]),float(sr["second_end_phase"]),alpha=0.15)
        ax.set_ylabel("Energy fraction")
        ax.set_title(f"{metric}: 2nd independent direction operator-energy profile")
        ax.grid(False)
    axes[-1].set_xlabel("Normalized beat phase")
    fig.tight_layout()
    fig.savefig(out/"WFP_FIGURE_SECOND_DIRECTION_OPERATOR_ENERGY_WIDTH8.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"WFP_FIGURE_SECOND_DIRECTION_OPERATOR_ENERGY_WIDTH8.pdf",bbox_inches="tight")
    plt.close(fig)

    # Coefficients first vs second at width=8.
    fig,axes=plt.subplots(3,2,figsize=(10,10),sharex=True)
    for row,metric in enumerate(METRICS):
        fr=first_dirs[
            (first_dirs["metric"]==metric)&
            (first_dirs["width_points"]==8)
        ].iloc[0]
        sr=summary_df[
            (summary_df["metric"]==metric)&
            (summary_df["width_points"]==8)
        ].iloc[0]
        c1=np.array([fr[f"c_axis{j}"] for j in range(1,D+1)],float)
        c2=np.array([sr[f"second_c_axis{j}"] for j in range(1,D+1)],float)
        axes[row,0].bar(np.arange(1,D+1),c1)
        axes[row,0].set_title(f"{metric}: first coefficients")
        axes[row,0].set_ylabel("Coefficient")
        axes[row,0].grid(False)
        axes[row,1].bar(np.arange(1,D+1),c2)
        axes[row,1].set_title(
            f"{metric}: second coefficients | Axis5-6 weight="
            f"{float(sr['second_axis5_6_squared_weight']):.3f}"
        )
        axes[row,1].grid(False)
    axes[-1,0].set_xlabel("Frozen WF-P axis")
    axes[-1,1].set_xlabel("Frozen WF-P axis")
    fig.tight_layout()
    fig.savefig(out/"WFP_FIGURE_FIRST_SECOND_COEFFICIENTS_WIDTH8.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"WFP_FIGURE_FIRST_SECOND_COEFFICIENTS_WIDTH8.pdf",bbox_inches="tight")
    plt.close(fig)

    result: Dict[str,Any]={
        "schema_version":1,
        "work_package":"WF-P",
        "stage":"5C",
        "decision":"WFP_SECOND_LOCALIZED_DIRECTION_COMPLETE",
        "scientific_role":"explicitly_exploratory_post_stage5b_characterization",
        "waveform_arrays_opened":False,
        "clinical_labels_accessed":False,
        "frozen_dimension":8,
        "basis_changed":False,
        "directions_searched_beyond_second":False,
        "independence_definition":"c2 orthogonal to Stage5B c1 in frozen B8 coefficient/shape-L2 geometry",
        "spatial_separation_forced":False,
        "summary":summary_df.to_dict(orient="records"),
        "boundary":[
            "This analysis was defined after Stage 5B and is exploratory.",
            "Only the second independent localized direction is examined; no third or higher direction is searched.",
            "The second direction may occupy the same phase region as the first.",
            "No direction or phase interval is labeled dicrotic notch automatically.",
            "No clinical labels are accessed and the frozen B8 basis is unchanged."
        ],
        "input_hashes":{
            "frozen_spec_sha256":sha256_file(spec_path),
            "coordinates_sha256":sha256_file(coord_path),
            "patient_scores_sha256":sha256_file(scores_path),
            "stage5b_result_sha256":sha256_file(stage5b_json),
            "stage5b_first_directions_sha256":sha256_file(first_dir_path),
            "stage5b_best_windows_sha256":sha256_file(first_win_path),
        }
    }
    (out/"WFP_SECOND_LOCALIZED_DIRECTION.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    lines=[
        "WF-P SECOND INDEPENDENT LOCALIZED DIRECTION",
        "===========================================",
        "Decision: WFP_SECOND_LOCALIZED_DIRECTION_COMPLETE",
        "Scientific role: EXPLICITLY EXPLORATORY POST-STAGE5B CHARACTERIZATION",
        "Waveform arrays opened: NO",
        "Clinical labels accessed: NO",
        "Frozen dimension: 8",
        "Primary basis changed: NO",
        "Third-or-higher localized directions searched: NO",
        "Spatial separation from first window forced: NO",
        "",
        "First vs second independent localized directions:",
    ]
    for _,r in summary_df.iterrows():
        lines.append(
            f"  {r['metric']} width={int(r['width_points'])}: "
            f"first loc={r['first_localization_fraction']:.6f}; "
            f"second loc={r['second_localization_fraction']:.6f}; "
            f"second/first={r['second_to_first_fraction']:.6f}; "
            f"first phase={r['first_start_phase']:.6f}..{r['first_end_phase']:.6f}; "
            f"second phase={r['second_start_phase']:.6f}..{r['second_end_phase']:.6f}; "
            f"window overlap={r['first_second_window_overlap_fraction']:.3f}; "
            f"second Axis5-6 squared weight={r['second_axis5_6_squared_weight']:.6f}"
        )
    lines += [
        "",
        "Boundary:",
        "  Do NOT label the second direction as dicrotic notch from numbers alone.",
        "  Inspect the frozen width=8 figure once, then stop this branch.",
        "  After this, proceed to Stage 6A outside-B8 residual localization.",
    ]
    (out/"WFP_SECOND_LOCALIZED_DIRECTION.txt").write_text(
        "\n".join(lines)+"\n",encoding="utf-8"
    )
    print("\n".join(lines))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
