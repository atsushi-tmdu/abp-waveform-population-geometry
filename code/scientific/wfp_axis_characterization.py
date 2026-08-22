#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 5 — morphology-axis characterization.

Scientific role:
Descriptive characterization of the already-frozen d=8 WF-P discovery coordinate
system. No waveform arrays and no clinical labels are opened.

Primary displays:
- population mean morphology
- mean ± 1 empirical score SD along each frozen axis

Primary Fourier characterization:
- fraction of each axis lying in the SAME fixed Fourier d=8 subspace used in Stage 3
- residual axis after projection onto that fixed Fourier d=8 subspace

Important:
The displayed +/- curves are NOT renormalized after displacement. This preserves
the actual frozen linear coordinate geometry. Axis signs are arbitrary up to the
previous deterministic sign convention and must not be assigned physiological
"positive/negative" meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = 8
P = 64

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def principal_project(U: np.ndarray, v: np.ndarray) -> np.ndarray:
    return U @ (U.T @ v)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-results", required=True)
    ap.add_argument("--geometry-audit", required=True)
    ap.add_argument("--discovery-script", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = Path(args.discovery_results).expanduser().resolve()
    geometry_path = Path(args.geometry_audit).expanduser().resolve()
    discovery_script = Path(args.discovery_script).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_AXIS_CHARACTERIZATION":
        raise SystemExit("FAIL: Stage 5 frozen spec status invalid")
    if spec.get("analysis_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("FAIL: Stage 5 analysis script hash differs from frozen spec")
    if spec.get("discovery_script_sha256") != sha256_file(discovery_script):
        raise SystemExit("FAIL: discovery script hash differs from frozen spec")
    if spec.get("geometry_audit_sha256") != sha256_file(geometry_path):
        raise SystemExit("FAIL: geometry audit hash differs from frozen spec")

    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("decision") != "WFP_GEOMETRY_COMPARATOR_AUDIT_COMPLETE":
        raise SystemExit("FAIL: Stage 4 geometry audit not complete")
    if int(geometry.get("dimension_fixed", -1)) != D:
        raise SystemExit("FAIL: Stage 4 dimension is not 8")
    if geometry.get("clinical_labels_accessed") is not False:
        raise SystemExit("FAIL: Stage 4 clinical-label boundary violated")

    coord_path = results / "WFP_DISCOVERY_COMMON_COORDINATES.npz"
    scores_path = results / "wfp_patient_scores_DISCOVERY_PRIVATE.csv"
    rel_path = results / "wfp_axis_reliability.csv"
    disc_result_path = results / "WFP_DISCOVERY_RESULTS.json"

    for p in (coord_path, scores_path, rel_path, disc_result_path):
        if not p.is_file():
            raise SystemExit(f"FAIL: required input missing: {p}")

    if spec.get("coordinates_sha256") != sha256_file(coord_path):
        raise SystemExit("FAIL: coordinate file hash differs from frozen spec")
    if spec.get("patient_scores_sha256") != sha256_file(scores_path):
        raise SystemExit("FAIL: patient-score file hash differs from frozen spec")
    if spec.get("axis_reliability_sha256") != sha256_file(rel_path):
        raise SystemExit("FAIL: axis-reliability file hash differs from frozen spec")

    disc = load_module(discovery_script, "wfp_discovery_frozen")
    with np.load(coord_path, allow_pickle=False) as z:
        mean_shape = np.asarray(z["population_mean"], dtype=float)
        basis = np.asarray(z["between_basis"], dtype=float)
        eigvals = np.asarray(z["between_eigenvalues"], dtype=float)

    if mean_shape.shape != (P,) or basis.shape != (P, D) or eigvals.shape != (D,):
        raise SystemExit(
            f"FAIL: coordinate dimensions unexpected: mean={mean_shape.shape}, "
            f"basis={basis.shape}, eigvals={eigvals.shape}"
        )

    scores = pd.read_csv(scores_path)
    reliability = pd.read_csv(rel_path)
    zcols = [f"z{j}" for j in range(1, D + 1)]
    if any(c not in scores.columns for c in zcols):
        raise SystemExit("FAIL: patient score columns z1...z8 not found")
    if len(scores) != 978:
        raise SystemExit(f"FAIL: expected 978 discovery score rows, found {len(scores)}")

    score_sd = scores[zcols].std(axis=0, ddof=1).to_numpy(float)
    score_mean = scores[zcols].mean(axis=0).to_numpy(float)
    replicate_sd = np.sqrt(np.clip(eigvals, 0.0, None))

    fourier_full = np.asarray(disc.fourier_basis(), dtype=float)
    if fourier_full.shape[0] != P or fourier_full.shape[1] < D:
        raise SystemExit("FAIL: frozen Fourier basis shape invalid")
    F8 = fourier_full[:, :D]

    phase = np.arange(P, dtype=float) / P
    curve_df = pd.DataFrame({"phase": phase, "population_mean": mean_shape})
    rows = []
    low_components = np.zeros_like(basis)
    residual_components = np.zeros_like(basis)

    rel_map = {
        int(r["axis"]): r
        for _, r in reliability.iterrows()
    }

    for j in range(D):
        b = basis[:, j]
        low = principal_project(F8, b)
        residual = b - low
        low_components[:, j] = low
        residual_components[:, j] = residual

        total_energy = float(np.dot(b, b))
        low_energy = float(np.dot(low, low))
        residual_energy = float(np.dot(residual, residual))
        low_fraction = low_energy / total_energy if total_energy > 0 else np.nan

        sd = float(score_sd[j])
        plus = mean_shape + sd * b
        minus = mean_shape - sd * b

        curve_df[f"axis{j+1}_minus1sd"] = minus
        curve_df[f"axis{j+1}_plus1sd"] = plus
        curve_df[f"axis{j+1}_basis"] = b
        curve_df[f"axis{j+1}_fourier_d8_component"] = low
        curve_df[f"axis{j+1}_nonfourier_residual"] = residual

        rr = rel_map.get(j + 1, {})
        rows.append({
            "axis": j + 1,
            "empirical_score_mean": float(score_mean[j]),
            "empirical_score_sd": sd,
            "sqrt_replicate_eigenvalue": float(replicate_sd[j]),
            "odd_even_score_correlation": float(rr.get("odd_even_score_correlation", np.nan)),
            "fourier_d8_fraction_of_axis_energy": low_fraction,
            "nonfourier_fraction_of_axis_energy":
                residual_energy / total_energy if total_energy > 0 else np.nan,
            "max_abs_basis_phase_index": int(np.argmax(np.abs(b))),
            "max_abs_basis_phase": float(phase[int(np.argmax(np.abs(b)))]),
        })

    table = pd.DataFrame(rows)
    curve_df.to_csv(out / "wfp_axis_shape_curves.csv", index=False)
    table.to_csv(out / "wfp_axis_characterization.csv", index=False)

    np.savez_compressed(
        out / "WFP_AXIS_FOURIER_DECOMPOSITION.npz",
        population_mean=mean_shape.astype(np.float64),
        between_basis=basis.astype(np.float64),
        fourier_d8_basis=F8.astype(np.float64),
        fourier_d8_components=low_components.astype(np.float64),
        nonfourier_residual_components=residual_components.astype(np.float64),
        empirical_score_sd=score_sd.astype(np.float64),
    )

    # One overview figure. These are descriptive morphology curves, not hypothesis tests.
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
    axes = np.asarray(axes).ravel()
    for j, ax in enumerate(axes):
        sd = float(score_sd[j])
        ax.plot(phase, mean_shape, linewidth=1.4, label="Population mean")
        ax.plot(phase, mean_shape + sd * basis[:, j], linewidth=1.2, linestyle="--", label="+1 SD")
        ax.plot(phase, mean_shape - sd * basis[:, j], linewidth=1.2, linestyle=":", label="-1 SD")
        frac = float(table.loc[table["axis"] == j+1, "fourier_d8_fraction_of_axis_energy"].iloc[0])
        rel = float(table.loc[table["axis"] == j+1, "odd_even_score_correlation"].iloc[0])
        ax.set_title(f"Axis {j+1} | Fourier-d8 fraction={frac:.3f} | odd/even r={rel:.3f}")
        ax.set_ylabel("shape_norm")
        ax.grid(False)
    axes[-1].set_xlabel("Normalized beat phase")
    axes[-2].set_xlabel("Normalized beat phase")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "WFP_FIGURE_AXIS_SHAPES.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "WFP_FIGURE_AXIS_SHAPES.pdf", bbox_inches="tight")
    plt.close(fig)

    # Axis-wise low-Fourier fraction.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(table["axis"].astype(str), table["fourier_d8_fraction_of_axis_energy"])
    ax.set_xlabel("Frozen WF-P axis")
    ax.set_ylabel("Fraction of axis energy in fixed Fourier d=8 subspace")
    ax.set_ylim(0, 1.02)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / "WFP_FIGURE_AXIS_FOURIER_FRACTION.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "WFP_FIGURE_AXIS_FOURIER_FRACTION.pdf", bbox_inches="tight")
    plt.close(fig)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": 5,
        "decision": "WFP_AXIS_CHARACTERIZATION_COMPLETE",
        "scientific_role": "descriptive_characterization_of_frozen_discovery_basis",
        "dimension": D,
        "waveform_arrays_opened": False,
        "clinical_labels_accessed": False,
        "dimension_reselected": False,
        "basis_changed": False,
        "display_rule": "population mean +/- 1 empirical population score SD along each frozen axis; no renormalization",
        "axis_sign_interpretation": "arbitrary up to frozen deterministic sign convention",
        "fourier_reference": "same fixed Fourier d=8 subspace used in Stage 3 comparator",
        "axis_table": rows,
        "input_hashes": {
            "stage5_spec_sha256": sha256_file(spec_path),
            "geometry_audit_sha256": sha256_file(geometry_path),
            "discovery_script_sha256": sha256_file(discovery_script),
            "coordinates_sha256": sha256_file(coord_path),
            "patient_scores_sha256": sha256_file(scores_path),
            "axis_reliability_sha256": sha256_file(rel_path),
        },
        "boundary": [
            "No axis is assigned a physiological or clinical name at this stage.",
            "No age, sex, diagnosis, treatment, outcome, lactate, or other clinical labels are accessed.",
            "Axis signs are not physiological directions.",
            "The frozen d=8 coordinate system remains unchanged.",
        ]
    }
    result_path = out / "WFP_AXIS_CHARACTERIZATION.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "WF-P FROZEN AXIS CHARACTERIZATION",
        "================================",
        "Decision: WFP_AXIS_CHARACTERIZATION_COMPLETE",
        "Scientific role: DESCRIPTIVE CHARACTERIZATION OF FROZEN DISCOVERY BASIS",
        "Dimension: 8",
        "Waveform arrays opened: NO",
        "Clinical labels accessed: NO",
        "Dimension reselected: NO",
        "Primary basis changed: NO",
        "",
        "Axis summary:",
    ]
    for r in rows:
        lines.append(
            f"  Axis {r['axis']}: score SD={r['empirical_score_sd']:.6f}; "
            f"sqrt(rep eig)={r['sqrt_replicate_eigenvalue']:.6f}; "
            f"odd/even r={r['odd_even_score_correlation']:.6f}; "
            f"Fourier-d8 fraction={r['fourier_d8_fraction_of_axis_energy']:.6f}; "
            f"non-Fourier fraction={r['nonfourier_fraction_of_axis_energy']:.6f}"
        )
    lines += [
        "",
        "Display rule:",
        "  population mean +/- 1 empirical population score SD along each frozen axis",
        "  curves are NOT renormalized after displacement",
        "",
        "Boundary:",
        "  Do not assign physiological or clinical names to axes yet.",
        "  Do not interpret + versus - as beneficial/pathological or high/low physiology.",
        "  Next clinical/covariate work requires a separate frozen linkage specification.",
    ]
    (out / "WFP_AXIS_CHARACTERIZATION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
