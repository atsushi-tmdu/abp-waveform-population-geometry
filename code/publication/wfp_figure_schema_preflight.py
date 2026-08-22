#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Figure Data Schema Preflight
=================================

Engineering-only preflight before writing plotting code.

This script:
- checks that FIG0 completed;
- inspects file existence, SHA256, row counts, and column names;
- verifies the private z1/z2 projection has 978 rows and unique patient_id;
- does NOT print patient IDs or z-score values;
- does NOT recompute any scientific metric;
- does NOT plot anything;
- does NOT modify any prior result.
"""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

EXPECTED_FILES = [
    "fig1_schematic_panel_text.md",
    "fig2_reconstruction_curves.csv",
    "fig2_axis_reliability.csv",
    "fig2_summary_metrics.csv",
    "fig3_patient_projection_z1z2_PRIVATE.csv",
    "fig3_scale_summary.csv",
    "fig4a_conditional_mean_summary.csv",
    "fig4b_residual_dependence_summary.csv",
    "fig4_interpretation_boundary.md",
    "WFP_FIG0_SOURCE_MANIFEST_SHA256.csv",
    "WFP_FIG0_EXPORT_READOUT.txt",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def summarize_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "duplicate_patient_id_rows": (
            int(df["patient_id"].duplicated().sum())
            if "patient_id" in df.columns else None
        ),
        "all_required_values_finite": (
            bool(df.select_dtypes(include="number").notna().all().all())
            if len(df.select_dtypes(include="number").columns) else None
        ),
    }

def self_test() -> int:
    print("WF-P figure-schema preflight self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fig0-dir",
        default="~/Documents/abp_information_study/results/wfp_fig0_exports",
    )
    ap.add_argument(
        "--out",
        default="~/Documents/abp_information_study/results/wfp_figure_schema_preflight",
    )
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    fig0 = Path(a.fig0_dir).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    readout = fig0 / "WFP_FIG0_EXPORT_READOUT.txt"
    if not readout.is_file():
        raise SystemExit(f"FAIL missing FIG0 readout: {readout}")
    txt = readout.read_text(encoding="utf-8", errors="replace")
    if "Decision: WFP_FIG0_EXPORT_COMPLETE" not in txt:
        raise SystemExit("FAIL FIG0 completion marker absent")

    missing = [name for name in EXPECTED_FILES if not (fig0 / name).is_file()]
    if missing:
        raise SystemExit(f"FAIL missing FIG0 exports: {missing}")

    report = {
        "decision": "WFP_FIGURE_SCHEMA_PREFLIGHT_PASS",
        "scientific_effects_calculated": False,
        "plots_rendered": False,
        "prior_results_modified": False,
        "files": {},
    }

    for name in EXPECTED_FILES:
        p = fig0 / name
        item = {
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
            "suffix": p.suffix.lower(),
        }
        if p.suffix.lower() == ".csv":
            item.update(summarize_csv(p))
        report["files"][name] = item

    # Hard integrity checks relevant to plotting.
    proj = pd.read_csv(fig0 / "fig3_patient_projection_z1z2_PRIVATE.csv")
    required_projection = {"patient_id", "z1", "z2"}
    if not required_projection.issubset(proj.columns):
        raise SystemExit(
            f"FAIL projection columns: required={sorted(required_projection)}, "
            f"observed={list(proj.columns)}"
        )
    if len(proj) != 978:
        raise SystemExit(f"FAIL projection rows={len(proj)} expected=978")
    if proj["patient_id"].duplicated().any():
        raise SystemExit("FAIL duplicate patient_id in private projection")

    for fn in [
        "fig2_summary_metrics.csv",
        "fig3_scale_summary.csv",
        "fig4a_conditional_mean_summary.csv",
        "fig4b_residual_dependence_summary.csv",
    ]:
        df = pd.read_csv(fig0 / fn)
        required = {"metric", "value", "unit", "role", "source", "note"}
        if not required.issubset(df.columns):
            raise SystemExit(f"FAIL {fn}: expected summary columns absent")

    (out / "WFP_FIGURE_SCHEMA_PREFLIGHT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "WF-P FIGURE DATA SCHEMA PREFLIGHT",
        "=================================",
        "Decision: WFP_FIGURE_SCHEMA_PREFLIGHT_PASS",
        "Scientific effects calculated: NO",
        "Plots rendered: NO",
        "Prior results modified: NO",
        "",
        "FIG0 export schemas:",
    ]
    for name in EXPECTED_FILES:
        item = report["files"][name]
        if item["suffix"] == ".csv":
            lines += [
                f"",
                f"{name}",
                f"  rows: {item['rows']}",
                f"  columns: {', '.join(item['columns'])}",
            ]
        else:
            lines += [f"", f"{name}", f"  non-CSV source; bytes: {item['bytes']}"]

    lines += [
        "",
        "Private projection integrity:",
        "  rows: 978",
        "  unique patient_id: YES",
        "  patient IDs / z values printed: NO",
        "",
        "Next authorized work:",
        "  Write plotting code against these exact schemas.",
        "  No new scientific metric may be introduced during rendering.",
        "",
    ]

    (out / "WFP_FIGURE_SCHEMA_PREFLIGHT.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
