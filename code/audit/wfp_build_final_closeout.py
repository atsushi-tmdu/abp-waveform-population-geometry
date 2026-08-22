#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P final scientific closeout builder
======================================

Purpose
-------
Create a source-driven final closeout package AFTER all declared WF-P analyses
have completed. This utility does not run any new scientific analysis.

It:
1. verifies the expected terminal decisions from discovery / WFP0 / Stage7B /
   scale audit / Stage7C / Stage7B-NL / Stage7B-RD;
2. parses the already-reported authoritative readouts;
3. writes a single authoritative results summary (CSV/JSON/MD);
4. writes a final scientific status document;
5. writes a figure manifest and figure-QC checklist;
6. writes a SHA256 source manifest.

It does NOT:
- open raw waveform arrays;
- fit a model;
- change B8;
- modify any prior result;
- create a new scientific effect;
- relabel any axis as Ztrait or Zstate.

Interpretation boundary carried into closeout
---------------------------------------------
The measured constitutional attributes showed weak OOF conditional-mean
predictability across the frozen linear Stage7B models and the separate
low-df nonlinear Stage7B-NL sensitivity. However, residual model-free
dependence persisted after Stage7B-RD pipeline/fold artifact controls.
Therefore the final closeout must NOT claim statistical independence or
complete orthogonality of B8 from age/sex/height.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EXPECTED = {
    "discovery": "WFP_VALIDATION1000_DISCOVERY",
    "stage7b": "WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE",
    "scale": "WFP_BETWEEN_WITHIN_SCALE_AUDIT_COMPLETE",
    "stage7c": "WFP_STAGE7C_ASSOCIATION_COMPLETE",
    "stage7b_nl": "NO_MATERIAL_OOF_MEAN_GAIN_BUT_RESIDUAL_MODEL_FREE_DEPENDENCE_FLAG",
    "stage7b_rd": "RESIDUAL_DEPENDENCE_PERSISTS_AFTER_PIPELINE_AND_FOLD_CONTROLS",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def find_one(root: Path, patterns: Iterable[str], label: str) -> Path:
    hits: List[Path] = []
    for pattern in patterns:
        hits.extend(sorted(root.glob(pattern)))
    # unique exact paths
    uniq = []
    seen = set()
    for p in hits:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    if len(uniq) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one file, found {len(uniq)}: "
            + ", ".join(str(x) for x in uniq)
        )
    return uniq[0]

def get_float(txt: str, pattern: str, label: str) -> float:
    m = re.search(pattern, txt, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not parse {label} with pattern {pattern!r}")
    return float(m.group(1))

def get_int(txt: str, pattern: str, label: str) -> int:
    return int(round(get_float(txt, pattern, label)))

def metric(section: str, key: str, value, unit: str, role: str, source: str, note: str="") -> Dict[str, object]:
    return {
        "section": section,
        "metric": key,
        "value": value,
        "unit": unit,
        "role": role,
        "source": source,
        "note": note,
    }

def assert_contains(txt: str, needle: str, label: str) -> None:
    if needle not in txt:
        raise RuntimeError(f"{label}: required marker absent: {needle}")

def self_test() -> int:
    d = """WF-P VALIDATION1000 DISCOVERY READOUT
Decision: WFP_VALIDATION1000_DISCOVERY_COMPLETE
Source n: 1000
Analysable n: 978
Primary replicate-corrected positive spectrum:
  effective rank: 5.78163
  d90: 6
  d95: 8
Selected common-basis dimension: 8
  CV R2 all: 0.964044
  Half-split subspace overlap median: 0.97767
  Within-window variance captured by between basis: 0.92315
  Between/within projector overlap: 0.94511
"""
    if get_int(d, r"Analysable n:\s*(\d+)", "analysable") != 978:
        raise RuntimeError("parser self-test failed")
    if abs(get_float(d, r"CV R2 all:\s*([+\-0-9.eE]+)", "cv") - .964044) > 1e-12:
        raise RuntimeError("parser self-test failed")
    print("WF-P final closeout builder self-test: PASS")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_final_closeout")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    root = Path(a.project_root).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Resolve authoritative terminal readouts. Exact known names first; fallback
    # patterns are confined to the declared result directory only.
    source_paths = {}

    discovery_dir = root / "results" / "wfp_discovery_validation1000"
    source_paths["discovery_readout"] = find_one(
        discovery_dir,
        ["WFP_DISCOVERY_READOUT.txt", "*DISCOVERY*READOUT*.txt"],
        "discovery readout",
    )

    # WFP0 is intentionally discovered within its frozen result directory because
    # historical naming varied slightly across engineering revisions.
    wfp0_dir = root / "results" / "wfp0_minimal_identifiability"
    source_paths["wfp0_readout"] = find_one(
        wfp0_dir,
        ["*READOUT*.txt", "*readout*.txt"],
        "WFP0 readout",
    )

    source_paths["stage7b_readout"] = (
        root / "results" / "wfp_stage7b_constitutional" /
        "WFP_STAGE7B_CONSTITUTIONAL_READOUT.txt"
    )
    source_paths["scale_readout"] = (
        root / "results" / "wfp_between_within_scale" /
        "WFP_BETWEEN_WITHIN_SCALE_READOUT.txt"
    )
    source_paths["stage7c_readout"] = (
        root / "results" / "wfp_stage7c_association" /
        "WFP_STAGE7C_ASSOCIATION_READOUT.txt"
    )
    source_paths["stage7b_nl_readout"] = (
        root / "results" / "wfp_stage7b_nl" /
        "WFP_STAGE7B_NL_READOUT.txt"
    )
    source_paths["stage7b_rd_readout"] = (
        root / "results" / "wfp_stage7b_rd" /
        "WFP_STAGE7B_RD_READOUT.txt"
    )

    # Frozen specs/results also enter the source-manifest when present.
    optional_paths = {
        "discovery_results_json": discovery_dir / "WFP_DISCOVERY_RESULTS.json",
        "stage7b_spec": root / "freeze" / "wfp_stage7b_constitutional" / "WFP_STAGE7B_CONSTITUTIONAL_FROZEN_SPEC.json",
        "stage7b_results_json": root / "results" / "wfp_stage7b_constitutional" / "WFP_STAGE7B_CONSTITUTIONAL_RESULTS.json",
        "scale_results_json": root / "results" / "wfp_between_within_scale" / "WFP_BETWEEN_WITHIN_SCALE_RESULTS.json",
        "stage7c_spec": root / "freeze" / "wfp_stage7c_association" / "WFP_STAGE7C_ASSOCIATION_FROZEN_SPEC.json",
        "stage7c_results_json": root / "results" / "wfp_stage7c_association" / "WFP_STAGE7C_ASSOCIATION_RESULTS.json",
        "stage7b_nl_spec": root / "freeze" / "wfp_stage7b_nl" / "WFP_STAGE7B_NL_FROZEN_SPEC.json",
        "stage7b_nl_results_json": root / "results" / "wfp_stage7b_nl" / "WFP_STAGE7B_NL_RESULTS.json",
        "stage7b_rd_spec": root / "freeze" / "wfp_stage7b_rd" / "WFP_STAGE7B_RD_FROZEN_SPEC.json",
        "stage7b_rd_results_json": root / "results" / "wfp_stage7b_rd" / "WFP_STAGE7B_RD_RESULTS.json",
    }

    for label, p in source_paths.items():
        if not p.is_file():
            raise SystemExit(f"FAIL missing authoritative source: {label}: {p}")

    texts = {k: read_text(p) for k, p in source_paths.items()}

    # Terminal decision checks.
    assert_contains(texts["discovery_readout"], "Analysable n: 978", "discovery")
    assert_contains(texts["stage7b_readout"], EXPECTED["stage7b"], "Stage7B")
    assert_contains(texts["scale_readout"], EXPECTED["scale"], "scale audit")
    assert_contains(texts["stage7c_readout"], EXPECTED["stage7c"], "Stage7C")
    assert_contains(texts["stage7b_nl_readout"], EXPECTED["stage7b_nl"], "Stage7B-NL")
    assert_contains(texts["stage7b_rd_readout"], EXPECTED["stage7b_rd"], "Stage7B-RD")

    # WFP0 has historical naming; pin by scientific markers instead.
    assert_contains(texts["wfp0_readout"], "0.136957", "WFP0 conventional OOF R2")
    assert_contains(texts["wfp0_readout"], "0.863038", "WFP0 residual trace fraction")

    rows: List[Dict[str, object]] = []

    d = texts["discovery_readout"]
    rows += [
        metric("population_geometry","source_n",get_int(d,r"Source n:\s*(\d+)","source n"),"patients","authoritative","discovery"),
        metric("population_geometry","analysable_n",get_int(d,r"Analysable n:\s*(\d+)","analysable n"),"patients","authoritative","discovery"),
        metric("population_geometry","effective_rank",get_float(d,r"effective rank:\s*([+\-0-9.eE]+)","effective rank"),"dimensionless","authoritative","discovery"),
        metric("population_geometry","d90",get_int(d,r"d90:\s*(\d+)","d90"),"dimensions","authoritative","discovery"),
        metric("population_geometry","d95",get_int(d,r"d95:\s*(\d+)","d95"),"dimensions","authoritative","discovery"),
        metric("population_geometry","selected_B8_dimension",get_int(d,r"Selected common-basis dimension:\s*(\d+)","B8 d"),"dimensions","authoritative","discovery"),
        metric("population_geometry","cv_r2_all",get_float(d,r"CV R2 all:\s*([+\-0-9.eE]+)","cv r2"),"R2","authoritative","discovery"),
        metric("population_geometry","half_split_overlap_median",get_float(d,r"Half-split subspace overlap median:\s*([+\-0-9.eE]+)","half split"),"overlap","authoritative","discovery"),
        metric("population_geometry","within_variance_captured_by_between_basis",get_float(d,r"Within-window variance captured by between basis:\s*([+\-0-9.eE]+)","within capture"),"fraction","authoritative","discovery"),
        metric("population_geometry","between_within_projector_overlap",get_float(d,r"Between/within projector overlap:\s*([+\-0-9.eE]+)","projector overlap"),"overlap","authoritative","discovery"),
    ]

    w0 = texts["wfp0_readout"]
    rows += [
        metric("representation_identifiability","conventional_to_B8_oof_r2",0.136957,"R2","authoritative","WFP0","level + log(scale) + log(duration)"),
        metric("representation_identifiability","conditioned_residual_trace_fraction",0.863038,"fraction","authoritative","WFP0"),
    ]

    s7 = texts["stage7b_readout"]
    rows += [
        metric("constitutional_mean","age_sex_oof_r2",get_float(s7,r"aggregate OOF R2:\s*([+\-0-9.eE]+)","Stage7B age sex"),"R2","authoritative","Stage7B"),
        metric("constitutional_mean","age_sex_residual_trace_fraction",get_float(s7,r"residual trace fraction:\s*([+\-0-9.eE]+)","Stage7B residual trace"),"fraction","authoritative","Stage7B"),
        metric("constitutional_mean","quadratic_age_sex_oof_r2",get_float(s7,r"age \+ age\^2 \+ sex OOF R2:\s*([+\-0-9.eE]+)","quadratic"),"R2","sensitivity","Stage7B"),
        metric("constitutional_mean","conventional_plus_age_sex_oof_r2",get_float(s7,r"conventional \+ age \+ sex OOF R2:\s*([+\-0-9.eE]+)","conv age sex"),"R2","secondary","Stage7B"),
        metric("constitutional_mean","age_sex_increment_beyond_conventional",get_float(s7,r"incremental age\+sex delta R2:\s*([+\-0-9.eE]+)","incremental age sex"),"delta_R2","secondary","Stage7B"),
        metric("constitutional_mean","height_complete_case_n",get_int(s7,r"Height complete-case n:\s*(\d+)","height n"),"patients","authoritative","Stage7B"),
        metric("constitutional_mean","height_delta_beyond_age_sex",get_float(s7,r"height delta R2 beyond age\+sex:\s*([+\-0-9.eE]+)","height delta"),"delta_R2","secondary","Stage7B"),
    ]

    sc = texts["scale_readout"]
    rows += [
        metric("between_within_scale","between_pairwise_rms",get_float(sc,r"pairwise RMS distance:\s*([+\-0-9.eE]+)","between rms"),"B8_distance","authoritative","scale audit"),
        metric("between_within_scale","between_pairwise_median",get_float(sc,r"pairwise distance median:\s*([+\-0-9.eE]+)","between median"),"B8_distance","authoritative","scale audit"),
        metric("between_within_scale","nearest_neighbor_median",get_float(sc,r"nearest-neighbor distance median:\s*([+\-0-9.eE]+)","nn median"),"B8_distance","authoritative","scale audit"),
        metric("between_within_scale","within_equal_patient_rms",get_float(sc,r"equal-patient within RMS:\s*([+\-0-9.eE]+)","within rms"),"B8_distance","authoritative","scale audit"),
        metric("between_within_scale","adjacent_60s_step_median",get_float(sc,r"adjacent 60-s step median:\s*([+\-0-9.eE]+)","adjacent"),"B8_distance","authoritative","scale audit"),
        metric("between_within_scale","odd_even_replicate_rms",get_float(sc,r"Odd/even replicate discrepancy:\s*\n\s*RMS distance:\s*([+\-0-9.eE]+)","replicate rms"),"B8_distance","authoritative","scale audit"),
        metric("between_within_scale","between_over_replicate",get_float(sc,r"between pairwise RMS / replicate RMS:\s*([+\-0-9.eE]+)","between rep ratio"),"ratio","authoritative","scale audit"),
        metric("between_within_scale","within_over_between",get_float(sc,r"within RMS / between pairwise RMS:\s*([+\-0-9.eE]+)","within between ratio"),"ratio","authoritative","scale audit"),
        metric("between_within_scale","within_over_replicate",get_float(sc,r"within RMS / replicate RMS:\s*([+\-0-9.eE]+)","within rep ratio"),"ratio","authoritative","scale audit"),
        metric("between_within_scale","p95_block_displacement_ge_nn_fraction",get_float(sc,r"patients with p95 block displacement >= NN:\s*([+\-0-9.eE]+)","p95 nn"),"fraction","secondary","scale audit"),
    ]

    c = texts["stage7c_readout"]
    rows += [
        metric("chronic_phenotype_mapping","analysis_n",get_int(c,r"Analysis n:\s*(\d+)","stage7c n"),"patients","authoritative","Stage7C"),
        metric("chronic_phenotype_mapping","baseline_oof_r2",get_float(c,r"baseline OOF R2:\s*([+\-0-9.eE]+)","stage7c baseline"),"R2","authoritative","Stage7C"),
        metric("chronic_phenotype_mapping","joint_8phenotype_oof_r2",get_float(c,r"joint 8-phenotype OOF R2:\s*([+\-0-9.eE]+)","stage7c joint"),"R2","authoritative","Stage7C"),
        metric("chronic_phenotype_mapping","phenotype_block_delta_oof_r2",get_float(c,r"phenotype-block incremental OOF R2:\s*([+\-0-9.eE]+)","stage7c delta"),"delta_R2","authoritative","Stage7C"),
        metric("chronic_phenotype_mapping","FDR_significant_global_phenotypes",0,"count","authoritative","Stage7C","8 prespecified global phenotype tests"),
    ]

    nl = texts["stage7b_nl_readout"]
    rows += [
        metric("constitutional_nonlinear_sensitivity","M0_linear_age_sex_oof_r2",get_float(nl,r"M0 linear age \+ sex OOF R2:\s*([+\-0-9.eE]+)","M0"),"R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","M1_RCS_age_sex_oof_r2",get_float(nl,r"M1 RCS4\(age\) \+ sex OOF R2:\s*([+\-0-9.eE]+)","M1"),"R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","M1_minus_M0",get_float(nl,r"delta M1-M0:\s*([+\-0-9.eE]+)","M1-M0"),"delta_R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","M2_RCS_age_by_sex_oof_r2",get_float(nl,r"M2 RCS4\(age\) \* sex OOF R2:\s*([+\-0-9.eE]+)","M2"),"R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","M2_minus_M0",get_float(nl,r"delta M2-M0 total nonlinear\+interaction:\s*([+\-0-9.eE]+)","M2-M0"),"delta_R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","H2_RCS_age_height_oof_r2",get_float(nl,r"H2 RCS4\(age\) \+ sex \+ RCS4\(height\) OOF R2:\s*([+\-0-9.eE]+)","H2"),"R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","nonlinear_height_increment",get_float(nl,r"nonlinear height increment H2-H1h:\s*([+\-0-9.eE]+)","height NL"),"delta_R2","sensitivity","Stage7B-NL"),
        metric("constitutional_nonlinear_sensitivity","height_by_sex_increment",get_float(nl,r"height-by-sex increment H3-H2:\s*([+\-0-9.eE]+)","height sex"),"delta_R2","sensitivity","Stage7B-NL"),
        metric("constitutional_residual_dependence","full_residual_dcor",get_float(nl,r"Full age\+sex vs OOF residual B8:\s*dCor=([+\-0-9.eE]+)","full dcor"),"dCor","sensitivity","Stage7B-NL"),
        metric("constitutional_residual_dependence","height_residual_dcor",get_float(nl,r"Height age\+sex\+height vs OOF residual B8:\s*dCor=([+\-0-9.eE]+)","height dcor"),"dCor","sensitivity","Stage7B-NL"),
    ]

    rd = texts["stage7b_rd_readout"]
    rows += [
        metric("constitutional_residual_dependence","full_pipeline_null_q95",get_float(rd,r"pipeline-replay null median/q95:\s*[+\-0-9.eE]+\s*/\s*([+\-0-9.eE]+)","full pipeline q95"),"dCor","artifact_control","Stage7B-RD"),
        metric("constitutional_residual_dependence","full_pipeline_holm_p",get_float(rd,r"pipeline-replay Holm p:\s*([+\-0-9.eE]+)","full holm"),"p","artifact_control","Stage7B-RD"),
        metric("constitutional_residual_dependence","height_pipeline_null_q95",get_float(rd,r"Height complete-case n=693:\s*[\s\S]*?pipeline-replay null median/q95:\s*[+\-0-9.eE]+\s*/\s*([+\-0-9.eE]+)","height pipeline q95"),"dCor","artifact_control","Stage7B-RD"),
    ]
    rows += [
        metric("constitutional_residual_dependence","height_pipeline_holm_p",get_float(rd,r"Height complete-case n=693:\s*[\s\S]*?pipeline-replay Holm p:\s*([+\-0-9.eE]+)","height holm"),"p","artifact_control","Stage7B-RD"),
        metric("constitutional_residual_dependence","full_within_fold_p",get_float(rd,r"Full cohort n=978:\s*[\s\S]*?within-fold diagnostic p:\s*([+\-0-9.eE]+)","full blocked p"),"p","artifact_control","Stage7B-RD"),
        metric("constitutional_residual_dependence","height_within_fold_p",get_float(rd,r"Height complete-case n=693:\s*[\s\S]*?within-fold diagnostic p:\s*([+\-0-9.eE]+)","height blocked p"),"p","artifact_control","Stage7B-RD"),
    ]

    # Write authoritative summary CSV.
    csvp = out / "WFP_AUTHORITATIVE_RESULTS_SUMMARY.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section","metric","value","unit","role","source","note"])
        w.writeheader()
        w.writerows(rows)

    # JSON.
    result_json = {
        "decision": "WFP_FINAL_CLOSEOUT_COMPLETE",
        "scientific_effect_analysis_added": False,
        "frozen_B8_changed": False,
        "metrics": rows,
        "interpretation_boundary": {
            "supported": [
                "A reproducible low-dimensional population-common morphology space was identified in the WF-P discovery cohort.",
                "Patient central-morphology points are broadly separated relative to short-window replicate discrepancy.",
                "Short-window within-patient movement is non-negligible but smaller than between-patient separation.",
                "Measured age/sex/height provided little OOF conditional-mean predictability across prespecified linear and low-df nonlinear models.",
                "Coarse chronic phenotype labels did not improve OOF prediction beyond the prespecified baseline.",
                "Residual model-free dependence on constitutional variables remained detectable after pipeline/fold artifact controls.",
            ],
            "not_supported": [
                "B8 is statistically independent of age/sex/height.",
                "B8 is completely orthogonal to constitutional attributes.",
                "Any B8 axis is a stable trait or physiological state.",
                "Residual constitutional dependence is specifically a nonlinear mean effect.",
                "Residual constitutional dependence has been localized to variance/covariance structure.",
                "Disease-specific or treatment-response meaning of B8.",
                "External generalizability beyond the studied MIMIC setting.",
            ],
        },
    }
    (out / "WFP_AUTHORITATIVE_RESULTS_SUMMARY.json").write_text(
        json.dumps(result_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Markdown summary.
    md = [
        "# WF-P Authoritative Results Summary",
        "",
        "**Status:** FINAL CLOSEOUT SOURCE SUMMARY",
        "",
        "This document aggregates already-completed WF-P analyses. It does not introduce a new scientific effect.",
        "",
        "## Core result hierarchy",
        "",
        "1. A low-dimensional patient-balanced population morphology space was identified; the frozen interface dimension is B8.",
        "2. Between-patient separation is much larger than odd/even replicate discrepancy, while 60-s within-patient movement is non-negligible.",
        "3. Conventional level/scale/timing variables and measured constitutional variables provide limited OOF mean predictability for B8.",
        "4. Low-df nonlinear age/height models and limited interactions do not materially improve OOF mean prediction.",
        "5. Residual constitutional dependence remains detectable after pipeline-replay and within-fold artifact controls; therefore independence/orthogonality is NOT claimed.",
        "6. Eight prespecified coarse chronic phenotypes add no OOF predictive value as a block, and no global phenotype association survives BH-FDR 0.05.",
        "",
        "## Selected authoritative numbers",
        "",
        "| Section | Metric | Value |",
        "|---|---|---:|",
    ]
    selected_keys = [
        "effective_rank","d95","cv_r2_all","half_split_overlap_median",
        "conventional_to_B8_oof_r2","age_sex_oof_r2",
        "between_pairwise_rms","within_equal_patient_rms","odd_even_replicate_rms",
        "M1_minus_M0","M2_minus_M0","full_residual_dcor","height_residual_dcor",
        "phenotype_block_delta_oof_r2","FDR_significant_global_phenotypes",
    ]
    bykey = {r["metric"]: r for r in rows}
    for k in selected_keys:
        r = bykey[k]
        md.append(f"| {r['section']} | {k} | {r['value']} |")
    md += [
        "",
        "## Final wording boundary",
        "",
        "Recommended wording:",
        "",
        "> The frozen B8 coordinates showed little out-of-sample mean predictability from age, sex, and height across prespecified linear, low-degree nonlinear, and limited-interaction models. However, model-free dependence of the residual B8 coordinates on these constitutional variables remained detectable after pipeline-replay and fold-stratified artifact controls.",
        "",
        "Do **not** write that B8 is independent of, unrelated to, or fully orthogonal to age/sex/height.",
        "",
        "## WF3 interface",
        "",
        "WF-P does not label any axis as Ztrait or Zstate. The next scientific question is whether the frozen coordinates show reproducible same-patient longitudinal movement and whether that movement maps to time-varying physiology/treatment/recovery.",
    ]
    (out / "WFP_AUTHORITATIVE_RESULTS_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Final scientific status.
    status = """# WF-P Final Scientific Status

## Decision

**WFP_FINAL_SCIENTIFIC_CLOSEOUT: COMPLETE**

No further WF-P scientific-effect analysis is authorized as part of this closeout.

## Completed stages

- Population-common B8 discovery / derivation
- Representation-identifiability audit (WFP0)
- Constitutional mapping (Stage 7B)
- Between–within scale audit
- Chronic phenotype mapping (Stage 7C)
- Nonlinear constitutional sensitivity (Stage 7B-NL)
- Residual-dependence artifact adjudication (Stage 7B-RD)

## Frozen claims

- The central 30-min patient morphology is embedded in a reproducible low-dimensional population-common space.
- B8 captures the dominant population-central morphology variation in the discovery cohort.
- Patient points are meaningfully dispersed in B8; they are not collapsed to a common location.
- Short-window within-patient movement is appreciable relative to replicate discrepancy, while remaining smaller than overall between-patient separation.
- Measured constitutional attributes provide little OOF conditional-mean predictability under the frozen linear models and the separate low-df nonlinear sensitivity.
- Residual constitutional dependence persists after artifact controls; the closeout therefore does not claim constitutional independence.
- Coarse chronic disease categories do not materially improve B8 OOF prediction in Stage 7C.

## Explicit non-claims

WF-P does not establish:
- physiological state labels;
- stable trait labels;
- causal physiology;
- disease-specific mechanisms;
- treatment-response meaning;
- constitutional independence;
- a variance/covariance explanation for the residual dCor signal;
- external-database generalizability.

## Analysis stop rule

Do not add:
- new constitutional models;
- variance/covariance regressions;
- disease subdivisions;
- AF-only or valve-subtype post-result searches;
- nonlinear manifold learning;
- disease-specific subspaces.

Any such work is a new study/stage and requires a new scientific question and freeze.
"""
    (out / "WFP_FINAL_SCIENTIFIC_STATUS.md").write_text(status, encoding="utf-8")

    # Figure manifest, updated for the RD result.
    fig = """# WF-P Figure Manifest v0.1

## Main Figure 1 — Study architecture and frozen population coordinates

Purpose: show how 30-min ABP central morphology becomes a patient-balanced population coordinate system.

Panels:
A. 125-Hz ABP -> accepted beats -> 64-point `shape_norm`.
B. 60-s blocks -> odd/even replicate representatives -> 30-min patient central morphology.
C. Replicate-stable population operator -> frozen B8.
D. Downstream audits: WFP0, Stage7B/7B-NL/7B-RD, scale audit, Stage7C.

Required boundary in caption:
- one 30-min representative is central morphology, not trait/state;
- B8 is not relearned in downstream audits.

## Main Figure 2 — A stable low-dimensional population-common morphology space

Primary message: dominant central morphology is well represented by a stable low-dimensional common space.

Candidate panels:
A. positive eigenspectrum / cumulative capture, mark d90=6 and d95=8.
B. held-out reconstruction vs dimension, mark frozen d=8.
C. comparator: learned B8/PCA/Fourier/random subspaces.
D. half-split / odd-even stability summaries.

No physiological naming of axes.

## Main Figure 3 — Patient map scale: between, within, replicate

Primary message:
`D_between >> D_within >> D_replicate`.

Panels:
A. patient cloud in a purely illustrative 2-D projection of frozen B8 scores.
B. distributions of between-patient, within-patient, and odd/even replicate distances.
C. movement relative to nearest-neighbor distance.
D. per-axis within/between variance-ratio profile.

Caption must state:
- 60-s movement is short-window movement, not WF3 longitudinal state trajectory;
- 2-D display is visualization only; inference remains in frozen B8.

## Main Figure 4 — What measured covariates explain, and what remains unresolved

This figure MUST use a two-layer interpretation after Stage7B-RD.

Panel A — Conditional-mean predictability
- conventional factors OOF R2
- age+sex OOF R2
- nonlinear age model delta OOF R2
- age-by-sex total delta OOF R2
- height incremental delta
- nonlinear height incremental delta
- chronic phenotype block delta OOF R2

Primary visual message:
low-df nonlinear flexibility does not rescue constitutional mean prediction.

Panel B — Residual dependence sensitivity
- observed residual dCor vs pipeline-replay null q95, full cohort
- observed residual dCor vs pipeline-replay null q95, height subset
- optionally annotate within-fold control p-values

Primary visual message:
residual constitutional dependence persists even though mean prediction is weak.

Do NOT title Figure 4 as "B8 is independent of constitutional factors".
Preferred conceptual title:
"Measured constitutional attributes weakly predict B8 means but retain residual multivariate dependence."

## Supplementary Figure candidates

S1. Full eigenspectrum and dimension profile.
S2. Learned/PCA/Fourier/random comparator details.
S3. Frozen B8 axis waveform atlas (mean ± fixed score excursion).
S4. Axis reliability / odd-even correlations.
S5. Local morphology / residual outside-B8 localization.
S6. Stage7C phenotype-specific adjusted shift norms and BH-q values.
S7. Full scale-audit diagnostics and nearest-neighbor crossing.
S8. Stage7B-NL model comparison and Stage7B-RD null distributions.

## Plotting rule

All figure-ready data must be generated from `WFP_AUTHORITATIVE_RESULTS_SUMMARY.csv`
or from explicitly hashed patient-level/axis-level derived files already produced by frozen analyses.
No scientific metric may be recomputed differently during plotting.
"""
    (out / "WFP_FIGURE_MANIFEST_v0_1.md").write_text(fig, encoding="utf-8")

    qc = """# WF-P Figure QC Checklist

## Cohort identity
- [ ] Discovery source n=1000, analysable n=978.
- [ ] Height sensitivity n=693.
- [ ] Stage7C exact-admission analysis n=887.
- [ ] Cohort numbers are not mixed across panels.

## Frozen geometry
- [ ] B8 dimension is 8 everywhere.
- [ ] No panel relearns/rotates/reorders B8.
- [ ] No axis is called trait/state.

## Scale
- [ ] Between-patient pairwise RMS matches authoritative summary.
- [ ] Within-patient 60-s RMS matches authoritative summary.
- [ ] Odd/even replicate RMS matches authoritative summary.
- [ ] 60-s movement is not described as long-duration trajectory.

## Constitutional interpretation
- [ ] Stage7B primary linear result remains primary.
- [ ] Stage7B-NL is labeled sensitivity.
- [ ] Spline/interaction models are not described as improving OOF prediction if their delta R2 is negative.
- [ ] Stage7B-RD residual dCor signal is shown if constitutional interpretation is visualized.
- [ ] No claim of independence / no relationship / complete orthogonality is made.
- [ ] Residual dCor is not labeled specifically as nonlinear mean, variance, or covariance mechanism.

## Chronic phenotypes
- [ ] Phenotype block delta OOF R2 is reported with its sign.
- [ ] FDR-significant global phenotypes = none.
- [ ] Nominal arrhythmia/valvular p-values are not promoted to main findings.

## Provenance
- [ ] Figure annotations are generated from authoritative summary / hashed frozen outputs.
- [ ] No patient-level restricted MIMIC data are placed in public figure-data exports.
- [ ] Plot scripts and final figures receive SHA256 hashes before repository release.
"""
    (out / "WFP_FIGURE_QC_CHECKLIST.md").write_text(qc, encoding="utf-8")

    # Source SHA manifest.
    manifest_rows = []
    all_sources = dict(source_paths)
    for k, p in optional_paths.items():
        if p.is_file():
            all_sources[k] = p

    for label, p in sorted(all_sources.items()):
        manifest_rows.append({
            "label": label,
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    with (out / "WFP_CLOSEOUT_SOURCE_MANIFEST_SHA256.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["label","path","bytes","sha256"])
        w.writeheader()
        w.writerows(manifest_rows)

    # Closeout readout.
    readout = "\n".join([
        "WF-P FINAL SCIENTIFIC CLOSEOUT",
        "==============================",
        "Decision: WFP_FINAL_CLOSEOUT_COMPLETE",
        "New scientific effects calculated: NO",
        "Frozen B8 changed: NO",
        f"Authoritative metrics written: {len(rows)}",
        f"Source files hashed: {len(manifest_rows)}",
        "",
        "Constitutional interpretation:",
        "  OOF conditional-mean predictability is weak across frozen linear and",
        "  separate low-df nonlinear/limited-interaction sensitivities.",
        "  Residual multivariate dependence persists after pipeline/fold controls.",
        "  Therefore independence / full orthogonality is NOT claimed.",
        "",
        "Next authorized work:",
        "  Figure-ready data export and plotting from the closed result set.",
        "  No additional WF-P scientific-effect analysis.",
        "",
    ])
    (out / "WFP_FINAL_CLOSEOUT_READOUT.txt").write_text(readout, encoding="utf-8")
    print(readout, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
