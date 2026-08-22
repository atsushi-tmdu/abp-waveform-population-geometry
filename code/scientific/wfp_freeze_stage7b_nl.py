#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Stage 7B-NL before nonlinear B8 effect access."""

from pathlib import Path
import argparse
import hashlib
import json

EXPECTED_ANALYSIS_SHA256 = "5b88a4681587d98796a1a6fae3c877a0414153c770e1d3f9ccfba7a9e8869a75"

PHENOTYPE_FORBIDDEN = True

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file", required=True)
    ap.add_argument("--temporal-linkage", required=True)
    ap.add_argument("--height-preflight", required=True)
    ap.add_argument("--stage7b-spec", required=True)
    ap.add_argument("--stage7b-results-json", required=True)
    ap.add_argument("--stage7b-readout", required=True)
    ap.add_argument("--preflight-json", required=True)
    ap.add_argument("--analysis-script", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    score = Path(args.score_file).expanduser().resolve()
    temporal = Path(args.temporal_linkage).expanduser().resolve()
    height = Path(args.height_preflight).expanduser().resolve()
    stage7b_spec = Path(args.stage7b_spec).expanduser().resolve()
    stage7b_results = Path(args.stage7b_results_json).expanduser().resolve()
    stage7b_readout = Path(args.stage7b_readout).expanduser().resolve()
    preflight = Path(args.preflight_json).expanduser().resolve()
    analysis = Path(args.analysis_script).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for path in [
        score, temporal, height, stage7b_spec, stage7b_results,
        stage7b_readout, preflight, analysis
    ]:
        if not path.is_file():
            raise SystemExit(f"FAIL missing required file: {path}")

    if sha256_file(analysis) != EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("FAIL: Stage7B-NL analysis-script hash mismatch")

    pre = json.loads(preflight.read_text(encoding="utf-8"))
    if pre.get("decision") != "WFP_STAGE7B_NL_PREFLIGHT_PASS":
        raise SystemExit("FAIL: Stage7B-NL preflight did not PASS")
    if int(pre.get("full_cohort_n", -1)) != 978:
        raise SystemExit("FAIL: preflight full cohort n")
    if int(pre.get("height_complete_case_n", -1)) != 693:
        raise SystemExit("FAIL: preflight height subset n")
    if pre.get("height_column_detected") != "height_median_cm":
        raise SystemExit("FAIL: frozen height column is not height_median_cm")
    if pre.get("morphology_score_values_read") is not False:
        raise SystemExit("FAIL: preflight unexpectedly opened morphology effects")
    if pre.get("nonlinear_effects_calculated") is not False:
        raise SystemExit("FAIL: preflight unexpectedly calculated nonlinear effects")
    if pre.get("existing_stage7b_modified") is not False:
        raise SystemExit("FAIL: preflight claims Stage7B modification")

    stage7b = json.loads(stage7b_results.read_text(encoding="utf-8"))
    if stage7b.get("decision") != "WFP_STAGE7B_CONSTITUTIONAL_Q4Q5_COMPLETE":
        raise SystemExit("FAIL: Stage7B is not complete")
    if int(stage7b.get("source_n", -1)) != 978:
        raise SystemExit("FAIL: Stage7B source n")
    if int(stage7b.get("height_complete_case_n", -1)) != 693:
        raise SystemExit("FAIL: Stage7B height n")

    reference = {
        "full_linear_age_sex":
            float(stage7b["primary"]["aggregate_oof_r2"]),
        "height_subset_linear_age_sex":
            float(stage7b["height_secondary"]["age_sex_oof_r2_same_subset"]),
        "height_subset_linear_age_sex_height":
            float(stage7b["height_secondary"]["age_sex_height_oof_r2"]),
        "quadratic_age_sensitivity":
            float(stage7b["nonlinear_age_sensitivity"]["aggregate_oof_r2"]),
    }

    spec = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "7B-NL",
        "status": "FROZEN_BEFORE_STAGE7B_NL_EFFECT_ACCESS",
        "scientific_role":
            "post_stage7b_nonlinear_constitutional_sensitivity",
        "existing_stage7b_modified": False,
        "full_cohort_n": 978,
        "height_complete_case_n": 693,
        "height_column": "height_median_cm",
        "cv": {
            "folds": 5,
            "rule": "sha256('20260820:<patient_id>') mod 5",
            "primary_metric": "aggregate multivariate OOF R2",
        },
        "spline": {
            "type": "restricted cubic spline / natural cubic spline",
            "knots": 4,
            "training_fold_quantile_probabilities":
                [0.05, 0.35, 0.65, 0.95],
            "total_df_per_smooth": 3,
            "test_fold_used_to_select_knots": False,
        },
        "full_models": {
            "M0": "linear age + sex, Stage7B replay",
            "M1": "RCS4(age) + sex",
            "M2": "RCS4(age) * sex",
        },
        "height_models": {
            "H0": "linear age + sex, same n=693 subset",
            "H0h": "linear age + sex + height, Stage7B replay",
            "H1": "RCS4(age) + sex",
            "H1h": "RCS4(age) + sex + linear height",
            "H2": "RCS4(age) + sex + RCS4(height)",
            "H3": "H2 + sex * linear(height), one interaction df",
        },
        "primary_comparisons": [
            "M1-M0",
            "M2-M1 and M2-M0",
            "H1h-H1",
            "H2-H1h",
            "H3-H2",
        ],
        "effect_size_interpretation_bands": {
            "delta_oof_r2_lt_0.01":
                "no material predictive improvement",
            "delta_oof_r2_0.01_to_lt_0.03":
                "modest nonlinear information",
            "delta_oof_r2_ge_0.03":
                "material nonlinear information requiring reinterpretation",
        },
        "model_free_sensitivity": {
            "measure": "biased sample distance correlation",
            "full_target":
                "X=(age,sex) vs OOF residual B8 after M0",
            "height_target":
                "X=(age,sex,height) vs OOF residual B8 after H0h",
            "standardize_columns": True,
            "patient_permutations": 999,
            "plus_one_p": True,
            "seed": 20260824,
            "multiplicity":
                "Holm adjustment across the two residual-dCor tests",
            "interpretation":
                "sensitivity only; may reflect nonlinear mean, variance, "
                "or higher-order dependence",
        },
        "stage7b_reference_oof_r2": reference,
        "axis_specific_oof_r2": "DESCRIPTIVE_ONLY",
        "forbidden": [
            "modifying Stage7B",
            "changing or relearning B8",
            "post-effect knot or df selection",
            "random forest",
            "neural network",
            "gradient boosting",
            "kernel-ridge prediction ceiling",
            "hyperparameter search",
            "additional interactions",
            "disease/treatment/outcome access",
            "automatic variance/distributional modeling",
            "Ztrait/Zstate labeling",
        ],
        "analysis_script_sha256": sha256_file(analysis),
        "score_file_sha256": sha256_file(score),
        "temporal_linkage_sha256": sha256_file(temporal),
        "height_preflight_sha256": sha256_file(height),
        "stage7b_spec_sha256": sha256_file(stage7b_spec),
        "stage7b_results_sha256": sha256_file(stage7b_results),
        "stage7b_readout_sha256": sha256_file(stage7b_readout),
        "preflight_json_sha256": sha256_file(preflight),
    }

    spec_path = out / "WFP_STAGE7B_NL_FROZEN_SPEC.json"
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    txt = (
        "WF-P STAGE 7B-NL SPEC FREEZE\n"
        "============================\n"
        "Decision: WFP_STAGE7B_NL_SPEC_FREEZE_PASS\n"
        "Existing Stage7B modified: NO\n"
        "Full cohort n: 978\n"
        "Height complete-case n: 693\n"
        "Height definition: height_median_cm\n"
        "Age smooth: RCS4 with training-fold 5/35/65/95% knots\n"
        "Age x sex: one prespecified spline interaction family\n"
        "Height smooth: RCS4 in complete cases\n"
        "Height x sex: one secondary 1-df linear interaction\n"
        "Primary quantity: patient-level 5-fold OOF R2 and delta R2\n"
        "Model-free sensitivity: residual dCor, 999 permutations\n"
        "Model fishing / hyperparameter search: NO\n"
        "Variance/distributional extension automatically authorized: NO\n"
        "Ztrait/Zstate labels authorized: NO\n"
        f"Analysis SHA256: {sha256_file(analysis)}\n"
        f"Frozen spec SHA256: {sha256_file(spec_path)}\n"
    )
    (out / "WFP_STAGE7B_NL_FROZEN_SPEC.txt").write_text(
        txt, encoding="utf-8"
    )
    print(txt, end="")

if __name__ == "__main__":
    main()
