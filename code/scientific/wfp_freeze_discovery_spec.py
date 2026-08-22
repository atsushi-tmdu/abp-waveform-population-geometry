#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze WF-P Validation1000 discovery specification before effect access."""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_RUN1 = "811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01"
EXPECTED_ANALYSIS = "a928ae3c3a81ebf9ba662cbde819d4384c7c8b13d96565ce29f32e4315d1c4ca"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--source-freeze", default="~/Documents/abp_information_study/results/wfp_cohort_source_freeze/WFP_COHORT_SOURCE_FREEZE.json")
    ap.add_argument("--smoke", default="~/Documents/abp_information_study/results/wfp_morphology_smoke/WFP_MORPHOLOGY_SMOKE.json")
    ap.add_argument("--block-yield", default="~/Documents/abp_information_study/results/wfp_block_yield/WFP_BLOCK_YIELD.json")
    ap.add_argument("--analysis-script", default="~/Documents/abp_information_study/code/wfp_validation1000_discovery.py")
    ap.add_argument("--out", default="~/Documents/abp_information_study/freeze")
    args=ap.parse_args()

    project=Path(args.project_root).expanduser().resolve()
    sf=Path(args.source_freeze).expanduser().resolve()
    sm=Path(args.smoke).expanduser().resolve()
    by=Path(args.block_yield).expanduser().resolve()
    ana=Path(args.analysis_script).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for p in (sf,sm,by,ana):
        if not p.exists():
            raise SystemExit(f"FAIL missing: {p}")

    source=json.loads(sf.read_text())
    smoke=json.loads(sm.read_text())
    block=json.loads(by.read_text())

    if source.get("cohort_roles",{}).get("validation1000") != "discovery_derivation":
        raise SystemExit("FAIL: Validation1000 source role is not discovery_derivation")
    if smoke.get("decision") != "WFP_MORPHOLOGY_SMOKE_PASS":
        raise SystemExit("FAIL: morphology smoke not PASS")
    if block.get("decision") != "WFP_BLOCK_YIELD_PASS_FREEZE_THRESHOLD":
        raise SystemExit("FAIL: block-yield audit not PASS")
    if int(block.get("selected_min_beats_per_block",-1)) != 32:
        raise SystemExit("FAIL: selected min beats/block is not 32")

    run1=project/"code"/"wp2_run1_development50.py"
    if not run1.exists() or sha256_file(run1) != EXPECTED_RUN1:
        raise SystemExit("FAIL: authoritative Run-1 hash mismatch")
    if sha256_file(ana) != EXPECTED_ANALYSIS:
        raise SystemExit("FAIL: discovery analysis script hash mismatch")

    spec={
      "schema_version":1,
      "work_package":"WF-P",
      "status":"FROZEN_BEFORE_VALIDATION1000_WFP_DISCOVERY_ACCESS",
      "scientific_role":"discovery_derivation_only",
      "source_cohort":"Validation1000",
      "expected_source_n":1000,
      "representation":{
        "source_fs_hz":125,
        "phase_grid":64,
        "representation":"shape_norm",
        "beat_pipeline":"authoritative WF1/WF2 native-125-Hz peak detection, midpoint boundaries, locked 30-min QC"
      },
      "block_definition":{
        "block_sec":60,
        "min_accepted_beats_per_block":32,
        "min_total_blocks":6,
        "min_odd_blocks":3,
        "min_even_blocks":3,
        "block_weighting":"equal clock-time blocks within patient",
        "patient_weighting":"equal patients"
      },
      "patient_representative":{
        "block_shape":"normalize(mean accepted shape_norm beats within eligible 60-s block))",
        "coherence":"phase SD of unrenormalized block mean shape; retained separately",
        "all_rep":"normalize(mean eligible block central shapes)",
        "odd_rep":"normalize(mean odd-indexed eligible block central shapes)",
        "even_rep":"normalize(mean even-indexed eligible block central shapes)",
        "terminology":"30-min central morphology; not trait/state"
      },
      "primary_operator":"0.5*(Cov(odd,even)+Cov(even,odd))",
      "spectrum":{
        "positive_eigenvalues_define_reproducible_spectrum":True,
        "negative_spectral_mass_reported":True,
        "effective_rank_positive":True,
        "d90_positive":True,
        "d95_positive":True
      },
      "cross_validation":{
        "folds":5,
        "fold_seed":20260820,
        "max_dimension":24,
        "dimension_rule":"smallest d with CV R2_all >= max(0.90,0.95*R2_at_max_available_d) and CV R2_odd>=0.85 and CV R2_even>=0.85"
      },
      "comparators":[
        "ordinary patient-mean PCA with identical patient CV folds",
        "fixed low-harmonic Fourier subspaces at even dimensions",
        "200 deterministic matched random mean-zero subspaces at selected dimension"
      ],
      "stability":{
        "patient_halfsplit_repeats":100,
        "seed":20260822,
        "metric":"projector overlap at selected dimension"
      },
      "within_window_alignment":{
        "within_operator":"equal-patient mean covariance of eligible 60-s block central shapes about patient block mean",
        "metrics":[
          "within-window variance captured by between basis",
          "positive between variance captured by within basis",
          "projector overlap",
          "principal angles"
        ],
        "interpretation":"short-window precursor only; not WF3 long-duration covariance"
      },
      "deferred":[
        "age/sex clinical linkage and residualization",
        "long-duration WF3 trajectory",
        "Zstate or Ztrait identification",
        "independent confirmatory validation"
      ],
      "forbidden_after_access":[
        "changing primary representation",
        "changing p=64",
        "changing 60-s block length",
        "changing min beats/block=32",
        "changing patient eligibility 6/3/3",
        "changing odd/even replicate definition",
        "changing primary replicate-corrected operator",
        "changing CV folds/seed or dimension rule",
        "changing comparator family or random seeds",
        "using clinical labels to redefine morphology axes"
      ],
      "source_freeze_sha256":sha256_file(sf),
      "morphology_smoke_sha256":sha256_file(sm),
      "block_yield_sha256":sha256_file(by),
      "authoritative_run1_sha256":sha256_file(run1),
      "analysis_script_sha256":sha256_file(ana)
    }

    spec_path=out/"WFP_DISCOVERY_FROZEN_SPEC.json"
    spec_path.write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    digest=sha256_file(spec_path)
    txt=(
      "WF-P DISCOVERY SPECIFICATION FREEZE\n"
      "==================================\n"
      "Decision: WFP_DISCOVERY_SPEC_FREEZE_PASS\n"
      "Source cohort: Validation1000\n"
      "Scientific role: DISCOVERY_DERIVATION_ONLY\n"
      "Min accepted beats/block: 32\n"
      "Patient eligibility: total>=6, odd>=3, even>=3\n"
      "Primary operator: symmetric odd/even replicate cross-covariance\n"
      "Clinical labels authorized: NO\n"
      "Independent confirmatory validation: NO\n"
      f"Analysis script SHA256: {sha256_file(ana)}\n"
      f"Frozen spec SHA256: {digest}\n"
    )
    (out/"WFP_DISCOVERY_FROZEN_SPEC.txt").write_text(txt)
    print(txt,end="")

if __name__=="__main__":
    main()
