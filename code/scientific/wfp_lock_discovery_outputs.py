#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P Stage 4A — lock the completed Validation1000 discovery outputs.

This script does not open waveform arrays and does not calculate new scientific
effects. It verifies the completed discovery readout and writes a hash manifest.
"""

from pathlib import Path
import argparse, hashlib, json

EXPECTED_SPEC_SHA256 = "0aa1944f715d46d78e4c11b72d9086a34605f5770071c823fd2ebf9e6eb787b7"
EXPECTED_ANALYSIS_SHA256 = "a928ae3c3a81ebf9ba662cbde819d4384c7c8b13d96565ce29f32e4315d1c4ca"

REQUIRED_OUTPUTS = [
    "WFP_DISCOVERY_RESULTS.json",
    "WFP_DISCOVERY_READOUT.txt",
    "WFP_DISCOVERY_COMMON_COORDINATES.npz",
    "wfp_cv_reconstruction_curve.csv",
    "wfp_fourier_comparator_curve.csv",
    "wfp_axis_reliability.csv",
    "wfp_population_eigenspectra.csv",
    "wfp_patient_scores_DISCOVERY_PRIVATE.csv",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--analysis-script", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = Path(args.results).expanduser().resolve()
    spec = Path(args.spec).expanduser().resolve()
    analysis = Path(args.analysis_script).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if sha256_file(spec) != EXPECTED_SPEC_SHA256:
        raise SystemExit(
            "FAIL: frozen discovery spec hash mismatch\n"
            f"observed={sha256_file(spec)}\nexpected={EXPECTED_SPEC_SHA256}"
        )
    if sha256_file(analysis) != EXPECTED_ANALYSIS_SHA256:
        raise SystemExit(
            "FAIL: discovery analysis script hash mismatch\n"
            f"observed={sha256_file(analysis)}\nexpected={EXPECTED_ANALYSIS_SHA256}"
        )

    missing = [name for name in REQUIRED_OUTPUTS if not (results / name).is_file()]
    if missing:
        raise SystemExit("FAIL: missing discovery outputs: " + ", ".join(missing))

    result_path = results / "WFP_DISCOVERY_RESULTS.json"
    r = json.loads(result_path.read_text(encoding="utf-8"))

    checks = {
        "decision_common_basis_identified":
            r.get("decision") == "WFP_DISCOVERY_COMMON_BASIS_IDENTIFIED",
        "scientific_role_discovery":
            r.get("scientific_role") == "discovery_derivation_only",
        "source_n_1000": int(r.get("source_n", -1)) == 1000,
        "analysable_n_978": int(r.get("analysable_n", -1)) == 978,
        "frozen_exclusions_n_22": int(r.get("frozen_rule_exclusions_n", -1)) == 22,
        "selected_dimension_8":
            int(r.get("selected_basis", {}).get("dimension", -1)) == 8,
        "clinical_labels_false":
            r.get("clinical_labels_accessed") is False,
        "age_sex_false":
            r.get("age_sex_analysis_performed") is False,
        "confirmatory_false":
            r.get("independent_confirmatory_validation") is False,
    }
    if not all(checks.values()):
        bad = [k for k,v in checks.items() if not v]
        raise SystemExit("FAIL: discovery result identity mismatch: " + ", ".join(bad))

    manifest = []
    for name in REQUIRED_OUTPUTS:
        p = results / name
        manifest.append({
            "name": name,
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        })

    lock = {
        "schema_version": 1,
        "work_package": "WF-P",
        "stage": "4A",
        "decision": "WFP_DISCOVERY_OUTPUT_LOCK_PASS",
        "scientific_role": "post_discovery_integrity_lock",
        "waveform_arrays_opened": False,
        "new_scientific_effects_calculated": False,
        "source_n": 1000,
        "analysable_n": 978,
        "selected_common_basis_dimension": 8,
        "frozen_discovery_spec_sha256": sha256_file(spec),
        "discovery_analysis_script_sha256": sha256_file(analysis),
        "checks": checks,
        "output_manifest": manifest,
        "boundary": [
            "This lock preserves the completed Validation1000 discovery state.",
            "It does not convert discovery evidence into confirmatory validation.",
            "Clinical labels remain unopened for WF-P."
        ],
    }

    out_json = out / "WFP_DISCOVERY_OUTPUT_LOCK.json"
    out_json.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock_sha = sha256_file(out_json)

    txt = [
        "WF-P DISCOVERY OUTPUT LOCK",
        "==========================",
        "Decision: WFP_DISCOVERY_OUTPUT_LOCK_PASS",
        "Waveform arrays opened: NO",
        "New scientific effects calculated: NO",
        "Source n: 1000",
        "Analysable n: 978",
        "Selected common-basis dimension: 8",
        f"Frozen discovery spec SHA256: {sha256_file(spec)}",
        f"Discovery analysis script SHA256: {sha256_file(analysis)}",
        f"Output lock SHA256: {lock_sha}",
        "",
        "Boundary:",
        "  Discovery status remains DISCOVERY / DERIVATION ONLY.",
        "  Clinical labels remain unopened.",
    ]
    (out / "WFP_DISCOVERY_OUTPUT_LOCK.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
