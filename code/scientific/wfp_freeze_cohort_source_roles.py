#!/usr/bin/env python3
from pathlib import Path
import argparse, json, hashlib

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pf = Path(args.preflight).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not pf.exists():
        raise SystemExit(f"FAIL: preflight JSON not found: {pf}")

    d = json.loads(pf.read_text())
    decision = str(d.get("decision", ""))
    if "WFP_SOURCE_PREFLIGHT" not in decision:
        raise SystemExit(f"FAIL: unexpected preflight decision: {decision}")

    freeze = {
        "schema_version": 1,
        "work_package": "WF-P",
        "freeze_type": "cohort_source_role",
        "preflight_sha256": sha256_file(pf),
        "waveform_arrays_opened_by_this_script": False,
        "scientific_effects_opened_by_this_script": False,
        "cohort_roles": {
            "development50": "engineering_only",
            "validation1000": "discovery_derivation",
            "future_independent_cohort": "confirmatory_validation"
        },
        "legacy_counts": {
            "development50": 50,
            "validation1000": 1000,
            "overlap": 0
        },
        "authorization": {
            "development50_engineering_smoke": True,
            "validation1000_discovery_after_smoke": True,
            "confirmatory_validation": False
        }
    }

    out_json = out / "WFP_COHORT_SOURCE_FREEZE.json"
    out_json.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    digest = sha256_file(out_json)

    txt = (
        "WF-P COHORT SOURCE FREEZE\n"
        "=========================\n"
        "Decision: WFP_SOURCE_ROLE_FREEZE_PASS\n"
        "Development50 role: ENGINEERING_ONLY\n"
        "Validation1000 role: DISCOVERY_DERIVATION\n"
        "Future independent cohort role: CONFIRMATORY_VALIDATION\n"
        "Waveform arrays opened: NO\n"
        "Scientific effects opened: NO\n"
        "Validation1000 discovery after implementation smoke: AUTHORIZED\n"
        "Independent confirmatory validation: NOT AUTHORIZED\n"
        f"Freeze SHA256: {digest}\n"
    )
    (out / "WFP_COHORT_SOURCE_FREEZE.txt").write_text(txt)
    print(txt, end="")

if __name__ == "__main__":
    main()
