#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final public-safety/reproducibility audit of the WF-P GitHub staging tree.

Read-only with respect to the staging tree. Audit outputs are written outside
the repository so the audited tree is not mutated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import py_compile
import re
import subprocess
import tempfile
from pathlib import Path

RAW_SUFFIXES = {".npz", ".npy", ".dat", ".hea", ".wfdb", ".gz", ".pkl", ".pickle"}
ID_COLUMNS = {
    "patient_id", "subject_id", "hadm_id", "icustay_id",
    "record_id", "record_path", "case_id",
}
FAIL_PATH_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
]
CONCRETE_PID = re.compile(r"\bp\d{5,}\b", re.I)
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def self_test():
    print("WF-P public-staging audit self-test: PASS")
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--staging-root",default="~/Documents/abp-waveform-population-geometry")
    ap.add_argument("--out",default="~/Documents/wfp_public_staging_audit")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        return self_test()

    root=Path(a.staging_root).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"FAIL staging root missing: {root}")

    expected=[
        "README.md","CITATION.cff","CHANGELOG.md","LICENSE_CODE.txt",
        "DATA_LICENSE.md","FIGURES_AND_DOCUMENTATION_LICENSE.md",
        "PUBLIC_MANIFEST.csv","PUBLIC_MANIFEST.json","STAGING_PROVENANCE.json",
        "figures/Release_Figure_1_population_geometry.pdf",
        "figures/Release_Figure_2_between_within_scale.pdf",
        "interface/frozen_B8_basis_64x8.csv",
        "interface/population_center_64.csv",
        "interface/projection_spec.json",
        "tools/reproduce_release_assets.sh",
    ]
    missing=[x for x in expected if not (root/x).is_file()]
    if missing:
        raise SystemExit("FAIL missing expected files: "+", ".join(missing))

    failures=[]
    warnings=[]
    raw_files=[]
    large_files=[]
    path_hits=[]
    pid_hits=[]
    secret_hits=[]
    id_column_hits=[]
    compile_failures=[]

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel=str(p.relative_to(root))
        low=rel.lower()

        if p.suffix.lower() in RAW_SUFFIXES:
            raw_files.append(rel)
        if p.stat().st_size > 20*1024*1024:
            large_files.append((rel,p.stat().st_size))

        if p.suffix.lower() in {".md",".txt",".json",".csv",".cff",".py",".sh",".yaml",".yml",".toml"}:
            txt=p.read_text(encoding="utf-8",errors="replace")
            for pat in FAIL_PATH_PATTERNS:
                if pat.search(txt):
                    path_hits.append(rel)
                    break
            for pat in SECRET_PATTERNS:
                if pat.search(txt):
                    secret_hits.append(rel)
                    break
            # Concrete pNNNNN-like identifiers are forbidden outside source code.
            if p.suffix.lower() != ".py" and CONCRETE_PID.search(txt):
                pid_hits.append(rel)

        if p.suffix.lower()==".csv":
            try:
                with p.open(newline="",encoding="utf-8",errors="replace") as f:
                    r=csv.reader(f)
                    hdr=next(r,[])
                bad=[c for c in hdr if c.strip().lower() in ID_COLUMNS]
                if bad:
                    id_column_hits.append((rel,bad))
            except Exception as e:
                failures.append(f"CSV header read failed {rel}: {e}")

        if p.suffix.lower()==".py":
            try:
                # Compile outside the staging tree so this read-only audit
                # never creates __pycache__/pyc files in the repository.
                with tempfile.TemporaryDirectory(prefix="wfp_compile_") as ctd:
                    cfile = Path(ctd) / (
                        hashlib.sha256(rel.encode("utf-8")).hexdigest() + ".pyc"
                    )
                    py_compile.compile(
                        str(p),
                        cfile=str(cfile),
                        doraise=True,
                    )
            except Exception as e:
                compile_failures.append((rel,str(e)))

    if raw_files: failures.append("raw/restricted file suffixes present: "+", ".join(raw_files))
    if large_files: failures.append("files >20MB present: "+", ".join(x[0] for x in large_files))
    if path_hits: failures.append("absolute local path leakage: "+", ".join(sorted(set(path_hits))))
    if pid_hits: failures.append("concrete pNNNNN-like identifiers: "+", ".join(sorted(set(pid_hits))))
    if secret_hits: failures.append("secret-like strings: "+", ".join(sorted(set(secret_hits))))
    if id_column_hits: failures.append("identifier columns in CSV: "+repr(id_column_hits))
    if compile_failures: failures.append("Python syntax failures: "+repr(compile_failures))

    # Manifest integrity.
    mf=root/"PUBLIC_MANIFEST.csv"
    rows=list(csv.DictReader(mf.open(encoding="utf-8")))
    manifest_errors=[]
    for r in rows:
        p=root/r["file"]
        if not p.is_file():
            manifest_errors.append(f"missing:{r['file']}")
            continue
        if int(r["bytes"])!=p.stat().st_size:
            manifest_errors.append(f"bytes:{r['file']}")
        if r["sha256"]!=sha256_file(p):
            manifest_errors.append(f"sha:{r['file']}")
    if manifest_errors:
        failures.append("PUBLIC_MANIFEST integrity: "+", ".join(manifest_errors))

    # No extra files outside manifest except manifests themselves.
    listed={r["file"] for r in rows}
    actual={
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in {"PUBLIC_MANIFEST.csv","PUBLIC_MANIFEST.json"}
    }
    extras=sorted(actual-listed)
    if extras:
        failures.append("unmanifested files: "+", ".join(extras))

    # Scientific boundary text.
    readme=(root/"README.md").read_text(encoding="utf-8",errors="replace")
    # Normalize Markdown line wrapping before checking required wording.
    readme_norm = " ".join(readme.split())
    required_phrases=[
        "Independent confirmatory WF-P validation: **not yet performed**",
        "without relearning, rotating, or re-signing B8",
        "central morphology",
        "does **not** establish",
    ]
    for phrase in required_phrases:
        if phrase not in readme_norm:
            failures.append(f"README boundary phrase missing: {phrase}")

    # Interface metadata.
    meta=json.loads((root/"interface"/"WFP_B8_INTERFACE_v1.0.0.json").read_text())
    if meta.get("B8_changed") is not False:
        failures.append("interface metadata B8_changed is not false")
    if meta.get("scientific_effects_calculated_by_serialization") is not False:
        failures.append("interface metadata scientific effects flag is not false")

    # Reproduction smoke in a temporary directory.
    repro_status="NOT_RUN"
    try:
        with tempfile.TemporaryDirectory(prefix="wfp_repro_") as td:
            cmd=[
                "bash",str(root/"tools"/"reproduce_release_assets.sh"),td
            ]
            cp=subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=180)
            if cp.returncode!=0:
                failures.append("release reproduction smoke failed: "+cp.stderr[-1200:])
                repro_status="FAIL"
            else:
                needed=[
                    Path(td)/"Release_Figure_1_population_geometry.pdf",
                    Path(td)/"Release_Figure_2_between_within_scale.pdf",
                    Path(td)/"tables"/"Table_1_population_geometry.csv",
                ]
                if not all(x.is_file() for x in needed):
                    failures.append("release reproduction smoke missing expected outputs")
                    repro_status="FAIL"
                else:
                    repro_status="PASS"
    except Exception as e:
        failures.append(f"release reproduction smoke exception: {e}")
        repro_status="FAIL"

    decision="WFP_PUBLIC_STAGING_AUDIT_PASS" if not failures else "WFP_PUBLIC_STAGING_AUDIT_FAIL"

    report={
        "decision":decision,
        "staging_root":str(root),
        "raw_files_present":raw_files,
        "large_files":large_files,
        "absolute_path_hits":sorted(set(path_hits)),
        "patient_identifier_hits":sorted(set(pid_hits)),
        "secret_hits":sorted(set(secret_hits)),
        "identifier_column_hits":id_column_hits,
        "python_compile_failures":compile_failures,
        "manifest_errors":manifest_errors,
        "reproduction_smoke":repro_status,
        "warnings":warnings,
        "failures":failures,
        "git_initialized":(root/".git").exists(),
    }
    (out/"WFP_PUBLIC_STAGING_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    lines=[
        "WF-P PUBLIC GITHUB STAGING AUDIT",
        "================================",
        f"Decision: {decision}",
        f"Raw/restricted files present: {'YES' if raw_files else 'NO'}",
        f"Files >20MB present: {'YES' if large_files else 'NO'}",
        f"Absolute local path leakage: {'YES' if path_hits else 'NO'}",
        f"Concrete patient-like identifiers: {'YES' if pid_hits else 'NO'}",
        f"Identifier columns in CSV: {'YES' if id_column_hits else 'NO'}",
        f"Secret-like strings: {'YES' if secret_hits else 'NO'}",
        f"Python compile failures: {'YES' if compile_failures else 'NO'}",
        f"PUBLIC_MANIFEST integrity: {'PASS' if not manifest_errors else 'FAIL'}",
        f"Release reproduction smoke: {repro_status}",
        f"Git initialized: {'YES' if (root/'.git').exists() else 'NO'}",
        "",
    ]
    if failures:
        lines.append("Failures:")
        lines.extend("  - "+x for x in failures)
    else:
        lines += [
            "Public staging tree is suitable for first Git versioning.",
            "Next: inspect `git status` after `git init`, commit, create GitHub remote,",
            "then create an exact v1.0.0 GitHub release before Zenodo archiving.",
        ]
    txt="\n".join(lines)+"\n"
    (out/"WFP_PUBLIC_STAGING_AUDIT.txt").write_text(txt,encoding="utf-8")
    print(txt)
    return 0 if not failures else 2

if __name__=="__main__":
    raise SystemExit(main())
