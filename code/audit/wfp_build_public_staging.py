#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a separate public GitHub staging tree for WF-P.

This script DOES NOT:
- modify the private working directory;
- initialize Git;
- push to GitHub;
- upload to Zenodo;
- open raw waveform arrays;
- calculate scientific effects.

It copies only release-candidate code/freeze/results/interface/figures/tables
into a new staging directory and writes release metadata using relative paths.
Run the companion public-staging audit before `git init`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

VERSION = "1.0.0"

EXCLUDE_TOKENS = (
    "private",
    "patient_scores",
    "patient_score",
    "checkpoint",
    "cache",
    "__pycache__",
    ".bak",
    "backup",
    "crdownload",
)

FREEZE_EXCLUDE_TOKENS = (
    "case_filenames",
    "filename_manifest",
    "source_manifest",
    "private",
)

ALLOWED_FREEZE_SUFFIXES = {".json", ".md", ".txt"}
ALLOWED_CODE_SUFFIXES = {".py"}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"FAIL missing {label}: {path}")
    return path

def ensure_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"FAIL missing {label}: {path}")
    return path

def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def classify_code(name: str) -> str:
    low = name.lower()
    if any(x in low for x in ("render", "figure", "publication")):
        return "publication"
    if any(x in low for x in (
        "preflight", "audit", "closeout", "public_release",
        "serialize", "interface", "staging", "build_public"
    )):
        return "audit"
    return "scientific"

def safe_code_candidates(code_dir: Path):
    out = []
    for p in sorted(code_dir.glob("wfp_*.py")):
        low = p.name.lower()
        if any(tok in low for tok in EXCLUDE_TOKENS):
            continue
        out.append(p)
    # Required upstream dependency inherited from WF2.
    dep = code_dir / "wp2_run1_development50.py"
    if dep.is_file():
        out.append(dep)
    return out

def safe_freeze_candidates(freeze_dir: Path):
    out = []
    for p in sorted(freeze_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(freeze_dir))
        low = rel.lower()
        if "wfp" not in low:
            continue
        if p.suffix.lower() not in ALLOWED_FREEZE_SUFFIXES:
            continue
        if any(tok in low for tok in FREEZE_EXCLUDE_TOKENS):
            continue
        # The JSON serialization freeze contains 1000 case filenames and is private.
        if p.name == "WFP_INTERFACE_SERIALIZATION_FROZEN_SPEC.json":
            continue
        out.append(p)
    return out

def observed_environment():
    pkgs = ["numpy", "pandas", "scipy", "matplotlib", "pillow"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "NOT_INSTALLED"
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": versions,
    }

def manifest_files(root: Path):
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if rel in {"PUBLIC_MANIFEST.csv", "PUBLIC_MANIFEST.json"}:
            continue
        rows.append({
            "file": rel,
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        })
    return rows

def self_test():
    assert classify_code("wfp_validation1000_discovery.py") == "scientific"
    assert classify_code("wfp_render_release_publication_assets.py") == "publication"
    assert classify_code("wfp_public_release_preflight.py") == "audit"
    print("WF-P public-staging builder self-test: PASS")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--staging-root", default="~/Documents/abp-waveform-population-geometry")
    ap.add_argument("--publication-source-data",
                    default="~/Documents/abp_information_study/release_build/WFP_RELEASE_PUBLICATION_ASSETS/data")
    ap.add_argument("--author-name", default="Atsushi Senda")
    ap.add_argument("--orcid", default="https://orcid.org/0000-0002-0128-6800")
    ap.add_argument("--github-user", default="atsushi-tmdu")
    ap.add_argument("--repo-name", default="abp-waveform-population-geometry")
    ap.add_argument("--wf1-doi", default="10.5281/zenodo.21940412")
    ap.add_argument("--wf2-doi", default="10.5281/zenodo.22020208")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    project = Path(a.project_root).expanduser().resolve()
    stage = Path(a.staging_root).expanduser().resolve()
    pubsrc = Path(a.publication_source_data).expanduser().resolve()

    ensure_dir(project, "project root")
    ensure_dir(pubsrc, "publication source data")

    if stage.exists():
        raise SystemExit(
            f"FAIL staging directory already exists: {stage}\n"
            "Move/delete it manually only if you intend to rebuild. No files changed."
        )

    # Required closed/public-safe objects.
    interface = ensure_dir(project / "results" / "wfp_release_interface_v1", "release interface")
    interface_audit = ensure_file(
        project / "results" / "wfp_release_interface_v1_public_safety"
        / "WFP_RELEASE_INTERFACE_PUBLIC_SAFETY_AUDIT.txt",
        "interface public-safety audit",
    )
    ia = interface_audit.read_text(encoding="utf-8", errors="replace")
    if "Decision: WFP_RELEASE_INTERFACE_PUBLIC_SAFETY_PASS" not in ia:
        raise SystemExit("FAIL interface public-safety audit is not PASS")

    pubassets = ensure_dir(
        project / "results" / "wfp_release_publication_assets",
        "release publication assets",
    )
    fig1 = ensure_file(pubassets / "Release_Figure_1_population_geometry.pdf", "Release Figure 1")
    fig2 = ensure_file(pubassets / "Release_Figure_2_between_within_scale.pdf", "Release Figure 2")
    pubreadout = ensure_file(
        pubassets / "WFP_RELEASE_PUBLICATION_ASSETS_READOUT.txt",
        "publication assets readout",
    )
    if "Decision: WFP_RELEASE_PUBLICATION_ASSETS_COMPLETE" not in pubreadout.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise SystemExit("FAIL publication assets readout is not COMPLETE")

    closeout = ensure_dir(project / "results" / "wfp_final_closeout", "WF-P final closeout")
    closeout_readout = ensure_file(closeout / "WFP_FINAL_CLOSEOUT_READOUT.txt", "final closeout readout")
    if "Decision: WFP_FINAL_CLOSEOUT_COMPLETE" not in closeout_readout.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise SystemExit("FAIL WF-P final closeout not complete")

    # Create tree.
    dirs = [
        "code/scientific",
        "code/audit",
        "code/publication",
        "freeze",
        "results/aggregate",
        "results/figure_ready",
        "interface",
        "figures",
        "tables",
        "docs/reproducibility",
        "environment",
    ]
    for d in dirs:
        (stage / d).mkdir(parents=True, exist_ok=True)

    # Code: all current WF-P scripts + one inherited WF2 Run1 dependency.
    code_dir = project / "code"
    copied_code = []
    for src in safe_code_candidates(code_dir):
        kind = classify_code(src.name)
        dst = stage / "code" / kind / src.name
        copy_file(src, dst)
        copied_code.append(str(dst.relative_to(stage)))

    # Freeze material: WFP-related, excluding private filename/source manifests.
    copied_freeze = []
    for src in safe_freeze_candidates(project / "freeze"):
        rel = src.relative_to(project / "freeze")
        dst = stage / "freeze" / rel
        copy_file(src, dst)
        copied_freeze.append(str(dst.relative_to(stage)))

    # Interface: already public-safety audited.
    copied_interface = []
    for src in sorted(interface.iterdir()):
        if not src.is_file():
            continue
        low = src.name.lower()
        if any(tok in low for tok in EXCLUDE_TOKENS):
            continue
        dst = stage / "interface" / src.name
        copy_file(src, dst)
        copied_interface.append(str(dst.relative_to(stage)))

    # Final aggregate scientific summaries.
    aggregate_names = [
        "WFP_AUTHORITATIVE_RESULTS_SUMMARY.csv",
        "WFP_AUTHORITATIVE_RESULTS_SUMMARY.json",
        "WFP_AUTHORITATIVE_RESULTS_SUMMARY.md",
        "WFP_FINAL_SCIENTIFIC_STATUS.md",
        "WFP_FINAL_CLOSEOUT_READOUT.txt",
    ]
    for name in aggregate_names:
        src = closeout / name
        if src.is_file():
            copy_file(src, stage / "results" / "aggregate" / name)

    copy_file(pubreadout, stage / "results" / "aggregate" / pubreadout.name)
    copy_file(interface_audit, stage / "docs" / "reproducibility" / interface_audit.name)

    # Final release figures: one PDF per accepted figure.
    copy_file(fig1, stage / "figures" / "Release_Figure_1_population_geometry.pdf")
    copy_file(fig2, stage / "figures" / "Release_Figure_2_between_within_scale.pdf")

    # Captions / presentation boundary.
    for name in ["WFP_RELEASE_FIGURE_CAPTIONS.md", "WFP_RELEASE_PRESENTATION_BOUNDARY.md"]:
        src = pubassets / name
        if src.is_file():
            copy_file(src, stage / "figures" / name)

    # Final tables.
    table_dir = pubassets / "tables"
    if not table_dir.is_dir():
        raise SystemExit(f"FAIL publication table directory missing: {table_dir}")
    for src in sorted(table_dir.glob("*.csv")):
        copy_file(src, stage / "tables" / src.name)
    if (pubassets / "WFP_RELEASE_TABLES.md").is_file():
        copy_file(pubassets / "WFP_RELEASE_TABLES.md", stage / "tables" / "WFP_RELEASE_TABLES.md")

    # Minimal figure-ready aggregate inputs for deterministic re-rendering.
    fig0 = project / "results" / "wfp_fig0_exports"
    for name in [
        "fig2_reconstruction_curves.csv",
        "fig2_summary_metrics.csv",
    ]:
        copy_file(ensure_file(fig0 / name, name), stage / "results" / "figure_ready" / name)

    # Bundled aggregate source tables used by publication renderer.
    for src in sorted(pubsrc.glob("*.csv")):
        copy_file(src, stage / "results" / "figure_ready" / src.name)

    # Release renderer must be present under publication code.
    renderer = stage / "code" / "publication" / "wfp_render_release_publication_assets.py"
    if not renderer.is_file():
        raise SystemExit(
            "FAIL final release renderer was not copied. "
            "Expected code/wfp_render_release_publication_assets.py in private project."
        )

    # Reproduction helper.
    reproduce = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:-$ROOT/reproduced_release_assets}"

"$PYTHON" "$ROOT/code/publication/wfp_render_release_publication_assets.py" \
  --fig0-dir "$ROOT/results/figure_ready" \
  --data-dir "$ROOT/results/figure_ready" \
  --out "$OUT"

echo
echo "Reproduced release assets under: $OUT"
"""
    write_text(stage / "tools" / "reproduce_release_assets.sh", reproduce)
    (stage / "tools" / "reproduce_release_assets.sh").chmod(0o755)

    # Observed environment.
    env = observed_environment()
    write_text(
        stage / "environment" / "ENVIRONMENT.json",
        json.dumps(env, indent=2, sort_keys=True),
    )
    req = "\n".join(f"{k}=={v}" for k, v in env["packages"].items() if v != "NOT_INSTALLED")
    write_text(stage / "environment" / "requirements-observed.txt", req)

    github_url = f"https://github.com/{a.github_user}/{a.repo_name}"

    readme = f"""# WF-P: Population Geometry of Arterial Blood Pressure Waveform Morphology

**Release status:** pre-WF3 public freeze, version {VERSION} candidate.

WF-P asks whether low-dimensional arterial blood pressure (ABP) morphology
within individual patients is embedded in a reproducible population-common
morphology space. The principal release object is a patient-balanced,
replicate-stable, held-out-generalizable eight-dimensional coordinate system
(**B8**) for 30-min central ABP morphology.

This repository is a release-safe research compendium. It is intentionally not
a copy of the private MIMIC working directory.

## Why this release exists

The frozen B8 interface is being publicly versioned **before WF3 longitudinal
scientific effects are analyzed**. Later WF3 analyses are expected to project
longitudinal morphology into this exact coordinate system without relearning,
rotating, or re-signing B8.

This is a provenance/freeze claim, not a claim that WF-P was independently
externally validated.

## Scientific status

- Source role: discovery / derivation.
- Source cohort: MIMIC-III Validation1000 source; 978 patients were analysable
  under the frozen rules.
- Independent confirmatory WF-P validation: **not yet performed**.
- Frozen population dimension: **d95 = 8** (d90 = 6).
- Held-out B8 reconstruction R²: approximately **0.964**.
- Scale hierarchy in frozen B8:
  - between-patient RMS distance: **3.161**;
  - within-patient 60-s RMS movement: **0.679**;
  - odd/even replicate RMS discrepancy: **0.139**.
- Prespecified age/sex/height models showed weak out-of-sample conditional-mean
  prediction of B8, but residual multivariate dependence remained detectable.
- The prespecified coarse chronic-phenotype block did not materially improve
  B8 prediction beyond baseline covariates.

## Frozen interface

See [`interface/`](interface/).

The release includes:

- 64-vector population center;
- frozen 64×8 B8 basis;
- selected and full eigenvalue profiles;
- ordinary between-person covariance;
- replicate-corrected between-person operator;
- short-window within-person covariance;
- exact axis orientation convention;
- exact projection specification;
- release-local SHA256 manifest.

For an already normalized 64-vector central morphology `x64`, the frozen
row-vector projection is

```text
z = (x64 - population_center) @ frozen_B8_basis
```

Raw mmHg waveform samples are not projected directly.

## Interpretation boundary

WF-P alone does **not** establish that:

- a B8 axis is a physiological state;
- a B8 axis is a stable individual trait;
- B8 is statistically independent of age, sex, or height;
- B8 has disease-specific or treatment-response meaning;
- the MIMIC-derived geometry is externally generalizable.

The 30-min patient representative is called **central morphology**, not trait.

## Relationship to upstream releases

WF-P inherits waveform/beat definitions from prior frozen ABP work and does not
alter those upstream scientific definitions.

- **WF1 — arterial-pressure waveform dimensionality and sampling fidelity**  
  Zenodo DOI: `{a.wf1_doi}`
- **WF2 — arterial-pressure waveform geometry**  
  Zenodo DOI: `{a.wf2_doi}`

## Repository layout

```text
code/
  scientific/      frozen/result-bearing WF-P analysis sources
  audit/           integrity, interface, closeout, release audits
  publication/     final release-asset renderer
freeze/            release-safe frozen specifications/amendments
results/
  aggregate/       cohort-level authoritative summaries/readouts
  figure_ready/    minimal aggregate inputs for figure reproduction
interface/         frozen B8 release interface
figures/           two accepted release figures
tables/            release tables
tools/             relative-path reproduction helper
environment/       observed software environment
docs/
  reproducibility/ public-safety evidence
```

## Reproduce the release figures/tables

From the repository root:

```bash
./tools/reproduce_release_assets.sh
```

This reproduces presentation assets from aggregate release-safe inputs. It does
not download or expose patient-level MIMIC data.

## Data source and redistribution boundary

The upstream waveform source is the **MIMIC-III Waveform Database Matched
Subset, version 1.0** (PhysioNet DOI `10.13026/c2294b`).

This repository does **not** redistribute raw MIMIC waveforms, patient-level
B8 scores, patient/record identifiers, private clinical linkage tables,
checkpoints, or local execution archives.

## Licensing

This repository uses a mixed-license structure:

- source code: MIT License (`LICENSE_CODE.txt`);
- aggregate derived numeric results: ODbL 1.0 notice (`DATA_LICENSE.md`) to the
  extent database rights apply;
- figures and documentation: copyright {a.author_name}; see
  `FIGURES_AND_DOCUMENTATION_LICENSE.md`.

## Citation

Citation metadata are provided in `CITATION.cff`.

GitHub repository: {github_url}

A version-specific Zenodo DOI should be added only after an exact tagged GitHub
release is archived.
"""
    write_text(stage / "README.md", readme)

    cff = f"""cff-version: 1.2.0
message: "If you use this release, please cite the archived software/research release. Please also cite the accompanying manuscript when available."
title: "WF-P: Population Geometry of Arterial Blood Pressure Waveform Morphology"
type: software
authors:
  - family-names: "Senda"
    given-names: "Atsushi"
    orcid: "{a.orcid}"
abstract: "Release-safe code, frozen aggregate outputs, and a versioned B8 interface for a patient-balanced, replicate-stable population geometry of arterial blood pressure waveform morphology."
keywords:
  - "arterial blood pressure"
  - "waveform morphology"
  - "population geometry"
  - "principal components"
  - "reproducible research"
repository-code: "{github_url}"
url: "{github_url}"
version: "{VERSION}"
"""
    write_text(stage / "CITATION.cff", cff)

    changelog = f"""# Changelog

## {VERSION} — pre-release staging

Initial public WF-P release candidate.

### Included

- frozen discovery/derivation scientific code and release-safe specifications;
- frozen B8 interface for later WF3 use;
- public-safety-audited aggregate covariance/operator files;
- two final release figures;
- five aggregate release tables;
- minimal aggregate figure-ready inputs;
- relative-path reproduction helper;
- observed software environment.

### Data-safety boundary

No raw MIMIC waveform/clinical files, patient-level B8 scores, patient or
record identifiers, checkpoints, private case-filename manifests, or local
absolute-path manifests are distributed.

### Scientific boundary

This release is a pre-WF3 public freeze. It does not claim independent
confirmatory WF-P validation or trait/state interpretation of B8 axes.
"""
    write_text(stage / "CHANGELOG.md", changelog)

    mit = f"""MIT License

Copyright (c) 2026 {a.author_name}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    write_text(stage / "LICENSE_CODE.txt", mit)

    data_license = """# Data license notice

Release-safe aggregate numeric results under `results/`, `interface/`, and
`tables/` are distributed under the Open Data Commons Open Database License
(ODbL) version 1.0 to the extent database rights apply.

Full license terms:
https://opendatacommons.org/licenses/odbl/1-0/

This notice does not grant redistribution rights to the underlying MIMIC-III
source data. Raw and patient-level MIMIC-derived data are not included here.
"""
    write_text(stage / "DATA_LICENSE.md", data_license)

    fig_license = f"""# Figures and documentation

Copyright (c) 2026 {a.author_name}.

The figures and documentation in this repository are provided for scientific
review, citation, and reproducibility of this release. No additional reuse
license is granted unless stated in the corresponding publication or file.

Code and aggregate numeric result files are governed separately by the license
notices in this repository.
"""
    write_text(stage / "FIGURES_AND_DOCUMENTATION_LICENSE.md", fig_license)

    gitignore = """.DS_Store
__pycache__/
*.pyc
.venv/
venv/
reproduced_release_assets/
local_audit/
"""
    write_text(stage / ".gitignore", gitignore)

    provenance = {
        "release_candidate_version": VERSION,
        "scientific_effects_calculated_by_staging": False,
        "raw_waveforms_opened_by_staging": False,
        "patient_level_files_intentionally_included": False,
        "frozen_B8_changed": False,
        "source_project": "private local working tree; not distributed",
        "upstream_releases": {
            "WF1_DOI": a.wf1_doi,
            "WF2_DOI": a.wf2_doi,
        },
        "code_files_copied": copied_code,
        "freeze_files_copied": copied_freeze,
        "interface_files_copied": copied_interface,
    }
    write_text(stage / "STAGING_PROVENANCE.json", json.dumps(provenance, indent=2, sort_keys=True))

    # Public relative-path manifest.
    rows = manifest_files(stage)
    with (stage / "PUBLIC_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    write_text(
        stage / "PUBLIC_MANIFEST.json",
        json.dumps({"schema_version": 1, "files": rows}, indent=2, sort_keys=True),
    )

    readout = f"""WF-P PUBLIC GITHUB STAGING BUILD
================================
Decision: WFP_PUBLIC_STAGING_BUILD_COMPLETE
Scientific effects calculated: NO
Raw waveform arrays opened: NO
Patient-level files intentionally copied: NO
Frozen B8 changed: NO
Git initialized: NO
GitHub push performed: NO
Zenodo upload performed: NO

Staging root:
  {stage}

Release candidate:
  {a.github_user}/{a.repo_name}
  version {VERSION}

Copied:
  code files: {len(copied_code)}
  freeze files: {len(copied_freeze)}
  interface files: {len(copied_interface)}
  release figures: 2

Next step:
  Run wfp_audit_public_staging.py against this staging tree.
  Do NOT run git init until that audit reports PASS.
"""
    print(readout)
    write_text(stage.parent / f"{stage.name}_BUILD_READOUT.txt", readout)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
