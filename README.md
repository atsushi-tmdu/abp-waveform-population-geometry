# WF-P: Population Geometry of Arterial Blood Pressure Waveform Morphology

**Release status:** pre-WF3 public freeze, version 1.0.0 candidate.

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
  Zenodo DOI: `10.5281/zenodo.21940412`
- **WF2 — arterial-pressure waveform geometry**  
  Zenodo DOI: `10.5281/zenodo.22020208`

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
- figures and documentation: copyright Atsushi Senda; see
  `FIGURES_AND_DOCUMENTATION_LICENSE.md`.

## Citation

Citation metadata are provided in `CITATION.cff`.

GitHub repository: https://github.com/atsushi-tmdu/abp-waveform-population-geometry

A version-specific Zenodo DOI should be added only after an exact tagged GitHub
release is archived.
