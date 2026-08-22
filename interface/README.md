# WF-P frozen B8 interface v1.0.0

This directory contains release-safe aggregate objects for applying the frozen
WF-P population morphology coordinate system in later work, including WF3.

## Key files

- `population_center_64.csv`: frozen 64-vector population center.
- `frozen_B8_basis_64x8.csv`: authoritative frozen 64x8 B8 basis.
- `selected_replicate_eigenvalues_8.csv`: eigenvalues corresponding to B8.
- `population_eigenspectra.csv`: full replicate-corrected and ordinary spectra.
- `Sigma_W_short_window_64x64.csv`: short-window within-person covariance.
- `Sigma_B_ordinary_64x64.csv`: ordinary covariance of 30-min patient central morphology.
- `S_rep_replicate_corrected_64x64.csv`: replicate-corrected symmetric between-person operator used as the final discovery primary operator.
- `S_rep_positive_64x64.csv`: positive-spectrum PSD form of `S_rep`.

Both ordinary `Sigma_B` and replicate-corrected `S_rep` are included because
they are distinct objects with distinct scientific roles.

## Projection

For an already normalized 64-vector central morphology `x64`:

`z = (x64 - population_center) @ frozen_B8_basis`

Do not project raw mmHg waveform samples directly.

## Orientation

Axis order is z1...z8. The exported basis preserves the exact stored
orientation from the frozen discovery artifact. No post-hoc sign flip, rotation,
or relearning is allowed.

## Replay integrity

- analysable n: 978
- frozen-rule exclusions: 22
- population-center max error: 0.000e+00
- Sigma_W max error: 0.000e+00
- selected-eigenvalue max error: 0.000e+00
- full S_rep spectrum max error: 9.975e-17
- full ordinary spectrum max error: 4.441e-16
- B8 projector max error: 0.000e+00

## Interpretation boundary

This interface does not label any B8 axis as physiological state or stable
trait, and does not imply constitutional independence, disease-specific meaning,
or treatment-response meaning.
