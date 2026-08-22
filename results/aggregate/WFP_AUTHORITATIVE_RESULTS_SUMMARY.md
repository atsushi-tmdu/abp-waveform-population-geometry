# WF-P Authoritative Results Summary

**Status:** FINAL CLOSEOUT SOURCE SUMMARY

This document aggregates already-completed WF-P analyses. It does not introduce a new scientific effect.

## Core result hierarchy

1. A low-dimensional patient-balanced population morphology space was identified; the frozen interface dimension is B8.
2. Between-patient separation is much larger than odd/even replicate discrepancy, while 60-s within-patient movement is non-negligible.
3. Conventional level/scale/timing variables and measured constitutional variables provide limited OOF mean predictability for B8.
4. Low-df nonlinear age/height models and limited interactions do not materially improve OOF mean prediction.
5. Residual constitutional dependence remains detectable after pipeline-replay and within-fold artifact controls; therefore independence/orthogonality is NOT claimed.
6. Eight prespecified coarse chronic phenotypes add no OOF predictive value as a block, and no global phenotype association survives BH-FDR 0.05.

## Selected authoritative numbers

| Section | Metric | Value |
|---|---|---:|
| population_geometry | effective_rank | 5.781627871619092 |
| population_geometry | d95 | 8 |
| population_geometry | cv_r2_all | 0.9640442363855233 |
| population_geometry | half_split_overlap_median | 0.9776711907405065 |
| representation_identifiability | conventional_to_B8_oof_r2 | 0.136957 |
| constitutional_mean | age_sex_oof_r2 | 0.029482 |
| between_within_scale | between_pairwise_rms | 3.161452 |
| between_within_scale | within_equal_patient_rms | 0.679064 |
| between_within_scale | odd_even_replicate_rms | 0.138692 |
| constitutional_nonlinear_sensitivity | M1_minus_M0 | -0.006409 |
| constitutional_nonlinear_sensitivity | M2_minus_M0 | -0.010637 |
| constitutional_residual_dependence | full_residual_dcor | 0.167447 |
| constitutional_residual_dependence | height_residual_dcor | 0.180459 |
| chronic_phenotype_mapping | phenotype_block_delta_oof_r2 | -0.007202 |
| chronic_phenotype_mapping | FDR_significant_global_phenotypes | 0 |

## Final wording boundary

Recommended wording:

> The frozen B8 coordinates showed little out-of-sample mean predictability from age, sex, and height across prespecified linear, low-degree nonlinear, and limited-interaction models. However, model-free dependence of the residual B8 coordinates on these constitutional variables remained detectable after pipeline-replay and fold-stratified artifact controls.

Do **not** write that B8 is independent of, unrelated to, or fully orthogonal to age/sex/height.

## WF3 interface

WF-P does not label any axis as Ztrait or Zstate. The next scientific question is whether the frozen coordinates show reproducible same-patient longitudinal movement and whether that movement maps to time-varying physiology/treatment/recovery.
