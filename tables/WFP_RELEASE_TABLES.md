# WF-P release tables

## Table 1 population geometry

| metric | value | unit | stage |
| --- | --- | --- | --- |
| Source patients | 1000 | patients | Discovery |
| Analysable patients | 978 | patients | Discovery |
| Positive-spectrum effective rank | 5.78163 | dimensionless | Discovery |
| d90 | 6 | dimensions | Discovery |
| d95 / frozen B8 dimension | 8 | dimensions | Discovery |
| Held-out B8 reconstruction R2 | 0.964044 | R2 | Discovery |
| Ordinary PCA R2 at d=8 | 0.964048 | R2 | Comparator |
| Fourier R2 at d=8 | 0.868815 | R2 | Comparator |
| Matched-random 95th-percentile R2 | 0.176313 | R2 | Comparator |
| Half-split subspace overlap median | 0.977671 | overlap | Stability |
| Within-window variance captured by B8 | 0.923148 | fraction | Between-within alignment |
| Between/within projector overlap | 0.945115 | overlap | Between-within alignment |

## Table 2 between within scale

| metric | value | unit |
| --- | --- | --- |
| Between-patient pairwise RMS | 3.16145 | B8 distance |
| Between-patient pairwise median | 2.45305 | B8 distance |
| Population-center radius RMS | 2.23434 | B8 distance |
| Nearest-neighbor median | 0.463522 | B8 distance |
| Nearest-neighbor q05 | 0.250757 | B8 distance |
| Nearest-neighbor q95 | 1.04523 | B8 distance |
| Within-patient equal-patient RMS | 0.679064 | B8 distance |
| Patient within-RMS median | 0.41661 | B8 distance |
| Block displacement median | 0.313912 | B8 distance |
| Block displacement q95 | 1.33806 | B8 distance |
| Adjacent 60-s step median | 0.190629 | B8 distance |
| Odd/even replicate RMS | 0.138692 | B8 distance |
| Odd/even replicate median | 0.062683 | B8 distance |
| Between / replicate | 22.7947 | ratio |
| Within / between | 0.214795 | ratio |
| Within / replicate | 4.89619 | ratio |
| Patients with p95 block displacement >= nearest-neighbor | 0.705521 | fraction |
| Patients with max block displacement >= nearest-neighbor | 0.805726 | fraction |

## Table 3 constitutional sensitivity

| stage | estimand | value | unit | note |
| --- | --- | --- | --- | --- |
| WFP0 | Conventional factors -> B8 | 0.136957 | OOF R2 | level + log(scale) + log(duration) |
| Stage7B | Age + sex -> B8 | 0.029482 | OOF R2 | primary linear constitutional model |
| Stage7B | Residual trace after age + sex | 0.970517 | fraction | linear conditioning only |
| Stage7B | Age + age^2 + sex | 0.026249 | OOF R2 | quadratic sensitivity |
| Stage7B | Height increment beyond age + sex | -0.001292 | delta OOF R2 | complete-case n=693 |
| Stage7B-NL | RCS age increment M1-M0 | -0.006409 | delta OOF R2 | 4-knot RCS |
| Stage7B-NL | RCS age-by-sex total M2-M0 | -0.010637 | delta OOF R2 | limited interaction |
| Stage7B-NL | Nonlinear height increment H2-H1h | -0.005638 | delta OOF R2 | complete-case n=693 |
| Stage7B-NL | Height-by-sex increment H3-H2 | -0.00567 | delta OOF R2 | 1-df secondary interaction |
| Stage7B-NL | Full residual dCor | 0.167447 | distance correlation | OOF-linear residuals |
| Stage7B-RD | Full pipeline-null q95 | 0.102055 | distance correlation | 999 pipeline replays |
| Stage7B-RD | Full pipeline-control Holm p | 0.002 | p value | artifact-control sensitivity |
| Stage7B-NL | Height-subset residual dCor | 0.180459 | distance correlation | n=693 |
| Stage7B-RD | Height pipeline-null q95 | 0.132398 | distance correlation | 999 pipeline replays |
| Stage7B-RD | Height pipeline-control Holm p | 0.002 | p value | artifact-control sensitivity |
| Stage7C | Baseline OOF R2 | 0.182252 | OOF R2 | age/sex + conventional factors |
| Stage7C | Joint 8-phenotype OOF R2 | 0.17505 | OOF R2 | exact admission n=887 |
| Stage7C | Phenotype block incremental OOF R2 | -0.007202 | delta OOF R2 | 8 phenotypes jointly |
| Stage7C | FDR-significant global phenotypes | 0 | count | BH-FDR 0.05 |

## Table 4 chronic phenotype mapping

| phenotype | exposed_n | marginal_delta_oof_r2 | unique_delta_oof_r2 | adjusted_shift_norm | shift_over_between_rms | shift_over_within_rms | permutation_p | BH_q | FDR05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| congestive_heart_failure | 238 | 6.3e-05 | -0.001355 | 0.222186 | 0.0703 | 0.3272 | 0.115442 | 0.296852 | False |
| cardiac_arrhythmias | 356 | 0.000924 | -0.000152 | 0.263054 | 0.0832 | 0.3874 | 0.0134933 | 0.107946 | False |
| valvular_disease | 111 | -0.001164 | -0.001381 | 0.335748 | 0.1062 | 0.4944 | 0.0454773 | 0.181909 | False |
| peripheral_vascular_disease | 69 | -0.001629 | -0.002233 | 0.27123 | 0.0858 | 0.3994 | 0.354323 | 0.46491 | False |
| hypertension | 522 | -0.001283 | -0.001319 | 0.11802 | 0.0373 | 0.1738 | 0.652174 | 0.652174 | False |
| diabetes | 255 | 0.000817 | 0.000433 | 0.199178 | 0.063 | 0.2933 | 0.148426 | 0.296852 | False |
| renal_failure | 138 | -9.5e-05 | -0.001084 | 0.231558 | 0.0732 | 0.341 | 0.236382 | 0.378211 | False |
| chronic_pulmonary_disease | 212 | -0.002466 | -0.002082 | 0.161457 | 0.0511 | 0.2378 | 0.406797 | 0.46491 | False |

## Table S1 axis reliability

| axis | odd_even_score_correlation | odd_sd | even_sd |
| --- | --- | --- | --- |
| 1 | 0.998595 | 1.44105 | 1.4468 |
| 2 | 0.998007 | 1.09114 | 1.09913 |
| 3 | 0.998149 | 0.854182 | 0.854904 |
| 4 | 0.997707 | 0.623249 | 0.625387 |
| 5 | 0.998351 | 0.43982 | 0.440699 |
| 6 | 0.995308 | 0.401598 | 0.403214 |
| 7 | 0.996841 | 0.371272 | 0.371158 |
| 8 | 0.99537 | 0.323949 | 0.324188 |
