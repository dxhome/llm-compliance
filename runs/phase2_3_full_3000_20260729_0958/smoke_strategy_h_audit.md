# Phase 2.3 Strategy H Smoke Audit

## Execution

- H completed 120/120 smoke steps from the Full2000 LoRA initialization.
- Training loss decreased from 9.9239 at step 1 to 3.1260 at step 115.
- Step 40, 80, and 120 model and training-state checkpoints were written under
  `artifacts/smoke_h/checkpoints`.
- No NaN, Inf, traceback, or formal-training checkpoint was observed.

## Pair-suite120 gate

The first stratified 50-record chunk contained 17 clean, 17 direct, and 16
indirect records. It produced:

| Class | Recall | F1 |
| --- | ---: | ---: |
| clean | 1.000 | 0.507 |
| direct | 0.000 | 0.000 |
| indirect | 0.000 | 0.000 |

- Accuracy: 0.340
- Macro F1: 0.169
- Prediction behaviour: all 50 records were predicted as clean.

## Decision

- H violates the hard zero-recall rule for both direct and indirect.
- Pair-suite evaluation was stopped after the completed first chunk; mixed150
  was not run.
- H is not a formal-training candidate. `artifacts/checkpoints` remains empty.

## Diagnosis

The conservative D balance term and generic margin did restore a strong clean
decision region relative to F, but together with the reduced diagonal loss they
over-corrected F and removed its direct/indirect separation. The loss curve is
therefore not a sufficient selection signal. Any next smoke must explicitly
protect direct and indirect recall while adding clean calibration more locally,
rather than applying a global clean-favouring correction.
