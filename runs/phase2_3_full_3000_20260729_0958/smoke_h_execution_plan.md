# Phase 2.3 Strategy H Smoke Plan

## Objective

Test a balanced F-D hybrid without starting formal training. H keeps F's usable
direct/indirect separation while adding a conservative version of D's set-level
balance loss to prevent clean collapse.

## Configuration

- Start from the existing Full2000 step-2000 LoRA checkpoint.
- Train exactly 120 smoke steps on the audited 900-record, 300/300/300 balanced
  boundary dataset with complete clean/direct/indirect triplets.
- Use F's low learning rates and flat classifier, but reduce diagonal ranking
  pressure from 0.80/0.40 to 0.35/0.20.
- Add D's generic margin and coverage mechanism conservatively: logit margin
  0.30 and triplet balance weight 0.15.
- Retain a smaller direct margin of 0.20. Do not use E's 0.90 direct margin.
- Write only to `artifacts/smoke_h/checkpoints`; formal `artifacts/checkpoints`
  must remain empty.

## Gates

1. One-step dry run must load Full2000, form ordered triplets, and write only
   under the smoke directory.
2. During training, stop only on NaN/Inf, process failure, or absent checkpoint.
3. After step 120, require a complete checkpoint and run pair-suite120 first.
4. If any class recall is zero, stop H evaluation and do not run mixed150.
5. Only if pair-suite passes all recalls >= 0.35, direct F1 >= 0.35, and macro
   F1 >= 0.45 may mixed150 run. Formal training remains user-confirmed only.
