# Phase 2.3 Strategy F2 Smoke Plan

## Objective

Improve the standard Benchmark v2 direct boundary without changing the F data,
initialization, classifier structure, or triplet ranking loss.

## Evidence from F

On the frozen 150-record Benchmark v2 smoke set, F was the only strategy with
non-zero recall for all three classes, but direct F1 was 0.3373. Of 53 direct
records, 14 were correct, 25 were predicted indirect, and 14 were predicted
clean. The intervention must therefore strengthen the direct class against
both competing classes.

## Single-variable Change

- F2 is identical to F except `direct_margin: 0.55` instead of `0.35`.
- Keep flat classification, generic logit margin 0.25, diagonal triplet loss
  0.80/0.40, paired triplets, seed 411, learning rates, and 120-step budget.
- Start from the same Full2000 LoRA-only checkpoint; do not initialize from F.

## Execution and Gates

1. Run a one-step dry run and verify a complete ordered triplet, finite loss,
   and smoke-only checkpoint output.
2. Run exactly 120 smoke steps, saving steps 40/80/120. Stop only for NaN/Inf,
   process failure, or missing smoke checkpoint.
3. Evaluate step 120 once on the same frozen Benchmark v2 smoke 150 with
   batch size 1 and complete all records; do not early-stop on class recall.
4. F2 is eligible for a later full benchmark only if all three recalls are
   non-zero, direct F1 >= 0.35, macro F1 >= 0.45, and it does not regress F's
   clean/indirect recall to zero.

## Safety

- Write only under `artifacts/smoke_f2/` and `logs/smoke_strategy_f2_*`.
- Do not create or write `artifacts/checkpoints` and do not start formal
  3000-step training.
