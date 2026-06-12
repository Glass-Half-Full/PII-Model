# Model changelog

Balanced GIRP accuracy per tagged version (newest first).

## Unreleased — high-precision pivot (binary PII-present flag on real free text)

Infrastructure for a pivot toward HIGH-PRECISION binary "PII present / absent" flagging, measured on
REAL complex free text. No new model weights — metrics are unchanged until the real-data baseline is run
on the GPU box. Added: a binary-flag precision metric (`evaluate.binary_flag_metrics`, with bootstrap CIs,
flowing through the sweep/headline/REPORT/CLI); a held-out REAL free-text eval set builder
(`gold_data/build_real.py` + `ingest_tab` for the Text Anonymization Benchmark) writing
`data/gold/real-v1/` (test-only, never trained); `eval_binary.py` (precision-recall-vs-threshold curve,
precision@recall-floor headline, per-source breakdown); precision-first loop gates (`loop_iter.decide` /
`eval_gate.gate_check` gate on binary precision + an absolute recall floor, balanced accuracy demoted to a
warning; `extract_errors` reviews false positives first); a model-assisted spot-check labeling harness
(`spotcheck.py`); a neutral-by-default per-label threshold mechanism (`aupii.PER_LABEL_THRESHOLDS`,
Lever A) consulted by both the evaluator and the production twins; and Stage-2 negative/anti-forgetting
handling (`train_lora` explicit no-entity negatives + `--max-chars` OOM guard). Runbook:
`PRECISION_LEVERS.md`.

**Measured binary PII-present baselines (mode A, current v1.1.0 weights, dev-machine CPU):**
- Synthetic gold v1 (564 rows, 482 present / 82 absent): base engine **94.5% precision @ 97.1% recall**
  (thr 0.75); hybrid **94.4% @ 97.7%** (thr 0.75) — confirms the confusion-matrix estimate, validates
  the metric on real model output.
- Real ECHR legal prose (Text Anonymization Benchmark, 127 docs, all PII-present): hybrid **person
  precision 100% / recall 94.5%**, binary recall 95.3%, and only **1 spurious element across 127
  multi-thousand-char documents** — real free text is far less over-classified than the PII-dense
  synthetic set. TAB has ~0 PII-absent rows, so real binary PRECISION still needs the PII-sparse PIILO
  essays (Kaggle).
- Lever-A per-entity sweep: DOB/licence/passport false positives are high-confidence (raising the
  threshold alone cannot reach 90% precision), so they need validator hardening (Lever C), not Lever A;
  `person` is already ≈95% precision. The remaining real-data steps (PIILO baseline, Stage-2) are in
  `RUNBOOK_REAL_DATA.md`.

## v1.1.0 — 2026-06-06 (fine-tune, threshold 0.7, iter 3)

Balanced GIRP accuracy 84.9% (95% CI 81.7–88.0); under 6.2%; over 8.3%; health-under 0.0%.
Gold: v1. Gate: ACCEPT (held-out test; +0.1pp within noise). Parent: v1.0.2.
Stage-2 LoRA fine-tune (r=8) on 90 accumulated hard examples + 150 synthetic, 3 epochs on Apple MPS (~40s). Balanced 84.7%->84.9% (+0.1pp, WITHIN CI noise). Milestone: the Stage-2 pipeline now runs end-to-end (train->merge->load->eval) with NO regression and no catastrophic forgetting (Highly recall stays 100%). Real Stage-2 gains require a larger hard-example pool accumulated over more loop iterations; the 25 detection-gap IDs from one iteration are too few to move zero-shot recall materially.

## v1.0.2 — 2026-06-06 (system-tuning, threshold 0.7, iter 2)

Balanced GIRP accuracy 84.7% (95% CI 81.6–87.9); under 6.2%; over 8.5%; health-under 0.0%.
Gold: v1. Gate: ACCEPT (held-out test). Parent: v1.0.1.
Stage-1: (a) raise health-pass threshold 0.4->0.6 (the 0.4 pass hallucinated health conditions in long finance/EDIFACT text; at 0.6 over-class drops, Highly recall stays 100%, health-under stays 0%); (b) require a date separator for DOB validation (bare YYYYMMDD finance timestamps were tagged DOB) -> DOB precision 32%->46%, 0 recall cost (all 32 gold DOBs have a separator). Over-class 9.6%->8.5%; Private recall +1.5pp.

## v1.0.1 — 2026-06-06 (system-tuning, threshold 0.7, iter 1)

Balanced GIRP accuracy 84.4% (95% CI 81.2–87.6); under 5.7%; over 9.6%; health-under 0.0%.
Gold: v1. Gate: ACCEPT (held-out test). Parent: v1.0.0.
Stage-1: suppress low-precision birthplace + mother's maiden from OUTPUT (kept in extraction to avoid perturbing person detection — first attempt removing them from the label set regressed -1.7pp and was REJECTED by the held-out gate); fix _is_personish to accept surname 'Citizen'. 25 detection-gap rows -> data/hard_examples.jsonl.

## v1.0.0 — 2026-06-06 (baseline, threshold 0.7, iter 0)

Balanced GIRP accuracy 82.9% (95% CI 79.6–86.1); under 5.7%; over 11.2%; health-under 1.4%.
Gold: v1. Gate: baseline.
Baseline: current hybrid on real gold v1 (564 test rows). First honest, CI-backed measurement.

