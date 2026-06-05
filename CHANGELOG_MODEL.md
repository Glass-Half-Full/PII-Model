# Model changelog

Balanced GIRP accuracy per tagged version (newest first).

## v1.0.1 — 2026-06-06 (system-tuning, threshold 0.7, iter 1)

Balanced GIRP accuracy 84.4% (95% CI 81.2–87.6); under 5.7%; over 9.6%; health-under 0.0%.
Gold: v1. Gate: ACCEPT (held-out test). Parent: v1.0.0.
Stage-1: suppress low-precision birthplace + mother's maiden from OUTPUT (kept in extraction to avoid perturbing person detection — first attempt removing them from the label set regressed -1.7pp and was REJECTED by the held-out gate); fix _is_personish to accept surname 'Citizen'. 25 detection-gap rows -> data/hard_examples.jsonl.

## v1.0.0 — 2026-06-06 (baseline, threshold 0.7, iter 0)

Balanced GIRP accuracy 82.9% (95% CI 79.6–86.1); under 5.7%; over 11.2%; health-under 1.4%.
Gold: v1. Gate: baseline.
Baseline: current hybrid on real gold v1 (564 test rows). First honest, CI-backed measurement.

