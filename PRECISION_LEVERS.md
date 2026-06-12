# Precision levers — driving binary PII-present precision up on real free text

The per-iteration runbook for **Phase 3** of the high-precision pivot. It assumes the held-out real set
(`data/gold/real-v1/`) and the honest baseline (`data/eval/real-baseline/`) exist (build/eval commands
are in the session handoff / README). Every lever is applied **TDD-first** and accepted **only** by the
precision-first gate: binary precision must hold/improve, binary recall must stay ≥ the floor
(default 0.97), health-tier under-classification must stay 0%.

## One iteration

```bash
# 1. eval the current model on the real set (one inference pass -> full PR curve)
python eval_binary.py --gold data/gold/real-v1/test.jsonl --engine hybrid \
    --sweep 0.30:0.90:0.02 --operating-threshold 0.7 --recall-floor 0.97 \
    --out data/eval/precision-NNN/before --binary-mode A

# 2. see the precision enemies (FP-ranked; prints the top spurious/false-positive elements)
python loop_iter.py errors --eval data/eval/precision-NNN/before --out errors/precision-NNN.jsonl
#    -> the "top spurious (false-positive) elements" list IS your lever priority order.
#    Also read data/eval/precision-NNN/before/{REPORT.md, mismatches.jsonl} (direction=false_positive).

# 3. pick the lever for the dominant FP element (below); add a FAILING test, make it pass,
#    keep ALL of test_failures.py / test_girp.py green.

# 4. re-eval to .../after, then gate
python eval_binary.py --gold data/gold/real-v1/test.jsonl --out data/eval/precision-NNN/after ...
python loop_iter.py decide --before data/eval/precision-NNN/before --after data/eval/precision-NNN/after
#    accept ONLY on ACCEPT (binary precision up / held, recall ≥ floor, health 0%).

# 5. on accept: python eval_gate.py --gold data/gold/real-v1/test.jsonl   (must PASS)
#    then tag a PATCH (Stage-1 lever) via metrics_io.tag_model.
```

## Levers — routed by where the false positive originates

**Lever A — per-label threshold.** For **fuzzy/zero-shot** FPs (person, address, date of birth, phone,
passport, driver's licence). Raise the per-label confidence bar in `aupii.PER_LABEL_THRESHOLDS` — a
single map consulted by **both** `evaluate.derive` (eval) and the production twins, so eval == prod.
Empty by default (no change). Calibrate per label L from the curve:

```bash
for t in 0.70 0.75 0.80 0.85 0.90; do
  # set aupii.PER_LABEL_THRESHOLDS = {"<L>": t}, then:
  python eval_binary.py --gold data/gold/real-v1/test.jsonl --out /tmp/sweep_${L}_$t ... ; done
# pick the smallest t where L's per-entity precision clears your target (≥0.90) AND binary recall ≥ floor.
```

- DOB / bank are *isolation-Public/Private* — suppressing a weak one rarely flips the binary flag (the
  name usually still fires), so they are cheap precision wins.
- Driver's licence / passport / TFN are *Confidential-in-isolation* — suppressing one **can** drop the
  flag; validate binary recall stays ≥ floor before accepting.
- ⚠️ **Activation:** the production-twin branch (`if PER_LABEL_THRESHOLDS:` in `aupii.py`) only runs when
  the map is non-empty and cannot be exercised on the no-presidio dev Mac. Before trusting it in
  production, run `python test_aupii.py` **and** a hybrid smoke (`python selfcheck.py`) on the RTX box.

**Lever B — structured tightening.** For **Presidio-side** FPs (bank account, tax file number, medicare,
credit card). These come from the checksum engine, *not* the fuzzy model, so raising `PER_LABEL_THRESHOLDS`
does nothing. Instead raise the Presidio score floor (`evaluate.PRESIDIO_THR` / `presidio_elements`
`score_threshold`) for the offending entity, or add required `context=[...]` to its recognizer (see the
`AU_BSB` recognizer in `aupii.build_analyzer`).

**Lever C — validator hardening.** For **format-identifiable** FPs (a "licence" that is a bare year, a
"DOB" that is a timestamp). Tighten `girp.is_valid_entity(label, value)` — a pure function both paths
call. Add the FP as a `(label, value, False, why)` case **and** a real positive as a `(.., True, ..)`
recall guard in `VALIDATION_CASES`. The iter-002 DOB-separator rule (`girp.py`) is the template.

**Lever D — corroboration.** For **lone weak-entity** FPs (a single DOB/licence with no other PII
flipping a field to flagged). Require a second signal (another element, a context keyword, or structured
confirmation) before a lone weak entity raises the flag — a post-filter over the derived elements+spans
in `evaluate.derive` and the twins (leave `classify_elements` untouched). Recall-risky; adopt only for
the narrow lone-entity case and validate against the floor.

## Stage-2 escalation

When Stage-1 levers plateau, the FP rows they can't fix plus the spot-check negatives
(`python spotcheck.py route ...`) accumulate in `data/hard_examples.jsonl`. On the RTX box:

```bash
python train_lora.py --data data/hard_examples.jsonl --out ./model-finetuned --max-chars 2000
```

No-entity negatives teach suppression (full GIRP vocab → empty), the synthetic blend preserves recall.
Re-eval on real-v1 + the binary-precision gate; `metrics_io.tag_model` bumps a MINOR.
