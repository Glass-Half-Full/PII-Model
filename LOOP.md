# Recursive improvement loop — runbook (Claude as reviewer)

This is the procedure Claude (Claude Code, dev-time) follows to iteratively improve the GIRP PII
classifier. Production inference stays 100% offline; this loop runs only at development time.
Headline metric: **balanced GIRP accuracy** (mean of the four per-tier recalls). Stage 1 tunes the
deterministic system (thresholds, GIRP rules, validation, regex, mappings); Stage 2 fine-tunes weights.

## One iteration

```
branch: loop/iter-NNN
 1 EVALUATE   python evaluate.py --gold data/gold/v1/test.jsonl --engine hybrid \
                  --sweep 0.3:0.9:0.1 --out-version iter-NNN/before
 2 EXTRACT    python loop_iter.py errors --eval data/eval/iter-NNN/before \
                  --out errors/iter-NNN.jsonl --cap 80
 3 REVIEW     Claude reads errors/iter-NNN.jsonl and applies the rubric below to each error,
              writing one correction record per error to reviews/iter-NNN.jsonl.
 4 PLAN       Group reviews by root_cause; pick the highest-yield Stage-1 bucket this iteration.
 5 TDD FIX    Add a FAILING test to test_loop_regressions.py reproducing the bug -> make the
              minimal edit (girp.py / aupii.py) -> green -> full suite green.
 6 RE-EVAL    python evaluate.py ... --out-version iter-NNN/after
 7 DECIDE     python loop_iter.py decide --before data/eval/iter-NNN/before \
                  --after data/eval/iter-NNN/after          (writes loop_state.json)
 8 ACCUMULATE detection-gap rows -> data/hard_examples.jsonl ; gold fixes -> data/gold_fixes.jsonl
 9 TAG/REVERT accept -> metrics_io.tag_model (PATCH) + commit ; reject -> git restore the edit
```

**Stop when** (whichever first): balanced accuracy plateaus (K=3 iterations with < 0.2pp gain);
MAX_ITERS=15; or the remaining errors are all `gold_label_error` / `ambiguous` (nothing Stage-1
tunable left). On plateau with a non-empty `data/hard_examples.jsonl`, escalate to **Stage 2**
(see train_lora.py) — fine-tune on the accumulated hard examples, eval-gate, tag a MINOR, re-enter.

## Error taxonomy (what went wrong)

| tag | meaning |
|---|---|
| `false_positive` | predicted an element not in gold |
| `false_negative` | missed a gold element |
| `wrong_tier_over` | predicted tier rank > gold (over-classified) |
| `wrong_tier_under` | predicted tier rank < gold (**dangerous**) |
| `span_boundary` | right label, wrong char span |
| `gold_label_error` | gold is wrong; model is right (or an eval artifact, e.g. health synonym) |
| `ambiguous` | genuinely undecidable from the text alone |

## Root-cause taxonomy (why) → fix locus

| root_cause | fix locus | stage |
|---|---|---|
| `threshold_too_high` / `threshold_too_low` | global default or `aupii.GLINER_FUZZY_GROUPS` | 1 |
| `girp_rule_bug` | `girp.classify_elements` | 1 |
| `validation_bug` | `girp.is_valid_entity` | 1 |
| `regex_backstop_gap` | `girp.regex_elements` / `aupii._PHONE_RX` | 1 |
| `mapping_gap` | `aupii.PRESIDIO2GIRP` / label sets | 1 |
| `detection_gap_needs_training` | `data/hard_examples.jsonl` | 2 |
| `dataset_label_noise` | `data/gold_fixes.jsonl` (no code change) | — |

## Tune-vs-train decision rule

Read each error's `probe` field (computed by evaluate.py):
- `rescuable_at_floor` non-empty → the model DOES surface the missed element at threshold 0.3 but it
  was dropped at 0.7 or by validation/mapping → **Stage 1** (threshold / validation / mapping fix).
- `detection_gap` non-empty (and not rescuable) → the model NEVER surfaces it even at 0.3 →
  **Stage 2** (route the gold `{text, spans}` to `data/hard_examples.jsonl`).
- `spurious_elements` non-empty → a false positive → check `is_valid_entity` / threshold; if the model
  fires on bait the validator should reject, it's a `validation_bug` (Stage 1). Also synthesise a hard
  negative (same text, spurious span omitted) into `data/hard_examples.jsonl` for Stage 2.

On `wrong_tier_under`, when torn between tune and train, prefer the compliance-safe Stage-1 fix
(over-classify before under-classify) AND route to hard examples so the model is trained later.

## Claude review rubric (apply to each record in errors/iter-NNN.jsonl)

> 1. **Verify the gold.** Read the text. Is the gold tier correct under the GIRP rules (README table /
>    `girp.classify_elements`)? Invariants: lone name = Private; name + any combination element
>    (phone/email/DOB/address/birthplace/maiden) = Confidential; card/TFN/Medicare/passport/licence in
>    isolation = Confidential; health alone or health+name = Confidential; **health + any Confidential
>    element = Highly Confidential**; bank account is Private-in-isolation and does NOT escalate. If the
>    gold is wrong (or the mismatch is only a health-synonym artifact — model `medical condition`/`illness`
>    vs gold `health condition`, same tier), tag `gold_label_error`, route `gold_fixes`, STOP.
> 2. **Tag what went wrong** (`error_tags`). If `RANK[pred] < RANK[gold]` it is `wrong_tier_under` — prioritise.
> 3. **Find the root cause (one primary).** Use the `probe`: rescuable_at_floor → threshold/validation/mapping;
>    detection_gap → `detection_gap_needs_training`; spurious_elements that should have been rejected →
>    `validation_bug`. Inspect the relevant function (`is_valid_entity`, `classify_elements`,
>    `PRESIDIO2GIRP`) to localise.
> 4. **Decide fixability** (`stage1_tune` vs `stage2_train`) per the rule above.
> 5. **Write the correction record** (schema below) with a one-sentence `claude_rationale` and `routes`.

## Correction record schema (reviews/iter-NNN.jsonl — one per error)

```json
{"id": "...", "text": "...",
 "predicted": {"tier": "...", "elements": [...]}, "gold": {"tier": "...", "elements": [...], "spans": [[label,start,end]]},
 "error_tags": ["wrong_tier_under","false_negative"],
 "root_cause": "validation_bug", "fixability": "stage1_tune",
 "routes": ["gold_fixes"|"hard_examples"|"regression_test"],
 "claude_rationale": "one sentence", "reviewer": "claude-opus-4-8", "iter": 1}
```

## TDD fix discipline (Stage 1)

Every accepted fix starts as a FAILING test in `test_loop_regressions.py` (pure-function, offline,
dual `_run()`/pytest pattern). Constraints that gate each edit:
- prefer per-group threshold overrides in `aupii.GLINER_FUZZY_GROUPS` over moving the global default;
- `classify_elements` edits must keep `test_girp.py::test_monotonic_rank_ordering` green;
- `is_valid_entity` / regex edits must keep ALL `test_failures.py` bait arrays green (highest blast radius);
- full suite green (`test_girp.py`, `test_failures.py`, `test_aupii.py`, `test_loop_regressions.py`) before TAG.
A reverted iteration keeps its regression test only if it still passes against the reverted code.
