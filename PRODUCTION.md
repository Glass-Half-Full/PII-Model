# Improving the model & production readiness

Evidence-based findings from benchmarking on a public labeled PII dataset (see `BASELINE.md`),
plus a roadmap to a production-grade solution.

## Adversarial review — resolved
A Codex adversarial review flagged three issues; all are addressed:
- **[HIGH] Health + Confidential-ID under-classification** — `classify_elements` now escalates to
  *Highly Confidential* when sensitive info combines with **any** Confidential element (name+combination
  **or** card/TFN/Medicare/passport/licence). Regression tests added (`test_girp.py`, `test_failures.py`).
- **[MED] Hybrid tests weren't a gate** — `test_aupii.py` now skips cleanly without the extras; base
  suite (`test_girp.py`, `test_failures.py`) stays green; deps tiered.
- **[MED] Offline escape hatch** — `aupii.load_gliner_pii()` is now local/offline by default
  (`local_files_only`, offline env, revision pin, fail-fast). `selfcheck.py` proves no network is used.

New gates from the review's roadmap: `eval_gate.py` (labeled-holdout release gate with health-miss bar),
`test_failures.py` (failure-mode bait), `selfcheck.py` (auditable offline), and the recursive
enhancement tooling `weak_label.py` + `train_lora.py` (Part C below).

## What we tested (so you don't have to guess)

| Change | Micro-F1 | Speed | GIRP acc / over-class | Verdict |
|---|---|---|---|---|
| **base-v1** (205M) @0.5 | 76.9% | 16.6 rows/s | 57.3% / 36% | baseline |
| **large-v1** (435M) @0.5 | 77.2% | 5.6 rows/s | 54.3% / 39% | ❌ 3× slower, **no gain**, slightly worse |
| **privacy-filter-PII** @0.5 | 76.8% | 13.9 rows/s | 56.0% / 36% | ➖ better card/phone recall, worse address; no net win |
| per-type confidence thresholds | ≤77% | — | ≤59% | ❌ no gain over a global threshold |
| **base-v1 @ threshold 0.7** | 77.3% | 16.6 rows/s | **62.7% / 30%** | ✅ best balance — now the default |
| base-v1 @ threshold 0.9 | 76.1% | 16.6 rows/s | 65.3% / 24% | ✅ for minimal false flags (some recall loss) |

**Conclusion:** *model capacity is not the bottleneck* — the larger model gives ~0% gain for 3× cost.
The error is zero-shot **label ambiguity** on numeric IDs (account ↔ card ↔ IMEI ↔ IP). Gains come from
the rules/validation layer (done), threshold tuning (done), and — for a step change — **fine-tuning**.

## Recommended production configuration
- **`threshold=0.7`** (new default). Use `0.5` for maximum recall (compliance-first, fewest misses),
  `0.85–0.9` to minimise false flags.
- **GPU with fp16** (auto-selected); CPU is fine for batch jobs. OOM auto-recovery is built in.
- Keep `validate=True` (format validation) and the Luhn regex backstop on.
- **Real card data:** cards are Luhn-valid in production, so enabling a Luhn-strict card check (see
  `regex_elements`) sharply cuts account→card false positives — the public benchmark uses non-Luhn
  synthetic cards, so it understates this.

## Additional measures to improve further (prioritised)

1. **Fine-tune on your labeled data — the biggest lever.** gliner2 supports LoRA fine-tuning.
   Label a few thousand of *your* rows (or weak-label with the current model, then human-correct the
   disagreements), fine-tune, and expect a step change in precision on your distribution. Needs a GPU.
   Outline:
   - Build a span-labeled training set in gliner2's format (text + entity spans/labels).
   - `model.apply_lora(r=8)`, train with a small `TrainingConfig(fp16=True, batch_size=4, …)`.
   - `model.merge_lora()` → `model.save_pretrained("./model-finetuned")` and ship it locally.
2. **Calibrate on YOUR data.** Re-run a `benchmark.py`-style eval on a labeled sample of real rows to
   pick the threshold and confirm the GIRP element mapping for your distribution (the public benchmark
   is a worst-case proxy; real customer notes are far less PII-dense).
3. **Human-in-the-loop review band.** Auto-accept high-confidence (>0.85); route borderline
   (0.7–0.85) to a reviewer. Cuts both false flags and misses where it matters most.
4. **Per-column label scoping.** Let each column declare expected PII types (a “sentiment” column
   needn’t scan for TFNs). Fewer competing labels → less ambiguity → fewer false positives.
5. **Domain allow/deny lists.** Maintain allowlists for known non-PII tokens that misfire (product
   names, internal codes) and denylists for must-flag terms. Cheap, high precision.
6. **Stronger structured-PII regexes.** Add checksum-validated patterns (SSN, TFN, Medicare, IBAN/BSB)
   to complement the model on well-formatted IDs.
7. **Monitoring & drift.** Log the level distribution + confidences over time; alert on shifts;
   re-evaluate periodically against a held-out labeled set.

## Operational hardening (already in place / recommended)
- **Offline & private:** 100% local, no API calls — data never leaves the host. ✅
- **Robust:** format validation, regex backstop, phone normalisation, CUDA-OOM auto-recovery. ✅
- **Auditable:** `explain()` returns the level + which rule fired; log it alongside the output.
- **Reproducible:** pin the model + code; record the run config (threshold, label set) with results.
- **Throughput:** batch + GPU fp16; lower `batch_size` only if VRAM is tight (auto-handled).

## Bottom line
The current pipeline is a solid, production-shippable **baseline** (micro-F1 ~77%, email 100%, names/
phones/addresses ~80%+, conservative over-classification you can tune). The next real accuracy jump is
**fine-tuning on labeled domain data**, not a bigger off-the-shelf model.
