# Real-data high-precision runbook (run on the RTX 2050 box)

The binary-precision foundation (metric, real-set builder, eval tool, precision gates, spot-check
harness, Lever-A mechanism) is built, unit-tested, and committed. The steps below are the **model
executions** that finish the job — they need the GPU box (torch + gliner2 + presidio installed) and the
`kaggle` CLI configured; they cannot run on the no-presidio dev machine. Model runtime stays offline
(`HF_HUB_OFFLINE=1`); only dataset downloads use the network. `python` = your interpreter with the stack.

## 0 · Confirm the new metric on existing gold (no Kaggle needed)
```bash
python evaluate.py --gold data/gold/v1/test.jsonl --engine hybrid --sweep 0.3:0.9:0.1 --out-version synth-binary-base
```
`data/eval/synth-binary-base/REPORT.md` now has a **Binary PII-present flag** section + `bin-P%`/`bin-R%`
sweep columns. Records the synthetic binary baseline (expected ≈94% precision / ≈98% recall).

## 1 · Source real free-text — the PII-sparse essays are what make precision measurable
```bash
kaggle competitions download -c pii-detection-removal-from-educational-data -p data/_raw/kaggle-pii
# accept the competition rules once on kaggle.com, then unzip so data/_raw/kaggle-pii/train.json exists
```

## 2 · Build the held-out REAL set (PIILO + TAB; TAB auto-downloads, no creds)
```bash
python -m gold_data.build_real --version real-v1 \
    --kaggle-path data/_raw/kaggle-pii/train.json --n-kaggle 1500 --n-tab 800
python -c "import json; print(json.load(open('data/gold/real-v1/manifest.json'))['counts'])"
```
Want a healthy **pii_absent** count (the rows where false flags happen). If it's ~0, raise `--n-kaggle`.

## 3 · The honest real-data binary-precision baseline  ← the number to review
```bash
python eval_binary.py --gold data/gold/real-v1/test.jsonl --engine hybrid --sweep 0.30:0.90:0.02 \
    --operating-threshold 0.7 --recall-floor 0.97 --out data/eval/real-baseline --binary-mode A
```
Read `data/eval/real-baseline/REPORT.md`: precision @ recall≥97%, the full PR curve, and the per-source
split (kaggle-pii vs tab). Set the real recall floor from this, then iterate.

## 4 · Iterate precision (Phase 3) — see PRECISION_LEVERS.md
Per iteration: `eval_binary` → `loop_iter.py errors` (FP-ranked; prints the top spurious elements) →
pick the lever for the dominant FP element → TDD → re-eval → `loop_iter.py decide` → `eval_gate.py` → tag.
Lever A example — calibrate from the per-entity curve, then **verify the production path on this box**:
```bash
# set aupii.PER_LABEL_THRESHOLDS = {"date of birth": 0.9, ...}
python test_aupii.py && python selfcheck.py        # exercises the guarded twin path with presidio present
```

## 5 · Model-assisted labeling to grow the training pool (the ground-truth loop)
```bash
python spotcheck.py queue --gold data/gold/real-v1/test.jsonl --out reviews/spotcheck-001.jsonl --cap 40
# mark each row's "verdict": model_right | gold_right | ambiguous, then:
python spotcheck.py route --reviews reviews/spotcheck-001.jsonl
# -> data/gold_fixes.jsonl + data/hard_examples.jsonl (confirmed false positives become suppression negatives)
```

## 6 · Stage-2 fine-tune when Stage-1 plateaus
```bash
python train_lora.py --data data/hard_examples.jsonl --out ./model-finetuned --max-chars 2000
python eval_binary.py --gold data/gold/real-v1/test.jsonl --model-dir ./model-finetuned/final \
    --out data/eval/finetuned --recall-floor 0.97
python loop_iter.py decide --before data/eval/real-baseline --after data/eval/finetuned   # binary-precision gate
python selfcheck.py                                                                        # offline load check
# accept -> metrics_io.tag_model (MINOR)
```

All gates are precision-first: an iteration is accepted only if **binary precision holds/improves AND
binary recall stays ≥ floor AND health-tier under-classification = 0%**.
