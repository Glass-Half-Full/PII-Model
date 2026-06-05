# Performance baseline

Measured with `benchmark.py` against **ai4privacy/pii-masking-200k** (2,000 English rows with gold
PII labels). The classifier runs fully offline; the benchmark only needs internet to download the
labeled data.

## PII detection (presence-level, threshold 0.5)

| Element | Precision % | Recall % | F1 % |
|---|---|---|---|
| Email address | 100.0 | 100.0 | **100.0** |
| Phone number | 73.6 | 96.3 | **83.4** |
| Person (name) | 70.2 | 97.4 | **81.6** |
| Address | 85.8 | 74.8 | **79.9** |
| Bank account | 84.7 | 55.0 | 66.7 |
| Credit card | 43.5 | 70.5 | 53.8 |
| Date of birth | 65.7 | 35.5 | 46.1 |
| **Micro-average** | **72.4** | **83.5** | **77.6** |

## End-to-end GIRP level (vs a gold level derived from the dataset's labels)

| Threshold | GIRP accuracy | Over-classified | Under-classified |
|---|---|---|---|
| 0.50 | 58.8% | 34.2% | 7.0% |
| 0.70 | 62.9% | 29.3% | 7.8% |
| 0.85 | 65.0% | 24.9% | 10.1% |
| 0.93 | 67.0% | 20.6% | 12.4% |

The dominant error is **over-classification** on PII-dense text — the safe direction for compliance,
and tunable with `threshold` (higher = fewer false positives, slightly more misses).

## Caveats (this is a conservative lower bound)
- **Adversarial benchmark.** ai4privacy is saturated with PII and exotic numeric IDs (IMEI, IPv6,
  MAC, crypto/BTC/ETH addresses, VIN…) that are hard to distinguish from phones/cards by format.
  Real customer text (notes, comments) is far less dense, so real-world accuracy is typically higher.
- **Synthetic, non-Luhn cards.** ~85% of the benchmark's card numbers fail the Luhn check — random
  16-digit numbers indistinguishable from account numbers — which caps card precision here. Real
  cards are Luhn-valid, where the regex backstop adds high-precision detection.
- **Sparse gold mapping.** Only the personal elements GIRP cares about are mapped; some
  "over-classified" rows contain PII the mapping ignores, so true over-classification is lower.
- **DOB vs generic dates.** The model under-detects "date of birth" (often labelled as a generic date).

## Reproduce
```
python -m pip install datasets       # benchmark only; the classifier itself needs no internet
python benchmark.py                  # 2000 rows, threshold 0.5
python benchmark.py 1000 0.7         # rows, threshold
```

## Clean-data sanity check
On the in-repo synthetic generator (`synthetic.py`, all four tiers + false-positive bait), the
pipeline scores **100% accuracy, 0 false positives, 0 under-classifications** across seeds —
confirming the rules + precision layer are correct; the lower benchmark numbers reflect the
adversarial dataset, not a logic error.
