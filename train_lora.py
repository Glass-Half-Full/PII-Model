"""LoRA fine-tuning for the GIRP PII detector — Stage 2 of the recursive-improvement loop.

Runs on a GPU (e.g. the 4 GB RTX 2050). Stage 2 is triggered when Stage-1 system tuning plateaus
(see LOOP.md) and the hard-example pool (data/hard_examples.jsonl, grown by loop_iter.py) is large
enough. It fine-tunes the zero-shot detector on the accumulated hard examples so the model surfaces
the spans it currently misses at any threshold (detection gaps) and stops firing on bait (the
no-entity / suppression rows), then re-enters the loop on the new weights.

Pipeline:
  1. loop_iter.py accumulate  -> data/hard_examples.jsonl  ({text, spans, reasons, gold_level})
  2. build_training_set       -> gliner2 InputExample dicts ({text, entities:{label:[values]}}),
                                 augmented with class-balanced synthetic-au rows (anti-forgetting).
  3. LoRA fine-tune (this script) on the GPU; merge; save to ./model-finetuned.
  4. evaluate.py + eval_gate.py on the new weights; accept only on a balanced-accuracy gain with no
     health-under regression; metrics_io.tag_model bumps a MINOR version. Re-run selfcheck.py.
  5. Re-enter Stage 1 on ./model-finetuned (load_local_model(model_dir="./model-finetuned")).

Usage (on the GPU box):
    python train_lora.py --data data/hard_examples.jsonl --out ./model-finetuned --epochs 3
"""
import argparse
import collections
import json
import os


def _spans_to_entities(text, spans):
    """gliner2 trains on {label: [entity_text, ...]}; convert char spans to entity strings."""
    ents = collections.defaultdict(list)
    for s in spans:
        ents[s["label"]].append(text[s["start"]:s["end"]])
    return {k: v for k, v in ents.items()}


def load_examples(path):
    """Back-compat: read a gold/hard JSONL of {"text", "spans":[{label,start,end}]}."""
    return [json.loads(l) for l in open(path) if l.strip()]


def _entities_for(row, explicit_negatives):
    """Entity dict for one hard-example row. A confirmed no-entity NEGATIVE (false positive the model
    must learn to suppress) is emitted with the FULL GIRP label vocabulary mapped to empty lists, so
    the trainer sees "these labels were queried and found nothing here" rather than an unlabelled blank."""
    ents = _spans_to_entities(row["text"], row["spans"])
    if explicit_negatives and row.get("negative") and not ents:
        from girp import GIRP_PII_LABELS
        return {lbl: [] for lbl in GIRP_PII_LABELS}
    return ents


def build_training_set(hard_path="data/hard_examples.jsonl", augment_n=200, seed=0,
                       out_path="data/training_set.jsonl", max_chars=None, explicit_negatives=True):
    """Convert the hard-example pool to gliner2 InputExample dicts and blend in class-balanced
    synthetic-au rows (so fine-tuning does not catastrophically forget broad recall).

    ``max_chars`` drops over-long real rows (TAB judgments run to thousands of chars) so a 4 GB card
    doesn't OOM. ``explicit_negatives`` teaches suppression of the high-FP labels on no-entity rows.
    Returns the list of {"text", "entities"} dicts; also writes them to out_path if given. Offline and
    GPU-free, so it is unit-testable.
    """
    examples, skipped_long = [], 0
    for r in load_examples(hard_path):
        if max_chars and len(r["text"]) > max_chars:
            skipped_long += 1
            continue
        examples.append({"text": r["text"], "entities": _entities_for(r, explicit_negatives)})
    n_real = len(examples)
    n_neg = sum(1 for e in examples if not any(e["entities"].values()))
    if augment_n:
        from synthetic import generate_synthetic_dataset
        for row in generate_synthetic_dataset(augment_n, seed=seed, return_spans=True):
            ents = collections.defaultdict(list)
            for (label, s, e) in row["spans"]:
                ents[label].append(row["text"][s:e])
            examples.append({"text": row["text"], "entities": dict(ents)})
    n_pos_aug = sum(1 for e in examples[n_real:] if any(e["entities"].values()))
    if n_neg and n_pos_aug < n_neg:
        print(f"[train_lora] WARNING: {n_neg} suppression negatives but only {n_pos_aug} synthetic "
              f"positives — raise --augment so negatives don't erode recall on the weak labels.")
    if skipped_long:
        print(f"[train_lora] skipped {skipped_long} rows longer than {max_chars} chars (4 GB OOM guard)")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hard_examples.jsonl", help="hard-example JSONL")
    ap.add_argument("--base", default=".", help="local base model dir (gliner2 weights)")
    ap.add_argument("--out", default="./model-finetuned")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)        # small for 4 GB VRAM
    ap.add_argument("--grad-accum", type=int, default=8)        # keep effective batch reasonable
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--augment", type=int, default=200, help="synthetic-au rows to blend in")
    ap.add_argument("--max-chars", type=int, default=2000,
                    help="drop hard rows longer than this (4 GB OOM guard); 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true", help="force CPU (impractically slow)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    examples = build_training_set(args.data, augment_n=args.augment, seed=args.seed,
                                  max_chars=args.max_chars or None)
    n_neg = sum(1 for e in examples if not any(e["entities"].values()))
    print(f"training set: {len(examples)} examples "
          f"({len(examples) - n_neg} with entities, {n_neg} no-entity/suppression negatives)")

    has_accel = torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if not has_accel and not args.allow_cpu:
        raise SystemExit(
            "No CUDA/MPS accelerator detected. LoRA fine-tuning targets the RTX 2050 (4 GB) or Apple "
            "MPS. Re-run on an accelerator, or pass --allow-cpu (impractically slow). The training set "
            "has been written to data/training_set.jsonl regardless.")

    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    torch.manual_seed(args.seed)
    base = GLiNER2.from_pretrained(args.base)
    # apply_lora RETURNS a PeftModel wrapping the base; we must train and merge THAT (not `base`).
    peft_model = base.apply_lora(r=args.lora_r, alpha=2 * args.lora_r)
    try:
        peft_model.gradient_checkpointing_enable()             # trade compute for memory on 4 GB
    except Exception as e:
        print(f"[train_lora] gradient checkpointing unavailable ({e}); continuing")

    cfg = TrainingConfig(
        output_dir=os.path.join(args.out, "_trainer"), num_epochs=args.epochs,
        batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        fp16=torch.cuda.is_available(), seed=args.seed, save_adapter_only=True,
        num_workers=0, pin_memory=False,          # avoid spawn issues; MPS has no pinned memory
        eval_strategy="no", save_best=False, logging_steps=10,
    )
    GLiNER2Trainer(peft_model, cfg).train(train_data=examples)

    # Merge the LoRA deltas into the base weights and save a plain, loadable checkpoint.
    merged = peft_model.merge_and_unload()
    ckpt = os.path.join(args.out, "final")
    merged.save_pretrained(ckpt, merge_lora=False, save_adapter_only=False)
    import shutil
    for name in ("special_tokens_map.json", "added_tokens.json"):   # belt-and-braces for the loader
        src, dst = os.path.join(args.base, name), os.path.join(ckpt, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"[train_lora] merged fine-tuned checkpoint at {ckpt}. Next:\n"
          f"  python evaluate.py --gold data/gold/v1/test.jsonl --model-dir {ckpt} --out-version finetuned/after\n"
          f"  python loop_iter.py decide --before data/eval/iter-002/after --after data/eval/finetuned/after\n"
          f"  python selfcheck.py   # confirm the fine-tuned weights still run fully offline")


if __name__ == "__main__":
    main()
