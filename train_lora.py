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


def build_training_set(hard_path="data/hard_examples.jsonl", augment_n=200, seed=0,
                       out_path="data/training_set.jsonl"):
    """Convert the hard-example pool to gliner2 InputExample dicts and blend in class-balanced
    synthetic-au rows (so fine-tuning does not catastrophically forget broad recall).

    Returns the list of {"text", "entities"} dicts; also writes them to out_path if given.
    Offline and GPU-free, so it is unit-testable.
    """
    examples = []
    for r in load_examples(hard_path):
        examples.append({"text": r["text"], "entities": _spans_to_entities(r["text"], r["spans"])})
    if augment_n:
        from synthetic import generate_synthetic_dataset
        for row in generate_synthetic_dataset(augment_n, seed=seed, return_spans=True):
            ents = collections.defaultdict(list)
            for (label, s, e) in row["spans"]:
                ents[label].append(row["text"][s:e])
            examples.append({"text": row["text"], "entities": dict(ents)})
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true", help="force CPU (impractically slow)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    examples = build_training_set(args.data, augment_n=args.augment, seed=args.seed)
    print(f"training set: {len(examples)} examples "
          f"({sum(1 for e in examples if e['entities'])} with entities, "
          f"{sum(1 for e in examples if not e['entities'])} no-entity negatives)")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "No CUDA detected. LoRA fine-tuning targets the RTX 2050 (4 GB). Re-run on the GPU box, "
            "or pass --allow-cpu to force CPU (impractically slow). The training set has been written "
            "to data/training_set.jsonl regardless.")

    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    torch.manual_seed(args.seed)
    model = GLiNER2.from_pretrained(args.base)
    model.apply_lora(r=args.lora_r, alpha=2 * args.lora_r)     # LoRA keeps the update small/regularized
    try:
        model.gradient_checkpointing_enable()                 # trade compute for memory on 4 GB
    except Exception as e:
        print(f"[train_lora] gradient checkpointing unavailable ({e}); continuing")

    cfg = TrainingConfig(
        output_dir=args.out, num_epochs=args.epochs,
        batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        fp16=torch.cuda.is_available(), seed=args.seed, save_adapter_only=False,
    )
    GLiNER2Trainer(model, cfg).train(train_data=examples)
    model.merge_lora()
    model.save_pretrained(args.out)
    print(f"[train_lora] merged LoRA + saved to {args.out}. Next: "
          f"python evaluate.py --gold data/gold/v1/test.jsonl --model-dir {args.out} --out-version finetuned ; "
          f"python eval_gate.py --gold data/gold/v1/dev.jsonl ; python selfcheck.py")


if __name__ == "__main__":
    main()
