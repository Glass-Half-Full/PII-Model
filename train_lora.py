"""LoRA fine-tuning scaffold for the GIRP PII detector — runs on a GPU (e.g. RTX 2050).

SCAFFOLD: this is the training entry point for the recursive enhancement loop. It is intentionally
not executed in CI (it needs a GPU + labeled data). Verify the trainer call against your installed
gliner2 version before a real run — the gliner2 training API surface can change between releases.

Pipeline (see PRODUCTION.md / AUPII.md):
  1. Build gold data with weak_label.py -> human review -> data/gold.jsonl (entity-span labels).
  2. Mix in augmentation + hard negatives (the false-positive bait: IMEI/IP/crypto/account numbers,
     role/gender nouns, pronouns) so the model learns to NOT fire on them.
  3. LoRA fine-tune with the stability settings below; evaluate each epoch with eval_gate.py.
  4. Merge LoRA, save to a LOCAL dir, record revision + sha256 in models.lock.
  5. Iterate (active learning) until the eval gate passes AND seed-variance is low.

Usage (on a GPU box):
    python train_lora.py --data data/gold.jsonl --out ./model-finetuned --epochs 3 --seed 0
"""
import argparse
import json
import os


def load_examples(path):
    """gold.jsonl: {"text": ..., "spans": [{"label": <GIRP element>, "start": int, "end": int}, ...]}."""
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labeled gold JSONL")
    ap.add_argument("--base", default=".", help="local base model dir (gliner2 weights)")
    ap.add_argument("--out", default="./model-finetuned")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)        # small for 4 GB VRAM
    ap.add_argument("--grad-accum", type=int, default=4)        # keep effective batch reasonable
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from gliner2 import GLiNER2

    if not torch.cuda.is_available():
        print("[train_lora] WARNING: no CUDA detected — fine-tuning on CPU is impractically slow.")
    torch.manual_seed(args.seed)

    examples = load_examples(args.data)
    print(f"loaded {len(examples)} labeled examples; base={args.base}; out={args.out}")

    model = GLiNER2.from_pretrained(args.base)
    # --- Stability levers (apply via gliner2's training API) ---------------------------------
    # * LoRA adapters keep the fine-tune small and regularized:
    #       model.apply_lora(r=args.lora_r)
    # * fp16 + small batch + grad-accum to fit 4 GB VRAM.
    # * Class-balance across the 4 GIRP tiers; add hard negatives so FP bait is trained as negative.
    # * Early-stop on eval_gate metrics; average/ensemble a few seeds to cut variance.
    #
    # The exact trainer call depends on your gliner2 version. Typical shape:
    #
    #   from gliner2 import TrainingConfig, Trainer            # verify names in your install
    #   cfg = TrainingConfig(fp16=True, batch_size=args.batch_size,
    #                        gradient_accumulation_steps=args.grad_accum,
    #                        lora_r=args.lora_r, num_epochs=args.epochs, seed=args.seed)
    #   model.apply_lora(r=args.lora_r)
    #   Trainer(model, cfg).train(examples, eval_fn=run_eval_gate)
    #   model.merge_lora()
    #
    # model.save_pretrained(args.out)
    raise SystemExit(
        "train_lora.py is a scaffold: wire up your gliner2 version's Trainer/TrainingConfig (see the "
        "comment above), then re-run on a GPU. After training, point load_local_model() at --out and "
        "re-run eval_gate.py to confirm gains before shipping."
    )


if __name__ == "__main__":
    main()
