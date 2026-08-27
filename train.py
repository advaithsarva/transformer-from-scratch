"""Train the transformer and measure it against baselines that can beat it.

    python train.py                          # bracket task, ~4 min on CPU
    python train.py --task markov            # the oracle / leak-detector task
    python train.py --text mybook.txt        # real text; no baseline exists
    python train.py --steps 4000 --n-layer 6
    python train.py --json                   # machine readable summary

Two numbers, depending on the task:

**brackets (default)** -- accuracy at choosing which of `)]}` comes next,
broken down by how far back the matching opener is. Chance is 0.333. An
order-k count model can only be right when the distance is at most k, so the
long-distance buckets are where attention has to earn its place.

**markov** -- bits per character against a computable oracle. Comparable
across models because the floor is known. A causal model scoring *below* the
oracle has leaked the future; train.py says so loudly rather than celebrating.

Loss is always reported in bits, never nats, because bits compare to the
oracle and to the count models. "val loss 1.73" on its own says nothing.
"""

import argparse
import json
import math
import time
from collections import defaultdict

import torch

import data as data_mod
from model import Transformer

NATS_TO_BITS = 1.0 / math.log(2)


def lr_at(step, base_lr, warmup, total):
    """Linear warmup, then cosine decay to a tenth of the peak.

    Warmup exists because the first steps of a randomly initialised
    transformer produce large gradients through the attention softmax, and a
    full-size step there can push the model somewhere it does not recover
    from. Pre-norm blocks make this far less severe than it was for the
    original post-norm design, so it is insurance rather than load bearing --
    `--warmup 0` measures training without it.
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def evaluate(model, split, block_size, batch_size, batches, generator, device):
    """Average loss over random windows, in bits per token."""
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = data_mod.get_batch(split, block_size, batch_size, generator, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return (total / batches) * NATS_TO_BITS


@torch.no_grad()
def closer_accuracy_model(model, tokenizer, val_text, block_size, device="cpu",
                          max_spots=1500):
    """Score the model only where a closing bracket is next.

    Scored the same way as the n-gram baseline: take the preceding characters
    as context, predict the next one, count a hit only for the exact right
    closer. The model gets `block_size` characters of context and the count
    model gets k -- which is the whole comparison, and why the context each
    one is given has to be stated rather than assumed.

    Restricted to the three closing brackets? **No.** The model predicts over
    the full vocabulary and must land on the right closer unaided. Masking to
    the three closers would hand it a third of the problem and inflate every
    number here.
    """
    model.eval()
    spots = data_mod.closer_positions(val_text)[:max_spots]
    if not spots:
        return None

    correct = 0
    by_distance = defaultdict(lambda: [0, 0])

    for spot in spots:
        i = spot["index"]
        context = val_text[max(0, i - block_size):i]
        if not context:
            continue
        ids = torch.tensor([tokenizer.encode(context)], device=device)
        logits, _ = model(ids)
        predicted = tokenizer.decode([int(logits[0, -1].argmax())])

        hit = predicted == spot["char"]
        correct += hit
        bucket = data_mod.distance_bucket(spot["distance"])
        by_distance[bucket][0] += hit
        by_distance[bucket][1] += 1

    model.train()
    return {
        "accuracy": correct / len(spots),
        "n": len(spots),
        "by_distance": {k: {"correct": v[0], "total": v[1],
                            "accuracy": v[0] / v[1] if v[1] else 0.0}
                        for k, v in sorted(by_distance.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["markov", "brackets"], default="markov")
    ap.add_argument("--text", help="train on a text file instead of a generated corpus")
    ap.add_argument("--symbols", type=int, default=200_000)
    ap.add_argument("--order", type=int, default=1,
                    help="Markov source order. 1 is learnable and the model reaches "
                         "the oracle floor on it; 3 is a random 1728-entry lookup "
                         "table that an n-gram count model is optimal for by "
                         "construction and that this model cannot beat uniform on. "
                         "See RESULTS.md.")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--bpe-merges", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100, help="0 disables warmup")
    ap.add_argument("--learned-pos", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--eval-spots", type=int, default=1200)
    ap.add_argument("--save", help="path to write the checkpoint")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = data_mod.load_text(args.text) if args.text else None
    ds = data_mod.build_dataset(
        text=text, n_symbols=args.symbols, tokenizer=args.tokenizer,
        bpe_merges=args.bpe_merges, seed=args.seed, task=args.task,
        **({"order": args.order} if args.task == "markov" else {}),
    )

    model = Transformer(
        vocab_size=ds["vocab_size"], n_embd=args.n_embd, n_head=args.n_head,
        n_layer=args.n_layer, block_size=args.block_size, dropout=args.dropout,
        learned_pos=args.learned_pos,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(args.seed)

    if not args.json:
        print(f"task {ds['task']} | device {device} | vocab {ds['vocab_size']} | "
              f"{model.parameter_count():,} parameters | "
              f"{len(ds['train']):,} train / {len(ds['val']):,} val tokens")
        if "oracle_bits_per_char" in ds:
            print(f"oracle floor: {ds['oracle_bits_per_char']:.4f} bits/char "
                  f"(nothing honest scores below this)")
        print(f"\n{'step':>6}{'lr':>10}{'train':>10}{'val':>10}")

    history = []
    started = time.perf_counter()

    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps) if args.warmup else args.lr
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = data_mod.get_batch(ds["train"], args.block_size, args.batch_size, gen, device)
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip before stepping. Attention produces occasional very large
        # gradients; one unclipped spike shows up only as a curve that jumps
        # and never recovers.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            train_bits = evaluate(model, ds["train"], args.block_size,
                                  args.batch_size, args.eval_batches, gen, device)
            val_bits = evaluate(model, ds["val"], args.block_size,
                                args.batch_size, args.eval_batches, gen, device)
            history.append({"step": step, "lr": round(lr, 6),
                            "train_bits": round(train_bits, 4),
                            "val_bits": round(val_bits, 4)})
            if not args.json:
                print(f"{step:>6}{lr:>10.5f}{train_bits:>10.4f}{val_bits:>10.4f}")

    elapsed = time.perf_counter() - started

    summary = {
        "task": ds["task"],
        "params": model.parameter_count(),
        "steps": args.steps,
        "seconds": round(elapsed, 1),
        "tokenizer": args.tokenizer,
        "chars_per_token": round(ds["chars_per_token"], 3),
        "final_val_bits": history[-1]["val_bits"],
        "history": history,
        "config": {k: v for k, v in vars(args).items() if k != "json"},
    }

    if ds["task"] == "brackets":
        model_score = closer_accuracy_model(
            model, ds["tokenizer"], ds["val_text"], args.block_size, device, args.eval_spots)
        baselines = {
            f"order{k}_counts": data_mod.closer_accuracy_ngram(
                ds["train_text"], ds["val_text"][:len(ds["val_text"])], k)
            for k in (3, 5, 8)
        }
        summary["closer_accuracy"] = {"transformer": model_score, **baselines}

        if not args.json:
            buckets = ["1-4", "5-8", "9-16", "17+"]
            print(f"\nClosing-bracket accuracy (chance = 0.333, {model_score['n']} positions)")
            print(f"{'model':<22}{'overall':>9}" + "".join(f"{b:>9}" for b in buckets))
            for name, r in [("transformer", model_score)] + list(baselines.items()):
                cells = "".join(
                    f"{r['by_distance'][b]['accuracy']:>9.3f}" if b in r["by_distance"]
                    else f"{'-':>9}" for b in buckets)
                print(f"{name:<22}{r['accuracy']:>9.3f}{cells}")
            best_ngram = max(b["accuracy"] for b in baselines.values())
            print(f"\ntransformer {model_score['accuracy']:.3f} vs best n-gram "
                  f"{best_ngram:.3f}  ({model_score['accuracy'] - best_ngram:+.3f})")
            far = model_score["by_distance"].get("17+")
            if far:
                ng_far = max(b["by_distance"].get("17+", {"accuracy": 0})["accuracy"]
                             for b in baselines.values())
                print(f"at distance 17+: transformer {far['accuracy']:.3f} vs "
                      f"n-gram {ng_far:.3f} -- this is the bucket n-grams cannot reach")

    if ds["task"] == "markov":
        v, t, V = ds["val_symbols"], ds["train_symbols"], ds["vocab_size"]
        summary["baselines"] = {
            "uniform": round(math.log2(V), 4),
            "unigram": round(data_mod.unigram_bits_per_symbol(t, v, V), 4),
            "order1_counts": round(data_mod.count_model_bits_per_symbol(t, v, V, 1), 4),
            "order3_counts": round(data_mod.count_model_bits_per_symbol(t, v, V, 3), 4),
            "oracle": round(ds["oracle_bits_per_char"], 4),
        }
        summary["val_bits_per_char"] = round(
            history[-1]["val_bits"] / ds["chars_per_token"], 4)
        summary["ratio_to_oracle"] = round(
            summary["val_bits_per_char"] / ds["oracle_bits_per_char"], 4)

        if not args.json:
            print(f"\n{'model':<20}{'bits/char':>12}{'vs oracle':>12}")
            rows = list(summary["baselines"].items())
            rows.insert(len(rows) - 1, ("transformer", summary["val_bits_per_char"]))
            for name, bits in rows:
                mark = "  <-- floor" if name == "oracle" else ""
                print(f"{name:<20}{bits:>12.4f}"
                      f"{bits / summary['baselines']['oracle']:>11.3f}x{mark}")

        if summary["val_bits_per_char"] < ds["oracle_bits_per_char"] - 1e-6:
            print("\nWARNING: scored BELOW the theoretical floor. That is impossible "
                  "for an honest causal model -- check the mask for a future leak.")

    if args.save:
        torch.save({"model": model.state_dict(), "config": vars(args),
                    "vocab_size": ds["vocab_size"],
                    "itos": getattr(ds["tokenizer"], "itos", None)}, args.save)
        summary["checkpoint"] = args.save

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\ntrained in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
