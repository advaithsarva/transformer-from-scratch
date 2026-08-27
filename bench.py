"""Ablations, each one a claim the README makes with a number behind it.

    python bench.py                    # all ablations, ~15 min on CPU
    python bench.py --only embedding-scale
    python bench.py --steps 600        # quicker, noisier
    python bench.py --json

Every run is seeded identically and differs in exactly one thing, so the
column that changes is the one being measured.  Results go in RESULTS.md with
the command that produced them.
"""

import argparse
import json
import math
import time

import torch
import torch.nn as nn

import data as data_mod
from model import Transformer, sinusoidal_positions
from train import lr_at

NATS_TO_BITS = 1.0 / math.log(2)


def train_once(ds, steps, seed=42, lr=3e-3, warmup=100, block_size=64, batch_size=32,
               n_embd=128, n_head=4, n_layer=4, dropout=0.1, learned_pos=False,
               embed_scale=None, tie_weights=True):
    """One training run.  Returns validation bits per token and wall time."""
    torch.manual_seed(seed)
    model = Transformer(ds["vocab_size"], n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                        block_size=block_size, dropout=dropout, learned_pos=learned_pos)

    if not tie_weights:
        model.head = nn.Linear(n_embd, ds["vocab_size"], bias=False)
        nn.init.normal_(model.head.weight, std=0.02)
    if embed_scale is not None:
        with torch.no_grad():
            model.tok_emb.weight.mul_(embed_scale)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(seed)
    started = time.perf_counter()

    for step in range(steps):
        current = lr_at(step, lr, warmup, steps) if warmup else lr
        for group in opt.param_groups:
            group["lr"] = current
        x, y = data_mod.get_batch(ds["train"], block_size, batch_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    elapsed = time.perf_counter() - started
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(40):
            x, y = data_mod.get_batch(ds["val"], block_size, batch_size, gen)
            total += model(x, y)[1].item()
    return (total / 40) * NATS_TO_BITS, elapsed, model.parameter_count()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--symbols", type=int, default=200_000)
    ap.add_argument("--order", type=int, default=1,
                    help="Markov order of the source. 1 is learnable; at 3 the "
                         "source is a random lookup table nothing generalises "
                         "from, so every ablation returns uniform and the "
                         "column being measured says nothing.")
    ap.add_argument("--only", help="run one ablation by name")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ds = data_mod.build_dataset(n_symbols=args.symbols, seed=42, order=args.order)
    V, oracle = ds["vocab_size"], ds["oracle_bits_per_char"]
    t, v = ds["train_symbols"], ds["val_symbols"]

    report = {
        "steps": args.steps,
        "order": args.order,
        "corpus_symbols": args.symbols,
        "vocab_size": V,
        "baselines": {
            "uniform": round(math.log2(V), 4),
            "unigram": round(data_mod.unigram_bits_per_symbol(t, v, V), 4),
            "order1_counts": round(data_mod.count_model_bits_per_symbol(t, v, V, 1), 4),
            "order2_counts": round(data_mod.count_model_bits_per_symbol(t, v, V, 2), 4),
            "order3_counts": round(data_mod.count_model_bits_per_symbol(t, v, V, 3), 4),
            "order5_counts": round(data_mod.count_model_bits_per_symbol(t, v, V, 5), 4),
            "oracle": round(oracle, 4),
        },
        "ablations": {},
    }

    if not args.json:
        print(f"corpus {args.symbols:,} symbols, order {args.order}, vocab {V}, "
              f"{args.steps} steps per run")
        print(f"\n{'baseline':<22}{'bits/char':>11}{'vs oracle':>11}")
        for name, bits in report["baselines"].items():
            print(f"{name:<22}{bits:>11.4f}{bits / oracle:>10.3f}x")
        print()

    # Each entry: one thing changed from the default, everything else identical.
    ablations = {
        "embedding-scale": [
            ("sinusoidal, no scaling", dict(embed_scale=None)),
            ("sinusoidal, x sqrt(n_embd)", dict(embed_scale=math.sqrt(128))),
            ("learned positions", dict(learned_pos=True)),
        ],
        "warmup": [
            ("warmup 100 steps", dict(warmup=100)),
            ("no warmup", dict(warmup=0)),
        ],
        "depth": [
            ("2 layers", dict(n_layer=2)),
            ("4 layers", dict(n_layer=4)),
            ("6 layers", dict(n_layer=6)),
        ],
        "heads": [
            ("1 head", dict(n_head=1)),
            ("4 heads", dict(n_head=4)),
            ("8 heads", dict(n_head=8)),
        ],
        "context": [
            ("block_size 8", dict(block_size=8)),
            ("block_size 32", dict(block_size=32)),
            ("block_size 64", dict(block_size=64)),
        ],
        "weight-tying": [
            ("tied", dict(tie_weights=True)),
            ("untied", dict(tie_weights=False)),
        ],
    }

    # The embedding-scale fix is the default everywhere else, since without it
    # nothing trains at all -- see RESULTS.md section 3.
    default_fix = dict(embed_scale=math.sqrt(128))

    for group, runs in ablations.items():
        if args.only and args.only != group:
            continue
        report["ablations"][group] = []
        if not args.json:
            print(f"--- {group} ---")
            print(f"{'variant':<30}{'bits/char':>11}{'vs oracle':>11}{'params':>11}{'sec':>8}")

        for label, kwargs in runs:
            merged = dict(default_fix)
            merged.update(kwargs)
            bits, secs, params = train_once(ds, args.steps, **merged)
            row = {"variant": label, "bits_per_char": round(bits, 4),
                   "ratio_to_oracle": round(bits / oracle, 4),
                   "params": params, "seconds": round(secs, 1)}
            report["ablations"][group].append(row)
            if not args.json:
                print(f"{label:<30}{bits:>11.4f}{bits / oracle:>10.3f}x{params:>11,}{secs:>8.1f}")
        if not args.json:
            print()

    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
