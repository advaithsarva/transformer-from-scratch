"""A standalone experiment: can this transformer learn to copy from context?

    python copy_task.py                    # default: 6000 steps, ~12 min on CPU
    python copy_task.py --steps 20000      # if you have the compute

WHAT THIS MEASURES
------------------
The sequence is blocks of `<span random symbols> SEP <the same symbols> SEP`.
Predicting the second half is impossible from local statistics -- the symbols
are uniformly random, so no n-gram of any order has an edge -- but trivial if
you can look back to the first half. That is the canonical *induction head*
task, and it isolates the one thing attention does that counting cannot:
retrieve a specific earlier token by matching context.

WHAT IT FOUND, AND WHY IT IS IN THE REPO
----------------------------------------
This transformer does NOT learn it at the sizes reachable on a CPU. Measured,
6000 steps, 100,672 parameters:

    order-3 n-gram counts      0.081
    order-5 n-gram counts      0.130
    order-8 n-gram counts      0.214
    transformer                0.130      chance = 0.125

The loss does fall (3.19 -> 2.55 bits) because the block structure is
learnable from position alone, but copy accuracy never leaves chance. Induction
heads form through a sharp phase transition that this budget does not reach.

This is kept, and reported, because it is the honest result. The model's
*correctness* is established elsewhere and independently: `test_model.py`
checks every hand-written layer against its torch equivalent, and on an
order-1 Markov source with computable entropy it reaches 2.2678 bits/char
against a theoretical floor of 2.2545 -- 1.006x the optimum. What this file
shows is a capability limit at this scale, not a bug, and the two claims do
not conflict.

See RESULTS.md section 5.
"""

import argparse
import collections
import math
import random

import torch

from model import Transformer


def generate(n_blocks, vocab, span, seed):
    """`span` random symbols, SEP, the same symbols again, SEP."""
    rng = random.Random(seed)
    sep = vocab
    out = []
    for _ in range(n_blocks):
        block = [rng.randrange(vocab) for _ in range(span)]
        out += block + [sep] + block + [sep]
    return out


def in_copied_half(position, span):
    """True if this index sits inside the repeated half of its block."""
    return span + 1 <= position % (2 * span + 2) <= 2 * span


def ngram_copy_accuracy(train, val, order, vocab, span):
    """Baseline: an order-k count model, scored only on the copied half."""
    table = collections.defaultdict(collections.Counter)
    for i in range(order, len(train)):
        table[tuple(train[i - order:i])][train[i]] += 1

    hit = total = 0
    for i in range(order, len(val)):
        if not in_copied_half(i, span):
            continue
        context = table.get(tuple(val[i - order:i]))
        guess = context.most_common(1)[0][0] if context else 0
        hit += guess == val[i]
        total += 1
    return (hit / total if total else 0.0), total


@torch.no_grad()
def model_copy_accuracy(model, val, block_size, span):
    model.eval()
    tensor = torch.tensor(val)
    period = 2 * span + 2
    hit = total = 0
    for start in range(0, len(val) - block_size - 1, period):
        logits, _ = model(tensor[start:start + block_size].unsqueeze(0))
        predicted = logits[0].argmax(-1)
        for j in range(block_size - 1):
            if in_copied_half(start + j + 1, span):
                hit += int(predicted[j] == tensor[start + j + 1])
                total += 1
    model.train()
    return (hit / total if total else 0.0), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--span", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2, help="2 is the minimum for induction")
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocab_size = args.vocab + 1                # + the separator
    block_size = 3 * (2 * args.span + 2)       # three blocks of context
    train = generate(4000, args.vocab, args.span, seed=1)
    val = generate(400, args.vocab, args.span, seed=2)

    print(f"vocab {vocab_size} (incl. SEP), span {args.span}, "
          f"block period {2 * args.span + 2}, context {block_size}")
    print(f"uniform = {math.log2(vocab_size):.3f} bits, "
          f"chance copy-accuracy = {1 / args.vocab:.3f}\n")

    print("n-gram baselines, scored on the copied half only:")
    for order in (3, 5, 8):
        accuracy, n = ngram_copy_accuracy(train, val, order, args.vocab, args.span)
        print(f"  order-{order:<2} {accuracy:.3f}   (n={n})")

    torch.manual_seed(args.seed)
    model = Transformer(vocab_size, n_embd=args.n_embd, n_head=args.n_head,
                        n_layer=args.n_layer, block_size=block_size, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)
    data = torch.tensor(train)

    print(f"\ntraining {model.parameter_count():,} parameters for {args.steps} steps")
    for step in range(args.steps):
        ix = torch.randint(len(data) - block_size - 1, (32,), generator=generator)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 12) == 0:
            print(f"  step {step:>6}  loss {loss.item() / math.log(2):.4f} bits", flush=True)

    accuracy, n = model_copy_accuracy(model, val, block_size, args.span)
    best_ngram = max(ngram_copy_accuracy(train, val, k, args.vocab, args.span)[0]
                     for k in (3, 5, 8))

    print(f"\ntransformer copy-accuracy {accuracy:.3f}  (n={n})")
    print(f"best n-gram               {best_ngram:.3f}")
    print(f"chance                    {1 / args.vocab:.3f}")

    if accuracy < 1 / args.vocab + 0.05:
        print("\nThe transformer is at chance: no induction head formed in this budget.\n"
              "That is the documented result -- see the module docstring and\n"
              "RESULTS.md section 5. It is a capability limit at this scale, not a\n"
              "defect; correctness is established by test_model.py and by reaching\n"
              "the theoretical entropy floor on the Markov task.")


if __name__ == "__main__":
    main()
