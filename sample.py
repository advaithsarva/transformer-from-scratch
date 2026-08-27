"""Generate text from a trained checkpoint.

    python sample.py --checkpoint model.pt
    python sample.py --checkpoint model.pt --prompt "the " --tokens 400
    python sample.py --checkpoint model.pt --temperature 0.6 --top-k 5
    python sample.py --checkpoint model.pt --greedy

On the default Markov corpus the output is not meant to be readable -- the
source has no words, only order-3 statistics. What to look for instead is the
**symbol frequency profile**, printed underneath: a model that has learned the
source produces a distribution close to the training data's. A model that has
collapsed produces one or two symbols forever, which the printed profile makes
obvious and a wall of text does not.
"""

import argparse
import sys
from collections import Counter

import torch

from model import Transformer


def load(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = Transformer(
        vocab_size=ckpt["vocab_size"], n_embd=cfg["n_embd"], n_head=cfg["n_head"],
        n_layer=cfg["n_layer"], block_size=cfg["block_size"], dropout=0.0,
        learned_pos=cfg.get("learned_pos", False),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("itos") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--greedy", action="store_true", help="always take the most likely token")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, itos = load(args.checkpoint)
    stoi = {c: i for i, c in itos.items()}

    if args.prompt:
        missing = set(args.prompt) - stoi.keys()
        if missing:
            print(f"characters not in this model's vocabulary: {sorted(missing)}", file=sys.stderr)
            print(f"vocabulary is: {''.join(sorted(stoi))}", file=sys.stderr)
            return 2
        idx = torch.tensor([[stoi[c] for c in args.prompt]])
    else:
        idx = torch.zeros((1, 1), dtype=torch.long)

    # Greedy is top_k=1, so there is one sampling path rather than two.
    top_k = 1 if args.greedy else args.top_k
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=top_k)[0].tolist()
    text = "".join(itos[i] for i in out)

    print(text)

    counts = Counter(text)
    total = len(text)
    print(f"\n{'symbol':>8}{'count':>8}{'share':>9}")
    for sym, n in counts.most_common():
        print(f"{sym!r:>8}{n:>8}{n / total:>8.1%}")

    if len(counts) == 1:
        print("\nWARNING: one symbol only. The model has collapsed -- it is not "
              "sampling a distribution, it is stuck.")
    elif len(counts) < model.head.out_features / 3:
        print(f"\nNote: only {len(counts)} of {model.head.out_features} symbols appeared. "
              "Low temperature or a small top-k will do this; so will an undertrained model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
