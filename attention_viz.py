"""Attention weight heatmaps, one panel per head per layer.

    python attention_viz.py --checkpoint model.pt --prompt abcabcabc
    python attention_viz.py --checkpoint model.pt --out attention.png
    python attention_viz.py --checkpoint model.pt --ascii     # no matplotlib

Reading the pictures
--------------------
Row t is "where position t looked".  Every row is lower triangular and sums to
1, because the mask forbids the future and the softmax normalises what is
left.  Three shapes come up again and again:

  diagonal band   attends to itself and a couple back -- a local n-gram head
  vertical stripe every position attends to one particular token -- usually
                  position 0, which acts as a null sink when the head has
                  nothing to say
  offset diagonal a fixed distance back -- "previous token" heads, the pieces
                  an induction circuit is built from

A first row that is entirely one cell is not a finding: position 0 can only
attend to itself, so that cell is always exactly 1.0.
"""

import argparse
import sys

import torch

from model import Transformer


def load(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = Transformer(
        vocab_size=ckpt["vocab_size"], n_embd=cfg["n_embd"], n_head=cfg["n_head"],
        n_layer=cfg["n_layer"], block_size=cfg["block_size"], dropout=0.0,
        learned_pos=cfg.get("learned_pos", False),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("itos") or {}


def ascii_heatmap(weights, tokens, layer, head):
    """Terminal fallback, so the tool works over ssh and in CI."""
    shades = " .:-=+*#%@"
    print(f"\nlayer {layer} head {head}   (row = querying position, column = attended position)")
    print("      " + "".join(t[:1] for t in tokens))
    for i, row in enumerate(weights):
        cells = "".join(
            shades[min(len(shades) - 1, int(float(v) * len(shades)))] if j <= i else " "
            for j, v in enumerate(row)
        )
        print(f"{tokens[i][:4]:>4}  {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="abcabcabcabc")
    ap.add_argument("--out", default="attention.png")
    ap.add_argument("--ascii", action="store_true", help="print in the terminal instead")
    ap.add_argument("--max-tokens", type=int, default=24, help="crop the prompt so cells stay readable")
    args = ap.parse_args()

    model, itos = load(args.checkpoint)
    stoi = {c: i for i, c in itos.items()}

    missing = set(args.prompt) - stoi.keys()
    if missing:
        print(f"characters not in this model's vocabulary: {sorted(missing)}", file=sys.stderr)
        print(f"vocabulary is: {''.join(sorted(stoi))}", file=sys.stderr)
        return 2

    ids = [stoi[c] for c in args.prompt][: min(args.max_tokens, model.block_size)]
    tokens = [itos[i] for i in ids]

    with torch.no_grad():
        _, _, attentions = model(torch.tensor([ids]), return_attention=True)

    n_layer, n_head = len(attentions), attentions[0].shape[1]

    if args.ascii:
        for layer, w in enumerate(attentions):
            for head in range(n_head):
                ascii_heatmap(w[0, head], tokens, layer, head)
        return 0

    try:
        import matplotlib
        matplotlib.use("Agg")           # no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; use --ascii instead", file=sys.stderr)
        return 2

    fig, axes = plt.subplots(n_layer, n_head, figsize=(2.1 * n_head, 2.1 * n_layer),
                             squeeze=False)
    for layer in range(n_layer):
        for head in range(n_head):
            ax = axes[layer][head]
            # vmin=0, vmax=1 fixed across every panel: with per-panel scaling a
            # head that attends uniformly looks identical to one that attends
            # sharply, which is the opposite of what the picture is for.
            ax.imshow(attentions[layer][0, head], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"L{layer} H{head}", fontsize=8)
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, fontsize=5)
            ax.set_yticklabels(tokens, fontsize=5)

    fig.suptitle(f"attention weights  |  prompt: {''.join(tokens)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({n_layer} layers x {n_head} heads)")

    # A number to go with the picture: entropy says how focused each head is.
    # 0 bits = looks at exactly one token, log2(t) = spreads evenly.
    print(f"\n{'layer':>6}{'head':>6}{'mean entropy (bits)':>22}{'shape':>18}")
    for layer in range(n_layer):
        for head in range(n_head):
            w = attentions[layer][0, head]
            ent = []
            for t in range(1, w.shape[0]):
                row = w[t, : t + 1]
                ent.append(float(-(row * torch.log2(row.clamp_min(1e-12))).sum()))
            mean_ent = sum(ent) / len(ent)
            diag = float(torch.diagonal(w).mean())
            first = float(w[1:, 0].mean())
            shape = ("self" if diag > 0.5 else
                     "sink->pos0" if first > 0.5 else
                     "focused" if mean_ent < 1.0 else "diffuse")
            print(f"{layer:>6}{head:>6}{mean_ent:>22.3f}{shape:>18}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
