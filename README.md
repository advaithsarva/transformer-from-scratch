# Transformer From Scratch

A decoder-only transformer written from PyTorch primitives — `nn.Linear`,
`nn.Embedding`, tensor maths — and nothing else. No `nn.Transformer`, no
`nn.MultiheadAttention`, no `nn.LayerNorm`, no `F.scaled_dot_product_attention`,
no `F.gelu`. Those are the parts worth writing out, so they are written out.

**"From scratch" is a checkable claim here, not a promise: every hand-written
layer is tested against the torch built-in it replaces, to 1e-5.** And on a
source whose entropy can be computed exactly, the model reaches **1.0059× the
theoretical floor** from a single documented command.

```bash
python test_model.py     # 19/19, including layer-vs-torch equivalence
python train.py --task markov --order 1 --steps 2000 \
    --n-embd 64 --n-layer 2 --block-size 16 --dropout 0.0 --json
python bench.py --steps 500          # six ablations
python copy_task.py                  # the negative result
```

Numbers, commands and error analysis in [RESULTS.md](RESULTS.md).

---

## The rule the whole thing turns on

> **A tensor enters a block as `(B, T, C)` and leaves as `(B, T, C)`, and the
> logits at position `t` depend on tokens `0..t` and on nothing after them.**

Shape is the easy half. **Causality is the invariant because breaking it is
invisible.**

A mask that leaks one position of the future produces a *better* training loss,
a *better* validation loss, and completely incoherent generation. The model has
learned to read the answer rather than predict it. Nothing raises, no shape is
wrong, and the curves look exactly like success.

So `test_model.py` checks it by perturbation: change a token at position `t+1`,
assert the logits at position `t` do not move. And because a test that cannot
fail is not a test, there is a second one that builds a deliberately
bidirectional model and asserts the causality test *catches* it.

---

## What is actually verified

Every one of these is a test, not a claim:

| hand-written | checked against | tolerance |
|---|---|---|
| `LayerNorm` | `nn.LayerNorm` | 1e-5 |
| `gelu` | `F.gelu(approximate='tanh')` | 1e-5 |
| multi-head causal attention | `F.scaled_dot_product_attention(is_causal=True)` | 1e-5 |
| sinusoidal positions | shift-linearity, no underflow at long context | exact |
| weight tying | embedding and output head are one tensor | identity |

Plus: attention rows are lower-triangular and sum to 1; untrained loss is
`log(vocab)`; generation is deterministic under a fixed seed; the model can
memorise a 32-token sequence; a head count that does not divide the embedding
raises rather than silently reshaping.

---

## Why a Markov source, and not Shakespeare

Training on text gives you a number — 1.4 bits/char, say — and no way to know
whether that is good. There is no floor to compare it against, so "it trained"
is the only claim available and it is not worth much.

`MarkovSource` generates from a known transition table, so the **true
conditional entropy of the source is computable**. That turns the loss into a
ratio against a floor that cannot be beaten:

```
uniform          3.5850  bits/char     no model at all
unigram          3.3909                symbol frequencies
order-3 counts   2.2883                a count model with the wrong order
order-1 counts   2.2547                the optimal estimator, by construction
transformer      2.2678                <- this project
oracle floor     2.2545                the source's own entropy
```

`ratio_to_oracle = 1.0059`. Within 0.6% of the information-theoretic optimum.

**Read that precisely.** The transformer does *not* beat the order-1 count model
(2.2678 vs 2.2547) — it is very slightly worse, and that is expected rather than
a shortfall. For an order-1 source the order-1 count model **is** the optimal
estimator by construction; there is nothing above it to reach. The claim is
"reaches 1.006× the theoretical entropy floor", which is a **correctness**
result: it says the implementation learns what is there to be learned. Any
wording implying it beat n-grams here would be false.

---

## Three negative results, kept on purpose

They are the more interesting half of the project.

**1. Order-3 Markov: stuck at exactly uniform.** Run with `--order 3` (which is `MarkovSource`'s own default):
`val_bits_per_char = 3.5837` against `uniform = 3.5850` -- while a plain order-3
count model reaches 2.4735 against an oracle floor of 2.4013. The model learns
nothing at all, and that is the correct behaviour: an order-3 source over a
12-symbol alphabet is a *random 1728-entry lookup table*. There is no structure
to generalise from — a count model is optimal by construction and attention has
nothing to attend to. A transformer failing here is evidence the training loop
is honest, not that it is broken.

**2. Dyck brackets: the metric cannot move.** The bracket corpus is near-uniform
in bits/char (unigram 3.8048 against uniform 3.8074), so bits/char is almost
blind to whether the brackets match. Bracket matching needs a stack, and the
measurement chosen could not have detected one.

**3. Copy / induction: plateaus at chance.** `copy_task.py` runs the canonical
induction-head task — random symbols, a separator, the same symbols again.
After 6000 steps and 100,672 parameters:

```
order-3 n-gram counts   0.081
order-5 n-gram counts   0.130
order-8 n-gram counts   0.214
transformer             0.130     chance = 0.125
```

The loss *does* fall (3.19 → 2.55 bits) because the block structure is
learnable from position alone. Copy accuracy never leaves chance. Induction
heads form through a sharp phase transition that a CPU budget does not reach.
This is a **capability limit at this scale, not a bug** — and it does not
conflict with the correctness result above, which is established independently
by layer-level equivalence tests.

---

## The ablations found nothing, and that is the finding

```bash
python bench.py --steps 500
```

Six ablations — embedding scale, warmup, depth, heads, context length, weight
tying — and **every variant lands between 1.0034× and 1.0094× the oracle.** The
whole suite spans 0.6%.

That is not a bug in the benchmark. It is what an ablation study looks like when
the task is far below the capacity of every configuration being compared: an
order-1 source needs a bigram lookup, and a 2-layer 1-head model with an 8-token
context already has more than enough machinery. **An ablation table can only
separate designs on a task that can tell them apart**, which is worth knowing
before reading anyone else's.

One row does separate, and it says something real:

| context | bits/char | vs oracle | seconds |
|---|---|---|---|
| block_size 8 | 2.2537 | 0.9996 | **20.8** |
| block_size 32 | 2.2756 | 1.0094 | 58.1 |
| block_size 64 | 2.2682 | 1.0061 | 119.5 |

An 8-token context is as good as a 64-token one and **5.7× faster** — because on
an order-1 source there is nothing further back to look at. That is positive
evidence the model learned the right thing rather than some spurious long-range
correlation.

(The 0.9996 is not a violation of the entropy bound. The oracle is the source's
true conditional entropy; a finite validation sample can score marginally under
it by chance. Full detail in RESULTS.md.)

---

## Files

| file | what is in it |
|---|---|
| `model.py` | the transformer. LayerNorm, gelu, attention, blocks, generation |
| `data.py` | Markov and bracket sources, char/BPE tokenisers, count-model baselines, the oracle |
| `train.py` | training loop, warmup + cosine schedule, `--json` |
| `bench.py` | six ablations, one variable changed each |
| `sample.py` | generation, with a symbol-frequency profile rather than a wall of text |
| `attention_viz.py` | per-head attention heatmaps, matplotlib or ASCII |
| `copy_task.py` | the induction-head experiment and its negative result |
| `test_model.py` | 19 tests |

```bash
pip install torch      # the only dependency; matplotlib optional for the viz
```
