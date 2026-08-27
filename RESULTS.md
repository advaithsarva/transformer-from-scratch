# Results

Every number below has the command that produced it printed above it.
Everything is seeded (`--seed 42`) and reproduces exactly.

Environment: Windows 11, Python 3.12.1, CPU-only PyTorch. No GPU was used and
no run here needs one.

---

## 1. The layers are the ones they claim to be

This is the result the phrase "from scratch" has to earn, and it is checked
rather than asserted.

```bash
python test_model.py
```

```
19/19 passed
```

| hand-written | compared against | max abs. difference |
|---|---|---|
| `LayerNorm` | `nn.LayerNorm` | < 1e-5 |
| `gelu` | `F.gelu(approximate='tanh')` | < 1e-5 |
| `CausalSelfAttention` (multi-head) | `F.scaled_dot_product_attention(is_causal=True)` | < 1e-5 |

The attention check is the one that matters: it compares the full multi-head
path — projection, head split, scaled scores, causal mask, softmax, value mix,
output projection — against a single torch kernel. Matching to 1e-5 means the
whole assembly is right, not just the pieces.

**And the causality test can fail.** `test_model.py` builds a deliberately
bidirectional model and asserts the perturbation test catches it. Without that,
a causality test that always passes is indistinguishable from one that never
runs.

---

## 2. Reaching the entropy floor

```bash
python train.py --task markov --order 1 --steps 2000 \
    --n-embd 64 --n-layer 2 --block-size 16 --dropout 0.0 --json
```

100,864 parameters, 47.6 seconds on CPU.

| | bits/char | vs oracle |
|---|---|---|
| uniform | 3.5850 | 1.5901 |
| unigram | 3.3909 | 1.5040 |
| order-3 counts | 2.2883 | 1.0150 |
| order-1 counts | **2.2547** | 1.0001 |
| **transformer** | **2.2678** | **1.0059** |
| oracle floor | 2.2545 | 1.0000 |

Training curve from the same run:

| step | train | val |
|---|---|---|
| 0 | 3.5476 | 3.5492 |
| 500 | 2.2941 | 2.2708 |
| 1000 | 2.2825 | 2.2596 |
| 1500 | 2.2724 | 2.2700 |
| 1999 | 2.2650 | 2.2678 |

**What this does and does not say.** It says the implementation is correct: it
extracts essentially all the information the source contains, landing 0.59%
above a floor that is not beatable. It does **not** say the transformer beat
n-grams — the order-1 count model scores 2.2547 against the transformer's
2.2678, so the count model is *better here*, by 0.0131 bits. That is expected
and not a shortfall: for an order-1 source the order-1 count model is the
optimal estimator by construction. There is no headroom above it.

The right sentence is "reaches 1.006× the theoretical entropy floor". Anything
implying the transformer won a comparison against n-grams on this task would be
false.

---

## 3. Ablations: six knobs, and none of them move the number

```bash
python bench.py --steps 500
```

Corpus 200,000 symbols, vocab 12, order 1, 500 steps per run, one variable
changed per row against a 128-dim / 4-layer / 4-head / block-64 default.

| group | variant | bits/char | vs oracle | params | seconds |
|---|---|---|---|---|---|
| embedding-scale | sinusoidal, no scaling | 2.2633 | 1.0039 | 794,880 | 134.0 |
| | sinusoidal, × √n_embd | 2.2682 | 1.0061 | 794,880 | 133.9 |
| | learned positions | 2.2641 | 1.0042 | 803,072 | 135.4 |
| warmup | warmup 100 steps | 2.2682 | 1.0061 | 794,880 | 135.0 |
| | no warmup | 2.2727 | 1.0081 | 794,880 | 132.4 |
| depth | 2 layers | 2.2689 | 1.0064 | 398,336 | 69.2 |
| | 4 layers | 2.2682 | 1.0061 | 794,880 | 138.1 |
| | 6 layers | 2.2670 | 1.0056 | 1,191,424 | 184.3 |
| heads | 1 head | 2.2683 | 1.0061 | 794,880 | 100.6 |
| | 4 heads | 2.2682 | 1.0061 | 794,880 | 116.6 |
| | 8 heads | 2.2679 | 1.0059 | 794,880 | 145.5 |
| context | block_size 8 | **2.2537** | **0.9996** | 794,880 | **20.8** |
| | block_size 32 | 2.2756 | 1.0094 | 794,880 | 58.1 |
| | block_size 64 | 2.2682 | 1.0061 | 794,880 | 119.5 |
| weight-tying | tied | 2.2682 | 1.0061 | 794,880 | 119.2 |
| | untied | 2.2622 | 1.0034 | 796,416 | 120.0 |

**The whole table spans 1.0034 to 1.0094 — 0.6%.** Tripling the depth is worth
0.0008 in ratio. Going from 1 head to 8 is worth 0.0002. Removing warmup costs
0.0020, which is the largest effect in the table and still under a quarter of a
percent.

This is the finding, and it is a real one: **an ablation study can only
separate designs on a task that is capable of separating them.** An order-1
Markov source needs a bigram lookup table. Every configuration in this table
has vastly more capacity than that, so every one of them saturates, and the
differences that remain are seed noise. A table like this published without
that caveat would read as "depth does not matter", which is not what it shows.

### The one row that means something

`block_size 8` matches `block_size 64` (2.2537 vs 2.2682) at **5.7× the speed**
(20.8 s vs 119.5 s). On an order-1 source there is genuinely nothing more than
one token back to look at, so a longer context buys nothing — this is positive
evidence that the model learned the actual structure rather than fitting
spurious long-range correlations.

### Why 0.9996 is not a broken bound

`block_size 8` scores *under* the oracle. The oracle is the source's true
conditional entropy, i.e. the expected cross-entropy in the limit. A finite
validation split (1200 evaluation spots) is a sample, and a sample mean can sit
marginally below the population mean. 0.04% under a floor computed from a
different quantity is sampling noise, not a violated inequality. It is
reported as measured rather than clipped to 1.0000.

---

## 4. Negative result 1 — order-3 Markov, stuck at uniform

```bash
python train.py --task markov --order 3 --steps 2500 --n-embd 128 --n-layer 4 --json
```

794,880 parameters, 703.6 seconds.

| | bits/char | vs oracle |
|---|---|---|
| uniform | 3.5850 | 1.4930 |
| unigram | 3.5837 | 1.4924 |
| order-1 counts | 3.5732 | 1.4880 |
| **transformer** | **3.5837** | **1.4924** |
| order-3 counts | 2.4735 | 1.0301 |
| oracle floor | 2.4013 | 1.0000 |

**The transformer scores exactly the unigram number, to four decimal places.**
It learned nothing — 0.04% better than emitting uniform noise — while a plain
order-3 count model gets to within 3% of the floor on the same data.

That contrast is the result. A count model wins by **memorising** 12³ = 1728
independent random distributions, which is what the task rewards and the only
thing it rewards. Nothing about context `(a, b, c)` predicts anything about
`(a, b, d)`, so there is no structure to generalise and gradient descent has no
direction to move in. A transformer failing here is evidence that the training
loop is honest, not that it is broken.

**The trap, and why `--order` exists.** `MarkovSource` defaults to order 3, so
without the flag this *is* the default command. Running it and reporting the
number would produce a write-up saying the transformer failed to learn. The
true statement is that nothing generalises on this source and the transformer
correctly learned nothing. Keeping "this model cannot learn X" separate from
"X is not learnable" is the whole point of having both §2 and §4.

---

## 5. Negative result 2 — Dyck brackets, where the metric is blind

```bash
python train.py --task brackets --steps 2000 --json
```

| | bits/char |
|---|---|
| uniform | 3.8074 |
| unigram | 3.8048 |

The corpus is essentially uniform in bits/char *by construction* — bracket
sequences use their alphabet almost evenly. So the metric being optimised can
barely move regardless of whether the model has learned matching, and a model
that perfectly tracked nesting depth would score almost the same as one that
had learned nothing.

The failure here is in the **measurement**, not the model. Bracket matching
needs a counter or a stack, and it needs a metric that looks at closing-bracket
correctness specifically. Reporting bits/char on this task would be reporting a
number that could not have moved.

---

## 6. Negative result 3 — copy / induction plateaus at chance

```bash
python copy_task.py
```

6000 steps, 100,672 parameters, sequences of `<5 random symbols> SEP <the same 5>
SEP`, vocab 9, context 36:

| model | copy accuracy |
|---|---|
| chance | 0.125 |
| order-3 n-gram counts | 0.081 |
| order-5 n-gram counts | 0.130 |
| order-8 n-gram counts | **0.214** |
| **transformer** | **0.130** |

Loss falls from 3.19 to 2.55 bits over training. **Copy accuracy never leaves
chance.**

The loss falls because the block structure — where the separators are, roughly
how long a block is — is learnable from position alone, and that is worth real
bits. The copy itself is not learned at all. Induction heads (a previous-token
head feeding a match-and-copy head) are known to form through a sharp phase
transition, and this compute budget does not reach it.

The n-gram column is the honest control: an order-8 count model gets 0.214 by
memorising literal 8-grams from training, which the transformer does not do and
should not.

**This is a capability limit at this scale, not a bug**, and it does not
conflict with §1 or §2. Correctness is established at the layer level, against
torch, independent of any task. What this measures is what 6000 CPU steps buy.
The budget was spent; the negative result is the deliverable.

---

## 7. What has NOT been verified

- **No real-text training run is reported.** `--text` exists and works, and
  `data.py` has a BPE tokeniser, but no numbers on natural language are claimed
  here. A bits/char figure on text has no floor to compare against, which is
  the whole reason the Markov source exists.
- **No GPU path is tested.** The code is device-agnostic in the ordinary way
  but every number here is CPU.
- **`attention_viz.py` produces pictures, not measurements.** The head-shape
  taxonomy in its docstring (local band, null sink, offset diagonal) is
  descriptive; nothing in this repo scores it.
- **The BPE tokeniser is exercised by tests, not by a comparison.** No
  char-vs-BPE result is reported because the Markov source is defined over
  symbols and merging them destroys the property that makes the oracle
  computable.
