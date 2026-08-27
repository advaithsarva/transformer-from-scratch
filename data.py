"""Corpora, tokenizers, and the reason this project can report an honest number.

The measurement problem
-----------------------
Train a language model on Shakespeare, get a validation loss of 1.48, and you
have learned almost nothing.  Is 1.48 good?  Compared to what?  The best
achievable loss on natural text is unknown, so the number floats free and any
architecture change can be spun as an improvement.

Two generated corpora answer it, and they answer different halves.

**MarkovSource** -- an order-k Markov chain with a known transition table. The
true conditional distribution is known, so the best achievable cross-entropy
can be computed exactly: the *oracle*. That turns "loss went down" into "the
model reached 1.00x the theoretical floor", a claim that can be checked and can
fail. It is also a leak detector -- a causal model that scores *below* the
oracle has read its own answers.

**BracketSource** -- nested brackets with filler, the default. It has no
closed-form oracle, but it has something better for benchmarking: a subset of
positions where n-gram counting *provably* cannot win. Predicting which of
three closing brackets comes next requires matching an opener that may be
arbitrarily far back, and a model that only sees the last k characters is
guessing once the group is longer than k.

Why two, rather than picking one: an order-k Markov source is a random lookup
table, so an order-k count model is its optimal estimator *by construction* and
a transformer has nothing to generalise from. Measured, this transformer
reaches the oracle on an order-1 source and never beats uniform on an order-3
one -- correct behaviour on a task that rewards memorisation over structure.
Benchmarking the architecture on it would be measuring the wrong thing. The
Markov source proves the implementation is right; the bracket source shows what
attention is for.

``--text yourfile.txt`` trains on real text, and then neither an oracle nor a
bracket metric exists, and RESULTS.md says so.
"""

import json
import math
from collections import Counter, defaultdict

import torch


# --------------------------------------------------------------------------
# a source whose entropy we know
# --------------------------------------------------------------------------

class MarkovSource:
    """An order-k Markov chain over a small alphabet, with a fixed seed.

    ``concentration`` controls how predictable it is.  The transition rows are
    drawn from a Dirichlet: values below 1 give peaky rows (predictable, low
    entropy), values above 1 give flat rows (near uniform, high entropy).  The
    default 0.35 produces text with real structure that is still not trivial.
    """

    def __init__(self, vocab_size=12, order=3, seed=42, concentration=0.35):
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if order < 1:
            raise ValueError("order must be at least 1")

        self.vocab_size = vocab_size
        self.order = order
        self.alphabet = [chr(ord("a") + i) for i in range(vocab_size)]

        g = torch.Generator().manual_seed(seed)
        n_states = vocab_size ** order
        # Dirichlet via normalised Gamma draws -- torch has no Dirichlet
        # sampler on a plain generator, and Gamma(alpha,1)/sum is the standard
        # construction.
        gamma = torch._standard_gamma(
            torch.full((n_states, vocab_size), concentration), generator=g
        )
        self.table = gamma / gamma.sum(dim=1, keepdim=True)

    def _state(self, context):
        """Encode the last `order` symbols as a base-`vocab_size` integer."""
        s = 0
        for symbol in context[-self.order:]:
            s = s * self.vocab_size + symbol
        return s

    def generate(self, n, seed=0):
        """Produce `n` symbols as a list of ints."""
        g = torch.Generator().manual_seed(seed)
        out = torch.randint(0, self.vocab_size, (self.order,), generator=g).tolist()
        for _ in range(n):
            probs = self.table[self._state(out)]
            out.append(int(torch.multinomial(probs, 1, generator=g).item()))
        return out[self.order:]

    def oracle_bits_per_symbol(self, sequence):
        """The best cross-entropy any model can achieve on this sequence.

        Averages -log2 P(next | true state) using the *real* table.  This is a
        floor in expectation: no model, however large, beats the process that
        generated the data.  A model scoring below it has seen its answers --
        which is exactly what a leaking causal mask does, and why this number
        is also a leak detector.
        """
        total, count = 0.0, 0
        for i in range(self.order, len(sequence)):
            probs = self.table[self._state(sequence[i - self.order:i])]
            total += -math.log2(max(float(probs[sequence[i]]), 1e-12))
            count += 1
        return total / count


# --------------------------------------------------------------------------
# tokenizers
# --------------------------------------------------------------------------

class CharTokenizer:
    """One token per character.  No vocabulary decisions, no merge table, and
    every string round-trips exactly -- which makes it the right default for
    checking that the *model* works."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        missing = set(text) - self.stoi.keys()
        if missing:
            raise ValueError(
                f"characters not in the vocabulary: {sorted(missing)[:5]}. "
                "Build the tokenizer on the full corpus before encoding."
            )
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class BPETokenizer:
    """Byte pair encoding, trained by repeatedly merging the commonest pair.

    The algorithm is three lines of idea: count adjacent pairs, merge the most
    frequent into a new symbol, repeat.  Do it 500 times and common sequences
    become single tokens.

    Why bother, when char-level already works: a token covering 2.5 characters
    means `block_size` tokens of context reaches 2.5x further into the text for
    the same compute.  Attention is quadratic in sequence length, so that is a
    6x saving, not a 2.5x one.  RESULTS.md has the measured compression.

    ponytail: the merge loop is O(merges x corpus) because it re-counts every
    pair each round.  Fine for the corpora here (seconds).  For megabytes, keep
    an index from pair to positions and update incrementally.
    """

    def __init__(self, text, n_merges=256):
        self.base = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.base)}
        self.merges = {}

        ids = [self.stoi[c] for c in text]
        next_id = len(self.base)
        vocab = {i: c for c, i in self.stoi.items()}

        for _ in range(n_merges):
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:
                break                       # nothing repeats; more merges are noise
            self.merges[best] = next_id
            vocab[next_id] = vocab[best[0]] + vocab[best[1]]
            ids = self._apply(ids, best, next_id)
            next_id += 1

        self.itos = vocab

    @staticmethod
    def _apply(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        missing = set(text) - self.stoi.keys()
        if missing:
            raise ValueError(f"characters not in the vocabulary: {sorted(missing)[:5]}")
        ids = [self.stoi[c] for c in text]
        # Merges must be replayed in the order they were learned: a later merge
        # can depend on the symbol an earlier one created.
        for pair, new_id in self.merges.items():
            ids = self._apply(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# --------------------------------------------------------------------------
# baselines the transformer has to beat
# --------------------------------------------------------------------------

def count_model_bits_per_symbol(train_ids, val_ids, vocab_size, order, alpha=1.0):
    """An n-gram count model with add-alpha smoothing, scored on the val split.

    This is the boring version the transformer has to beat.  At order 3 it is
    the *same model class* as the generator, so with enough data it converges
    to the oracle -- which makes it the strongest honest baseline available and
    a much harder target than "uniform" or "unigram".
    """
    counts = defaultdict(Counter)
    for i in range(order, len(train_ids)):
        counts[tuple(train_ids[i - order:i])][train_ids[i]] += 1

    total, n = 0.0, 0
    denom_default = alpha * vocab_size
    for i in range(order, len(val_ids)):
        ctx = counts.get(tuple(val_ids[i - order:i]))
        if ctx is None:
            p = 1.0 / vocab_size                      # unseen context: uniform
        else:
            p = (ctx[val_ids[i]] + alpha) / (sum(ctx.values()) + denom_default)
        total += -math.log2(p)
        n += 1
    return total / n


def unigram_bits_per_symbol(train_ids, val_ids, vocab_size, alpha=1.0):
    """Ignore context entirely; just use character frequency."""
    counts = Counter(train_ids)
    total_count = len(train_ids) + alpha * vocab_size
    total = sum(
        -math.log2((counts[s] + alpha) / total_count) for s in val_ids
    )
    return total / len(val_ids)


# --------------------------------------------------------------------------
# assembling a dataset
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# a task that needs attention, not counting
# --------------------------------------------------------------------------

class BracketSource:
    """Nested brackets with filler: a language n-gram models provably cannot
    learn, and attention can.

    Why this exists
    ---------------
    MarkovSource above has a computable oracle, which is exactly what is needed
    to prove the implementation is correct -- and on an order-1 source this
    transformer reaches it (see RESULTS.md). But as a *benchmark* it is the
    wrong task, and measuring it honestly says so:

        an order-k Markov source is a random lookup table. An order-k count
        model is its optimal estimator by construction, and a transformer has
        nothing to generalise from -- only 12^3 arbitrary rows to memorise.

    So the default corpus is this instead. A sequence like

        ( a b [ c d ] e ) { f }

    requires, at every closing bracket, remembering which bracket opened the
    current group -- arbitrarily far back, across arbitrary filler. A count
    model of order k sees only the last k characters, so once the group is
    longer than k it is guessing between three closers. Attention can look
    straight back at the opener.

    That makes the comparison sharp and falsifiable: `closer_accuracy_ngram`
    scores *only* the positions where a closing bracket is next -- the subset
    of the task where n-grams must fail and attention need not.
    """

    OPEN = "([{"
    CLOSE = ")]}"
    FILLER = "abcdefgh"

    def __init__(self, seed=42, max_depth=4, filler_range=(0, 3), open_prob=0.42):
        self.seed = seed
        self.max_depth = max_depth
        self.filler_range = filler_range
        self.open_prob = open_prob
        self.alphabet = sorted(set(self.OPEN + self.CLOSE + self.FILLER + " "))

    def generate(self, n_chars, seed=None):
        """Emit until we have n_chars. Always closes what it opens."""
        import random
        rng = random.Random(self.seed if seed is None else seed)
        out, stack = [], []

        while len(out) < n_chars:
            for _ in range(rng.randint(*self.filler_range)):
                out.append(rng.choice(self.FILLER))

            if stack and (len(stack) >= self.max_depth or rng.random() > self.open_prob):
                out.append(self.CLOSE[stack.pop()])       # close the innermost
            elif len(stack) < self.max_depth:
                i = rng.randrange(len(self.OPEN))
                stack.append(i)
                out.append(self.OPEN[i])
            else:
                out.append(" ")

        while stack:                       # close anything still open
            out.append(self.CLOSE[stack.pop()])
        return "".join(out[:n_chars])


def closer_positions(text):
    """Indices where the character is a closing bracket, with the distance back
    to its matching opener.

    Distance is the measurement that matters: a count model of order k can only
    be right when the distance is at most k. Bucketing by distance shows
    exactly where n-grams fall off, and whether attention does.
    """
    stack, out = [], []
    for i, ch in enumerate(text):
        if ch in BracketSource.OPEN:
            stack.append(i)
        elif ch in BracketSource.CLOSE:
            opener = stack.pop() if stack else None
            if opener is not None and i > 0:
                out.append({"index": i, "char": ch, "distance": i - opener})
    return out


def distance_bucket(d):
    if d <= 4:
        return "1-4"
    if d <= 8:
        return "5-8"
    if d <= 16:
        return "9-16"
    return "17+"


def closer_accuracy_ngram(train_text, val_text, order):
    """Baseline: an order-k count model, scored ONLY where a closer is next.

    Backs off to shorter contexts when the full one is unseen, which is the
    charitable version -- an unsmoothed model scores lower still.
    """
    tables = [defaultdict(Counter) for _ in range(order + 1)]
    for k in range(order + 1):
        for i in range(k, len(train_text)):
            tables[k][train_text[i - k:i]][train_text[i]] += 1

    correct = 0
    by_distance = defaultdict(lambda: [0, 0])
    targets = closer_positions(val_text)

    for spot in targets:
        i = spot["index"]
        guess = None
        for k in range(order, -1, -1):              # longest context first
            ctx = val_text[max(0, i - k):i]
            if len(ctx) == k and tables[k][ctx]:
                guess = tables[k][ctx].most_common(1)[0][0]
                break
        hit = guess == spot["char"]
        correct += hit
        bucket = distance_bucket(spot["distance"])
        by_distance[bucket][0] += hit
        by_distance[bucket][1] += 1

    return {
        "accuracy": correct / len(targets) if targets else 0.0,
        "n": len(targets),
        "by_distance": {k: {"correct": v[0], "total": v[1],
                            "accuracy": v[0] / v[1] if v[1] else 0.0}
                        for k, v in sorted(by_distance.items())},
    }


def build_dataset(text=None, n_symbols=200_000, val_fraction=0.1,
                  tokenizer="char", bpe_merges=256, seed=42, task="markov",
                  **source_kwargs):
    """Return everything training and evaluation need.

    task="brackets"  the default. Adds `train_text`/`val_text` so the
                     closing-bracket metric can be computed. No oracle.
    task="markov"    adds `oracle_bits_per_char` and the symbol sequences the
                     count-model baselines need.
    text=...         real text. Neither extra is available, and nothing in
                     this file invents one.
    """
    source = None
    if text is not None:
        task = "text"
    elif task == "brackets":
        source = BracketSource(seed=seed, **source_kwargs)
        text = source.generate(n_symbols, seed=seed + 1)
    elif task == "markov":
        source = MarkovSource(seed=seed, **source_kwargs)
        symbols = source.generate(n_symbols, seed=seed + 1)
        text = "".join(source.alphabet[s] for s in symbols)
    else:
        raise ValueError(f"unknown task {task!r}; use 'brackets', 'markov', or pass text=")

    split = int(len(text) * (1 - val_fraction))
    if split < 100 or len(text) - split < 100:
        raise ValueError(
            f"corpus of {len(text)} characters is too small to split; need a few hundred"
        )

    tok = CharTokenizer(text) if tokenizer == "char" else BPETokenizer(text, bpe_merges)

    data = {
        "task": task,
        "tokenizer": tok,
        "train": torch.tensor(tok.encode(text[:split]), dtype=torch.long),
        "val": torch.tensor(tok.encode(text[split:]), dtype=torch.long),
        "vocab_size": tok.vocab_size,
        "n_chars": len(text),
        "chars_per_token": len(text) / len(tok.encode(text)),
        "train_text": text[:split],
        "val_text": text[split:],
    }

    if task == "markov":
        # The oracle is defined over source symbols, which is characters here,
        # so it is only directly comparable to a char-level model. BPE changes
        # the unit; RESULTS.md converts via chars_per_token rather than
        # comparing the two numbers directly.
        val_symbols = [source.alphabet.index(c) for c in text[split:]]
        data["oracle_bits_per_char"] = source.oracle_bits_per_symbol(val_symbols)
        data["source"] = source
        data["val_symbols"] = val_symbols
        data["train_symbols"] = [source.alphabet.index(c) for c in text[:split]]

    return data


def get_batch(data, block_size, batch_size, generator=None, device="cpu"):
    """Sample a batch of (context, next-token) pairs.

    `y` is `x` shifted by one, so position t predicts t+1 for every t at once.
    That is the whole reason a causal transformer trains efficiently: one
    forward pass yields block_size training signals rather than one.
    """
    if len(data) <= block_size:
        raise ValueError(f"split of {len(data)} tokens is shorter than block_size {block_size}")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def load_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    print("=== bracket task (default) ===")
    d = build_dataset(n_symbols=60_000)
    spots = closer_positions(d["val_text"])
    print(f"  vocab {d['vocab_size']}, {len(d['train']):,} train / {len(d['val']):,} val chars")
    print(f"  {len(spots)} closing brackets in val, max match distance "
          f"{max(s['distance'] for s in spots)}")
    print(f"  sample: {d['train_text'][:80]}")
    print()
    print("  n-gram baselines on closing-bracket choice (chance = 0.333):")
    for k in (3, 5, 8):
        r = closer_accuracy_ngram(d["train_text"], d["val_text"], k)
        buckets = "  ".join(f"{b}:{v['accuracy']:.2f}" for b, v in r["by_distance"].items())
        print(f"    order-{k}: {r['accuracy']:.3f} overall   {buckets}")

    print()
    print("=== markov task (has a computable oracle) ===")
    m = build_dataset(n_symbols=50_000, task="markov")
    print(json.dumps({
        "vocab_size": m["vocab_size"],
        "oracle_bits_per_char": round(m["oracle_bits_per_char"], 4),
        "uniform_bits_per_char": round(math.log2(m["vocab_size"]), 4),
        "order3_counts_bits_per_char": round(count_model_bits_per_symbol(
            m["train_symbols"], m["val_symbols"], m["vocab_size"], 3), 4),
    }, indent=4))
