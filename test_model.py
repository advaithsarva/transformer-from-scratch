"""Plain asserts, no pytest, no network, no training.  About 3 seconds.

    python test_model.py

Two kinds of test, and both are needed.

**Equivalence.** Every hand-written layer is compared against the torch
built-in it replaces -- LayerNorm against nn.LayerNorm, gelu against F.gelu,
the attention against F.scaled_dot_product_attention.  torch is not used in
model.py; it is used here as the oracle.  "Built from scratch" is otherwise an
unfalsifiable claim.

**Causality.** The invariant.  Checked by perturbation rather than by reading
the mask, because reading the mask is how you convince yourself it is fine.
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (Block, FeedForward, LayerNorm, MultiHeadSelfAttention,
                   Transformer, gelu, sinusoidal_positions)

torch.manual_seed(0)
results = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        results.append((name, False, str(exc)))
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append((name, True, ""))


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


# --------------------------------------------------------------------------
# equivalence with torch
# --------------------------------------------------------------------------

def test_layernorm_matches_torch():
    """Including the eps-inside-the-sqrt detail and the biased variance.  Put
    eps outside, or use the unbiased variance, and this drifts in the fifth
    decimal -- enough to change training, not enough to notice by eye."""
    x = torch.randn(4, 9, 32)
    mine, theirs = LayerNorm(32), nn.LayerNorm(32)
    with torch.no_grad():
        theirs.weight.copy_(mine.gain)
        theirs.bias.copy_(mine.bias)
    assert close(mine(x), theirs(x)), f"max diff {(mine(x) - theirs(x)).abs().max():.2e}"


def test_layernorm_normalises_per_token_not_per_batch():
    """The BatchNorm confusion, made concrete: each token's 32 features must
    have mean 0 and variance 1 on their own.  Normalising over the batch
    instead would make inference depend on what else is in the batch."""
    x = torch.randn(4, 9, 32) * 7 + 3
    out = LayerNorm(32)(x)
    assert close(out.mean(dim=-1), torch.zeros(4, 9), tol=1e-4), "per-token mean is not 0"
    assert close(out.var(dim=-1, unbiased=False), torch.ones(4, 9), tol=1e-3), \
        "per-token variance is not 1"


def test_gelu_matches_torch():
    """The tanh approximation, so compare against F.gelu(approximate='tanh')."""
    x = torch.linspace(-6, 6, 500)
    assert close(gelu(x), F.gelu(x, approximate="tanh")), \
        f"max diff {(gelu(x) - F.gelu(x, approximate='tanh')).abs().max():.2e}"


def test_gelu_is_not_relu():
    """It must pass small negatives through. If someone 'simplifies' it to a
    ReLU this is the only test that notices."""
    x = torch.tensor([-0.5, -1.0])
    assert (gelu(x) < 0).all(), "gelu returned non-negative values for negative input"
    assert (gelu(x) > -0.2).all(), "gelu should be a small negative, not a large one"


def test_attention_matches_torch_sdpa():
    """THE equivalence test.  My multi-head causal attention against
    F.scaled_dot_product_attention with is_causal=True, sharing weights.

    This is what makes 'no nn.MultiheadAttention' a checkable claim.  It
    catches the scale factor, the mask placement, the head reshape and the
    transpose order in one go -- and every one of those is wrong-but-plausible
    if you get it subtly wrong."""
    B, T, C, H = 2, 7, 32, 4
    x = torch.randn(B, T, C)

    attn = MultiHeadSelfAttention(C, H, block_size=T, dropout=0.0).eval()
    mine = attn(x)

    q, k, v = attn.qkv(x).split(C, dim=2)
    q = q.view(B, T, H, C // H).transpose(1, 2)
    k = k.view(B, T, H, C // H).transpose(1, 2)
    v = v.view(B, T, H, C // H).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).contiguous().view(B, T, C))

    assert close(mine, ref, tol=1e-5), f"max diff {(mine - ref).abs().max():.2e}"


def test_attention_scale_is_head_dim_not_embedding_dim():
    """Dividing by sqrt(n_embd) instead of sqrt(head_dim) is the classic
    off-by-a-factor.  It does not crash and it does not look wrong; it just
    over-flattens the softmax so the model learns slowly and nobody knows why.

    With 4 heads the two differ by a factor of 2 in the scale, which changes
    the attention weights measurably -- so if this passes with the wrong
    constant, the test is broken, not the code."""
    B, T, C, H = 1, 6, 32, 4
    x = torch.randn(B, T, C)
    attn = MultiHeadSelfAttention(C, H, block_size=T, dropout=0.0).eval()
    _, w = attn(x, return_attention=True)

    q, k, _ = attn.qkv(x).split(C, dim=2)
    q = q.view(B, T, H, C // H).transpose(1, 2)
    k = k.view(B, T, H, C // H).transpose(1, 2)

    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    right = F.softmax(((q @ k.transpose(-2, -1)) / math.sqrt(C // H))
                      .masked_fill(mask == 0, float("-inf")), dim=-1)
    wrong = F.softmax(((q @ k.transpose(-2, -1)) / math.sqrt(C))
                      .masked_fill(mask == 0, float("-inf")), dim=-1)

    assert close(w, right), "attention does not use sqrt(head_dim)"
    assert not close(w, wrong, tol=1e-3), "the test cannot tell the two scales apart"


# --------------------------------------------------------------------------
# the invariant: causality
# --------------------------------------------------------------------------

def test_future_tokens_cannot_change_past_logits():
    """THE test.  Change the last token of the input; every logit before it
    must be bit-for-bit unchanged.

    Perturbation rather than mask inspection, because inspecting the mask is
    how you talk yourself into believing it is right.  A leak here makes
    training loss *better* and generation incoherent, and nothing raises."""
    torch.manual_seed(1)
    model = Transformer(vocab_size=20, n_embd=32, n_head=4, n_layer=3,
                        block_size=16, dropout=0.0).eval()

    idx = torch.randint(0, 20, (1, 12))
    with torch.no_grad():
        base, _ = model(idx)
        for position in (11, 8, 5):
            poked = idx.clone()
            poked[0, position] = (poked[0, position] + 7) % 20
            other, _ = model(poked)
            drift = (base[0, :position] - other[0, :position]).abs().max().item()
            assert drift == 0.0, (
                f"changing token {position} moved the logits at earlier positions "
                f"by {drift:.2e} -- the causal mask leaks the future"
            )


def test_attention_weights_are_lower_triangular_and_sum_to_one():
    """Two properties of a correct causal softmax.  The second is what breaks
    if you zero the future *after* the softmax instead of masking with -inf
    before it: the rows stop summing to 1 and the normalisation itself carries
    information backwards."""
    model = Transformer(vocab_size=20, n_embd=32, n_head=4, n_layer=2,
                        block_size=16, dropout=0.0).eval()
    with torch.no_grad():
        _, _, attentions = model(torch.randint(0, 20, (1, 10)), return_attention=True)

    for layer, w in enumerate(attentions):
        upper = torch.triu(w, diagonal=1)
        assert upper.abs().max().item() == 0.0, \
            f"layer {layer}: attends to the future, max weight {upper.abs().max():.2e}"
        assert close(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), tol=1e-5), \
            f"layer {layer}: attention rows do not sum to 1 -- masked after softmax?"


def test_the_causality_test_can_actually_fail():
    """Proof the test above is load bearing.  Build the same model with the
    mask replaced by all-ones -- fully bidirectional -- and assert the check
    catches it.  Without this, a mask that was never applied would pass
    everything silently."""
    torch.manual_seed(1)
    model = Transformer(vocab_size=20, n_embd=32, n_head=4, n_layer=3,
                        block_size=16, dropout=0.0).eval()
    for block in model.blocks:
        block.attn.mask.fill_(1.0)          # the bug, deliberately introduced

    idx = torch.randint(0, 20, (1, 12))
    with torch.no_grad():
        base, _ = model(idx)
        poked = idx.clone()
        poked[0, 11] = (poked[0, 11] + 7) % 20
        other, _ = model(poked)
        drift = (base[0, :11] - other[0, :11]).abs().max().item()

    assert drift > 1e-6, (
        "a fully bidirectional model passed the causality check; "
        "the check is decoration"
    )


# --------------------------------------------------------------------------
# shapes, positional encoding, generation
# --------------------------------------------------------------------------

def test_every_sublayer_preserves_shape():
    """The other half of the invariant: (B, T, C) in, (B, T, C) out."""
    x = torch.randn(3, 11, 64)
    for name, layer in [
        ("LayerNorm", LayerNorm(64)),
        ("FeedForward", FeedForward(64, dropout=0.0)),
        ("MultiHeadSelfAttention", MultiHeadSelfAttention(64, 8, 11, dropout=0.0)),
        ("Block", Block(64, 8, 11, dropout=0.0)),
    ]:
        assert layer(x).shape == x.shape, f"{name} changed shape to {layer(x).shape}"


def test_head_count_must_divide_the_embedding():
    """Raise, do not clamp.  Silently dropping the remainder channels would
    lose real capacity and never be noticed."""
    try:
        MultiHeadSelfAttention(30, 4, block_size=8)
        raise AssertionError("n_embd=30 with 4 heads was accepted; 30/4 is not an integer")
    except ValueError:
        pass


def test_sequence_longer_than_block_size_raises():
    model = Transformer(vocab_size=10, block_size=8, n_embd=32, n_head=4, n_layer=1)
    try:
        model(torch.randint(0, 10, (1, 9)))
        raise AssertionError("a 9-token sequence was accepted with block_size 8")
    except ValueError:
        pass


def test_positional_encoding_is_shift_linear():
    """The property that justifies sinusoidal encoding: PE[pos+k] is a fixed
    linear map of PE[pos], the same map at every position.  That is what lets
    one head learn 'three tokens back' once instead of per position.

    Checked via the sin/cos angle-addition identity on a single frequency
    pair, which is where the property actually comes from."""
    pe = sinusoidal_positions(64, 32)
    k = 5
    for pair in (0, 4, 10):
        i, j = 2 * pair, 2 * pair + 1
        theta = math.asin(float(pe[1, i]))                 # frequency of this pair
        cos_k, sin_k = math.cos(k * theta), math.sin(k * theta)
        for pos in (3, 17, 40):
            want_sin = float(pe[pos, i]) * cos_k + float(pe[pos, j]) * sin_k
            want_cos = float(pe[pos, j]) * cos_k - float(pe[pos, i]) * sin_k
            assert abs(want_sin - float(pe[pos + k, i])) < 1e-4, \
                f"pair {pair} pos {pos}: shift by {k} is not the expected rotation"
            assert abs(want_cos - float(pe[pos + k, j])) < 1e-4, \
                f"pair {pair} pos {pos}: shift by {k} is not the expected rotation"


def test_positional_encoding_does_not_underflow():
    """Computed in log space for a reason: the direct 1/10000**(2i/d) form
    underflows for large d and silently zeroes the slow dimensions, which
    destroys long-range position information without any error."""
    pe = sinusoidal_positions(128, 256)
    assert pe.abs().max() <= 1.0 + 1e-6, "positional encoding is out of [-1, 1]"
    slow = pe[:, -2:]
    assert slow.abs().max() > 1e-3, "the slowest dimensions underflowed to zero"
    assert not torch.isnan(pe).any(), "positional encoding contains NaN"


def test_weight_tying_shares_one_matrix():
    """Not two matrices that happen to be equal -- the same object.  A copy
    would drift apart during training and quietly double the parameter cost."""
    model = Transformer(vocab_size=50, n_embd=32, n_head=4, n_layer=1, block_size=8)
    assert model.head.weight is model.tok_emb.weight, "the weights are not tied"


def test_generation_is_deterministic_under_a_seed():
    """Same seed, same sample.  Otherwise nothing in RESULTS.md reproduces."""
    model = Transformer(vocab_size=20, n_embd=32, n_head=4, n_layer=2,
                        block_size=16, dropout=0.0)
    start = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(7)
    first = model.generate(start, 20)
    torch.manual_seed(7)
    assert torch.equal(first, model.generate(start, 20)), "generation is not reproducible"


def test_generation_respects_the_context_window():
    """Generating past block_size must crop the context, not raise or corrupt."""
    model = Transformer(vocab_size=20, n_embd=32, n_head=4, n_layer=2,
                        block_size=8, dropout=0.0)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 30)
    assert out.shape == (1, 31), f"expected 31 tokens, got {out.shape}"
    assert (out >= 0).all() and (out < 20).all(), "generated a token outside the vocabulary"


def test_untrained_loss_is_about_log_vocab():
    """A randomly initialised model should score ln(vocab_size) nats -- pure
    uniform guessing.  Much lower means the initialisation is leaking; much
    higher means it started saturated and will train badly."""
    model = Transformer(vocab_size=64, n_embd=64, n_head=4, n_layer=2,
                        block_size=16, dropout=0.0).eval()
    idx = torch.randint(0, 64, (8, 16))
    with torch.no_grad():
        _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(64)) < 0.25, \
        f"initial loss {loss.item():.3f}, expected about {math.log(64):.3f}"


def test_the_model_can_memorise_a_tiny_sequence():
    """The smallest end-to-end proof that gradients flow and the optimiser is
    wired up: overfit 32 tokens until the loss is near zero.  If the plumbing
    is broken this is where it shows, in two seconds instead of twenty
    minutes."""
    torch.manual_seed(3)
    model = Transformer(vocab_size=10, n_embd=64, n_head=4, n_layer=2,
                        block_size=32, dropout=0.0)
    idx = torch.randint(0, 10, (1, 33))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert loss.item() < 0.05, \
        f"could not memorise 32 tokens in 300 steps (loss {loss.item():.4f}) -- check the gradients"


TESTS = [
    ("layernorm matches torch", test_layernorm_matches_torch),
    ("layernorm normalises per token", test_layernorm_normalises_per_token_not_per_batch),
    ("gelu matches torch", test_gelu_matches_torch),
    ("gelu is not relu", test_gelu_is_not_relu),
    ("attention matches torch sdpa", test_attention_matches_torch_sdpa),
    ("attention scales by sqrt(head_dim)", test_attention_scale_is_head_dim_not_embedding_dim),
    ("future cannot change past logits", test_future_tokens_cannot_change_past_logits),
    ("attention is triangular and sums to 1", test_attention_weights_are_lower_triangular_and_sum_to_one),
    ("the causality test can fail", test_the_causality_test_can_actually_fail),
    ("every sublayer preserves shape", test_every_sublayer_preserves_shape),
    ("head count must divide embedding", test_head_count_must_divide_the_embedding),
    ("too-long sequence raises", test_sequence_longer_than_block_size_raises),
    ("positional encoding is shift-linear", test_positional_encoding_is_shift_linear),
    ("positional encoding does not underflow", test_positional_encoding_does_not_underflow),
    ("weight tying shares one matrix", test_weight_tying_shares_one_matrix),
    ("generation is deterministic", test_generation_is_deterministic_under_a_seed),
    ("generation respects the context window", test_generation_respects_the_context_window),
    ("untrained loss is about log(vocab)", test_untrained_loss_is_about_log_vocab),
    ("the model can memorise 32 tokens", test_the_model_can_memorise_a_tiny_sequence),
]


def main():
    for name, fn in TESTS:
        check(name, fn)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, err in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      {err}")
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
