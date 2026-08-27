"""A decoder-only transformer built from PyTorch primitives.

Allowed: nn.Linear, nn.Embedding, nn.Dropout, tensor maths.
Not allowed anywhere in this file: nn.Transformer, nn.TransformerEncoder,
nn.MultiheadAttention, nn.LayerNorm, F.scaled_dot_product_attention,
F.gelu.  Those are the parts worth writing out, so they are written out.

``test_model.py`` checks each one against the torch built-in it replaces, so
"from scratch" is a claim with a number behind it rather than a promise.

THE INVARIANT
-------------
Two halves, and the second is the one that bites.

1. **Shape.** A tensor enters a block as ``(B, T, C)`` and leaves as
   ``(B, T, C)``.  Every sublayer is shape preserving.

2. **Causality.** The logits at position ``t`` depend on tokens ``0..t`` and
   on nothing after them.

Causality is the invariant because breaking it is invisible.  A mask that
leaks one position of future produces a *better* training loss, a *better*
validation loss, and completely incoherent generation -- the model has learned
to read the answer rather than predict it.  Nothing raises.  The curves look
like success.  ``test_model.py`` checks it by perturbing a future token and
asserting the past logits do not move.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# pieces that torch would otherwise hand us
# --------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Normalise each token's feature vector to zero mean, unit variance.

    Note this is *per token*, over the C dimension -- not over the batch.  That
    is the whole difference from BatchNorm and the reason transformers work
    with any batch size, including one, and why inference does not need running
    statistics.

    The eps sits inside the sqrt, matching torch.  Outside it, the gradient is
    wrong for near-zero variance.
    """

    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # biased variance (divide by n, not n-1), which is what torch uses here
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gain * (x - mean) / torch.sqrt(var + self.eps) + self.bias


def gelu(x):
    """Gaussian Error Linear Unit, tanh approximation -- the one GPT-2 used.

    Reads as "scale x by the probability that a standard normal is below x".
    Unlike ReLU it is smooth and lets small negatives through, which matters
    because a dead ReLU unit in a residual stream stays dead.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
    ))


def sinusoidal_positions(seq_len, dim, device=None):
    """The original positional encoding from Attention Is All You Need.

    Position ``pos``, dimension pair ``i``:

        PE[pos, 2i]   = sin(pos / 10000^(2i/dim))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/dim))

    Why this shape rather than "just use the integer position": each dimension
    is a sinusoid of a different wavelength, from about 2 up to 10000*2pi.  Low
    dimensions oscillate fast and encode fine position, high dimensions
    oscillate slowly and encode coarse position -- a smooth binary counter.

    The property that earns it its place: PE[pos+k] is a fixed linear function
    of PE[pos] for any fixed offset k, because shifting a sine by k is a
    rotation.  So a head can learn "attend three tokens back" as a single
    linear map, independent of where in the sequence it is.  Learned embeddings
    have to discover that separately for every position.

    Computed in log space -- exp(-log(10000) * 2i/dim) rather than
    1/10000**(2i/dim) -- because the direct form underflows to zero for large
    dim in float32 and silently kills the slow dimensions.
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    idx = torch.arange(0, dim, 2, dtype=torch.float, device=device)
    inv_freq = torch.exp(-math.log(10000.0) * idx / dim)

    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)
    return pe


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, written out.

    The shape dance, which is the part that takes a whiteboard:

        x            (B, T, C)
        qkv          (B, T, 3C)          one Linear, then split
        q, k, v      (B, T, C) each
        reshape      (B, T, H, C/H)      split channels across heads
        transpose    (B, H, T, C/H)      heads become a batch dimension
        q @ k^T      (B, H, T, T)        every token scores every token
        mask + softmax
        @ v          (B, H, T, C/H)
        transpose    (B, T, H, C/H)
        reshape      (B, T, C)           heads concatenated back
        out proj     (B, T, C)

    Heads are not separate modules.  They are one Linear whose output is
    *viewed* as H separate subspaces, which is why multi-head attention costs
    almost nothing over single-head: the same matmul, reinterpreted.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must divide evenly into n_head ({n_head}); "
                "heads split the channel dimension and a remainder has nowhere to go"
            )
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular ones.  Registered as a buffer, not a parameter: it
        # has no gradient but must move with .to(device) and be saved with the
        # model, and a plain attribute does neither.
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x, return_attention=False):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scale by sqrt(head_dim), NOT by n_embd.  q.k is a sum of head_dim
        # products; if q and k have unit-variance entries that sum has variance
        # head_dim, so without the scaling the softmax input grows with head
        # size, saturates, and the gradient through it goes to zero.  Getting
        # this constant wrong does not crash -- it just stops learning.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # -inf before the softmax, not 0 after it.  Zeroing afterwards leaves
        # the denominator containing future tokens, so rows no longer sum to 1
        # and information leaks backwards through the normalisation.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        weights = att
        att = self.attn_dropout(att)

        y = att @ v                                    # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        return (y, weights) if return_attention else y


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back.

    Applied to each position independently -- no mixing across time.  That
    division of labour is the architecture in one line: **attention moves
    information between positions, the MLP processes it within a position.**
    The 4x is from the original paper and has stuck ever since.
    """

    def __init__(self, n_embd, dropout=0.1, expansion=4):
        super().__init__()
        self.fc = nn.Linear(n_embd, expansion * n_embd)
        self.proj = nn.Linear(expansion * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block, pre-norm.

        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))

    Pre-norm (normalise going in) rather than the original post-norm
    (normalise coming out).  Post-norm needs a learning-rate warmup to train at
    all past a few layers, because the residual path passes through a LayerNorm
    and the identity gradient gets rescaled at every layer.  Pre-norm leaves a
    clean additive path from the loss to the embedding, which is why every
    model after GPT-2 uses it.  Warmup is still in train.py, and still helps,
    but it is no longer load bearing.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            delta, weights = self.attn(self.ln1(x), return_attention=True)
            x = x + delta
            return x + self.mlp(self.ln2(x)), weights
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class Transformer(nn.Module):
    """Decoder-only language model.

    Positional encoding is sinusoidal and fixed (a buffer, not a parameter),
    matching the original paper rather than GPT's learned table.  ``train.py
    --learned-pos`` switches it, and RESULTS.md measures both.
    """

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1, learned_pos=False):
        super().__init__()
        self.block_size = block_size
        self.learned_pos = learned_pos

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        if learned_pos:
            self.pos_emb = nn.Embedding(block_size, n_embd)
        else:
            self.register_buffer("pos_enc", sinusoidal_positions(block_size, n_embd))

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: the embedding and the output projection are the same
        # matrix. Saves vocab_size * n_embd parameters and consistently helps
        # small models -- measured in RESULTS.md rather than assumed.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """N(0, 0.02) -- the GPT-2 initialisation.

        torch's Linear default is uniform scaled by 1/sqrt(fan_in), which is
        fine for a few layers and too large once residuals accumulate: each
        block adds to the stream, so activation variance grows with depth and
        the model starts in saturation.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                f"sequence of {T} exceeds block_size {self.block_size}; "
                "truncate the context rather than silently cropping it here"
            )

        x = self.tok_emb(idx)
        if self.learned_pos:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        else:
            x = x + self.pos_enc[:T]
        x = self.drop(x)

        attentions = []
        for block in self.blocks:
            if return_attention:
                x, weights = block(x, return_attention=True)
                attentions.append(weights)
            else:
                x = block(x)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.reshape(B * T)
            )

        if return_attention:
            return logits, loss, attentions
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample continuations one token at a time.

        The context is cropped to the last block_size tokens, because the
        positional encoding has no entry beyond that.  Cropping here is correct
        -- it is a genuine limit of the architecture, not a parameter being
        clamped to hide a caller's mistake.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive; use top_k=1 for greedy decoding")

        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / temperature     # last position only

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k)[0][:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx

    def parameter_count(self, trainable_only=True):
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not trainable_only)
