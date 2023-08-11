import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = - 1)

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# expert

def FeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# main class

class SoftMoE(Module):
    def __init__(
        self,
        dim,
        *,
        seq_len = None,
        num_experts = 4,
        num_slots = None,
        expert_mult = 4,
        dropout = 0.,
        geglu = False
    ):
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), 'either seq_len, or num_slots must be passed into SoftMoE'

        num_slots = default(num_slots, seq_len // num_experts)

        self.norm = RMSNorm(dim)

        self.slot_norm = RMSNorm(dim)
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))

        expert_klass = GLUFeedForward if geglu else FeedForward

        self.experts = nn.ModuleList([
            expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        is_image = x.ndim == 4

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)

        logits = einsum('b n d, e s d -> b n e s', x, slot_embeds)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> e b s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = []
        for slots_per_expert, expert in zip(slots, self.experts):
            out.append(expert(slots_per_expert))

        out = torch.stack(out)

        # combine back out

        out = rearrange(out, 'e b s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')

        return out
