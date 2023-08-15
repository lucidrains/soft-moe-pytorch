import math

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = - 1)

def pad_to_multiple(
    tensor,
    multiple,
    dim = -1,
    value = 0
):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return False, tensor

    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

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

class DynamicSlotsSoftMoE(Module):
    def __init__(
        self,
        dim,
        *,
        num_experts = 4,
        expert_mult = 4,
        dropout = 0.,
        geglu = False
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.num_experts = num_experts

        self.to_slot_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_experts, bias = False),
            Rearrange('b n (e d) -> b e n d', e = num_experts),
            RMSNorm(dim)
        )

        expert_klass = GLUFeedForward if geglu else FeedForward

        self.experts = nn.ModuleList([
            expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
        ])

    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        seq_len, is_image, num_experts = x.shape[-2], x.ndim == 4, self.num_experts

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        x = self.norm(x)

        # dynamic slot embeds
        # first average consecutive tokens, by number of experts
        # then, for each position, project out to that number of expert slot tokens
        # there should be # slots ~= sequence length, like in a usual MoE with 1 expert

        is_padded, x = pad_to_multiple(x, num_experts, dim = -2)

        if is_padded:
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool)

            _, mask = pad_to_multiple(mask, num_experts, dim = -1, value = False)

        x_segmented = rearrange(x, 'b (n e) d -> b n e d', e = num_experts)

        if exists(mask):
            segmented_mask = rearrange(mask, 'b (n e) -> b n e', e = num_experts)
            x_segmented = x_segmented.masked_fill(~rearrange(segmented_mask, '... -> ... 1'), 0.)

        # perform a masked mean

        if exists(mask):
            num = reduce(x_segmented, 'b n e d -> b n d', 'sum')
            den = reduce(segmented_mask.float(), 'b n e -> b n 1', 'sum').clamp(min = 1e-5)
            x_consecutive_mean = num / den
            slots_mask = segmented_mask.any(dim = -1)
        else:
            x_consecutive_mean = reduce(x_segmented, 'b n e d -> b n d', 'mean')

        # project to get dynamic slots embeddings
        # could potentially inject sinusoidal positions here too before projection

        slot_embeds = self.to_slot_embeds(x_consecutive_mean)

        logits = einsum('b n d, b e s d -> b n e s', x, slot_embeds)

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            slots_mask = rearrange(slots_mask, 'b s -> b 1 1 s')

            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
            logits = logits.masked_fill(~slots_mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

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

        return out[:, :seq_len]
