import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack

from soft_moe_pytorch.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    has_only_one_value
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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

# experts

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()
        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts

        # for now naively all gather across batch dimension if distributed, optimize later

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            assert has_only_one_value(seq_sizes), 'number of tokens per expert must be the same'

            x, batch_sizes = self.all_gather(x)

            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # the experts in use on the rank
        # for now, make sure number of machines is right multiple

        if world_size <= num_experts:
            assert divisible_by(num_experts, world_size), 'if number of machines is less than the number of experts, the number of experts must be divisible by number of machines'
            num_experts_per_rank = num_experts // world_size
            expert_start_index = rank * num_experts_per_rank
        else:
            assert divisible_by(world_size, num_experts), 'if number of machines is greater than number of experts, machines must be divisible by number of experts, so experts are evenly distributed'
            num_experts_per_rank = 1
            expert_start_index = rank // num_experts

        expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')
            x = rearrange(x, '(r eb) n d -> r eb n d', r = world_size)
            x = split_by_rank(x)
            x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        outs = torch.stack(outs)

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs = split_by_rank(outs)

        assert outs.shape == shape
        return outs

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
        geglu = False,
        is_distributed = None
    ):
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), 'either seq_len, or num_slots must be passed into SoftMoE'

        num_slots = default(num_slots, seq_len // num_experts)

        self.norm = RMSNorm(dim)

        self.slot_norm = RMSNorm(dim)
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))

        expert_klass = GLUFeedForward if geglu else FeedForward

        self.experts = Experts(
            experts = [expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)],
            is_distributed = is_distributed
        )

    def forward(self, x, mask = None):
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

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = self.experts(slots)

        # combine back out

        out = rearrange(out, ' b e s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')

        return out
