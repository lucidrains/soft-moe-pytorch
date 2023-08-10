<img src="./soft-moe.1.png" width="450px"></img>

<img src="./soft-moe.2.png" width="450px"></img>

## Soft MoE - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2308.00951">Soft MoE (Mixture of Experts)</a>, proposed by Brain's Vision team, in Pytorch.

This MoE has only been made to work with non-autoregressive encoder. However, some recent <a href="https://arxiv.org/abs/2305.18295">text-to-image models</a> have started using MoE with great results, so may be a fit there.

## Install

```bash
$ pip install soft-moe-pytorch
```

## Usage

```python
import torch
from soft_moe_pytorch import SoftMoE

moe = SoftMoE(
    dim = 512,         # model dimensions
    seq_len = 1024,    # max sequence length (will automatically calculate number of slots as seq_len // num_experts) - you can also set num_slots directly
    num_experts = 4    # number of experts - (they suggest number of experts should be high enough that each of them get only 1 slot. wonder if that is the weakness of the paper?)
)

x = torch.randn(1, 1024, 512)

out = moe(x) + x # (1, 1024, 512) - add in a transformer in place of a feedforward at a certain layer (here showing the residual too)
```

## Citations

```bibtex
@misc{puigcerver2023sparse,
	title 	= {From Sparse to Soft Mixtures of Experts}, 
	author 	= {Joan Puigcerver and Carlos Riquelme and Basil Mustafa and Neil Houlsby},
	year 	= {2023},
	eprint 	= {2308.00951},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG}
}
```
