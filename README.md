<img src="./soft-moe.1.png" width="450px"></img>

<img src="./soft-moe.2.png" width="450px"></img>

## Soft MoE - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2308.00951">Soft MoE (Mixture of Experts)</a>, proposed by Brain's Vision team, in Pytorch.

This MoE has only been made to work with non-autoregressive encoder. However, some recent <a href="https://arxiv.org/abs/2305.18295">text-to-image models</a> have started using MoE with great results, so may be a fit there.

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
