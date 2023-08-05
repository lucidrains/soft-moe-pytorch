## Soft MoE - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2308.00951">Soft MoE</a>, proposed by Brain's Vision team, in Pytorch.

<a href="https://github.com/arankomat">Aran</a> notes that it has not been tested with autoregressive transformers. However, some recent <a href="https://arxiv.org/abs/2305.18295">text-to-image models</a> have started using MoE with great results, so may be a fit there.

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
