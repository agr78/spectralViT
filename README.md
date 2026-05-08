# Spectral Vision Transformer

Repository for [_Spectral Vision Transformer for Efficient Tokenization with Limited Data_](https://openreview.net/pdf?id=cH5gmxBepK).

![Schematic of the spectral ViT with parameterized linear projection $\mathbf{E}$. This embedding results in spectral tokens $\mathbf{s}$ with inherent rank or
hierarchical order. Our approach introduces a global inductive bias and spectral positional embeddings $\mathbf{e}_{pos}$ by mode rather than local patch embedding](spectralViT_dm.png)

## Summary
Efficient image tokenization using projections on to spectral bases to reduce the number of parameters associated with fitting vision transformers.

## Contents
Various tokenizations can be found in `bases.ipynb` <br/>
Sample size varying pattern classifications can be found in `pattern.ipynb` <br/>
The spatially invariant Fourier basis example can be found in `objects.ipynb` <br/>
Sex classification for all models can be found in `ixi.ipynb` <br/>
Candidate selection can be found in `dbs.ipynb` <br/>
