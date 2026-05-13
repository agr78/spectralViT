# Spectral Vision Transformer

Repository for [_Spectral Vision Transformer for Efficient Tokenization with Limited Data_](https://arxiv.org/abs/2605.12026).

![Schematic of the spectral ViT with parameterized linear projection $\mathbf{E}$. This embedding results in spectral tokens $\mathbf{s}$ with inherent rank or
hierarchical order. Our approach introduces a global inductive bias and spectral positional embeddings $\mathbf{e}_{pos}$ by mode rather than local patch embedding](figs/spectralViT_dm.png)

## Summary
Efficient image tokenization using projections on to spectral bases to reduce the number of parameters associated with fitting vision transformers.

## Contents
Various tokenizations can be found in [`bases.ipynb`](https://github.com/agr78/SpectralViT/blob/main/bases.ipynb) <br/>
Sample size varying pattern classifications can be found in [`pattern.ipynb`](https://github.com/agr78/SpectralViT/blob/main/sim.ipynb) <br/>
The spatially invariant Fourier basis example can be found in [`objects.ipynb`](https://github.com/agr78/SpectralViT/blob/main/spatial_invariance.ipynb) <br/>
Sex classification for all models can be found in [`ixi.ipynb`](https://github.com/agr78/SpectralViT/blob/main/ixi.ipynb) <br/>
Candidate selection can be found in [`dbs.ipynb`](https://github.com/agr78/SpectralViT/blob/main/dbs.ipynb) <br/>

## Publications
If this code is used, please cite the following:
> [Preprint](http://arxiv.org/abs/2605.12026): A. G. Roberts et al., ‘Spectral Vision Transformer for Efficient Tokenization with Limited Data’, arXiv [cs.CV]. 2026.
> 

## BibTex

```bibtex
@misc{Roberts2026spectralViT},
      title={Spectral Vision Transformer for Efficient Tokenization with Limited Data}, 
      author={Alexandra G. Roberts and 
            Maneesh John and
            Jinwei Zhang and 
            Dominick Romano and 
            Mert Sisman and 
            Ki Sueng Choi and 
            Heejong Kim and 
            Mert R. Sabuncu and 
            Thanh D. Nguyen and 
            Alexey V. Dimov and 
            Pascal Spincemaille and 
            Brian H. Kopell and 
            Yi Wang},
      year={2026},
      eprint={2605.12026},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.12026}, 
}
```

## Contact
Please direct questions to [Alexandra G. Roberts](https://github.com/agr78) at agr78@cornell.edu.
