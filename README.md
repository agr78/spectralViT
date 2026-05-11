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

## Publications
If this code is used, please cite the following:
> [Preprint]([https://doi.org/10.1227/neu.0000000000003721](https://openreview.net/attachment?id=cH5gmxBepK&name=pdf)): A. G. Roberts et al., Spectral Vision Transformer for Efficient Tokenization with Limited Data, 2026
> 

## BibTex

```bibtex
@article{Roberts_RadDBS-QSM_2025,
  title    = "Technical feasibility of quantitative susceptibility mapping
              radiomics for predicting deep brain stimulation outcomes in
              Parkinson disease",
  author   = "Roberts, Alexandra G and Zhang, Jinwei and Tozlu, Ceren and
              Romano, Dominick and Akkus, Sema and Kim, Heejong and Sabuncu,
              Mert R and Spincemaille, Pascal and Li, Jianqi and Wang, Yi and
              Wu, Xi and Kopell, Brian H",
  journal  = "Neurosurgery",
  month    =  sep,
  year     =  2025,
  keywords = "Deep brain stimulation; Machine learning; Parkinson disease;
              Quantitative susceptibility mapping; Radiomics; Regression",
  language = "en"
}
```

## Contact
Please direct questions to [Alexandra G. Roberts](https://github.com/agr78) at agr78@cornell.edu.
