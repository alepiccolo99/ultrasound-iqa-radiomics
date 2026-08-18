# Third-party software provenance

This repository does **not** redistribute third-party FR-IQA or learned-perceptual source code, model repositories, or pretrained weights.

The main radiomics paper-reproduction workflow under `matlab/` and `python/` does not require these external benchmark implementations. The entries below document the external software used for benchmark analyses in the associated study.

## Learned perceptual baselines

### LPIPS

- Metric: Learned Perceptual Image Patch Similarity (LPIPS)
- Upstream repository: https://github.com/richzhang/PerceptualSimilarity
- Frozen repository commit used in the study: `082bb24f84c091ea94de2867d34c4544f68e0963`
- Study configuration: AlexNet backbone, LPIPS v0.1 weights
- Upstream license: BSD 2-Clause
- Redistribution in this repository: **No**

The study used the upstream implementation and verified the frozen repository commit and relevant model-file hashes before scoring.

### DISTS

- Metric: Deep Image Structure and Texture Similarity (DISTS)
- Upstream repository: https://github.com/dingkeyan93/DISTS
- Frozen repository commit used in the study: `1267d8cb626c98706db3697422701c56a85ebf2e`
- Upstream license: MIT
- Redistribution in this repository: **No**

The study used the upstream implementation and verified the frozen repository commit and relevant model-file hashes before scoring.

## Classical full-reference IQA baselines

The study evaluated the following classical full-reference metrics:

- PSNR
- SSIM
- MS-SSIM
- IW-SSIM
- FSIM
- VSNR
- IFC
- VIF

PSNR was evaluated with the MATLAB implementation. The remaining classical metrics were evaluated using the legacy/original MATLAB implementations preserved in the study environment.

Those third-party MATLAB implementations are **not redistributed here** because the legacy bundle contains code from multiple upstream sources with heterogeneous provenance and licensing. Users wishing to reproduce those benchmark calculations should obtain the corresponding implementations from their original authors/distribution sources and consult the associated article references.

## Scope of the public repository

The absence of the benchmark source repositories does not affect the primary radiomics reproduction workflow provided here.

The public repository contains:

- the radiomic feature-extraction code;
- the grouped nested model-development code;
- exact derived study modeling inputs;
- subjective targets and metadata;
- outer-fold assignments;
- model configuration and seed;
- reference outputs and an automated reproduction verifier.

The learned and classical FR-IQA baselines were comparator analyses and are not dependencies of `run_grouped_nested_validation.py`.

## Environment note

`requirements.txt` specifies the Python environment needed for the primary grouped nested modeling workflow. It intentionally does not install LPIPS, DISTS, PyTorch, torchvision, or other benchmark-specific dependencies.

## Citations

For scientific use of LPIPS, DISTS, or any classical FR-IQA baseline, cite the corresponding original publication and implementation as appropriate. See the reference list of the associated manuscript for the study-specific citations.
