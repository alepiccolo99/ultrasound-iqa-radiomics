# Ultrasound IQA Radiomics

Code and derived data for the study:

**A Radiomics-Based Machine Learning Framework for Full-Reference Ultrasound Image Quality Assessment Across Varying Acquisition Settings**

This repository contains the radiomic feature-extraction workflow, the reference-grouped nested model-development pipeline, derived study data needed to reproduce the primary modeling results, and a reusable workflow for applying the same analysis structure to another full-reference ultrasound IQA dataset.

## Scope

The repository supports two distinct workflows:

1. **Paper reproduction** — reproduce the primary grouped nested-validation results reported in the manuscript from the exact derived modeling inputs.
2. **Custom dataset analysis** — extract the same 280 radiomic descriptors from user-supplied reference/test ultrasound images and run the same reference-grouped nested model-development protocol when subjective targets are available.

The custom workflow is a model-development and validation workflow. It is not a pretrained deployment model.

## Repository structure

```text
ultrasound-iqa-radiomics/
├── config/
│   └── paper_modeling_config.json
├── data/
│   ├── paper/
│   │   ├── outer_fold_assignments.csv
│   │   ├── paper_modeling_data.mat
│   │   ├── reference_radiomic_features.csv
│   │   ├── roi_coordinates.csv
│   │   ├── roi_masks/
│   │   │   ├── kidney_esaote_roi_mask.png
│   │   │   ├── kidney_telemed_roi_mask.png
│   │   │   ├── thigh_esaote_roi_mask.png
│   │   │   ├── thigh_telemed_roi_mask.png
│   │   │   └── roi_mask_metadata.csv
│   │   ├── sample_metadata.csv
│   │   ├── subjective_scores.csv
│   │   └── test_radiomic_features.csv
│   └── user_input/
│       ├── dataset_manifest_template.csv
│       ├── roi_coordinates_template.csv
│       ├── subjective_scores_template.csv
│       ├── reference_images/
│       └── test_images/
├── matlab/
│   ├── extract_radiomic_features.m
│   ├── prepare_radiomic_dataset.m
│   └── radiomic_feature_names.m
├── python/
│   └── run_grouped_nested_validation.py
├── tests/
│   ├── reference_results/
│   └── verify_paper_reproduction.py
├── third_party/
│   └── README.md
├── .gitignore
├── LICENSE
├── LICENSING.md
├── requirements.txt
└── README.md
```

## Radiomic representation

For each reference/test image pair, the workflow extracts 56 radiomic features from each of five domains:

- native B-mode ROI;
- first-level Haar `LL`;
- first-level Haar `LH`;
- first-level Haar `HL`;
- first-level Haar `HH`.

This gives **280 descriptors per image**.

The public extractor uses the same study settings:

- rectangular ROI;
- same ROI coordinates for one reference image and all test images in that reference group;
- native B-mode scale;
- one-level Haar decomposition;
- signed wavelet coefficients retained;
- fixed-bin-number discretization with 32 bins;
- no resegmentation;
- no spatial resampling.

## Software

The paper workflow was validated with:

- MATLAB **R2026a Update 4** (`26.1.0.3312084`);
- Medical Imaging Toolbox;
- Image Processing Toolbox;
- Wavelet Toolbox;
- Python **3.13.9**;
- NumPy `2.5.1`;
- pandas `3.0.5`;
- SciPy `1.18.0`;
- scikit-learn `1.9.0`;
- joblib `1.5.3`.

Create and activate a Python environment, then install:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the paper modeling results

From the repository root:

```bash
python python/run_grouped_nested_validation.py --input-mode paper
```

The paper mode reads:

```text
data/paper/paper_modeling_data.mat
```

This MAT file contains the exact derived numerical inputs used by the public reproduction workflow. The CSV files under `data/paper/` are also provided as transparent, human-readable derived tables.

The analysis performs:

- six full-reference radiomic representations;
- zero-variance filtering;
- training-fold standardization;
- Spearman redundancy filtering at `|rho| > 0.90`;
- LASSO feature selection over eight candidate values;
- regression-model selection over the predefined study search space;
- outer leave-one-reference-group-out validation;
- inner leave-one-training-reference-group-out model selection;
- deterministic tie-breaking by RMSE, MAE, R2, then candidate order.

The master seed is:

```text
20250111
```

The complete configuration is stored in:

```text
config/paper_modeling_config.json
```

### Expected aggregate performance

| Metric | Value |
|---|---:|
| MAE | 0.11986515 |
| RMSE | 0.14618090 |
| R2 | 0.71958929 |
| PLCC | 0.87019786 |
| SROCC | 0.89296008 |

## Verify the reproduction

After running paper mode:

```bash
python tests/verify_paper_reproduction.py --strict-exact
```

In the validated reference environment, this produces:

```text
PAPER REPRODUCTION VERIFICATION: PASS
```

The verifier checks all 192 out-of-reference predictions, the four outer-fold metric rows, the four selected outer-fold pipelines, and all 34 selected-feature records.

For cross-system comparisons, the verifier can also be run without `--strict-exact`; its default numerical tolerance is `rtol=1e-10, atol=1e-10`.

## Apply the workflow to another dataset

### 1. Prepare images

Place user-supplied images in:

```text
data/user_input/reference_images/
data/user_input/test_images/
```

Private or third-party images should not be committed to the repository.

### 2. Prepare the dataset manifest

Use:

```text
data/user_input/dataset_manifest_template.csv
```

Required columns:

```text
SampleID,ReferenceID,ReferenceFile,TestFile
```

Each `ReferenceID` must map to exactly one reference image. A reference group may contain any number of associated test images.

### 3. Define one ROI per reference group

Use:

```text
data/user_input/roi_coordinates_template.csv
```

Required columns:

```text
ReferenceID,X,Y,Width,Height
```

Coordinates follow the MATLAB `imcrop` rectangle convention:

```text
[X, Y, Width, Height]
```

The same ROI coordinates are applied to the reference image and every associated test image in that group.

### 4. Extract radiomic features in MATLAB

Example:

```matlab
addpath("matlab")

summary = prepare_radiomic_dataset( ...
    "data/user_input/dataset_manifest.csv", ...
    "data/user_input/roi_coordinates.csv", ...
    "data/user_input/reference_images", ...
    "data/user_input/test_images", ...
    "data/user_input/derived");
```

The wrapper writes:

```text
reference_radiomic_features.csv
test_radiomic_features.csv
radiomic_dataset.mat
```

`radiomic_dataset.mat` stores the exact MATLAB doubles and SciPy-compatible metadata used by Python custom mode.

### 5. Add subjective targets

For supervised grouped model development, create:

```text
data/user_input/subjective_scores.csv
```

using the template:

```text
data/user_input/subjective_scores_template.csv
```

Required columns:

```text
SampleID,ReferenceID,SubjectiveTarget
```

`SampleID` and `ReferenceID` must match the radiomic dataset exactly.

If subjective targets are not available, the MATLAB radiomic extraction can still be used, but the supervised Python model-development workflow cannot be run.

### 6. Run custom grouped nested validation

Example:

```bash
python python/run_grouped_nested_validation.py \
  --input-mode custom \
  --custom-data data/user_input/derived/radiomic_dataset.mat \
  --scores-file data/user_input/subjective_scores.csv \
  --output-dir results/custom
```

The custom mode applies the same grouped nested model-development structure as paper mode. It supports a generic number of reference groups, provided at least three groups are available.

## Study data included in this repository

The public derived study data include:

- sample-level metadata for all 192 test images;
- exact rectangular ROI coordinates for the four reference groups;
- full-resolution binary ROI masks and ROI-mask metadata for the four reference groups;
- observer-level pair-mean subjective scores and final subjective targets;
- per-reference and per-test radiomic feature tables;
- exact derived modeling inputs;
- explicit outer-fold assignments;
- model-development configuration and master seed;
- expected predictions, fold metrics, selected pipelines, and selected features used by the reproduction verifier.

## Image availability

The original ultrasound images used in the study are **not distributed in this repository**.

The repository therefore provides the code, derived numerical data, ROI coordinates, full-resolution binary ROI masks, grouping information, subjective targets, split assignments, seeds, and model configuration needed to reproduce the principal radiomics/modeling analysis without redistributing the original images.

The reusable MATLAB workflow is provided so that the same extraction and grouped model-development structure can be applied to user-supplied images.

## Reproducibility status

The public workflow was validated against the frozen study analysis:

- paper mode reproduced all 192 predictions, four selected pipelines, four fold-metric rows, and 34 selected-feature records exactly in the reference Python environment;
- the MATLAB dataset wrapper reproduced the original frozen radiomic feature matrices exactly from the study images;
- the end-to-end custom workflow reproduced the paper results within `1e-10` numerical tolerance when run from the original images.

## License

Different parts of this repository are distributed under different terms:

- software, analysis code, configuration files, verification code, and reusable input templates: **BSD 3-Clause License**;
- derived research data under `data/paper/` and `tests/reference_results/`: **Creative Commons Attribution 4.0 International (CC BY 4.0)**;
- original ultrasound images: **not distributed and not licensed through this repository**;
- third-party software: governed by its respective upstream licenses and not redistributed here.

See [`LICENSE`](LICENSE) for the BSD 3-Clause text and [`LICENSING.md`](LICENSING.md) for the complete scope of the licensing terms. Third-party provenance is documented in [`third_party/README.md`](third_party/README.md).

## Citation

If you use this repository, please cite the associated article:

> *A Radiomics-Based Machine Learning Framework for Full-Reference Ultrasound Image Quality Assessment Across Varying Acquisition Settings.*

The final bibliographic citation and DOI will be added after publication.

## Contact

For questions about the repository:

**Alessandro Piccolo**<br>
Scuola Superiore Sant'Anna<br>
alessandro.piccolo@santannapisa.it
