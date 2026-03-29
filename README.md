# MRNet Multiplane Fusion Research

This repository contains notebook-based experiments for multi-plane knee MRI classification inspired by MRNet. The work in `MRNet Hybrid/` compares three model variants and uses a two-stage fusion pipeline:

1. Train separate experts for each MRI plane (`axial`, `coronal`, `sagittal`)
2. Fuse plane-level logits with a logistic-regression classifier

The target labels are binary tasks:
- `abnormal`
- `acl`
- `meniscus`

## What Is In This Repo

```text
MRNet Hybrid/
	mrnet_multiplane_fusion_npy acl with test.ipynb
	mrnet_multiplane_fusion_npy meniscus with test.ipynb
	mrnet_custom_cnn_only_abnormal.ipynb
	mrnet_custom_cnn_only_acl.ipynb
	mrnet_custom_cnn_only_meniscus.ipynb
	mrnet_transfer_learning_only_abnormal.ipynb
	mrnet_transfer_learning_only_acl.ipynb
	mrnet_transfer_learning_only_meniscus .ipynb
	runs_mrnet_hybrid_fusion_npy/
	runs_mrnet_custom_cnn_only/
	runs_mrnet_transfer_learning_only/
```

## Notebook Families

### 1) Hybrid fusion notebooks
Files:
- `MRNet Hybrid/mrnet_multiplane_fusion_npy acl with test.ipynb`
- `MRNet Hybrid/mrnet_multiplane_fusion_npy meniscus with test.ipynb`

Configuration pattern:
- `RUNS_DIR = ./runs_mrnet_hybrid_fusion_npy`
- `epochs = 30`
- `use_custom_cnn = True`
- `use_cbam = True`
- `use_slice_attention = True`
- `backbone = densenet121` (config supports others)

Purpose:
- Full hybrid model (pretrained backbone + custom CNN branch)
- Attention-enabled feature aggregation
- Validation + held-out test evaluation
- Visualization pipeline (Grad-CAM and attention maps)

### 2) Custom CNN only notebooks
Files:
- `MRNet Hybrid/mrnet_custom_cnn_only_abnormal.ipynb`
- `MRNet Hybrid/mrnet_custom_cnn_only_acl.ipynb`
- `MRNet Hybrid/mrnet_custom_cnn_only_meniscus.ipynb`

Configuration pattern:
- `RUNS_DIR = ./runs_mrnet_custom_cnn_only`
- `epochs = 10`
- `use_custom_cnn = True`
- `use_cbam = False`
- `use_slice_attention = False`

Purpose:
- Ablation-style run where attention modules are off
- Compare simpler custom branch behavior against hybrid variant

### 3) Transfer learning only notebooks
Files:
- `MRNet Hybrid/mrnet_transfer_learning_only_abnormal.ipynb`
- `MRNet Hybrid/mrnet_transfer_learning_only_acl.ipynb`
- `MRNet Hybrid/mrnet_transfer_learning_only_meniscus .ipynb`

Configuration pattern:
- `RUNS_DIR = ./runs_mrnet_transfer_learning_only`
- `epochs = 10`
- `use_custom_cnn = False`
- `use_cbam = False`
- `use_slice_attention = False`

Purpose:
- Pure transfer-learning baseline
- Uses backbone features without custom branch or attention modules

## Data Expectations

Notebooks expect a local dataset root at:
- `./dataset`

Expected structure:

```text
dataset/
	train-abnormal.csv
	train-acl.csv
	train-meniscus.csv
	valid-abnormal.csv
	valid-acl.csv
	valid-meniscus.csv
	train/
		axial/*.npy
		coronal/*.npy
		sagittal/*.npy
	valid/
		axial/*.npy
		coronal/*.npy
		sagittal/*.npy
```

Data handling implemented in notebooks:
- Label load from CSV (`exam_id`, binary `label`)
- Stratified re-split of original MRNet training set
- Original validation split reused as held-out test set
- Volume standardization to 25 slices
- Intensity normalization and augmentation pipeline

## Pipeline Summary

### Stage 1: Plane experts
For each plane (`axial`, `coronal`, `sagittal`):
- Build model variant according to notebook family
- Train with focal loss support
- Save best checkpoint by validation AUC

Typical checkpoint names:
- Hybrid: `hybrid_expert_<task>_<plane>.pt`
- Custom only: `customcnn_expert_<task>_<plane>.pt`
- Transfer only: `transferlearning_expert_<task>_<plane>.pt`

### Stage 2: Fusion classifier
- Collect per-plane logits into a tabular feature set
- Train logistic regression fusion model
- Save fusion model and logits CSV

Saved outputs:
- `plane_logits_<task>.csv`
- `fusion_<task>.joblib`
- `test_logits_<task>.csv`
- `evaluation_plots_<task>.png`
- `test_evaluation_<task>.png`

## Verified Output Inventory (Current Repository State)

### `runs_mrnet_hybrid_fusion_npy/`
Contains complete artifacts for all three tasks (`abnormal`, `acl`, `meniscus`) including:
- expert checkpoints (`hybrid_expert_*`)
- fusion models (`fusion_abnormal.joblib`, `fusion_acl.joblib`, `fusion_meniscus.joblib`)
- validation/test logits CSVs for all tasks
- evaluation plot images
- framework, preprocessing, and table figure folders
- task-specific visualization folders:
	- `visualizations_abnormal/`
	- `visualizations_acl/`
	- `visualizations_meniscus/`

### `runs_mrnet_custom_cnn_only/`
Contains currently generated ACL artifacts:
- `customcnn_expert_acl_{axial,coronal,sagittal}.pt`
- `fusion_acl.joblib`
- `plane_logits_acl.csv`
- `test_logits_acl.csv`
- evaluation/test plots

### `runs_mrnet_transfer_learning_only/`
Contains currently generated ACL artifacts:
- `transferlearning_expert_acl_{axial,coronal,sagittal}.pt`
- `fusion_acl.joblib`
- `plane_logits_acl.csv`
- `test_logits_acl.csv`
- evaluation/test plots
- `visualizations_acl/` exists but is empty in the current snapshot

## CSV Schema

The logits CSV files use this structure:

```csv
split,exam_id,y,logit_axial,logit_coronal,logit_sagittal
```

Where:
- `split` is usually `train`/`valid` for fusion training or `test` for held-out evaluation
- `exam_id` is zero-padded exam identifier
- `y` is binary ground-truth label
- `logit_*` are raw model logits per plane

## Environment Setup

This repo is notebook-first (no packaged Python module yet). A practical setup:

```powershell
conda create -n mrnet-hybrid python=3.10 -y
conda activate mrnet-hybrid

pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn tqdm opencv-python pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional, used by Grad-CAM visualization cells
pip install grad-cam scikit-image

jupyter lab
```

## How To Run

1. Open one notebook in `MRNet Hybrid/`.
2. Verify these config variables early in the notebook:
	 - `DATA_ROOT`
	 - `TASK`
	 - `RUNS_DIR`
	 - `HYPER` settings
3. Run cells in order.
4. Check artifacts in the corresponding `runs_*` directory.

## Notes And Caveats

- Notebook filenames include spaces (for example `mrnet_transfer_learning_only_meniscus .ipynb` has a trailing space before `.ipynb`).
- Some output text inside notebooks appears to contain stale print paths from earlier runs. The configuration cells (`RUNS_DIR`, `TASK`) are the authoritative source.
- The repository currently stores generated model artifacts (`.pt`, `.joblib`, `.csv`, `.png`) directly under `MRNet Hybrid/runs_*`.

## License

Apache License 2.0. See `LICENSE`.