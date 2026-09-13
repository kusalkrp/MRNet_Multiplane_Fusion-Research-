# TriPlane Health — Multi-Plane Knee MRI Diagnostic Workstation & Research

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF.svg?style=flat&logo=vite)](https://vitejs.dev)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-13%2F13%20Passing-00ff88.svg?style=flat)](#automated-testing--verification)

**TriPlane Health** is a PACS-grade clinical decision support system and research platform for automated anterior cruciate ligament (ACL) injury detection from multi-planar knee MRI scans (axial, coronal, and sagittal).

The system pairs a high-performance **FastAPI inference microservice** (`acl-inference-api`) hosting 9 PyTorch deep learning checkpoints and 3 Stage-2 Logistic Regression fusion models with an **OLED pitch-black & emerald green diagnostic workstation client** (`acl-dashboard`), alongside the original multiplane research experiments (`MRNet Hybrid/`).

---

## Table of Contents

- [Production System Architecture](#production-system-architecture)
- [Repository Structure](#repository-structure)
- [Production Implementation Details](#production-implementation-details)
  - [1. Multi-Plane Expert Models](#1-multi-plane-expert-models)
  - [2. Exact Preprocessing Pipeline](#2-exact-preprocessing-pipeline)
  - [3. Stage-2 Logistic Regression Fusion](#3-stage-2-logistic-regression-fusion)
  - [4. Explainability (Grad-CAM)](#4-explainability-grad-cam)
  - [5. Telemetry & SQLite Audit Trail](#5-telemetry--sqlite-audit-trail)
  - [6. PACS Diagnostic Dashboard](#6-pacs-diagnostic-dashboard)
- [How To Run (Production System)](#how-to-run-production-system)
  - [Prerequisites](#prerequisites)
  - [1. Start the Inference API](#1-start-the-inference-api)
  - [2. Start the PACS Dashboard](#2-start-the-pacs-dashboard)
  - [3. Access the Workstation](#3-access-the-workstation)
- [Automated Testing & Verification](#automated-testing--verification)
- [API Endpoints Reference](#api-endpoints-reference)
- [Checkpoint Provenance Mapping](#checkpoint-provenance-mapping)
- [Research Notebook Experiments (Original Research)](#research-notebook-experiments-original-research)
  - [What Is In This Repo](#what-is-in-this-repo)
  - [Notebook Families](#notebook-families)
  - [Data Expectations](#data-expectations)
  - [Pipeline Summary](#pipeline-summary)
  - [Verified Output Inventory](#verified-output-inventory-current-repository-state)
  - [CSV Schema](#csv-schema)
  - [Notebook Environment Setup](#notebook-environment-setup)
  - [How To Run Research Notebooks](#how-to-run-research-notebooks)
  - [Notes And Caveats](#notes-and-caveats)
- [License](#license)

---

## Production System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               TriPlane Health PACS Workstation               │
│                       (acl-dashboard)                        │     │
│   • Multi-Model Live Comparison (Actual vs Predicted Label)  │
│   • Real-Time Radial Probability Gauge & Triage Alerting     │
│   • Multi-Plane Fusion Contribution Weight Breakdown         │
│   • On-Demand Grad-CAM Slice Heatmap Visualizer              │
│   • SQLite Audit Trail Drawer (21 CFR Part 11 Aligned)       │
└──────────────────────────────▲───────────────────────────────┘
                               │ HTTP JSON / REST API (Port 8000)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 FastAPI Inference Service                    │
│                    (acl-inference-api)                       │
│  ├── Model Registry (In-Memory Cached Models):               │
│  │   ├── Proposed Hybrid (DenseNet121 + Custom CNN + CBAM)   │
│  │   ├── Transfer Learning (DenseNet121 Baseline)            │
│  │   └── Custom CNN (4-Block CNN Trained from Scratch)       │
│  ├── Stage-2 Logistic Regression Fusion Classifiers          │
│  ├── Scientific Preprocessing (valid_tf path verbatim)       │
│  ├── Slice-Level Grad-CAM Activation Engine                  │
│  └── SQLite Telemetry & Audit Logger (audit.db)              │
└──────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
MRNet_Multiplane_Fusion-Research-/
├── acl-inference-api/              # FastAPI Inference Microservice
│   ├── checkpoints/                # 9 PyTorch .pt checkpoints + 3 .joblib fusion models
│   ├── models/                     # PyTorch architecture modules
│   │   ├── hybrid_expert.py        # DenseNet121 + Custom CNN + CBAM + Slice Attention
│   │   ├── custom_cnn_expert.py    # 4-block scratch CNN with slice max pooling
│   │   └── transfer_expert.py      # Pretrained DenseNet121 backbone expert
│   ├── tests/                      # Automated test suite (contracts, regression, fusion)
│   │   ├── test_contracts.py       # API contract, schema, and error-handling tests
│   │   ├── test_fusion_weights.py  # Mathematical verification of logistic regression weights
│   │   └── test_golden_regression.py # Exact held-out test case regression (Exams 1130 & 1172)
│   ├── audit.py                    # SQLite audit trail logging and retrieval
│   ├── gradcam.py                  # PyTorch Grad-CAM computation & JET heatmap rendering
│   ├── inference.py                # Model registry and multiplane prediction pipeline
│   ├── main.py                     # FastAPI application & route declarations
│   ├── preprocessing.py            # MRI slice standardization and tensor normalization
│   ├── requirements.txt            # Python dependencies
│   └── schemas.py                  # Pydantic request/response schemas
│
├── acl-dashboard/                  # Modern PACS Diagnostic Web Client
│   ├── index.html                  # Diagnostic layout, triage queue, benchmark table, Grad-CAM
│   ├── package.json                # Vite configuration and scripts
│   ├── vite.config.js              # Vite server settings (Port 5173)
│   └── src/
│       ├── app.js                  # Client logic, API integration, and reactive state
│       └── style.css               # Pitch-black (#000000) & neon green (#00ff88) design system
│
├── dataset/                        # MRNet MRI volumes (.npy) and labels
│   ├── valid-acl.csv               # Held-out validation/test ground truth labels
│   └── valid/                      # 120 held-out volumes per plane (axial, coronal, sagittal)
│
├── MRNet Hybrid/                   # Research and training experiment notebooks
│   ├── mrnet_multiplane_fusion_npy acl with test.ipynb
│   ├── mrnet_custom_cnn_only_acl.ipynb
│   ├── mrnet_transfer_learning_only_acl.ipynb
│   └── runs_mrnet_*/               # Original training checkpoints and output CSVs
│
├── toolspec.md                     # Technical specification and architectural requirements
└── README.md                       # Repository overview and setup guide
```

---

## Production Implementation Details

### 1. Multi-Plane Expert Models

The backend hosts three distinct model paradigms for each of the three anatomical MRI planes (`axial`, `coronal`, `sagittal`), totaling 9 PyTorch checkpoints:

1. **Proposed Hybrid Architecture (`proposed`)**:
   - Combines a pretrained **DenseNet121** feature extractor with a parallel **Custom CNN** branch.
   - Applies **CBAM (Convolutional Block Attention Module)** for channel and spatial attention refinement.
   - Aggregates multi-slice representations using **Slice Attention Pooling** with learnable softmax scoring.
   - Evaluated held-out test AUC: **`0.954`** (Accuracy: `0.925`, F1-Score: `0.916`).
2. **Transfer Learning Baseline (`transfer_learning`)**:
   - DenseNet121 backbone pretrained on ImageNet.
   - Slices aggregated via standard **Slice Max Pooling**.
   - Evaluated held-out test AUC: **`0.892`** (Accuracy: `0.742`, F1-Score: `0.644`).
3. **Custom CNN Baseline (`custom_cnn`)**:
   - 4-stage convolutional neural network trained entirely from scratch.
   - Slices aggregated via **Slice Max Pooling**.
   - Evaluated held-out test AUC: **`0.827`** (Accuracy: `0.708`, F1-Score: `0.533`).

#### Held-Out Test Set Benchmark Comparison (Table 11 from Dissertation)

| Model Architecture | AUC | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN + Simple Fusion (Baseline)** | 0.827 | 0.708 | 0.952 | 0.370 | 0.970 | 0.533 |
| **Transfer Learning Only + Simple Fusion** | 0.892 | 0.742 | 0.848 | 0.519 | 0.939 | 0.644 |
| **Proposed Hybrid Multi-Plane + Learned Fusion** | **0.954** | **0.925** | **0.925** | **0.907** | **0.939** | **0.916** |
| *Gains vs Custom CNN* | *+0.127* | *+0.217* | *-0.027* | *+0.537* | *-0.031* | *+0.383* |
| *Gains vs Transfer Learning* | *+0.062* | *+0.183* | *+0.077* | *+0.388* | *0.000* | *+0.272* |

### 2. Exact Preprocessing Pipeline

To eliminate model drift and ensure identical outputs to the published research, `acl-inference-api/preprocessing.py` strictly replicates the notebook `valid_tf` evaluation pipeline:

- **Volume Standardization**: Standardizes variable-depth MRI scans to exactly **25 slices** using hybrid Shannon entropy slice ranking (`select_slices_hybrid_entropy`).
- **Spatial Resizing**: Resizes each slice to **224 × 224** pixels via bilinear interpolation.
- **Intensity Normalization**: Slices undergo per-volume min-max scaling, global z-score normalization, per-slice tensor normalization, and standard ImageNet channel standardization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Shape Invariance**: Guaranteed output tensor shape of `(1, 25, 3, 224, 224)`.

### 3. Stage-2 Logistic Regression Fusion

Rather than simple probability averaging, the system feeds the 3 plane logits $[l_{\text{axial}}, l_{\text{coronal}}, l_{\text{sagittal}}]$ into a calibrated **Stage-2 Logistic Regression classifier**:

$$P(\text{ACL Tear}) = \sigma\left(w_{\text{axial}} \cdot l_{\text{axial}} + w_{\text{coronal}} \cdot l_{\text{coronal}} + w_{\text{sagittal}} \cdot l_{\text{sagittal}} + b\right)$$

The exact loaded fusion parameters are verified across models:
- **Proposed Hybrid**: $w_{\text{axial}} = +3.3338,\; w_{\text{coronal}} = +2.5667,\; w_{\text{sagittal}} = -0.4003,\; b = -4.4974$
- **Transfer Learning**: $w_{\text{axial}} = +0.9641,\; w_{\text{coronal}} = +0.8821,\; w_{\text{sagittal}} = +0.2479,\; b = -2.3735$
- **Custom CNN**: $w_{\text{axial}} = +0.7163,\; w_{\text{coronal}} = +0.3158,\; w_{\text{sagittal}} = +0.4811,\; b = -1.8448$

### 4. Explainability (Grad-CAM)

The API provides on-demand visual interpretability via `POST /gradcam`:
- Generates slice-level gradient activation maps for the **Proposed Hybrid** (targeting the CBAM attention layer) and **Transfer Learning** (targeting `features.denseblock4`).
- Computes OpenCV **JET colormap attention heatmaps**, blended overlays (35% heatmap / 65% grayscale MRI), and peak activation coordinate hotspots $(x, y)$ pointing to intra-articular tear locations.
- Safely rejects unsupported architectures (`custom_cnn`) with an informative HTTP 422.

### 5. Telemetry & SQLite Audit Trail

Per medical software audit guidelines (21 CFR Part 11 / PACS traceability), `audit.py` records every diagnostic inference event to `audit.db`:
- Indexed fields: Timestamp (ISO-8601 UTC), Client IP, Exam ID, Model Architecture, Fused Probability, Predicted Label, Plane Logits, and Exact Execution Latency (ms).
- Queryable in real-time via `GET /audit-logs` and viewable in the dashboard modal drawer.

### 6. PACS Diagnostic Dashboard

Designed with a high-contrast **OLED Pitch Black (`#000000` / `#0a0a0a`) and Neon Emerald Green (`#00ff88`)** medical UI:
- **Case Intake Queue**: Instant 1-click loading and triage filtering (All, High Risk, Normal) across 120 held-out test cases.
- **Manual Case Upload**: Drag-and-drop support for three custom plane `.npy` files.
- **Radial Probability Gauge**: Instant visual display of ACL tear likelihood with clear triage headlines.
- **Multi-Model Benchmark Table**: Direct comparison of all 3 architectures featuring clean side-by-side **`Actual Label`** and **`Predicted Label`** columns.
- **Stage-2 Fusion Weight Bars**: Dynamic visualization of how each plane influenced the final diagnosis.
- **Interactive Grad-CAM Viewer**: Interactive plane selector and slice depth scrubber (0 to 24) for reviewing anatomical focus regions.

---

## How To Run (Production System)

### Prerequisites

Ensure you have the following installed:
- **Python 3.10+** (Tested on Python 3.10, 3.11, and 3.12)
- **Node.js 18+** & `npm`
- **Git**

---

### 1. Start the Inference API

Open a terminal in the project root:

```bash
# Navigate to API directory
cd acl-inference-api

# (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server via Uvicorn
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will initialize, load all 9 PyTorch checkpoints and 3 fusion models into memory, and listen at:
- **API URL**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:8000/docs`

---

### 2. Start the PACS Dashboard

Open a second terminal:

```bash
# Navigate to dashboard directory
cd acl-dashboard

# Install Vite dependencies
npm install

# Start the development server
npm run dev
```

The Vite dev server will launch at:
- **Dashboard URL**: `http://localhost:5173`

---

### 3. Access the Workstation

1. Open `http://localhost:5173` in your browser.
2. The header will display `API Online (3 models)` with active latency telemetry.
3. Select any held-out exam from the **Case Intake Queue** (e.g., **Exam #1172** for Tear, **Exam #1130** for Normal Knee).
4. Review the verdict, examine the **`Actual Label`** vs **`Predicted Label`** benchmark table, observe plane weighting, and click **"Compute Grad-CAM Attention"** to inspect slice activations.

---

## Automated Testing & Verification

The inference service includes a comprehensive automated test suite validating contracts, edge cases, fusion coefficients, and golden held-out test case reproduction:

```bash
# Run pytest from the repository root
python -m pytest acl-inference-api/tests -v
```

### Test Suite Summary (13/13 Passed)

| Test Suite | Focus Area | Status |
|---|---|:---:|
| `test_contracts.py` | `/health`, `/models`, Pydantic validation, 404/422 errors, SQLite audit logging | **PASSED** (8 tests) |
| `test_fusion_weights.py` | Exact coefficient checks ($w_{\text{axial}}, w_{\text{coronal}}, w_{\text{sagittal}}, b$) across all 3 fusion models | **PASSED** (2 tests) |
| `test_golden_regression.py` | Held-out test case reproduction: **Exam #1130** ($p=0.005\%$, Normal) and **Exam #1172** ($p=93.1\%$, Tear) | **PASSED** (3 tests) |

#### Verified PyTest Console Run Output

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.3.4, pluggy-1.6.0
rootdir: E:\Corse works\Research\MRNet_Multiplane_Fusion-Research-
collected 13 items

acl-inference-api/tests/test_contracts.py::test_health_endpoint PASSED                    [  7%]
acl-inference-api/tests/test_contracts.py::test_models_metadata_endpoint PASSED          [ 15%]
acl-inference-api/tests/test_contracts.py::test_predict_success PASSED                    [ 23%]
acl-inference-api/tests/test_contracts.py::test_unknown_model_returns_404 PASSED          [ 30%]
acl-inference-api/tests/test_contracts.py::test_malformed_npy_shape_returns_422 PASSED   [ 38%]
acl-inference-api/tests/test_contracts.py::test_gradcam_custom_cnn_rejected_with_422 PASSED [ 46%]
acl-inference-api/tests/test_contracts.py::test_gradcam_proposed_returns_overlay PASSED  [ 53%]
acl-inference-api/tests/test_contracts.py::test_audit_logs_recorded PASSED              [ 61%]
acl-inference-api/tests/test_fusion_weights.py::test_fusion_weights_exact_match PASSED   [ 69%]
acl-inference-api/tests/test_fusion_weights.py::test_fusion_weights_distinct_across_models PASSED [ 76%]
acl-inference-api/tests/test_golden_regression.py::test_golden_case_1130_normal PASSED   [ 84%]
acl-inference-api/tests/test_golden_regression.py::test_golden_case_1172_acl_tear PASSED [ 92%]
acl-inference-api/tests/test_golden_regression.py::test_compare_endpoint_runs_all_models PASSED [100%]

======================= 13 passed, 9 warnings in 43.22s =======================
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System liveness probe, CUDA status, and list of loaded models |
| `GET` | `/models` | Architectural metadata, target layers, and paper benchmark AUCs |
| `GET` | `/cases` | List available held-out validation cases and ground truth labels |
| `GET` | `/cases/{exam_id}` | Base64-encoded volumes for an exam across all 3 planes |
| `POST` | `/predict` | Single-model multi-plane inference and probability fusion |
| `POST` | `/predict/compare` | Multi-model live comparison across Proposed, Transfer, and Custom CNN |
| `POST` | `/gradcam` | On-demand slice-level Grad-CAM overlay, heatmap, and hotspot coordinates |
| `GET` | `/audit-logs` | Retrieve recent inference audit logs from SQLite |

---

## Checkpoint Provenance Mapping

The canonical names in the API checkpoints folder map to the training notebook outputs as follows:

| Notebook Output Location | Canonical API Name | Description |
|---|---|---|
| `runs_mrnet_hybrid_fusion_npy/hybrid_expert_acl_{plane}.pt` | `hybrid_expert_acl_{plane}.pt` | Proposed Hybrid expert checkpoints |
| `runs_mrnet_custom_cnn_only/customcnn_expert_acl_{plane}.pt` | `custom_cnn_expert_acl_{plane}.pt` | Custom CNN expert checkpoints |
| `runs_mrnet_transfer_learning_only/transferlearning_expert_acl_{plane}.pt` | `transfer_expert_acl_{plane}.pt` | Transfer Learning checkpoints |
| `runs_mrnet_hybrid_fusion_npy/fusion_acl.joblib` | `fusion_acl_hybrid.joblib` | Stage-2 Hybrid fusion model |
| `runs_mrnet_custom_cnn_only/fusion_acl.joblib` | `fusion_acl_custom_cnn.joblib` | Stage-2 Custom CNN fusion model |
| `runs_mrnet_transfer_learning_only/fusion_acl.joblib` | `fusion_acl_transfer.joblib` | Stage-2 Transfer Learning fusion model |

---

## Research Notebook Experiments (Original Research)

This section contains the original documentation for the notebook-based experiments for multi-plane knee MRI classification inspired by MRNet. The work in `MRNet Hybrid/` compares three model variants and uses a two-stage fusion pipeline:

1. Train separate experts for each MRI plane (`axial`, `coronal`, `sagittal`)
2. Fuse plane-level logits with a logistic-regression classifier

The target labels are binary tasks:
- `abnormal`
- `acl`
- `meniscus`

### What Is In This Repo

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

### Notebook Families

#### 1) Hybrid fusion notebooks
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

#### 2) Custom CNN only notebooks
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

#### 3) Transfer learning only notebooks
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

### Data Expectations

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

### Pipeline Summary

#### Stage 1: Plane experts
For each plane (`axial`, `coronal`, `sagittal`):
- Build model variant according to notebook family
- Train with focal loss support
- Save best checkpoint by validation AUC

Typical checkpoint names:
- Hybrid: `hybrid_expert_<task>_<plane>.pt`
- Custom only: `customcnn_expert_<task>_<plane>.pt`
- Transfer only: `transferlearning_expert_<task>_<plane>.pt`

#### Stage 2: Fusion classifier
- Collect per-plane logits into a tabular feature set
- Train logistic regression fusion model
- Save fusion model and logits CSV

Saved outputs:
- `plane_logits_<task>.csv`
- `fusion_<task>.joblib`
- `test_logits_<task>.csv`
- `evaluation_plots_<task>.png`
- `test_evaluation_<task>.png`

### Verified Output Inventory (Current Repository State)

#### `runs_mrnet_hybrid_fusion_npy/`
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

#### `runs_mrnet_custom_cnn_only/`
Contains currently generated ACL artifacts:
- `customcnn_expert_acl_{axial,coronal,sagittal}.pt`
- `fusion_acl.joblib`
- `plane_logits_acl.csv`
- `test_logits_acl.csv`
- evaluation/test plots

#### `runs_mrnet_transfer_learning_only/`
Contains currently generated ACL artifacts:
- `transferlearning_expert_acl_{axial,coronal,sagittal}.pt`
- `fusion_acl.joblib`
- `plane_logits_acl.csv`
- `test_logits_acl.csv`
- evaluation/test plots
- `visualizations_acl/` exists but is empty in the current snapshot

### CSV Schema

The logits CSV files use this structure:

```csv
split,exam_id,y,logit_axial,logit_coronal,logit_sagittal
```

Where:
- `split` is usually `train`/`valid` for fusion training or `test` for held-out evaluation
- `exam_id` is zero-padded exam identifier
- `y` is binary ground-truth label
- `logit_*` are raw model logits per plane

### Notebook Environment Setup

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

### How To Run Research Notebooks

1. Open one notebook in `MRNet Hybrid/`.
2. Verify these config variables early in the notebook:
	 - `DATA_ROOT`
	 - `TASK`
	 - `RUNS_DIR`
	 - `HYPER` settings
3. Run cells in order.
4. Check artifacts in the corresponding `runs_*` directory.

### Notes And Caveats

- Notebook filenames include spaces (for example `mrnet_transfer_learning_only_meniscus .ipynb` has a trailing space before `.ipynb`).
- Some output text inside notebooks appears to contain stale print paths from earlier runs. The configuration cells (`RUNS_DIR`, `TASK`) are the authoritative source.
- The repository currently stores generated model artifacts (`.pt`, `.joblib`, `.csv`, `.png`) directly under `MRNet Hybrid/runs_*`.

---

## License

Apache License 2.0. See `LICENSE`.