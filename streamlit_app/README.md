# Knee MRI Injury Detection - Streamlit Research Dashboard

## 🦴 Enhanced Deep Learning Approach for Automated Knee Injury Detection in MRI Scans

**Author:** K.R. Punchihewa  
**University:** Coventry University  
**Program:** MSc Data Science & Machine Learning

---

## 📋 Overview

This Streamlit application showcases research on deep learning approaches for automated knee injury detection using MRI scans. It provides an interactive interface for:

- Research overview and key findings
- Model architecture comparisons (CNN-only, Transfer Learning, Hybrid)
- Performance metrics visualization
- Grad-CAM interpretability analysis
- Interactive prediction demo

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the research directory:**
   ```bash
   cd Research/streamlit_app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

---

## 📁 Project Structure

```
streamlit_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── assets/
    ├── architectures/        # Model architecture diagrams
    │   ├── cnn_only_architecture.png
    │   ├── transfer_learning_architecture.png
    │   ├── hybrid_architecture.png
    │   └── combined_architecture.png
    └── samples/              # Sample MRI images for demo

Code/  (parent folder)
├── runs_mrnet_custom_cnn_only/      # CNN-only model results
├── runs_mrnet_transfer_learning_only/  # Transfer learning results
└── runs_mrnet_hybrid_fusion_npy/    # Hybrid model results
    ├── evaluation_plots_*.png
    ├── test_evaluation_*.png
    ├── gradcam_*.png
    └── visualizations_*/
```

---

## 🎨 Adding Architecture Diagrams

To display model architecture diagrams, add PNG/SVG images to `assets/architectures/`:

1. **cnn_only_architecture.png** - Custom CNN architecture diagram
2. **transfer_learning_architecture.png** - DenseNet121 transfer learning diagram
3. **hybrid_architecture.png** - Proposed hybrid architecture with CBAM
4. **combined_architecture.png** - Side-by-side comparison (optional)

### Recommended Tools for Creating Diagrams:
- [diagrams.net (draw.io)](https://diagrams.net) - Free, web-based
- [NN-SVG](https://alexlenail.me/NN-SVG/) - Neural network visualizations
- [Edraw.AI](https://www.edraw.ai/) - Free version available
- [Lucidchart](https://www.lucidchart.com/) - Free tier available

### Diagram Style Guidelines:
- Use block/flowchart format
- Show: Input MRI → Layers/Branches → Fusion → Attention → Output
- Label branches (e.g., "Custom CNN for knee-specific features")
- Use arrows for data flow
- Different colors for different branches
- Include a legend
- Export as high-resolution PNG (300 DPI) or SVG

---

## 📊 Performance Metrics

The app displays actual performance metrics from your trained models. To update with your exact values, edit the `PERFORMANCE_DATA` dictionary in `app.py`:

```python
PERFORMANCE_DATA = {
    "Custom CNN Only": {
        "abnormal": {"AUC": 0.82, "Accuracy": 0.78, ...},
        "acl": {"AUC": 0.85, "Accuracy": 0.80, ...},
        ...
    },
    ...
}
```

---

## ☁️ Deployment to Streamlit Community Cloud

### Step 1: Prepare Repository

1. Create a new GitHub repository
2. Upload the `streamlit_app/` folder contents
3. Also upload or link to the `Code/` folder with model results

### Step 2: Configure for Deployment

Create a `.streamlit/config.toml` file:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Step 3: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Set main file path to `app.py`
5. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

---

## 🔧 Customization

### Updating Results Paths

If your results are in different locations, update the paths in `app.py`:

```python
BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "Code"

CNN_ONLY_DIR = CODE_DIR / "runs_mrnet_custom_cnn_only"
TRANSFER_DIR = CODE_DIR / "runs_mrnet_transfer_learning_only"
HYBRID_DIR = CODE_DIR / "runs_mrnet_hybrid_fusion_npy"
```

### Adding Model Inference

To enable actual model predictions, uncomment the PyTorch dependencies in `requirements.txt` and implement the inference logic in the Interactive Prediction section.

---

## 📝 Citation

If you use this work, please cite:

```
Punchihewa, K.R. (2026). Enhanced Deep Learning Approach for Automated 
Knee Injury Detection in MRI Scans: A Comparative Study Against 
State-of-the-Art Models. MSc Thesis, Coventry University.
```

---

## ⚠️ Disclaimer

- This application uses only public de-identified data (MRNet - Stanford ML Group)
- For research and educational purposes only
- Not intended for clinical diagnosis

---

## 📧 Contact

**K.R. Punchihewa**  
Coventry University  
MSc Data Science & Machine Learning
