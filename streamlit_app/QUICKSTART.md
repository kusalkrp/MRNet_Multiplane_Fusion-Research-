# 🚀 Quick Start Guide

## How to Launch the App

### Method 1: Double-click the Batch File (Easiest)
1. Double-click `run_app.bat`
2. The app will automatically open in your browser

### Method 2: Command Line
```bash
# Navigate to the app directory
cd C:\Users\ganee\Desktop\Research\streamlit_app

# Activate conda environment
conda activate gpu

# Run the app
streamlit run app.py
```

### Method 3: From any directory
```bash
# Activate environment
conda activate gpu

# Run with full path
streamlit run "C:\Users\ganee\Desktop\Research\streamlit_app\app.py"
```

---

## Accessing the App

Once running, the app will be available at:
- **Local URL:** http://localhost:8501
- **Network URL:** http://192.168.x.x:8501 (for access from other devices)

The browser should open automatically. If not, manually open the local URL above.

---

## Stopping the App

Press `Ctrl+C` in the terminal/command window to stop the server.

---

## Dark Theme ✅

The app now features a dark theme for better viewing experience!

---

## Troubleshooting

### "streamlit is not recognized"
Install Streamlit in your conda environment:
```bash
conda activate gpu
pip install streamlit plotly pillow
```

### Port already in use
If port 8501 is occupied, Streamlit will automatically try the next available port (8502, 8503, etc.)

### Can't see architecture diagrams
Run the diagram generator:
```bash
python generate_diagrams.py
```

---

## Features

✅ **Home/Overview** - Research summary and key findings  
✅ **Model Architectures** - Visual diagrams of CNN, Transfer Learning, and Hybrid models  
✅ **Interactive Prediction** - Upload MRI images for prediction (demo mode)  
✅ **Performance Comparison** - Charts and metrics across all models  
✅ **Interpretability Analysis** - Grad-CAM visualizations and attention analysis  

---

## Next Steps

1. **Add Real Architecture Diagrams:** The generator creates placeholder diagrams. For publication-quality diagrams, create them using tools like diagrams.net or Edraw.AI and save to `assets/architectures/`

2. **Update Performance Metrics:** Edit the `PERFORMANCE_DATA` dictionary in `app.py` with your exact experimental results

3. **Enable Model Inference:** Uncomment PyTorch dependencies in `requirements.txt` and implement actual model loading for real-time predictions

4. **Deploy to Cloud:** Follow the README.md instructions to deploy to Streamlit Community Cloud for public access

---

**Enjoy your interactive research dashboard! 🦴📊**
