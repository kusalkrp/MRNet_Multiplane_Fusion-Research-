"""
Verification script for PyTorch Inference Pipeline in Streamlit App
"""

import sys
from pathlib import Path
import numpy as np

# Ensure streamlit_app is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

import inference_engine as ie

def test_pipeline():
    print("=== Testing Inference Engine ===")
    print("PyTorch Device:", ie.DEVICE)
    
    # 1. Test sample exams list
    sample_ids = ie.get_available_sample_ids()
    print("Available dataset sample IDs:", sample_ids[:5], "Total:", len(sample_ids))
    assert len(sample_ids) > 0, "No sample IDs found!"
    
    test_id = sample_ids[0]
    print(f"\n--- Testing with Exam ID: {test_id} ---")
    
    # 2. Load volumes
    ax, co, sa = ie.load_dataset_exam_planes(test_id)
    assert ax is not None and co is not None and sa is not None, "Failed to load volumes"
    print("Volume shapes - Axial:", ax.shape, "Coronal:", co.shape, "Sagittal:", sa.shape)
    
    # 3. Ground truth labels
    gt = ie.get_exam_ground_truth(test_id)
    print("Ground truth labels:", gt)
    
    # 4. Test Multiplane Inference for Hybrid Model (ACL task)
    print("\n--- Running Hybrid Model Multi-Plane Fusion (ACL task) ---")
    res_hybrid = ie.run_multiplane_inference(
        model_key="hybrid",
        task="acl",
        axial_vol=ax,
        coronal_vol=co,
        sagittal_vol=sa
    )
    
    if "error" in res_hybrid:
        print("ERROR in Hybrid inference:", res_hybrid["error"])
    else:
        print("Hybrid Model Result:")
        print("  - Prediction Label:", res_hybrid["prediction_label"])
        print("  - Fused Probability:", f"{res_hybrid['fused_prob']:.4f}")
        print("  - Confidence:", f"{res_hybrid['confidence']:.1f}%")
        print("  - Logits:", res_hybrid["logits"])
        print("  - Probabilities:", res_hybrid["probs"])
    
    # 5. Test Grad-CAM Heatmap Generation
    print("\n--- Generating Grad-CAM Heatmap (Axial Plane) ---")
    orig_img, heat_img, overlay_img = ie.generate_gradcam_heatmap(
        model_key="hybrid",
        task="acl",
        plane="axial",
        volume=ax,
        target_slice_idx=12
    )
    
    if orig_img is not None:
        print("Grad-CAM generation SUCCESS!")
        print("  - Original Image Size:", orig_img.size)
        print("  - Heatmap Image Size:", heat_img.size)
        print("  - Overlay Image Size:", overlay_img.size)
    else:
        print("Grad-CAM generation failed or returned None")
    
    print("\n=== ALL INFERENCE PIPELINE TESTS COMPLETED ===")

if __name__ == "__main__":
    test_pipeline()
