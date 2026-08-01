"""
Architecture Diagram Generator for Knee MRI Research

This script generates architecture diagrams for the three models:
1. Custom CNN Only
2. Transfer Learning Only
3. Hybrid (Proposed)

Run this script to generate PNG diagrams in the assets/architectures folder.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "assets" / "architectures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_block(ax, x, y, width, height, text, color='#667eea', text_color='white', fontsize=9):
    """Create a rounded rectangle block with text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='#333', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)
    return box


def create_arrow(ax, start, end, color='#333'):
    """Create an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))


def generate_cnn_only_diagram():
    """Generate Custom CNN Only architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(5, 15.5, 'Custom CNN Only Architecture', fontsize=16, fontweight='bold', 
            ha='center', color='#2d5a87')
    
    # Input
    create_block(ax, 5, 14, 4, 0.8, 'Input MRI Volume\n(S × 1 × 224 × 224)', '#4ecdc4')
    
    # Arrow
    create_arrow(ax, (5, 13.6), (5, 12.8))
    
    # Custom CNN block
    create_block(ax, 5, 12, 5, 1.5, 'Custom CNN Feature Extractor\n\nConv(7×7, 32) → ReLU → MaxPool\nConv(5×5, 64) → ReLU → MaxPool\nConv(3×3, 128) → ReLU → MaxPool\nConv(3×3, 256) → ReLU → AdaptivePool', '#667eea')
    
    # Arrow
    create_arrow(ax, (5, 11.25), (5, 10.4))
    
    # Features
    create_block(ax, 5, 9.8, 3.5, 0.6, 'Slice Features (S × 256)', '#f093fb')
    
    # Arrow
    create_arrow(ax, (5, 9.5), (5, 8.6))
    
    # Max pool
    create_block(ax, 5, 8, 3.5, 0.8, 'Max Pooling\nacross Slices', '#ff6b6b')
    
    # Arrow
    create_arrow(ax, (5, 7.6), (5, 6.8))
    
    # Aggregated features
    create_block(ax, 5, 6.2, 3.5, 0.6, 'Aggregated Features (256)', '#f093fb')
    
    # Arrow
    create_arrow(ax, (5, 5.9), (5, 5.0))
    
    # Dropout
    create_block(ax, 5, 4.4, 2.5, 0.6, 'Dropout (0.4)', '#ffd93d')
    
    # Arrow
    create_arrow(ax, (5, 4.1), (5, 3.2))
    
    # Classification
    create_block(ax, 5, 2.6, 3.5, 0.8, 'Classification Head\nDense(256 → 1)', '#667eea')
    
    # Arrow
    create_arrow(ax, (5, 2.2), (5, 1.4))
    
    # Output
    create_block(ax, 5, 0.8, 3, 0.7, 'Sigmoid → Prediction', '#43e97b')
    
    # Legend
    ax.text(0.5, 0.3, '[X] No Transfer Learning  [X] No CBAM  [X] No Slice Attention', 
            fontsize=9, color='#888')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cnn_only_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'cnn_only_architecture.png'}")


def generate_transfer_learning_diagram():
    """Generate Transfer Learning Only architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(5, 15.5, 'Transfer Learning Only Architecture', fontsize=16, fontweight='bold', 
            ha='center', color='#2d5a87')
    
    # Input
    create_block(ax, 5, 14, 4.5, 0.8, 'Input MRI Volume\n(S × 3 × 224 × 224)', '#4ecdc4')
    ax.text(5, 13.2, '[Grayscale → RGB conversion]', fontsize=8, ha='center', color='#888', style='italic')
    
    # Arrow
    create_arrow(ax, (5, 13.0), (5, 12.2))
    
    # DenseNet block
    create_block(ax, 5, 11.5, 5.5, 1.5, 'DenseNet121 Backbone\n(ImageNet Pretrained)\n\nDense Blocks × 4\nTransition Layers × 3\nGlobal Avg Pool', '#38ef7d')
    
    # Arrow
    create_arrow(ax, (5, 10.75), (5, 9.8))
    
    # Features
    create_block(ax, 5, 9.2, 3.5, 0.6, 'Slice Features (S × 1024)', '#f093fb')
    
    # Arrow
    create_arrow(ax, (5, 8.9), (5, 8.0))
    
    # Max pool
    create_block(ax, 5, 7.4, 3.5, 0.8, 'Max Pooling\nacross Slices', '#ff6b6b')
    
    # Arrow
    create_arrow(ax, (5, 7.0), (5, 6.2))
    
    # Aggregated features
    create_block(ax, 5, 5.6, 3.5, 0.6, 'Aggregated Features (1024)', '#f093fb')
    
    # Arrow
    create_arrow(ax, (5, 5.3), (5, 4.4))
    
    # Dropout
    create_block(ax, 5, 3.8, 2.5, 0.6, 'Dropout (0.4)', '#ffd93d')
    
    # Arrow
    create_arrow(ax, (5, 3.5), (5, 2.6))
    
    # Classification
    create_block(ax, 5, 2.0, 3.5, 0.8, 'Classification Head\nDense(1024 → 1)', '#667eea')
    
    # Arrow
    create_arrow(ax, (5, 1.6), (5, 0.8))
    
    # Output
    create_block(ax, 5, 0.2, 3, 0.7, 'Sigmoid → Prediction', '#43e97b')
    
    # Legend
    ax.text(0.5, -0.3, '[OK] Transfer Learning  [X] No Custom CNN  [X] No CBAM  [X] No Slice Attention', 
            fontsize=9, color='#888')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transfer_learning_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'transfer_learning_architecture.png'}")


def generate_hybrid_diagram():
    """Generate Hybrid (Proposed) architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(7, 19.5, 'Hybrid Architecture (Proposed)', fontsize=18, fontweight='bold', 
            ha='center', color='#2d5a87')
    ax.text(7, 19.0, '* Novel Contribution: Dual-Branch + CBAM + Slice Attention *', 
            fontsize=11, ha='center', color='#11998e', style='italic')
    
    # Input
    create_block(ax, 7, 18, 5, 0.8, 'Input MRI Volume (S × 1 × 224 × 224)', '#4ecdc4')
    
    # Split arrows
    ax.annotate('', xy=(4, 16.6), xytext=(7, 17.6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(10, 16.6), xytext=(7, 17.6),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # Custom CNN Branch (left)
    create_block(ax, 4, 15.8, 4.5, 1.2, 'Custom CNN Branch\n\nConv layers (1→256)\nMRI-specific features', '#667eea')
    ax.text(4, 14.8, 'Domain-specific', fontsize=8, ha='center', color='#667eea', style='italic')
    
    # DenseNet Branch (right)
    create_block(ax, 10, 15.8, 4.5, 1.2, 'DenseNet121 Branch\n\n(ImageNet Pretrained)\nGeneral visual features', '#38ef7d')
    ax.text(10, 14.8, 'Transfer learning', fontsize=8, ha='center', color='#38ef7d', style='italic')
    
    # Arrows to concat
    ax.annotate('', xy=(5.5, 13.8), xytext=(4, 15.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(8.5, 13.8), xytext=(10, 15.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # Feature dimensions labels
    ax.text(4, 15.0, '256 dim', fontsize=8, ha='center', color='#333',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#667eea'))
    ax.text(10, 15.0, '1024 dim', fontsize=8, ha='center', color='#333',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#38ef7d'))
    
    # Concatenation
    create_block(ax, 7, 13.2, 4, 0.8, 'Feature Concatenation\n(1280 dim)', '#f093fb')
    
    # Arrow to CBAM
    create_arrow(ax, (7, 12.8), (7, 12.0))
    
    # CBAM Block (highlighted as novel)
    cbam_box = FancyBboxPatch(
        (3.5, 9.5), 7, 2.3,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='#fff3cd', edgecolor='#ffc107', linewidth=3
    )
    ax.add_patch(cbam_box)
    ax.text(7, 11.4, '** CBAM Attention Module **', fontsize=11, fontweight='bold',
            ha='center', color='#856404')
    
    # Channel Attention
    create_block(ax, 5, 10.3, 2.8, 0.7, 'Channel Attention\n"What to focus on"', '#ff6b6b', fontsize=8)
    
    # Spatial Attention
    create_block(ax, 9, 10.3, 2.8, 0.7, 'Spatial Attention\n"Where to focus"', '#4ecdc4', fontsize=8)
    
    # Arrow between channel and spatial
    ax.annotate('', xy=(7.5, 10.3), xytext=(6.5, 10.3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    
    # Arrow from CBAM
    create_arrow(ax, (7, 9.5), (7, 8.6))
    
    # Refined features
    create_block(ax, 7, 8.0, 4, 0.7, 'Attention-Refined Features\n(S × 1280)', '#f093fb')
    
    # Arrow to slice attention
    create_arrow(ax, (7, 7.65), (7, 6.8))
    
    # Slice Attention Block (highlighted)
    slice_box = FancyBboxPatch(
        (4, 5.2), 6, 1.5,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='#d4edda', edgecolor='#28a745', linewidth=3
    )
    ax.add_patch(slice_box)
    ax.text(7, 6.3, '** Slice Attention Aggregation **', fontsize=10, fontweight='bold',
            ha='center', color='#155724')
    ax.text(7, 5.7, 'Learnable importance weights per slice', fontsize=8,
            ha='center', color='#155724', style='italic')
    
    # Arrow from slice attention
    create_arrow(ax, (7, 5.2), (7, 4.4))
    
    # Aggregated features
    create_block(ax, 7, 3.8, 4, 0.7, 'Weighted Aggregated Features\n(1280)', '#f093fb')
    
    # Arrow
    create_arrow(ax, (7, 3.45), (7, 2.6))
    
    # Dropout
    create_block(ax, 7, 2.0, 2.5, 0.6, 'Dropout (0.4)', '#ffd93d')
    
    # Arrow
    create_arrow(ax, (7, 1.7), (7, 0.9))
    
    # Classification
    create_block(ax, 7, 0.3, 3.5, 0.8, 'Classification Head\nDense(1280 → 1)', '#667eea')
    
    # Arrow
    create_arrow(ax, (7, -0.1), (7, -0.8))
    
    # Output
    create_block(ax, 7, -1.4, 3.5, 0.7, 'Sigmoid → Prediction', '#43e97b')
    
    # Legend
    ax.text(0.5, -2.0, '[OK] Dual-Branch  [OK] Transfer Learning  [OK] Custom CNN  [OK] CBAM  [OK] Slice Attention', 
            fontsize=10, color='#155724', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hybrid_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'hybrid_architecture.png'}")


def generate_comparison_diagram():
    """Generate a comparison diagram showing all three architectures side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    
    architectures = [
        ("Custom CNN Only", ['Input MRI', 'Custom CNN\n(256 dim)', 'Max Pool', 'Classifier', 'Output'],
         ['#4ecdc4', '#667eea', '#ff6b6b', '#667eea', '#43e97b']),
        ("Transfer Learning", ['Input MRI', 'DenseNet121\n(1024 dim)', 'Max Pool', 'Classifier', 'Output'],
         ['#4ecdc4', '#38ef7d', '#ff6b6b', '#667eea', '#43e97b']),
        ("Hybrid (Proposed)", ['Input MRI', 'Dual-Branch\nCNN+DenseNet', 'CBAM\nAttention', 
                               'Slice\nAttention', 'Classifier', 'Output'],
         ['#4ecdc4', '#f093fb', '#ff6b6b', '#28a745', '#667eea', '#43e97b'])
    ]
    
    for idx, (title, blocks, colors) in enumerate(architectures):
        ax = axes[idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        ax.text(5, 13.5, title, fontsize=14, fontweight='bold', ha='center',
                color='#2d5a87' if idx < 2 else '#11998e')
        
        n_blocks = len(blocks)
        spacing = 10 / (n_blocks + 1)
        
        for i, (block_text, color) in enumerate(zip(blocks, colors)):
            y = 12 - (i + 1) * spacing * 1.2
            create_block(ax, 5, y, 4.5, 0.9, block_text, color, fontsize=9)
            
            if i < n_blocks - 1:
                ax.annotate('', xy=(5, y - 0.55), xytext=(5, y - 0.45),
                            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        
        # Highlight hybrid
        if idx == 2:
            highlight = FancyBboxPatch(
                (0.5, 0.5), 9, 13,
                boxstyle="round,pad=0.1,rounding_size=0.2",
                facecolor='none', edgecolor='#11998e', linewidth=3, linestyle='--'
            )
            ax.add_patch(highlight)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'combined_architecture.png'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Architecture Diagrams")
    print("=" * 60)
    
    generate_cnn_only_diagram()
    generate_transfer_learning_diagram()
    generate_hybrid_diagram()
    generate_comparison_diagram()
    
    print("=" * 60)
    print("✅ All diagrams generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)
