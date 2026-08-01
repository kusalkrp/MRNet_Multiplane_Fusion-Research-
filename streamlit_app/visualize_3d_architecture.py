"""
Advanced 3D Interactive Neural Network Architecture Visualizations
for Knee MRI Injury Detection Research.

Each model is rendered with individual layers, sub-components,
tensor dimensions, data-flow annotations, and interpretability markers.
"""

import plotly.graph_objects as go
import numpy as np


# ── colour palette ──────────────────────────────────────────────────────────
C = dict(
    input='#6c757d',
    conv='#667eea',
    bn='#5a6fd6',
    relu='#7c8cf0',
    pool='#764ba2',
    dense='#f093fb',
    dropout='#c678dd',
    output='#ff6b6b',
    sigmoid='#e74c3c',
    backbone='#1a73e8',
    dense_block='#1565c0',
    transition='#0d47a1',
    gap='#9c27b0',
    concat='#f1c40f',
    cbam_ch='#27ae60',
    cbam_sp='#2ecc71',
    cbam_frame='#1abc9c',
    slice_att='#e67e22',
    slice_fc='#d35400',
    gradcam='rgba(255,0,0,0.25)',
    skip='rgba(150,150,150,0.15)',
    flow='rgba(180,200,255,0.5)',
    flow_cnn='rgba(102,126,234,0.6)',
    flow_tl='rgba(26,115,232,0.6)',
    flow_att='rgba(46,204,113,0.6)',
    # Multi-plane fusion pipeline colours
    fusion_lr='#ff9800',
    fusion_sc='#ffb74d',
    axial='#42a5f5',
    coronal='#66bb6a',
    sagittal='#ab47bc',
)


# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _box(xc, yc, zc, w, h, d, color, name, info, opacity=0.82, flat=False):
    """Mesh3d box centred at (xc, yc, zc)."""
    hw, hh, hd = w / 2, h / 2, d / 2
    xs = [xc-hw, xc+hw, xc+hw, xc-hw, xc-hw, xc+hw, xc+hw, xc-hw]
    ys = [yc-hh, yc-hh, yc+hh, yc+hh, yc-hh, yc-hh, yc+hh, yc+hh]
    zs = [zc-hd, zc-hd, zc-hd, zc-hd, zc+hd, zc+hd, zc+hd, zc+hd]
    i = [7,0,0,0,4,4,6,6,4,0,3,2]
    j = [3,4,1,2,5,6,5,2,0,1,6,3]
    k = [0,7,2,3,6,7,1,1,5,5,7,6]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        hovertext=f"<b>{name}</b><br>{info}",
        hoverinfo='text', showscale=False,
        lighting=dict(ambient=0.55, diffuse=0.6, specular=0.3, fresnel=0.2),
        lightposition=dict(x=500, y=500, z=500),
        flatshading=flat,
    )


def _wire_box(xc, yc, zc, w, h, d, color, name="", dash='dot'):
    """Wireframe outline – 12 edges of a box."""
    hw, hh, hd = w/2, h/2, d/2
    corners = np.array([
        [xc-hw,yc-hh,zc-hd],[xc+hw,yc-hh,zc-hd],[xc+hw,yc+hh,zc-hd],[xc-hw,yc+hh,zc-hd],
        [xc-hw,yc-hh,zc+hd],[xc+hw,yc-hh,zc+hd],[xc+hw,yc+hh,zc+hd],[xc-hw,yc+hh,zc+hd],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color=color, width=2, dash=dash),
        name=name, hoverinfo='none', showlegend=False,
    )


def _line(a, b, color='rgba(200,200,200,0.45)', width=3):
    return go.Scatter3d(
        x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]],
        mode='lines', line=dict(color=color, width=width),
        hoverinfo='none', showlegend=False,
    )


def _arrow_line(a, b, color='rgba(200,200,200,0.6)', width=3):
    """Line with a small cone at the end to indicate direction."""
    traces = [_line(a, b, color, width)]
    dx, dy, dz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
    l = max(np.sqrt(dx*dx+dy*dy+dz*dz), 1e-6)
    traces.append(go.Cone(
        x=[b[0]], y=[b[1]], z=[b[2]],
        u=[dx/l*0.4], v=[dy/l*0.4], w=[dz/l*0.4],
        colorscale=[[0, color], [1, color]],
        sizemode='absolute', sizeref=0.35,
        showscale=False, hoverinfo='none',
        anchor='tip',
    ))
    return traces


def _label(x, y, z, text, size=9, color='#ccc'):
    return go.Scatter3d(
        x=[x], y=[y], z=[z], mode='text',
        text=[text], textposition='top center',
        textfont=dict(size=size, color=color, family='Consolas, monospace'),
        hoverinfo='none', showlegend=False,
    )


def _dim_label(x, y, z, text, size=8, color='#aaa'):
    """Smaller tensor-shape annotation."""
    return go.Scatter3d(
        x=[x], y=[y], z=[z], mode='text',
        text=[text], textposition='bottom center',
        textfont=dict(size=size, color=color, family='Consolas'),
        hoverinfo='none', showlegend=False,
    )


def _gradcam_marker(xc, yc, zc, radius=0.6):
    """Translucent red sphere indicating Grad-CAM hook point."""
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 12)
    xs = xc + radius * np.outer(np.cos(u), np.sin(v))
    ys = yc + radius * np.outer(np.sin(u), np.sin(v))
    zs = zc + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0,'rgba(255,60,60,0.3)'],[1,'rgba(255,60,60,0.3)']],
        showscale=False, hoverinfo='text',
        hovertext='<b>Grad-CAM Hook</b><br>Gradient captured here<br>for interpretability analysis',
    )


def _layout(title, height=600, camera=None):
    cam = camera or dict(eye=dict(x=1.6, y=1.4, z=1.0))
    return dict(
        title=dict(text=title, font=dict(size=16, color='#ddd')),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data', bgcolor='rgba(0,0,0,0)',
            camera=cam,
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        paper_bgcolor='rgba(14,17,23,1)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=height,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SHARED – compact Stage 2 multi-plane fusion annotation
# ═══════════════════════════════════════════════════════════════════════════

def _add_stage2_fusion(fig, start_x, y=0):
    """Append compact Stage 2 multi-plane fusion visualisation to a figure."""
    S2 = start_x + 2.0
    # Wireframe around Stage 2
    fig.add_trace(_wire_box(S2 + 3.5, y, 0, 10, 6.5, 4, 'rgba(255,152,0,0.18)'))
    fig.add_trace(_label(S2 + 3.5, y, 4.0,
        'STAGE 2:  Multi-Plane Logistic Regression Fusion',
        size=10, color='#ff9800'))
    # Three plane logit boxes
    for dy, plane, col in [(2.0, 'Axial', C['axial']),
                           (0.0, 'Coronal', C['coronal']),
                           (-2.0, 'Sagittal', C['sagittal'])]:
        fig.add_trace(_box(S2, y + dy, 0, 0.6, 1.0, 1.0, col,
            f'{plane} Expert \u2192 logit',
            f'Same architecture trained on {plane} plane<br>'
            f'Outputs scalar logit z_{plane[0].lower()}', opacity=0.7))
        fig.add_trace(_label(S2, y + dy, 0.85,
            f'z_{plane[0].lower()}', size=8, color=col))
        for tr in _arrow_line((S2 + 0.3, y + dy, 0),
                              (S2 + 2.0, y + dy * 0.15, 0), col, 2):
            fig.add_trace(tr)
    # StandardScaler
    fig.add_trace(_box(S2 + 2.5, y, 0, 0.6, 1.8, 1.5, C['fusion_sc'],
        'StandardScaler',
        'Z-score normalise the 3 logits<br>Fitted on training set',
        opacity=0.85))
    fig.add_trace(_label(S2 + 2.5, y, 1.3, 'Scaler', size=8, color='#ffb74d'))
    fig.add_trace(_dim_label(S2 + 2.5, y, -1.3, '(N, 3)', size=7))
    for tr in _arrow_line((S2 + 2.8, y, 0), (S2 + 3.7, y, 0), '#ff9800', 3):
        fig.add_trace(tr)
    # Logistic Regression
    fig.add_trace(_box(S2 + 4.3, y, 0, 1.0, 2.2, 2.0, C['fusion_lr'],
        'Logistic Regression',
        'sklearn LogisticRegression (max_iter=2000)<br>'
        "P = \u03c3(w_a\u00b7z'_a + w_c\u00b7z'_c + w_s\u00b7z'_s + b)<br>"
        'Learns optimal weighting of 3 planes',
        opacity=0.85))
    fig.add_trace(_label(S2 + 4.3, y, 1.6, 'Logistic Reg.', size=9, color='#ff9800'))
    fig.add_trace(_dim_label(S2 + 4.3, y, -1.6, "\u03c3(w\u00b7z' + b)", size=7))
    for tr in _arrow_line((S2 + 4.8, y, 0), (S2 + 5.8, y, 0), '#43e97b', 4):
        fig.add_trace(tr)
    # Fused output
    fig.add_trace(_box(S2 + 6.3, y, 0, 0.8, 0.8, 0.8, '#43e97b',
        'Fused Prediction',
        'Combined probability from all 3 planes<br>P(injury) \u2208 [0, 1]',
        opacity=0.9))
    fig.add_trace(_label(S2 + 6.3, y, 0.9, 'Fused P(injury)', size=9, color='#43e97b'))


# ═══════════════════════════════════════════════════════════════════════════
# 1) CUSTOM CNN  –  fully expanded layer by layer
# ═══════════════════════════════════════════════════════════════════════════

def create_3d_cnn_model():
    fig = go.Figure()

    X = 0  # running x-position

    # ── Input ──
    fig.add_trace(_box(X, 0, 0, 0.4, 8, 8, C['input'],
        'Input MRI Slice',
        'Shape: (1 x 224 x 224)<br>Single-channel grayscale<br>Min-max + Z-score normalised'))
    fig.add_trace(_label(X, 0, 4.5, 'Input'))
    fig.add_trace(_dim_label(X, 0, -4.5, '1x224x224'))

    # ── Conv Block 1 ──
    X += 3
    fig.add_trace(_box(X, 0, 0, 0.6, 7.5, 7.5, C['conv'],
        'Conv2d  (1->32)',
        'Kernel: 7x7 | Stride: 2 | Pad: 3<br>Params: 1,600<br>Output: 32x112x112'))
    fig.add_trace(_label(X, 0, 4.3, 'Conv 7x7'))
    X += 1.0
    fig.add_trace(_box(X, 0, 0, 0.3, 7.2, 7.2, C['bn'],
        'BatchNorm2d(32)',
        'Running mean and var<br>Affine: True'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.25, 7.0, 7.0, C['relu'],
        'ReLU',
        'Activation: max(0, x)<br>In-place: True'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.3, 5.5, 5.5, C['pool'],
        'MaxPool2d',
        'Kernel: 2x2 | Stride: 2<br>Output: 32x56x56'))
    fig.add_trace(_dim_label(X, 0, -3.3, '32x56x56'))
    for tr in _arrow_line((0.2, 0, 0), (X-0.15, 0, 0), C['flow_cnn']): fig.add_trace(tr)

    # ── Conv Block 2 ──
    X += 2.0
    fig.add_trace(_box(X, 0, 0, 0.6, 5.2, 5.2, C['conv'],
        'Conv2d (32->64)',
        'Kernel: 5x5 | Stride: 1 | Pad: 2<br>Params: 51,264<br>Output: 64x56x56'))
    fig.add_trace(_label(X, 0, 3.2, 'Conv 5x5'))
    X += 1.0
    fig.add_trace(_box(X, 0, 0, 0.3, 5.0, 5.0, C['bn'],
        'BatchNorm2d(64)', 'Normalisation'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.25, 4.8, 4.8, C['relu'], 'ReLU', 'Activation'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.3, 3.8, 3.8, C['pool'],
        'MaxPool2d',
        'Kernel: 2x2 | Stride: 2<br>Output: 64x28x28'))
    fig.add_trace(_dim_label(X, 0, -2.5, '64x28x28'))

    # ── Conv Block 3 ──
    X += 2.0
    fig.add_trace(_box(X, 0, 0, 0.6, 3.5, 3.5, C['conv'],
        'Conv2d (64->128)',
        'Kernel: 3x3 | Stride: 1 | Pad: 1<br>Params: 73,856<br>Output: 128x28x28'))
    fig.add_trace(_label(X, 0, 2.5, 'Conv 3x3'))
    X += 1.0
    fig.add_trace(_box(X, 0, 0, 0.3, 3.3, 3.3, C['bn'], 'BatchNorm2d(128)', 'Normalisation'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.25, 3.1, 3.1, C['relu'], 'ReLU', 'Activation'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.3, 2.5, 2.5, C['pool'],
        'MaxPool2d',
        'Kernel: 2x2 | Stride: 2<br>Output: 128x14x14'))
    fig.add_trace(_dim_label(X, 0, -1.8, '128x14x14'))

    # ── Conv Block 4 ──
    X += 2.0
    fig.add_trace(_box(X, 0, 0, 0.6, 2.2, 2.2, C['conv'],
        'Conv2d (128->256)',
        'Kernel: 3x3 | Stride: 1 | Pad: 1<br>Params: 295,168<br>Output: 256x14x14'))
    fig.add_trace(_label(X, 0, 1.8, 'Conv 3x3'))
    X += 1.0
    fig.add_trace(_box(X, 0, 0, 0.3, 2.0, 2.0, C['bn'], 'BatchNorm2d(256)', 'Normalisation'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.25, 1.8, 1.8, C['relu'], 'ReLU', 'Activation'))
    X += 0.7
    fig.add_trace(_box(X, 0, 0, 0.3, 1.0, 1.0, C['pool'],
        'AdaptiveAvgPool2d',
        'Output: 256x1x1<br>Global spatial reduction'))
    fig.add_trace(_dim_label(X, 0, -1.0, '256x1x1'))

    # Grad-CAM hook point (last conv output)
    gcx = X - 1.65
    fig.add_trace(_gradcam_marker(gcx, 0, 0, 0.45))
    fig.add_trace(_label(gcx, 0, -2.5, 'Grad-CAM Hook', size=8, color='#ff6b6b'))

    # ── Flatten + FC ──
    X += 2.0
    fig.add_trace(_box(X, 0, 0, 0.15, 4, 0.15, C['dense'],
        'Flatten -> FC(256)',
        'Shape: (batch, 256)<br>Fully connected projection'))
    fig.add_trace(_label(X, 0, 2.5, 'FC 256'))
    fig.add_trace(_dim_label(X, 0, -2.5, 'Bx256'))

    # ── Dropout ──
    X += 1.2
    fig.add_trace(_box(X, 0, 0, 0.1, 3.5, 0.1, C['dropout'],
        'Dropout(0.3)',
        'Regularisation<br>p = 0.3'))
    fig.add_trace(_label(X, 0, 2.3, 'Drop 0.3'))

    # ── Slice Max Pool ──
    X += 2.0
    # Show multiple faint slices converging
    for s in range(-2, 3):
        fig.add_trace(_box(X - 0.3*abs(s), s*0.8, 0, 0.1, 2.5, 0.1, C['pool'],
            f'Slice {s+3}', f'Feature vector from slice {s+3}', opacity=0.3))
    fig.add_trace(_box(X+0.5, 0, 0, 0.3, 3.0, 0.3, C['pool'],
        'Max Pool (Slices)',
        'Aggregation: max across 25 slices<br>Input: (25, 256) -> Output: (256,)<br>Selects strongest activation per feature'))
    fig.add_trace(_label(X+0.5, 0, 2.0, 'Slice MaxPool'))
    fig.add_trace(_dim_label(X+0.5, 0, -2.0, '25x256 -> 256'))

    # ── Classifier ──
    X += 3.0
    fig.add_trace(_box(X, 0, 0, 0.4, 1.5, 0.4, C['dense'],
        'FC (256->1)',
        'Final linear projection<br>Params: 257'))
    fig.add_trace(_label(X, 0, 1.3, 'FC 1'))

    X += 1.2
    fig.add_trace(_box(X, 0, 0, 0.6, 0.6, 0.6, C['sigmoid'],
        'Sigmoid',
        'sig(x) = 1/(1+e^(-x))<br>Output: probability [0, 1]'))
    fig.add_trace(_label(X, 0, 0.8, 'Sigmoid'))
    fig.add_trace(_dim_label(X, 0, -0.8, 'P(injury)'))

    # overall flow arrow (Stage 1 per-plane output)
    for tr in _arrow_line((X-0.3, 0, 0), (X+0.8, 0, 0), '#43e97b', 4): fig.add_trace(tr)
    fig.add_trace(_label(X+1.3, 0, 0, 'Logit', color='#43e97b'))

    # Stage 1 label
    fig.add_trace(_label(14, 0, 5.5,
        '\u25c4\u2500\u2500 STAGE 1: Per-Plane Expert (trained \u00d73: axial, coronal, sagittal) \u2500\u2500\u25ba',
        size=10, color='#667eea'))

    # Append Stage 2 fusion
    _add_stage2_fusion(fig, X + 1.5)

    fig.update_layout(**_layout(
        'Custom CNN Expert (Stage 1) + Multi-Plane Fusion (Stage 2)',
        height=600,
        camera=dict(eye=dict(x=1.8, y=1.2, z=0.8))
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2) TRANSFER LEARNING  –  DenseNet121 expanded
# ═══════════════════════════════════════════════════════════════════════════

def create_3d_transfer_model():
    fig = go.Figure()
    X = 0

    # ── Input ──
    fig.add_trace(_box(X, 0, 0, 0.4, 8, 8, C['input'],
        'Input MRI Slice',
        'Shape: (1 x 224 x 224)<br>Grayscale'))
    fig.add_trace(_label(X, 0, 4.5, 'Input'))
    fig.add_trace(_dim_label(X, 0, -4.5, '1x224x224'))

    # ── Grayscale -> RGB conversion ──
    X += 1.8
    fig.add_trace(_box(X, 0, 0, 0.2, 7.8, 7.8, '#888',
        'Channel Repeat (1->3)',
        'Converts single-channel grayscale<br>to 3-channel pseudo-RGB<br>Required by ImageNet backbone'))
    fig.add_trace(_label(X, 0, 4.3, 'Gray->RGB'))
    fig.add_trace(_dim_label(X, 0, -4.3, '3x224x224'))
    for tr in _arrow_line((0.2, 0, 0), (X-0.1, 0, 0), C['flow_tl']): fig.add_trace(tr)

    # ── DenseNet121 Stem ──
    X += 2.2
    fig.add_trace(_box(X, 0, 0, 0.6, 7.0, 7.0, C['backbone'],
        'Stem Conv + BN + ReLU + Pool',
        'Conv2d(3->64, k=7, s=2, p=3)<br>BatchNorm -> ReLU -> MaxPool(3,s=2)<br>Output: 64x56x56'))
    fig.add_trace(_label(X, 0, 4.0, 'DenseNet Stem'))
    fig.add_trace(_dim_label(X, 0, -4.0, '64x56x56'))
    for tr in _arrow_line((X-1.8, 0, 0), (X-0.3, 0, 0), C['flow_tl']): fig.add_trace(tr)

    # ── Dense Blocks ──
    db_info = [
        ('Dense Block 1', '6 layers x BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)<br>Growth rate: 32<br>Output: 256x56x56', 5.5, '256x56x56'),
        ('Transition 1',  'BN -> ReLU -> Conv(1x1) -> AvgPool(2x2)<br>Compression: theta=0.5<br>Output: 128x28x28', 4.5, '128x28x28'),
        ('Dense Block 2', '12 layers x (BN-ReLU-Conv-BN-ReLU-Conv)<br>Growth rate: 32<br>Output: 512x28x28', 4.0, '512x28x28'),
        ('Transition 2',  'Compression -> AvgPool<br>Output: 256x14x14', 3.0, '256x14x14'),
        ('Dense Block 3', '24 layers x (BN-ReLU-Conv-BN-ReLU-Conv)<br>Growth rate: 32<br>Output: 1024x14x14', 3.5, '1024x14x14'),
        ('Transition 3',  'Compression -> AvgPool<br>Output: 512x7x7', 2.0, '512x7x7'),
        ('Dense Block 4', '16 layers x (BN-ReLU-Conv-BN-ReLU-Conv)<br>Growth rate: 32<br>Output: 1024x7x7', 2.5, '1024x7x7'),
    ]
    colors_db = [C['dense_block'], C['transition'], C['dense_block'],
                 C['transition'], C['dense_block'], C['transition'], C['dense_block']]

    prev_x = X
    X += 2.2
    for idx, (name, info, size, dim_txt) in enumerate(db_info):
        fig.add_trace(_box(X, 0, 0, 0.8, size, size, colors_db[idx], name, info))
        fig.add_trace(_label(X, 0, size/2+0.4, name, size=8))
        fig.add_trace(_dim_label(X, 0, -size/2-0.4, dim_txt, size=7))
        for tr in _arrow_line((prev_x+0.4, 0, 0), (X-0.4, 0, 0), C['flow_tl'], 2): fig.add_trace(tr)
        prev_x = X
        X += 1.6

    # Grad-CAM hook at last dense block
    fig.add_trace(_gradcam_marker(prev_x, 0, 0, 0.5))
    fig.add_trace(_label(prev_x, 0, -2.2, 'Grad-CAM Hook', size=8, color='#ff6b6b'))

    # Wireframe around entire backbone
    backbone_mid = (4 + prev_x) / 2
    backbone_w = prev_x - 4 + 2
    fig.add_trace(_wire_box(backbone_mid, 0, 0, backbone_w, 8, 8, 'rgba(26,115,232,0.3)',
        'DenseNet121 Backbone (ImageNet Pretrained)'))
    fig.add_trace(_label(backbone_mid, 0, 5.0, 'DenseNet121 Backbone (Pretrained)', size=10, color='#64b5f6'))

    # ── Global Average Pooling ──
    X += 1.0
    fig.add_trace(_box(X, 0, 0, 0.3, 1.5, 1.5, C['gap'],
        'Global Average Pool',
        'AdaptiveAvgPool2d(1)<br>1024x7x7 -> 1024x1x1'))
    fig.add_trace(_label(X, 0, 1.3, 'GAP'))
    fig.add_trace(_dim_label(X, 0, -1.3, '1024'))
    for tr in _arrow_line((prev_x+0.4, 0, 0), (X-0.15, 0, 0), C['flow_tl']): fig.add_trace(tr)

    # ── Slice MaxPool ──
    X += 2.5
    for s in range(-2, 3):
        fig.add_trace(_box(X-0.3*abs(s), s*0.7, 0, 0.08, 2, 0.08, C['pool'],
            f'Slice {s+3}', 'Feature vector per slice', opacity=0.25))
    fig.add_trace(_box(X+0.4, 0, 0, 0.3, 2.5, 0.3, C['pool'],
        'Max Pool (Slices)',
        'Aggregation: max over 25 slices<br>(25, 1024) -> (1024,)'))
    fig.add_trace(_label(X+0.4, 0, 1.8, 'Slice MaxPool'))
    fig.add_trace(_dim_label(X+0.4, 0, -1.8, '25x1024 -> 1024'))

    # ── Classifier + Sigmoid ──
    X += 3.0
    fig.add_trace(_box(X, 0, 0, 0.4, 1.5, 0.4, C['dense'],
        'FC (1024->1)',
        'Final linear projection<br>Params: 1,025'))
    fig.add_trace(_label(X, 0, 1.3, 'FC 1'))
    X += 1.2
    fig.add_trace(_box(X, 0, 0, 0.6, 0.6, 0.6, C['sigmoid'],
        'Sigmoid', 'sig(x) = 1/(1+e^(-x))<br>Output: P(injury)'))
    fig.add_trace(_label(X, 0, 0.8, 'Sigmoid'))

    for tr in _arrow_line((X-0.3, 0, 0), (X+0.8, 0, 0), '#43e97b', 4): fig.add_trace(tr)
    fig.add_trace(_label(X+1.3, 0, 0, 'Logit', color='#43e97b'))

    # Stage 1 label
    fig.add_trace(_label(12, 0, 5.5,
        '\u25c4\u2500\u2500 STAGE 1: Per-Plane Expert (trained \u00d73: axial, coronal, sagittal) \u2500\u2500\u25ba',
        size=10, color='#667eea'))

    # Append Stage 2 fusion
    _add_stage2_fusion(fig, X + 1.5)

    fig.update_layout(**_layout(
        'Transfer Learning Expert (Stage 1) + Multi-Plane Fusion (Stage 2)',
        height=620,
        camera=dict(eye=dict(x=1.8, y=1.3, z=0.9))
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3) HYBRID (PROPOSED)  –  dual branch + CBAM internals + slice attention
# ═══════════════════════════════════════════════════════════════════════════

def create_3d_hybrid_model():
    fig = go.Figure()

    Y_CNN  =  6   # upper branch
    Y_TL   = -6   # lower branch

    # === INPUT ===
    fig.add_trace(_box(0, 0, 0, 0.5, 9, 9, C['input'],
        'Input MRI Volume',
        'Shape: (S x 1 x 224 x 224)<br>S = 25 standardised slices<br>Single-channel grayscale'))
    fig.add_trace(_label(0, 0, 5.2, 'Input Volume'))
    fig.add_trace(_dim_label(0, 0, -5.2, '25x1x224x224'))

    # ════════════════════════════════════════════
    #  UPPER BRANCH  --  Custom CNN
    # ════════════════════════════════════════════
    bx = 4
    # Wireframe grouper
    fig.add_trace(_wire_box(7, Y_CNN, 0, 14, 5, 5, 'rgba(102,126,234,0.25)', 'Custom CNN Branch'))
    fig.add_trace(_label(7, Y_CNN, 3.2, 'Custom CNN Branch', size=10, color='#92a8ee'))

    cnn_layers = [
        (bx,    Y_CNN, 0, 0.5, 4, 4,   C['conv'], 'Conv 7x7 (1->32)',   'k=7, s=2, p=3 | BN | ReLU<br>Out: 32x112x112'),
        (bx+1.5,Y_CNN, 0, 0.3, 3, 3,   C['pool'], 'MaxPool',           '2x2 -> 32x56x56'),
        (bx+3,  Y_CNN, 0, 0.5, 2.8, 2.8,C['conv'],'Conv 5x5 (32->64)',  'k=5, p=2 | BN | ReLU<br>Out: 64x56x56'),
        (bx+4.5,Y_CNN, 0, 0.3, 2.2, 2.2,C['pool'],'MaxPool',           '2x2 -> 64x28x28'),
        (bx+6,  Y_CNN, 0, 0.5, 2.0, 2.0,C['conv'],'Conv 3x3 (64->128)', 'k=3, p=1 | BN | ReLU<br>Out: 128x28x28'),
        (bx+7.5,Y_CNN, 0, 0.3, 1.5, 1.5,C['pool'],'MaxPool',           '2x2 -> 128x14x14'),
        (bx+9,  Y_CNN, 0, 0.5, 1.3, 1.3,C['conv'],'Conv 3x3 (128->256)','k=3, p=1 | BN | ReLU<br>Out: 256x14x14'),
        (bx+10.5,Y_CNN,0, 0.3, 0.6, 0.6,C['pool'],'AdaptiveAvgPool',   '256x1x1 -> flatten -> 256'),
    ]
    # split line from input to branch
    for tr in _arrow_line((0.25, 0, 0), (bx-0.25, Y_CNN, 0), C['flow_cnn'], 3): fig.add_trace(tr)
    prev_pos = None
    for (x, y, z, w, h, d, col, name, info) in cnn_layers:
        fig.add_trace(_box(x, y, z, w, h, d, col, name, info))
        if prev_pos:
            for tr in _arrow_line(prev_pos, (x-w/2, y, z), C['flow_cnn'], 2): fig.add_trace(tr)
        prev_pos = (x+w/2, y, z)

    fig.add_trace(_dim_label(bx+10.5, Y_CNN, -1.2, '-> 256-d'))

    # Grad-CAM on last CNN conv
    fig.add_trace(_gradcam_marker(bx+9, Y_CNN, 0, 0.35))
    fig.add_trace(_label(bx+9, Y_CNN, -1.5, 'Grad-CAM Hook', size=7, color='#ff6b6b'))

    # ════════════════════════════════════════════
    #  LOWER BRANCH  --  DenseNet121
    # ════════════════════════════════════════════
    fig.add_trace(_wire_box(7, Y_TL, 0, 14, 6, 6, 'rgba(26,115,232,0.25)', 'Transfer Learning Branch'))
    fig.add_trace(_label(7, Y_TL, 3.8, 'DenseNet121 Branch (Pretrained)', size=10, color='#64b5f6'))

    for tr in _arrow_line((0.25, 0, 0), (bx-0.25, Y_TL, 0), C['flow_tl'], 3): fig.add_trace(tr)

    # RGB conversion
    fig.add_trace(_box(bx, Y_TL, 0, 0.2, 4.5, 4.5, '#777',
        'Gray->RGB', 'Channel repeat (1->3)<br>3x224x224'))
    fig.add_trace(_dim_label(bx, Y_TL, -2.8, '3x224x224', size=7))

    tl_layers = [
        (bx+2,  Y_TL, 0, 0.5, 4, 4,   C['backbone'], 'Stem+DB1',      'Conv7->BN->ReLU->Pool<br>DenseBlock1 (6 layers)<br>Transition1<br>Out: 128x28x28'),
        (bx+4,  Y_TL, 0, 0.5, 3, 3,   C['dense_block'],'DB2+Trans2',   'DenseBlock2 (12 layers)<br>Transition2<br>Out: 256x14x14'),
        (bx+6,  Y_TL, 0, 0.5, 2.5, 2.5,C['dense_block'],'DB3+Trans3',  'DenseBlock3 (24 layers)<br>Transition3<br>Out: 512x7x7'),
        (bx+8,  Y_TL, 0, 0.5, 2, 2,   C['dense_block'],'DenseBlock4',  'DenseBlock4 (16 layers)<br>Out: 1024x7x7'),
        (bx+10, Y_TL, 0, 0.3, 0.8, 0.8,C['gap'],       'GAP',          'AdaptiveAvgPool2d(1)<br>1024x7x7 -> 1024'),
    ]
    prev_pos = (bx+0.1, Y_TL, 0)
    for (x, y, z, w, h, d, col, name, info) in tl_layers:
        fig.add_trace(_box(x, y, z, w, h, d, col, name, info))
        for tr in _arrow_line(prev_pos, (x-w/2, y, z), C['flow_tl'], 2): fig.add_trace(tr)
        prev_pos = (x+w/2, y, z)

    fig.add_trace(_dim_label(bx+10, Y_TL, -1.5, '-> 1024-d'))

    # Grad-CAM on last DenseBlock
    fig.add_trace(_gradcam_marker(bx+8, Y_TL, 0, 0.4))
    fig.add_trace(_label(bx+8, Y_TL, -1.8, 'Grad-CAM Hook', size=7, color='#ff6b6b'))

    # ════════════════════════════════════════════
    #  FUSION  --  Concatenation
    # ════════════════════════════════════════════
    FX = 16
    fig.add_trace(_box(FX, 0, 0, 0.8, 5, 5, C['concat'],
        'Feature Concatenation',
        'Concatenate along feature dim<br>256 (CNN) + 1024 (DenseNet) = 1280<br>Shape per slice: (1280,)'))
    fig.add_trace(_label(FX, 0, 3.2, 'Concat [256+1024]'))
    fig.add_trace(_dim_label(FX, 0, -3.2, '1280-d / slice'))

    # Merge arrows
    for tr in _arrow_line((bx+10.8, Y_CNN, 0), (FX-0.4, 1, 0), C['flow_cnn'], 3): fig.add_trace(tr)
    for tr in _arrow_line((bx+10.3, Y_TL, 0), (FX-0.4, -1, 0), C['flow_tl'], 3): fig.add_trace(tr)

    # ════════════════════════════════════════════
    #  CBAM  --  Channel + Spatial (detailed internals)
    # ════════════════════════════════════════════
    CX = 20
    # Overall CBAM wireframe
    fig.add_trace(_wire_box(CX+1.5, 0, 0, 7, 7, 7, 'rgba(46,204,113,0.3)', 'CBAM Attention Block'))
    fig.add_trace(_label(CX+1.5, 0, 4.2, 'CBAM Attention Module', size=10, color='#2ecc71'))
    for tr in _arrow_line((FX+0.4, 0, 0), (CX-0.6, 0, 0), C['flow_att'], 3): fig.add_trace(tr)

    # ── Channel Attention sub-path (upper) ──
    ca_y = 2.5
    fig.add_trace(_box(CX-0.3, ca_y, 0, 0.5, 1.2, 1.2, C['cbam_ch'],
        'Global AvgPool',
        'Squeeze spatial dims<br>(B,1280,H,W) -> (B,1280,1,1)'))
    fig.add_trace(_box(CX-0.3, ca_y-2.5, 0, 0.5, 1.2, 1.2, C['cbam_ch'],
        'Global MaxPool',
        'Squeeze spatial dims<br>(B,1280,H,W) -> (B,1280,1,1)'))
    fig.add_trace(_label(CX-0.3, ca_y, 1.2, 'AvgPool', size=7))
    fig.add_trace(_label(CX-0.3, ca_y-2.5, -1.2, 'MaxPool', size=7))

    fig.add_trace(_box(CX+1.2, ca_y-1.25, 0, 0.6, 1.8, 0.8, C['cbam_ch'],
        'Shared MLP',
        'FC(1280->80) -> ReLU -> FC(80->1280)<br>Reduction ratio r=16<br>Weight-shared for both pooled vectors'))
    fig.add_trace(_label(CX+1.2, ca_y-1.25, 1.4, 'Shared MLP', size=8))

    fig.add_trace(_box(CX+2.6, ca_y-1.25, 0, 0.4, 1.0, 0.4, C['cbam_ch'],
        'Add + Sigmoid (Mc)',
        'Element-wise add -> Sigmoid<br>Channel attention map Mc in R^1280'))
    fig.add_trace(_label(CX+2.6, ca_y-1.25, 1.0, 'Sigmoid(+)', size=8))
    fig.add_trace(_dim_label(CX+2.6, ca_y-1.25, -1.0, 'Mc: 1280', size=7))

    # Connections inside channel attention
    for tr in _arrow_line((CX+0.0, ca_y, 0), (CX+0.9, ca_y-0.8, 0), C['flow_att'], 2): fig.add_trace(tr)
    for tr in _arrow_line((CX+0.0, ca_y-2.5, 0), (CX+0.9, ca_y-1.7, 0), C['flow_att'], 2): fig.add_trace(tr)
    for tr in _arrow_line((CX+1.5, ca_y-1.25, 0), (CX+2.4, ca_y-1.25, 0), C['flow_att'], 2): fig.add_trace(tr)

    # ── Spatial Attention sub-path (lower) ──
    sa_y = -2.8
    fig.add_trace(_box(CX+0.5, sa_y, 0, 0.5, 1.0, 1.0, C['cbam_sp'],
        'Channel AvgPool + MaxPool',
        'Pool across channels<br>(B,1280,H,W) -> 2x(B,1,H,W)<br>Concatenate -> (B,2,H,W)'))
    fig.add_trace(_label(CX+0.5, sa_y, 1.0, 'Ch-Pool', size=7))

    fig.add_trace(_box(CX+2.0, sa_y, 0, 0.5, 1.3, 1.3, C['cbam_sp'],
        'Conv2d(2->1, k=7) + Sigmoid',
        'Spatial attention conv<br>7x7 kernel | Pad: 3<br>Sigmoid -> Spatial map Ms in R^(HxW)'))
    fig.add_trace(_label(CX+2.0, sa_y, 1.3, 'Conv7x7 + Sigmoid', size=7))
    fig.add_trace(_dim_label(CX+2.0, sa_y, -1.3, 'Ms: HxW', size=7))

    for tr in _arrow_line((CX+0.75, sa_y, 0), (CX+1.75, sa_y, 0), C['flow_att'], 2): fig.add_trace(tr)

    # ── CBAM output: multiply ──
    fig.add_trace(_box(CX+3.5, 0, 0, 0.5, 2.0, 2.0, C['cbam_frame'],
        'Feature x Mc x Ms',
        'Element-wise multiplication<br>F_prime = F * Mc * Ms<br>Refined features: (B, 1280, H, W)'))
    fig.add_trace(_label(CX+3.5, 0, 1.5, 'F*Mc*Ms', size=8))
    fig.add_trace(_dim_label(CX+3.5, 0, -1.5, '1280-d (refined)'))

    for tr in _arrow_line((CX+2.8, ca_y-1.25, 0), (CX+3.25, 0.5, 0), C['flow_att'], 2): fig.add_trace(tr)
    for tr in _arrow_line((CX+2.25, sa_y, 0), (CX+3.25, -0.5, 0), C['flow_att'], 2): fig.add_trace(tr)

    # Grad-CAM at CBAM output
    fig.add_trace(_gradcam_marker(CX+3.5, 0, 0, 0.4))
    fig.add_trace(_label(CX+3.5, 0, -2.5, 'Grad-CAM Hook', size=7, color='#ff6b6b'))

    # ════════════════════════════════════════════
    #  SLICE ATTENTION  --  detailed
    # ════════════════════════════════════════════
    SX = CX + 6.5
    fig.add_trace(_wire_box(SX+1.5, 0, 0, 5, 5, 5, 'rgba(230,126,34,0.3)', 'Slice Attention'))
    fig.add_trace(_label(SX+1.5, 0, 3.2, 'Slice Attention Aggregation', size=10, color='#e67e22'))
    for tr in _arrow_line((CX+3.75, 0, 0), (SX-0.2, 0, 0), C['flow_att'], 3): fig.add_trace(tr)

    # Show individual slice features stacked
    for s in range(7):
        zy = -1.5 + s * 0.5
        fig.add_trace(_box(SX, zy, 0, 0.1, 2, 0.1, C['slice_att'],
            f'Slice {s+1} Feature',
            f'Feature vector for slice {s+1}<br>Shape: (1280,)',
            opacity=0.35 + 0.05*s))

    fig.add_trace(_box(SX+1.2, 0, 0, 0.5, 1.5, 0.5, C['slice_fc'],
        'Attention FC',
        'FC(1280->1) per slice<br>Learns importance score<br>for each MRI slice'))
    fig.add_trace(_label(SX+1.2, 0, 1.2, 'FC->Score', size=8))

    fig.add_trace(_box(SX+2.3, 0, 0, 0.3, 1.0, 0.3, C['slice_att'],
        'Softmax',
        'Normalise attention scores<br>alpha = softmax(scores)<br>Sum(alpha_i) = 1'))
    fig.add_trace(_label(SX+2.3, 0, 0.9, 'Softmax', size=8))

    fig.add_trace(_box(SX+3.2, 0, 0, 0.5, 1.8, 1.8, C['slice_att'],
        'Weighted Sum',
        'Output = Sum(alpha_i * f_i)<br>Aggregates 25 slices into<br>single 1280-d vector<br>using learned weights'))
    fig.add_trace(_label(SX+3.2, 0, 1.5, 'Sum(alpha_i * f_i)', size=8))
    fig.add_trace(_dim_label(SX+3.2, 0, -1.5, '1280-d'))

    for tr in _arrow_line((SX+0.05, 0, 0), (SX+0.95, 0, 0), C['flow_att'], 2): fig.add_trace(tr)
    for tr in _arrow_line((SX+1.45, 0, 0), (SX+2.15, 0, 0), C['flow_att'], 2): fig.add_trace(tr)
    for tr in _arrow_line((SX+2.45, 0, 0), (SX+2.95, 0, 0), C['flow_att'], 2): fig.add_trace(tr)

    # ════════════════════════════════════════════
    #  CLASSIFICATION HEAD
    # ════════════════════════════════════════════
    HX = SX + 5.5
    for tr in _arrow_line((SX+3.45, 0, 0), (HX-0.2, 0, 0), '#ccc', 3): fig.add_trace(tr)

    fig.add_trace(_box(HX, 0, 0, 0.3, 2.0, 0.3, C['dropout'],
        'Dropout(0.3)', 'Regularisation<br>p = 0.3'))
    fig.add_trace(_label(HX, 0, 1.5, 'Drop', size=8))

    HX += 1.2
    fig.add_trace(_box(HX, 0, 0, 0.4, 1.5, 0.4, C['dense'],
        'FC (1280->1)',
        'Final projection<br>Params: 1,281<br>Single output logit'))
    fig.add_trace(_label(HX, 0, 1.3, 'FC 1'))

    HX += 1.2
    fig.add_trace(_box(HX, 0, 0, 0.6, 0.6, 0.6, C['sigmoid'],
        'Sigmoid',
        'sig(x) = 1/(1+e^(-x))<br>Output in [0, 1]'))
    fig.add_trace(_label(HX, 0, 0.8, 'Sigmoid'))

    for tr in _arrow_line((HX-0.3, 0, 0), (HX+0.8, 0, 0), '#43e97b', 4): fig.add_trace(tr)
    fig.add_trace(_label(HX+1.5, 0, 0, 'Logit', color='#43e97b', size=11))

    # Stage 1 label
    fig.add_trace(_label(16, 0, 10,
        '\u25c4\u2500\u2500 STAGE 1: Per-Plane Expert (trained \u00d73: axial, coronal, sagittal) \u2500\u2500\u25ba',
        size=10, color='#667eea'))

    # Append Stage 2 fusion
    _add_stage2_fusion(fig, HX + 1.5)

    # ── Legend-like annotations ──
    fig.add_trace(_label(0, Y_CNN+3.5, 0, 'Custom CNN (domain-specific features)', size=8, color='#92a8ee'))
    fig.add_trace(_label(0, Y_TL-4.5, 0, 'DenseNet121 (pretrained general features)', size=8, color='#64b5f6'))
    fig.add_trace(_label(CX+1.5, -5, 0, 'CBAM: Channel + Spatial Attention', size=8, color='#2ecc71'))
    fig.add_trace(_label(SX+1.5, -4.5, 0, 'Slice Attention: Learnable aggregation', size=8, color='#e67e22'))
    fig.add_trace(_label(CX+1.5, -6, 0, 'Red spheres = Grad-CAM hook points for interpretability', size=8, color='#ff6b6b'))
    fig.add_trace(_label(HX+5, -7, 0, 'Orange wireframe = Stage 2 logistic regression fusion', size=8, color='#ff9800'))

    fig.update_layout(**_layout(
        'Hybrid Expert (Stage 1) + Multi-Plane Fusion (Stage 2)',
        height=720,
        camera=dict(eye=dict(x=1.6, y=1.5, z=1.1))
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4) COMPLETE FUSION PIPELINE  –  full two-stage overview
# ═══════════════════════════════════════════════════════════════════════════

def create_3d_fusion_pipeline():
    """Full two-stage multi-plane fusion pipeline (all 3 models share this structure)."""
    fig = go.Figure()

    # ── STAGE 1 header + wireframe ──
    fig.add_trace(_wire_box(5, 0, 0, 13, 17, 5, 'rgba(102,126,234,0.12)'))
    fig.add_trace(_label(5, 0, 9.5, 'STAGE 1:  Per-Plane Expert Training',
                         size=13, color='#667eea'))

    # ── MRI Exam input ──
    fig.add_trace(_box(-2.5, 0, 0, 0.6, 14, 5, C['input'],
        'MRI Examination',
        'Full knee MRI exam<br>Contains 3 orthogonal planes:<br>'
        'Axial \u00b7 Coronal \u00b7 Sagittal<br>Each plane \u2248 25 slices'))
    fig.add_trace(_label(-2.5, 0, 8, 'MRI Exam', size=11))

    # ── Three plane experts ──
    planes = [
        ('Axial',    5.5,  C['axial'],    'Top-down cross-sections'),
        ('Coronal',  0.0,  C['coronal'],  'Front-back cross-sections'),
        ('Sagittal', -5.5, C['sagittal'], 'Side cross-sections'),
    ]

    for plane, py, col, desc in planes:
        # Plane volume
        fig.add_trace(_box(0.5, py, 0, 0.4, 3.5, 3.5, col,
            f'{plane} Volume', f'{desc}<br>~25 slices per exam', opacity=0.6))
        fig.add_trace(_label(0.5, py, 2.2, plane, size=9, color=col))
        fig.add_trace(_dim_label(0.5, py, -2.2, '25 slices'))
        # Arrow from exam to plane
        for tr in _arrow_line((-2.2, 0, 0), (0.3, py, 0), col, 2):
            fig.add_trace(tr)
        # Expert box
        fig.add_trace(_box(5, py, 0, 6, 3.5, 3.5, col,
            f'{plane} Expert',
            f'<b>{plane} Per-Plane Expert</b><br><br>'
            f'<i>Hybrid model:</i><br>'
            f'CustomCNN(256-d) + DenseNet121(1024-d)<br>'
            f'\u2192 Concat(1280) \u2192 CBAM \u2192 SliceAttention<br>'
            f'\u2192 FC \u2192 Sigmoid<br><br>'
            f'<i>Architecture varies by model type</i><br>'
            f'(see individual model tabs for detail)',
            opacity=0.65))
        fig.add_trace(_label(5, py, 2.2, f'{plane} Expert', size=11, color='white'))
        for tr in _arrow_line((0.7, py, 0), (2, py, 0), col, 3):
            fig.add_trace(tr)
        # Logit output
        fig.add_trace(_box(9.5, py, 0, 1.0, 1.2, 1.2, col,
            f'logit_{plane[0].lower()}',
            f'Raw scalar logit from {plane} expert<br>Pre-sigmoid activation',
            opacity=0.8))
        fig.add_trace(_label(9.5, py, 1.0, f'z_{plane[0].lower()}', size=10, color=col))
        for tr in _arrow_line((8, py, 0), (9, py, 0), col, 3):
            fig.add_trace(tr)

    # ── STAGE 2 header + wireframe ──
    S2 = 13.5
    fig.add_trace(_wire_box(S2 + 5, 0, 0, 13, 17, 5, 'rgba(255,152,0,0.12)'))
    fig.add_trace(_label(S2 + 5, 0, 9.5, 'STAGE 2:  Logistic Regression Fusion',
                         size=13, color='#ff9800'))

    # Converge arrows from 3 logits to collect box
    for _, py, col, _ in planes:
        for tr in _arrow_line((10, py, 0), (S2 - 0.5, py * 0.2, 0), '#ffb74d', 2):
            fig.add_trace(tr)

    # Collect logits
    fig.add_trace(_box(S2, 0, 0, 1.2, 4, 2.5, C['fusion_sc'],
        'Collect Logits',
        'Stack logits into feature vector<br>'
        'X = [z_a, z_c, z_s]  shape: (N, 3)<br>'
        'N = number of exams', opacity=0.8))
    fig.add_trace(_label(S2, 0, 2.5, 'Collect Logits', size=10, color='#ffb74d'))
    fig.add_trace(_dim_label(S2, 0, -2.5, '[z_a, z_c, z_s]  (N, 3)'))

    # StandardScaler
    S2 += 3.5
    fig.add_trace(_box(S2, 0, 0, 1.2, 3.5, 2.5, C['fusion_sc'],
        'StandardScaler',
        'Z-score normalisation per feature<br>'
        "x' = (x \u2212 \u03bc) / \u03c3<br>"
        'Fitted on training logits only',
        opacity=0.85))
    fig.add_trace(_label(S2, 0, 2.3, 'StandardScaler', size=10, color='#ffb74d'))
    fig.add_trace(_dim_label(S2, 0, -2.3, '(N, 3) normalised'))
    for tr in _arrow_line((S2 - 2.3, 0, 0), (S2 - 0.6, 0, 0), '#ff9800', 3):
        fig.add_trace(tr)

    # Logistic Regression
    S2 += 4
    fig.add_trace(_box(S2, 0, 0, 2.0, 5, 3.5, C['fusion_lr'],
        'Logistic Regression',
        '<b>sklearn.linear_model.LogisticRegression</b><br>'
        'max_iter = 2000<br><br>'
        "P(injury) = \u03c3(w_a\u00b7z'_a + w_c\u00b7z'_c + w_s\u00b7z'_s + b)<br><br>"
        'Learns optimal weighting of the 3 planes<br>'
        'Typically w_sagittal > w_axial > w_coronal for ACL',
        opacity=0.85))
    fig.add_trace(_label(S2, 0, 3.2, 'Logistic Regression', size=12, color='#ff9800'))
    fig.add_trace(_dim_label(S2, 0, -3.2, "P = \u03c3(w\u00b7z' + b)", size=9))
    for tr in _arrow_line((S2 - 2.8, 0, 0), (S2 - 1, 0, 0), '#ff9800', 3):
        fig.add_trace(tr)

    # Final fused output
    S2 += 3.5
    fig.add_trace(_box(S2, 0, 0, 1.5, 2, 2, '#43e97b',
        'Fused Prediction',
        'Final fused probability P(injury)<br>'
        'Combines evidence from all 3 MRI planes<br>'
        'Output \u2208 [0, 1]',
        opacity=0.9))
    fig.add_trace(_label(S2, 0, 1.8, 'P(injury)', size=13, color='#43e97b'))
    for tr in _arrow_line((S2 - 2, 0, 0), (S2 - 0.75, 0, 0), '#43e97b', 4):
        fig.add_trace(tr)
    for tr in _arrow_line((S2 + 0.75, 0, 0), (S2 + 2, 0, 0), '#43e97b', 5):
        fig.add_trace(tr)
    fig.add_trace(_label(S2 + 2.5, 0, 0, 'FINAL\nDIAGNOSIS', size=11, color='#43e97b'))

    # ── Bottom annotations ──
    fig.add_trace(_label(5, -9, 0,
        'All 3 model types share this pipeline \u2014 only the internal per-plane expert differs',
        size=9, color='#aaa'))
    fig.add_trace(_label(5, -10, 0,
        'Custom CNN: 4-layer CNN (256-d)  |  Transfer Learning: DenseNet121 (1024-d)  |  Hybrid: CNN+DenseNet+CBAM+SliceAttn (1280-d)',
        size=8, color='#888'))

    fig.update_layout(**_layout(
        'Complete Two-Stage Multi-Plane Fusion Pipeline',
        height=700,
        camera=dict(eye=dict(x=1.3, y=1.6, z=1.2))
    ))
    return fig
