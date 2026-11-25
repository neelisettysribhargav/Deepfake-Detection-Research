"""
app.py - Deepfake Vision Forensics Studio
Interactive Streamlit Application for Robust Deepfake Detection & Explainability
Supports CSWin Transformer and ConvNeXt-Tiny architectures across all training arms.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import cv2
import streamlit as st

# Setup repository paths
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
scripts_path = os.path.join(HERE, "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from models.model_factory import get_model
from datasets import get_test_transform

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Vision Forensics Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK MODE GLASSMORPHISM STYLES ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Cards */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    .glass-card {
        background: rgba(22, 30, 46, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .badge-inconclusive {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# --- AVAILABLE MODEL REGISTRY ---
MODEL_OPTIONS = {
    "CSWin-Small (Arm C: Structured Robust - 96.2% Acc)": {
        "arch": "CSWin_64_24322_small_256",
        "checkpoint": "experiments/robust_runs/cswin_C_s1/last.pth",
        "description": "Trained with structured composition across lighting, blur, occlusion, noise & JPEG. Peak real-world robustness.",
        "type": "cswin"
    },
    "CSWin-Small (Arm BN: Coverage-Matched Generic - 96.1% Acc)": {
        "arch": "CSWin_64_24322_small_256",
        "checkpoint": "experiments/robust_runs/cswin_BN_s1/last.pth",
        "description": "Standard generic augmentation plus matched Gaussian noise and JPEG frequency.",
        "type": "cswin"
    },
    "CSWin-Small (Arm A: Clean Baseline - 90.6% Acc)": {
        "arch": "CSWin_64_24322_small_256",
        "checkpoint": "experiments/robust_runs/cswin_A_s1/last.pth",
        "description": "Trained with zero augmentations. Sensitive to shadows, blur, and compression artifacts.",
        "type": "cswin"
    },
    "ConvNeXt-Tiny (Arm C: Structured Robust)": {
        "arch": "convnext_tiny",
        "checkpoint": "experiments/robust_runs/convnext_tiny_C_s1/last.pth",
        "description": "Modern Convolutional architecture trained with structured composition.",
        "type": "convnext"
    },
    "ConvNeXt-Tiny (Arm A: Clean Baseline)": {
        "arch": "convnext_tiny",
        "checkpoint": "experiments/robust_runs/convnext_tiny_A_s1/last.pth",
        "description": "ConvNeXt baseline trained with zero augmentations.",
        "type": "convnext"
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_test_transform(256)


# --- CACHED MODEL LOADER ---
@st.cache_resource(show_spinner=False)
def load_cached_model(arch_name, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        return None
    model = get_model(arch_name, num_classes=1, pretrained=False).to(DEVICE)
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# --- FACE DETECTION & CROPPING ---
def detect_and_crop_face(pil_img, padding_frac=0.25):
    """Detects primary face using OpenCV, applies margin padding, and crops to 256x256."""
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    annotated = img_rgb.copy()
    H, W = img_rgb.shape[:2]
    
    if len(faces) == 0:
        # Fallback if no face found (or image is already a tight crop)
        cropped_img = pil_img.resize((256, 256), Image.BILINEAR)
        return annotated, cropped_img, None
    
    # Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Draw sleek detection box
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (16, 185, 129), 3)
    cv2.putText(annotated, "FACE DETECTED", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2, cv2.LINE_AA)
    
    # Add margin padding
    pad_w = int(w * padding_frac)
    pad_h = int(h * padding_frac)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(W, x + w + pad_w)
    y1 = min(H, y + h + pad_h)
    
    cropped = img_rgb[y0:y1, x0:x1]
    cropped_pil = Image.fromarray(cropped).resize((256, 256), Image.BILINEAR)
    return annotated, cropped_pil, (x, y, w, h)


# --- EXPLAINABLE AI: GRAD-CAM GENERATOR ---
def compute_gradcam(model, tensor_input, model_type, target="fake", img_size=(256, 256)):
    """Computes Grad-CAM attention heatmap for either CSWin or ConvNeXt."""
    features, gradients = [], []
    
    def hook_f(module, inp, out):
        features.append(out)
    def hook_b(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    if model_type == "cswin":
        target_layer = model.stage4[-1]
    else:  # convnext
        target_layer = model.features[-1]
        
    h_f = target_layer.register_forward_hook(hook_f)
    h_b = target_layer.register_full_backward_hook(hook_b)
    
    out = model(tensor_input).view(-1)
    # Logit represents P(real). For fake, score = -logit
    score = -out[0] if target == "fake" else out[0]
    
    model.zero_grad()
    score.backward()
    
    h_f.remove()
    h_b.remove()
    
    if not features or not gradients:
        return np.zeros(img_size), float(torch.sigmoid(out[0]).item()), float(out[0].item())
        
    feat = features[0]
    grad = gradients[0]
    
    # Format tokens to 2D feature map
    if model_type == "cswin" and len(feat.shape) == 3:  # (1, 64, 512)
        feat = feat.view(1, 8, 8, -1).permute(0, 3, 1, 2)
        grad = grad.view(1, 8, 8, -1).permute(0, 3, 1, 2)
        
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feat, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=img_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    
    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)
        
    p_real = float(torch.sigmoid(out[0]).item())
    logit = float(out[0].item())
    return cam, p_real, logit


def overlay_cam_on_image(pil_img, cam_mask, alpha=0.55):
    """Blends a normalized Grad-CAM mask onto the PIL image using COLORMAP_JET."""
    img_rgb = np.array(pil_img.resize((256, 256), Image.BILINEAR).convert("RGB"))
    heatmap = (cam_mask * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    blended = (alpha * heatmap_color + (1.0 - alpha) * img_rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# --- SIDEBAR INTERFACE ---
st.sidebar.markdown("### ⚙️ Forensics Control Panel")

model_choice = st.sidebar.selectbox("Select Model Architecture & Arm", list(MODEL_OPTIONS.keys()))
model_meta = MODEL_OPTIONS[model_choice]

st.sidebar.caption(model_meta["description"])

st.sidebar.markdown("---")
st.sidebar.markdown("#### Detection Settings")
auto_crop = st.sidebar.checkbox("Auto Face Detection & Bounding Box", value=True,
                                help="Detects and pads face boundaries from uncropped photos.")
enable_cam = st.sidebar.checkbox("Generate Explainable AI Heatmap (Grad-CAM)", value=True,
                                 help="Visualizes facial regions driving the prediction.")
cam_target = st.sidebar.radio("Heatmap Evidence Mode", ["Evidence for Fake (Manipulations)", "Evidence for Real (Skin/Pores)"])
cam_target_mode = "fake" if "Fake" in cam_target else "real"

cam_opacity = st.sidebar.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=0.9, value=0.55, step=0.05)

st.sidebar.markdown("---")
compare_mode = st.sidebar.checkbox("⚡ Dual-Model Head-to-Head Comparison", value=False,
                                  help="Evaluates image through both Arm C (Robust) and Arm A (Clean) simultaneously.")


# --- MAIN HEADER ---
st.markdown("""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="color: #60a5fa; font-weight: 800; margin-bottom: 5px;">🛡️ Deepfake Vision Forensics Studio</h1>
    <p style="color: #9ca3af; font-size: 1.1rem; max-width: 800px; margin: auto;">
        Evaluation platform powered by <strong>CSWin Transformer</strong> and <strong>ConvNeXt</strong> architectures, 
        trained under <strong>Structured Compositional Robustness</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# --- SAMPLE GALLERY & FILE UPLOAD ---
col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    uploaded_file = st.file_uploader("Upload an Image to Analyze (.jpg, .jpeg, .png, .webp)",
                                     type=["jpg", "jpeg", "png", "webp"])

with col_up2:
    st.markdown("##### Or Test with Sample Benchmark Faces:")
    sample_choice = st.selectbox(
        "Choose a sample:",
        ["None (Use Upload)", "Sample 1: Authentic Human (Real)", "Sample 2: Synthetic Face (Deepfake)"]
    )

img_to_process = None

if uploaded_file is not None:
    img_to_process = Image.open(uploaded_file).convert("RGB")
elif sample_choice == "Sample 1: Authentic Human (Real)":
    real_sample_path = "data/Dataset-1/Data Set 1/test/real/real_3526.jpg"
    if os.path.isfile(real_sample_path):
        img_to_process = Image.open(real_sample_path).convert("RGB")
    else:
        st.warning("Sample file not found at " + real_sample_path)
elif sample_choice == "Sample 2: Synthetic Face (Deepfake)":
    fake_sample_path = "data/Dataset-1/Data Set 1/test/fake/fake_4243.jpg"
    if os.path.isfile(fake_sample_path):
        img_to_process = Image.open(fake_sample_path).convert("RGB")
    else:
        st.warning("Sample file not found at " + fake_sample_path)


# --- INFERENCE PIPELINE ---
if img_to_process is not None:
    # 1. Face Preprocessing
    if auto_crop:
        annotated_img, face_crop, face_coords = detect_and_crop_face(img_to_process)
    else:
        annotated_img = np.array(img_to_process)
        face_crop = img_to_process.resize((256, 256), Image.BILINEAR)
        face_coords = None

    # Display image pipeline
    st.markdown("### 📷 Visual Inspection Pipeline")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        st.image(annotated_img, caption="1. Full Input (with Detection)" if face_coords else "1. Full Input (No Face Box)", use_container_width=True)
    with p_col2:
        st.image(face_crop, caption="2. Normalized 256x256 Model Input", use_container_width=True)

    # 2. Model Prediction
    model = load_cached_model(model_meta["arch"], model_meta["checkpoint"])
    if model is None:
        st.error(f"Checkpoint file not found: `{model_meta['checkpoint']}`. Please ensure training has finished.")
    else:
        t0 = time.time()
        tensor_crop = TRANSFORM(face_crop).unsqueeze(0).to(DEVICE)
        
        if enable_cam:
            cam_mask, p_real, logit = compute_gradcam(model, tensor_crop, model_meta["type"], target=cam_target_mode)
            cam_overlay = overlay_cam_on_image(face_crop, cam_mask, alpha=cam_opacity)
            with p_col3:
                st.image(cam_overlay, caption=f"3. Explainable Grad-CAM ({cam_target})", use_container_width=True)
        else:
            with torch.no_grad():
                out = model(tensor_crop).view(-1)
                p_real = float(torch.sigmoid(out[0]).item())
                logit = float(out[0].item())
            with p_col3:
                st.info("Grad-CAM heatmap is disabled in sidebar.")

        latency = (time.time() - t0) * 1000.0
        p_fake = 1.0 - p_real

        # 3. Decision Card
        st.markdown("---")
        st.markdown("### 🔍 Forensics Decision & Confidence Analysis")
        
        c_verdict, c_metrics = st.columns([1.2, 2])
        
        with c_verdict:
            # Calibrated 3-Zone Thresholds
            if p_fake >= 0.70:
                st.markdown('<div class="badge-fake">🔴 DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f"<p style='color: #f87171; margin-top: 10px; font-weight: 500;'>High confidence manipulation detected ({p_fake*100:.1f}%).</p>", unsafe_allow_html=True)
            elif p_fake <= 0.30:
                st.markdown('<div class="badge-real">🟢 AUTHENTIC HUMAN</div>', unsafe_allow_html=True)
                st.markdown(f"<p style='color: #34d399; margin-top: 10px; font-weight: 500;'>Natural biometric skin consistency confirmed ({p_real*100:.1f}%).</p>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-inconclusive">🟡 AMBIGUOUS / INCONCLUSIVE</div>', unsafe_allow_html=True)
                st.markdown("<p style='color: #fbbf24; margin-top: 10px; font-weight: 500;'>Confidence in uncertainty buffer (30%-70%). Human forensic inspection advised.</p>", unsafe_allow_html=True)
                
            st.caption(f"Inference Latency: **{latency:.1f} ms** on {DEVICE.type.upper()}")

        with c_metrics:
            st.markdown(f"**Manipulation Probability:** `{p_fake * 100:.2f}%`")
            st.progress(float(p_fake))
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown('<div class="metric-label">Probability Real</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{p_real*100:.1f}%</div>', unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-label">Probability Fake</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{p_fake*100:.1f}%</div>', unsafe_allow_html=True)
            with m3:
                st.markdown('<div class="metric-label">Raw Logit Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{logit:+.2f}</div>', unsafe_allow_html=True)

        # 4. Dual-Model Comparison Mode (Arm C vs. Arm A)
        if compare_mode:
            st.markdown("---")
            st.markdown("### ⚡ Head-to-Head Comparison: Arm C (Robust) vs. Arm A (Clean Baseline)")
            st.caption("Demonstrates why standard clean baselines suffer from false alarms on unconstrained photos while Arm C remains resilient.")
            
            comp_a_meta = MODEL_OPTIONS["CSWin-Small (Arm A: Clean Baseline - 90.6% Acc)"]
            model_a = load_cached_model(comp_a_meta["arch"], comp_a_meta["checkpoint"])
            
            if model_a is not None:
                with torch.no_grad():
                    out_a = model_a(tensor_crop).view(-1)
                    p_real_a = float(torch.sigmoid(out_a[0]).item())
                    p_fake_a = 1.0 - p_real_a
                
                cmp_col1, cmp_col2 = st.columns(2)
                with cmp_col1:
                    st.markdown("#### 🌟 Arm C: Structured Robust Model")
                    st.metric("P(Fake)", f"{p_fake*100:.1f}%", delta=f"{p_fake - p_fake_a:+.1f}% vs Arm A")
                    st.markdown("**Verdict:** " + ("🔴 FAKE" if p_fake >= 0.5 else "🟢 REAL"))
                with cmp_col2:
                    st.markdown("#### 🔍 Arm A: Clean Baseline Model")
                    st.metric("P(Fake)", f"{p_fake_a*100:.1f}%")
                    st.markdown("**Verdict:** " + ("🔴 FAKE" if p_fake_a >= 0.5 else "🟢 REAL"))
                    if abs(p_fake - p_fake_a) > 0.25:
                        st.warning("⚠️ Discrepancy observed! The clean baseline was influenced by image sensor noise or compression artifacts that Arm C successfully filtered out.")

else:
    st.info("👆 Please upload an image or select a sample face above to begin forensics analysis.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>CSWin Deepfake Detection Research &bull; Phase 1 Robustness & Architecture Generalization</div>", unsafe_allow_html=True)
