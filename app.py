import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time
import os
import json

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Railway Fault Detector",
    page_icon="🚂",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap");

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: "Inter", sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    min-height: 100vh;
}

.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
}

.hero-header p { color: #94a3b8; font-size: 1rem; margin: 0; }

.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.08) 100%);
    border: 1.5px solid rgba(16,185,129,0.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: fadeSlideIn 0.5s ease-out;
}

.result-danger {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%);
    border: 1.5px solid rgba(239,68,68,0.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: fadeSlideIn 0.5s ease-out;
}

.result-title { font-size: 1.9rem; font-weight: 800; margin: 0.4rem 0; }
.result-safe .result-title  { color: #34d399; }
.result-danger .result-title { color: #f87171; }

.result-subtitle { font-size: 0.95rem; color: #94a3b8; margin: 0; }

.confidence-label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #64748b; margin: 1.2rem 0 0.3rem 0;
}

.confidence-value { font-size: 2.5rem; font-weight: 700; margin: 0; }
.result-safe  .confidence-value { color: #6ee7b7; }
.result-danger .confidence-value { color: #fca5a5; }

.bar-wrap { background: rgba(255,255,255,0.06); border-radius: 99px; height: 8px; margin-top: 0.5rem; overflow: hidden; }
.bar-fill-safe   { background: linear-gradient(90deg, #059669, #34d399); height: 8px; border-radius: 99px; }
.bar-fill-danger { background: linear-gradient(90deg, #dc2626, #f87171); height: 8px; border-radius: 99px; }

[data-testid="stSidebar"] { background: rgba(15,12,41,0.85); border-right: 1px solid rgba(99,102,241,0.15); }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
# NOTE on class order: this MUST match tf.keras.utils.image_dataset_from_directory's
# alphabetical folder ordering used at training time: ['Defective', 'Non defective'].
# The model's single sigmoid output is P(class index 1) = P("Non defective").
CLASS_NAMES = ["Defective", "Non defective"]
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# Load model config produced by train_best.py (if available)
_cfg_path = os.path.join(BASE_DIR, "model_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    _has_new_model = os.path.exists(os.path.join(BASE_DIR, _cfg["model_file"]))
else:
    _cfg = {}
    _has_new_model = False

MODEL_OPTIONS = {}
if _has_new_model:
    MODEL_OPTIONS["🏆 Best Final Model (EfficientNetV2S, trained)"] = _cfg["model_file"]
MODEL_OPTIONS["Best Model (EfficientNetB3)"] = "best_model.keras"

# IMG_SIZE per model (new model uses 260x260; best_model.keras uses 225x225)
MODEL_SIZES = {
    _cfg.get("model_file", ""): tuple(_cfg["img_size"]) if "img_size" in _cfg else (260, 260),
    "best_model.keras": (224, 224),
}

# Decision thresholds tuned on the validation set (see notebook: "Best validation
# threshold" cell). These are the actual class-split points, NOT just a "low
# confidence" cosmetic cutoff — using 0.5 here instead of the tuned value is what
# was systematically pushing borderline "Non defective" images into "Defective".
TUNED_THRESHOLDS = {
    "best_model.keras": 0.50,
}


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return tf.keras.models.load_model(path)


def preprocess_image(img_pil: Image.Image, size: tuple, crop_aspect: bool = True) -> np.ndarray:
    """Mirror the training pipeline with optional aspect-ratio center cropping:
    - correct EXIF rotation (phone photos carry orientation tags PIL won't auto-apply)
    - resize with bilinear interpolation, preserving aspect ratio if crop_aspect is True
    - keep raw [0, 255] float values (EfficientNet's Rescaling/Normalization layers
      are baked into the model itself — do NOT divide by 255 here)
    """
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    if crop_aspect:
        img_resized = ImageOps.fit(img_pil, size, method=Image.Resampling.BILINEAR)
    else:
        img_resized = img_pil.resize(size, Image.Resampling.BILINEAR)
        
    arr = np.array(img_resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Model Selection")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    model_path  = MODEL_OPTIONS[model_label]
    img_size    = MODEL_SIZES.get(model_path, (224, 224))

    is_new_model    = _has_new_model and model_path == _cfg.get("model_file", "")
    default_thresh  = float(_cfg.get("threshold", 0.5)) if is_new_model else TUNED_THRESHOLDS.get(model_path, 0.5)

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider(
        "Decision threshold",
        0.20, 0.80, default_thresh, 0.01,
        help=(
            "This is the actual cutoff used to decide Defective vs Non defective "
            "(tuned on the validation set), not just a warning label."
        ),
    )
    crop_aspect = st.checkbox(
        "Preserve Aspect Ratio (Center Crop)", value=True,
        help="Prevents squashing rectangular photos into squares during model resize.",
    )
    calibrate_conf = st.checkbox(
        "Probability Calibration (Threshold Scaling)", value=True,
        help="Scales raw model probabilities relative to the decision boundary to reflect true prediction certainty.",
    )
    show_raw = st.checkbox("Show raw probabilities", value=False)

    st.markdown("---")
    st.markdown("## 📋 About")
    st.markdown("""
This app uses a CNN model trained on railway track images to detect structural defects.

**Classes:**
- 🔴 Defective
- 🟢 Non defective
""")

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div style="font-size:3rem; margin-bottom:0.3rem;">🚂</div>
    <h1>Railway Fault Detector</h1>
    <p>AI-powered track inspection — upload an image to detect defects instantly</p>
</div>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────
model_file = os.path.join(BASE_DIR, model_path)

if not os.path.exists(model_file):
    st.error(f"Model file not found: `{model_path}`. Run `train_best.py` first or choose another model.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model(model_file)

st.success(f"Model loaded — **{model_label}**  |  Input: {img_size[0]}×{img_size[1]}", icon="✅")

# ─── Upload Section ───────────────────────────────────────────────────────────
st.markdown("### 📷 Upload Track Image")

uploaded = st.file_uploader(
    "Drag & drop or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    img_pil_raw = Image.open(uploaded)

    col_img, col_info = st.columns([1, 1], gap="large")
    with col_img:
        st.markdown("**Preview**")
        st.image(ImageOps.exif_transpose(img_pil_raw), use_container_width=True, caption=uploaded.name)

    with col_info:
        st.markdown("**Image Details**")
        st.markdown(f"""
| Property  | Value |
|-----------|-------|
| Filename  | `{uploaded.name}` |
| Size      | `{img_pil_raw.width} × {img_pil_raw.height} px` |
| File size | `{uploaded.size / 1024:.1f} KB` |
""")

    st.markdown("---")

    # ── Predict ──────────────────────────────────────────────────────────────
    with st.spinner("Analysing image..."):
        img_batch = preprocess_image(img_pil_raw, img_size, crop_aspect)

        start_t = time.time()
        raw_pred = model.predict(img_batch, verbose=0)
        elapsed = time.time() - start_t

    # ── Decide class using the ACTUAL decision threshold ────────────────────
    # This is the core fix: the class must be decided against confidence_threshold,
    # not a hardcoded 0.5 argmax, or the slider does nothing and the tuned threshold
    # from the notebook is silently ignored.
    if raw_pred.shape[-1] == 1:
        prob_non_defective = float(raw_pred[0][0])
        prob_defective     = 1.0 - prob_non_defective
        probs = [prob_defective, prob_non_defective]

        is_defective = prob_non_defective < confidence_threshold
        pred_idx     = 0 if is_defective else 1

        if calibrate_conf:
            eps = 1e-7
            p_clamped = float(np.clip(prob_non_defective, eps, 1.0 - eps))
            t_clamped = float(np.clip(confidence_threshold, eps, 1.0 - eps))
            logit_p = float(np.log(p_clamped / (1.0 - p_clamped)))
            logit_t = float(np.log(t_clamped / (1.0 - t_clamped)))
            scaled_logit = (logit_p - logit_t) / 0.45
            calibrated_p = float(1.0 / (1.0 + np.exp(-scaled_logit)))
            confidence = (1.0 - calibrated_p) if is_defective else calibrated_p
        else:
            confidence = prob_defective if is_defective else prob_non_defective
    else:
        probs      = raw_pred[0].tolist()
        pred_idx   = int(np.argmax(probs))
        confidence = float(max(probs))

    pred_class = CLASS_NAMES[pred_idx]
    # "Uncertain" now means the score is close to the decision boundary, not just
    # below an arbitrary confidence bar.
    uncertain = abs(probs[1] - confidence_threshold) < 0.08

    # ── Result Card ──────────────────────────────────────────────────────────
    st.markdown("### 🔬 Detection Result")

    if uncertain:
        st.warning(f"⚠️ This prediction is close to the decision boundary ({confidence*100:.1f}% confidence). Consider a clearer image.")

    is_defective = (pred_idx == 0)
    card_class   = "result-danger" if is_defective else "result-safe"
    icon         = "⚠️" if is_defective else "✅"
    fill_class   = "bar-fill-danger" if is_defective else "bar-fill-safe"
    bar_pct      = f"{confidence*100:.1f}%"
    msg          = "Structural fault detected — immediate inspection recommended." if is_defective else "Track appears to be in good condition."

    st.markdown(f"""
<div class="{card_class}">
    <div style="font-size:3rem">{icon}</div>
    <div class="result-title">{pred_class}</div>
    <div class="result-subtitle">{msg}</div>
    <div class="confidence-label">Confidence</div>
    <div class="confidence-value">{confidence*100:.1f}%</div>
    <div class="bar-wrap"><div class="{fill_class}" style="width:{bar_pct}"></div></div>
    <p style="color:#64748b; font-size:0.8rem; margin-top:1rem;">Inference time: {elapsed*1000:.1f} ms</p>
</div>
""", unsafe_allow_html=True)

    if show_raw:
        st.markdown("#### 📊 Class Probabilities")
        for cls, p in zip(CLASS_NAMES, probs):
            st.markdown(f"**{cls}:** {p*100:.2f}%")
            st.progress(p)
        st.caption(f"Raw model sigmoid output (P(\"Non defective\")): {probs[1]:.4f} — decision threshold: {confidence_threshold:.2f}")

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", pred_class)
    m2.metric("Confidence", f"{confidence*100:.1f}%")
    m3.metric("Inference",  f"{elapsed*1000:.0f} ms")

else:
    st.markdown("""
<div style="
    border: 2px dashed rgba(99,102,241,0.3); border-radius: 16px;
    padding: 3rem 2rem; text-align: center; color: #475569; margin-top: 1rem;
">
    <div style="font-size:3rem">📸</div>
    <p style="font-size:1.1rem; font-weight:500; margin:0.5rem 0">No image uploaded yet</p>
    <p style="font-size:0.9rem; margin:0">Upload a railway track photo above to run fault detection</p>
</div>
""", unsafe_allow_html=True)