import os
import io
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from huggingface_hub import hf_hub_download

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background-color: #080C14;
    color: #E8EDF5;
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1200px;
}
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}
/* ── Hero Header ── */
.hero-header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 2.5rem 0 1rem;
    border-bottom: 1px solid rgba(99, 179, 237, 0.15);
    margin-bottom: 2.5rem;
}
.hero-icon {
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, #1A3A5C, #0D5E8A);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    flex-shrink: 0;
    box-shadow: 0 0 30px rgba(29, 119, 178, 0.4);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #E8EDF5 0%, #63B3ED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.2rem;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    color: #6B8BA4;
    margin: 0;
    letter-spacing: 0.03em;
}
.hero-badge {
    margin-left: auto;
    background: rgba(99, 179, 237, 0.08);
    border: 1px solid rgba(99, 179, 237, 0.2);
    color: #63B3ED;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
}
/* ── Cards ── */
.card {
    background: #0D1520;
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63B3ED;
    margin: 0 0 1rem;
}
/* ── Model Selector ── */
.stSelectbox label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #63B3ED !important;
}
.stSelectbox > div > div {
    background: #111B28 !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(99, 179, 237, 0.5) !important;
}
/* ── File Uploader ── */
.stFileUploader label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #63B3ED !important;
}
[data-testid="stFileUploader"] section {
    background: #111B28 !important;
    border: 2px dashed rgba(99, 179, 237, 0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
    padding: 2rem !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99, 179, 237, 0.5) !important;
}
[data-testid="stFileUploader"] section p {
    color: #6B8BA4 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(99, 179, 237, 0.1) !important;
    border: 1px solid rgba(99, 179, 237, 0.3) !important;
    color: #63B3ED !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
}
/* ── Remove button fix — prevent popup ── */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button,
button[title="Remove file"] {
    all: unset !important;
    cursor: pointer !important;
    color: #FC8181 !important;
    font-size: 0.75rem !important;
    padding: 0.2rem 0.5rem !important;
    border-radius: 6px !important;
    background: rgba(252,129,129,0.08) !important;
    border: 1px solid rgba(252,129,129,0.2) !important;
    display: inline-flex !important;
    align-items: center !important;
    transition: background 0.2s !important;
}
button[title="Remove file"]:hover {
    background: rgba(252,129,129,0.18) !important;
}
/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(99, 179, 237, 0.15) !important;
}
/* ── Result Boxes ── */
.result-box {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.result-pneumonia {
    background: rgba(254, 178, 178, 0.06);
    border: 1px solid rgba(254, 178, 178, 0.25);
}
.result-pneumonia::before { background: linear-gradient(90deg, #FC8181, #F56565); }
.result-normal {
    background: rgba(154, 230, 180, 0.06);
    border: 1px solid rgba(154, 230, 180, 0.25);
}
.result-normal::before { background: linear-gradient(90deg, #68D391, #48BB78); }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
}
.result-pneumonia .result-label { color: #FC8181; }
.result-normal .result-label { color: #68D391; }
.result-confidence {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 300;
    color: #6B8BA4;
    margin: 0 0 1.2rem;
}
/* ── Confidence Bar ── */
.conf-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.conf-bar-fill-pneumonia {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #FC8181, #F56565);
}
.conf-bar-fill-normal {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #68D391, #48BB78);
}
.conf-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
}
.result-pneumonia .conf-pct { color: #FC8181; }
.result-normal .conf-pct { color: #68D391; }
/* ── Info Grid ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.info-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
.info-chip-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4A6880;
    margin: 0 0 0.2rem;
}
.info-chip-value {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #C8D8E8;
    margin: 0;
}
/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255, 214, 0, 0.04);
    border: 1px solid rgba(255, 214, 0, 0.15);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 1.2rem;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
}
.disclaimer-icon { font-size: 0.95rem; flex-shrink: 0; margin-top: 1px; }
.disclaimer-text {
    font-size: 0.78rem;
    color: #8A9BAB;
    line-height: 1.5;
    font-style: italic;
}
/* ── Clear button ── */
.stButton > button {
    background: rgba(252,129,129,0.08) !important;
    border: 1px solid rgba(252,129,129,0.25) !important;
    color: #FC8181 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.8rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: rgba(252,129,129,0.18) !important;
}
.stSpinner > div { border-top-color: #63B3ED !important; }
hr { border-color: rgba(99, 179, 237, 0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ── Thresholds ─────────────────────────────────────────────────────────────────
CNN_THRESHOLD = 0.5
TL_THRESHOLD  = 0.57


# ── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)

    cnn_path = hf_hub_download(
        repo_id="Tejas-ML/pneumonia-detection-models",
        filename="pneumonia_detection_model.keras",
        local_dir="models"
    )
    tl_path = hf_hub_download(
        repo_id="Tejas-ML/pneumonia-detection-models",
        filename="transfer_learning_model.keras",
        local_dir="models"
    )

    cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
    tl_model  = tf.keras.models.load_model(tl_path,  compile=False, safe_mode=False)
    return cnn_model, tl_model


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, model_type: str) -> np.ndarray:
    img = img.resize((224, 224))
    if model_type == "Custom CNN (Grayscale)":
        img = img.convert("L")
        arr = np.array(img)[..., np.newaxis]
    else:
        img = img.convert("RGB")
        arr = np.array(img)
    return np.expand_dims(arr, axis=0)


# ── Session state for image ────────────────────────────────────────────────────
# Fix: store image bytes in session state so clearing works without popup
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "img_name" not in st.session_state:
    st.session_state.img_name = None
if "img_size" not in st.session_state:
    st.session_state.img_size = None


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🫁</div>
    <div>
        <p class="hero-title">PneumoScan AI</p>
        <p class="hero-sub">Deep Learning · Chest X-Ray Analysis · Clinical Decision Support</p>
    </div>
    <div class="hero-badge">Research Tool</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.1], gap="large")

with left:
    model_choice = st.selectbox(
        "Select Model",
        ["Custom CNN (Grayscale)", "Transfer Learning (RGB)"],
        help="Custom CNN uses grayscale input. Transfer Learning (DenseNet121) uses RGB."
    )

    if model_choice == "Custom CNN (Grayscale)":
        arch, input_fmt, desc = "Custom CNN", "224×224 Grayscale", "Trained from scratch on chest X-ray dataset."
    else:
        arch, input_fmt, desc = "DenseNet121", "224×224 RGB", "Fine-tuned DenseNet121 pretrained on ImageNet."

    st.markdown(f"""
    <div class="card" style="margin-top:0.8rem;">
        <p class="card-title">Model Details</p>
        <div class="info-grid" style="margin-top:0;">
            <div class="info-chip">
                <p class="info-chip-label">Architecture</p>
                <p class="info-chip-value">{arch}</p>
            </div>
            <div class="info-chip">
                <p class="info-chip-label">Input Format</p>
                <p class="info-chip-value">{input_fmt}</p>
            </div>
        </div>
        <p style="font-size:0.8rem;color:#4A6880;margin:0.8rem 0 0;font-style:italic;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ── File uploader — read bytes immediately to avoid double-read bug ────────
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray",
        type=["jpg", "jpeg", "png"],
        help="Accepts JPG or PNG chest X-ray images.",
        key="file_uploader"
    )

    # Store bytes in session state as soon as file is uploaded
    if uploaded_file is not None:
        st.session_state.img_bytes = uploaded_file.read()   # read once here
        st.session_state.img_name  = uploaded_file.name
        img_for_display = Image.open(io.BytesIO(st.session_state.img_bytes))
        st.session_state.img_size  = img_for_display.size

    # Manual clear button — fixes the popup issue when clicking ✕
    if st.session_state.img_bytes is not None:
        st.image(
            Image.open(io.BytesIO(st.session_state.img_bytes)),
            use_container_width=True,
            caption=""
        )
        st.markdown(f"""
        <p style="font-size:0.75rem;color:#4A6880;text-align:center;margin-top:0.4rem;">
            {st.session_state.img_name} &nbsp;·&nbsp;
            {st.session_state.img_size[0]}×{st.session_state.img_size[1]}px
        </p>
        """, unsafe_allow_html=True)

        if st.button("✕ Remove image"):
            st.session_state.img_bytes = None
            st.session_state.img_name  = None
            st.session_state.img_size  = None
            st.rerun()


# ── Right column ───────────────────────────────────────────────────────────────
with right:
    if st.session_state.img_bytes is None:
        st.markdown("""
        <div style="
            height: 380px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 2px dashed rgba(99,179,237,0.1);
            border-radius: 16px;
            color: #2E4A60;
            text-align: center;
            padding: 2rem;
        ">
            <div style="font-size:3.5rem;margin-bottom:1rem;opacity:0.4;">🔬</div>
            <p style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;
                      color:#2E4A60;margin:0 0 0.4rem;">No image uploaded</p>
            <p style="font-size:0.82rem;color:#1E3245;max-width:220px;line-height:1.5;margin:0;">
                Upload a chest X-ray on the left to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        cnn_model, tl_model = load_models()

        # Use buffered bytes — no second file.read() needed
        img = Image.open(io.BytesIO(st.session_state.img_bytes))
        arr = preprocess(img, model_choice)

        with st.spinner("Analysing image…"):
            time.sleep(0.3)
            model     = cnn_model if model_choice == "Custom CNN (Grayscale)" else tl_model
            threshold = CNN_THRESHOLD if model_choice == "Custom CNN (Grayscale)" else TL_THRESHOLD
            prediction = model.predict(arr, verbose=0)

        prob         = float(prediction[0][0])
        is_pneumonia = prob > threshold
        confidence   = prob if is_pneumonia else 1 - prob
        conf_pct     = confidence * 100

        if is_pneumonia:
            st.markdown(f"""
            <div class="result-box result-pneumonia">
                <p class="result-label">⚠ Pneumonia Detected</p>
                <p class="result-confidence">Model confidence score</p>
                <p class="conf-pct">{conf_pct:.1f}%</p>
                <div class="conf-bar-track" style="margin-top:0.8rem;">
                    <div class="conf-bar-fill-pneumonia" style="width:{conf_pct:.1f}%"></div>
                </div>
                <div class="info-grid">
                    <div class="info-chip">
                        <p class="info-chip-label">Raw Probability</p>
                        <p class="info-chip-value">{prob:.4f}</p>
                    </div>
                    <div class="info-chip">
                        <p class="info-chip-label">Classification</p>
                        <p class="info-chip-value">PNEUMONIA</p>
                    </div>
                </div>
            </div>
            
            <div class="card" style="margin-top:1.2rem;border-color:rgba(252,129,129,0.2);">
                <p class="card-title" style="color:#FC8181;">⚕ Recommended Precautions</p>
                <p style="font-size:0.82rem;color:#6B8BA4;margin:0 0 1rem;font-style:italic;">
                    Pneumonia indicators detected. Please consult a licensed physician or visit an emergency room immediately. The following are general precautions only and do not replace professional medical advice.
                </p>
                <div style="display:flex;flex-direction:column;gap:0.8rem;">
                    <div class="info-chip" style="border-left:3px solid #FC8181;">
                        <p class="info-chip-label" style="font-size:0.72rem;">01 &nbsp;·&nbsp; SEEK IMMEDIATE MEDICAL ATTENTION</p>
                        <p style="font-size:0.85rem;color:#C8D8E8;margin:0;line-height:1.6;">Visit a doctor, clinic, or emergency room as soon as possible. Pneumonia requires professional diagnosis and treatment — which may include antibiotics for bacterial pneumonia or antivirals for viral pneumonia. Do not self-medicate.</p>
                    </div>
                    <div class="info-chip" style="border-left:3px solid #FC8181;">
                        <p class="info-chip-label" style="font-size:0.72rem;">02 &nbsp;·&nbsp; REST AND LIMIT PHYSICAL ACTIVITY</p>
                        <p style="font-size:0.85rem;color:#C8D8E8;margin:0;line-height:1.6;">Complete bed rest is essential. Your immune system needs energy to fight the infection — avoid all strenuous physical activity, work, or exercise until you have fully recovered and been cleared by a doctor.</p>
                    </div>
                    <div class="info-chip" style="border-left:3px solid #FC8181;">
                        <p class="info-chip-label" style="font-size:0.72rem;">03 &nbsp;·&nbsp; MAINTAIN ADEQUATE HYDRATION</p>
                        <p style="font-size:0.85rem;color:#C8D8E8;margin:0;line-height:1.6;">Drink plenty of water, warm broths, and herbal teas. Staying well hydrated helps thin mucus secretions, ease breathing, reduce fever, and prevent dehydration — a common complication of respiratory infections.</p>
                    </div>
                    <div class="info-chip" style="border-left:3px solid #FC8181;">
                        <p class="info-chip-label" style="font-size:0.72rem;">04 &nbsp;·&nbsp; ISOLATE AND PREVENT TRANSMISSION</p>
                        <p style="font-size:0.85rem;color:#C8D8E8;margin:0;line-height:1.6;">Pneumonia can be contagious depending on its cause. Wear a mask in shared spaces, cover coughs and sneezes with a tissue, wash hands frequently with soap, and avoid close contact with elderly, infants, or immunocompromised individuals.</p>
                    </div>
                    <div class="info-chip" style="border-left:3px solid #FC8181;">
                        <p class="info-chip-label" style="font-size:0.72rem;">05 &nbsp;·&nbsp; MONITOR BREATHING AND OXYGEN LEVELS</p>
                        <p style="font-size:0.85rem;color:#C8D8E8;margin:0;line-height:1.6;">Use a pulse oximeter if available to monitor blood oxygen saturation. If your SpO₂ drops below 94%, you experience rapid breathing, chest pain, confusion, or bluish lips — seek emergency medical care immediately.</p>
                    </div>
                    <div class="info-chip" style="border-left:3px solid #FC8181;">
                        <p class="info-chip-label" style="font-size:0.72rem;">06 &nbsp;·&nbsp; AVOID LUNG IRRITANTS</p>
                        <p style="font-size:0.85rem;color:#C8D8E8;margin:0;line-height:1.6;">Completely avoid cigarette smoke, air pollution, dust, chemical fumes, and cold dry air. These irritants inflame already compromised lung tissue and can significantly slow recovery or worsen symptoms.</p>
                    </div>
                </div>
            </div>
    
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-normal">
                <p class="result-label">✓ Normal</p>
                <p class="result-confidence">Model confidence score</p>
                <p class="conf-pct">{conf_pct:.1f}%</p>
                <div class="conf-bar-track" style="margin-top:0.8rem;">
                    <div class="conf-bar-fill-normal" style="width:{conf_pct:.1f}%"></div>
                </div>
                <div class="info-grid">
                    <div class="info-chip">
                        <p class="info-chip-label">Raw Probability</p>
                        <p class="info-chip-value">{prob:.4f}</p>
                    </div>
                    <div class="info-chip">
                        <p class="info-chip-label">Classification</p>
                        <p class="info-chip-value">NORMAL</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-grid" style="margin-top:0.8rem;">
            <div class="info-chip">
                <p class="info-chip-label">Model Used</p>
                <p class="info-chip-value">{arch}</p>
            </div>
            <div class="info-chip">
                <p class="info-chip-label">Decision Threshold</p>
                <p class="info-chip-value">{threshold}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <span class="disclaimer-icon">⚠️</span>
            <p class="disclaimer-text">
                This tool is intended for <strong>research and educational purposes only</strong>.
                It is not a substitute for professional medical diagnosis.
                Always consult a qualified radiologist or physician.
            </p>
        </div>
        """, unsafe_allow_html=True)