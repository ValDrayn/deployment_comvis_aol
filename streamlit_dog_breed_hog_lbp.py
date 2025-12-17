import streamlit as st
import cv2
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern

st.set_page_config(
    page_title="Dog Breed Detector",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2e86c1;
    }
    .stAlert {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "dog_model_hog_lbp_augmented.pkl"
IMG_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

@st.cache_resource
def load_model():
    try:
        data = joblib.load(MODEL_PATH)
        return data["svm"], data["scaler"], data["classes"]
    except FileNotFoundError:
        st.error(f"❌ File '{MODEL_PATH}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None, None, None

svm, scaler, classes = load_model()

def extract_feature_with_vis(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    feature = np.hstack([hog_feat, lbp_hist])

    lbp_vis = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    lbp_vis = cv2.cvtColor(lbp_vis.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return feature, lbp_vis

def process_prediction(feat):
    t0 = time.time()

    scores = svm.decision_function(feat)[0]
    scores = scores.astype(np.float64)
    
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    infer_ms = (time.time() - t0) * 1000
    best_margin = np.max(scores) 

    results = {
        classes[i]: float(probs[i])
        for i in range(len(classes))
    }
    
    return results, best_margin, infer_ms

def run_benchmark_logic(feat, n_runs):
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        svm.decision_function(feat)
        times.append((time.time() - t0) * 1000)
    return np.mean(times), np.std(times), times

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/616/616408.png",
        width=90
    )

    st.title("Settings")

    st.markdown(
        f"""
        <div style="
            padding: 14px;
            border-radius: 12px;
            background: linear-gradient(135deg, #e0f2fe, #f8fafc);
            border: 1px solid #bae6fd;
            margin-bottom: 12px;
        ">
            <h4 style="margin-top: 0;">Fixed Parameters</h4>
            <ul style="margin: 0; padding-left: 5px;">
                <li><b>Image Size</b>: {IMG_SIZE}</li>
                <li><b>LBP Radius</b>: {LBP_RADIUS}</li>
                <li><b>Model</b>: SVM + HOG + LBP</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )



st.title("🐶 Dog Breed Classifier")
st.markdown("Analisis ras anjing menggunakan *Classical Computer Vision* (HOG + LBP) dan SVM.")

if svm is None:
    st.stop()

tab_detect, tab_bench = st.tabs(["🖼️ Single Detection", "⚡ Performance Benchmark"])

with tab_detect:
    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        feat_vector = None
        vis_image = None
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, caption="Input Image", use_container_width=True, channels="RGB")
            
            with st.spinner("Extracting HOG & LBP features..."):
                raw_feat, vis_image = extract_feature_with_vis(img_bgr)
                feat_vector = scaler.transform(raw_feat.reshape(1, -1))

    with col_result:
        st.subheader("2. Analysis Results")
        
        if feat_vector is not None:
            if st.button("🔍 Identify Breed", type="primary", use_container_width=True):
                
                results, margin, infer_ms = process_prediction(feat_vector)
                
                top_class = max(results, key=results.get)
                top_prob = results[top_class]

                st.success(f"**Prediction:** {top_class}")
                
                m1, m2 = st.columns(2)
                m1.metric("Confidence", f"{top_prob*100:.1f}%")
                m2.metric("Inference Time", f"{infer_ms:.2f} ms")

                st.markdown("##### Probability Distribution")
                df_res = pd.DataFrame(list(results.items()), columns=["Breed", "Probability"])
                df_res = df_res.sort_values(by="Probability", ascending=True)
                
                st.bar_chart(df_res.set_index("Breed"), color="#2e86c1", height=250)

                with st.expander("🔬 View Feature Visualization (LBP)"):
                    st.image(vis_image, caption="Local Binary Pattern (Texture Feature)", use_container_width=True)
                    st.caption("Gambar ini merepresentasikan tekstur yang dilihat oleh model.")
        else:
            st.info("👈 Silakan upload gambar di panel kiri untuk memulai analisis.")

with tab_bench:
    st.subheader("System Performance Test")
    st.markdown("Menguji stabilitas waktu inferensi model pada mesin ini.")
    
    b_col1, b_col2 = st.columns([1, 2])
    
    with b_col1:
        n_runs = st.slider("Number of Loops", 10, 200, 20)
        run_btn = st.button("🚀 Start Benchmark", type="primary")

    with b_col2:
        if run_btn:
            if feat_vector is None:
                st.warning("⚠️ Harap upload gambar di tab 'Single Detection' terlebih dahulu untuk mengekstrak fitur.")
            else:
                with st.spinner(f"Running prediction {n_runs} times..."):
                    mean_t, std_t, times = run_benchmark_logic(feat_vector, n_runs)
                
                st.markdown(f"""
                | Metric | Value |
                | :--- | :--- |
                | **Avg Latency** | `{mean_t:.2f} ms` |
                | **Stability (Std)** | `± {std_t:.2f} ms` |
                | **Total Time** | `{sum(times)/1000:.3f} s` |
                """)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(times, marker='o', markersize=4, linestyle='-', color='#2e86c1', alpha=0.7)
                ax.axhline(mean_t, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_t:.2f}ms')
                ax.fill_between(
                    range(len(times)), 
                    mean_t - std_t, 
                    mean_t + std_t, 
                    color='red', 
                    alpha=0.1, 
                    label='Std Dev Range (Stability)'
                )
                
                ax.set_title("Inference Stability Check")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Time (ms)")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                
                st.pyplot(fig)