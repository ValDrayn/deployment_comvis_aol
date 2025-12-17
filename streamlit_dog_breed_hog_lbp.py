import streamlit as st
import cv2
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern

st.set_page_config(
    page_title="Dog Breed Detector + FAST",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #2e86c1; }
    .stAlert { padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "dog_model_hog_lbp_augmented_fast.pkl"
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

def get_roi_with_fast(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold=30)
    keypoints = fast.detect(gray, None)
    
    x, y, w, h = 0, 0, img_bgr.shape[1], img_bgr.shape[0]
    
    if keypoints and len(keypoints) > 10:
        points = np.array([kp.pt for kp in keypoints])
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        pad = 25
        x = int(max(0, x_min - pad))
        y = int(max(0, y_min - pad))
        w = int(min(img_bgr.shape[1], x_max + pad) - x)
        h = int(min(img_bgr.shape[0], y_max + pad) - y)
        
        if w > 40 and h > 40:
            return (x, y, w, h), img_bgr[y:y+h, x:x+w]

    return (x, y, w, h), img_bgr

def extract_feature_with_vis(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True
    )

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    feature = np.hstack([hog_feat, lbp_hist])

    lbp_vis = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    lbp_vis = cv2.cvtColor(lbp_vis.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return feature, lbp_vis

def process_pipeline(img_bgr):
    t0 = time.time()
    
    (x, y, w, h), roi_img = get_roi_with_fast(img_bgr)
    
    raw_feat, vis_img = extract_feature_with_vis(roi_img)
    feat_vector = scaler.transform(raw_feat.reshape(1, -1))
    
    scores = svm.decision_function(feat_vector)[0]
    scores = scores.astype(np.float64)
    scores -= np.max(scores) 
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    
    infer_ms = (time.time() - t0) * 1000
    
    return {
        "box": (x, y, w, h),
        "roi": roi_img,
        "vis": vis_img,
        "probs": probs,
        "feat": feat_vector,
        "time": infer_ms
    }

def run_benchmark_logic(feat, n_runs):
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        svm.decision_function(feat)
        times.append((time.time() - t0) * 1000)
    return np.mean(times), np.std(times), times

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=90)
    st.title("Settings")
    st.markdown(f"""
        <div style="padding: 14px; border-radius: 12px; background: linear-gradient(135deg, #e0f2fe, #f8fafc); border: 1px solid #bae6fd;">
            <h4 style="margin-top: 0;">System Info</h4>
            <ul style="margin: 0; padding-left: 15px;">
                <li><b>Algorithm</b>: FAST + HOG + LBP</li>
                <li><b>Classifier</b>: SVM RBF</li>
                <li><b>Img Size</b>: {IMG_SIZE}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.info("💡 **Cara Kerja:**\nSistem mencari titik sudut (FAST), memotong area objek, lalu mengklasifikasikannya.")

st.title("🐶 Smart Dog Detector")
st.markdown("Deteksi lokasi dan klasifikasi ras anjing secara Real-Time.")

if svm is None:
    st.stop()

tab_detect, tab_bench = st.tabs(["🖼️ Smart Detection", "⚡ Performance Benchmark"])

if 'last_feat_vector' not in st.session_state:
    st.session_state['last_feat_vector'] = None

with tab_detect:
    col_input, col_result = st.columns([1.2, 1], gap="medium")

    with col_input:
        st.subheader("1. Input Image")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Detecting & Analyzing..."):
                res = process_pipeline(img_bgr)
                st.session_state['last_feat_vector'] = res['feat'] # Simpan untuk benchmark
            
            img_display = img_bgr.copy()
            x, y, w, h = res["box"]
            
            top_idx = np.argmax(res['probs'])
            top_class = classes[top_idx]
            top_prob = res['probs'][top_idx]
            
            color = (0, 255, 0) if top_class != "non_dog" else (0, 0, 255)
            
            cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 3)
            label_text = f"{top_class.replace('anjing_','').upper()} ({top_prob*100:.0f}%)"
            cv2.putText(img_display, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption=f"Processed Image ({res['time']:.2f} ms)", use_container_width=True)
            
            with st.expander("🔍 Lihat Area Deteksi (ROI) & Fitur LBP"):
                r1, r2 = st.columns(2)
                r1.image(cv2.cvtColor(res['roi'], cv2.COLOR_BGR2RGB), caption="Cropped ROI", use_container_width=True)
                r2.image(res['vis'], caption="LBP Texture View", use_container_width=True)

    with col_result:
        st.subheader("2. Analysis Results")
        
        if uploaded_file:
            clean_name = top_class.replace("anjing_", "").title()
            
            if top_class == "non_dog":
                st.error(f"⛔ Terdeteksi: **Bukan Anjing (Non-Dog)**")
            else:
                st.success(f"🐶 Terdeteksi: **{clean_name}**")
            
            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{top_prob*100:.1f}%")
            m2.metric("Pipeline Time", f"{res['time']:.2f} ms")

            # Chart Distribusi
            st.markdown("##### Probability Distribution")
            df_res = pd.DataFrame({
                "Breed": classes,
                "Probability": res['probs']
            }).sort_values(by="Probability", ascending=True)
            
            st.bar_chart(df_res.set_index("Breed"), color="#2e86c1", height=250)
            
        else:
            st.info("👈 Upload gambar untuk melihat keajaiban detektor FAST + SVM.")

with tab_bench:
    st.subheader("System Performance Test")
    st.markdown("Menguji stabilitas waktu inferensi model pada mesin ini.")
    
    b_col1, b_col2 = st.columns([1, 2])
    
    with b_col1:
        n_runs = st.slider("Number of Loops", 10, 200, 20)
        run_btn = st.button("🚀 Start Benchmark", type="primary")

    with b_col2:
        if run_btn:
            feat = st.session_state['last_feat_vector']
            
            if feat is None:
                st.warning("⚠️ Harap upload gambar di tab 'Smart Detection' dulu agar sistem punya data fitur untuk dites.")
            else:
                with st.spinner(f"Running prediction {n_runs} times..."):
                    mean_t, std_t, times = run_benchmark_logic(feat, n_runs)
                
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
                ax.fill_between(range(len(times)), mean_t - std_t, mean_t + std_t, color='red', alpha=0.1, label='Stability Range')
                
                ax.set_title("Inference Stability Check")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Time (ms)")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                
                st.pyplot(fig)