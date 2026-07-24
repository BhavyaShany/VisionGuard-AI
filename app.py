import torch

# Directly override torch.load to ignore the strict weights_only check globally
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

import os
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
import numpy as np
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from zoneinfo import ZoneInfo
import torch
torch.serialization.add_safe_globals([torch.serialization.Storage])
try:
    import ultralytics.nn.tasks
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except ImportError:
    pass
  
# Set page config with improved styling
st.set_page_config(
    page_title="AI CCTV Surveillance",
    layout="wide",
    page_icon="👷",
    initial_sidebar_state="expanded"
)

# Load YOLOv8 model
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")
    st.stop()

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

model = load_yolo_model(MODEL_PATH)
CLASS_NAMES = model.names
LOG_FILE = "violation_logs.csv"

if "stream_active" not in st.session_state:
    st.session_state.stream_active = False
if "stream_source" not in st.session_state:
    st.session_state.stream_source = None

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        /* Main background: Ultra modern deep dark blue/gray */
        .main {
            background-color: #0b0f19 !important; 
            color: #f1f5f9;
            font-family: 'Segoe UI', sans-serif;
        }
        
        /* Interactive Action Buttons: Radiant Electric Purple/Pink */
        .stButton>button {
            background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 22px;
            font-weight: 700;
            border: none !important;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
        }

        .stSelectbox, .stTextInput, .stRadio>div {
            background-color:#191970	 !important;
            color: #0f172a !important;
            border-radius: 8px;
            padding: 10px;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(0, 242, 254, 0.2);
        }
        
        /* Make sure label text on top of inputs is bright white/slate */
        label, p {
            color: #e2e8f0 !important;
        }

        /* Violation Alerts: Vibrant Neon Red/Orange Alert Card with glowing border */
        .violation-card {
            border-left: 6px solid #ff3366;
            background-color: #1e151a;
            padding: 14px;
            margin-bottom: 14px;
            border-radius: 8px;
            color: #ff4d79;
            font-weight: 600;
            box-shadow: 0 0 15px rgba(255, 51, 102, 0.25);
            border: 1px solid rgba(255, 51, 102, 0.2);
        }

        /* Sidebar: High contrast dark layout to separate options */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 2px solid #3b82f6;
        }

        /* Headings: Bright Neon Green and Cyan accents */
        h1 { 
            color: #00ffcc !important;
            text-shadow: 0 0 12px rgba(0, 255, 204, 0.3);
            font-weight: 800;
        }
        h2, h3, h4 { 
            color: #38bdf8 !important; 
            font-weight: 700;
        }

        .dark-warning-box {
            background-color: rgba(255, 179, 0, 0.05) !important;
            border: 1px solid rgba(255, 179, 0, 0.35) !important;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 15px 0 25px 0;
        }
        .dark-warning-title {
            color: #ffcc00 !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            margin-bottom: 12px !important;
        }
        .dark-warning-list {
            list-style: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .dark-warning-item {
            color: #e2e8f0 !important;
            font-size: 14px !important;
            margin-bottom: 6px !important;
            display: flex;
            align-items: center;
        }
</style>
""", unsafe_allow_html=True)

# Main header
colored_header(
    label="👷 VisionGuard- AI",
    description="An AI-powered CCTV surveillance system for real-time detection of PPE compliance using YOLOv8.",
    color_name="blue-70",
)

# Sidebar with layout
with st.sidebar:
    if os.path.exists("home.jpeg"):
        st.image("home.jpeg", use_container_width=True)
    
    st.markdown("""
    <div style="margin-top: 20px;">
        <div style="display: flex; align-items: center; gap: 10px; background-color: #003366; color: white; padding: 10px 16px; border-radius: 8px; font-weight: bold; font-size: 1.1rem;">
            ⚙️ Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    source_type = st.radio(
        "Select Input Source",
        ['Browser Webcam (Photo)', 'Upload Video', 'Upload Image', 'RTSP IP Camera', 'OpenCV Webcam (Local Only)'],
        index=0,
        help="Choose the source for surveillance feed"
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Model", "YOLOv8", "Active")
    status_col2.metric("FPS", "30", "Live")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This AI surveillance system detects:
    - No-SafetyVest
    - No-HardHat
    - No-Mask
    - No-Gloves
    - Machines
    - Person   
    - Items from Construction site        
    """)

def log_violation_memory(class_name, confidence):
    """Logs data to CSV safely without blocking video frames."""
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame(
        [[timestamp, class_name, round(confidence * 100, 2)]],
        columns=["Timestamp", "Violation", "Confidence"]
    )
    
    if not os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, index=False)
    else:
        entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    return timestamp

def process_frame(frame, alert_placeholder):
    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()
    violation_count = 0
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        confidence = float(box.conf[0])
        
        if "NO" in class_name.upper():
            violation_count += 1
            timestamp = log_violation_memory(class_name, confidence)
            
            alert_placeholder.markdown(f"""
            <div class="violation-card">
                <strong>⚠️ New Violation Detected</strong><br>
                Type: {class_name}<br>
                Confidence: {round(confidence * 100, 2)}%<br>
                Time: {timestamp}
            </div>
            """, unsafe_allow_html=True)
            
    return annotated_frame, results, violation_count

def display_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("❌ Unable to open the selected video source.")
        return

    st_frame = st.empty()
    alert_placeholder = st.empty()

    stop_button = st.button("🛑 Stop Stream", key="stop_stream_btn")
    if stop_button:
        st.session_state.stream_active = False

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    p1 = metrics_col1.empty()
    p2 = metrics_col2.empty()
    p3 = metrics_col3.empty()

    style_metric_cards(background_color="#f8f9fa", border_left_color="#0068c9", box_shadow="0 2px 8px rgba(0,0,0,0.1)")

    fail_count = 0
    while cap.isOpened() and st.session_state.stream_active:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 15:
                st.warning("⚠️ Stream connection lost.")
                break
            continue
        fail_count = 0

        start_time = time.time()
        annotated_frame, results, violation_count = process_frame(frame, alert_placeholder)
        processing_time = (time.time() - start_time) * 1000

        p1.metric("Processing Time", f"{processing_time:.1f} ms")
        p2.metric("Objects Detected", len(results.boxes))
        p3.metric("Violations Tracked", violation_count)

        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(0.01)

    cap.release()

# Tabs Setup
tab1, tab2 = st.tabs(["Live Monitoring", "Violation Logs"])

with tab1:

    if source_type == 'Browser Webcam (Photo)':
        st.info("ℹ️ Captures single photos via your web browser interface.")
        captured_image = st.camera_input("Take a photo for PPE detection")
        alert_placeholder = st.empty()
        
        if captured_image:
            file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated_frame, results, violation_count = process_frame(frame, alert_placeholder)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            with col2:
                st.metric("Violations Detected", violation_count)

    elif source_type == 'Upload Video':
        with stylable_container(
            key="upload_container", 
            css_styles="{border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 8px; padding: 20px;}"):
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
            if uploaded_file:
                temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_video_path, 'wb') as f:
                    f.write(uploaded_file.read())
                st.session_state.stream_active = True
                display_video(temp_video_path)

    elif source_type == 'OpenCV Webcam (Local Only)':
        st.markdown("""<div class="dark-warning-box">
            <div class="dark-warning-title">🌐 Webcam access is disabled in cloud deployments. Try these instead:</div>
            <ul class="dark-warning-list">
                <li class="dark-warning-item">➡️ &nbsp;<span style="font-size:16px;">📁</span>&nbsp; Upload a video file</li>
                <li class="dark-warning-item">➡️ &nbsp;<span style="font-size:16px;">📡</span>&nbsp; Use RTSP stream</li>
                <li class="dark-warning-item">➡️ &nbsp;<span style="font-size:16px;">💻</span>&nbsp; Run locally for webcam</li>
            </ul>
        </div> 
    """, unsafe_allow_html=True)
        if os.environ.get('IS_STREAMLIT_CLOUD'):
            st.error("OpenCV local hardware hooks are inaccessible on cloud servers!")
        else:
            if st.button("🎥 Start Local Webcam Feed"):
                st.session_state.stream_active = True
            if st.session_state.stream_active:
                display_video(0)

    elif source_type == 'Upload Image':
        uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        alert_placeholder = st.empty()
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated_frame, results, violation_count = process_frame(frame, alert_placeholder)
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    elif source_type == 'RTSP IP Camera':
        rtsp_url = st.text_input("Enter RTSP Stream URL", placeholder="rtsp://username:password@192.168.1.100:554/stream1")
        if rtsp_url and st.button("📡 Start RTSP Stream", type="primary"):
            st.session_state.stream_active = True
            display_video(rtsp_url)

with tab2:
    st.markdown("### 📄 Recent Violation Logs")
    if os.path.exists(LOG_FILE):
        try:
            df_logs = pd.read_csv(LOG_FILE)
            if not df_logs.empty:
                # 1. Preview Table Setup (showing newest logs at the top)
                st.dataframe(
                    df_logs.tail(10).sort_values("Timestamp", ascending=False), 
                    use_container_width=True, 
                    hide_index=True
                )
                
                # 2. Stable Action Bar Layout
                col1, col2 = st.columns(2)
                with col1:
                    # FIX: Convert data to an immutable byte array before rendering the button
                    csv_bytes = df_logs.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Full Log", 
                        data=csv_bytes, 
                        file_name=f"violation_report_{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_btn_stable", # Unique explicit key prevents state loss
                        use_container_width=True
                    )
                with col2:
                    # Streamlined unique key mapping for Clear button
                    if st.button("🗑️ Clear Logs", key="clear_logs_stable_btn", use_container_width=True):
                        if os.path.exists(LOG_FILE):
                            os.remove(LOG_FILE)
                            st.success("Logs cleared successfully!")
                            st.rerun()
                
                # 3. Analytics Charts
                st.markdown("---")
                st.markdown("### 📊 Violation Statistics")
                st.bar_chart(df_logs["Violation"].value_counts())
            else:
                st.info("ℹ️ No violations logged yet.")
        except Exception as e:
            st.error(f"Error processing logs: {e}")
    else:
        st.info("ℹ️ No violations logged yet.")





st.markdown("""<hr style="margin-top: 3rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 0.9em; color: grey; margin-bottom: 2rem;'>
        👨‍💻 Developed by: Bhavya Shany  |  2501730024 |
        <a href='https://github.com/BhavyaShany/VisionGuard-AI' target='_blank' style='color: grey; text-decoration: none;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style='vertical-align: middle; margin-left: 4px;'>
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
        </a>
    </div>
""", unsafe_allow_html=True)
