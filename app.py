import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os


# ------------------------ #
# ⚙️ Streamlit Page Config
# ------------------------ #
st.set_page_config(
    page_title=" AI Student Focus Detection",
    layout="centered"
)

# ------------------------ #
# Load YOLOv8 Model
# ------------------------ #
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ------------------------ #
# Custom CSS

st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css');

    html, body, [class*="css"] {
        margin: 0 !important;
        padding: 0 !important;
        overflow-x: hidden !important;
    }

    body {
        background-color: #FAFAFA;
        font-family: "Inter", sans-serif;
        color: #111827;
    }

    /* Reduce Streamlit default padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* General text styles */
    .center { text-align: center; }
    .main-title { font-size: 3rem; font-weight: 800; color: #111827; }
    .green { color: #16A34A; }
    .subtext { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem; }

    /* Header */
    .focus-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        background-color: white;
        border-bottom: 1px solid #E5E7EB;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .focus-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.5rem;
        font-weight: 700;
        color: #000000;
    }
    .focus-logo-icon {
        background-color: #16A34A;
        color: white;
        border-radius: 10px;
        padding: 8px;
        font-size: 1.1rem;
    }

    /* Footer */
    .footer {
        color: #6B7280;
        text-align: center;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }

    /* Reduce white space below Streamlit widgets */
    section[data-testid="stSidebar"] > div:first-child,
    section[data-testid="stVerticalBlock"] {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }

    footer, #MainMenu, header {
        visibility: hidden;
        height: 0;
    }

    </style>
""", unsafe_allow_html=True)


# ------------------------ #
# Header Components
# ------------------------ #
def focus_header():
    st.markdown("""
    <div class="focus-header">
        <div class="focus-logo">
            <div class="focus-logo-icon"><i class="fa-solid fa-brain"></i></div>
            FocusDetect
        </div>
    </div>
    """, unsafe_allow_html=True)


# fotter componentes
def footer():
    st.markdown("""
    <hr style='margin-top:3rem;margin-bottom:2rem;border: none;border-top:1px solid #E5E7EB;'>
    <div style='display:flex;flex-wrap:wrap;justify-content:space-evenly;align-items:flex-start;gap:2rem;'>
        <!-- Left -->
        <div style='flex:1;min-width:230px;'>
            <div style='display:flex;align-items:center;gap:10px;'>
                <div class="focus-logo-icon"><i class="fa-solid fa-brain"></i></div>
                <h3 style='margin:0;color:#000000;font-weight:600;'>FocusDetect</h3>
            </div>
            <p style='margin-top:8px;color:#6B7280;font-size:0.95rem;line-height:1.4;'>
                AI-powered student focus detection using YOLOv8 for smarter classroom monitoring.
            </p>
        </div>
        <!-- Middle -->
        <div style='flex:1;min-width:150px;'>
            <h4 style='margin-bottom:0.5rem;font-weight:600;color:#111827;'>Quick Links</h4>
            <p style='margin:4px 0;color:#6B7280;line-height:1.6;'>
                <a href='#' style='color:#6B7280;text-decoration:none;'><i class="fa-solid fa-house"></i> &nbsp;Home</a><br>
                <a href='#' style='color:#6B7280;text-decoration:none;'><i class="fa-solid fa-image"></i> &nbsp;Upload Image</a><br>
                <a href='#' style='color:#6B7280;text-decoration:none;'><i class="fa-solid fa-camera-retro"></i>&nbsp; Live Detection</a><br>
                <a href='#' style='color:#6B7280;text-decoration:none;'><i class="fa-solid fa-circle-info"></i> &nbsp; About</a>
            </p>
        </div><!-- Right -->
        <div style='flex:1;min-width:150px;'>
            <h4 style='margin-bottom:0.5rem;font-weight:600;color:#111827;'>Connect</h4>
            <div style='display:flex;gap:10px;'>
                <a href='https://github.com/' target='_blank'
                   style='color:#000000;display:inline-flex;align-items:center;justify-content:center;
                          width:40px;height:40px;border:1px solid #D1D5DB;border-radius:10px;
                          text-decoration:none;font-size:1.2rem; '><i class="fa-brands fa-github"></i></a>
                <a href='https://twitter.com/' target='_blank'
                   style='color:#000000;display:inline-flex;align-items:center;justify-content:center;
                          width:40px;height:40px;border:1px solid #D1D5DB;border-radius:10px;
                          text-decoration:none;font-size:1.2rem;'><i class="fa-brands fa-x-twitter"></i></a>
            </div>
        </div>
    </div>

    <p style='text-align:center;color:#9CA3AF;font-size:0.85rem;margin-top:2rem;'><span style='color:#000000;'><i class="fa-regular fa-copyright"></i></span>
     2025 FocusDetect. Built with <span style='color:#EF4444;'><i class="fa-solid fa-heart"></i></span> using YOLOv8 and Streamlit.
    </p>
    """, unsafe_allow_html=True)

# ------------------------ #
# Page Navigation
# ------------------------ #
if "page" not in st.session_state:
    st.session_state.page = "landing"


# ------------------------ #
# PAGE 1: Landing
# ------------------------ #
if st.session_state.page == "landing":
    focus_header()
    st.markdown("""
    <div class='center'>
        <p style='background:#ECFDF5;color:#065F46;
           display:inline-block;padding:0.4rem 1rem;border-radius:25px;font-weight:600'>
           <i class="fa-solid fa-brain"></i> Powered by YOLOv8 AI Technology
        </p>
        <h1 class='main-title'>AI-Powered Student <span class='green'>Focus Detection</span></h1>
        <p class='subtext'>
            Revolutionize classroom monitoring with real-time AI detection.<br>
            Upload images or use live webcam to instantly detect student focus levels.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Upload Image", use_container_width=True,):
            st.session_state.page = "upload"
            st.rerun()
    with col2:
        if st.button("Take Picture ", use_container_width=True):
            st.session_state.page = "webcam"
            st.rerun()

    footer()


# ------------------------ #
# PAGE 2: Upload Image
# ------------------------ #
elif st.session_state.page == "upload":
    focus_header()
    st.markdown("<h2 class='center'>Upload Image for Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p class='center subtext'>Upload a classroom photo to detect focus using YOLOv8</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop or select an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("""### <div><i class="fa-solid fa-brain"></i> AI Detection Results</div>""",unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(temp_path, caption=" Original Image", use_column_width=True)
        with col2:
            results = model.predict(source=temp_path, conf=0.25)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption=" Detection Results", use_column_width=True)


    if st.button(" Back to Home", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()
    footer()


# ------------------------ #
# PAGE 3: Live Webcam with Side-by-Side Result
# ------------------------ #
elif st.session_state.page == "webcam":
    focus_header()
    st.markdown("<h2 class='center'>Take Picture From Camera</h2>", unsafe_allow_html=True)
    st.markdown("<p class='center subtext'>Real-time student focus detection using your webcam</p>", unsafe_allow_html=True)

    picture = st.camera_input(" Take Photo for Focus Detection")

    if picture is not None:
        img = Image.open(picture)
        img_array = np.array(img)

        # Run YOLO prediction
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()

        # Display in side-by-side layout
        st.markdown("<h3 style='text-align:center;'> AI Detection Results</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption=" Original Image", use_column_width=True)
        with col2:
            st.image(res_plotted, caption=" Detection Results", use_column_width=True)

        # Count classes
        classes = [model.names[int(c)] for c in results[0].boxes.cls]
        focus_count = classes.count("focus")
        unfocus_count = classes.count("unfocus")

        

    if st.button(" Back to Home", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()
    footer()
