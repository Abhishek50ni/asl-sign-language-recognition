import os
import sys
import json
import numpy as np
from PIL import Image
import streamlit as st

# Smart path — works on both local Windows and Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ASL Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
IMG_SIZE = 96
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_mobilenet_final.keras")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")
CONFIDENCE_THRESHOLD = 0.70
# ---------------------------------------------------------
# Load model and class indices (cached — loads only once)
# ---------------------------------------------------------
@st.cache_resource  
# cache_resource keeps model in memory across reruns
# without this, model reloads on every interaction (very slow)
def load_model_cached():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_class_mapping():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    # Invert: {name: idx} -> {idx: name}
    return {v: k for k, v in class_indices.items()}


def preprocess_image(image):
    """
    Preprocesses a PIL Image for model prediction.
    Must match training preprocessing exactly:
    - Resize to 96x96
    - Normalize to [0,1]
    - Add batch dimension
    """
    image = image.convert("RGB")                    # ensure RGB (not RGBA/grayscale)
    image = image.resize((IMG_SIZE, IMG_SIZE))       # resize to match training
    img_array = np.array(image) / 255.0             # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)   # add batch dim: (1,96,96,3)
    return img_array


def predict(model, index_to_class, image):
    """
    Runs model prediction on a PIL Image.
    Returns predicted class, confidence, and all class probabilities.
    """
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    predicted_class = index_to_class[predicted_index]

    # Build top 5 predictions for display
    top5_indices = np.argsort(predictions[0])[::-1][:5]
    top5 = [(index_to_class[i], float(predictions[0][i])) for i in top5_indices]

    return predicted_class, confidence, top5


# ---------------------------------------------------------
# Custom CSS for professional styling
# ---------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }

    .predicted-letter {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
    }

    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Word display box */
    .word-box {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.3rem;
        color: #333;
        margin: 1rem 0;
        min-height: 70px;
    }

    /* Info cards */
    .info-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
    }

    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤟 ASL Recognition")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["🏠 Home & Predict", "📷 Live Webcam", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Model:** MobileNetV2
        - **Test Accuracy:** 94.76%
        - **Classes:** 29 ASL signs
        - **Input Size:** 96×96 RGB
        """)

        st.markdown("---")
        st.markdown("### 🔡 Supported Signs")
        st.markdown("A–Z, SPACE, DELETE, NOTHING")

        st.markdown("---")
        st.caption("Built with TensorFlow, OpenCV & Streamlit")

    return page


# ---------------------------------------------------------
# Page 1: Home & Image Upload Prediction
# ---------------------------------------------------------
def render_home_page(model, index_to_class):
    st.markdown(
        '<div class="main-header">🤟 ASL Sign Language Recognition</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Upload an image of a hand sign to get instant prediction</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a hand showing an ASL sign"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 🔮 Prediction")

        if uploaded_file:
            with st.spinner("Analyzing sign..."):
                predicted_class, confidence, top5 = predict(model, index_to_class, image)

            if confidence >= CONFIDENCE_THRESHOLD:
                # Main prediction display
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="predicted-letter">{predicted_class}</div>
                    <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(
                    f"Low confidence prediction: **{predicted_class}** ({confidence*100:.1f}%)\n\n"
                    "Try a clearer image with better lighting and plain background."
                )

            # Confidence bar
            st.markdown("#### Confidence Score")
            st.progress(confidence)

            # Top 5 predictions
            st.markdown("#### Top 5 Predictions")
            for cls, prob in top5:
                col_name, col_bar, col_pct = st.columns([1, 3, 1])
                with col_name:
                    st.markdown(f"**{cls}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.markdown(f"{prob*100:.1f}%")

        else:
            st.info("👆 Upload an image to get started")

            # Show example signs guide
            st.markdown("### 💡 Tips for Best Results")
            st.markdown("""
            - ✅ Use a **plain background** (white or black works best)
            - ✅ Ensure **good lighting** on your hand
            - ✅ Keep hand **centered** in the image
            - ✅ Show a **clear, static** sign
            - ✅ Avoid shadows on the hand
            """)


# ---------------------------------------------------------
# Page 2: Live Webcam Prediction
# ---------------------------------------------------------
def render_webcam_page(model, index_to_class):
    st.markdown("## 📷 Live Webcam Prediction")
    st.markdown("Use your webcam to predict ASL signs in real time.")
    st.markdown("---")

    # Initialize session state for word building
    if "current_word" not in st.session_state:
        st.session_state.current_word = ""
    if "last_letter" not in st.session_state:
        st.session_state.last_letter = ""
    if "letter_count" not in st.session_state:
        st.session_state.letter_count = 0

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### 📸 Camera Feed")
        # Streamlit's built-in camera input
        # captures one frame at a time on button press
        camera_image = st.camera_input(
            "Hold your sign steady and capture",
            help="Click the camera button to capture your hand sign"
        )

    with col2:
        st.markdown("### 🔮 Live Prediction")

        if camera_image:
            image = Image.open(camera_image)

            with st.spinner("Predicting..."):
                predicted_class, confidence, top5 = predict(model, index_to_class, image)

            # Display prediction
            if confidence >= CONFIDENCE_THRESHOLD:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="predicted-letter">{predicted_class}</div>
                    <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.progress(confidence)

                # Word building logic
                st.markdown("---")
                st.markdown("### 📝 Word Builder")

                btn_col1, btn_col2, btn_col3 = st.columns(3)

                with btn_col1:
                    if st.button("➕ Add Letter", use_container_width=True):
                        if predicted_class == "space":
                            st.session_state.current_word += " "
                        elif predicted_class == "del":
                            st.session_state.current_word = st.session_state.current_word[:-1]
                        elif predicted_class != "nothing":
                            st.session_state.current_word += predicted_class

                with btn_col2:
                    if st.button("🗑️ Delete Last", use_container_width=True):
                        st.session_state.current_word = st.session_state.current_word[:-1]

                with btn_col3:
                    if st.button("🔄 Reset Word", use_container_width=True):
                        st.session_state.current_word = ""

            else:
                st.warning(
                    f"Low confidence: **{predicted_class}** ({confidence*100:.1f}%)\n\n"
                    "Try better lighting or a clearer hand position."
                )

        else:
            st.info("📸 Click the camera button above to capture a sign")

        # Word display (always visible)
        st.markdown("### 🔤 Current Word")
        word_display = st.session_state.current_word if st.session_state.current_word else "..."
        st.markdown(
            f'<div class="word-box">{word_display}</div>',
            unsafe_allow_html=True
        )

        # Full reset button
        if st.button("🔄 Clear Everything", use_container_width=True):
            st.session_state.current_word = ""
            st.rerun()


# ---------------------------------------------------------
# Page 3: About
# ---------------------------------------------------------
def render_about_page():
    st.markdown("## ℹ️ About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🎯 Project Goals")
        st.markdown("""
        This project was built to:
        - Learn CNN architecture from scratch
        - Implement Transfer Learning with MobileNetV2
        - Build a complete image classification pipeline
        - Deploy a real-world Deep Learning model

        Every line of code was written and understood from scratch
        — no pre-built pipelines or copied repositories.
        """)

        st.markdown("### 🧠 Models Built")
        st.markdown("""
        **1. Custom CNN (Baseline)**
        - Built from scratch: Conv2D → MaxPool → Dense
        - Input: 64×64 RGB
        - Parameters: 686,941 (~2.62 MB)
        - Test Accuracy: **95.34%**

        **2. MobileNetV2 (Production)**
        - Two-phase transfer learning
        - Phase 1: Frozen base, head only
        - Phase 2: Fine-tuned top 30 layers
        - Input: 96×96 RGB
        - Test Accuracy: **94.76%**
        - Val Accuracy: **85.04%** ← better generalization
        """)

    with col2:
        st.markdown("### 📊 Performance")

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Test Accuracy", "94.76%", "+14% vs random")
        with m2:
            st.metric("Val Accuracy", "85.04%", "+4.35% vs CNN")
        with m3:
            st.metric("Classes", "29", "A-Z + 3")

        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        | Component | Tool |
        |---|---|
        | Deep Learning | TensorFlow 2.19 / Keras 3 |
        | Base Model | MobileNetV2 (ImageNet) |
        | Computer Vision | OpenCV |
        | Web App | Streamlit |
        | Image Processing | PIL / Pillow |
        | Data Analysis | NumPy, Pandas |
        | Visualization | Matplotlib, Seaborn |
        """)

        st.markdown("### 📂 Dataset")
        st.markdown("""
        **ASL Alphabet Dataset (Kaggle)**
        - 87,000 training images
        - 29 balanced classes (3,000/class)
        - 200×200 RGB images
        - [View Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
        """)

    st.markdown("---")
    st.markdown("### 🔑 Key Learnings")

    kl1, kl2, kl3 = st.columns(3)
    with kl1:
        st.info("**CNN Architecture**\nLearned how Conv2D filters detect edges → curves → hand shapes across layers")
    with kl2:
        st.info("**Transfer Learning**\nTwo-phase strategy: frozen extraction then fine-tuning prevents catastrophic forgetting")
    with kl3:
        st.info("**Domain Gap**\nModel achieving 94%+ test accuracy can still face challenges on out-of-distribution real-world input")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:gray;'>Built with ❤️ using TensorFlow, OpenCV & Streamlit</p>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Main App Entry Point
# ---------------------------------------------------------
def main():
    apply_custom_css()

    # Load model once (cached)
    with st.spinner("Loading ASL Recognition Model..."):
        model = load_model_cached()
        index_to_class = load_class_mapping()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to correct page
    if page == "🏠 Home & Predict":
        render_home_page(model, index_to_class)
    elif page == "📷 Live Webcam":
        render_webcam_page(model, index_to_class)
    elif page == "ℹ️ About":
        render_about_page()


if __name__ == "__main__":
    main()
