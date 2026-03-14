import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Food Freshness AI",
    page_icon="🍎",
    layout="wide"
)

# ---------------------------------
# CUSTOM STYLE
# ---------------------------------
st.markdown("""
<style>

.main-title{
font-size:42px;
font-weight:700;
color:#2e7d32;
}

.result-card{
padding:25px;
border-radius:12px;
background-color:#f5f5f5;
text-align:center;
font-size:24px;
font-weight:600;
}

.footer{
text-align:center;
color:gray;
font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# CONFIG
# ---------------------------------
IMG_SIZE = 224
CLASS_NAMES = ["Fresh", "Spoiled"]

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("⚙️ Settings")

category = st.sidebar.selectbox(
    "Select Category",
    ["Fruit", "Vegetable"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI model detects if food is fresh or rotten.")

# ---------------------------------
# LOAD MODEL (.pb SavedModel)
# ---------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# ---------------------------------
# IMAGE PREPROCESS
# ---------------------------------
def preprocess(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# ---------------------------------
# PREDICTION
# ---------------------------------
def predict(image):

    if model is None:
        return "Model not loaded", 0

    processed = preprocess(image)

    prediction = model.predict(processed)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    label = CLASS_NAMES[predicted_class]

    return label, confidence


# ---------------------------------
# HEADER
# ---------------------------------
st.markdown('<p class="main-title">🍎 Food Freshness Detection</p>', unsafe_allow_html=True)

st.write("Upload an image of a **fruit or vegetable** to check if it is **Fresh or Rotten**.")

st.markdown("---")

# ---------------------------------
# MAIN LAYOUT
# ---------------------------------
col1, col2 = st.columns(2)

# ---------------------------------
# IMAGE UPLOAD
# ---------------------------------
with col1:

    st.subheader("📤 Upload Image")

    uploaded = st.file_uploader(
        "Upload a fruit or vegetable image",
        type=["jpg","png","jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):

            label, confidence = predict(image)

            with col2:

                st.subheader("🧠 Prediction Result")

                st.markdown(
                f'<div class="result-card">{label}</div>',
                unsafe_allow_html=True
                )

                st.write("Confidence Score")

                st.progress(float(confidence))

                st.write(f"{confidence*100:.2f}%")

# ---------------------------------
# CAMERA DETECTION
# ---------------------------------
st.markdown("---")

st.subheader("📷 Camera Detection")

camera = st.camera_input("Take a picture")

if camera:

    image = Image.open(camera)

    st.image(image, width=400)

    label, confidence = predict(image)

    st.markdown(
    f'<div class="result-card">{label}</div>',
    unsafe_allow_html=True
    )

    st.progress(float(confidence))

    st.write(f"Confidence: {confidence*100:.2f}%")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")

st.markdown(
'<p class="footer">AI Model • Fruit & Vegetable Freshness Detection</p>',
unsafe_allow_html=True
)