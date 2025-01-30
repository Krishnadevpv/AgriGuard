import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure the page
st.set_page_config(
    page_title="AgriGuard",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated styling with a fresh color palette
st.markdown("""
    <style>
        .stApp {
            background-color: #1B1B2F;
            color: #E94560;
        }
        .stButton > button {
            background-color: #E94560;
            color: #FFF !important;
            border-radius: 10px;
            font-weight: bold;
            border: none;
            padding: 0.7rem 1.5rem;
        }
        .css-1d391kg {
            background-color: #162447;
        }
        h2, h3, h1 {
            color: #E94560 !important;
            font-weight: bold !important;
            text-align: center;
        }
        p, li {
            color: #FFF !important;
            font-size: 18px !important;
            text-align: center;
        }
        a {
            color: #E94560 !important;
            font-weight: 500;
        }
        .taskbar-text {
            color: #FFF !important;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .container {
            display: flex;
            justify-content: space-around;
        }
        .box {
            background-color: #162447;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            width: 30%;
        }
    </style>
""", unsafe_allow_html=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Taskbar
st.markdown("""
    <div class='taskbar-text'>
        🌻 AgriGuard - AI for Plant Health Monitoring
    </div>
""", unsafe_allow_html=True)
app_mode = st.selectbox("Select Page", ["Home", "Disease Recognition"], index=0)

if app_mode == "Home":
    st.markdown("<h1>AgriGuard: AI-Powered Plant Disease Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='container'>
            <div class='box'>
                <h3>About AgriGuard</h3>
                <p>
                    AgriGuard is a smart AI-based platform that helps farmers detect plant diseases early.
                    Simply upload an image of a leaf, and our deep learning model will analyze and provide accurate diagnostics.
                </p>
            </div>
            <div class='box'>
                <h3>How It Works</h3>
                <p>
                    1. Upload a plant leaf image<br>
                    2. AI scans and processes the image<br>
                    3. Get instant disease identification and recommendations
                </p>
            </div>
            <div class='box'>
                <h3>Technologies Used</h3>
                <p>
                    - Streamlit<br>
                    - TensorFlow & Keras<br>
                    - NumPy<br>
                    - Pillow
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image for Analysis:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        if st.button("Analyze Disease"):
            st.snow()
            st.write("AI Analysis Result")
            result_index = model_prediction(test_image)
            class_names = ['Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
                           'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 'Corn - Gray Leaf Spot',
                           'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black Rot',
                           'Grape - Black Measles', 'Grape - Leaf Blight', 'Grape - Healthy', 'Orange - Citrus Greening',
                           'Peach - Bacterial Spot', 'Peach - Healthy', 'Pepper - Bacterial Spot', 'Pepper - Healthy',
                           'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy',
                           'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
                           'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold',
                           'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot',
                           'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy']
            st.success(f"Diagnosis: {class_names[result_index]}")

# Footer
st.markdown("<hr style='margin: 30px 0px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #E94560;'>AgriGuard © 2025 | AI for Smart Farming</p>", unsafe_allow_html=True)
