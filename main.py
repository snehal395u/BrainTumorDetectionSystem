import os
import json
import hashlib
import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from PIL import Image

# ===================== CONFIG =====================
OPENROUTER_API_KEY = "YOUR_API_KEY"

working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    working_dir,
    "models",
    "cnn-parameters-improvement-01-0.579117.keras"
)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_cnn_model():

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found:\n{model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ CNN Model Loaded Successfully")
        return model

    except Exception as e:
        st.error(f"⚠ Error loading model: {e}")
        return None
model = load_cnn_model()
# ===================== UTILS =====================
prediction_cache = {}

def generate_cache_key(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def preprocess_image(image, target_size=(140, 140)):  
    img = image.convert("RGB")
    img = img.resize(target_size)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict(image):
    if model is None:
        st.error("Model not loaded properly.")
        return "Model not loaded"

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    st.write("Raw prediction:", prediction)

    # ✅ CASE 1: Binary model (single output neuron)
    if prediction.shape[-1] == 1:
        prob = float(prediction[0][0])
        st.write("Probability:", prob)

        if prob > 0.7:
            class_idx = 1
        else:
            class_idx = 0

    # ✅ CASE 2: Softmax (2 output neurons)
    else:
        class_idx = np.argmax(prediction)

    st.write("Predicted class index:", class_idx)

    # ⚠ IMPORTANT: YOU MAY NEED TO SWAP THIS
    label = "Tumor Detected 🧠" if class_idx == 1 else "No Tumor ✅"

    return label
def fetch_ai_explanation(result):
    prompt = f"""
    The MRI scan result is: {result}.
    Explain this in simple medical terms and suggest next steps.
    """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠ AI API Error: {e}"

# ===================== UI =====================
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.markdown("""
    <style>

        /* Sidebar Blue-Green Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #00B4D8, #2E8B57);
            color: white;
        }

        [data-testid="stSidebar"] h1 {
            color: white;
        }

        /* Main Heading */
        .main-heading {
            text-align: center;
            font-size: 38px;
            background: linear-gradient(90deg, #0077B6, #2E8B57);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }

        /* Button Styling */
        div.stButton > button {
            background: linear-gradient(90deg, #00B4D8, #2E8B57);
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            transition: 0.3s ease;
        }

        div.stButton > button:hover {
            transform: scale(1.05);
        }

        /* Image Styling */
        img {
            border-radius: 12px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }

        hr {
            border: 1px solid #2E8B57;
        }
        .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #0077B6, #2E8B57);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
        }

    </style>
    <div class="footer">
        © 2026 Snehal Jadhav | Brain Tumor Detection System 🧠 | All Rights Reserved
    </div>
""", unsafe_allow_html=True)
st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Demo", "Tech Talk","Developer Info"])

# ===================== HOME =====================
if page == "Home":
    st.title("🧠 Brain Tumor Detection System")

    st.markdown("""
    ## 📌 Project Overview
    This project uses **Deep Learning (CNN)** to classify MRI brain scans.

    ### ✅ Model Used
    - Convolutional Neural Network (CNN)
    - Binary Classification: Tumor / No Tumor
    - Trained using TensorFlow/Keras
    """)

    # ✅ FIXED RELATIVE IMAGE PATHS
    acc_path = os.path.join(working_dir, "Accuracy.PNG")
    loss_path = os.path.join(working_dir, "Loss.PNG")

    if os.path.exists(acc_path):
        st.image(acc_path, caption="Model Accuracy")

    if os.path.exists(loss_path):
        st.image(loss_path, caption="Model Loss")

# ===================== DEMO =====================
elif page == "Demo":
    st.title("📷 MRI Scan Analysis")

    uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded MRI", width=300)

        if st.button("🔍 Analyze Scan"):
            result = predict(image)
            st.success(f"Prediction: {result}")

            with st.spinner("🤖 Generating AI Explanation..."):
                explanation = fetch_ai_explanation(result)

            st.info(explanation)


# ===================== TECH TALK =====================
elif page == "Developer Info":
    st.markdown("<h1 class='main-heading'>🧠 Brain Tumor Detection System 🚀</h1>", unsafe_allow_html=True)

    developer_image_path = "C:/Users/USER/OneDrive/Desktop/srushti31/doctor/Brain-Tumor-Detection-master/Brain-Tumor-Detection-master/me.jpeg"
    st.image(developer_image_path, caption="Snehal Jadhav", width=500)

    st.markdown("""
        ## 📬 Connect with Me
        - 📧 **Email:** snehalljadhav395@gmail.com  
        - 🔗 **GitHub:** https://github.com/snehal395u  
        - 🔗 **LinkedIn:** https://www.linkedin.com/in/snehal-jadhav-1b1101305/  

        <p style='text-align: center; font-size: 16px;'>
            <b>Let's innovate in AI & Healthcare together! 🚀</b>
        </p>

        <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style='text-align: center; color: #0077B6;'>👋 Hey! I'm Snehal Jadhav</h2>
        <p style='text-align: center; font-size: 18px;'>
        Passionate <b>AI Engineer & Software Developer</b> currently pursuing 
        <b>BSc Computer Science (Third Year)</b>.  
        I specialize in <b>Deep Learning, Medical Imaging AI, Backend Development</b> 
        and <b>Data Structures & Algorithms</b>.
        </p>
        <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
        ## 🧠 About This Project  
        This is a <b>Brain Tumor Detection System</b> built using 
        <b>Deep Learning (CNN)</b> to classify MRI brain scans into:
        <ul>
            <li>Tumor Detected</li>
            <li>No Tumor</li>
        </ul>

        The model is trained on MRI datasets and deployed using 
        <b>Streamlit</b> for real-time prediction.

        ### 🛠️ Tech Stack Used:
        - TensorFlow / Keras
        - NumPy
        - Pillow (PIL)
        - Python
        - Streamlit
        - OpenRouter API
        <br>
        <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
        ## 💡 Technical Expertise
        - 🔹 Java, Spring Boot, Microservices  
        - 🔹 React / Next.js  
        - 🔹 Deep Learning (CNN, TensorFlow, PyTorch)  
        - 🔹 Java,Python (DSA Focused)  
        - 🔹 MySQL, MongoDB, Firebase  
        - 🔹 REST API Development  
        <hr>
    """, unsafe_allow_html=True)



# ==============================
# 🎤 TECH TALK PAGE
# ==============================

elif page == "Tech Talk":
    st.markdown("<h1 class='main-heading'>🎤 Tech Talk - Brain Tumor AI</h1>", unsafe_allow_html=True)

    st.markdown("""
    ## 🧠 System Architecture
    """)

    st.markdown("""
    - **Input:** MRI Brain Scan Image  
    - **Preprocessing:** Resize, Normalize, Convert to Array  
    - **Model:** Convolutional Neural Network (CNN)  
    - **Output:** Tumor / No Tumor Prediction  
    """)

    st.markdown("""
    ## ⚙️ Model Workflow
    1. Image Upload  
    2. Image Preprocessing  
    3. CNN Feature Extraction  
    4. Dense Layers Classification  
    5. Final Prediction Display  
    """)

    st.markdown("""
    ## 🛠️ Technologies Used
    - Python  
    - TensorFlow / Keras  
    - NumPy  
    - Pillow (PIL)  
    - Streamlit  
    - OpenRouter API  
    """)

    st.markdown("""
    ## 👨‍💻 Developer
    **Snehal Jadhav**  
    AI & Healthcare Software Developer  

    🔗 GitHub: https://github.com/snehal395u  
    🔗 LinkedIn: https://www.linkedin.com/in/snehal-jadhav-1b1101305/  
    """)
