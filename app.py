import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

PROJECT_DIR = Path(__file__).resolve().parent

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🧠 Model Information")

    st.write(
        "This application uses a fine-tuned EfficientNetB0 "
        "deep learning model for four-class brain MRI "
        "image classification."
    )

    st.divider()

    st.subheader("Classification Classes")

    for class_name in CLASS_NAMES:
        st.write(f"• {class_name}")

    st.divider()

    st.subheader("Model Performance")

    st.metric("Test Accuracy", "78.46%")
    st.metric("Precision", "82.31%")
    st.metric("Recall", "78.46%")
    st.metric("F1-Score", "78.69%")

# ============================================================
# MODEL FINDER
# ============================================================

def find_model():
    preferred_names = [
        "efficientnetb0_transfer.h5",
        "efficientnetb0_transfer.keras",
        "efficientnetb0.h5",
        "efficientnetb0.keras",
        "transfer_model.h5",
        "transfer_model.keras",
        "brain_tumor_model.h5",
        "brain_tumor_model.keras",
        "model.h5",
        "model.keras",
    ]

    for name in preferred_names:
        candidate = PROJECT_DIR / name
        if candidate.exists():
            return candidate

    candidates = list(PROJECT_DIR.glob("*.h5"))
    candidates += list(PROJECT_DIR.glob("*.keras"))
    candidates += list(PROJECT_DIR.glob("*/*.h5"))
    candidates += list(PROJECT_DIR.glob("*/*.keras"))

    if not candidates:
        return None

    preferred = [
        p for p in candidates
        if any(
            word in p.name.lower()
            for word in ["efficientnet", "transfer", "brain", "tumor"]
        )
    ]

    return preferred[0] if preferred else candidates[0]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_path = find_model()

if model_path is None:
    st.error(
        "No .h5 or .keras model was found. "
        "Place your trained model in the same folder as app.py "
        "and restart Streamlit."
    )
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as e:
    st.error("The model could not be loaded.")
    st.exception(e)
    st.stop()

# ============================================================
# TITLE
# ============================================================

st.title("🧠 Brain Tumor MRI Classification")

st.caption(
    "AI-assisted classification of brain MRI images using "
    "EfficientNetB0 transfer learning"
)

st.divider()

# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.container(border=True):
    st.subheader("🔬 AI-Assisted MRI Analysis")

    st.write(
        "Upload a brain MRI image to obtain a predicted classification "
        "across four categories. The application uses a fine-tuned "
        "EfficientNetB0 model trained using transfer learning."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "🧠 **4 Classes**\n\n"
            "Glioma, Meningioma, No Tumor and Pituitary"
        )

    with col2:
        st.info(
            "⚡ **Deep Learning**\n\n"
            "EfficientNetB0 Transfer Learning"
        )

    with col3:
        st.info(
            "📊 **Model Evaluation**\n\n"
            "Accuracy, Precision, Recall & F1-Score"
        )

# ============================================================
# MODEL STATUS
# ============================================================

st.success(
    f"🟢 Model Ready — {model_path.name} loaded successfully."
)

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):
    processed = preprocess_image(image)

    predictions = model.predict(processed, verbose=0)
    predictions = np.asarray(predictions)

    if predictions.ndim > 1:
        probabilities = predictions[0]
    else:
        probabilities = predictions

    probabilities = np.asarray(probabilities, dtype=float)

    # Convert logits to probabilities if required.
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(np.sum(probabilities), 1.0, atol=0.05)
    ):
        probabilities = tf.nn.softmax(probabilities).numpy()

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs, "
            f"but {len(CLASS_NAMES)} classes are configured."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    return predicted_class, confidence, probabilities

# ============================================================
# UPLOAD
# ============================================================

st.divider()

st.header("📤 Upload Brain MRI Image")

st.write(
    "Upload a JPG, JPEG, or PNG brain MRI image for "
    "AI-assisted classification."
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image.",
)

# ============================================================
# PREDICTION RESULT
# ============================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col_image, col_result = st.columns(2)

        with col_image:
            st.subheader("Uploaded MRI")
            st.image(
                image,
                caption="Uploaded MRI Image",
                use_container_width=True,
            )

        with st.spinner("Analyzing MRI image..."):
            predicted_class, confidence, probabilities = predict_image(image)

        with col_result:
            st.subheader("Prediction Result")

            if predicted_class == "No Tumor":
                st.success(f"Prediction: {predicted_class}")
            else:
                st.warning(f"Prediction: {predicted_class}")

            st.metric("Confidence", f"{confidence:.2f}%")

        # --------------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------------

        st.subheader("📊 Class Probabilities")

        probability_data = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(probability_data)

        st.subheader("Probability Details")

        for class_name, probability in probability_data.items():
            st.write(f"**{class_name}:** {probability * 100:.2f}%")
            st.progress(
                min(max(float(probability), 0.0), 1.0)
            )

    except Exception as e:
        st.error("Unable to process this MRI image.")
        st.exception(e)

else:
    st.info(
        "👆 Upload an MRI image above to start the classification."
    )

# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This application is an **AI-assisted research and educational
    prototype**. It is not a medical diagnostic system and should not
    be used as a substitute for professional medical evaluation,
    radiological interpretation, or clinical diagnosis.

    Model probabilities represent the output of the trained machine
    learning model and should not be interpreted as clinical certainty.
    """
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Brain Tumor MRI Classification Project • "
    "EfficientNetB0 Transfer Learning • TensorFlow • Streamlit"
)

st.caption("Research & Educational Prototype")
