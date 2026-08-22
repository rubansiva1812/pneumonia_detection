"""
Pneumonia Detection — Streamlit App

Loads the serialized best model (model/best_model.keras) and its
deployment_config.json (written by the final notebook's Section 14),
lets the user upload a chest X-ray (.dcm, .png, or .jpg), preprocesses it
with the SAME pipeline used during training, and displays the predicted
class with per-class probabilities.

IMPORTANT: this app does not hard-code image size / channel / normalization
choices — it reads them from deployment_config.json so it always matches
whatever model was actually selected as best in the notebook.
"""

import json
import os

import numpy as np
import streamlit as st
import cv2
import pydicom
from PIL import Image
import tensorflow as tf

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "deployment_config.json")


@st.cache_resource
def load_model_and_config():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, config


def normalize(img_uint8: np.ndarray, scheme: str) -> np.ndarray:
    """Mirrors the notebook's normalize_for_scratch_cnn / normalize_for_vgg16 /
    normalize_for_resnet50 functions exactly — must stay in sync with
    Section 3.4 of the notebook if that logic ever changes."""
    if scheme == "rescale":
        return img_uint8.astype(np.float32) / 255.0
    elif scheme == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(img_uint8.astype(np.float32))
    elif scheme == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(img_uint8.astype(np.float32))
    else:
        raise ValueError(f"Unknown normalization scheme: {scheme}")


def load_and_preprocess(uploaded_file, target_size: int, as_rgb: bool, norm_scheme: str):
    """Handles .dcm as well as standard image formats, so the tool is usable
    even when a rendered PNG/JPG export of an X-ray is all that's available,
    though .dcm is the format the model was actually trained on."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        dcm = pydicom.dcmread(uploaded_file)
        pixels = dcm.pixel_array.astype(np.float32)
        display_img = pixels.copy()
        pixels = (
            (pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-8) * 255.0
        ).astype(np.uint8)
    else:
        pil_img = Image.open(uploaded_file).convert("L")  # force grayscale
        pixels = np.array(pil_img)
        display_img = pixels.copy()

    resized = cv2.resize(pixels, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if as_rgb:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        resized = resized[..., np.newaxis]

    normalized = normalize(resized, norm_scheme)
    return display_img, np.expand_dims(normalized, axis=0)


def main():
    st.set_page_config(page_title="Pneumonia Detection — Screening Support", layout="centered")
    st.title("Chest X-Ray Pneumonia Detection")

    st.warning(
        "**Disclaimer:** this is an AI-assisted screening / decision-support "
        "tool intended to help prioritize cases for review. It is **not** a "
        "diagnostic device and does not replace evaluation by a qualified "
        "radiologist or physician."
    )

    model, config = load_model_and_config()

    if model is None:
        st.error(
            f"No model found at `{MODEL_PATH}`. Copy `best_model.keras` and "
            f"`deployment_config.json` (produced by Section 13/14 of the "
            f"final notebook) into the `model/` folder before running this app."
        )
        st.stop()

    st.caption(f"Loaded model: **{config['model_name']}**")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image (.dcm, .png, or .jpg)",
        type=["dcm", "png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        display_img, model_input = load_and_preprocess(
            uploaded_file,
            target_size=config["target_size"],
            as_rgb=config["as_rgb"],
            norm_scheme=config["normalize"],
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(display_img, caption="Uploaded X-ray", clamp=True, use_column_width=True)

        with st.spinner("Running inference..."):
            probs = model.predict(model_input, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        class_names = config["class_names"]

        with col2:
            st.subheader(f"Prediction: {class_names[pred_idx]}")
            st.metric("Confidence", f"{probs[pred_idx] * 100:.1f}%")

        st.subheader("Class Probabilities")
        for cname, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            st.write(f"**{cname}**")
            st.progress(float(p))
            st.caption(f"{p * 100:.2f}%")

        st.info(
            "Remember: this prediction is a screening aid only. Clinical "
            "decisions should be made by a qualified healthcare professional "
            "using full clinical context, not this tool's output alone."
        )


if __name__ == "__main__":
    main()
