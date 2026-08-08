import io
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from project_utils import IMG_SIZE, TARGET_CLASSES


# Load the best model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "best_model.keras"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)


# Create FastAPI application
app = FastAPI(
    title="AID Scene Classification API"
)


# Shared prediction function
def classify_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    best_index = int(
        np.argmax(probabilities)
    )

    return best_index, probabilities


# Check that the application is running
@app.get("/")
def root():
    return {
        "message": "AID Scene Classification API is running."
    }


# FastAPI prediction endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Could not read the image.",
        )

    best_index, probabilities = classify_image(
        image
    )

    return {
        "predicted_class": TARGET_CLASSES[
            best_index
        ],
        "class_index": best_index,
        "confidence": float(
            probabilities[best_index]
        ),
    }


# Gradio prediction
def gradio_predict(image):
    if image is None:
        raise gr.Error(
            "Please upload an image."
        )

    best_index, probabilities = classify_image(
        image
    )

    predicted_class = TARGET_CLASSES[
        best_index
    ]

    confidence = probabilities[
        best_index
    ]

    return (
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )


# Simple upload interface
interface = gr.Interface(
    fn=gradio_predict,

    inputs=gr.Image(
        type="pil",
        sources="upload",
        label="Upload an aerial image",
    ),

    outputs=gr.Textbox(
        label="Prediction result",
    ),

    title="AID Scene Classification",

    flagging_mode="never",
)


# Add Gradio to FastAPI
app = gr.mount_gradio_app(
    app,
    interface,
    path="/gradio",
)