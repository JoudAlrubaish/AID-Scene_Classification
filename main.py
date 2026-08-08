from pathlib import Path
import io
import gradio as gr
import numpy as np
import tensorflow as tf

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from PIL import (
    Image,
    UnidentifiedImageError,
)

from project_utils import (
    IMG_SIZE,
    TARGET_CLASSES,
)


# Resolve the model path relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "best_model.keras"
)


# Load the winning CNN model
model = tf.keras.models.load_model(
    MODEL_PATH
)


app = FastAPI(
    title="AID Scene Classification API",
    description=(
        "Classifies aerial images into one of "
        "the selected AID scene categories."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    """Check that the API is running."""
    return {
        "message": "AID Scene Classification API is running.",
        "model": model.name,
        "number_of_classes": len(TARGET_CLASSES),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    """Predict the aerial scene class of an uploaded image."""

    # Check the uploaded file content type
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    try:
        # Read the uploaded file
        image_bytes = await file.read()

        # Decode and prepare the image
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as an image.",
        )

    # Convert the image to a float32 NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Add the batch dimension:
    # (160, 160, 3) -> (1, 160, 160, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    # Generate class probabilities
    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    # Select the class with the highest probability
    best_index = int(
        np.argmax(probabilities)
    )

    predicted_class = TARGET_CLASSES[
        best_index
    ]

    confidence = float(
        probabilities[best_index]
    )

    return {
        "predicted_class": predicted_class,
        "class_index": best_index,
        "confidence": confidence,
    }
def gradio_predict(image):
    """Predict an uploaded image from the Gradio interface."""

    if image is None:
        raise gr.Error("Please upload an image.")

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

    predicted_class = TARGET_CLASSES[
        best_index
    ]

    confidence = float(
        probabilities[best_index]
    )

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(
            TARGET_CLASSES,
            probabilities,
        )
    }

    result_text = (
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    return class_probabilities, result_text
gradio_interface = gr.Interface(
    fn=gradio_predict,

    inputs=gr.Image(
        type="pil",
        label="Upload an aerial image",
    ),

    outputs=[
        gr.Label(
            num_top_classes=3,
            label="Top predictions",
        ),

        gr.Textbox(
            label="Prediction result",
            lines=2,
        ),
    ],

    title="AID Scene Classification",

    description=(
        "Upload an aerial image to classify it "
        "into one of the selected AID scene categories."
    ),

    flagging_mode="never",
)


app = gr.mount_gradio_app(
    app,
    gradio_interface,
    path="/gradio",
)