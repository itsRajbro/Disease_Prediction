from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
import keras
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODELS = {
    "cxr": {
        "path": "model/cxr_model.h5",
        "classes": ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"],
        "img_size": (224, 224)
    },
    "malaria": {
        "path": "model/malaria_model.keras",
        "classes": ["Parasitized", "Uninfected"],
        "img_size": (224, 224)
    },
    "ocular": {
        "path": "model/ocular_model.keras",
        "classes": ["A", "C", "D", "G", "H", "M", "N", "O"],
        "img_size": (224, 224)
    }
}

models = {}

# Load all models at startup
@app.on_event("startup")
async def load_models():
    for disease_type, config in MODELS.items():
        model_path = config["path"]
        if model_path.endswith(".keras"):
            models[disease_type] = keras.saving.load_model(model_path)
        else:
            import tensorflow as tf
            models[disease_type] = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded {disease_type} model from {model_path}")
def preprocess_image(image_bytes, img_size, grayscale=False):
    """Handle both RGB and grayscale preprocessing"""
    mode = "L" if grayscale else "RGB"
    img = Image.open(io.BytesIO(image_bytes)).convert(mode)
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel for grayscale
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    best_prediction = None
    max_confidence = 0.0

    for model_name, model in models.items():
        if model_name in ["cardio", "cardio_scaler"]:
            continue
        config = MODELS[model_name]
        try:
            # Use grayscale preprocessing if configured
            input_data = preprocess_image(
                contents, 
                config["img_size"], 
                grayscale=config.get("grayscale", False)
            )
            prediction = model.predict(input_data)
            confidence = float(np.max(prediction))
            class_index = int(np.argmax(prediction, axis=1)[0])
            
            logger.info(f"{model_name} prediction confidence: {confidence:.4f}")
            
            if confidence > max_confidence:
                max_confidence = confidence
                best_prediction = {
                    "disease_type": model_name.upper(),
                    "diagnosis": config["classes"][class_index],
                    "confidence": round(confidence, 4)
                }
        except Exception as e:
            logger.error(f"Error in {model_name} model: {str(e)}")
            continue

    return best_prediction or {"error": "No valid predictions"}
