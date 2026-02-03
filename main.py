from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
import keras
import joblib
import pandas as pd
import logging
import tempfile
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler


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
MODELS: Dict[str, Dict[str, Any]] = {
    "cxr": {
        "path": "model/cxr_model.h5",
        "classes": ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"],
        "img_size": (224, 224),
        "grayscale": False
    },
    "malaria": {
        "path": "model/malaria_model.keras",
        "classes": ["Parasitized", "Uninfected"],
        "img_size": (224, 224),
        "grayscale": True  # ConvNeXt expects RGB but malaria model uses grayscale
    },
    "ocular": {
        "path": "model/ocular_model.keras",
        "classes": ["A", "C", "D", "G", "H", "M", "N", "O"],
        "img_size": (224, 224),
        "grayscale": False
    }
}

models: Dict[str, Any] = {}

def preprocess_image(image_bytes: bytes, img_size: tuple, grayscale: bool = False) -> np.ndarray:
    """Preprocess images with configurable color mode"""
    try:
        mode = "L" if grayscale else "RGB"
        img = Image.open(io.BytesIO(image_bytes)).convert(mode)
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        
        # Add channel dimension if grayscale
        if grayscale and len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
            
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ValueError("Invalid image file")

@app.on_event("startup")
async def load_models():
    """Load all models at startup with proper error handling"""
    for disease_type, config in MODELS.items():
        model_path = config["path"]
        try:
            if model_path.endswith(".keras"):
                models[disease_type] = keras.saving.load_model(model_path)
            else:
                import tensorflow as tf
                models[disease_type] = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded {disease_type} model")
        except Exception as e:
            logger.error(f"Failed to load {disease_type} model: {str(e)}")
            raise

    # Load cardio model if available
    logger.info("Cardio model disabled (image-based prediction only)")


def preprocess_cardio_input(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Process cardiovascular input data"""
    try:
        df['bmi'] = df['weight'] / (df['height']/100)**2
        df = pd.get_dummies(df, columns=['cholesterol', 'gluc'], drop_first=True)
        
        # Ensure expected columns
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        # Scale features
        num_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        df[num_features] = scaler.transform(df[num_features])
        return df[expected_cols]
    except Exception as e:
        logger.error(f"Cardio preprocessing failed: {str(e)}")
        raise

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Unified prediction endpoint for all models"""
    contents = await file.read()
    filename = file.filename.lower()

    if filename.endswith(('.jpg', '.jpeg', '.png')):
        best_prediction = None
        max_confidence = 0.0

        for model_name, model in models.items():
            if model_name in ["cardio", "cardio_scaler"]:
                continue

            config = MODELS.get(model_name)
            if not config:
                continue

            try:
                # Get preprocessing parameters from config
                grayscale = config.get("grayscale", False)
                input_data = preprocess_image(contents, config["img_size"], grayscale)
                
                # Validate input shape
                expected_shape = (1, *config["img_size"], 1 if grayscale else 3)
                if input_data.shape != expected_shape:
                    logger.error(f"Input shape mismatch for {model_name}. Expected {expected_shape}, got {input_data.shape}")
                    continue

                prediction = model.predict(input_data)
                confidence = float(np.max(prediction))
                class_index = int(np.argmax(prediction, axis=1)[0])
                
                logger.info(f"{model_name} prediction: {config['classes'][class_index]} (confidence: {confidence:.4f})")

                if confidence > max_confidence:
                    max_confidence = confidence
                    best_prediction = {
                        "diagnosis": config["classes"][class_index],
                        "confidence": round(confidence, 4),
                        "model_used": model_name.upper()
                    }

            except Exception as e:
                logger.error(f"{model_name} model failed: {str(e)}", exc_info=True)
                continue

        return best_prediction or {"error": "No valid predictions from any model"}

    # Handle CSV/cardio prediction
    return {"error": "Only image-based prediction is supported"}

