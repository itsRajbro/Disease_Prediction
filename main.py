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
    # "skin": {
    #     "path": "models/skin_model.h5",
    #     "classes": ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"],
    #     "img_size": (224, 224)
    # },
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
    # Load cardio model and its scaler
    models["cardio"] = joblib.load("model/cardio_model.joblib")
    models["cardio_scaler"] = joblib.load("model/cardio_scaler.joblib")
    logger.info("Loaded cardio model and scaler")

def preprocess_image(image_bytes, img_size):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_cardio_input(df, scaler):
    # Data cleaning and feature engineering as in training
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df = pd.get_dummies(df, columns=['cholesterol', 'gluc'], drop_first=True)
    # Ensure all columns match training
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    # Scale numerical features
    num_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    df[num_features] = scaler.transform(df[num_features])
    return df

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    # Check if file is an image
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        best_prediction = None
        max_confidence = 0.0
        for model_name, model in models.items():
            if model_name in ["cardio", "cardio_scaler"]:
                continue
            try:
                config = MODELS[model_name]
                input_data = preprocess_image(contents, config["img_size"])
                prediction = model.predict(input_data)
                confidence = float(np.max(prediction))
                class_index = int(np.argmax(prediction, axis=1)[0])
                logger.info(f"{model_name} prediction confidence: {confidence:.4f}")
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_prediction = {
                        # "disease_type": model_name.upper(),
                        "diagnosis": config["classes"][class_index],
                        "confidence": round(confidence, 4)
                    }
            except Exception as e:
                logger.error(f"Error in {model_name} model: {str(e)}")
                continue
        if not best_prediction:
            return {"error": "Could not process image with any available model"}
        logger.info(f"Best image prediction: {best_prediction}")
        return best_prediction

    # Otherwise, assume it's a CSV/text for cardio
    try:
        # Read uploaded CSV/text as DataFrame
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp.flush()
            df = pd.read_csv(tmp.name)
        scaler = models["cardio_scaler"]
        model = models["cardio"]
        X = preprocess_cardio_input(df, scaler)
        prediction = model.predict_proba(X)
        confidence = float(np.max(prediction))
        class_index = int(np.argmax(prediction, axis=1)[0])
        logger.info(f"Cardio prediction confidence: {confidence:.4f}")
        return {
            "disease_type": "CARDIO",
            "diagnosis": "Disease" if class_index == 1 else "No Disease",
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        logger.error(f"Error in cardio model: {str(e)}")
        return {"error": "Could not process file as image or cardio data"}

