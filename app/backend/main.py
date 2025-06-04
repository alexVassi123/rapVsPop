from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# __file__ is inside: app/backend/
current_file_dir = os.path.dirname(__file__)

# Go up THREE levels: backend → app → rapVsPop → project root
project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

model_path = os.path.join(
    project_root,
    "inference",
    "rap-versus-pop-classifier",
    "INPUT_model_path",
    "best_model.h5"
)

model = tf.keras.models.load_model(model_path)

class LyricsRequest(BaseModel):
    lyrics: str

@app.post("/predict")
async def predict_genre(request: LyricsRequest):
    try:
        prediction = model.predict([request.lyrics])
        
        # Assuming binary classification: 0=rap, 1=pop
        genre = "pop" if prediction[0][0] > 0.5 else "rap"
        confidence = float(prediction[0][0] if genre == "pop" else 1 - prediction[0][0])
        
        return {
            "genre": genre,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}