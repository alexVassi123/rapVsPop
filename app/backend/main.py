from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
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


model_path = os.path.join("/app", "inference", "rap-versus-pop-classifier", "INPUT_model_path", "best_model.h5")
tokenizer_path = os.path.join("/app", "inference", "preprocessed_folder", "tokenizer.json")

model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer_json_str = f.read()
tokenizer = tokenizer_from_json(tokenizer_json_str)

class LyricsRequest(BaseModel):
    lyrics: str

api_router = APIRouter()

@api_router.post("/predict")
async def predict_genre(request: LyricsRequest):
    try:

        sequenced_lyrics = tokenizer.texts_to_sequences([request.lyrics])
        padded_lyrics = pad_sequences(sequenced_lyrics, maxlen=100, padding='post', truncating='post')
        
        prediction = model.predict(padded_lyrics)
        
        # Assuming binary classification: 0=rap, 1=pop
        genre = "pop" if prediction[0][0] > 0.5 else "rap"
        confidence = float(prediction[0][0] if genre == "pop" else 1 - prediction[0][0])
        
        return {
            "genre": genre,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(api_router)