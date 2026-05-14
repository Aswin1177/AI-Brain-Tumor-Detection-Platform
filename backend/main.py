from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from predictor import predict_mri
from reasoning import generate_reasoning

app = FastAPI()
@app.get("/")
def home():
    return {
        "message": "Brain Tumor Detection API Running"
    }
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "error": "Invalid image"
        }

    predicted_class, confidence = predict_mri(image)
    reasoning = generate_reasoning(
        predicted_class,
        confidence
    )
    return {
        "prediction": predicted_class,

        "confidence": round(confidence * 100, 2),
        
        "reasoning": reasoning
    }