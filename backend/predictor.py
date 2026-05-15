import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


model = load_model("/models/Brain_tumour_model.h5")

categories = [
    'pituitary',
    'no_tumor',
    'glioma',
    'meningioma']

def predict_mri(image):
    image = cv2.resize(image, (224,224))
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_class = categories[predicted_index]
    confidence = float(np.max(prediction))
    return predicted_class, confidence