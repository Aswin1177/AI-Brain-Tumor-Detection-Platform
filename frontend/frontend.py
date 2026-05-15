import streamlit as st
import requests
import numpy as np
import cv2
st.title("Brain Tumour Detection and Reasoning")
uploaded=st.file_uploader("Upload MRI image",type=['jpg','png','jpeg'])
if uploaded is not None:
    image=np.asarray(bytearray(uploaded.read()),dtype=np.uint8)
    image=cv2.imdecode(image,1)
    st.image(image, caption="Uploaded image", use_column_width=True)
    response = requests.post(
        "http://13.207.1.57:8000/predict",
        files={
            "file": uploaded.getvalue()
        }
    )
    result=response.json()
    st.write(
        "Prediction:",
        result["prediction"]
    )
    st.write(
        "Confidence:",
        result["confidence"],
        "%"
    )
    st.write("Reasoning:")
    st.write(result["reasoning"])