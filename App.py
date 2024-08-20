import streamlit as st 
import numpy as np 
import cv2 
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Intel Image Classification")

Model = load_model("model.h5")


img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if img is not None:

    newimg = Image.open(img)
    st.image(newimg, caption="Uploaded Image")

    image = np.array(newimg)
    image = cv2.resize(image, (100, 100))  
    image = image.astype("float32") / 255  
    image = np.expand_dims(image, axis=0)  

    if st.button('Predict'):
        pred = np.argmax(Model.predict(image), axis=1)
        categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        st.write("Prediction is",categories[pred[0]])
else:
    st.write("You should add an image") 
