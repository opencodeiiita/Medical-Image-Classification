import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import cv2
import numpy as np

    # Load the ResNet50 model
model = keras.models.load_model("tlear_resnet50_model.h5")


if __name__ == "__main__":
        st.title("Keras Model Deployment")
        st.write("Upload an image and click on the button to make predictions.")

        # Upload image through Streamlit
        img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if img is not None:
            # Display the uploaded image
            st.image(img, caption="Uploaded Image.", use_column_width=True)
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

            img = np.array(img)
            img = cv2.resize(img,(256,256))

            img = preprocess_input(img)
            img = np.reshape(img,(1,256,256,3))
            predictions = model.predict(img)
        
            if predictions>0.5:
                st.write("Predictions:","INFECTED")
            else:
                st.write("Predictions:","NORMAL")
