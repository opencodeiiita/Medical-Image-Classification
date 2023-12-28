import streamlit as st
import tensorflow as tf
import base64
import time
from PIL import Image
import numpy as np
import json 
import requests 
from streamlit_lottie import st_lottie 
  
url = requests.get( 
    "https://lottie.host/0ea6f299-fe52-414e-8063-f54283c7c577/YTa5T7FZRz.json") 
# Creating a blank dictionary to store JSON file, 
# as their structure is similar to Python Dictionary 
url_json = dict() 

back = requests.get( 
    "https://lottie.host/dc8eb2d2-dc26-4bff-a908-20668f49437a/7rui3mxxXI.json") 
# Creating a blank dictionary to store JSON file, 
# as their structure is similar to Python Dictionary 
back_json = dict() 
  
if url.status_code == 200: 
    url_json = url.json() 
else: 
    print("Error in the URL") 

if back.status_code == 200: 
    back_json = back.json() 
else: 
    print("Error in the URL") 
# Load the saved model
model = tf.keras.models.load_model('MyCNN.h5')

# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((256, 256))  # Resize the image to match the model input size
    img_array = np.array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    return predictions

# Streamlit app code
st.set_page_config(
    page_title="Classification of Medical X-Rays",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar: 
    st_lottie(back_json , 
              height=350, 
              width=350)
st.sidebar.title("Classification of Medical X-Rays")
st.sidebar.write(
    "In recent years, the intersection of medical imaging and deep learning has witnessed unprecedented advancements, revolutionizing the landscape of healthcare. One notable application that has gained substantial attention is medical image classification using Convolutional Neural Networks (CNNs). As we embark on this project, we delve into the realm of leveraging cutting-edge deep learning techniques to augment traditional medical image analysis"
)
st.markdown(
    f'''
        <style>
            .sidebar {{
                width: 400px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

# Main content

st.title("Classification of Medical X-Rays")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image with border
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    # Perform prediction
    if st.button("Predict"):
        result = predict(uploaded_file)
        progress_bar = st.progress(0)
        status_text = st.empty()
          
         
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f'Progress: {i}%')
            time.sleep(0.00)
        
        status_text.text('Done!')
        
        # Display the prediction results
        st.write("Prediction Results:")
        prediction_label = "Normal" if result >= 0.5 else "Infected"
        if prediction_label == "Normal":
            st.balloons()
            st.success(f"The image is predicted as {prediction_label}")
        else :
            st.snow()
            st.warning(f'Warning : The image is predicted as {prediction_label}', icon="⚠️")
          
expander = st.expander("View Training , Validation and Testing Results")
expander.write("Following are the results : ")
expander.write("Confusion Matrix: ")                      
expander.image("./ConfusionMatrix.png", use_column_width=True)
expander.write("Graphs: ")    
expander.image("./Graphs.png", use_column_width=True)

# Footer
st.markdown("---")
st.write("Developed by Atharva Chavan")
st.write("Copyright © 2023. All rights reserved.")