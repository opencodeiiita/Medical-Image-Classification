import streamlit as st
import tensorflow as tf
import base64
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('MyCNN.h5')

# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((256, 256))  # Resize image to match model input size
    img_array = np.array(img)  # Convert image to NumPy array
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    

    return predictions

# Streamlit app code
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")
st.title("Deep Learning")
st.header("Medical Image Classification")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-opacity: 0.5;
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0.5);
}}

[data-testid="stTitle"] {{
background: rgba(255,255,255,0.5);
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Perform prediction
    result = predict(uploaded_file)

    # Display the prediction results
    st.write("Prediction Results:")
    if result>=0.5 :
        st.write("Normal : 1")
    else:
        st.write("Infected : 0")
