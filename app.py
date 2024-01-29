import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('tomato_model.h5')

# Image preprocessing function
def preprocess_image(image): 
    image = image.resize((256, 256))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.title('Image Classification')

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
c1, c2= st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)


    preprocessed_image = preprocess_image(image)
    c1.header('Input Image')
    c1.image(preprocessed_image)
    # c1.write(preprocessed_image.shape)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    class_Name = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
    predicted_class = np.argmax(predictions[0])
    # print(predicted_class)
    # print(class_Name[predicted_class])

    c2.header('Output')
    c2.subheader('Predicted class :')
    c2.write(class_Name[predicted_class])
    
