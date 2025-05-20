import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = 'image_cnn_classifier.h5'

model = load_model(MODEL_PATH)

class_names = ['anu','bharti','deepak','manidhar','sudh']

st.set_page_config(page_title = "🖼️ Image Classification App",layout = 'centered',  page_icon="📷")

st.sidebar.title("upload your image")

st.markdown("""
This application uses a **Convolutional Neural Network (CNN)** to classify images into one of the following categories:
- Anusha
- Bharat
- Naveen
- Suresh
- Vishal
""")

upload_file = st.sidebar.file_uploader("choose your image" ,  type = ["jpg" , 'jpeg','png'])

from PIL import Image

# this will upload the image in 3 channels RGB format in streamlit application
if upload_file is not None :  
    img = Image.open(upload_file).convert('RGB')
    st.image(img,caption="your image")
    
    image_resized = img.resize((128,128))  #pre-processing
    img_array = image.img_to_array(image_resized)/255.0  #converting the image to numpy array
    image_batch = np.expand_dims(img_array,axis=0)  
    
    prediction = model.predict(image_batch)  #prediction 
    predicted_class = class_names[np.argmax(prediction)]  #predicting the highest probability class
    
    st.success(f"This image is predicted to be :{predicted_class}")
    
    st.subheader("Welcome to Image Classifier! 👋")
    print(prediction)
    for index,score in  enumerate(prediction[0]):
        st.write(f"{class_names[index]}: {score}")

    st.markdown("---")
st.markdown("🔍 Built with Streamlit & TensorFlow | 🧠 CNN Architecture")