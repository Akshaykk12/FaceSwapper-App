import streamlit as st
import numpy as np
from PIL import Image
import cv2 
import insightface 
from insightface.app import FaceAnalysis
import tempfile
import os
import base64

app = FaceAnalysis(name="./buffalo_l/")  
app.prepare(ctx_id=0, det_size=(640,640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)

# Function to process images using your machine learning model
def generate_output(poster, facial):
    # Convert NumPy arrays to image files
    poster_path = save_temp_image(poster)
    facial_path = save_temp_image(facial)

    poster = cv2.imread(poster_path)
    facial = cv2.imread(facial_path)

    facial_faces = app.get(facial)

    # Check if facial_faces is not empty before accessing elements
    if facial_faces:
        facial_face = facial_faces[0]
        bbox = facial_face['bbox']
        bbox = [int(b) for b in bbox]

        faces = app.get(poster)

        for face in faces:
            poster = swapper.get(poster, face, facial_face, paste_back=True)

        # Use a temporary file for the result
        result_fd, result_path = tempfile.mkstemp(suffix=".jpg")
        os.close(result_fd)

        cv2.imwrite(result_path, poster)

        # Remove temporary image files
        os.remove(poster_path)
        os.remove(facial_path)

        return result_path
    else:
        st.warning("No faces found in the provided facial image.")
        return None
    
# Function to create a download link
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'

# Function to save a NumPy array as a temporary image file
def save_temp_image(image):
    temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(temp_fd)

    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return temp_path

# Streamlit app
st.title("Face Swapper App")

col1, col2, col3= st.columns([3,1,3])

# Upload two images
uploaded_image1 = col1.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
uploaded_image2 = col3.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

# Display the uploaded images
if uploaded_image1 is not None and uploaded_image2 is not None:
    col1.image([Image.open(uploaded_image1)], caption=["Background Image"])
    col3.image([Image.open(uploaded_image2)], caption=["Face Image"])

# Generate button
if st.button("Generate"):
    if uploaded_image1 is not None and uploaded_image2 is not None:
        # Load images
        image1 = np.array(Image.open(uploaded_image1))
        image2 = np.array(Image.open(uploaded_image2))

        # Generate output using your machine learning model
        generated_output = generate_output(image1, image2)

        if generated_output:
            # Display the generated image using PIL
            st.image(Image.open(generated_output), caption="Generated Image", width=200)

            # Add a download button
            st.markdown(get_binary_file_downloader_html(generated_output, 'Download Image'), unsafe_allow_html=True)
    else:
        st.warning("Please upload both Background Image and Face Image before generating.")