import streamlit as st
import pandas as pd
import os
from PIL import Image


st.title("Depth Estimation Tool")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "test-imgs")
CSV_PATH = os.path.join(IMAGE_FOLDER, "results_detections.csv")

# predictions
predictions = pd.read_csv(CSV_PATH)
st.write("### predictions")
st.write(predictions)

# images
images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images.sort()
st.write("### images")
st.write(images)

