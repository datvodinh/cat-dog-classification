import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from demo_model import *
import time

st.title('Cat Dog Classification')
st.text('Author: Võ Đình Đạt')


st.write('***')
st.header("Upload Image: ")
old_option = None
image = st.file_uploader('Upload Picture', type=["png", "jpg", "jpeg"])

if image is not None:
    st.image(image)

st.write('***')

st.header("Select Model: ")

option = st.selectbox(
    'Choose Model?',
    ("None",'resnet18', 'cnn'))

if image is not None and option!="None":
    if old_option != option:
        
        st.header("Probability: ")
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.0001)
            my_bar.progress(percent_complete + 1)

        prob = predict(Image.open(image),option)[0]
        prob = np.exp(prob) / np.sum(np.exp(prob))
        st.markdown(f"""
            ```
            Cat: {prob[0]*100:.2f}%
            Dog: {prob[1]*100:.2f}%
            ```
        """)
        if prob[0] > prob[1]:
            st.header("=> This is a Cat 😼")
        else:
            st.header("=> This is a Dog 🐶")
        old_option = option
# except:
#     st.text('Error: Picture must be RGB (3 channels), not RBGA (4 channels)!')    






