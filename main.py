import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from model import *

st.title('Cat Dog Classification')
st.text('Author: Võ Đình Đạt')
st.write('***')
image = st.file_uploader('Upload Picture', type=["png", "jpg", "jpeg"])
try:
    if image is not None:
        st.image(image)
        st.header("Probability: ")
        prob = predict(Image.open(image))[0]
        prob = np.exp(prob) / np.sum(np.exp(prob))
        st.markdown(f"""
            ```
            Cat: {prob[0]*100:.2f}%
            Dog: {prob[1]*100:.2f}%
            ```
        """)
        if prob[0] > prob[1]:
            st.header("-> This is a Cat")
        else:
            st.header("-> This is a Dog")
except:
    st.text('Error: Picture must be RGB (3 channels), not RBGA (4 channels)!')    






