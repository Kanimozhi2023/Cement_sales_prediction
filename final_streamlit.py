import pandas as pd
import streamlit as st
import numpy as np
import statsmodels 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from PIL import Image
#page configuration
image = Image.open('inno.png')
st.image(image,width = 600)
st.title('Time Series Cement Sales Forecasting Using Streamlit')
uploaded_file = st.file_uploader(r'dataset.xlsx', type=['xlsx'])
future_length = 15
if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    cement['month'] = cement['month'].apply(lambda x: x.strftime('%D-%M-%Y'))
    
    hwe_model_mul_add = ExponentialSmoothing(cement["sales"][:130], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
    
    newdata_pred = hwe_model_mul_add.predict(len(cement['sales']), len(cement['sales'])-3+future_length)
    
    
    st.subheader("For exponential model")
    pred = pd.DataFrame(newdata_pred, columns=['sales'])
    st.write("Sales Forecast: ", pred)

     
