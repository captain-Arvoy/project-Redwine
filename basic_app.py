import streamlit as st
import pandas as pd
import numpy as np
st.title("Learning project -2")
DATA_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
nav = st.sidebar.radio("Navigation",["Home","EDA","visualisation"])
def load_data(nrows):
    data = pd.read_csv(DATA_URL,nrows=nrows)
    lowercase = lambda x: str(x).lower()
    return data
if nav == "Home":
    st.write("Home")
if nav == "EDA":
    st.write("EDA")
if nav == "visualisation":
    st.write("visualisation")
    data_load_state = st.text('Loading data...')
    data = load_data(1000)
    data_load_state.text('')
    st.subheader('Raw data')
    st.write(data)




