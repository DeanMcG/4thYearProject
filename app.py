#Imports
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

#Read csv file into variable df
df = pd.read_csv("GOOG.csv")

#For debugging purposes
#print(df)
st.title('Stock Forecast')

stocks = ("GOOG", "TSLA", "AAPL")
st.selectbox('Select Stock To Forecast', stocks, index = 0)

st.select_slider("Years of Prediction", options = [1,2,3,4,5])

st.header("Current Data")
st.table(data = df.tail())

st.header("Current Graph")

