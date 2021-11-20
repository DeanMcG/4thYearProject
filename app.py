#Imports
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


#Read csv file into variable df
df = pd.read_csv("GOOG.csv")
current_df = pd.read_csv("GOOG.csv")
forecast_df = pd.read_csv("GOOG.csv")

#For debugging purposes
#print(df)
st.title('Stock Forecast')

#Select Stocks
stocks = ("GOOG", "TSLA", "AAPL")
st.selectbox('Select Stock To Forecast', stocks, index = 0)

#Slider Bar
st.select_slider("Years of Prediction", options = [1,2,3,4,5])

#Current Data Table
st.header("Current Data")
st.table(data = current_df.tail())

#Current Graph
def plot_current():
    st.header("Current Graph")
    current_df["Date"]=pd.to_datetime(current_df.Date,format="%Y-%m-%d")

    fig = plt.figure(figsize=(12,8))

    plt.plot(
        current_df["Date"],
        current_df["Close"],
    )

    plt.xlabel('Date')
    plt.ylabel('Close')

    st.write(fig)
plot_current()

#Forecast Data Table - currently same as current table
st.header("Current Data")
st.table(data = forecast_df.tail())

#Forecast Graph - currently same as current graph
def plot_forecast():
    st.header("Forecast Graph")
    forecast_df["Date"]=pd.to_datetime(forecast_df.Date,format="%Y-%m-%d")

    fig = plt.figure(figsize=(12,8))

    plt.plot(
        forecast_df["Date"],
        forecast_df["Close"],
    )

    plt.xlabel('Date')
    plt.ylabel('Close')

    st.write(fig)
plot_forecast()