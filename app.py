#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.preprocessing import MinMaxScaler

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler

import pymongo
from pymongo import MongoClient
import dns




#Read csv file into variable df

myclient = pymongo.MongoClient("mongodb+srv://admin:Password@stockcluster.cuzo7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
mydb = myclient["Stocks"]
mycol = mydb["GOOG"]

df=pd.DataFrame(list(mycol.find()))
df = df.drop('_id', axis = 1)

current_df=pd.DataFrame(list(mycol.find()))
current_df = current_df.drop('_id', axis = 1)

forecast_df=pd.DataFrame(list(mycol.find()))
forecast_df = forecast_df.drop(['_id'], axis = 1)



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

    current_df['Close'] = current_df['Close'].astype(float)

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
st.header("Forecast Data")
st.table(data = forecast_df.tail())

#Forecast Graph - currently same as current graph
def plot_forecast():
    st.header("Forecast Graph")
    forecast_df["Date"]=pd.to_datetime(forecast_df.Date,format="%Y-%m-%d")

    forecast_df['Close'] = forecast_df['Close'].astype(float)

    fig = plt.figure(figsize=(12,8))

    plt.plot(
        forecast_df["Date"],
        forecast_df["Close"],
    )

    plt.xlabel('Date')
    plt.ylabel('Close')

    st.write(fig)
plot_forecast()