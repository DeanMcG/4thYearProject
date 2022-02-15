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

df=pd.read_csv("GOOG5.csv")
#df=pd.DataFrame(list(mycol.find()))
#df = df.drop('_id', axis = 1)

current_df =pd.read_csv("GOOG5.csv")
#current_df=pd.DataFrame(list(mycol.find()))
#current_df = current_df.drop('_id', axis = 1)

#forecast_df=pd.read_csv("GOOG5.csv")
#forecast_df=pd.DataFrame(list(mycol.find()))
#forecast_df = forecast_df.drop(['_id'], axis = 1)



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

data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]


new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)
final_dataset=new_dataset.values

train_data=final_dataset[0:935,:]
valid_data=final_dataset[935:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

lstm_model.save("saved_lstm_model.h5")

fig = plt.figure(figsize=(12,8))

train_data=new_dataset[:935]
valid_data=new_dataset[935:]
valid_data['Predictions']=predicted_closing_price


forecast_df = valid_data

#Forecast Data Table
st.header("Forecast Data")
st.table(data = forecast_df.tail())

#Forecast Graph
def plot_forecast():
    st.header("Forecast Graph")
    


    forecast_df['Close'] = forecast_df['Close'].astype(float)

    fig1 = plt.figure(figsize=(12,8))

    
    plt.plot(train_data["Close"], label="Train Data")
    plt.plot(forecast_df['Close'], label="Actual Closing Price")
    plt.plot(forecast_df['Predictions'], label="Predicted Closing Price")                    
    plt.legend(loc="upper left")

    plt.xlabel('Date')
    plt.ylabel('Close')

    st.write(fig1)
plot_forecast()