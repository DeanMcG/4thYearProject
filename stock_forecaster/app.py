#Imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pyrebase


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.preprocessing import MinMaxScaler

from matplotlib.pylab import rcParams

from sklearn.preprocessing import MinMaxScaler

import pymongo
from pymongo import MongoClient
import dns





firebaseConfig = {
  'apiKey': "AIzaSyBSS49dtXl5xqejyYX698iKhQRXp-GWvC4",
  'authDomain': "stockforecaster-2e532.firebaseapp.com",
  'projectId': "stockforecaster-2e532",
  'databaseURL' : "https://console.firebase.google.com/project/stockforecaster-2e532/database/stockforecaster-2e532-default-rtdb/data/~2F",
  'storageBucket': "stockforecaster-2e532.appspot.com",
  'messagingSenderId': "929676977143",
  'appId': "1:929676977143:web:2cf2c4df5c035abae584f2",
  'measurementId': "G-ZD4VT06YFJ"
};

#Authentication with Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

#Firebase database
db = firebase.database()
storage = firebase.storage()

st.sidebar.title("Stock Forecaster")

#Authenticate User
choice = st.sidebar.selectbox('Signup/Login', ['Sign Up', 'Login'])
email = st.sidebar.text_input('Please Enter Email Address')
password = st.sidebar.text_input('Please Enter Password', type = 'password')

if choice == 'Sign Up':
    username = st.sidebar.text_input('Please Input Your Username', value= 'Eg.Steve123')
    submit = st.sidebar.button('Create Account')

    if submit:
        user = auth.create_user_with_email_and_password(email,password)
        st.success('Account Created')
        st.balloons()

        user = auth.sign_in_with_email_and_password(email,password)
        db.child(user['localId']).child("Username").set(username)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title('Welcome' + username)
        st.info('Login using dropdown')
if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email,password)

#Title
st.title('Stock Forecast')

#Select Stocks
stocks = ("GOOG", "TSLA", "AAPL")
stock_selection = st.selectbox('Select Stock To Forecast', stocks, index = 0)


#Slider Bar
#st.select_slider("Years of Prediction", options = [1,2,3,4,5])







def database():

    myclient = pymongo.MongoClient("mongodb+srv://admin:Password@stockcluster.cuzo7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    mydb = myclient["Stocks"]
    mycol = mydb[stock_selection]

    return mydb, mycol


def populateDataframe(mycol):
    df=pd.DataFrame(list(mycol.find()))
    df = df.drop('_id', axis = 1)
    df = df.sort_values(by="Date")

    return df

#Current Data Table
def currentTable():
    current_df = df
    current_df.reset_index(drop = True, inplace = True)
    st.header("Current Data")
    st.table(data = current_df.tail())

    return current_df


#Current Graph
def currentGraph(current_df):
    current_df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    current_df.index=current_df['Date']

    current_df['Close'] = current_df['Close'].astype(float)
    fig = plt.figure(figsize=(16,8))

    plt.plot(current_df["Close"],label='Close Price history')

    st.write(fig)



def createModel(df):
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


    return new_dataset, predicted_closing_price


def forecast_graph():
    train_data=new_dataset[:935]
    valid_data=new_dataset[935:]
    valid_data['Predictions']=predicted_closing_price

    st.header("Forecast Data")
    st.table(data = valid_data.tail())

    fig1 = plt.figure(figsize=(16,8))

    plt.plot(train_data["Close"], label="Train Data")
    plt.plot(valid_data['Close'], label="Actual Closing Price")
    plt.plot(valid_data['Predictions'], label="Predicted Closing Price")                    
    plt.legend(loc="upper left")

    st.write(fig1)

mydb, mycol = database()
populateDataframe(mycol)
df = populateDataframe(mycol)
current_df = currentTable()
currentGraph(current_df)

generate_forecast = st.button('Generate Forecast')
if generate_forecast:
    new_dataset, predicted_closing_price = createModel(df)
    forecast_graph()
