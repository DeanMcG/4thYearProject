#Imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import hashlib

import sqlite3
connection = sqlite3.connect('user.db')
c = connection.cursor()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.preprocessing import MinMaxScaler

from matplotlib.pylab import rcParams

from sklearn.preprocessing import MinMaxScaler

import pymongo
from pymongo import MongoClient
import dns

#SQL Database Management
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	connection.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

#Hashlib - password security
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hash(password, hashed_text):
    if hash_password(password) == hashed_text:
        return hashed_text
    return False

#MongoDB stock database connection
def database(stock_selection):

    myclient = pymongo.MongoClient("mongodb+srv://admin:Password@stockcluster.cuzo7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    mydb = myclient["Stocks"]
    mycol = mydb[stock_selection]

    return mydb, mycol


#Populates pandas dataframe with stock information
def populateDataframe(mycol):
    df=pd.DataFrame(list(mycol.find()))
    df = df.drop('_id', axis = 1)
    df = df.sort_values(by="Date")

    return df

#Current Data Table
def currentTable(df):
    current_df = df
    current_df.reset_index(drop = True, inplace = True)
    st.header("Current Data")
    st.table(data = current_df.tail())

    return current_df


#Current Graph
def currentGraph(current_df, df):
    current_df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    current_df.index=current_df['Date']

    current_df['Close'] = current_df['Close'].astype(float)
    fig = plt.figure(figsize=(16,8))

    plt.plot(current_df["Close"],label='Close Price history')

    st.write(fig)


#LSTM model creation
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

#Forecast data
def forecast_graph(new_dataset, predicted_closing_price):
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

def main():

    menu = ['Home','Login','Signup']
    choice = st.sidebar.selectbox('Menu', menu)



    if choice == 'Home':
        st.title('Home')
        st.warning('Please sign up or log in to view dashboard, thanks!')


    elif choice == 'Login':
        st.subheader('Login')
        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password', type = 'password')
        if st.sidebar.checkbox('Login'):
            create_usertable()
            hashed_password = hash_password(password)
            result = login_user(username, check_hash(password, hashed_password))
            if result:
                st.success('Logged In as {}'.format(username))

                #Title
                st.title('Stock Forecast')

                #Select Stocks
                stocks = ("GOOG", "TSLA", "AAPL")
                stock_selection = st.selectbox('Select Stock To Forecast', stocks, index = 0)

                mydb, mycol = database(stock_selection)
                populateDataframe(mycol)
                df = populateDataframe(mycol)
                current_df = currentTable(df)
                currentGraph(current_df, df)

                generate_forecast = st.button('Generate Forecast')
                if generate_forecast:
                    new_dataset, predicted_closing_price = createModel(df)
                    forecast_graph(new_dataset, predicted_closing_price)
            else:
                st.warning('Incorrect Username or Password')
            

    elif choice == 'Signup':
        st.subheader('Signup')
        new_user = st.text_input('Username')
        new_password = st.text_input('Password', type='password')

        if st.button('Signup'):
            create_usertable()
            add_userdata(new_user, hash_password(new_password))
            st.success('Account created successfully!')
            st.balloons()
            st.info('Please go to login menu to use new account')


if __name__ == '__main__':
    main()
