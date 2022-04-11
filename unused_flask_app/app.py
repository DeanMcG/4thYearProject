from flask import Flask, render_template, session, redirect, Response
from functools import wraps
import pymongo
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from sklearn.preprocessing import MinMaxScaler

from matplotlib.pylab import rcParams

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf
from plotly import graph_objs as go
from datetime import date

app = Flask(__name__)
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'

# Database

client = pymongo.MongoClient("mongodb+srv://admin:Password@stockcluster.cuzo7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.user_login


# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap


# Routes
from user import routes

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/dashboard/')
@login_required
def dashboard():
  return render_template('dashboard.html')

# @app.route('/plot.png')
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig


@app.route('/current')
def loadStockData():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = yf.download("GOOG", START, TODAY )
    data.reset_index(inplace = True)
    return data


def showCurrentTable(data):
    current_df = data
    fig = current_df.tail()
    fig.show()



# def database():

#     myclient = pymongo.MongoClient("mongodb+srv://admin:Password@stockcluster.cuzo7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
#     mydb = myclient["Stocks"]
#     mycol = mydb["GOOG"]

#     return mydb, mycol

# def stockSelector(mydb):

#     if stock_selection == "GOOG":
#         mycol = mydb["GOOG"]
#     elif stock_selection == "AAPL":
#         mycol = mydb["AAPL"]

#     return mydb, mycol

# def populateDataframe(mycol):
#     df=pd.DataFrame(list(mycol.find()))
#     df = df.drop('_id', axis = 1)
#     df = df.sort_values(by="Date")

#     return df

# #Current Data Table
# @app.route('/currentTable')
# def currentTable():
#     current_df = df
#     current_df.reset_index(drop = True, inplace = True)
#     st.header("Current Data")
#     st.table(data = current_df.tail())

#     return current_df


# #Current Graph
# def currentGraph(current_df):
#     current_df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
#     current_df.index=current_df['Date']

#     current_df['Close'] = current_df['Close'].astype(float)
#     fig = plt.figure(figsize=(16,8))

#     plt.plot(current_df["Close"],label='Close Price history')

#     st.write(fig)



# def createModel(df):
#     data=df.sort_index(ascending=True,axis=0)
#     new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

#     for i in range(0,len(data)):
#         new_dataset["Date"][i]=data['Date'][i]
#         new_dataset["Close"][i]=data["Close"][i]


#     new_dataset.index=new_dataset.Date
#     new_dataset.drop("Date",axis=1,inplace=True)
#     final_dataset=new_dataset.values

#     train_data=final_dataset[0:935,:]
#     valid_data=final_dataset[935:,:]

#     scaler=MinMaxScaler(feature_range=(0,1))
#     scaled_data=scaler.fit_transform(final_dataset)

#     x_train_data,y_train_data=[],[]

#     for i in range(60,len(train_data)):
#         x_train_data.append(scaled_data[i-60:i,0])
#         y_train_data.append(scaled_data[i,0])
        
#     x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

#     x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


#     lstm_model=Sequential()
#     lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
#     lstm_model.add(LSTM(units=50))
#     lstm_model.add(Dense(1))

#     inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
#     inputs_data=inputs_data.reshape(-1,1)
#     inputs_data=scaler.transform(inputs_data)

#     lstm_model.compile(loss='mean_squared_error',optimizer='adam')
#     lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

#     X_test=[]
#     for i in range(60,inputs_data.shape[0]):
#         X_test.append(inputs_data[i-60:i,0])
#     X_test=np.array(X_test)

#     X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#     predicted_closing_price=lstm_model.predict(X_test)
#     predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

#     lstm_model.save("saved_lstm_model.h5")

#     return new_dataset, predicted_closing_price


# def forecast_graph():
#     train_data=new_dataset[:935]
#     valid_data=new_dataset[935:]
#     valid_data['Predictions']=predicted_closing_price

#     st.header("Forecast Data")
#     st.table(data = valid_data.tail())

#     fig1 = plt.figure(figsize=(16,8))

#     plt.plot(train_data["Close"], label="Train Data")
#     plt.plot(valid_data['Close'], label="Actual Closing Price")
#     plt.plot(valid_data['Predictions'], label="Predicted Closing Price")                    
#     plt.legend(loc="upper left")

#     st.write(fig1)

# db, mycol = database()
# stockSelector(db)
# populateDataframe(mycol)
# df = populateDataframe(mycol)
# current_df = currentTable()
# currentGraph(current_df)
# new_dataset, predicted_closing_price = createModel(df)
# forecast_graph()