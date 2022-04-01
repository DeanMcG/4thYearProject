from flask import Flask, render_template
import pymongo
from pymongo import MongoClient


app = Flask(__name__)

#Database
client = pymongo.MongoClient("mongodb+srv://admin:Password@stockcluster.cuzo7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client["user_login"]
col = db["account_details"]

#Routes
from user import routes

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard/')
def dashboard():
    return render_template('dashboard.html')