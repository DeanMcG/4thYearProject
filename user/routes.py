from flask import Flask
from app import app
from user.models import User #User class from models.py

@app.route('/user/signup', methods =['GET'])
def signup():
    return User().signup()