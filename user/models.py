from flask import Flask, jsonify, request
from passlib.hash import pbkdf2_sha256
import uuid
from app import col


class User:

    def signup(self):
        print(request.form)
        
        user = {
            "_id": uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "password": request.form.get('password'),
        }

        #Encrypt password
        user['password'] = pbkdf2_sha256.encrypt(user['password'])

        col.insert_one(user)

        return jsonify(user), 200