from flask import Flask, request, render_template
from predictor import Predictor
from GetImageFromServer import GetImage
from send_pkl_data import send_pkl_data
import requests
import logging

app = Flask(__name__)

#Handling POST requests from camera
@app.route("/save-pkl-image", methods=['POST'])
def saveImage():
	pkl_form = request.form.to_dict()
	obj = GetImage(pkl_form)
	obj.write_image()
	return ' '

#Handling GET requests from IOS app
@app.route("/lot-data", methods=['GET'])
def sendParkingLotData():
	pkl_name = request.args.get('pkl_name', type=str)
	return send_pkl_data(pkl_name) 
